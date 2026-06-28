"""Status/health assembly — keeps storage internals out of the API layer."""

import logging
from typing import Optional

from models.schemas import HealthResponse
from services.embeddings import EmbeddingClient
from services.llm import LLMProvider
from services.processor import WikiProcessor
from repository.minio_client import MinioStorage

logger = logging.getLogger(__name__)


def build_status(storage: MinioStorage) -> dict:
    """Stats about the current wiki and snapshot."""
    wiki = storage.get_json("wiki.json") or {"apis": {}, "metadata": {}}
    snapshot = storage.get_json("markdowns_snapshot.json") or {}
    return {
        "status": "running",
        "wiki_size": len(wiki.get("apis", {})),
        "tracked_files": len(snapshot),
        "last_updated": wiki.get("metadata", {}).get("updated_at", "unknown"),
    }


async def build_health(
    storage: MinioStorage,
    llm: LLMProvider,
    processor: WikiProcessor,
    llm_provider_name: str,
    embedder: Optional[EmbeddingClient],
) -> HealthResponse:
    """Health check: Minio connectivity, LLM config, optional vector index."""
    minio_ok = storage.ping()

    llm_ok = False
    try:
        # Local config check only — validate_config() would make a live LLM
        # API call, and /health is polled.
        llm_ok = llm.is_configured()
    # Health probe must never crash on a misbehaving provider check; any error
    # just means "not OK".
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("LLM health check failed: %s", e)

    vector_ok = False
    if processor.vector_store is not None:
        try:
            vector_ok = await processor.vector_store.available()
        # Health probe must never crash on a vector-store error; treat any
        # failure as "index not available".
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Vector index health check failed: %s", e)

    return HealthResponse(
        status="ok" if minio_ok else "degraded",
        minio_connected=minio_ok,
        llm_configured=llm_ok,
        llm_provider=llm_provider_name,
        vector_index_connected=vector_ok,
        embeddings_configured=embedder is not None,
    )
