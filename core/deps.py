"""Dependency providers for FastAPI routes.

Each provider is an lru_cache(1) singleton built lazily on first request —
tests construct TestClient(app) at module import time without running the
lifespan, so nothing here may touch the network at import. Production
fail-fast is preserved by the lifespan warm-up in main.py, which calls
get_processor() once at startup.

Tests override these via app.dependency_overrides and reset cached
instances between tests with reset_singletons().
"""

import logging
from functools import lru_cache
from typing import Optional

from core.config import get_settings
from services.embeddings import EmbeddingClient
from services.llm import LLMProvider, LLMProviderFactory
from services.processor import WikiProcessor
from repository.minio_client import MinioStorage
from repository.pg_store import PGVectorStore, pg_store_from_env

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_storage() -> MinioStorage:
    return MinioStorage()


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    settings = get_settings()
    provider = LLMProviderFactory.create(settings.llm)
    logger.info(
        f"LLM provider initialized: {settings.llm.provider} / {settings.llm.model}"
    )
    return provider


@lru_cache(maxsize=1)
def get_embedder() -> Optional[EmbeddingClient]:
    cfg = get_settings().embeddings
    return EmbeddingClient(cfg) if cfg.is_enabled() else None


@lru_cache(maxsize=1)
def get_vector_store() -> Optional[PGVectorStore]:
    store = pg_store_from_env()
    if store is None:
        logger.warning(
            "PG_DSN not set — vector index DISABLED (dev mode). Search and reads "
            "fall back to wiki.json scans; set PG_DSN to enable semantic search."
        )
    else:
        logger.info(
            f"Vector index enabled (dim={store.dim}, "
            f"embeddings={'on' if get_embedder() else 'off — relational sync only'})"
        )
    return store


@lru_cache(maxsize=1)
def get_processor() -> WikiProcessor:
    return WikiProcessor(
        storage=get_storage(),
        llm=get_llm(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


def reset_singletons() -> None:
    """Test seam: drop all cached instances so the next request rebuilds."""
    for fn in (get_processor, get_vector_store, get_embedder, get_llm, get_storage):
        fn.cache_clear()
    get_settings.cache_clear()
