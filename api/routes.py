import logging
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from models.schemas import HealthResponse, ProcessRequest, ProcessResponse
from services.embeddings import EmbeddingClient, load_embedding_env
from services.llm import LLMProvider, LLMProviderFactory, load_from_env
from services.processor import WikiProcessor
from storage.minio_client import MinioStorage
from storage.pg_store import pg_store_from_env

logger = logging.getLogger(__name__)

router = APIRouter()

if not os.getenv("PROCESSOR_API_KEY"):
    logger.warning(
        "PROCESSOR_API_KEY not set — /process is UNAUTHENTICATED (dev mode). "
        "Set it before exposing this service beyond localhost."
    )


async def require_api_key(x_api_key: str = Header(default="")):
    """Reject /process calls without a valid X-API-Key when auth is enabled.

    Read at request time (not import time) so tests can toggle the env var.
    """
    expected = os.getenv("PROCESSOR_API_KEY")
    if not expected:
        return  # auth disabled (dev mode)
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# ---------------------------------------------------------------------------
# Singletons (created once at import time)
# ---------------------------------------------------------------------------

storage = MinioStorage()

_llm_config = load_from_env()
llm: LLMProvider = LLMProviderFactory.create(_llm_config)
logger.info(f"LLM provider initialized: {_llm_config.provider} / {_llm_config.model}")

_embed_config = load_embedding_env()
embedder = EmbeddingClient(_embed_config) if _embed_config.is_enabled() else None
vector_store = pg_store_from_env()
if vector_store is None:
    logger.warning(
        "PG_DSN not set — vector index DISABLED (dev mode). Search and reads "
        "fall back to wiki.json scans; set PG_DSN to enable semantic search."
    )
else:
    logger.info(
        f"Vector index enabled (dim={vector_store.dim}, "
        f"embeddings={'on' if embedder else 'off — relational sync only'})"
    )

processor = WikiProcessor(
    storage=storage, llm=llm, embedder=embedder, vector_store=vector_store
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_api_key)])
async def process(request: ProcessRequest):
    """Process markdown files and update the wiki.

    Supports both full wiki generation and app-level incremental updates:
    - If source_app is provided: performs app-level update (only updates that app's files)
    - If source_app is None: performs full wiki update from all markdowns
    """
    try:
        return await processor.process(
            markdowns=request.markdowns,
            timestamp=request.timestamp,
            source_app=request.source_app,
            source_version=request.source_version,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/admin/reindex", dependencies=[Depends(require_api_key)])
async def reindex():
    """Rebuild the PG vector index from wiki.json.

    Bootstrap path for enabling PG on an existing wiki, and the repair path
    for any wiki<->PG drift (e.g. PG was down during submissions)."""
    if processor.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector index disabled: PG_DSN is not configured")
    try:
        result = await processor.reindex()
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


@router.get("/status")
async def status():
    """Return stats about the current wiki and snapshot."""
    try:
        wiki = storage.get_json("wiki.json") or {"apis": {}, "metadata": {}}
        snapshot = storage.get_json("markdowns_snapshot.json") or {}
        return {
            "status": "running",
            "wiki_size": len(wiki.get("apis", {})),
            "tracked_files": len(snapshot),
            "last_updated": wiki.get("metadata", {}).get("updated_at", "unknown"),
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check: verify Minio connectivity and LLM configuration."""
    minio_ok = False
    llm_ok = False

    try:
        storage.client.bucket_exists(storage.bucket)
        minio_ok = True
    except Exception as e:
        logger.error(f"Minio health check failed: {e}")

    try:
        # Local config check only — validate_config() would make a live LLM
        # API call, and /health is polled.
        llm_ok = llm.is_configured()
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")

    vector_ok = False
    if processor.vector_store is not None:
        try:
            vector_ok = await processor.vector_store.available()
        except Exception as e:
            logger.error(f"Vector index health check failed: {e}")

    return HealthResponse(
        status="ok" if minio_ok else "degraded",
        minio_connected=minio_ok,
        llm_configured=llm_ok,
        llm_provider=_llm_config.provider,
        vector_index_connected=vector_ok,
        embeddings_configured=embedder is not None,
    )
