import logging
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from models.schemas import HealthResponse, ProcessRequest, ProcessResponse
from services.llm import LLMProvider, LLMProviderFactory, load_from_env
from services.processor import WikiProcessor
from storage.minio_client import MinioStorage

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

processor = WikiProcessor(storage=storage, llm=llm)


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

    return HealthResponse(
        status="ok" if minio_ok else "degraded",
        minio_connected=minio_ok,
        llm_configured=llm_ok,
        llm_provider=_llm_config.provider,
    )
