import logging
import os

from fastapi import APIRouter, HTTPException

from models.schemas import HealthResponse, ProcessRequest, ProcessResponse
from services.llm import MinimaxClient
from services.processor import WikiProcessor
from storage.minio_client import MinioStorage

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Singletons (created once at import time)
# ---------------------------------------------------------------------------

storage = MinioStorage()

_api_key = os.getenv("MINIMAX_API_KEY", "dummy-key")
if _api_key == "dummy-key":
    logger.warning("MINIMAX_API_KEY not set; LLM calls will fail")

llm = MinimaxClient(api_key=_api_key)
processor = WikiProcessor(storage=storage, llm=llm)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    """Process markdown files and update the wiki."""
    try:
        return await processor.process(request.markdowns, request.timestamp)
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
    """Health check: verify Minio connectivity and API key presence."""
    minio_ok = False
    minimax_ok = False

    try:
        storage.client.bucket_exists(storage.bucket)
        minio_ok = True
    except Exception as e:
        logger.error(f"Minio health check failed: {e}")

    try:
        key = os.getenv("MINIMAX_API_KEY", "")
        minimax_ok = bool(key and key != "dummy-key")
    except Exception as e:
        logger.error(f"Minimax health check failed: {e}")

    return HealthResponse(
        status="ok" if minio_ok else "degraded",
        minio_connected=minio_ok,
        minimax_accessible=minimax_ok,
    )
