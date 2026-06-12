import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import require_api_key
from core.deps import get_processor
from services.processor import WikiProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/admin/reindex", dependencies=[Depends(require_api_key)])
async def reindex(processor: WikiProcessor = Depends(get_processor)):
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
