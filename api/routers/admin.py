"""Admin endpoints: vector reindex, extraction recompile, concept rebuild."""

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
        raise HTTPException(
            status_code=503, detail="Vector index disabled: PG_DSN is not configured"
        )
    try:
        result = await processor.reindex()
        return {"status": "ok", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Reindex failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}") from e


@router.post("/admin/recompile", dependencies=[Depends(require_api_key)])
async def recompile(processor: WikiProcessor = Depends(get_processor)):
    """Re-run extraction over stored per-app snapshots (no re-ingest).

    Use after an extraction/prompt change to refresh entries from markdown
    already on record."""
    try:
        return {"status": "ok", **await processor.recompile()}
    except Exception as e:
        logger.error("Recompile failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recompile failed: {e}") from e


@router.post("/admin/rebuild-concepts", dependencies=[Depends(require_api_key)])
async def rebuild_concepts(processor: WikiProcessor = Depends(get_processor)):
    """Cross-app concept synthesis over the whole wiki (writes wiki.concepts)."""
    try:
        return {"status": "ok", **await processor.rebuild_concepts()}
    except Exception as e:
        logger.error("Concept rebuild failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Concept rebuild failed: {e}") from e
