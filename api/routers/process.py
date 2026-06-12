import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import require_api_key
from core.deps import get_processor
from models.schemas import ProcessRequest, ProcessResponse
from services.processor import WikiProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_api_key)])
async def process(
    request: ProcessRequest,
    processor: WikiProcessor = Depends(get_processor),
):
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
