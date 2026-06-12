import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from core.config import get_settings
from core.deps import get_embedder, get_llm, get_processor, get_storage
from models.schemas import HealthResponse
from services import system_service
from services.embeddings import EmbeddingClient
from services.llm import LLMProvider
from services.processor import WikiProcessor
from repository.minio_client import MinioStorage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def status(storage: MinioStorage = Depends(get_storage)):
    """Return stats about the current wiki and snapshot."""
    try:
        return system_service.build_status(storage)
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health(
    storage: MinioStorage = Depends(get_storage),
    llm: LLMProvider = Depends(get_llm),
    processor: WikiProcessor = Depends(get_processor),
    embedder: Optional[EmbeddingClient] = Depends(get_embedder),
):
    """Health check: verify Minio connectivity and LLM configuration."""
    return await system_service.build_health(
        storage=storage,
        llm=llm,
        processor=processor,
        llm_provider_name=get_settings().llm.provider,
        embedder=embedder,
    )
