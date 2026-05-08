from typing import Optional
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request body for /process endpoint"""
    markdowns: dict[str, str]  # {filename: content}
    timestamp: str
    trigger_info: dict


class ProcessResponse(BaseModel):
    """Response from /process endpoint"""
    status: str
    message: str
    wiki_url: Optional[str] = None
    changes_summary: dict = {}
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    minio_connected: bool
    minimax_accessible: bool
