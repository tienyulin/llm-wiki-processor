from typing import Optional
from pydantic import BaseModel, field_validator


class ProcessRequest(BaseModel):
    """Request body for /process endpoint"""
    markdowns: dict[str, str]  # {filename: content}
    timestamp: str
    trigger_info: dict
    source_app: Optional[str] = None  # e.g., "app-inventory"
    source_version: Optional[str] = None  # git commit sha or version tag

    @field_validator("markdowns")
    @classmethod
    def markdowns_must_not_be_empty(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("markdowns must contain at least one file")
        return v


class ProcessResponse(BaseModel):
    """Response from /process endpoint"""
    status: str  # "success" | "partial" | "failed"
    message: str
    wiki_url: Optional[str] = None
    changes_summary: dict = {}
    timestamp: str
    source_app: Optional[str] = None
    files_updated: list[str] = []
    validation_errors: list[dict] = []
    processing_time_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    minio_connected: bool
    llm_configured: bool
    llm_provider: str = "unknown"
    # Backward-compat alias (mirrors llm_configured)
    minimax_accessible: Optional[bool] = None

    def model_post_init(self, __context):
        if self.minimax_accessible is None:
            self.minimax_accessible = self.llm_configured
