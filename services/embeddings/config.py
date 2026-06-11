"""Embedding provider configuration."""

import os
from dataclasses import dataclass

from services.llm.exceptions import ConfigurationException


@dataclass
class EmbeddingConfig:
    """Configuration for the OpenAI-compatible embeddings endpoint.

    Unlike LLMConfig, an unconfigured embedder is not an error: embeddings
    are an optional feature, and is_enabled() gates every use.
    """

    base_url: str = ""
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dim: int = 1536
    batch_size: int = 64
    timeout_seconds: int = 30
    mock_mode: bool = False

    def __post_init__(self):
        if self.dim <= 0:
            raise ConfigurationException("EMBEDDING_DIM must be positive")
        if self.batch_size <= 0:
            raise ConfigurationException("EMBEDDING_BATCH_SIZE must be positive")
        if self.timeout_seconds <= 0:
            raise ConfigurationException("EMBEDDING_TIMEOUT must be positive")

    def is_enabled(self) -> bool:
        """Mock mode counts as enabled (mirrors LLMProvider.is_configured)."""
        return self.mock_mode or bool(self.base_url)


def load_embedding_env() -> EmbeddingConfig:
    """Load embedding configuration from environment variables.

    Supported env vars:
        EMBEDDING_BASE_URL    - OpenAI-compatible root URL (empty = disabled)
        EMBEDDING_API_KEY     - Bearer token (optional for local servers)
        EMBEDDING_MODEL       - Model name (default: text-embedding-3-small)
        EMBEDDING_DIM         - Vector dimension (default: 1536, must match DDL)
        EMBEDDING_BATCH_SIZE  - Texts per request (default: 64)
        EMBEDDING_TIMEOUT     - Request timeout in seconds (default: 30)
        MOCK_EMBEDDINGS       - "true" = deterministic local vectors, no network
    """
    return EmbeddingConfig(
        base_url=(os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/"),
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        timeout_seconds=int(os.getenv("EMBEDDING_TIMEOUT", "30")),
        mock_mode=os.getenv("MOCK_EMBEDDINGS", "false").lower() == "true",
    )
