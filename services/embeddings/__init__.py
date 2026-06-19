"""Embedding service: OpenAI-compatible /v1/embeddings client with mock mode.

Mirrors the structure of services/llm: config loaded from env, a thin async
client, and a deterministic mock mode (MOCK_EMBEDDINGS) for testing without
network access. Reuses the LLM exception types so callers handle one error
vocabulary.
"""

from .client import EmbeddingClient
from .config import EmbeddingConfig, load_embedding_env
from .mock import mock_embed
from .text import entry_to_text, knowledge_to_text

__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "load_embedding_env",
    "mock_embed",
    "entry_to_text",
    "knowledge_to_text",
]
