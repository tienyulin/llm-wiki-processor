"""Async client for OpenAI-compatible /v1/embeddings endpoints.

Works with OpenAI itself and self-hosted servers (Ollama, vLLM, LM Studio,
text-embeddings-inference). Error mapping follows
services/llm/providers/openai_compatible.py so callers see the same
exception vocabulary as LLM calls.
"""

import json
import logging
from typing import Any

import httpx

from services.llm.exceptions import (
    APIException,
    AuthenticationException,
    RateLimitException,
    ValidationException,
)

from .config import EmbeddingConfig
from .mock import mock_embed

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Batched embedding calls against {base_url}/v1/embeddings."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        if config.mock_mode:
            logger.warning("EmbeddingClient: running in MOCK mode")

    def is_enabled(self) -> bool:
        """True when embeddings are usable (mock mode or a configured base URL)."""
        return self.config.is_enabled()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        return h

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in order. Returns one vector per input text.

        Raises AuthenticationException / RateLimitException / APIException /
        ValidationException — callers decide whether that is fatal (the
        processor degrades to NULL vectors, it never fails the wiki write).
        """
        if not texts:
            return []
        if self.config.mock_mode:
            return [mock_embed(t, self.config.dim) for t in texts]
        if not self.config.base_url:
            raise APIException("Embeddings not configured (EMBEDDING_BASE_URL is empty)")

        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=float(self.config.timeout_seconds)) as client:
            for start in range(0, len(texts), self.config.batch_size):
                chunk = texts[start : start + self.config.batch_size]
                vectors.extend(await self._embed_chunk(client, chunk))
        return vectors

    async def _embed_chunk(self, client: httpx.AsyncClient, chunk: list[str]) -> list[list[float]]:
        url = f"{self.config.base_url}/v1/embeddings"
        body: dict[str, Any] = {"model": self.config.model, "input": chunk}
        if self.config.send_dimensions:
            body["dimensions"] = self.config.dim
        try:
            response = await client.post(
                url,
                headers=self._headers(),
                json=body,
            )
            if response.status_code == 401:
                raise AuthenticationException("Invalid API key for embeddings endpoint")
            if response.status_code == 429:
                raise RateLimitException("Embeddings rate limit exceeded")
            response.raise_for_status()

            data = response.json()["data"]
            # The API may return items out of order; "index" is authoritative
            # when present. Some OpenAI-compatible servers (e.g. Gemini's compat
            # endpoint) omit "index" and return items in request order — fall
            # back to that.
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
        # AuthenticationException / RateLimitException raised above are distinct
        # types from the handlers below, so they already propagate unchanged.
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            raise ValidationException(f"Unexpected embeddings response format: {e}") from e
        except httpx.HTTPStatusError as e:
            raise APIException(f"Embeddings API error: {e}") from e

        if len(vectors) != len(chunk):
            raise ValidationException(
                f"Embeddings count mismatch: sent {len(chunk)} texts, got {len(vectors)} vectors"
            )
        for vec in vectors:
            if len(vec) != self.config.dim:
                raise ValidationException(
                    f"Embedding dimension mismatch: expected {self.config.dim}, got {len(vec)} "
                    f"(check EMBEDDING_DIM against the model)"
                )
        return vectors
