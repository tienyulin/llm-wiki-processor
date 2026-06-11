"""OpenAI-Compatible LLM Provider

Supports any service that exposes the /v1/chat/completions endpoint:
  - Ollama  (http://localhost:11434/v1)
  - vLLM    (http://server:8000/v1)
  - LM Studio (http://localhost:1234/v1)
  - Custom internal LLM servers
"""

import json
import logging
import os
from typing import Dict, Any, Optional

import httpx

from ..base import LLMProvider
from ..config import LLMConfig
from ..exceptions import APIException, AuthenticationException, RateLimitException, ValidationException
from ..factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Provider for services with OpenAI-compatible /v1/chat/completions API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (config.base_url or "http://localhost:8000").rstrip("/")
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"
        if self.mock_mode:
            logger.warning("OpenAICompatibleProvider: running in MOCK mode")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        return h

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.mock_mode:
            return json.dumps({"apis": {}, "metadata": {}})

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        url = f"{self.base_url}/v1/chat/completions"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self._headers(),
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                        "max_tokens": tokens,
                    },
                    timeout=float(self.config.timeout_seconds),
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid API key for OpenAI-compatible provider")
                if response.status_code == 429:
                    raise RateLimitException("Rate limit exceeded")
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except (AuthenticationException, RateLimitException):
                raise
            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected response format: {e}")
            except httpx.HTTPStatusError as e:
                raise APIException(f"API error: {e}")

    async def validate_config(self) -> bool:
        if self.mock_mode:
            return True
        try:
            url = f"{self.base_url}/v1/chat/completions"
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._headers(),
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                    timeout=10.0,
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid API key")
                response.raise_for_status()
                return True
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"OpenAICompatibleProvider validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai-compatible",
            "model_name": self.config.model,
            "base_url": self.base_url,
            "max_context": 4096,
            "supports_streaming": True,
        }

# Register with factory
LLMProviderFactory.register("openai-compatible", OpenAICompatibleProvider)
