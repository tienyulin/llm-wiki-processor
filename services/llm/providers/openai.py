"""OpenAI LLM Provider"""

import json
import logging
from typing import Dict, Any, Optional

import httpx

from ..base import LLMProvider
from ..exceptions import (
    APIException,
    AuthenticationException,
    RateLimitException,
    ValidationException,
)
from ..factory import LLMProviderFactory

logger = logging.getLogger(__name__)

_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(LLMProvider):
    """OpenAI (GPT) LLM provider."""

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    _API_URL,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                        "max_tokens": tokens,
                    },
                    timeout=float(self.config.timeout_seconds),
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid OpenAI API key")
                if response.status_code == 429:
                    raise RateLimitException("OpenAI rate limit exceeded")
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected OpenAI response: {e}") from e
            except httpx.HTTPStatusError as e:
                raise APIException(f"OpenAI API error: {e}") from e

    async def validate_config(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _API_URL,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                    timeout=10.0,
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid OpenAI API key")
                response.raise_for_status()
                return True
        except AuthenticationException:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("OpenAIProvider validation failed: %s", e)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model_name": self.config.model,
            "max_context": 128000,
            "supports_streaming": True,
        }


# Register with factory
LLMProviderFactory.register("openai", OpenAIProvider)
