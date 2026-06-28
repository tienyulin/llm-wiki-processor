"""Anthropic Claude LLM Provider"""

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

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Anthropic temperature range is 0-1 (clamp if needed)
        temp = min(temperature if temperature is not None else self.config.temperature, 1.0)
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    _API_URL,
                    headers={
                        "x-api-key": self.config.api_key,
                        "anthropic-version": _ANTHROPIC_VERSION,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "max_tokens": tokens,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                    },
                    timeout=float(self.config.timeout_seconds),
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid Anthropic API key")
                if response.status_code == 429:
                    raise RateLimitException("Anthropic rate limit exceeded")
                response.raise_for_status()

                result = response.json()
                # Claude response: content[0].text
                return result["content"][0]["text"]

            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected Anthropic response: {e}") from e
            except httpx.HTTPStatusError as e:
                raise APIException(f"Anthropic API error: {e}") from e

    async def validate_config(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    _API_URL,
                    headers={
                        "x-api-key": self.config.api_key,
                        "anthropic-version": _ANTHROPIC_VERSION,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                    timeout=10.0,
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid Anthropic API key")
                response.raise_for_status()
                return True
        except AuthenticationException:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("AnthropicProvider validation failed: %s", e)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model_name": self.config.model,
            "max_context": 200000,
            "supports_streaming": True,
        }


# Register with factory
LLMProviderFactory.register("anthropic", AnthropicProvider)
