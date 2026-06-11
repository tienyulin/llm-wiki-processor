"""Azure OpenAI LLM Provider

Endpoint format:
  https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=...

Required env vars:
  LLM_API_KEY        - Azure API key
  LLM_BASE_URL       - https://{resource}.openai.azure.com
  LLM_MODEL          - Deployment name (e.g. "gpt-4")
  AZURE_API_VERSION  - Optional; default 2024-02-01
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

_DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        resource_url = (config.base_url or "").rstrip("/")
        api_version = os.getenv("AZURE_API_VERSION", _DEFAULT_API_VERSION)
        deployment = config.model
        self._url = (
            f"{resource_url}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    def _headers(self) -> dict:
        return {"api-key": self.config.api_key, "Content-Type": "application/json"}

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
                    self._url,
                    headers=self._headers(),
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                        "max_tokens": tokens,
                    },
                    timeout=float(self.config.timeout_seconds),
                )
                if response.status_code in (401, 403):
                    raise AuthenticationException("Invalid Azure OpenAI credentials")
                if response.status_code == 429:
                    raise RateLimitException("Azure OpenAI rate limit exceeded")
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except (AuthenticationException, RateLimitException):
                raise
            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected Azure OpenAI response: {e}")
            except httpx.HTTPStatusError as e:
                raise APIException(f"Azure OpenAI API error: {e}")

    async def validate_config(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    headers=self._headers(),
                    json={
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                    },
                    timeout=10.0,
                )
                if response.status_code in (401, 403):
                    raise AuthenticationException("Invalid Azure OpenAI credentials")
                response.raise_for_status()
                return True
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"AzureOpenAIProvider validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "azure",
            "model_name": self.config.model,
            "max_context": 128000,
            "supports_streaming": True,
        }

# Register with factory
LLMProviderFactory.register("azure", AzureOpenAIProvider)
LLMProviderFactory.register("azure-openai", AzureOpenAIProvider)  # alias
