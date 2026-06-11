"""Minimax LLM Provider"""

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

_API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"


class MinimaxProvider(LLMProvider):
    """Minimax LLM API provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"
        if self.mock_mode:
            logger.warning("MinimaxProvider: running in MOCK mode")

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Minimax API and return the assistant message content."""
        if self.mock_mode:
            return json.dumps({"apis": {}, "metadata": {}})

        temp = temperature if temperature is not None else self.config.temperature

        async with httpx.AsyncClient(verify=False) as client:
            try:
                response = await client.post(
                    _API_URL,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temp,
                    },
                    timeout=float(self.config.timeout_seconds),
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid Minimax API key")
                if response.status_code == 429:
                    raise RateLimitException("Minimax rate limit exceeded")
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except (AuthenticationException, RateLimitException):
                raise
            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected Minimax response: {e}")
            except httpx.HTTPStatusError as e:
                raise APIException(f"Minimax API error: {e}")

    async def validate_config(self) -> bool:
        if self.mock_mode or not self.config.api_key:
            logger.warning("MinimaxProvider: skipping validation (mock/no-key mode)")
            return True
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    _API_URL,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "temperature": 0.0,
                    },
                    timeout=10.0,
                )
                if response.status_code == 401:
                    raise AuthenticationException("Invalid Minimax API key")
                response.raise_for_status()
                return True
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"MinimaxProvider validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "minimax",
            "model_name": self.config.model,
            "max_context": 16000,
            "supports_streaming": False,
        }

# Register with factory
LLMProviderFactory.register("minimax", MinimaxProvider)
