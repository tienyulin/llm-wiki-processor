"""Anthropic Claude LLM Provider"""

import json
import logging
import re
from typing import Dict, Any, Optional

import httpx

from ..base import LLMProvider
from ..config import LLMConfig
from ..exceptions import APIException, AuthenticationException, RateLimitException, ValidationException
from ..factory import LLMProviderFactory

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config

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

            except (AuthenticationException, RateLimitException):
                raise
            except (KeyError, json.JSONDecodeError) as e:
                raise ValidationException(f"Unexpected Anthropic response: {e}")
            except httpx.HTTPStatusError as e:
                raise APIException(f"Anthropic API error: {e}")

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
        except Exception as e:
            logger.error(f"AnthropicProvider validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "anthropic",
            "model_name": self.config.model,
            "max_context": 200000,
            "supports_streaming": True,
        }

    def _extract_json(self, content: str) -> dict:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    async def generate_wiki(self, markdowns: dict) -> dict:
        combined = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in markdowns.items())
        prompt = (
            "Analyze the following API documentation markdown files and generate a structured wiki.\n\n"
            f"{combined}\n\n"
            "Task:\n"
            "1. Extract all API endpoints (method, path, description, parameters)\n"
            "2. Group by module/service\n"
            '3. Generate JSON structure: {"apis": {"module": {"endpoint": {...}}}, "metadata": {}}\n\n'
            "Output ONLY valid JSON, no markdown."
        )
        content = await self.generate(prompt, temperature=0.3)
        return self._extract_json(content)

    async def update_wiki(self, current_files: dict, changed_markdowns: dict, changes) -> dict:
        changed_content = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in changed_markdowns.items())
        wiki_summary = json.dumps(current_files, ensure_ascii=False, indent=2)[:2000]
        prompt = (
            f"Current Wiki (summarized):\n{wiki_summary}\n\n"
            f"Changes: {json.dumps(changes) if isinstance(changes, dict) else changes}\n\n"
            f"New/Modified Markdowns:\n{changed_content}\n\n"
            "Task:\n"
            "1. For new files: Extract APIs and add to wiki\n"
            "2. For modified files: Update related APIs\n"
            "3. For deleted files: Remove from wiki\n"
            "4. Maintain module structure and semantic relationships\n\n"
            "Output ONLY the updated wiki JSON."
        )
        content = await self.generate(prompt, temperature=0.2)
        return self._extract_json(content)


# Register with factory
LLMProviderFactory.register("anthropic", AnthropicProvider)
