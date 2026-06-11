"""Minimax LLM Provider"""

import json
import logging
import os
import re
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
            return self._mock_response(prompt)

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

    # ------------------------------------------------------------------
    # Helpers kept from original MinimaxClient
    # ------------------------------------------------------------------

    def extract_json(self, content: str) -> dict:
        """Strip <think> tags and parse JSON from LLM response."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def _mock_response(self, prompt: str) -> str:
        """Return deterministic mock JSON for testing."""
        if "generate a structured wiki" in prompt.lower():
            return json.dumps({
                "apis": {
                    "inventory": {
                        "GET /inventory": {"method": "GET", "path": "/inventory", "description": "取得所有庫存項目"},
                        "POST /inventory": {"method": "POST", "path": "/inventory", "description": "創建新的庫存項目"},
                        "GET /inventory/{id}": {"method": "GET", "path": "/inventory/{id}", "description": "取得單個庫存項目詳細資訊"},
                    },
                    "order": {
                        "GET /orders": {"method": "GET", "path": "/orders", "description": "取得所有訂單"},
                        "POST /orders": {"method": "POST", "path": "/orders", "description": "建立新訂單"},
                    },
                },
                "metadata": {"version": "1.0", "modules": ["inventory", "order"], "updated_at": "2026-05-09T05:24:25.343753"},
            })
        if "incremental update" in prompt.lower() or "for new files" in prompt.lower():
            return json.dumps({
                "apis": {
                    "inventory": {
                        "GET /inventory": {"method": "GET", "path": "/inventory", "description": "取得所有庫存項目"},
                        "POST /inventory": {"method": "POST", "path": "/inventory", "description": "創建新的庫存項目"},
                        "GET /inventory/{id}": {"method": "GET", "path": "/inventory/{id}", "description": "取得單個庫存項目詳細資訊"},
                    },
                    "order": {
                        "GET /orders": {"method": "GET", "path": "/orders", "description": "取得所有訂單"},
                        "POST /orders": {"method": "POST", "path": "/orders", "description": "建立新訂單"},
                    },
                    "payment": {
                        "POST /payments": {"method": "POST", "path": "/payments", "description": "建立新支付"},
                        "GET /payments/{id}": {"method": "GET", "path": "/payments/{id}", "description": "取得支付詳細資訊"},
                        "PUT /payments/{id}/status": {"method": "PUT", "path": "/payments/{id}/status", "description": "更新支付狀態"},
                    },
                },
                "metadata": {"version": "1.0", "modules": ["inventory", "order", "payment"], "updated_at": "2026-05-09T05:24:33.164815"},
            })
        return json.dumps({"apis": {}, "metadata": {}})

    # ------------------------------------------------------------------
    # High-level wiki methods (consumed by processor.py)
    # ------------------------------------------------------------------

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
        logger.info(f"MinimaxProvider: initial wiki generation ({len(combined)} chars)")
        content = await self.generate(prompt, temperature=0.3)
        return self.extract_json(content)

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
        logger.info("MinimaxProvider: incremental update")
        content = await self.generate(prompt, temperature=0.2)
        return self.extract_json(content)


# Register with factory
LLMProviderFactory.register("minimax", MinimaxProvider)
