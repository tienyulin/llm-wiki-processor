"""Abstract base class for all LLM providers"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}.-]*)")
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class LLMProvider(ABC):
    """
    Unified interface for LLM providers.

    All providers must implement:
        generate()      - send a prompt, get back text
        validate_config() - check API key / connectivity
        get_model_info() - return model metadata
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt.

        Returns:
            Raw text content from the model (callers do their own JSON parsing).

        Raises:
            AuthenticationException: bad API key
            RateLimitException: rate limit hit
            APIException: other API-level error
            ValidationException: unexpected response shape
        """

    @abstractmethod
    async def validate_config(self) -> bool:
        """Check that credentials and connectivity are working.

        Returns True on success; raises on failure.
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata: model_name, max_context, provider, …"""

    def is_configured(self) -> bool:
        """Cheap local configuration check — no API call.

        Unlike validate_config(), this is safe to call from frequently polled
        health endpoints. Mock mode counts as configured; otherwise an API key
        must be present (openai-compatible servers may run keyless, so
        providers without a key report unconfigured here).
        """
        if self._mock_mode():
            return True
        config = getattr(self, "config", None)
        return bool(config and config.api_key)

    # ------------------------------------------------------------------
    # High-level wiki methods, shared by all providers (consumed by
    # processor.py). Providers only supply generate(); prompts, JSON
    # extraction, and mock mode live here.
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_mode() -> bool:
        return os.getenv("MOCK_LLM", "false").lower() == "true"

    def extract_json(self, content: str) -> dict:
        """Strip <think> tags and parse JSON from an LLM response."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    @staticmethod
    def _mock_apis_from_markdowns(markdowns: Dict[str, str]) -> dict:
        """Derive deterministic API entries from the input markdowns.

        Mock responses must reflect the input (not canned data) so that
        integration and stress tests can verify each app's data actually
        reaches the wiki. One module per file; every file yields at least
        one entry.
        """
        apis: dict = {}
        for filename, content in markdowns.items():
            module = filename.rsplit(".", 1)[0]
            for suffix in ("_api", "_arch", "-api"):
                module = module.removesuffix(suffix)
            h1 = _H1_RE.search(content or "")
            description = h1.group(1).strip() if h1 else filename
            endpoints = _ENDPOINT_RE.findall(content or "")
            module_apis = apis.setdefault(module, {})
            if endpoints:
                for method, path in endpoints:
                    module_apis[f"{method} {path}"] = {
                        "method": method,
                        "path": path,
                        "description": description,
                    }
            else:
                # No recognizable endpoints — still surface the doc
                module_apis[f"DOC {filename}"] = {
                    "method": "DOC",
                    "path": filename,
                    "description": description,
                }
        return apis

    async def generate_wiki(self, markdowns: Dict[str, str]) -> dict:
        """Generate a full structured wiki from markdown files.

        Returns {"apis": {module: {api_key: {...}}}, "metadata": {...}}.
        Provenance (source_app/source_version) is stamped by the processor,
        not requested from the model.
        """
        if self._mock_mode():
            return {"apis": self._mock_apis_from_markdowns(markdowns), "metadata": {}}

        combined = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in markdowns.items())
        prompt = (
            "Analyze the following API documentation markdown files and generate a structured wiki.\n\n"
            f"{combined}\n\n"
            "Task:\n"
            "1. Extract all API endpoints (method, path, description, parameters)\n"
            "2. Group by module/service\n"
            '3. Generate JSON structure: {"apis": {"<module>": {"<METHOD /path>": {"method": ..., "path": ..., "description": ...}}}, "metadata": {}}\n\n'
            "Output ONLY valid JSON, no markdown."
        )
        logger.info(f"{type(self).__name__}: initial wiki generation ({len(combined)} chars)")
        content = await self.generate(prompt, temperature=0.3)
        return self.extract_json(content)

    async def update_wiki(self, current_apis: dict, changed_markdowns: Dict[str, str], changes) -> dict:
        """Regenerate one application's API entries.

        Args:
            current_apis: the app's existing entries, {module: {api_key: {...}}}
            changed_markdowns: the app's new/modified markdown files
            changes: change summary (dict or str), included in the prompt

        Returns {"apis": {module: {api_key: {...}}}} containing ONLY this
        app's entries — the processor merges them into the shared wiki.
        """
        if self._mock_mode():
            return {"apis": self._mock_apis_from_markdowns(changed_markdowns)}

        changed_content = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in changed_markdowns.items())
        current_summary = json.dumps(current_apis, ensure_ascii=False, indent=2)[:2000]
        prompt = (
            f"Current API entries for this application (summarized):\n{current_summary}\n\n"
            f"Changes: {json.dumps(changes) if isinstance(changes, dict) else changes}\n\n"
            f"New/Modified Markdowns:\n{changed_content}\n\n"
            "Task:\n"
            "1. Extract all API endpoints from the markdowns (method, path, description, parameters)\n"
            "2. Group by module/service\n"
            "3. Return ONLY this application's entries — do not include other applications\n\n"
            'Output ONLY valid JSON: {"apis": {"<module>": {"<METHOD /path>": {...}}}}'
        )
        logger.info(f"{type(self).__name__}: incremental update")
        content = await self.generate(prompt, temperature=0.2)
        return self.extract_json(content)
