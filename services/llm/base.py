"""Abstract base class for all LLM providers"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


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
        if os.getenv("MOCK_LLM", "false").lower() == "true":
            return True
        config = getattr(self, "config", None)
        return bool(config and config.api_key)
