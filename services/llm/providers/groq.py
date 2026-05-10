"""Groq LLM Provider

Groq exposes an OpenAI-compatible endpoint, so this simply subclasses
OpenAICompatibleProvider with a fixed base URL.
"""

from ..factory import LLMProviderFactory
from .openai_compatible import OpenAICompatibleProvider
from ..config import LLMConfig

_GROQ_BASE_URL = "https://api.groq.com"


class GroqProvider(OpenAICompatibleProvider):
    """Groq ultra-fast inference — OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        # Force base_url to Groq regardless of what was passed
        config.base_url = _GROQ_BASE_URL
        super().__init__(config)

    def get_model_info(self):
        info = super().get_model_info()
        info["provider"] = "groq"
        info["max_context"] = 32768
        return info


# Register with factory
LLMProviderFactory.register("groq", GroqProvider)
