"""LLM Provider Configuration"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import os

from .exceptions import ConfigurationException


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: str          # "openai", "anthropic", "gemini", "groq", "azure", "openai-compatible", "minimax"
    api_key: str           # API key/token
    model: str             # Model name (provider-specific)
    temperature: float = 0.7
    max_tokens: int = 4000
    base_url: Optional[str] = None  # For openai-compatible self-hosted services
    timeout_seconds: int = 60
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.provider:
            raise ConfigurationException("Provider must be specified")
        if self.provider == "openai-compatible" and not self.base_url:
            raise ConfigurationException("LLM_BASE_URL is required for openai-compatible provider")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationException("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ConfigurationException("max_tokens must be positive")


# Default model names per provider
_PROVIDER_DEFAULTS: Dict[str, str] = {
    "openai": "gpt-4-turbo",
    "anthropic": "claude-opus-4-7",
    "gemini": "gemini-2.0-flash",
    "groq": "mixtral-8x7b-32768",
    "azure": "gpt-4",
    "openai-compatible": "local-model",
    "minimax": "MiniMax-M2.7",
}


def load_from_env() -> LLMConfig:
    """Load LLM configuration from environment variables.

    Supported env vars:
        LLM_PROVIDER      - Provider name (default: minimax)
        LLM_API_KEY       - API key (falls back to MINIMAX_API_KEY for backward compat)
        LLM_MODEL         - Model name (defaults per provider)
        LLM_TEMPERATURE   - Temperature (default: 0.7)
        LLM_MAX_TOKENS    - Max tokens (default: 4000)
        LLM_BASE_URL      - Base URL (openai-compatible only)
        LLM_TIMEOUT       - Request timeout in seconds (default: 60)
    """
    provider = os.getenv("LLM_PROVIDER", "minimax").lower()

    # Backward-compatible API key fallback
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("MINIMAX_API_KEY", "")
    )

    model = os.getenv("LLM_MODEL") or _PROVIDER_DEFAULTS.get(provider, "")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    base_url = os.getenv("LLM_BASE_URL")
    timeout_seconds = int(os.getenv("LLM_TIMEOUT", "60"))

    # Mock mode performs deterministic extraction without calling any provider,
    # so no API key is required (matches the documented key-free mock setup).
    mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"

    # Require API key for providers other than openai-compatible (local may not need one)
    if not api_key and provider != "openai-compatible" and not mock_mode:
        raise ConfigurationException(
            f"API key required. Set LLM_API_KEY (or MINIMAX_API_KEY) for provider: {provider}"
        )

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
