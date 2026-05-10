"""LLM Provider Module

Public API:
    LLMProvider      - abstract base class
    LLMConfig        - configuration dataclass
    load_from_env()  - load config from environment variables
    LLMProviderFactory - create provider instances

Usage:
    from services.llm import load_from_env, LLMProviderFactory
    config = load_from_env()
    llm = LLMProviderFactory.create(config)
    result = await llm.generate_wiki(markdowns)
"""

from .base import LLMProvider
from .config import LLMConfig, load_from_env
from .factory import LLMProviderFactory
from .exceptions import (
    LLMException,
    AuthenticationException,
    RateLimitException,
    ConfigurationException,
    APIException,
    ValidationException,
)

# Import providers to trigger factory registration
import services.llm.providers  # noqa: F401

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "load_from_env",
    "LLMProviderFactory",
    "LLMException",
    "AuthenticationException",
    "RateLimitException",
    "ConfigurationException",
    "APIException",
    "ValidationException",
]
