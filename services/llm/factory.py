"""LLM Provider Factory"""

import logging
from typing import Type, Dict

from .base import LLMProvider
from .config import LLMConfig
from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """Creates LLM provider instances from configuration.

    Providers register themselves at import time:
        LLMProviderFactory.register("openai", OpenAIProvider)
    """

    _providers: Dict[str, Type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider class under a (case-insensitive) name."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Instantiate the provider described by config.

        Raises:
            ConfigurationException: if provider name is unknown
        """
        name = config.provider.lower()
        if name not in cls._providers:
            available = ", ".join(sorted(cls._providers))
            raise ConfigurationException(f"Unknown provider: '{name}'. Available: {available}")
        logger.info("Creating '%s' provider (model=%s)", name, config.model)
        return cls._providers[name](config)

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted names of all registered providers."""
        return sorted(cls._providers)
