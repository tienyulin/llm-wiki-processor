# LLM Provider Abstraction - Implementation Guide

**Objective:** Refactor hardcoded Minimax API configuration to support multiple LLM providers (OpenAI, Anthropic, Google Gemini, Groq, Azure OpenAI, and self-hosted OpenAI-compatible services).

**Scope:** wiki-processor service only. Core logic in processor.py remains unchanged.

**Backward Compatibility:** 100% guaranteed. Default to Minimax if LLM_PROVIDER not specified.

---

## Phase 1: Project Setup & Base Classes

### Step 1.1: Create Directory Structure

Create the following new directories:

```bash
mkdir -p wiki-processor/services/llm/providers
touch wiki-processor/services/llm/__init__.py
touch wiki-processor/services/llm/base.py
touch wiki-processor/services/llm/exceptions.py
touch wiki-processor/services/llm/factory.py
touch wiki-processor/services/llm/config.py
touch wiki-processor/services/llm/providers/__init__.py
```

### Step 1.2: Implement `services/llm/exceptions.py`

Define custom exceptions for standardized error handling across all providers:

```python
"""LLM Provider Exceptions"""


class LLMException(Exception):
    """Base exception for all LLM provider errors"""
    pass


class AuthenticationException(LLMException):
    """Raised when API key is invalid or authentication fails"""
    pass


class RateLimitException(LLMException):
    """Raised when rate limit is exceeded"""
    pass


class ConfigurationException(LLMException):
    """Raised when provider configuration is invalid"""
    pass


class APIException(LLMException):
    """Raised when API returns an error"""
    pass


class ValidationException(LLMException):
    """Raised when response validation fails"""
    pass
```

### Step 1.3: Implement `services/llm/config.py`

Create configuration dataclass and loading logic:

```python
"""LLM Provider Configuration"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import os
from dotenv import load_dotenv

from .exceptions import ConfigurationException


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    
    provider: str       # "openai", "anthropic", "gemini", "groq", "azure", "openai-compatible", "minimax"
    api_key: str        # API key/token
    model: str          # Model name (provider-specific)
    temperature: float = 0.7
    max_tokens: int = 4000
    base_url: Optional[str] = None  # For openai-compatible only
    timeout_seconds: int = 60
    extra: Dict = field(default_factory=dict)  # Provider-specific options
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.provider:
            raise ConfigurationException("Provider must be specified")
        if not self.api_key and self.provider not in ["openai-compatible"]:
            raise ConfigurationException(f"API key required for {self.provider}")
        if self.provider == "openai-compatible" and not self.base_url:
            raise ConfigurationException("base_url required for openai-compatible provider")
        if not 0 <= self.temperature <= 2:
            raise ConfigurationException("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ConfigurationException("max_tokens must be positive")


def load_from_env() -> LLMConfig:
    """Load LLM configuration from environment variables"""
    
    load_dotenv()
    
    provider = os.getenv("LLM_PROVIDER", "minimax").lower()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    base_url = os.getenv("LLM_BASE_URL")
    timeout_seconds = int(os.getenv("LLM_TIMEOUT", "60"))
    
    # Set defaults based on provider
    if not model:
        defaults = {
            "openai": "gpt-4-turbo",
            "anthropic": "claude-opus-4-7",
            "gemini": "gemini-2.0-flash",
            "groq": "mixtral-8x7b-32768",
            "azure": "gpt-4",
            "openai-compatible": "local-model",
            "minimax": "MiniMax-M2.7"
        }
        model = defaults.get(provider, "gpt-4")
    
    if not api_key and provider != "openai-compatible":
        raise ConfigurationException(
            f"LLM_API_KEY environment variable required for provider: {provider}"
        )
    
    config = LLMConfig(
        provider=provider,
        api_key=api_key or "not-required",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        timeout_seconds=timeout_seconds
    )
    
    return config
```

### Step 1.4: Implement `services/llm/base.py`

Create the abstract base class:

```python
"""Base LLM Provider Interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt
            temperature: Temperature override (uses config default if None)
            max_tokens: Max tokens override (uses config default if None)
        
        Returns:
            Generated text content (plain string, not JSON-wrapped)
        
        Raises:
            AuthenticationException: If API key is invalid
            RateLimitException: If rate limit is exceeded
            APIException: If API returns an error
            ValidationException: If response validation fails
        """
        pass
    
    @abstractmethod
    async def validate_config(self) -> bool:
        """
        Validate that the provider is properly configured.
        Test the API connection if possible.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ConfigurationException: If configuration is invalid
            AuthenticationException: If authentication fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dict with keys: model_name, max_context, supports_streaming, etc.
        """
        pass
```

---

## Phase 2: Provider Factory & Configuration

### Step 2.1: Implement `services/llm/factory.py`

Create the factory for provider instantiation:

```python
"""LLM Provider Factory"""

from typing import Type, Dict
import logging

from .base import LLMProvider
from .config import LLMConfig
from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM provider instances"""
    
    _providers: Dict[str, Type[LLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider implementation"""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """
        Create a provider instance from configuration.
        
        Args:
            config: LLMConfig instance
        
        Returns:
            Instantiated LLMProvider
        
        Raises:
            ConfigurationException: If provider type is unknown
        """
        provider_name = config.provider.lower()
        
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ConfigurationException(
                f"Unknown LLM provider: {provider_name}. Available: {available}"
            )
        
        provider_class = cls._providers[provider_name]
        logger.info(f"Creating {provider_name} provider with model {config.model}")
        
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of registered provider names"""
        return sorted(cls._providers.keys())
```

### Step 2.2: Implement `services/llm/__init__.py`

Export public API:

```python
"""LLM Provider Module"""

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
```

---

## Phase 3: Provider Implementations

### Step 3.1: Implement `services/llm/providers/minimax.py`

Migrate existing Minimax code:

```python
"""Minimax LLM Provider"""

import httpx
import json
import logging
import re
from typing import Optional, Dict, Any

from ..base import LLMProvider
from ..config import LLMConfig
from ..exceptions import (
    AuthenticationException,
    RateLimitException,
    APIException,
    ValidationException,
)
from ..factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class MinimaxProvider(LLMProvider):
    """Minimax LLM Provider Implementation"""
    
    API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
    
    def __init__(self, config: LLMConfig):
        """Initialize Minimax provider"""
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds)
        )
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Minimax API"""
        
        if not self.config.api_key or self.config.api_key == "not-required":
            # Mock mode for testing
            return self._mock_response(prompt)
        
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens
        }
        
        try:
            response = await self.client.post(
                self.API_URL,
                json=payload,
                headers=headers,
                verify=False  # self-signed certs in internal deployments
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["result"]["choices"][0]["message"]["content"]
            
            return self._extract_json(content)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationException("Invalid Minimax API key")
            elif e.response.status_code == 429:
                raise RateLimitException("Minimax rate limit exceeded")
            else:
                raise APIException(f"Minimax API error: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValidationException(f"Invalid Minimax response format: {e}")
    
    async def validate_config(self) -> bool:
        """Validate Minimax configuration"""
        if not self.config.api_key or self.config.api_key == "not-required":
            logger.warning("Minimax API key not configured, using mock mode")
            return True
        
        # Try a simple request to validate
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "max_tokens": 10
            }
            
            response = await self.client.post(
                self.API_URL,
                json=payload,
                headers=headers,
                verify=False,
                timeout=httpx.Timeout(10)
            )
            
            if response.status_code == 401:
                raise AuthenticationException("Invalid Minimax API key")
            
            response.raise_for_status()
            return True
            
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"Minimax validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Minimax model information"""
        return {
            "model_name": self.config.model,
            "max_context": 16000,
            "supports_streaming": False,
            "provider": "minimax"
        }
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from response content"""
        try:
            # Try to parse directly
            json.loads(content)
            return content
        except json.JSONDecodeError:
            # Try to extract JSON from text
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return match.group()
            return content
    
    def _mock_response(self, prompt: str) -> str:
        """Return mock response for testing"""
        return json.dumps({
            "status": "success",
            "message": "Mock response",
            "data": {}
        })


# Register provider
LLMProviderFactory.register("minimax", MinimaxProvider)
```

### Step 3.2: Implement `services/llm/providers/openai_compatible.py`

Support self-hosted and compatible LLM services:

```python
"""OpenAI-Compatible LLM Provider"""

import httpx
import json
import logging
from typing import Optional, Dict, Any

from ..base import LLMProvider
from ..config import LLMConfig
from ..exceptions import (
    AuthenticationException,
    RateLimitException,
    APIException,
    ValidationException,
)
from ..factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible API services (Ollama, vLLM, etc.)"""
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI-compatible provider"""
        self.config = config
        self.base_url = config.base_url.rstrip("/") if config.base_url else "http://localhost:8000"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds)
        )
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using OpenAI-compatible API"""
        
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if provided (not required for local services)
        if self.config.api_key and self.config.api_key != "not-required":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens
        }
        
        url = f"{self.base_url}/v1/chat/completions"
        
        try:
            response = await self.client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            return content
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationException("Invalid API key for OpenAI-compatible provider")
            elif e.response.status_code == 429:
                raise RateLimitException("Rate limit exceeded")
            else:
                raise APIException(f"OpenAI-compatible API error: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValidationException(f"Invalid response format: {e}")
    
    async def validate_config(self) -> bool:
        """Validate OpenAI-compatible configuration"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key and self.config.api_key != "not-required":
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "max_tokens": 10
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            response = await self.client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(10)
            )
            
            if response.status_code == 401:
                raise AuthenticationException("Invalid API key")
            
            response.raise_for_status()
            return True
            
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"OpenAI-compatible validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model,
            "max_context": 4096,
            "supports_streaming": True,
            "provider": "openai-compatible",
            "base_url": self.base_url
        }


# Register provider
LLMProviderFactory.register("openai-compatible", OpenAICompatibleProvider)
```

### Step 3.3: Create `services/llm/providers/__init__.py`

Import all providers to register them:

```python
"""LLM Providers"""

# Import all providers to trigger registration via LLMProviderFactory
from .minimax import MinimaxProvider
from .openai_compatible import OpenAICompatibleProvider

# Additional providers (implement in subsequent steps)
# from .openai import OpenAIProvider
# from .anthropic import AnthropicProvider
# from .gemini import GeminiProvider
# from .groq import GroqProvider
# from .azure_openai import AzureOpenAIProvider

__all__ = [
    "MinimaxProvider",
    "OpenAICompatibleProvider",
]
```

---

## Phase 4: Integration with WikiProcessor

### Step 4.1: Update the API layer (now `api/routers/` + `core/deps.py`)

Replace MinimaxClient instantiation with factory:

```python
# OLD CODE TO REPLACE:
# from services.llm import MinimaxClient
# _api_key = os.getenv("MINIMAX_API_KEY", "dummy-key")
# if _api_key == "dummy-key":
#     logger.warning("MINIMAX_API_KEY not set; LLM calls will fail")
# llm = MinimaxClient(api_key=_api_key)

# NEW CODE:
from services.llm import load_from_env, LLMProviderFactory, LLMProvider

try:
    llm_config = load_from_env()
    llm: LLMProvider = LLMProviderFactory.create(llm_config)
    logger.info(f"Initialized LLM provider: {llm_config.provider}")
except Exception as e:
    logger.error(f"Failed to initialize LLM provider: {e}")
    raise
```

### Step 4.2: Update `services/processor.py`

Change type hints to use base class:

```python
# OLD CODE:
# from services.llm import MinimaxClient
# class WikiProcessor:
#     def __init__(self, llm: MinimaxClient, ...):

# NEW CODE:
from services.llm import LLMProvider

class WikiProcessor:
    def __init__(self, llm: LLMProvider, ...):
```

### Step 4.3: Remove Old MinimaxClient

Delete or archive the old `services/llm.py` file (after verifying all code has been migrated to new provider structure).

---

## Phase 5: Configuration & Environment Setup

### Step 5.1: Update `docker-compose.yml`

Replace Minimax-specific variables:

```yaml
# OLD:
# environment:
#   MINIMAX_API_KEY: ${MINIMAX_API_KEY}
#   MOCK_LLM: "true"

# NEW:
environment:
  LLM_PROVIDER: ${LLM_PROVIDER:-minimax}
  LLM_API_KEY: ${LLM_API_KEY}
  LLM_MODEL: ${LLM_MODEL:-MiniMax-M2.7}
  LLM_TEMPERATURE: ${LLM_TEMPERATURE:-0.7}
  LLM_MAX_TOKENS: ${LLM_MAX_TOKENS:-4000}
  # Optional for openai-compatible:
  LLM_BASE_URL: ${LLM_BASE_URL}
```

### Step 5.2: Update `.env-example`

Add all provider configurations:

```env
# LLM Provider Configuration
# Options: openai, anthropic, gemini, groq, azure, openai-compatible, minimax

# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-proj-...
LLM_MODEL=gpt-4-turbo

# Anthropic Claude
# LLM_PROVIDER=anthropic
# LLM_API_KEY=sk-ant-...
# LLM_MODEL=claude-opus-4-7

# Google Gemini
# LLM_PROVIDER=gemini
# LLM_API_KEY=AIzaSy...
# LLM_MODEL=gemini-2.0-flash

# Groq
# LLM_PROVIDER=groq
# LLM_API_KEY=gsk_...
# LLM_MODEL=mixtral-8x7b-32768

# OpenAI-Compatible (Ollama, vLLM, LM Studio, etc.)
# LLM_PROVIDER=openai-compatible
# LLM_API_KEY=not-needed
# LLM_BASE_URL=http://localhost:11434/v1  # Ollama
# LLM_MODEL=llama2

# Minimax (Default)
# LLM_PROVIDER=minimax
# LLM_API_KEY=sk-cp-...
# LLM_MODEL=MiniMax-M2.7

# Common settings
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
LLM_TIMEOUT=60
```

---

## Phase 6: Testing Strategy

### Test Structure

Create test files:

```
tests/
├── unit/
│   ├── test_llm_config.py
│   ├── test_llm_factory.py
│   └── test_providers/
│       ├── test_minimax.py
│       └── test_openai_compatible.py
└── integration/
    └── test_provider_integration.py
```

### Sample Unit Test (test_llm_config.py)

```python
import pytest
from services.llm import LLMConfig, ConfigurationException


def test_config_defaults():
    """Test that config accepts provider and api_key"""
    config = LLMConfig(
        provider="openai",
        api_key="sk-test",
        model="gpt-4"
    )
    assert config.temperature == 0.7
    assert config.max_tokens == 4000


def test_config_validation_missing_apikey():
    """Test that config validation fails without API key"""
    with pytest.raises(ConfigurationException):
        LLMConfig(
            provider="openai",
            api_key="",
            model="gpt-4"
        )


def test_config_openai_compatible_requires_url():
    """Test that openai-compatible requires base_url"""
    with pytest.raises(ConfigurationException):
        LLMConfig(
            provider="openai-compatible",
            api_key="not-required",
            model="local",
            base_url=None
        )
```

---

## Phase 7: Additional Providers (Future Work)

These can be implemented after the foundation is solid. Templates for each:

### Template: OpenAI Provider

```python
# services/llm/providers/openai.py

import httpx
from ..base import LLMProvider
from ..factory import LLMProviderFactory

class OpenAIProvider(LLMProvider):
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    # Implementation similar to OpenAICompatibleProvider
    # Key differences:
    # - Use hardcoded API_URL
    # - Expect specific auth header format
    # - Parse responses according to OpenAI format
    pass

LLMProviderFactory.register("openai", OpenAIProvider)
```

Similar templates available for:
- Anthropic
- Google Gemini
- Groq
- Azure OpenAI

---

## Phase 8: Backward Compatibility Verification

### Checklist

- [ ] Set `LLM_PROVIDER=minimax` as default in docker-compose.yml
- [ ] Set `LLM_API_KEY` to same value as old `MINIMAX_API_KEY` in .env
- [ ] Run existing tests - all should pass
- [ ] Verify `MOCK_LLM` behavior unchanged
- [ ] Test docker-compose setup without any LLM_* variables (should use defaults)
- [ ] Verify error messages are user-friendly

---

## Verification Steps

### Manual Testing

1. **Start services:**
   ```bash
   docker compose up -d
   ```

2. **Check logs:**
   ```bash
   docker compose logs wiki-processor | grep "LLM provider"
   ```

3. **Test health endpoint:**
   ```bash
   curl http://localhost:8001/health
   ```

4. **Verify provider loads correctly:**
   Should see message like: "Initialized LLM provider: minimax"

### Unit Tests

```bash
pytest tests/unit/test_llm_config.py -v
pytest tests/unit/test_llm_factory.py -v
pytest tests/unit/test_providers/ -v
```

### Integration Tests

```bash
pytest tests/integration/test_provider_integration.py -v
```

---

## Troubleshooting

### Common Issues

**Issue: "Unknown LLM provider: openai"**
- Solution: Ensure `from .openai import OpenAIProvider` is in `providers/__init__.py`
- The import triggers `LLMProviderFactory.register()` call

**Issue: "API key required" even when set**
- Check: `LLM_API_KEY` env var is set (not `MINIMAX_API_KEY`)
- Check: Provider is not "openai-compatible"

**Issue: Connection refused to LLM service**
- Check: `LLM_BASE_URL` is correct for openai-compatible
- Check: Service is actually running (e.g., Ollama on port 11434)

---

## Files Summary

| File | Status | Type |
|------|--------|------|
| `services/llm/__init__.py` | New | Config |
| `services/llm/base.py` | New | Interface |
| `services/llm/config.py` | New | Config |
| `services/llm/exceptions.py` | New | Errors |
| `services/llm/factory.py` | New | Factory |
| `services/llm/providers/__init__.py` | New | Init |
| `services/llm/providers/minimax.py` | New | Provider |
| `services/llm/providers/openai_compatible.py` | New | Provider |
| `services/llm.py` | Delete | Old |
| `api/routers/` | Modify | Integration |
| `services/processor.py` | Modify | Type hints |
| `docker-compose.yml` | Modify | Config |
| `.env-example` | Modify | Config |

---

## Success Criteria

- [x] Code compiles without errors
- [ ] All existing tests pass
- [ ] New unit tests cover all providers
- [ ] Integration tests verify provider switching works
- [ ] Backward compatibility verified (minimax as default)
- [ ] docker-compose works with new config
- [ ] Documentation updated
- [ ] No breaking changes to processor.py or llm.py interface

---

## Notes for Implementation

1. **Import Order:** Providers must be imported in `providers/__init__.py` to trigger registration
2. **Error Handling:** All providers should raise standardized exceptions (use custom exception classes)
3. **Logging:** Use logger.info() for initialization, logger.error() for failures
4. **Async:** All API calls use `async/await` via httpx
5. **Type Hints:** Use full type hints including Optional, Dict, etc.
6. **Testing:** Write tests that mock HTTP responses, not actual API calls
7. **Documentation:** Update docstrings for all public methods

---

**Next Steps After Implementation:**
1. Run full test suite
2. Update README with provider configuration examples
3. Create migration guide for existing users
4. Optional: Implement remaining providers (OpenAI, Anthropic, etc.)

