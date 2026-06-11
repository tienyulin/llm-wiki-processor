"""Tests for LLM provider abstraction."""
import os
import asyncio
import pytest

# Set mock mode before importing providers
os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("LLM_PROVIDER", "minimax")
os.environ.setdefault("MINIMAX_API_KEY", "test-key")

from services.llm import LLMProviderFactory, load_from_env
from services.llm.providers import MinimaxProvider, OpenAICompatibleProvider
from services.llm.config import LLMConfig
from services.llm.exceptions import ConfigurationException


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    config = LLMConfig(provider="minimax", api_key="key", model="MiniMax-M2.7")
    assert config.temperature == 0.7
    assert config.max_tokens == 4000


def test_config_openai_compatible_requires_base_url():
    with pytest.raises(ConfigurationException):
        LLMConfig(provider="openai-compatible", api_key="", model="local", base_url=None)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_factory_creates_minimax():
    config = LLMConfig(provider="minimax", api_key="key", model="MiniMax-M2.7")
    llm = LLMProviderFactory.create(config)
    assert isinstance(llm, MinimaxProvider)


def test_factory_creates_openai_compatible():
    config = LLMConfig(
        provider="openai-compatible",
        api_key="not-needed",
        model="llama2",
        base_url="http://localhost:11434",
    )
    llm = LLMProviderFactory.create(config)
    assert isinstance(llm, OpenAICompatibleProvider)


def test_factory_unknown_provider_raises():
    config = LLMConfig(provider="unknown-xyz", api_key="key", model="m")
    with pytest.raises(ConfigurationException):
        LLMProviderFactory.create(config)


def test_all_expected_providers_registered():
    available = LLMProviderFactory.available()
    for expected in ["minimax", "openai", "anthropic", "gemini", "groq", "azure", "openai-compatible"]:
        assert expected in available, f"Expected '{expected}' not registered"


# ---------------------------------------------------------------------------
# Minimax mock-mode tests
# ---------------------------------------------------------------------------

@pytest.fixture
def minimax_mock():
    os.environ["MOCK_LLM"] = "true"
    config = LLMConfig(provider="minimax", api_key="test-key", model="MiniMax-M2.7")
    return MinimaxProvider(config)


async def test_minimax_mock_generate_wiki(minimax_mock):
    result = await minimax_mock.generate_wiki({"api.md": "# Test API\nGET /health"})
    assert isinstance(result, dict)
    assert "apis" in result
    assert "metadata" in result


async def test_minimax_mock_update_wiki(minimax_mock):
    current = {"apis": {}, "metadata": {}}
    markdowns = {"api.md": "# Test\nGET /health"}
    result = await minimax_mock.update_wiki(current, markdowns, {"added": ["api.md"]})
    assert isinstance(result, dict)
    assert "apis" in result


def test_minimax_get_model_info(minimax_mock):
    info = minimax_mock.get_model_info()
    assert info["provider"] == "minimax"
    assert "model_name" in info


# ---------------------------------------------------------------------------
# extract_json helper
# ---------------------------------------------------------------------------

def test_extract_json_valid(minimax_mock):
    data = minimax_mock.extract_json('{"apis": {}, "metadata": {}}')
    assert data == {"apis": {}, "metadata": {}}


def test_extract_json_with_think_tags(minimax_mock):
    content = "<think>reasoning here</think>\n{\"apis\": {}}"
    data = minimax_mock.extract_json(content)
    assert "apis" in data


def test_extract_json_embedded(minimax_mock):
    content = 'Some text {"apis": {"mod": {}}} more text'
    data = minimax_mock.extract_json(content)
    assert "apis" in data
