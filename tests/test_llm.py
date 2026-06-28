"""Tests for LLM provider abstraction.

pylint: pytest conventions — fixture injection reuses fixture names as test
arguments (redefined-outer-name); provider imports follow a deliberate
env-var setup so they read mock config (wrong-import-position).
"""

# pylint: disable=redefined-outer-name,wrong-import-position

import os
import pytest

# Set mock mode before importing providers
os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("LLM_PROVIDER", "minimax")
os.environ.setdefault("MINIMAX_API_KEY", "test-key")

from services.llm import LLMProviderFactory  # noqa: E402  (env set above first)
from services.llm.providers import (  # noqa: E402
    MinimaxProvider,
    OpenAICompatibleProvider,
)
from services.llm.config import LLMConfig  # noqa: E402
from services.llm.exceptions import ConfigurationException  # noqa: E402

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Unspecified temperature/max_tokens fall back to documented defaults."""
    config = LLMConfig(provider="minimax", api_key="key", model="MiniMax-M2.7")
    assert config.temperature == 0.7
    assert config.max_tokens == 4000


def test_config_openai_compatible_requires_base_url():
    """openai-compatible without a base_url is rejected at config time."""
    with pytest.raises(ConfigurationException):
        LLMConfig(provider="openai-compatible", api_key="", model="local", base_url=None)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


def test_factory_creates_minimax():
    """The factory builds a MinimaxProvider for provider='minimax'."""
    config = LLMConfig(provider="minimax", api_key="key", model="MiniMax-M2.7")
    llm = LLMProviderFactory.create(config)
    assert isinstance(llm, MinimaxProvider)


def test_factory_creates_openai_compatible():
    """The factory builds an OpenAICompatibleProvider for that provider."""
    config = LLMConfig(
        provider="openai-compatible",
        api_key="not-needed",
        model="llama2",
        base_url="http://localhost:11434",
    )
    llm = LLMProviderFactory.create(config)
    assert isinstance(llm, OpenAICompatibleProvider)


def test_factory_unknown_provider_raises():
    """An unregistered provider name raises ConfigurationException."""
    config = LLMConfig(provider="unknown-xyz", api_key="key", model="m")
    with pytest.raises(ConfigurationException):
        LLMProviderFactory.create(config)


def test_all_expected_providers_registered():
    """All seven shipped providers are registered with the factory."""
    available = LLMProviderFactory.available()
    for expected in [
        "minimax",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "azure",
        "openai-compatible",
    ]:
        assert expected in available, f"Expected '{expected}' not registered"


# ---------------------------------------------------------------------------
# Minimax mock-mode tests
# ---------------------------------------------------------------------------


@pytest.fixture
def minimax_mock():
    """A MinimaxProvider pinned to mock mode for deterministic, network-free tests."""
    os.environ["MOCK_LLM"] = "true"
    config = LLMConfig(provider="minimax", api_key="test-key", model="MiniMax-M2.7")
    return MinimaxProvider(config)


async def test_minimax_mock_generate_wiki(minimax_mock):
    """Mock-mode generate_wiki returns a wiki dict with apis/metadata."""
    result = await minimax_mock.generate_wiki({"api.md": "# Test API\nGET /health"})
    assert isinstance(result, dict)
    assert "apis" in result
    assert "metadata" in result


async def test_minimax_mock_update_wiki(minimax_mock):
    """Mock-mode update_wiki merges changes and returns a wiki dict."""
    current: dict = {"apis": {}, "metadata": {}}
    markdowns = {"api.md": "# Test\nGET /health"}
    result = await minimax_mock.update_wiki(current, markdowns, {"added": ["api.md"]})
    assert isinstance(result, dict)
    assert "apis" in result


def test_minimax_get_model_info(minimax_mock):
    """get_model_info reports the minimax provider and a model name."""
    info = minimax_mock.get_model_info()
    assert info["provider"] == "minimax"
    assert "model_name" in info


# ---------------------------------------------------------------------------
# extract_json helper
# ---------------------------------------------------------------------------


def test_extract_json_valid(minimax_mock):
    """extract_json parses a clean JSON object."""
    data = minimax_mock.extract_json('{"apis": {}, "metadata": {}}')
    assert data == {"apis": {}, "metadata": {}}


def test_extract_json_with_think_tags(minimax_mock):
    """extract_json ignores leading <think> reasoning before the JSON."""
    content = '<think>reasoning here</think>\n{"apis": {}}'
    data = minimax_mock.extract_json(content)
    assert "apis" in data


def test_extract_json_embedded(minimax_mock):
    """extract_json recovers a JSON object embedded in surrounding text."""
    content = 'Some text {"apis": {"mod": {}}} more text'
    data = minimax_mock.extract_json(content)
    assert "apis" in data
