"""Tests for MinimaxClient.extract_json (no network, no async)."""
import pytest
from services.llm import MinimaxClient


@pytest.fixture
def client():
    return MinimaxClient(api_key="test-key")


def test_clean_json_string(client):
    raw = '{"apis": {"users": {}}, "metadata": {}}'
    result = client.extract_json(raw)
    assert result == {"apis": {"users": {}}, "metadata": {}}


def test_json_with_think_tags(client):
    raw = "<think>Let me think about this...</think>\n{\"apis\": {}, \"metadata\": {}}"
    result = client.extract_json(raw)
    assert result == {"apis": {}, "metadata": {}}


def test_json_embedded_in_text(client):
    raw = 'Here is the wiki:\n{"apis": {"orders": {}}, "metadata": {}}\nEnd of response.'
    result = client.extract_json(raw)
    assert result == {"apis": {"orders": {}}, "metadata": {}}


def test_json_with_multiline_think_tags(client):
    raw = (
        "<think>\n"
        "I need to parse these markdowns carefully.\n"
        "Step 1: identify endpoints.\n"
        "</think>\n"
        '{"apis": {"payments": {"POST /pay": {}}}, "metadata": {"version": "1.0"}}'
    )
    result = client.extract_json(raw)
    assert result["apis"]["payments"]["POST /pay"] == {}
    assert result["metadata"]["version"] == "1.0"
