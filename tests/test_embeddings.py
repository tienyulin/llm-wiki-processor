"""Embedding module tests: mock determinism, text builder, client behavior.

All hermetic — the real-HTTP path is exercised through httpx.MockTransport.

pylint: tests white-box the client (protected-access on _embed_chunk) and the
MockTransport handlers ignore their request arg (unused-argument).
"""

# pylint: disable=protected-access,unused-argument

import json
import math

import httpx
import pytest

from services.embeddings import EmbeddingClient, EmbeddingConfig, entry_to_text, mock_embed
from services.embeddings.config import RequestTuning, load_embedding_env
from services.llm.exceptions import (
    AuthenticationException,
    ConfigurationException,
    RateLimitException,
    ValidationException,
)

# ---------------------------------------------------------------------------
# mock_embed
# ---------------------------------------------------------------------------


def _dot(a, b):
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def test_mock_embed_deterministic():
    """mock_embed is deterministic for the same text and dim."""
    assert mock_embed("hello world", 64) == mock_embed("hello world", 64)


def test_mock_embed_normalized():
    """mock_embed returns a unit-length vector."""
    vec = mock_embed("some text with several tokens", 128)
    assert math.isclose(math.sqrt(sum(c * c for c in vec)), 1.0, rel_tol=1e-9)


def test_mock_embed_empty_text_is_unit_vector():
    """Empty text maps to the canonical [1, 0, ...] unit vector."""
    vec = mock_embed("", 16)
    assert vec[0] == 1.0 and all(c == 0.0 for c in vec[1:])


def test_mock_embed_token_overlap_ranks_higher():
    """Shared tokens => higher cosine similarity; that's what makes mock
    semantic search assertable in integration tests."""
    query = mock_embed("inventory health", 1536)
    related = mock_embed("inventory health check status", 1536)
    unrelated = mock_embed("user login authentication token", 1536)
    assert _dot(query, related) > _dot(query, unrelated)
    assert _dot(query, related) > 0.5


def test_mock_embed_golden_values():
    """Pins the exact algorithm. mcp-server/tests/test_embeddings.py has the
    same golden test — if this fails after an intentional change, update the
    mcp-server copy of mock_embed and BOTH golden tests together."""
    vec = mock_embed("inventory | GET /inventory/health | Inventory Health Check", 8)
    expected = [0.872872, 0.0, 0.0, 0.218218, 0.0, 0.0, 0.0, -0.436436]
    assert vec == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# entry_to_text
# ---------------------------------------------------------------------------


def test_entry_to_text_full_entry():
    """entry_to_text joins description, module, key, and params."""
    text = entry_to_text(
        "inventory",
        "GET /inventory/{id}",
        {
            "method": "GET",
            "path": "/inventory/{id}",
            "description": "Get one item",
            "parameters": {"id": "string"},
        },
    )
    assert text == ('Get one item | inventory | GET /inventory/{id} | {"id": "string"}')


def test_entry_to_text_drops_empty_parts():
    """entry_to_text omits empty fields from the joined text."""
    assert (
        entry_to_text("mod", "DOC readme.md", {"description": "Intro"})
        == "Intro | mod | DOC readme.md"
    )


def test_entry_to_text_non_dict_detail():
    """entry_to_text handles a non-dict detail by stringifying it."""
    assert entry_to_text("mod", "KEY", "plain string") == "plain string | mod | KEY"


def test_entry_to_text_truncates_parameters():
    """entry_to_text caps the serialized params at the size limit."""
    big_params = {f"field_{i}": "x" * 50 for i in range(50)}
    text = entry_to_text("mod", "KEY", {"description": "d", "parameters": big_params})
    params_part = text.rsplit(" | ", maxsplit=1)[-1]
    assert len(params_part) <= 500


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_disabled_by_default(monkeypatch):
    """With no base URL or mock flag, embeddings are disabled."""
    for var in ("EMBEDDING_BASE_URL", "MOCK_EMBEDDINGS"):
        monkeypatch.delenv(var, raising=False)
    assert load_embedding_env().is_enabled() is False


def test_config_mock_mode_counts_as_enabled(monkeypatch):
    """MOCK_EMBEDDINGS=true counts as enabled."""
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.setenv("MOCK_EMBEDDINGS", "true")
    cfg = load_embedding_env()
    assert cfg.is_enabled() is True and cfg.mock_mode is True


def test_config_base_url_enables_and_strips_slash(monkeypatch):
    """A base URL enables embeddings and its trailing slash is stripped."""
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("MOCK_EMBEDDINGS", "false")
    cfg = load_embedding_env()
    assert cfg.is_enabled() is True and cfg.base_url == "http://localhost:11434"


def test_config_rejects_bad_dim():
    """A non-positive embedding dim is rejected at config time."""
    with pytest.raises(ConfigurationException):
        EmbeddingConfig(dim=0)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


async def test_client_mock_mode_no_network():
    """Mock-mode aembed returns deterministic vectors with no network."""
    client = EmbeddingClient(EmbeddingConfig(mock_mode=True, dim=32))
    vecs = await client.aembed(["a b c", "d e f"])
    assert len(vecs) == 2 and all(len(v) == 32 for v in vecs)
    assert vecs[0] == mock_embed("a b c", 32)


async def test_client_empty_input():
    """aembed([]) short-circuits to an empty list."""
    client = EmbeddingClient(EmbeddingConfig(mock_mode=True, dim=8))
    assert await client.aembed([]) == []


def _transport_client(handler, **config_kwargs):
    """EmbeddingClient whose HTTP layer is a MockTransport."""
    tuning_kwargs = {
        k: config_kwargs.pop(k) for k in ("batch_size", "timeout_seconds") if k in config_kwargs
    }
    if tuning_kwargs:
        config_kwargs["request"] = RequestTuning(**tuning_kwargs)
    config = EmbeddingConfig(base_url="http://embed.test", dim=4, **config_kwargs)
    client = EmbeddingClient(config)
    transport = httpx.MockTransport(handler)

    async def _embed_with_transport(texts):
        vectors = []
        async with httpx.AsyncClient(transport=transport) as http:
            for start in range(0, len(texts), config.batch_size):
                chunk = texts[start : start + config.batch_size]
                vectors.extend(await client._embed_chunk(http, chunk))
        return vectors

    return client, _embed_with_transport


async def test_client_batches_and_preserves_order():
    """Batched requests preserve per-chunk index order in the output."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        # Return items deliberately reversed: "index" must be authoritative.
        data = [
            {"index": i, "embedding": [float(len(calls)), float(i), 0.0, 0.0]}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    _, embed = _transport_client(handler, batch_size=2)
    vecs = await embed(["t0", "t1", "t2"])
    assert calls == [["t0", "t1"], ["t2"]]
    assert [v[1] for v in vecs] == [0.0, 1.0, 0.0]  # per-chunk index order restored


async def test_client_maps_auth_and_rate_limit_errors():
    """401/429 responses map to AuthenticationException/RateLimitException."""
    _, embed_401 = _transport_client(lambda r: httpx.Response(401))
    with pytest.raises(AuthenticationException):
        await embed_401(["x"])

    _, embed_429 = _transport_client(lambda r: httpx.Response(429))
    with pytest.raises(RateLimitException):
        await embed_429(["x"])


async def test_client_rejects_dim_mismatch():
    """A wrong-dimension response raises ValidationException."""

    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    _, embed = _transport_client(handler)  # config dim=4, response dim=2
    with pytest.raises(ValidationException):
        await embed(["x"])


async def test_client_rejects_count_mismatch():
    """A vector-count mismatch raises ValidationException."""

    def handler(request):
        return httpx.Response(200, json={"data": []})

    _, embed = _transport_client(handler)
    with pytest.raises(ValidationException):
        await embed(["x"])
