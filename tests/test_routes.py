"""Tests for FastAPI routes using TestClient.

Singletons are provided lazily via core.deps; tests inject mocks with
app.dependency_overrides (cleared by the autouse fixture in conftest.py).
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from core import deps
from main import app
from models.schemas import ProcessResponse

client = TestClient(app)


@contextmanager
def override(dep, replacement):
    """Context manager that overrides a FastAPI dependency, restoring it on exit."""
    app.dependency_overrides[dep] = lambda: replacement
    try:
        yield replacement
    finally:
        app.dependency_overrides.pop(dep, None)


@contextmanager
def mock_processor_override():
    """Override get_processor with a MagicMock processor for the duration."""
    mock_processor = MagicMock()
    with override(deps.get_processor, mock_processor):
        yield mock_processor


def test_health_returns_200_with_expected_keys():
    """/health returns 200 with all status fields incl. the backward-compat alias."""
    mock_storage = MagicMock()
    mock_storage.ping.return_value = True
    with override(deps.get_storage, mock_storage):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "minio_connected" in body
    # New fields
    assert "llm_configured" in body
    assert "llm_provider" in body
    # Backward-compat alias
    assert "minimax_accessible" in body
    # Vector-index layer (disabled in tests: no PG_DSN)
    assert body["vector_index_connected"] is False
    assert body["embeddings_configured"] is True  # MOCK_EMBEDDINGS=true in conftest


def test_status_returns_expected_shape():
    """/status reports wiki size, tracked files, and last-updated from storage."""
    fake_wiki = {
        "apis": {"users": {}, "orders": {}},
        "metadata": {"updated_at": "2024-01-01T00:00:00"},
    }
    fake_snapshot = {"a.md": "...", "b.md": "..."}

    mock_storage = MagicMock()
    mock_storage.get_json.side_effect = lambda key: (
        fake_wiki if key == "wiki.json" else fake_snapshot
    )
    with override(deps.get_storage, mock_storage):
        resp = client.get("/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["wiki_size"] == 2
    assert body["tracked_files"] == 2
    assert body["last_updated"] == "2024-01-01T00:00:00"


def test_process_returns_200_and_response_body():
    """/process returns the processor's ProcessResponse body on success."""
    fake_response = ProcessResponse(
        status="success",
        message="Wiki generated successfully",
        wiki_url="minio://wiki-data/wiki.json",
        changes_summary={"added": ["api.md"], "modified": [], "deleted": []},
        timestamp="2024-01-01T00:00:00",
    )

    with mock_processor_override() as mock_processor:
        mock_processor.process = AsyncMock(return_value=fake_response)
        resp = client.post(
            "/process",
            json={
                "markdowns": {"api.md": "# API docs"},
                "timestamp": "2024-01-01T00:00:00",
                "trigger_info": {"branch": "main"},
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["wiki_url"] == "minio://wiki-data/wiki.json"
    assert "api.md" in body["changes_summary"]["added"]


def test_process_rejects_empty_markdowns():
    """/process rejects an empty markdowns map with 422."""
    resp = client.post(
        "/process",
        json={
            "markdowns": {},
            "timestamp": "2024-01-01T00:00:00",
            "trigger_info": {"branch": "main"},
        },
    )
    assert resp.status_code == 422


_PAYLOAD = {
    "markdowns": {"api.md": "# API docs"},
    "timestamp": "2024-01-01T00:00:00",
    "trigger_info": {"branch": "main"},
}


def test_process_requires_api_key_when_configured(monkeypatch):
    """/process returns 401 for a missing or wrong key when one is configured."""
    monkeypatch.setenv("PROCESSOR_API_KEY", "secret-key")

    # Missing header
    assert client.post("/process", json=_PAYLOAD).status_code == 401
    # Wrong key
    resp = client.post("/process", json=_PAYLOAD, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_process_accepts_valid_api_key(monkeypatch):
    """/process accepts the correct X-API-Key when one is configured."""
    monkeypatch.setenv("PROCESSOR_API_KEY", "secret-key")
    fake_response = ProcessResponse(
        status="success",
        message="ok",
        timestamp="2024-01-01T00:00:00",
    )
    with mock_processor_override() as mock_processor:
        mock_processor.process = AsyncMock(return_value=fake_response)
        resp = client.post("/process", json=_PAYLOAD, headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200


def test_reindex_503_when_pg_disabled(monkeypatch):
    """/admin/reindex returns 503 when the vector store is disabled."""
    monkeypatch.delenv("PROCESSOR_API_KEY", raising=False)
    resp = client.post("/admin/reindex")
    assert resp.status_code == 503
    assert "PG_DSN" in resp.json()["detail"]


def test_reindex_requires_api_key(monkeypatch):
    """/admin/reindex returns 401 without the configured API key."""
    monkeypatch.setenv("PROCESSOR_API_KEY", "secret-key")
    assert client.post("/admin/reindex").status_code == 401


def test_reindex_runs_when_pg_enabled():
    """/admin/reindex runs and echoes the reindex result when PG is enabled."""
    with mock_processor_override() as mock_processor:
        mock_processor.vector_store = MagicMock()  # not None => enabled
        mock_processor.reindex = AsyncMock(return_value={"apps": 2, "entries": 5, "embedded": 5})
        resp = client.post("/admin/reindex")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "apps": 2, "entries": 5, "embedded": 5}


def test_process_open_when_auth_disabled(monkeypatch):
    """/process is open (200) when no PROCESSOR_API_KEY is configured."""
    monkeypatch.delenv("PROCESSOR_API_KEY", raising=False)
    fake_response = ProcessResponse(
        status="success",
        message="ok",
        timestamp="2024-01-01T00:00:00",
    )
    with mock_processor_override() as mock_processor:
        mock_processor.process = AsyncMock(return_value=fake_response)
        resp = client.post("/process", json=_PAYLOAD)
    assert resp.status_code == 200
