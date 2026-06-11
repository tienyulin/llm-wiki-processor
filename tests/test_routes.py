"""Tests for FastAPI routes using TestClient."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from models.schemas import ProcessResponse

client = TestClient(app)


def test_health_returns_200_with_expected_keys():
    with patch("api.routes.storage") as mock_storage:
        mock_storage.client.bucket_exists.return_value = True
        mock_storage.bucket = "wiki-data"
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


def test_status_returns_expected_shape():
    fake_wiki = {
        "apis": {"users": {}, "orders": {}},
        "metadata": {"updated_at": "2024-01-01T00:00:00"},
    }
    fake_snapshot = {"a.md": "...", "b.md": "..."}

    with patch("api.routes.storage") as mock_storage:
        mock_storage.get_json.side_effect = lambda key: (
            fake_wiki if key == "wiki.json" else fake_snapshot
        )
        resp = client.get("/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["wiki_size"] == 2
    assert body["tracked_files"] == 2
    assert body["last_updated"] == "2024-01-01T00:00:00"


def test_process_returns_200_and_response_body():
    fake_response = ProcessResponse(
        status="success",
        message="Wiki generated successfully",
        wiki_url="minio://wiki-data/wiki.json",
        changes_summary={"added": ["api.md"], "modified": [], "deleted": []},
        timestamp="2024-01-01T00:00:00",
    )

    with patch("api.routes.processor") as mock_processor:
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
    resp = client.post(
        "/process",
        json={
            "markdowns": {},
            "timestamp": "2024-01-01T00:00:00",
            "trigger_info": {"branch": "main"},
        },
    )
    assert resp.status_code == 422
