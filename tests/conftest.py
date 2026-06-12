"""Shared test setup: hermetic env defaults and a stubbed Minio SDK.

Dependencies are built lazily via core/deps.py providers, so there is no
import-order constraint anymore — but the env defaults and the Minio stub
must still be in place before the first request constructs the singletons.

Tests inject mocks with app.dependency_overrides; the autouse fixture below
clears overrides and cached singletons between tests.
"""
import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("MOCK_EMBEDDINGS", "true")

# Stub the Minio SDK class so MinioStorage() never opens a connection.
import repository.minio_client as _minio_client  # noqa: E402

_minio_client.Minio = MagicMock()


@pytest.fixture(autouse=True)
def _reset_dependency_state():
    """Fresh overrides and singletons for every test."""
    from core import deps
    from main import app

    app.dependency_overrides.clear()
    deps.reset_singletons()
    yield
    app.dependency_overrides.clear()
    deps.reset_singletons()
