"""Shared test setup: hermetic env defaults and a stubbed Minio SDK.

api/routes.py builds its storage/LLM singletons at import time, so the env
vars and the Minio stub must be in place before any test module imports
`main`. conftest.py is imported first during collection, which guarantees
that ordering.
"""
import os
from unittest.mock import MagicMock

os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("LLM_API_KEY", "test-key")

# Stub the Minio SDK class so MinioStorage() never opens a connection.
# Tests that exercise storage behavior patch `api.routes.storage` directly.
import storage.minio_client as _minio_client  # noqa: E402

_minio_client.Minio = MagicMock()
