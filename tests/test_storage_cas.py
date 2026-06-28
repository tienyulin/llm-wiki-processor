"""CAS (conditional write) tests for MinioStorage.

These run against a real MinIO server because they pin the behavior of the
private `Minio._put_object` header path — the hermetic Minio stub from
conftest.py cannot validate that. Skipped automatically when no server is
reachable on MINIO_ENDPOINT (default localhost:9000).

pylint: the storage/key fixtures are injected into tests under the same names
(redefined-outer-name) — standard pytest fixture convention.
"""

# pylint: disable=redefined-outer-name

import os
import socket
import uuid

import pytest
from minio import Minio

from repository.minio_client import MinioStorage

_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")


def _minio_reachable() -> bool:
    """True if a TCP connection to the MinIO endpoint succeeds."""
    host, _, port = _ENDPOINT.partition(":")
    try:
        with socket.create_connection((host, int(port or 9000)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _minio_reachable(), reason=f"no MinIO server at {_ENDPOINT}")


@pytest.fixture
def storage():
    """A MinioStorage backed by a real MinIO client against a test bucket."""
    # conftest stubs the Minio class inside repository.minio_client for hermetic
    # unit tests; rebuild a real client for these CAS tests.
    s = MinioStorage()
    s.client = Minio(
        _ENDPOINT,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    s.bucket = "cas-tests"
    if not s.client.bucket_exists(s.bucket):
        s.client.make_bucket(s.bucket)
    return s


@pytest.fixture
def key():
    """A unique object key per test, so cases don't collide."""
    return f"cas-{uuid.uuid4().hex}.json"


def test_get_json_with_etag_missing_key(storage, key):
    """A missing key reads back as (None, None)."""
    assert storage.get_json_with_etag(key) == (None, None)


def test_put_if_absent_then_conflict(storage, key):
    """put_json_if_absent succeeds once then fails on the existing key."""
    assert storage.put_json_if_absent(key, {"v": 1}) is True
    assert storage.put_json_if_absent(key, {"v": 2}) is False

    data, etag = storage.get_json_with_etag(key)
    assert data == {"v": 1}
    assert etag


def test_put_if_match_success_and_stale(storage, key):
    """put_json_if_match succeeds on a fresh etag and fails on a stale one."""
    storage.put_json_if_absent(key, {"v": 1})
    _, etag1 = storage.get_json_with_etag(key)

    assert storage.put_json_if_match(key, {"v": 2}, etag1) is True

    # The old etag is now stale
    assert storage.put_json_if_match(key, {"v": 3}, etag1) is False

    data, etag2 = storage.get_json_with_etag(key)
    assert data == {"v": 2}
    assert etag2 != etag1


async def test_async_facade_roundtrip(storage, key):
    """The async CAS facade round-trips absent/match writes and reads."""
    assert await storage.aput_json_if_absent(key, {"v": 1}) is True
    data, etag = await storage.aget_json_with_etag(key)
    assert data == {"v": 1}
    assert await storage.aput_json_if_match(key, {"v": 2}, etag) is True
    assert await storage.aget_json(key) == {"v": 2}
