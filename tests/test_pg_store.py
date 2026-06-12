"""PGVectorStore tests against a real Postgres+pgvector server.

Like test_storage_cas.py these need real infrastructure (HNSW indexes,
transactional semantics, multi-host DSN failover can't be stubbed) and skip
automatically when nothing is listening on PG_TEST_DSN's first host
(default localhost:5432). Start one with:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pg -e POSTGRES_DB=wiki \
        pgvector/pgvector:pg16
"""
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from services.embeddings import mock_embed
from services.llm.exceptions import ConfigurationException
from repository.pg_store import PGVectorStore, to_vector_literal

_DSN = os.getenv("PG_TEST_DSN", "postgresql://postgres:pg@localhost:5432/wiki")
_DIM = 32


def _pg_reachable() -> bool:
    hostspec = _DSN.split("@")[-1].split("/")[0].split(",")[0]
    host, _, port = hostspec.partition(":")
    try:
        with socket.create_connection((host, int(port or 5432)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _pg_reachable(), reason=f"no Postgres server for {_DSN}"
)


def _rows(n=3, module="inventory", offset=0):
    rows = []
    for i in range(offset, offset + n):
        api_key = f"GET /{module}/thing{i}"
        detail = {"method": "GET", "path": f"/{module}/thing{i}", "description": f"Get thing {i}"}
        text = f"{module} | {api_key} | Get thing {i}"
        rows.append({
            "module": module,
            "api_key": api_key,
            "description": f"Get thing {i}",
            "detail": detail,
            "embed_text": text,
            "embedding": mock_embed(text, _DIM),
            "embedding_model": "mock",
        })
    return rows


@pytest.fixture
async def store():
    s = PGVectorStore(_DSN, min_size=1, max_size=2)
    await s._ensure_open()
    async with s._pool.connection() as conn:
        await conn.execute("DROP TABLE IF EXISTS api_entries, index_state, app_sync CASCADE")
        await conn.commit()
    await s.ensure_schema(_DIM)
    yield s
    await s.aclose()


def _now():
    return datetime.now(timezone.utc)


async def test_ensure_schema_idempotent(store):
    await store.ensure_schema(_DIM)  # second run must be a no-op
    assert await store.count_entries() == (0, 0)


async def test_ensure_schema_rejects_dim_change(store):
    with pytest.raises(ConfigurationException, match="EMBEDDING_DIM"):
        await store.ensure_schema(_DIM * 2)


async def test_replace_app_entries_roundtrip(store):
    assert await store.replace_app_entries("app-a", "v1", _rows(3), _now())
    assert await store.count_entries() == (3, 3)

    # Re-sync with fewer entries: old rows for this app must disappear.
    assert await store.replace_app_entries("app-a", "v2", _rows(1, offset=10), _now())
    total, embedded = await store.count_entries()
    assert (total, embedded) == (1, 1)


async def test_replace_preserves_other_apps(store):
    await store.replace_app_entries("app-a", "v1", _rows(2, module="inventory"), _now())
    await store.replace_app_entries("app-b", "v1", _rows(2, module="billing"), _now())
    await store.replace_app_entries("app-a", "v2", _rows(1, module="inventory", offset=5), _now())
    total, _ = await store.count_entries()
    assert total == 3  # 1 inventory (app-a) + 2 billing (app-b)


async def test_key_ownership_transfer(store):
    """If another app re-publishes the same (module, api_key), last writer
    wins — mirrors the dict-merge semantics of _merge_app_entries."""
    await store.replace_app_entries("app-a", "v1", _rows(1), _now())
    await store.replace_app_entries("app-b", "v1", _rows(1), _now())  # same module/key
    total, _ = await store.count_entries()
    assert total == 1
    results = await store.semantic_search(mock_embed("inventory thing0", _DIM), top_k=1)
    assert results[0]["source_app"] == "app-b"


async def test_stale_sync_rejected(store):
    now = _now()
    assert await store.replace_app_entries("app-a", "v2", _rows(2), now)
    # A replay carrying an older timestamp must not clobber newer data.
    stale = now - timedelta(seconds=5)
    assert not await store.replace_app_entries("app-a", "v1", _rows(5, offset=20), stale)
    total, _ = await store.count_entries()
    assert total == 2


async def test_semantic_search_ordering(store):
    rows = []
    for module, key, text in [
        ("inventory", "GET /inventory/health", "inventory health check status"),
        ("auth", "POST /auth/login", "user login authentication token"),
        ("billing", "GET /billing/invoice", "fetch invoice for billing account"),
    ]:
        rows.append({
            "module": module, "api_key": key, "description": text,
            "detail": {"description": text}, "embed_text": text,
            "embedding": mock_embed(text, _DIM), "embedding_model": "mock",
        })
    await store.replace_app_entries("app-a", "v1", rows, _now())

    results = await store.semantic_search(mock_embed("inventory health", _DIM), top_k=3)
    assert results[0]["api_key"] == "GET /inventory/health"
    assert results[0]["score"] > results[-1]["score"]
    assert all(0.0 <= r["score"] <= 1.0001 for r in results)


async def test_null_embeddings_still_searchable_relationally(store):
    rows = _rows(2)
    for r in rows:
        r["embedding"] = None  # embeddings endpoint was down
    await store.replace_app_entries("app-a", "v1", rows, _now())
    assert await store.count_entries() == (2, 0)
    # NULL vectors are excluded from ANN, not an error.
    assert await store.semantic_search(mock_embed("anything", _DIM)) == []


async def test_rebuild_replaces_everything(store):
    await store.replace_app_entries("app-old", "v1", _rows(4), _now())
    total = await store.rebuild(
        {"app-a": _rows(2, module="m1"), "app-b": _rows(3, module="m2")},
        {"app-a": "v1", "app-b": "v2"},
    )
    assert total == 5
    counted, _ = await store.count_entries()
    assert counted == 5


async def test_multihost_dsn_skips_dead_host():
    """The HA-failover contract (future clusters need no code change):
    with target_session_attrs=read-write,
    libpq walks the host list until it finds a writable server, so a dead
    (or demoted) first host is skipped transparently."""
    user_info, _, rest = _DSN.partition("@")
    hostspec, _, dbpart = rest.partition("/")
    multi = f"{user_info}@localhost:9,{hostspec}/{dbpart}"
    sep = "&" if "?" in multi else "?"
    multi += f"{sep}target_session_attrs=read-write&connect_timeout=3"

    s = PGVectorStore(multi, min_size=1, max_size=1)
    try:
        assert await s.available() is True
    finally:
        await s.aclose()


def test_to_vector_literal():
    assert to_vector_literal(None) is None
    assert to_vector_literal([1, 2.5, -0.5]) == "[1.0,2.5,-0.5]"
