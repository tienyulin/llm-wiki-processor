"""Concurrency + per-app-object regression tests for WikiProcessor.

P3: each app's source of truth is its own object apps/<app>.json. Different apps
write different keys, so concurrent updates never contend — no lost writes by
construction. Same-app concurrent pushes are still guarded by ETag CAS. These
tests use an in-memory CAS storage whose every op yields control to force
coroutine interleaving.
"""

# pytest white-box conventions: tests poke storage internals (protected-access)
# and monkeypatch bound methods on the fake (method assignment).
# pylint: disable=protected-access

import asyncio
import uuid

from services.llm.config import LLMConfig
from services.llm.providers.minimax import MinimaxProvider
from services.processor import WikiProcessor, _AUDIT_PREFIX, _app_key


class InMemoryCASStorage:
    """Async, etag-aware stand-in for MinioStorage's facade."""

    def __init__(self):
        self.data: dict = {}
        self.etags: dict = {}

    def _put(self, key, value):
        """Store value under key and assign it a fresh etag."""
        self.data[key] = value
        self.etags[key] = uuid.uuid4().hex

    async def aget_json(self, key):
        """Async read mirroring MinioStorage.aget_json."""
        await asyncio.sleep(0)
        return self.data.get(key)

    async def aput_json(self, key, value):
        """Async unconditional write."""
        await asyncio.sleep(0)
        self._put(key, value)

    async def aget_json_with_etag(self, key):
        """Return (value, etag) or (None, None) when absent."""
        await asyncio.sleep(0)
        if key not in self.data:
            return None, None
        return self.data[key], self.etags[key]

    async def aput_json_if_match(self, key, value, etag):
        """Conditional write that succeeds only on an etag match."""
        await asyncio.sleep(0)
        if self.etags.get(key) != etag:
            return False
        self._put(key, value)
        return True

    async def aput_json_if_absent(self, key, value):
        """Conditional write that succeeds only when the key is absent."""
        await asyncio.sleep(0)
        if key in self.data:
            return False
        self._put(key, value)
        return True

    async def alist_files(self, prefix=""):
        """List keys under a prefix."""
        await asyncio.sleep(0)
        return [k for k in self.data if k.startswith(prefix)]

    def audit_entries(self):
        """Return all stored audit-log entries."""
        return [v for k, v in self.data.items() if k.startswith(_AUDIT_PREFIX)]


def make_processor(storage):
    """WikiProcessor wired to a mock LLM and the given storage."""
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="test-key", model="m"))
    return WikiProcessor(storage=storage, llm=llm)


def app_markdown(app: str) -> dict[str, str]:
    """Minimal one-endpoint README for the named app."""
    return {f"{app}_api.md": f"# {app} API\n\nGET /{app}/items\n"}


async def test_concurrent_app_updates_have_no_lost_writes():
    """20 apps pushing at once each land their own object — no lost writes."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    apps = [f"app-{i:02d}" for i in range(20)]
    results = await asyncio.gather(
        *[
            processor.process(app_markdown(app), "t", source_app=app, source_version="v1.0.0")
            for app in apps
        ]
    )
    assert all(r.status == "success" for r in results), [r.message for r in results]

    # Each app's object holds its own entry — no shared blob to lose writes on.
    for app in apps:
        obj = storage.data[_app_key(app)]
        assert obj["source_app"] == app
        assert f"GET /{app}/items" in obj["apis"][app]
    assert len(storage.audit_entries()) == len(apps)


async def test_resubmission_replaces_only_that_apps_object():
    """Re-pushing one app replaces only its object, leaving others intact."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    await processor.process(app_markdown("app-a"), "t", source_app="app-a", source_version="v1")
    await processor.process(app_markdown("app-b"), "t", source_app="app-b", source_version="v1")

    new_md = {"app-a_api.md": "# app-a API v2\n\nPOST /app-a/orders\n"}
    assert (
        await processor.process(new_md, "t", source_app="app-a", source_version="v2")
    ).status == "success"

    a = storage.data[_app_key("app-a")]["apis"]["app-a"]
    assert "POST /app-a/orders" in a and "GET /app-a/items" not in a
    assert a["POST /app-a/orders"]["source_version"] == "v2"
    # app-b's object untouched
    assert "GET /app-b/items" in storage.data[_app_key("app-b")]["apis"]["app-b"]


async def test_unchanged_resubmission_skips_write():
    """An identical resubmission detects no changes and skips the write."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    md = app_markdown("app-a")
    await processor.process(md, "t", source_app="app-a", source_version="v1")
    etag_before = storage.etags[_app_key("app-a")]

    result = await processor.process(md, "t", source_app="app-a", source_version="v1")
    assert result.status == "success" and "No changes" in result.message
    assert storage.etags[_app_key("app-a")] == etag_before  # object not rewritten


async def test_same_app_cas_conflict_is_retried():
    """A concurrent writer to the SAME app's object between read and write must
    not lose data — the CAS loop re-reads and retries."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)
    await processor.process(app_markdown("app-x"), "t", source_app="app-x", source_version="v1")

    original = storage.aput_json_if_match
    injected = False

    async def conflicting(key, value, etag):
        nonlocal injected
        if not injected and key == _app_key("app-x"):
            injected = True
            storage._put(key, {**storage.data[key], "marker": "intruder"})  # sneak a write
        return await original(key, value, etag)

    storage.aput_json_if_match = conflicting  # type: ignore[method-assign]
    new_md = {"app-x_api.md": "# app-x v2\n\nPOST /app-x/orders\n"}
    result = await processor.process(new_md, "t", source_app="app-x", source_version="v2")
    assert result.status == "success" and injected
    # app-x's update landed despite the mid-flight conflict.
    assert "POST /app-x/orders" in storage.data[_app_key("app-x")]["apis"]["app-x"]


async def test_aggregate_apps_merges_all_objects():
    """aggregate_apps merges every per-app object into one wiki view."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)
    await processor.process(app_markdown("app-a"), "t", source_app="app-a", source_version="v1")
    await processor.process(app_markdown("app-b"), "t", source_app="app-b", source_version="v1")

    agg = await processor.aggregate_apps()
    assert "GET /app-a/items" in agg["apis"]["app-a"]
    assert "GET /app-b/items" in agg["apis"]["app-b"]
