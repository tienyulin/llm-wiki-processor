"""Concurrency and schema-migration regression tests for WikiProcessor.

All apps share a single wiki.json. Phase 1 (LLM) runs concurrently; Phase 2
(merge + write) uses ETag CAS plus a process-local lock. These tests use an
in-memory CAS storage whose every operation yields control, forcing coroutine
interleaving so lost updates would surface immediately.
"""
import asyncio
import uuid

import pytest

from services.llm.config import LLMConfig
from services.llm.providers.minimax import MinimaxProvider
from services.processor import WikiProcessor, _AUDIT_PREFIX, _WIKI_KEY


class InMemoryCASStorage:
    """Async, etag-aware stand-in for MinioStorage's facade."""

    def __init__(self):
        self.data: dict = {}
        self.etags: dict = {}

    def _put(self, key, value):
        self.data[key] = value
        self.etags[key] = uuid.uuid4().hex

    async def aget_json(self, key):
        await asyncio.sleep(0)
        return self.data.get(key)

    async def aput_json(self, key, value):
        await asyncio.sleep(0)
        self._put(key, value)

    async def aget_json_with_etag(self, key):
        await asyncio.sleep(0)
        if key not in self.data:
            return None, None
        return self.data[key], self.etags[key]

    async def aput_json_if_match(self, key, value, etag):
        await asyncio.sleep(0)
        if self.etags.get(key) != etag:
            return False
        self._put(key, value)
        return True

    async def aput_json_if_absent(self, key, value):
        await asyncio.sleep(0)
        if key in self.data:
            return False
        self._put(key, value)
        return True

    def audit_entries(self):
        return [v for k, v in self.data.items() if k.startswith(_AUDIT_PREFIX)]


def make_processor(storage):
    # conftest sets MOCK_LLM=true: the base-class mock derives API entries
    # from the input markdowns, so per-app data is verifiable end to end.
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="test-key", model="m"))
    return WikiProcessor(storage=storage, llm=llm)


def app_markdown(app: str) -> dict[str, str]:
    return {f"{app}_api.md": f"# {app} API\n\nGET /{app}/items\n"}


async def test_concurrent_app_updates_have_no_lost_writes():
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    apps = [f"app-{i:02d}" for i in range(20)]
    results = await asyncio.gather(*[
        processor.process(
            markdowns=app_markdown(app),
            timestamp="2026-06-11T00:00:00",
            source_app=app,
            source_version="v1.0.0",
        )
        for app in apps
    ])

    assert all(r.status == "success" for r in results), [r.message for r in results]

    wiki = storage.data[_WIKI_KEY]
    assert wiki["schema_version"] == 2
    for app in apps:
        entries = wiki["apis"].get(app, {})
        assert f"GET /{app}/items" in entries, f"lost update for {app}"
        assert entries[f"GET /{app}/items"]["source_app"] == app

    assert len(storage.audit_entries()) == len(apps)


async def test_resubmission_replaces_only_that_apps_entries():
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    await processor.process(app_markdown("app-a"), "t", source_app="app-a", source_version="v1")
    await processor.process(app_markdown("app-b"), "t", source_app="app-b", source_version="v1")

    # app-a v2 exposes a different endpoint — old one must disappear
    new_md = {"app-a_api.md": "# app-a API v2\n\nPOST /app-a/orders\n"}
    result = await processor.process(new_md, "t", source_app="app-a", source_version="v2")
    assert result.status == "success"

    apis = storage.data[_WIKI_KEY]["apis"]
    assert "POST /app-a/orders" in apis["app-a"]
    assert "GET /app-a/items" not in apis.get("app-a", {})
    # app-b untouched
    assert "GET /app-b/items" in apis["app-b"]
    assert apis["app-a"]["POST /app-a/orders"]["source_version"] == "v2"


async def test_unchanged_resubmission_skips_llm():
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    md = app_markdown("app-a")
    await processor.process(md, "t", source_app="app-a", source_version="v1")
    etag_before = storage.etags[_WIKI_KEY]

    result = await processor.process(md, "t", source_app="app-a", source_version="v1")
    assert result.status == "success"
    assert "No changes" in result.message
    assert storage.etags[_WIKI_KEY] == etag_before  # wiki not rewritten


async def test_cas_conflict_is_retried():
    """A concurrent writer between Phase 1 read and Phase 2 write must not
    cause data loss — the CAS loop re-reads and re-merges."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)
    await processor.process(app_markdown("app-x"), "t", source_app="app-x", source_version="v1")

    original_if_match = storage.aput_json_if_match
    conflict_injected = False

    async def conflicting_if_match(key, value, etag):
        nonlocal conflict_injected
        if not conflict_injected:
            conflict_injected = True
            # Simulate another replica sneaking in a write first
            storage._put(_WIKI_KEY, {
                **storage.data[_WIKI_KEY],
                "apis": {**storage.data[_WIKI_KEY]["apis"],
                         "intruder": {"GET /intruder": {"source_app": "intruder",
                                                        "source_version": "v9"}}},
            })
        return await original_if_match(key, value, etag)

    storage.aput_json_if_match = conflicting_if_match

    result = await processor.process(
        app_markdown("app-y"), "t", source_app="app-y", source_version="v1"
    )
    assert result.status == "success"
    assert conflict_injected

    apis = storage.data[_WIKI_KEY]["apis"]
    # Both the intruder's write and app-y's update survived
    assert "GET /intruder" in apis["intruder"]
    assert "GET /app-y/items" in apis["app-y"]
    assert "GET /app-x/items" in apis["app-x"]


async def test_legacy_wiki_is_migrated_to_v2():
    """Pre-v2 wikis mixed structured entries with file-map strings; the
    structured part survives, file-map entries are dropped."""
    storage = InMemoryCASStorage()
    processor = make_processor(storage)

    storage._put(_WIKI_KEY, {
        "apis": {"inventory": {"GET /inventory": {"description": "list"}}},
        "metadata": {"version": "1.0"},
        "legacy_doc.md": "---\nsource_app: old-app\n---\n# Legacy",
    })

    result = await processor.process(
        app_markdown("app-new"), "t", source_app="app-new", source_version="v1"
    )
    assert result.status == "success"

    wiki = storage.data[_WIKI_KEY]
    assert wiki["schema_version"] == 2
    assert "legacy_doc.md" not in wiki
    assert "GET /inventory" in wiki["apis"]["inventory"]  # structured part kept
    assert "GET /app-new/items" in wiki["apis"]["app-new"]
