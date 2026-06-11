"""Regression tests for concurrent process() calls.

All apps share a single wiki.json, and the read-modify-write pipeline in
WikiProcessor.process() spans an awaited LLM call. Without the processor
lock, concurrent app updates read the same wiki state and overwrite each
other (lost updates). The fake LLM below yields control at the await point
to force that interleaving.
"""
import asyncio

from services.processor import WikiProcessor

_SNAPSHOT_KEY = "markdowns_snapshot.json"
_WIKI_KEY = "wiki.json"
_AUDIT_LOG_KEY = "wiki-audit-log.jsonl"


class InMemoryStorage:
    """Sync dict-backed stand-in for MinioStorage."""

    def __init__(self):
        self.data = {}

    def get_json(self, key):
        return self.data.get(key)

    def put_json(self, key, value):
        self.data[key] = value

    def get_file(self, key):
        return self.data.get(key)

    def put_file(self, key, content):
        self.data[key] = content


class YieldingFakeLLM:
    """Yields control mid-call so coroutine interleaving can occur."""

    async def generate_wiki(self, markdowns):
        await asyncio.sleep(0)
        return dict(markdowns)

    async def update_wiki(self, current_files, changed_markdowns, changes):
        await asyncio.sleep(0)
        return dict(changed_markdowns)


def app_markdown(app: str) -> dict[str, str]:
    return {
        f"{app}_api.md": f'---\nsource_app: "{app}"\n---\n\n# {app} API\n',
    }


async def test_concurrent_app_updates_have_no_lost_writes():
    storage = InMemoryStorage()
    processor = WikiProcessor(storage=storage, llm=YieldingFakeLLM())

    # Non-empty snapshot so process() takes the app-level update path
    storage.put_json(_SNAPSHOT_KEY, {"seed.md": "# seed"})
    storage.put_json(_WIKI_KEY, {"apis": {}, "metadata": {"version": "1.0"}})

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

    assert all(r.status == "success" for r in results)

    wiki = storage.get_json(_WIKI_KEY)
    for app in apps:
        assert f"{app}_api.md" in wiki, f"lost update for {app}"

    audit_lines = storage.get_file(_AUDIT_LOG_KEY).strip().splitlines()
    assert len(audit_lines) == len(apps)


async def test_app_update_on_structured_wiki_succeeds():
    """Regression: app-level merge used to crash on dict-valued wiki entries."""
    storage = InMemoryStorage()
    processor = WikiProcessor(storage=storage, llm=YieldingFakeLLM())

    storage.put_json(_SNAPSHOT_KEY, {"seed.md": "# seed"})
    storage.put_json(_WIKI_KEY, {
        "apis": {"inventory": {"GET /inventory": {"description": "list items"}}},
        "metadata": {"version": "1.0"},
    })

    result = await processor.process(
        markdowns=app_markdown("app-payment"),
        timestamp="2026-06-11T00:00:00",
        source_app="app-payment",
        source_version="v1.0.0",
    )

    assert result.status == "success"
    wiki = storage.get_json(_WIKI_KEY)
    # Structured entries are preserved and the app's file was merged in
    assert wiki["apis"] == {"inventory": {"GET /inventory": {"description": "list items"}}}
    assert "app-payment_api.md" in wiki
