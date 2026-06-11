"""Write-path wiring of the optional PG vector index (hermetic).

The contract under test: index sync is best-effort and strictly after the
CAS write — embedding or PG failures degrade (NULL vectors / audit flag) but
never fail the wiki write; processors without a vector store behave exactly
as before the layer existed.
"""
import pytest

from services.embeddings import EmbeddingClient, EmbeddingConfig
from services.processor import WikiProcessor
from tests.test_concurrency import InMemoryCASStorage, app_markdown, make_processor

_DIM = 16


class FakeVectorStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.schema_calls = 0
        self.replace_calls = []
        self.rebuilds = []
        self.dim = _DIM

    async def ensure_schema_once(self):
        self.schema_calls += 1

    async def replace_app_entries(self, source_app, source_version, rows, synced_at):
        if self.fail:
            raise ConnectionError("pg is down")
        self.replace_calls.append((source_app, source_version, rows, synced_at))
        return True

    async def rebuild(self, apps, versions):
        self.rebuilds.append((apps, versions))
        return sum(len(rows) for rows in apps.values())


class FailingEmbedder(EmbeddingClient):
    def __init__(self):
        super().__init__(EmbeddingConfig(mock_mode=True, dim=_DIM))

    async def aembed(self, texts):
        raise ConnectionError("embeddings endpoint unreachable")


def make_vector_processor(storage, vector_store, embedder="mock"):
    base = make_processor(storage)
    if embedder == "mock":
        embedder = EmbeddingClient(EmbeddingConfig(mock_mode=True, dim=_DIM))
    return WikiProcessor(
        storage=storage, llm=base.llm, embedder=embedder, vector_store=vector_store
    )


async def test_sync_called_with_embedded_rows():
    storage = InMemoryCASStorage()
    store = FakeVectorStore()
    processor = make_vector_processor(storage, store)

    result = await processor.process(
        app_markdown("app-a"), "t", source_app="app-a", source_version="v1"
    )
    assert result.status == "success"

    assert len(store.replace_calls) == 1
    source_app, version, rows, synced_at = store.replace_calls[0]
    assert (source_app, version) == ("app-a", "v1")
    assert synced_at.tzinfo is not None

    (row,) = rows
    assert row["module"] == "app-a"
    assert row["api_key"] == "GET /app-a/items"
    assert row["detail"]["source_app"] == "app-a"  # stamped before indexing
    assert len(row["embedding"]) == _DIM
    assert row["embedding_model"] == "text-embedding-3-small"
    assert "app-a" in row["embed_text"]


async def test_embedder_failure_degrades_to_null_vectors():
    storage = InMemoryCASStorage()
    store = FakeVectorStore()
    processor = make_vector_processor(storage, store, embedder=FailingEmbedder())

    result = await processor.process(
        app_markdown("app-a"), "t", source_app="app-a", source_version="v1"
    )
    assert result.status == "success"  # wiki write unaffected

    (_, _, rows, _) = store.replace_calls[0]
    assert all(r["embedding"] is None for r in rows)  # synced relationally


async def test_store_failure_keeps_wiki_success_and_flags_audit():
    storage = InMemoryCASStorage()
    processor = make_vector_processor(storage, FakeVectorStore(fail=True))

    result = await processor.process(
        app_markdown("app-a"), "t", source_app="app-a", source_version="v1"
    )
    assert result.status == "success"

    statuses = [e["status"] for e in storage.audit_entries()]
    assert "success" in statuses
    assert "success_index_sync_failed" in statuses


async def test_no_vector_store_means_no_index_activity():
    storage = InMemoryCASStorage()
    processor = make_processor(storage)  # embedder/vector_store default to None

    result = await processor.process(
        app_markdown("app-a"), "t", source_app="app-a", source_version="v1"
    )
    assert result.status == "success"
    assert processor.vector_store is None


async def test_unchanged_resubmission_skips_sync():
    storage = InMemoryCASStorage()
    store = FakeVectorStore()
    processor = make_vector_processor(storage, store)

    md = app_markdown("app-a")
    await processor.process(md, "t", source_app="app-a", source_version="v1")
    await processor.process(md, "t", source_app="app-a", source_version="v1")
    assert len(store.replace_calls) == 1  # second run took the no-change path


async def test_reindex_rebuilds_from_wiki():
    storage = InMemoryCASStorage()
    store = FakeVectorStore()
    processor = make_vector_processor(storage, store)

    await processor.process(app_markdown("app-a"), "t", source_app="app-a", source_version="v1")
    await processor.process(app_markdown("app-b"), "t", source_app="app-b", source_version="v2")

    result = await processor.reindex()
    assert result == {"apps": 2, "entries": 2, "embedded": 2}

    (apps, versions), = store.rebuilds
    assert set(apps) == {"app-a", "app-b"}
    assert versions == {"app-a": "v1", "app-b": "v2"}
    assert apps["app-a"][0]["embedding"] is not None


async def test_reindex_without_store_raises():
    processor = make_processor(InMemoryCASStorage())
    with pytest.raises(RuntimeError, match="PG_DSN"):
        await processor.reindex()
