"""Knowledge-document ingestion (prose/reference docs, not API specs).

Proves the wiki can hold general knowledge (Oracle, FastAPI how-tos) alongside
API entries, that doc type auto-detects, and that a knowledge doc links to an
API concept it mentions — the basis for cross-domain agent reasoning
(e.g. "data loss" → an Oracle flashback doc → the flashback-api /recover endpoint).
"""

# pylint: disable=redefined-outer-name  # pytest fixtures injected by name

from typing import Any

import pytest

from services.llm.config import LLMConfig
from services.llm.providers import MinimaxProvider
from services.processor import WikiProcessor, _looks_like_api
from tests.test_two_step_ingest import _FakeStorage  # reuse the in-memory storage


@pytest.fixture
def llm():
    """A Minimax provider in mock mode (conftest forces MOCK_LLM)."""
    return MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))


_ORACLE_MD = {
    "oracle-flashback.md": (
        "---\nsource_app: oracle-kb\n---\n"
        "# Oracle Flashback\n"
        "Oracle Flashback recovers data after accidental data loss without a full restore.\n"
        "## Flashback Table\n"
        "Rewinds a table to a past point in time.\n"
        "- FLASHBACK TABLE t TO TIMESTAMP ...\n"
        "- Recovers dropped or wrongly-updated rows\n"
    )
}
_FASTAPI_MD = {
    "fastapi-howto.md": (
        "---\nsource_app: fastapi-kb\n---\n"
        "# How to build a FastAPI endpoint\n"
        "Define a path operation with a decorator and a typed function.\n"
        "## Steps\n"
        "- Create an APIRouter\n"
        "- Add async def handlers with Pydantic models\n"
    )
}


def test_looks_like_api_classifier():
    """Endpoint-shaped markdown is API; endpoint-free prose is not."""
    assert _looks_like_api({"a.md": "GET /items list"}) is True
    assert _looks_like_api({"a.md": "# Guide\njust prose, no endpoints"}) is False


async def test_generate_knowledge_mock_extracts_structure(llm):
    """Mock knowledge extraction yields title/summary/topics/key_points."""
    k = await llm.generate_knowledge(_ORACLE_MD, source_app="oracle-kb")
    ((doc_id, entry),) = k.items()
    assert "oracle" in doc_id
    assert entry["title"] == "Oracle Flashback"
    assert "data loss" in entry["summary"].lower()
    assert "Flashback Table" in entry["topics"]
    assert any("dropped" in p.lower() for p in entry["key_points"])


async def test_process_auto_detects_knowledge(llm):
    """Prose with no endpoints auto-detects as knowledge, not API."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    # No doc_type given; prose with no endpoints -> knowledge.
    r = await proc.process(_ORACLE_MD, "t", source_app="oracle-kb", source_version="v1")
    assert r.status == "success"

    await proc.rebuild_concepts()  # P3: materialize the aggregate from per-app objects
    wiki = await storage.aget_json("wiki.json")
    assert wiki["apis"] == {}, "prose doc must not create API entries"
    assert wiki["knowledge"], "knowledge section populated"
    entry = next(iter(wiki["knowledge"].values()))
    assert entry["source_app"] == "oracle-kb"
    assert entry["sources"] == ["oracle-flashback.md"]


async def test_api_and_knowledge_coexist(llm):
    """API and knowledge pushes from different apps coexist in one wiki."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    await proc.process(
        {
            "flashback.md": (
                "---\nsource_app: flashback-api\n---\n"
                "# Flashback API\nPOST /recover start recovery\n"
            )
        },
        "t",
        source_app="flashback-api",
        source_version="v1",
    )  # api (auto)
    await proc.process(
        _ORACLE_MD, "t", source_app="oracle-kb", source_version="v1"
    )  # knowledge (auto)

    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    assert "flashback-api" in wiki["apis"]  # api push preserved
    assert any("oracle" in d for d in wiki["knowledge"])  # knowledge push preserved


async def test_concept_links_knowledge_to_api(llm):
    """The Oracle flashback doc mentions 'recover' → links to the flashback-api
    /recover concept, giving the agent a cross-domain bridge."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    await proc.process(
        {
            "flashback.md": (
                "---\nsource_app: flashback-api\n---\n"
                "# Flashback API\nPOST /recover start recovery\n"
            )
        },
        "t",
        source_app="flashback-api",
        source_version="v1",
    )
    await proc.process(_ORACLE_MD, "t", source_app="oracle-kb", source_version="v1")

    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    recover = wiki["concepts"]["recover"]
    # endpoint + knowledge doc both attached to the same concept
    assert any(r.startswith("flashback-api::") for r in recover["related"])
    assert any(r.startswith("knowledge::") for r in recover["related"])
