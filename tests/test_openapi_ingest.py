"""Deterministic OpenAPI ingestion + frontmatter type/tags (authoring standard).

OpenAPI path produces wiki entries directly from the spec — no LLM call (accurate,
no rate-limit). Frontmatter type/tags ride in the entry/knowledge dicts.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures injected by name

from typing import Any

import pytest

from services.llm.config import LLMConfig
from services.llm.providers import MinimaxProvider
from services.processor import (
    WikiProcessor,
    _apis_from_openapi,
    _parse_frontmatter,
    _readme_summary,
)
from tests.test_two_step_ingest import _FakeStorage


@pytest.fixture
def llm():
    """A Minimax provider in mock mode (conftest forces MOCK_LLM)."""
    return MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))


_SPEC = {
    "openapi": "3.1.0",
    "paths": {
        "/payments/charge": {
            "post": {
                "summary": "Charge a saved credit card",
                "parameters": [{"name": "idempotency_key"}],
            }
        },
        "/payments/refund": {"post": {"description": "Refund money back to a customer"}},
        "/health": {"get": {"summary": "health"}},
    },
}


def test_apis_from_openapi_pure():
    """The OpenAPI spec maps directly to wiki entries, no LLM involved."""
    apis = _apis_from_openapi(_SPEC, "payments-api")
    m = apis["payments-api"]
    assert m["POST /payments/charge"]["description"] == "Charge a saved credit card"
    assert m["POST /payments/charge"]["parameters"] == ["idempotency_key"]
    assert m["POST /payments/charge"]["sources"] == ["openapi.json"]
    assert m["POST /payments/refund"]["description"] == "Refund money back to a customer"
    assert set(m) == {"POST /payments/charge", "POST /payments/refund", "GET /health"}


def test_parse_frontmatter():
    """YAML frontmatter parses into a dict; absent frontmatter yields an empty dict."""
    fm = _parse_frontmatter(
        {"r.md": "---\ntype: how-to\nsource_app: kb\ntags: [a, b-c]\n---\n# X\nbody"}
    )
    assert fm["type"] == "how-to" and fm["source_app"] == "kb" and fm["tags"] == ["a", "b-c"]
    assert not _parse_frontmatter({"r.md": "# no frontmatter"})


def test_readme_summary_first_paragraph():
    """The README summary is its first prose paragraph after the H1."""
    md = {"r.md": "---\ntype: api\n---\n# Payments API\nProcess charges and refunds.\nPOST /x do"}
    assert _readme_summary(md) == "Process charges and refunds."


async def test_process_openapi_skips_llm_and_stores(llm):
    """process() with an openapi spec stores spec-derived entries, no LLM call."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    md = {
        "README.md": "---\ntype: api\nsource_app: payments-api\ntags: [billing]\n---\n"
        "# Payments API\nHandles charges and refunds for the platform."
    }
    r = await proc.process(md, "t", source_app="payments-api", source_version="v1", openapi=_SPEC)
    assert r.status == "success"
    assert "POST /payments/charge" in r.files_updated

    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    entry = wiki["apis"]["payments-api"]["POST /payments/charge"]
    assert entry["description"] == "Charge a saved credit card"  # from spec, not LLM
    assert entry["sources"] == ["openapi.json"]
    assert entry["tags"] == ["billing"]  # frontmatter tag stored
    # deterministic overview from README first paragraph
    assert "charges and refunds" in wiki["overviews"]["payments-api"]["text"].lower()


async def test_process_knowledge_stores_type_tags(llm):
    """A knowledge doc stores its frontmatter doc_type and tags."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    md = {
        "undo.md": "---\ntype: how-to\nsource_app: oracle-kb\ntags: [recovery]\n---\n"
        "# Undo a delete\nUse flashback to recover."
    }
    await proc.process(md, "t", source_app="oracle-kb", source_version="v1")
    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    k = next(iter(wiki["knowledge"].values()))
    assert k["doc_type"] == "how-to" and k["tags"] == ["recovery"]


async def test_openapi_only_change_reingests(llm):
    """An openapi-only change (README unchanged) must re-ingest, not no-op —
    this is the 'fill endpoint descriptions in code' workflow."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    md = {"README.md": "---\ntype: api\nsource_app: pay\n---\n# Pay\nHandles charges."}
    spec1 = {"openapi": "3.1.0", "paths": {"/charge": {"post": {"summary": "Charge a card"}}}}
    r1 = await proc.process(md, "t1", source_app="pay", source_version="v1", openapi=spec1)
    assert "POST /charge" in r1.files_updated

    # Same README, richer description in the spec (regenerated from code).
    spec2 = {
        "openapi": "3.1.0",
        "paths": {"/charge": {"post": {"summary": "對信用卡扣款收取款項"}}},
    }
    r2 = await proc.process(md, "t2", source_app="pay", source_version="v2", openapi=spec2)
    assert "POST /charge" in r2.files_updated, "openapi-only change was dropped as no-op"

    # Identical README + identical spec → genuine no-op.
    r3 = await proc.process(md, "t3", source_app="pay", source_version="v3", openapi=spec2)
    assert r3.files_updated == []
