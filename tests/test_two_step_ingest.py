"""Two-step chain-of-thought ingest + source-provenance traceability.

Covers:
  1. Mock path stamps a `sources` list on every entry.
  2. Real LLM path runs as two calls (analyze -> generate), not one pass.
  3. End-to-end: a real, multi-app markdown set flows through the full
     WikiProcessor.process() pipeline and lands in wiki.json with cross-app
     isolation, provenance, and source traceability intact.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures injected by name
# pylint: disable=super-init-not-called  # stub/recording providers skip the real init
# pylint: disable=unused-argument  # stub LLM signatures must match the real provider

import json
from typing import Any

import pytest

from services.llm.base import LLMProvider
from services.llm.config import LLMConfig
from services.llm.providers import MinimaxProvider
from services.processor import WikiProcessor, _app_key

# ---------------------------------------------------------------------------
# 1. Mock path: every entry carries `sources`
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """A Minimax provider in mock mode (conftest forces MOCK_LLM)."""
    return MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))


async def test_mock_generate_wiki_stamps_sources(mock_llm):
    """The mock generate path stamps a `sources` list on every entry."""
    wiki = await mock_llm.generate_wiki({"orders.md": "# Orders\nGET /orders\nPOST /orders"})
    entries = [e for mod in wiki["apis"].values() for e in mod.values()]
    assert entries, "expected at least one entry"
    for e in entries:
        assert e["sources"] == ["orders.md"]


async def test_mock_update_wiki_stamps_sources(mock_llm):
    """The mock update path also stamps `sources` on every entry."""
    wiki = await mock_llm.update_wiki(
        {}, {"users.md": "# Users\nDELETE /users/{id}"}, {"added": ["users.md"]}
    )
    entries = [e for mod in wiki["apis"].values() for e in mod.values()]
    assert entries
    assert all(e["sources"] == ["users.md"] for e in entries)


# ---------------------------------------------------------------------------
# 2. Real path: analyze -> generate (two calls)
# ---------------------------------------------------------------------------


class _RecordingProvider(LLMProvider):
    """Real (non-mock) provider whose generate() records prompts and returns
    canned analysis then canned JSON, so we can assert the two-step flow."""

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt, temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        if "generate the structured wiki JSON" in prompt:  # step 2
            return json.dumps(
                {
                    "apis": {
                        "billing": {
                            "GET /invoices": {
                                "method": "GET",
                                "path": "/invoices",
                                "description": "list invoices",
                                "sources": ["billing.md"],
                            }
                        }
                    },
                    "metadata": {},
                }
            )
        return "ANALYSIS: GET /invoices in billing, from billing.md"  # step 1

    async def validate_config(self):
        return True

    def get_model_info(self):
        return {"provider": "recording", "model_name": "x"}


async def test_real_path_is_two_step(monkeypatch):
    """The real (non-mock) path runs as analyze then generate (two calls)."""
    monkeypatch.setenv("MOCK_LLM", "false")
    p = _RecordingProvider()
    wiki = await p.generate_wiki({"billing.md": "# Billing\nGET /invoices"})

    assert len(p.prompts) == 2, "expected analyze + generate (two calls)"
    # Step 1 reasons (no JSON shape demanded); step 2 demands the JSON shape.
    assert "generate the structured wiki JSON" not in p.prompts[0]
    assert "generate the structured wiki JSON" in p.prompts[1]
    # Step 2 was grounded in step-1 analysis.
    assert "ANALYSIS:" in p.prompts[1]
    assert wiki["apis"]["billing"]["GET /invoices"]["sources"] == ["billing.md"]


# ---------------------------------------------------------------------------
# 3. End-to-end real case through the full pipeline
# ---------------------------------------------------------------------------


class _FakeStorage:
    """In-memory stand-in for MinioStorage covering the async methods the
    processor uses. ETag is a monotonically bumped string."""

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._etags: dict[str, str] = {}
        self._n = 0

    def _bump(self, key):
        self._n += 1
        self._etags[key] = f"etag-{self._n}"
        return self._etags[key]

    async def aget_json(self, key):
        """Return the stored value for key, or None."""
        return self._store.get(key)

    async def aget_json_with_etag(self, key):
        """Return the stored (value, etag) pair for key."""
        return self._store.get(key), self._etags.get(key)

    async def aput_json(self, key, value):
        """Unconditionally store value under key and bump its etag."""
        self._store[key] = value
        self._bump(key)
        return True

    async def aput_json_if_absent(self, key, value):
        """Store value only if key is absent (create-if-missing)."""
        if key in self._store:
            return False
        self._store[key] = value
        self._bump(key)
        return True

    async def aput_json_if_match(self, key, value, etag):
        """Store value only if the current etag matches (compare-and-set)."""
        if self._etags.get(key) != etag:
            return False
        self._store[key] = value
        self._bump(key)
        return True

    async def alist_files(self, prefix=""):
        """List stored keys under the given prefix."""
        return [k for k in self._store if k.startswith(prefix)]


# Realistic source docs from two different applications.
_FLASHBACK_MD = {
    "flashback.md": (
        "---\nsource_app: flashback-api\n---\n"
        "# Flashback Recovery API\n"
        "POST /recover  — start a flashback recovery job\n"
        "GET /recover/{id}  — poll recovery job status\n"
    )
}
_INVENTORY_MD = {
    "inventory.md": (
        "---\nsource_app: inventory-api\n---\n"
        "# Inventory API\n"
        "GET /items  — list stock items\n"
        "POST /items  — create a stock item\n"
    )
}


async def test_end_to_end_two_apps_real_case():
    """Two apps flow through the full pipeline with isolation and provenance intact."""
    storage: Any = _FakeStorage()
    llm = MinimaxProvider(
        LLMConfig(provider="minimax", api_key="k", model="m")
    )  # mock mode (conftest)
    proc = WikiProcessor(storage=storage, llm=llm)

    r1 = await proc.process(
        _FLASHBACK_MD, "2026-06-18T00:00:00", source_app="flashback-api", source_version="v1"
    )
    r2 = await proc.process(
        _INVENTORY_MD, "2026-06-18T00:01:00", source_app="inventory-api", source_version="v2"
    )

    assert r1.status == "success" and r2.status == "success"

    # P3: per-app objects are the source of truth; the aggregate wiki.json is
    # built on demand. rebuild_concepts() materializes it.
    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    apis = wiki["apis"]

    # Both apps coexist (cross-app isolation: neither overwrote the other).
    assert "flashback-api" in apis and "inventory-api" in apis

    # Every entry has provenance + source traceability.
    for module, endpoints in apis.items():
        for _api_key, detail in endpoints.items():
            assert detail["source_app"] == module
            assert detail["source_version"] in ("v1", "v2")
            assert isinstance(detail["sources"], list) and detail["sources"]

    # The flashback recover endpoint traces back to its source file.
    recover = apis["flashback-api"]["POST /recover"]
    assert recover["sources"] == ["flashback.md"]
    assert recover["source_version"] == "v1"

    # Per-app overview (item 5) was synthesized and stored for each app.
    assert "flashback-api" in wiki["overviews"]
    assert "recover" in wiki["overviews"]["flashback-api"]["text"].lower()


# ---------------------------------------------------------------------------
# 4. Overview + concepts + recompile through the processor
# ---------------------------------------------------------------------------


async def test_overview_mock_lists_endpoints():
    """The mock overview names the app and its endpoints."""
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))
    text = await llm.generate_overview(
        "billing", {"billing": {"GET /invoices": {"description": "list invoices"}}}
    )
    assert "billing" in text and "invoices" in text


async def test_concepts_mock_clusters_cross_app():
    """Two apps both exposing /recover collapse into one cross-app concept."""
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))
    apis = {
        "app-a": {"POST /recover": {"description": "recover", "source_app": "app-a"}},
        "app-b": {"GET /recover/{id}": {"description": "status", "source_app": "app-b"}},
    }
    concepts = await llm.generate_concepts(apis)
    assert "recover" in concepts
    assert sorted(concepts["recover"]["apps"]) == ["app-a", "app-b"]
    assert len(concepts["recover"]["related"]) == 2


async def test_rebuild_concepts_writes_to_wiki():
    """rebuild_concepts() persists concepts without clobbering apis/overviews."""
    storage: Any = _FakeStorage()
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))
    proc = WikiProcessor(storage=storage, llm=llm)

    await proc.process(_FLASHBACK_MD, "t", source_app="flashback-api", source_version="v1")
    await proc.process(_INVENTORY_MD, "t", source_app="inventory-api", source_version="v2")

    result = await proc.rebuild_concepts()
    assert result["concepts"] > 0
    wiki = await storage.aget_json("wiki.json")
    assert wiki["concepts"], "concepts should be persisted on wiki.json"
    # rebuild must not clobber existing apis/overviews.
    assert "flashback-api" in wiki["apis"]
    assert "flashback-api" in wiki["overviews"]


async def test_recompile_refreshes_from_snapshots():
    """recompile() rebuilds entries from per-app snapshots, re-versioned."""
    storage: Any = _FakeStorage()
    llm = MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))
    proc = WikiProcessor(storage=storage, llm=llm)

    await proc.process(_FLASHBACK_MD, "t", source_app="flashback-api", source_version="v1")
    result = await proc.recompile()

    assert "flashback-api" in result["recompiled_apps"]
    await proc.rebuild_concepts()
    wiki = await storage.aget_json("wiki.json")
    # Entries are back after recompile, stamped with the recompiled version.
    assert wiki["apis"]["flashback-api"]["POST /recover"]["source_version"] == "recompiled"


async def test_modeb_module_collapses_to_source_app():
    """A real LLM may name the module differently from source_app; the processor
    must re-key endpoints under source_app (the per-app model, like the OpenAPI
    path and the mock LLM)."""

    class _StubLLM:
        """Minimal LLM returning a module name that differs from source_app."""

        async def generate_wiki(self, markdowns, source_app=None):
            """Return one endpoint keyed under a module name != source_app."""
            return {
                "apis": {
                    "payments": {
                        "POST /pay": {"method": "POST", "path": "/pay", "description": "扣款"}
                    }
                }
            }

        async def generate_overview(self, app, apis):
            """Return a trivial per-app overview string."""
            return f"{app} overview"

    storage: Any = _FakeStorage()
    stub_llm: Any = _StubLLM()
    proc = WikiProcessor(storage=storage, llm=stub_llm)
    md = {
        "README.md": "---\ntype: api\nsource_app: payments-svc\n---\n# 金流\n扣款。\nPOST /pay 扣款"
    }
    r = await proc.process(md, "t", source_app="payments-svc", source_version="v1")
    assert r.status == "success"
    obj, _ = await proc.storage.aget_json_with_etag(_app_key("payments-svc"))
    assert obj is not None
    assert set(obj["apis"].keys()) == {"payments-svc"}, obj["apis"].keys()
    assert "POST /pay" in obj["apis"]["payments-svc"]
