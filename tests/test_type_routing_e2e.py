"""End-to-end: declared type routes a push to the right store (apis vs knowledge).

Complements the unit matrix in test_type_detection.py by driving the full
WikiProcessor.process() pipeline (mock LLM, in-memory storage) and asserting
where each push actually lands. The decisive cases are the ones a stray
endpoint line would have mis-routed before type became authoritative.
"""

# pylint: disable=redefined-outer-name  # pytest fixtures injected by name

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core import deps
from main import app
from services.llm.config import LLMConfig
from services.llm.providers import MinimaxProvider
from services.processor import WikiProcessor, _app_key
from tests.test_two_step_ingest import _FakeStorage  # reuse the in-memory storage

client = TestClient(app)


@pytest.fixture
def llm():
    """A Minimax provider in mock mode (conftest forces MOCK_LLM)."""
    return MinimaxProvider(LLMConfig(provider="minimax", api_key="k", model="m"))


async def _counts(llm, markdowns, *, app, doc_type=None, openapi=None):
    """Process one push and return (status, n_apis, n_knowledge) for the app."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    res = await proc.process(
        markdowns, "t", source_app=app, source_version="v1", doc_type=doc_type, openapi=openapi
    )
    obj = await storage.aget_json(_app_key(app)) or {}
    return res.status, len(obj.get("apis", {})), len(obj.get("knowledge", {}))


_API_README = {
    "README.md": (
        "---\ntype: api\nsource_app: pay\n---\n"
        "# Pay\n對信用卡扣款。\n## Endpoints\nPOST /charge — 對信用卡扣款\n"
    )
}
_HOWTO_README = {
    "README.md": (
        "---\ntype: how-to\nsource_app: kb\n---\n"
        "# 從誤刪救回資料\n誤刪後用 Oracle Flashback 回溯。\n"
    )
}
# A component doc that, against the skill's guidance, lets an endpoint signature
# slip into prose. The declared knowledge type must keep it out of `apis`.
_REFERENCE_WITH_STRAY_ENDPOINT = {
    "README.md": (
        "---\ntype: reference\nsource_app: nightly\ntags: [cronjob]\n---\n"
        "# Nightly Billing Job\n每晚 02:00 對到期帳單扣款；內部會打 POST /charge。\n"
    )
}


async def test_api_doc_lands_in_apis(llm):
    """type: api with body endpoints populates apis, not knowledge."""
    status, n_apis, n_knowledge = await _counts(llm, _API_README, app="pay")
    assert status == "success"
    assert n_apis >= 1
    assert n_knowledge == 0


async def test_knowledge_doc_lands_in_knowledge(llm):
    """type: how-to populates knowledge, not apis."""
    status, n_apis, n_knowledge = await _counts(llm, _HOWTO_README, app="kb")
    assert status == "success"
    assert n_apis == 0
    assert n_knowledge >= 1


async def test_reference_with_stray_endpoint_stays_knowledge(llm):
    """A declared reference doc with an endpoint signature in prose is knowledge —
    the type is authoritative; the endpoint heuristic does not pull it into apis."""
    status, n_apis, n_knowledge = await _counts(llm, _REFERENCE_WITH_STRAY_ENDPOINT, app="nightly")
    assert status == "success"
    assert n_apis == 0
    assert n_knowledge >= 1


async def test_doc_type_reference_passthrough_is_knowledge(llm):
    """Regression: an explicit doc_type that is neither "api" nor "knowledge"
    (here "reference") must route to knowledge. It previously passed straight
    through as the kind and, not being exactly "knowledge", went to the API
    path — so an endpoint-shaped prose line landed in apis."""
    md = {"README.md": "# Legacy notes\n散文，順帶提到 POST /charge 這個端點。\n"}
    status, n_apis, n_knowledge = await _counts(llm, md, app="legacy", doc_type="reference")
    assert status == "success"
    assert n_apis == 0
    assert n_knowledge >= 1


async def test_openapi_attached_is_deterministic_api(llm):
    """An attached OpenAPI spec ingests endpoints deterministically into apis."""
    spec = {
        "paths": {
            "/charge": {
                "post": {"summary": "對信用卡扣款", "parameters": []},
            }
        }
    }
    md = {"README.md": "---\ntype: api\nsource_app: pay2\n---\n# Pay2\n收款。\n"}
    status, n_apis, n_knowledge = await _counts(llm, md, app="pay2", openapi=spec)
    assert status == "success"
    assert n_apis >= 1
    assert n_knowledge == 0


# --- whole flow through the HTTP layer (router -> process -> storage) ---


def test_http_process_routes_knowledge_by_type(llm):
    """A type: how-to push over the real /process route lands in the app's
    knowledge store — exercising the router, request schema, processor and
    storage together (the only layers below this are real MinIO/PG)."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    app.dependency_overrides[deps.get_processor] = lambda: proc
    try:
        resp = client.post(
            "/process",
            json={
                "markdowns": {
                    "README.md": "---\ntype: how-to\nsource_app: kbh\n---\n# 救資料\n用 flashback 回溯。\n"
                },
                "timestamp": "t",
                "trigger_info": {},
                "source_app": "kbh",
                "source_version": "v1",
            },
        )
    finally:
        app.dependency_overrides.pop(deps.get_processor, None)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    obj = asyncio.run(storage.aget_json(_app_key("kbh"))) or {}
    assert len(obj.get("knowledge", {})) >= 1
    assert len(obj.get("apis", {})) == 0


async def test_mixed_batch_scale_routes_and_isolates(llm):
    """Many apps of mixed types pushed through one processor each land in the
    right store and stay isolated — routing holds at scale, not just for one."""
    storage: Any = _FakeStorage()
    proc = WikiProcessor(storage=storage, llm=llm)
    n = 30
    for i in range(n):
        if i % 2 == 0:
            md = {
                "README.md": (
                    f"---\ntype: api\nsource_app: api{i}\n---\n"
                    f"# A{i}\nx\n## Endpoints\nGET /r{i} — 取資源{i}\n"
                )
            }
            await proc.process(md, "t", source_app=f"api{i}", source_version="v1")
        else:
            md = {
                "README.md": (
                    f"---\ntype: reference\nsource_app: kb{i}\n---\n"
                    f"# K{i}\n元件{i}：每晚跑，內部打 POST /go{i}。\n"
                )
            }
            await proc.process(md, "t", source_app=f"kb{i}", source_version="v1")

    for i in range(n):
        if i % 2 == 0:
            obj = await storage.aget_json(_app_key(f"api{i}")) or {}
            assert len(obj.get("apis", {})) >= 1, f"api{i} should have endpoints"
            assert len(obj.get("knowledge", {})) == 0
        else:
            obj = await storage.aget_json(_app_key(f"kb{i}")) or {}
            assert len(obj.get("knowledge", {})) >= 1, f"kb{i} should be knowledge"
            # the stray endpoint in prose must not leak into apis
            assert len(obj.get("apis", {})) == 0, f"kb{i} stray endpoint leaked into apis"
