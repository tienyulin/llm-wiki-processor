"""Push-kind resolution — type is authoritative, endpoint heuristic is fallback.

The wiki-doc-author skill stamps a controlled frontmatter ``type``
(``api | tutorial | how-to | reference | explanation``); cronjob/worker/CLI
docs use ``type: reference`` + tags. The processor must trust that declared
type and only fall back to the "looks like endpoints" heuristic when nothing
declares a type — so a knowledge doc that merely *mentions* an endpoint, or a
component README that is not an API, is never mis-ingested as API.

These lock that contract: _resolve_kind matrix + frontmatter parsing.
"""

# pylint: disable=protected-access  # deliberately exercising the internal kind classifier

import pytest

from services.processor import WikiProcessor, _looks_like_api, _parse_frontmatter

_EP = {"a.md": "POST /charge — 對信用卡扣款"}  # endpoint-shaped (outside code fence)
_PROSE = {"a.md": "# Nightly Job\n每晚 02:00 對到期帳單扣款。"}  # no endpoint


def _kind(doc_type=None, openapi=None, fm_type=None, markdowns=None):
    """Thin wrapper over the static classifier with prose as the default body."""
    return WikiProcessor._resolve_kind(doc_type, openapi, fm_type, markdowns or _PROSE)


# --- explicit request doc_type (normalised to binary, never passed through) ---


def test_doc_type_api():
    """doc_type=api routes to the API path."""
    assert _kind(doc_type="api", markdowns=_PROSE) == "api"


def test_doc_type_knowledge():
    """doc_type=knowledge routes to knowledge even with endpoint-shaped body."""
    assert _kind(doc_type="knowledge", markdowns=_EP) == "knowledge"


@pytest.mark.parametrize("dt", ["reference", "tutorial", "how-to", "explanation"])
def test_doc_type_non_api_is_knowledge_not_passthrough(dt):
    """Regression: a non-api/knowledge doc_type used to pass straight through and,
    not being exactly "knowledge", route to the API path. Now normalised."""
    assert _kind(doc_type=dt, markdowns=_EP) == "knowledge"


@pytest.mark.parametrize("dt", ["API", "Api", " api "])
def test_doc_type_api_case_insensitive(dt):
    """doc_type matching is case- and whitespace-insensitive."""
    assert _kind(doc_type=dt, markdowns=_PROSE) == "api"


# --- attached OpenAPI spec -> deterministic API ingest ---


def test_openapi_present_is_api():
    """An attached OpenAPI spec means API ingest regardless of prose body."""
    assert _kind(openapi={"paths": {}}, markdowns=_PROSE) == "api"


# --- frontmatter type is authoritative when present ---


def test_fm_api():
    """frontmatter type=api -> api."""
    assert _kind(fm_type="api", markdowns=_PROSE) == "api"


@pytest.mark.parametrize("ft", ["tutorial", "how-to", "reference", "explanation"])
def test_fm_knowledge_types(ft):
    """A declared knowledge type wins even if the body has an endpoint line."""
    assert _kind(fm_type=ft, markdowns=_EP) == "knowledge"


@pytest.mark.parametrize("ft", ["API", "Api", " api "])
def test_fm_api_case_insensitive(ft):
    """frontmatter type matching is case- and whitespace-insensitive."""
    assert _kind(fm_type=ft, markdowns=_PROSE) == "api"


def test_fm_declared_non_api_with_endpoint_is_knowledge():
    """A component doc (e.g. someone writes type: cronjob) with a stray endpoint
    line must not be reclassified as API — the declared, non-api type is
    authoritative; the heuristic never overrides it."""
    assert _kind(fm_type="cronjob", markdowns=_EP) == "knowledge"


# --- heuristic only when nothing declares a type (legacy / no frontmatter) ---


def test_no_declaration_endpoint_is_api():
    """No declared type + endpoint-shaped body -> api (heuristic)."""
    assert _kind(markdowns=_EP) == "api"


def test_no_declaration_prose_is_knowledge():
    """No declared type + prose body -> knowledge (heuristic)."""
    assert _kind(markdowns=_PROSE) == "knowledge"


# --- precedence: doc_type > openapi > fm_type > heuristic ---


def test_precedence_doc_type_over_openapi_and_fm():
    """Explicit doc_type beats both an attached spec and frontmatter type."""
    assert (
        _kind(doc_type="knowledge", openapi={"paths": {}}, fm_type="api", markdowns=_EP)
        == "knowledge"
    )


def test_precedence_openapi_over_fm():
    """An attached spec beats frontmatter type (a spec means API ingest)."""
    assert _kind(openapi={"paths": {}}, fm_type="how-to", markdowns=_PROSE) == "api"


# --- the endpoint classifier itself ---


def test_looks_like_api_ignores_fenced_code():
    """Endpoint signatures inside fenced code don't count; outside prose does."""
    fenced = {"a.md": "# Guide\n```\nPOST /charge\n```\nprose only"}
    assert _looks_like_api(fenced) is False
    assert _looks_like_api({"a.md": "POST /charge — do it"}) is True


# --- frontmatter parsing (controlled subset) ---


def test_parse_frontmatter_type_and_tags():
    """Scalar fields and an inline list are parsed from the leading block."""
    md = {
        "r.md": (
            "---\ntype: reference\nsource_app: billing-nightly\n"
            "tags: [cronjob, billing]\n---\n# x\n"
        )
    }
    fm = _parse_frontmatter(md)
    assert fm["type"] == "reference"
    assert fm["tags"] == ["cronjob", "billing"]
    assert fm["source_app"] == "billing-nightly"


def test_parse_frontmatter_absent():
    """A doc with no leading --- block yields an empty mapping."""
    assert not _parse_frontmatter({"r.md": "# no frontmatter\nbody"})
