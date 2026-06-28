"""Canonical entry -> embedding-text builder.

The exact text that gets embedded for an API entry. Index time
(wiki-processor) and query-side tooling must agree on this format;
mcp-server/services/embeddings.py carries a byte-identical copy pinned by
golden tests in both suites.
"""

import json

_PARAMS_MAX_CHARS = 500


def entry_to_text(module: str, api_key: str, detail) -> str:
    """Build the text to embed for one API entry.

    Format: "{description} | {module} | {api_key} | {params-json}" with empty
    parts dropped. The description leads because it is the semantic, translatable
    part and must dominate the embedding — leading with the module name and the
    English METHOD/path otherwise pulls cross-language queries off-target (a
    Chinese query ranking every Chinese-described entry above an equally-relevant
    English one). The old format also repeated the endpoint, which is already
    carried verbatim by api_key ("{METHOD} {path}"); that duplication is dropped.
    Parameters JSON is truncated — field names matter for recall, not schemas.
    """
    if not isinstance(detail, dict):
        detail = {"description": str(detail)}

    params = detail.get("parameters")
    params_part = ""
    if params:
        params_part = json.dumps(params, ensure_ascii=False, sort_keys=True)[:_PARAMS_MAX_CHARS]

    description = str(detail.get("description") or "").strip()

    parts = [description, module, api_key, params_part]
    return " | ".join(p for p in parts if p)


def knowledge_to_text(_doc_id: str, entry) -> str:
    """Build the text to embed for one knowledge document.

    ``_doc_id`` is accepted to mirror ``entry_to_text``'s keyed signature but is
    not part of the embedded text — knowledge vectors are content-only.

    Concatenates title | summary | topics | key_points — the distilled fields,
    which for these short docs act as one focused chunk (good embedding recall
    without diluting the vector across a long raw document).
    """
    if not isinstance(entry, dict):
        entry = {"summary": str(entry)}
    parts = [
        str(entry.get("title") or ""),
        str(entry.get("summary") or ""),
        " ".join(entry.get("topics", []) or []),
        " ".join(entry.get("key_points", []) or []),
    ]
    return " | ".join(p for p in parts if p.strip())
