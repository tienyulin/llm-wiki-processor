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

    Format: "{module} | {api_key} | {METHOD /path} | {description} | {params-json}"
    with empty parts dropped. Parameters JSON is truncated — field names are
    what matters for recall, not full nested schemas.
    """
    if not isinstance(detail, dict):
        detail = {"description": str(detail)}

    method = str(detail.get("method") or "").strip()
    path = str(detail.get("path") or "").strip()
    endpoint = " ".join(p for p in (method, path) if p)

    params = detail.get("parameters")
    params_part = ""
    if params:
        params_part = json.dumps(params, ensure_ascii=False, sort_keys=True)[:_PARAMS_MAX_CHARS]

    description = str(detail.get("description") or "").strip()

    parts = [module, api_key, endpoint, description, params_part]
    return " | ".join(p for p in parts if p)


def knowledge_to_text(doc_id: str, entry) -> str:
    """Build the text to embed for one knowledge document.

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
