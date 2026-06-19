"""Concept-link dominant-app filter — guards the false-link bug found in the
stress run: a generic doc (FastAPI how-to) was flatly similar to endpoints
across many apps and got linked to unrelated concepts (items, refunds).

Numbers below are the real measured cosines from that run.
"""
from repository.pg_store import _dominant_app_links

T, M = 0.63, 0.05  # production defaults

# (module, api_key, source_app, score), sorted score desc — as the query returns.
ORACLE = [
    ("flashback-api", "GET /recover/{id}", "flashback-api", 0.773),
    ("flashback-api", "POST /recover", "flashback-api", 0.768),
    ("flashback-api", "GET /health", "flashback-api", 0.735),
    ("payments-api", "POST /refunds", "payments-api", 0.530),
]
SYN = [
    ("flashback-api", "POST /recover", "flashback-api", 0.656),
    ("flashback-api", "GET /recover/{id}", "flashback-api", 0.636),
    ("flashback-api", "GET /health", "flashback-api", 0.605),
    ("inventory-api", "PATCH /items/{id}", "inventory-api", 0.565),
]
JWT = [
    ("auth-api", "POST /token/refresh", "auth-api", 0.699),
    ("auth-api", "POST /logout", "auth-api", 0.646),
    ("auth-api", "GET /me", "auth-api", 0.636),
    ("auth-api", "POST /login", "auth-api", 0.633),
    ("auth-api", "GET /health", "auth-api", 0.628),
    ("payments-api", "POST /refunds", "payments-api", 0.620),
]
FASTAPI = [  # generic how-to — flat across apps, must link to NOTHING
    ("inventory-api", "POST /items", "inventory-api", 0.648),
    ("inventory-api", "GET /items", "inventory-api", 0.635),
    ("payments-api", "POST /refunds", "payments-api", 0.633),
    ("inventory-api", "PATCH /items/{id}", "inventory-api", 0.630),
    ("payments-api", "POST /charges", "payments-api", 0.626),
]


def test_specific_docs_link_to_their_app():
    assert all(m == "flashback-api" for m, _, _ in _dominant_app_links(ORACLE, T, M))
    assert ("flashback-api", "POST /recover", 0.656) in _dominant_app_links(SYN, T, M)
    jwt = _dominant_app_links(JWT, T, M)
    assert jwt and all(m == "auth-api" for m, _, _ in jwt)


def test_generic_doc_links_to_nothing():
    # The bug: fastapi-howto used to false-link to items + refunds.
    assert _dominant_app_links(FASTAPI, T, M) == []


def test_floor_excludes_weak_same_app_matches():
    # syn-kb: GET /health (0.605) is same-app but below the 0.63 floor → dropped.
    keys = [k for _, k, _ in _dominant_app_links(SYN, T, M)]
    assert "GET /health" not in keys
    assert "POST /recover" in keys


def test_empty():
    assert _dominant_app_links([], T, M) == []
