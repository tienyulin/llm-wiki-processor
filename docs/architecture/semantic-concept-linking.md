# Semantic concept linking (robustness)

Concept links used to be **substring** only: a knowledge doc joined concept
`recover` only if its text literally contained "recover". Synonyms broke it — a
doc saying "roll back / undo" was findable by hybrid *search* but never *linked*
into the cross-domain concept the agent relies on for multi-hop.

## Fix

`rebuild_concepts` now augments substring links with **semantic** ones: for each
knowledge doc, the nearest API entries by embedding cosine (reusing the vectors
already in `knowledge_entries` / `api_entries`) contribute the doc to those
endpoints' concepts. Substring links are kept, so this only *adds* recall.

```
substring links (precise)  ∪  semantic links (cosine ≥ threshold)
```

No new infra — one SQL cosine join (`PGVectorStore.knowledge_api_links`). No-op
when embeddings/PG are off (links stay substring-only).

## Threshold — measured, not guessed

knowledge-doc ↔ nearest API endpoint cosine (bge-small) on the live corpus:

| knowledge doc | nearest API | cosine | should link |
|---|---|---|---|
| oracle-kb:oracle-flashback | /recover | 0.773 | yes |
| runbook-kb:incident-runbook | /recover | 0.660 | yes |
| **syn-kb:undo-guide** (synonyms only) | /recover | **0.656** | **yes** |
| fastapi-kb:fastapi-howto | /recover | 0.599 | **no** (unrelated) |

The relevant/irrelevant gap is 0.599 → 0.656. Floor **0.63** sits in it (and in the
0.60–0.64 range reported in entity-linking papers). Tune via
`CONCEPT_LINK_MIN_COSINE`.

## Result (live)

`get_concept "recover"` before vs after, for the synonym doc:

| | substring only | + semantic |
|---|---|---|
| syn-kb:undo-guide linked | **no** | **yes** |
| fastapi-kb (unrelated) linked | no | **no** (correctly excluded) |
| concept spans | 3 apps | **4 apps** |

Via Claude: *"I need to roll back a table after a bad write — undo it"* →
`get_concept "recover"` surfaced the API **and** all three knowledge docs
(including the synonym doc) in one lookup.

## Stress-test finding → dominant-app gate

On a bigger corpus (4 services, 6 knowledge docs) a fixed cosine floor produced
**false links**: a generic "how to build a FastAPI endpoint" doc sat at a flat
0.61–0.65 cosine to endpoints across *many* apps (build-an-endpoint ≈ any
endpoint) and linked to `items`, `refunds`, … A specific doc instead concentrates
on one app:

| doc | best app | next *other* app | gap | link? |
|---|---|---|---|---|
| oracle-flashback | flashback 0.773 | payments 0.530 | 0.243 | ✓ |
| jwt | auth 0.699 | payments 0.620 | 0.079 | ✓ |
| syn-kb | flashback 0.656 | inventory 0.565 | 0.091 | ✓ |
| **fastapi-howto** | inventory 0.648 | payments 0.633 | **0.015** | ✗ |

Fix: a doc links semantically only when its **best app dominates** —
`best − best_of_any_other_app >= CONCEPT_LINK_MARGIN` (0.05). 0.05 cleanly
separates all four. Then that app's endpoints with cosine >= `CONCEPT_LINK_MIN_COSINE`
(0.63) are linked. Pure logic in `_dominant_app_links` (unit-tested on these numbers).

Result: the egregious semantic false-links (refunds, charges) are gone.

**Residual, by design:** substring linking still links a doc that *literally
contains* a token — the FastAPI how-to writes `@app.post('/items')`, so it links
to `items`. That's a real mention, not a hallucination, and low-harm (the agent
verifies via hybrid search). Raise the floor/margin or add an LLM tag pass only
if literal-mention links become noisy.
