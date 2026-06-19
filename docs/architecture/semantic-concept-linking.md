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

## Honest note

The relevant/irrelevant margin here is thin (~0.06) — bge-small rates all
database-ish text fairly similar. Substring links stay as the high-precision
backbone; semantic linking adds synonym recall above a conservative floor. For a
larger / noisier corpus, raise `CONCEPT_LINK_MIN_COSINE` or add a light LLM tag
pass (the SAG-distilled idea) if false links appear.
