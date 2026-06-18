# wiki-processor API

Base URL: `http://localhost:8001`

## `POST /process`
Ingest one application's markdown, extract API entries via the LLM, and apply an
**app-level incremental update** to the shared wiki (MinIO) using an optimistic
ETag CAS loop. Best-effort syncs the entries into the PG/pgvector index.

Request:
```json
{
  "markdowns": {"api.md": "# ... markdown ..."},
  "timestamp": "2026-06-14T00:00:00",
  "trigger_info": {"source": "ci"},
  "source_app": "my-app",          // app identity; entries stamped + replaced per app
  "source_version": "git-sha"
}
```
Response (200):
```json
{
  "status": "success",
  "message": "Wiki updated successfully",
  "source_app": "my-app",
  "files_updated": ["POST /my-app/items"],
  "changes_summary": {"added": [...], "modified": [...], "deleted": []},
  "processing_time_ms": 1234
}
```
Auth: send `X-API-Key: $PROCESSOR_API_KEY` when `PROCESSOR_API_KEY` is set
(empty = open dev mode). Errors are returned as `{"status":"failed", ...}` with
HTTP 200 (the LLM/embedding failures degrade gracefully).

### API entry shape (in `wiki.json` and `GET /get_api_detail`)
```json
{
  "method": "POST",
  "path": "/recover",
  "description": "start a flashback recovery job",
  "sources": ["flashback.md"],   // markdown file(s) this entry was extracted from
  "source_app": "flashback-api", // stamped by the processor (LLM output is not trusted for provenance)
  "source_version": "v1.0"
}
```
`sources` gives per-entry traceability back to the originating markdown; `source_app`
/`source_version` are stamped by the processor and used for app-level replace/merge.

### Extraction (two-step, real LLM only)
The LLM path runs a two-step chain of thought: **analyze** the docs (endpoints,
modules, contradictions, originating file) → **generate** the final JSON grounded in
that analysis, with a `sources` list per entry. Mock mode (`MOCK_LLM=true`) derives
the same shape deterministically from the input markdown.

## `GET /status`
`{"status":"running","wiki_size":<app count>,"tracked_files":...,"last_updated":...}`

## `POST /admin/reindex`
Rebuild the PG index from MinIO (`{"status":"ok","apps":N,"entries":M,"embedded":M}`).
Returns 503 when PG is disabled (`PG_DSN` empty).

## `POST /admin/recompile`
Re-run extraction over stored per-app snapshots without re-ingesting — use after
an extraction/prompt change. `{"status":"ok","recompiled_apps":[...],"count":N}`.

## `POST /admin/rebuild-concepts`
Cross-app concept synthesis over the whole wiki; writes `wiki.concepts`. Run after
a batch of pushes (not per-ingest — it scans the whole wiki).
`{"status":"ok","concepts":N}`.

`wiki.json` also carries `overviews` (`{app: {text, updated_at}}`, refreshed per
app ingest) and `concepts` (`{name: {description, related, apps}}`, built by
`/admin/rebuild-concepts`). Both are served by mcp-server.

## `GET /health`
`{"status":"ok","minio_connected":true,"llm_provider":...,"minimax_accessible":...,
  "vector_index_connected":...,"embeddings_configured":...}`

See also: [llm-provider-abstraction](llm-provider-abstraction.md),
[concurrency](concurrency.md), and the platform's
`docs/architecture/vector-search.md`.
