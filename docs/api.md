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

## `GET /status`
`{"status":"running","wiki_size":<app count>,"tracked_files":...,"last_updated":...}`

## `POST /admin/reindex`
Rebuild the PG index from MinIO (`{"status":"ok","apps":N,"entries":M,"embedded":M}`).
Returns 503 when PG is disabled (`PG_DSN` empty).

## `GET /health`
`{"status":"ok","minio_connected":true,"llm_provider":...,"minimax_accessible":...,
  "vector_index_connected":...,"embeddings_configured":...}`

See also: [llm-provider-abstraction](llm-provider-abstraction.md),
[concurrency](concurrency.md), and the platform's
`docs/architecture/vector-search.md`.
