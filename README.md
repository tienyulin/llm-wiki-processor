# wiki-processor

Ingestion + indexing service for the LLM Wiki platform. Receives an application's
markdown, uses an LLM to extract structured API entries, applies an **app-level
incremental update** to the shared wiki in MinIO (optimistic ETag CAS — safe under
concurrent pushes from many apps), and best-effort syncs the entries into a
Postgres/pgvector index for keyword + semantic search.

Part of the [llm-wiki-mcp platform](https://github.com/tienyulin/llm-wiki-mcp);
deployable on its own.

## Architecture
```
POST /process ──> LLM extraction ──> MinIO wiki.json (CAS, source of truth)
                                          └─> PG/pgvector index (derived, best-effort)
```
- `api/` — FastAPI routes (`/process`, `/status`, `/admin/reindex`, `/health`)
- `services/` — `processor.py` (CAS pipeline), `llm/` (7-provider abstraction), `embeddings/`
- `repository/` — `minio_client.py`, `pg_store.py`
- `core/` — config + dependency injection

## Quickstart (standalone)
```bash
cp .env.example .env          # keep MOCK_LLM=true for a no-key run
docker compose up -d --build  # brings up minio + pg + wiki-processor
curl localhost:8001/health
```
Run without the vector index: `PG_DSN= docker compose up -d --build minio wiki-processor`.

## Develop in a Dev Container
This repo ships a [`.devcontainer/`](.devcontainer/). In VS Code / Cursor:
**Reopen in Container** — it builds this service + its deps (minio, pg), mounts the
source live at `/app`, and gives you an isolated Python env (no host pollution,
no clash with the other services). Inside the container:
```bash
python -m pytest         # run the tests
python main.py           # run the service (:8001); edits reflect live
```

## Push an app's docs
```bash
curl -X POST localhost:8001/process -H 'Content-Type: application/json' -d '{
  "markdowns":{"api.md":"# My API\n\nGET /my-app/items - list items"},
  "timestamp":"2026-06-14T00:00:00","trigger_info":{"source":"manual"},
  "source_app":"my-app","source_version":"v1"}'
```

## Configuration
See [`.env.example`](.env.example). Key vars: `LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`
(or `MOCK_LLM=true`), `MINIO_*`, `PG_DSN` (empty disables the index),
`EMBEDDING_*` (+ `EMBEDDING_SEND_DIMENSIONS=true` for Gemini).

## Tests
```bash
python -m pytest            # hermetic unit tests (Minio SDK + LLM stubbed)
```
Real-MinIO CAS tests and real-PG store tests auto-skip when those servers are unreachable.

## Docs
- [LLM provider abstraction](docs/llm-provider-abstraction.md)
- [Concurrency model (CAS write pipeline)](docs/concurrency.md)
- [API reference](docs/api.md)
- Cross-cutting (platform): `docs/architecture/vector-search.md`, `docs/architecture/service-layering.md`
