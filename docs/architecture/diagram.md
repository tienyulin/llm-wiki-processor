# wiki-processor — architecture

Ingestion + indexing service. Layered: `api/` (HTTP) → `core/` (DI) →
`services/` (logic) → `repository/` (MinIO + PG).

## Internal layering

```mermaid
flowchart TD
    subgraph api["api/ (FastAPI routers)"]
        R1["process.py<br/>POST /process"]
        R2["admin.py<br/>/admin/reindex · recompile · rebuild-concepts"]
        R3["system.py<br/>/status · /health"]
    end
    DEPS["core/deps.py<br/>dependency injection / singletons"]
    PROC["services/processor.py<br/>WikiProcessor (CAS pipeline)"]
    LLM["services/llm/<br/>LLMProvider + factory (7 providers)"]
    EMB["services/embeddings/<br/>EmbeddingClient"]
    MINIO["repository/minio_client.py<br/>wiki.json (CAS, source of truth)"]
    PG["repository/pg_store.py<br/>pgvector index (derived)"]

    R1 --> DEPS
    R2 --> DEPS
    R3 --> DEPS
    DEPS --> PROC
    PROC --> LLM
    PROC --> EMB
    PROC --> MINIO
    PROC -.best-effort.-> PG
```

## /process pipeline (two-phase, multi-replica safe)

```mermaid
flowchart TD
    A["markdown in"] --> P1
    subgraph P1["Phase 1 — read + LLM (no lock, concurrent)"]
        RW["read wiki + snapshot"] --> CH{changes?}
        CH -->|none| SKIP["return: unchanged"]
        CH -->|yes| GEN["generate_wiki / update_wiki<br/>(two-step: analyze → generate)"]
        GEN --> STAMP["_stamp source_app/version<br/>+ sources[] per entry"]
        STAMP --> OVR["generate_overview(app)"]
        OVR --> EMBED["embed rows (if PG on)"]
    end
    EMBED --> P2
    subgraph P2["Phase 2 — merge + write (bounded CAS loop)"]
        MERGE["_merge_app_entries<br/>keep other apps + concepts/overviews"] --> WR{"If-Match ETag"}
        WR -->|conflict| RR["re-read, retry"] --> MERGE
        WR -->|ok| DONE[("wiki.json")]
    end
    DONE --> SNAP["write app snapshot"]
    SNAP --> SYNC["PG sync (best-effort)"]
    SYNC --> INV["notify mcp-server /cache/invalidate"]
```

See [`docs/concurrency.md`](../concurrency.md) for the CAS contract,
[`docs/llm-provider-abstraction.md`](../llm-provider-abstraction.md) for the LLM
layer, and [`docs/api.md`](../api.md) for endpoints.
