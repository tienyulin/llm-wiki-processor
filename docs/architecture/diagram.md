# wiki-processor —— 架構

匯入 + 索引服務。分層：`api/`（HTTP）→ `core/`（DI 依賴注入）→ `services/`（邏輯）
→ `repository/`（MinIO + PG）。

> 名詞：**DI（dependency injection，依賴注入）** = 由外部把相依物件傳進來，便於替換/測試；
> **CAS** = ETag 樂觀鎖；**每-app 物件** = `apps/<app>.json`，一次推送只寫自己的（P3）。

## 內部分層

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
    MINIO["repository/minio_client.py<br/>apps/&lt;app&gt;.json (CAS, source of truth)"]
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

## /process pipeline（兩階段，多 replica 安全）

```mermaid
flowchart TD
    A["markdown in"] --> P1
    subgraph P1["Phase 1 — read + LLM (no lock, concurrent)"]
        RW["read app object + snapshot"] --> CH{changes?}
        CH -->|none| SKIP["return: unchanged"]
        CH -->|yes| GEN["generate_wiki / update_wiki / generate_knowledge<br/>(two-step: analyze → generate)"]
        GEN --> STAMP["_stamp source_app/version<br/>+ sources[] per entry"]
        STAMP --> OVR["generate_overview(app)"]
        OVR --> EMBED["embed rows (if PG on)"]
    end
    EMBED --> P2
    subgraph P2["Phase 2 — write this app's object (bounded CAS loop)"]
        BUILD["_build_app_object<br/>(this app only — no other apps in the file)"] --> WR{"If-Match ETag on apps/&lt;app&gt;.json"}
        WR -->|conflict| RR["re-read, retry"] --> BUILD
        WR -->|ok| DONE[("apps/&lt;app&gt;.json")]
    end
    DONE --> SNAP["write app snapshot"]
    SNAP --> SYNC["PG sync (best-effort)"]
    SYNC --> INV["notify mcp-server /cache/invalidate"]
```

> P3：寫入只動該 app 自己的物件（O(1)，各 app 各自一把 CAS 鎖、無全域鎖、不互卡）。
> 彙總 `wiki.json`（concepts/overviews + 合併視圖）由 `/admin/rebuild-concepts` 按需重建。

CAS 契約見 [`docs/concurrency.md`](../concurrency.md)；LLM 層見
[`docs/llm-provider-abstraction.md`](../llm-provider-abstraction.md)；端點見
[`docs/api.md`](../api.md)。
</content>
