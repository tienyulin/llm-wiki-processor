# wiki-processor API

Base URL：`http://localhost:8001`

> 名詞：**CAS（Compare-And-Swap）** = 寫入前比對版本標記（ETag），相符才寫的樂觀鎖；
> **每-app 物件** = 每個 app 的資料各存一個檔 `apps/<app>.json`；**best-effort** = 盡力，
> 失敗不中斷主流程。

## `POST /process`
收某個 app 的 markdown，用 LLM 抽出 API 條目，對該 app 做 **app 級增量更新**：寫入它自己的
物件 `apps/<app>.json`（用 ETag CAS 樂觀鎖，各 app 各自一把鎖、互不競爭）。並盡力把條目
同步進 PG/pgvector 索引。

Request：
```json
{
  "markdowns": {"api.md": "# ... markdown ..."},
  "timestamp": "2026-06-14T00:00:00",
  "trigger_info": {"source": "ci"},
  "source_app": "my-app",          // app 身分；條目以此 stamp 並按 app 取代
  "source_version": "git-sha",
  "doc_type": "api"                // "api" | "knowledge"；省略則自動判斷
}
```

**文件型別。** 推送解析為兩種 kind —— `api` 或 `knowledge`，決定抽取方式：
- `api` —— 把 endpoint 抽進 `wiki.apis`。
- `knowledge` —— 從**散文/參考文件**（如 Oracle、FastAPI how-to、cronjob/worker 元件 README）抽出結構化條目
  （title、summary、topics、key_points、`doc_type`、`tags`）進 `wiki.knowledge`，讓 wiki 也裝 agent 能推理的通用知識。

**判定優先序（type 權威，heuristic 只是最後手段）：**
1. 明確的 `doc_type`（請求欄位）—— 正規化後只看是不是 `api`（不分大小寫）；其餘一律 `knowledge`。
2. 附了 `openapi` spec → `api`（確定性匯入）。
3. 來源文件 frontmatter 的 `type` —— **有宣告就權威**：只有 `type: api` 是 api，
   其餘宣告型別（`tutorial`/`how-to`/`reference`/`explanation`，或 cronjob/worker 用的 `reference`）都是 `knowledge`。
4. **都沒宣告**時才用內容 heuristic：散文中（程式碼區塊外）出現 `METHOD /path` → `api`，否則 `knowledge`。

→ 因此 wiki-doc-author 產的合規文件（一定帶受控 `type`）永遠走宣告，不會被散文中順帶出現的
endpoint 句子誤判；heuristic 僅服務沒有 frontmatter 的舊文件。

知識條目帶同樣的 `sources`/`source_app`/`source_version` 出處；提到某概念 token 的知識文件
（如 Oracle「flashback」文件寫了「recover」）會被 `/admin/rebuild-concepts` 連到該概念
—— 橋接領域（knowledge ⇄ API）。

知識條目也會 embed 進 `knowledge_entries` pgvector 表（向量 + trigram 索引），所以
mcp-server 提供 **hybrid**（語意 + 關鍵字）知識搜尋。與 API 索引一樣 best-effort；
`/admin/recompile` 可從 snapshot 重建。

Response（200）：
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
驗證：設了 `PROCESSOR_API_KEY` 時要帶 `X-API-Key: $PROCESSOR_API_KEY`（空 = 開放 dev 模式）。
錯誤以 `{"status":"failed", ...}` + HTTP 200 回傳（LLM/embedding 失敗優雅降級）。

### API 條目格式（在每-app 物件 / 彙總 `wiki.json` / `GET /get_api_detail`）
```json
{
  "method": "POST",
  "path": "/recover",
  "description": "start a flashback recovery job",
  "sources": ["flashback.md"],   // 此條目從哪個 markdown 抽出
  "source_app": "flashback-api", // 由 processor stamp（出處不信任 LLM 輸出）
  "source_version": "v1.0"
}
```
`sources` 提供每條目回溯到原始 markdown 的可追溯性；`source_app`/`source_version` 由
processor stamp，用於 app 級取代/合併。

### 抽取（兩段式，僅真 LLM）
真 LLM 路徑跑兩段式 chain of thought：**analyze**（分析文件：endpoint、module、矛盾、
來源檔）→ **generate**（基於分析產生最終 JSON，每條目附 `sources`）。比 single-pass
讀寫更少幻覺。Mock 模式（`MOCK_LLM=true`）從輸入 markdown 確定性推導出同樣格式。

## `GET /status`
`{"status":"running","wiki_size":<app 數>,"tracked_files":...,"last_updated":...}`

## `POST /admin/reindex`
從每-app 物件重建 PG 索引（`{"status":"ok","apps":N,"entries":M,"embedded":M}`）。
PG 關閉時（`PG_DSN` 空）回 503。

## `POST /admin/recompile`
對已存的每-app snapshot 重跑抽取，不需重新匯入 —— 抽取/prompt 改動後用。
`{"status":"ok","recompiled_apps":[...],"count":N}`。

## `POST /admin/rebuild-concepts`
跨應用概念合成，並重建**衍生彙總 `wiki.json`**（合併所有 app + concepts + overviews）。
批次推送後或排程跑（非每次匯入 —— 它讀全部 app）。`{"status":"ok","concepts":N,...}`。

> P3 後：每-app 物件 `apps/<app>.json` 是真相來源；彙總 `wiki.json`（給 mcp 讀 concepts/
> overviews + fallback）由本端點按需重建。`wiki.json` 含 `overviews`、`concepts`、
> `knowledge`，全部由 mcp-server 提供。

## `GET /health`
`{"status":"ok","minio_connected":true,"llm_provider":...,"minimax_accessible":...,
  "vector_index_connected":...,"embeddings_configured":...}`

另見：[llm-provider-abstraction](llm-provider-abstraction.md)、
[concurrency](concurrency.md)、平台的 `docs/architecture/vector-search.md`。
</content>
