# wiki-processor

> 👉 想看這個服務**實際怎麼運作（含真實紀錄）**：[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)。

LLM Wiki 平台的**寫入端（ingestion + indexing）**服務。收某個 app 的 markdown，
用 LLM 抽出結構化的 API 條目，對 MinIO 裡的 wiki 做**應用級增量更新（app-level
incremental update）**，並盡力（best-effort）把條目同步進 Postgres/pgvector 索引供
關鍵字 + 語意搜尋。

隸屬 [llm-wiki-mcp 平台](https://github.com/tienyulin/llm-wiki-mcp)，也可獨立部署。

> **名詞**
> - **ingestion**：把外部文件「收進來、處理、入庫」的過程。
> - **CAS（Compare-And-Swap）**：寫入前比對版本標記（ETag），相符才寫 —— 樂觀鎖，防並發互蓋。
> - **embedding / 向量**：文字轉數字陣列，供語意搜尋。
> - **best-effort（盡力而為）**：失敗不會中斷主流程，事後可重建。

## 架構
```
POST /process ──> LLM 抽取 ──> MinIO：每-app 物件 apps/<app>.json（真相來源，CAS 寫入）
                                   └─> Postgres/pgvector 索引（衍生、盡力同步）
彙總 wiki.json（概念/總覽/合併視圖）由 /admin/rebuild-concepts 按需重建
```
- `api/` — FastAPI 路由（`/process`、`/status`、`/health`、`/admin/{reindex,recompile,rebuild-concepts}`）
- `services/` — `processor.py`（CAS pipeline + 概念/總覽）、`llm/`（7 家供應商抽象 + 兩段式抽取 + 限流退避）、`embeddings/`、`vector_sync.py`（PG 索引同步）
- `repository/` — `minio_client.py`、`pg_store.py`
- `core/` — 設定 + 依賴注入

> **每-app 物件**：每個 app 的資料存自己的檔，一次推送只重寫自己的檔（O(1)），
> 100+ app 同時更新不互卡。彙總 `wiki.json` 是衍生物，批次後重建。

## 快速開始
用**共用 infra**（[llm-wiki-infra](https://github.com/tienyulin/llm-wiki-infra)：一套
MinIO + Postgres 在 `llm-wiki-net` 網路上），所以可與其他服務並跑、不撞埠。先起
infra 一次，再起本服務：
```bash
# 1) 共用 infra（一次，從隔壁的 llm-wiki-infra clone）
(cd ../llm-wiki-infra && docker compose up -d)
# 2) 本服務
cp .env.example .env          # 不想填 key → 保持 MOCK_LLM=true
docker compose up -d --build
curl localhost:8001/health
```
不要向量索引：在 `.env` 設 `PG_DSN=`（空字串）。

## Dev Container（容器內開發）
本 repo 附 [`.devcontainer/`](.devcontainer/)。**先起共用 infra**
（`cd ../llm-wiki-infra && docker compose up -d`），再於 VS Code / Cursor：
**Reopen in Container** —— 建置本服務、原始碼即時掛載於 `/app`、獨立 Python 環境、
接上共用的 `llm-wiki-net`。容器內：
```bash
python -m pytest         # 跑測試
python main.py           # 跑服務（:8001）；改檔即時反映
```

## 推一個 app 的文件
```bash
curl -X POST localhost:8001/process -H 'Content-Type: application/json' -d '{
  "markdowns":{"api.md":"# My API\n\nGET /my-app/items - 列出項目"},
  "timestamp":"2026-06-20T00:00:00","trigger_info":{"source":"manual"},
  "source_app":"my-app","source_version":"v1"}'
```
`doc_type` 省略時自動判斷：內容含 HTTP endpoint → `api`，否則當 `knowledge`（知識文件）。

## 設定
見 [`.env.example`](.env.example)。重點變數：`LLM_PROVIDER`/`LLM_API_KEY`/`LLM_MODEL`
（或 `MOCK_LLM=true`）、`MINIO_*`、`PG_DSN`（空字串關閉索引）、`EMBEDDING_*`
（Gemini 需 `EMBEDDING_SEND_DIMENSIONS=true`）。
限流調校：`LLM_MAX_RETRIES`、`LLM_RETRY_BASE_SECONDS`、`LLM_MAX_CONCURRENCY`。

## 測試
```bash
python -m pytest            # 隔離單元測試（MinIO SDK + LLM 都被 stub 掉）
```
需要真 MinIO 的 CAS 測試、需要真 Postgres 的索引測試，在伺服器連不上時自動跳過（skip）。

## 文件
- [架構圖](docs/architecture/diagram.md) — 分層 + 兩段式 `/process` pipeline
- [LLM 供應商抽象](docs/llm-provider-abstraction.md)
- [並發模型（CAS 寫入 pipeline）](docs/concurrency.md)
- [API 參考](docs/api.md)
</content>
