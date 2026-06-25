# wiki-processor 實際怎麼運作（含真實紀錄）

> 這是**寫入端**。它收某個 app 的 markdown，叫 LLM 抽成結構，寫進該 app 自己的 MinIO 檔
> （正本）+ Postgres 向量（索引）。下面用**真 Minimax-M3 + 本地 bge-small embedding** 現場
> 跑出來的真實輸出，拆開這個「黑盒子」。擷取 2026-06-25；LLM 非確定性，你重跑數字會略不同。
>
> 想先看整套（跨四個服務）端到端流程：見平台 repo 的 `docs/HOW-IT-WORKS.md`。這份只聚焦
> wiki-processor 內部。

> **名詞**：**LLM** 大型語言模型；**embedding/向量** 文字轉數字、意思近數字近；
> **CAS（Compare-And-Swap）** 寫入前比對版本（ETag），相符才寫的樂觀鎖；
> **每-app 物件** 每個 app 各存一個檔 `apps/<app>.json`；**best-effort** 盡力、失敗不中斷主流程。

---

## 一筆 `/process` 的內部

```
markdown ─▶ Phase 1（不上鎖、可並發）            ─▶ Phase 2（CAS 寫自己的檔）
            兩段式抽取：analyze → generate           寫 apps/<app>.json
            蓋 source_app/sources、生 overview        ↓（best-effort）
            算 embedding                              同步 Postgres 索引 + audit + 通知 mcp 清快取
```

---

## 步驟 1（核心）— 兩段式抽取（two-step extraction）

真 LLM 路徑跑**兩段** chain-of-thought（思路鏈）：先 **analyze**（讀文件、列出 endpoint /
module / 矛盾 / 來源檔），再 **generate**（基於分析產生最終 JSON）。比一次到位更少幻覺。

把 payments.md 丟進第 1 段 `_analyze`，真 M3 輸出（節錄）：
```
# API Documentation Analysis

## Endpoints
| HTTP Method | Path | Purpose | Source File |
|-------------|------|---------|-------------|
| POST | /payments/charge | Charge a saved credit card to collect payment. | payments.md |
| POST | /payments/refund | Refund money back to a customer. | payments.md |

## Module/Service
- **Payments API** (from payments.md): Handles payment collection and refund operations …

## Contradictions or Duplicates
- No contradictions or duplicate definitions were found …
```
**這代表什麼：** LLM 先「想清楚」有哪些 endpoint、屬於哪個 module、有沒有衝突，第 2 段才把它
變成乾淨 JSON。`MOCK_LLM=true` 時改用確定性規則從 markdown 推導同樣結構（免 key、可測）。
**壞了會怎樣：** LLM 回傳格式怪（多包一層、加 think 標籤）→ `extract_json` 容錯解析。

---

## 步驟 2 — 寫進自己的 MinIO 檔（CAS，per-app）

generate 後，processor 蓋上 `source_app`/`sources`（不信任 LLM 自報出處），並對
**`apps/payments-api.json`** 做 CAS 寫入。真內容（節錄）：
```json
{
  "schema_version": 2, "source_app": "payments-api", "source_version": "v1",
  "apis": { "Payments": {
    "POST /payments/charge": {"method":"POST","path":"/payments/charge",
      "description":"Charge a saved credit card to collect payment.",
      "sources":["payments.md"],"source_app":"payments-api","source_version":"v1"},
    "POST /payments/refund": {"...":"..."} } },
  "knowledge": {},
  "overview": "The payments-api service ... charges ... and refunds ...（真 LLM 生的總覽）",
  "updated_at": "2026-06-22T13:49:03.963944"
}
```
**這代表什麼：** 一次推送只重寫**這一個 app 的檔**（O(1)），不碰別人。LLM 還順手生了 `overview`。
**壞了會怎樣：** 兩個請求同時改同一 app → CAS 比 ETag，輸的重讀重試（最多 5 次）。不同 app
各自一檔 → 根本不競爭。這就是「每-app 物件」（P3）的好處：寫入延遲不隨 app 數變慢。

每次推送另寫一筆 audit（append-only）：
```json
{"timestamp":"2026-06-22T13:49:03.980953","source_app":"payments-api","files_count":1,
 "status":"success","files_updated":["POST /payments/charge","POST /payments/refund"]}
```

---

## 步驟 3 — 算 embedding、同步 Postgres 索引（best-effort）

`embed_text`（攤平的字串，非原始 markdown）丟給 embedding 模型 → 384 個數字，存進 `api_entries`：
```
api_key         | POST /payments/charge
embed_text      | Payments | POST /payments/charge | POST /payments/charge | Charge a saved credit card to collect payment.
dim             | 384
embedding_model | bge-small
向量前 8 維       | {-0.061822545,-0.055060707,-0.015334889,0.025303239,0.027342524,-0.073736265,0.027584018,0.044032466}
```
知識文件走一樣的路，存到 `knowledge_entries`。
**這代表什麼：** 索引是**衍生副本**，查詢端拿這串數字比語意。
**壞了會怎樣：** embedding 服務掛 → 存 NULL 向量（仍可關鍵字）；PG 掛 → 整個同步跳過、記
audit、不影響 MinIO 正本；之後 `POST /admin/reindex` 從每-app 物件重建。

---

## 步驟 4 — 限流保護（rate-limit backoff）

大量 app 併發推送會撞 LLM 供應商的 rate limit（429）。`services/llm/base.py` 的
`_generate_retry`：對 `RateLimitException`/`APIException` 做**指數退避重試**（含 jitter），
並用**信號量**限制同時呼叫 LLM 的數量（`LLM_MAX_CONCURRENCY`，預設 3）。
**實證**：併發 8、真 M3 → 失敗率 **65% → 0%**（稍慢但全成功）。可調：`LLM_MAX_RETRIES`、
`LLM_RETRY_BASE_SECONDS`、`LLM_MAX_CONCURRENCY`。

---

## 管理端點
- `POST /admin/reindex` — 從每-app 物件重建 Postgres 索引。
- `POST /admin/recompile` — 對已存 snapshot 重跑抽取（抽取/prompt 改動後用）。
- `POST /admin/rebuild-concepts` — 跨應用概念合成 + 重建彙總 `wiki.json`（批次/排程，非每次推送）。

## 自己重跑
```bash
curl -X POST localhost:8001/process -H 'Content-Type: application/json' -d '{
  "markdowns":{"payments.md":"# Payments API\nPOST /payments/charge - Charge a saved credit card."},
  "timestamp":"...","trigger_info":{},"source_app":"payments-api","source_version":"v1"}'
docker exec wiki-processor python -c "import json;from repository.minio_client import MinioStorage;print(json.dumps(MinioStorage().get_json('apps/payments-api.json'),ensure_ascii=False,indent=2))"
```
更多：[docs/api.md](api.md)、[docs/architecture/diagram.md](architecture/diagram.md)、
[docs/concurrency.md](concurrency.md)、[docs/llm-provider-abstraction.md](llm-provider-abstraction.md)。
</content>
