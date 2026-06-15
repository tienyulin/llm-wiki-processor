# 並發控制設計（Concurrency）

**狀態：** v2 — ETag CAS + 兩階段更新（2026-06-11）
**取代：** v1 的單一 process-local `asyncio.Lock`（會序列化 LLM 呼叫，且無法保護多副本）

---

## 問題

所有應用共用同一個 `wiki.json`。更新流程「讀 wiki → LLM 呼叫 → 寫 wiki」
橫跨一次秒級的 awaited LLM 呼叫，沒有並發控制時後寫者會覆蓋先寫者
（實測 20 個並發更新只有 1 個存活）。v1 用一把大鎖修正了正確性，但代價是
LLM 呼叫被完全序列化，且鎖是 process-local —— 多副本部署仍會 race。

## 解法：兩階段 + ETag CAS

```
Phase 1（無鎖、完全並行）
  讀 wiki + ETag ──► 取出該 app 現有 entries ──► LLM 呼叫（秒級）

Phase 2（process-local lock + CAS loop，毫秒級）
  merge（替換該 app 的 entries）──► put_json_if_match(etag)
       ▲                                    │ 412 conflict
       └──── 重讀 wiki + ETag ◄─────────────┘   （上限 5 次，jitter backoff）
```

- **跨副本安全**：寫入用 MinIO 條件寫入（`If-Match: <etag>`，首次建立用
  `If-None-Match: *`）。另一個副本搶先寫入時本副本收到 412，重讀、重 merge、
  重試 —— LLM 結果不變，重試只是毫秒級的 merge+write。
- **同進程突發**：Phase 2 另外包一把 process-local `asyncio.Lock`。
  純 CAS 在 N 路同進程突發下每輪只有一個贏家，重試預算會被耗盡；
  鎖把進程內的 Phase 2 串行化（鎖內只有一次 storage roundtrip），
  CAS 重試只需應付跨副本衝突。
- **吞吐**：LLM 呼叫（瓶頸）完全並行。100 個並發 app 實測（mock LLM、
  真實 MinIO）：100/100 成功、無 lost update。

## minio-py 實作備註

minio-py 7.2.x 的公開 `put_object` 不接受自訂 headers；條件寫入透過
`Minio._put_object(bucket, key, data, headers)`（私有但穩定，內部單次 PUT
路徑）。412 以 `S3Error(code="PreconditionFailed")` 拋出。
`wiki-processor/tests/test_storage_cas.py` 直接對真實 MinIO 驗證此行為，
minio 版本升級若破壞此 API 會立即被測試抓到。

## 周邊設計

- **Audit log**：每筆一個 append-only 物件（`audit/{iso-ts}-{uuid8}.json`），
  消除共用 NDJSON 檔的 read-modify-write 爭用。讀取 = list `audit/` prefix。
- **Snapshots**：app-level 更新寫 `snapshots/{app}.json`（v1 會覆寫全域
  snapshot、吃掉其他 app 的記錄）；無 `source_app` 的全量更新才寫全域
  `markdowns_snapshot.json`。內容未變的重複提交直接跳過 LLM。
- **Storage 非阻塞**：minio-py 是同步的；async 呼叫端透過
  `MinioStorage` 的 async facade（`asyncio.to_thread`）存取，
  mcp-server 的 wiki 讀取同樣走 `to_thread`。
- **快取一致性**：wiki-processor 成功更新後呼叫 mcp-server
  `POST /cache/invalidate`（`MCP_SERVER_URL`，best-effort）。

## 資料模型（schema v2）

```json
{
  "schema_version": 2,
  "apis": {"<module>": {"<METHOD /path>": {"...": "...", "source_app": "...", "source_version": "..."}}},
  "metadata": {"version": "...", "created_at": "...", "updated_at": "..."}
}
```

- provenance 由 processor 蓋章，不信任 LLM 輸出
- app-level 更新 = 移除該 `source_app` 的所有 entries 後併入新 entries
- v1 的混合形態（結構化 + file-map 字串條目）已移除；舊 wiki 讀取時
  lazy 遷移（保留結構化部分、丟棄 file-map 條目並記 warning），
  受影響的 app 在下次提交時自動重建

## 驗證

- `wiki-processor/tests/test_concurrency.py` — 20 路並發無 lost update、
  CAS 衝突注入（模擬另一副本插隊）、重提交取代自身 entries、v2 遷移
- `wiki-processor/tests/test_storage_cas.py` — 真實 MinIO 條件寫入行為
- `tests/stress/test_mock_stress.py` — 100 路並發 + 隔離 + audit（in-memory）
- `tests/stress/test_real_service_stress.py` — 100 路並發打真實服務 +
  真實 MinIO：per-app 完整性逐一驗證
