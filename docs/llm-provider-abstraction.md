# LLM Provider 抽象層

用一個介面（`LLMProvider`）罩住多家 LLM 後端，執行時由環境變數挑選。`processor.py`
從不 import 具體 provider —— 從 factory 拿一個，呼叫共用的高階方法。

> 名詞：**ABC（Abstract Base Class）** = 抽象基底類別，定義介面；**factory** = 工廠模式，
> 依名字產生對應實作；**two-step extraction** = 先分析、再產生 JSON 的兩段式抽取。

## 結構（`services/llm/`）
- `base.py` —— `LLMProvider` ABC。各 provider 實作 `generate()`；wiki 方法
  （`generate_wiki`、`update_wiki`、`generate_overview`、`generate_concepts`、
  `generate_knowledge`）、prompt、JSON 抽取、mock 模式、**限流退避**都住這裡。
- `config.py` —— `LLMConfig` + `load_from_env()`。
- `factory.py` —— `LLMProviderFactory.create(config)` 依 provider 名字產生。
- `providers/` —— 每個後端一個檔。

## Providers
`minimax`（預設）、`openai`、`anthropic`、`gemini`、`groq`、`azure`、
`openai-compatible`（Ollama / vLLM / LM Studio）。

## 設定
```env
LLM_PROVIDER=minimax        # 必填
LLM_API_KEY=sk-...          # 沒設則回退 MINIMAX_API_KEY
LLM_MODEL=MiniMax-M3        # 必填
LLM_BASE_URL=http://...     # 僅 openai-compatible 需要
LLM_TEMPERATURE=0.3         # 選填
LLM_MAX_TOKENS=4000         # 選填
MOCK_LLM=true               # 不呼叫 API；從輸入確定性產出
# 限流保護（併發推送撞 429 時）
LLM_MAX_RETRIES=4           # 指數退避重試次數
LLM_RETRY_BASE_SECONDS=2    # 退避基礎秒數
LLM_MAX_CONCURRENCY=3       # 同時呼叫 LLM 的上限（信號量）
```
`MOCK_LLM=true` 是測試與 quickstart 用的免 key 路徑：抽取從輸入 markdown 確定性推導，
所以仍反映各 app 的真實內容。

## 抽取（兩段式）
真 LLM 路徑跑兩段式 chain of thought：**analyze**（分析文件：endpoint、module、矛盾、
來源檔）→ **generate**（基於分析產生最終 JSON，每條目附 `sources`）。比 single-pass
讀寫更少幻覺。見 `base.py` 的 `_analyze` / `_generate_from_analysis`。

## 限流保護
大量 app 併發推送會撞 LLM 供應商的 rate limit（429）。`base.py` 的 `_generate_retry`
對 `RateLimitException`/`APIException` 做指數退避重試（含 jitter），並用信號量限制同時
呼叫 LLM 的數量 —— 讓 processor 不超過供應商速率、把其餘排隊，把失敗轉成「稍慢但成功」。

## 新增一個 provider
1. 加 `providers/<name>.py`，類別繼承 `LLMProvider`，實作 `generate()`、
   `validate_config()`、`get_model_info()`。
2. 註冊（`@LLMProviderFactory.register("<name>")` 並在 `providers/__init__.py` import）。
3. 在 `.env.example` 加一筆。

錯誤拋 `exceptions.py` 裡的型別化例外（`AuthenticationException`、`RateLimitException`、
`APIException`…），讓呼叫端統一處理失敗。
</content>
