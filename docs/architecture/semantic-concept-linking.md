# 語意概念連結（robustness）

概念連結以前只有**子字串**：一份知識文件要文字裡literally含 "recover" 才會被連到
`recover` 概念。同義詞會破功 —— 只寫 "roll back / undo" 的文件雖然 hybrid *搜尋*找得到，
卻從不被*連*進 agent 做 multi-hop 所靠的跨領域概念。

> 名詞：**cosine** = 向量相似度；**dominant-app gate** = 「主導 app 邊界」，用來擋泛用文件誤連。

## 修法

`rebuild_concepts` 在子字串連結之外，再加**語意**連結：對每份知識文件，取 embedding
cosine 最近的 API 條目（重用 `knowledge_entries` / `api_entries` 已有的向量），把該文件
貢獻到那些 endpoint 的概念。子字串連結保留，所以這只**增加**召回。

```
子字串連結（精準）  ∪  語意連結（cosine ≥ 門檻）
```

不加新基礎設施 —— 一個 SQL cosine join（`PGVectorStore.knowledge_api_links`）。
embedding/PG 關閉時 no-op（連結維持只有子字串）。

## 門檻 —— 量測得來，非猜的

知識文件 ↔ 最近 API endpoint 的 cosine（bge-small）於 live 語料：

| 知識文件 | 最近 API | cosine | 該連嗎 |
|---|---|---|---|
| oracle-kb:oracle-flashback | /recover | 0.773 | 該 |
| runbook-kb:incident-runbook | /recover | 0.660 | 該 |
| **syn-kb:undo-guide**（純同義詞） | /recover | **0.656** | **該** |
| fastapi-kb:fastapi-howto | /recover | 0.599 | **不該**（無關） |

相關/不相關落差是 0.599 → 0.656。下限 **0.63** 落在其中（也在 entity-linking 論文常見的
0.60–0.64 範圍）。用 `CONCEPT_LINK_MIN_COSINE` 調。

## 結果（live）

同義詞文件在 `get_concept "recover"` 前後對比：

| | 只有子字串 | + 語意 |
|---|---|---|
| syn-kb:undo-guide 被連 | **否** | **是** |
| fastapi-kb（無關）被連 | 否 | **否**（正確排除） |
| 概念橫跨 | 3 apps | **4 apps** |

透過 Claude：*"我要在誤寫後回溯一張表 —— undo it"* → `get_concept "recover"` 一次查出
API **和**三份知識文件（含同義詞那份）。

## 壓測發現 → dominant-app 邊界

在更大語料（4 服務、6 知識文件）上，固定 cosine 下限會產生**誤連**：一份泛用的
「how to build a FastAPI endpoint」文件對*很多* app 的 endpoint 都平平地 0.61–0.65
（建 endpoint ≈ 任何 endpoint），被連到 `items`、`refunds`…。而專一的文件會集中在一個 app：

| 文件 | 最佳 app | 次佳的*其他* app | 落差 | 連嗎 |
|---|---|---|---|---|
| oracle-flashback | flashback 0.773 | payments 0.530 | 0.243 | ✓ |
| jwt | auth 0.699 | payments 0.620 | 0.079 | ✓ |
| syn-kb | flashback 0.656 | inventory 0.565 | 0.091 | ✓ |
| **fastapi-howto** | inventory 0.648 | payments 0.633 | **0.015** | ✗ |

修法：文件只有在其**最佳 app 主導**時才做語意連結 ——
`最佳 − 任一其他 app 的最佳 >= CONCEPT_LINK_MARGIN`（0.05）。0.05 乾淨分開四者。然後把該
app 中 cosine >= `CONCEPT_LINK_MIN_COSINE`（0.63）的 endpoint 連上。純邏輯在
`_dominant_app_links`（用這些數字做單元測試）。

結果：嚴重的語意誤連（refunds、charges）消失。

**設計上的殘留**：子字串連結仍會連到「文字裡literally含某 token」的文件 —— FastAPI
how-to 寫了 `@app.post('/items')`，所以連到 `items`。那是真的提及、非幻覺，低傷害
（agent 會用 hybrid 搜尋驗證）。只有當這種「字面提及」連結變吵時，才調高下限/邊界或加
LLM 標籤 pass。
</content>
