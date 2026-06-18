# LLM Provider Abstraction

One interface (`LLMProvider`) over many LLM backends, picked at runtime by env
var. `processor.py` never imports a concrete provider — it gets one from the
factory and calls the shared high-level methods.

## Layout (`services/llm/`)
- `base.py` — `LLMProvider` ABC. Providers implement `generate()`; the wiki
  methods (`generate_wiki`, `update_wiki`, `generate_overview`,
  `generate_concepts`), prompts, JSON extraction, and mock mode live here.
- `config.py` — `LLMConfig` + `load_from_env()`.
- `factory.py` — `LLMProviderFactory.create(config)` by provider name.
- `providers/` — one file per backend.

## Providers
`minimax` (default), `openai`, `anthropic`, `gemini`, `groq`, `azure`,
`openai-compatible` (Ollama / vLLM / LM Studio).

## Configuration
```env
LLM_PROVIDER=minimax        # required
LLM_API_KEY=sk-...          # falls back to MINIMAX_API_KEY
LLM_MODEL=MiniMax-M2.7      # required
LLM_BASE_URL=http://...     # openai-compatible only
LLM_TEMPERATURE=0.3         # optional
LLM_MAX_TOKENS=4000         # optional
MOCK_LLM=true               # no API calls; deterministic output from input
```
`MOCK_LLM=true` is the keyless path used by tests and quickstart: extraction is
derived deterministically from the input markdown, so it still reflects each
app's real content.

## Extraction (two-step)
The real LLM path runs a two-step chain of thought: **analyze** the docs
(endpoints, modules, contradictions, originating file) → **generate** the final
JSON grounded in that analysis, with a per-entry `sources` list. Less
hallucination than a single read-and-write pass. See `base.py` `_analyze` /
`_generate_from_analysis`.

## Add a provider
1. Add `providers/<name>.py` with a class subclassing `LLMProvider`,
   implementing `generate()`, `validate_config()`, `get_model_info()`.
2. Register it (decorate with `@LLMProviderFactory.register("<name>")` and import
   it in `providers/__init__.py`).
3. Add an entry to `.env.example`.

Errors raise the typed exceptions in `exceptions.py`
(`AuthenticationException`, `RateLimitException`, `APIException`, …) so callers
handle failures uniformly.
