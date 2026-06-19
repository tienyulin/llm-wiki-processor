"""Abstract base class for all LLM providers"""

import asyncio
import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .exceptions import APIException, RateLimitException

logger = logging.getLogger(__name__)

_ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}.-]*)")
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class LLMProvider(ABC):
    """
    Unified interface for LLM providers.

    All providers must implement:
        generate()      - send a prompt, get back text
        validate_config() - check API key / connectivity
        get_model_info() - return model metadata
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt.

        Returns:
            Raw text content from the model (callers do their own JSON parsing).

        Raises:
            AuthenticationException: bad API key
            RateLimitException: rate limit hit
            APIException: other API-level error
            ValidationException: unexpected response shape
        """

    @abstractmethod
    async def validate_config(self) -> bool:
        """Check that credentials and connectivity are working.

        Returns True on success; raises on failure.
        """

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata: model_name, max_context, provider, …"""

    def is_configured(self) -> bool:
        """Cheap local configuration check — no API call.

        Unlike validate_config(), this is safe to call from frequently polled
        health endpoints. Mock mode counts as configured; otherwise an API key
        must be present (openai-compatible servers may run keyless, so
        providers without a key report unconfigured here).
        """
        if self._mock_mode():
            return True
        config = getattr(self, "config", None)
        return bool(config and config.api_key)

    # ------------------------------------------------------------------
    # High-level wiki methods, shared by all providers (consumed by
    # processor.py). Providers only supply generate(); prompts, JSON
    # extraction, and mock mode live here.
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_mode() -> bool:
        return os.getenv("MOCK_LLM", "false").lower() == "true"

    def _llm_semaphore(self) -> "asyncio.Semaphore":
        """Lazily-built cap on concurrent LLM calls (per process).

        A fleet of apps pushing at once drove the provider into sustained 429s;
        retry alone only cut failures from 65% to ~43%. Bounding concurrency so
        the processor never exceeds the provider's rate — queuing the rest — is
        the real lever. Tunable via LLM_MAX_CONCURRENCY (default 3).
        """
        sem = getattr(self, "_sem", None)
        if sem is None:
            sem = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "3")))
            self._sem = sem
        return sem

    async def _generate_retry(self, prompt: str, **kwargs) -> str:
        """generate() bounded by a concurrency cap, with exponential backoff on
        rate-limit / transient errors. Tunable via LLM_MAX_RETRIES (default 4),
        LLM_RETRY_BASE_SECONDS (default 2), LLM_MAX_CONCURRENCY (default 3)."""
        attempts = int(os.getenv("LLM_MAX_RETRIES", "4"))
        base = float(os.getenv("LLM_RETRY_BASE_SECONDS", "2"))
        for attempt in range(attempts + 1):
            try:
                async with self._llm_semaphore():
                    return await self.generate(prompt, **kwargs)
            except (RateLimitException, APIException) as e:
                if attempt == attempts:
                    raise
                delay = base * (2 ** attempt) + random.uniform(0, base)
                logger.warning(
                    f"{type(self).__name__}: {type(e).__name__}, retry "
                    f"{attempt + 1}/{attempts} in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

    def extract_json(self, content: str) -> dict:
        """Strip <think> tags and parse JSON from an LLM response."""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # LLMs may wrap the JSON in code fences or append trailing text or a
            # second object (e.g. MiniMax-M2.7/M3). Decode the first complete
            # JSON object starting at the first '{' and ignore anything after it.
            start = content.find("{")
            if start != -1:
                obj, _ = json.JSONDecoder().raw_decode(content[start:])
                return obj
            raise

    @staticmethod
    def _mock_apis_from_markdowns(
        markdowns: Dict[str, str], source_app: Optional[str] = None
    ) -> dict:
        """Derive deterministic API entries from the input markdowns.

        Mock responses must reflect the input (not canned data) so that
        integration and stress tests can verify each app's data actually
        reaches the wiki. Every file yields at least one entry.

        Module key matches what a real LLM produces for the same input: the
        pushing application's identity (``source_app``) when known, else the
        file stem. This keeps mock and real extraction producing identical
        module keys (e.g. "flashback-api").
        """
        apis: dict = {}
        for filename, content in markdowns.items():
            if source_app:
                module = source_app
            else:
                module = filename.rsplit(".", 1)[0]
                for suffix in ("_api", "_arch", "-api"):
                    module = module.removesuffix(suffix)
            h1 = _H1_RE.search(content or "")
            description = h1.group(1).strip() if h1 else filename
            # Scan for endpoints outside fenced code blocks so example/comment
            # lines (e.g. "bash ... # POST /process") aren't harvested as real
            # endpoints — this matches how a real LLM reads the document.
            scan_text = re.sub(r"```.*?```", "", content or "", flags=re.DOTALL)
            endpoints = _ENDPOINT_RE.findall(scan_text)
            module_apis = apis.setdefault(module, {})
            if endpoints:
                for method, path in endpoints:
                    module_apis[f"{method} {path}"] = {
                        "method": method,
                        "path": path,
                        "description": description,
                        "sources": [filename],  # provenance: which markdown produced this
                    }
            else:
                # No recognizable endpoints — still surface the doc
                module_apis[f"DOC {filename}"] = {
                    "method": "DOC",
                    "path": filename,
                    "description": description,
                    "sources": [filename],
                }
        return apis

    async def generate_wiki(
        self, markdowns: Dict[str, str], source_app: Optional[str] = None
    ) -> dict:
        """Generate a full structured wiki from markdown files.

        Returns {"apis": {module: {api_key: {...}}}, "metadata": {...}}.
        Provenance (source_app/source_version) is stamped by the processor,
        not requested from the model. ``source_app`` is forwarded to mock
        extraction so mock module keys match real-LLM output.
        """
        if self._mock_mode():
            return {
                "apis": self._mock_apis_from_markdowns(markdowns, source_app),
                "metadata": {},
            }

        combined = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in markdowns.items())
        logger.info(f"{type(self).__name__}: initial wiki generation ({len(combined)} chars)")
        analysis = await self._analyze(combined)
        return await self._generate_from_analysis(combined, analysis)

    # ------------------------------------------------------------------
    # Two-step chain-of-thought extraction. Step 1 reads the docs and reasons
    # about structure/contradictions; step 2 emits the final JSON grounded in
    # that analysis. Splitting the calls yields higher-quality, less
    # hallucinated output than a single read-and-write pass.
    # ------------------------------------------------------------------

    async def _analyze(self, combined_markdown: str, context: str = "") -> str:
        """Step 1: reason about the docs. Returns free-form analysis text."""
        prompt = (
            "You are analyzing API documentation. Read carefully and produce a concise "
            "structured analysis (plain text, not JSON):\n"
            "- Every API endpoint you find: HTTP method, path, one-line purpose, and the "
            "exact source filename (from the '## File: <name>' headers) it came from.\n"
            "- The module/service each endpoint belongs to.\n"
            "- Any contradictions or duplicate definitions across files.\n\n"
            f"{context}"
            f"{combined_markdown}"
        )
        return await self._generate_retry(prompt, temperature=0.2)

    async def _generate_from_analysis(self, combined_markdown: str, analysis: str) -> dict:
        """Step 2: emit final JSON grounded in the step-1 analysis."""
        prompt = (
            "Using your analysis below, generate the structured wiki JSON.\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Source documents:\n{combined_markdown}\n\n"
            "Output ONLY valid JSON, no markdown, in this exact shape:\n"
            '{"apis": {"<module>": {"<METHOD /path>": {"method": "...", "path": "...", '
            '"description": "...", "sources": ["<source filename>"]}}}, "metadata": {}}\n'
            "Every endpoint MUST include a non-empty \"sources\" list naming the markdown "
            "file(s) it was extracted from."
        )
        content = await self._generate_retry(prompt, temperature=0.3)
        return self.extract_json(content)

    async def update_wiki(
        self,
        current_apis: dict,
        changed_markdowns: Dict[str, str],
        changes,
        source_app: Optional[str] = None,
    ) -> dict:
        """Regenerate one application's API entries.

        Args:
            current_apis: the app's existing entries, {module: {api_key: {...}}}
            changed_markdowns: the app's new/modified markdown files
            changes: change summary (dict or str), included in the prompt
            source_app: pushing app identity, used as the mock module key so
                mock output matches real-LLM extraction.

        Returns {"apis": {module: {api_key: {...}}}} containing ONLY this
        app's entries — the processor merges them into the shared wiki.
        """
        if self._mock_mode():
            return {
                "apis": self._mock_apis_from_markdowns(changed_markdowns, source_app)
            }

        changed_content = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in changed_markdowns.items())
        current_summary = json.dumps(current_apis, ensure_ascii=False, indent=2)[:2000]
        logger.info(f"{type(self).__name__}: incremental update")
        context = (
            f"Current API entries for this application (summarized):\n{current_summary}\n\n"
            f"Changes: {json.dumps(changes) if isinstance(changes, dict) else changes}\n\n"
            "This is an incremental update for ONE application — only consider the "
            "new/modified files below; do not invent entries for other applications.\n\n"
        )
        analysis = await self._analyze(changed_content, context=context)
        return await self._generate_from_analysis(changed_content, analysis)

    # ------------------------------------------------------------------
    # Per-app overview (item 5) and cross-app concepts (item 2).
    # ------------------------------------------------------------------

    @staticmethod
    def _api_lines(apis: dict) -> list[str]:
        """Flatten {module: {api_key: {description}}} to 'api_key — description'."""
        lines = []
        for endpoints in (apis or {}).values():
            if not isinstance(endpoints, dict):
                continue
            for api_key, detail in endpoints.items():
                desc = detail.get("description", "") if isinstance(detail, dict) else ""
                lines.append(f"{api_key} — {desc}" if desc else api_key)
        return lines

    @staticmethod
    def _concept_token(api_key: str) -> str:
        """First meaningful path segment of a 'METHOD /a/b' key — the mock's
        deterministic concept handle (e.g. 'GET /items/{id}' -> 'items')."""
        parts = api_key.split(None, 1)
        path = parts[1] if len(parts) > 1 else parts[0]
        for seg in path.strip("/").split("/"):
            if seg and not seg.startswith("{"):
                return seg.lower()
        return "general"

    async def generate_overview(self, app: str, app_apis: dict) -> str:
        """One-paragraph synthesis of an app's surface. Mock is deterministic."""
        lines = self._api_lines(app_apis)
        if self._mock_mode():
            return f"{app}: {len(lines)} endpoint(s). " + "; ".join(lines)
        prompt = (
            f"Write a concise one-paragraph overview of the '{app}' service based on its "
            "API endpoints below. State its purpose and main capabilities. Plain text only.\n\n"
            + "\n".join(lines)
        )
        return (await self._generate_retry(prompt, temperature=0.3)).strip()

    async def generate_concepts(self, apis: dict, knowledge: dict | None = None) -> dict:
        """Cross-app concept synthesis over the WHOLE wiki.

        Returns {concept: {"description", "related": ["module::api_key" |
        "knowledge::doc_id", ...], "apps": [...]}}. Mock clusters endpoints by
        shared first path segment, then links knowledge docs that *mention* a
        concept token — so an Oracle "flashback" doc and a flashback-api
        `/recover` endpoint land on the same concept (cross-domain reasoning).
        """
        if self._mock_mode():
            concepts: dict = {}
            for module, endpoints in (apis or {}).items():
                if not isinstance(endpoints, dict):
                    continue
                for api_key, detail in endpoints.items():
                    token = self._concept_token(api_key)
                    app = detail.get("source_app", module) if isinstance(detail, dict) else module
                    c = concepts.setdefault(
                        token, {"description": f"Concept '{token}'.", "related": [], "apps": []}
                    )
                    c["related"].append(f"{module}::{api_key}")
                    if app not in c["apps"]:
                        c["apps"].append(app)
            # Link knowledge docs to any concept token they mention.
            for doc_id, entry in (knowledge or {}).items():
                if not isinstance(entry, dict):
                    continue
                text = " ".join([
                    entry.get("title", ""), entry.get("summary", ""),
                    " ".join(entry.get("topics", [])), " ".join(entry.get("key_points", [])),
                ]).lower()
                app = entry.get("source_app", "")
                for token, c in concepts.items():
                    if token in text:
                        c["related"].append(f"knowledge::{doc_id}")
                        if app and app not in c["apps"]:
                            c["apps"].append(app)
            return concepts

        catalogue = "\n".join(
            f"{module}::{api_key} — {detail.get('description', '') if isinstance(detail, dict) else ''}"
            for module, endpoints in (apis or {}).items() if isinstance(endpoints, dict)
            for api_key, detail in endpoints.items()
        )
        prompt = (
            "Identify cross-cutting concepts shared across these API endpoints (e.g. "
            "authentication, pagination, recovery). For each concept list the endpoints "
            "that implement it. Output ONLY valid JSON:\n"
            '{"<concept>": {"description": "...", "related": ["<module>::<api_key>", ...], '
            '"apps": ["<app>", ...]}}\n\n'
            f"Endpoints:\n{catalogue}"
        )
        return self.extract_json(await self._generate_retry(prompt, temperature=0.3))

    # ------------------------------------------------------------------
    # Knowledge documents — prose/reference docs (Oracle, FastAPI how-tos),
    # not API specs. This is what lets the wiki hold general knowledge the
    # Karpathy llm-wiki way, so an agent can reason across services AND domains.
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_id(source_app: str, filename: str) -> str:
        stem = filename.rsplit(".", 1)[0]
        return f"{source_app}:{stem}" if source_app else stem

    async def generate_knowledge(
        self, markdowns: Dict[str, str], source_app: Optional[str] = None
    ) -> dict:
        """Extract a structured knowledge entry per prose document.

        Returns {doc_id: {title, summary, topics: [...], key_points: [...]}}.
        Mock derives everything deterministically from the markdown (title from
        the H1, summary from the opening text, topics from headings, key_points
        from bullet/numbered lines) so retrieval still reflects real content.
        """
        if self._mock_mode():
            out: dict = {}
            for filename, content in markdowns.items():
                content = content or ""
                # Drop YAML frontmatter for the body view.
                body = content
                if body.startswith("---"):
                    end = body.find("---", 3)
                    if end != -1:
                        body = body[end + 3:]
                h1 = _H1_RE.search(content)
                title = h1.group(1).strip() if h1 else filename
                headings = re.findall(r"^#{1,6}\s+(.+)$", body, re.MULTILINE)
                points = re.findall(r"^\s*(?:[-*]|\d+\.)\s+(.+)$", body, re.MULTILINE)
                # Summary: first non-heading, non-bullet prose, generously sized
                # so keyword search over the entry finds in-body terms.
                prose = "\n".join(
                    ln for ln in body.splitlines()
                    if ln.strip() and not ln.lstrip().startswith(("#", "-", "*"))
                    and not re.match(r"^\s*\d+\.\s", ln)
                )
                out[self._doc_id(source_app, filename)] = {
                    "title": title,
                    "summary": prose.strip()[:600],
                    "topics": [h.strip() for h in headings][:20],
                    "key_points": [p.strip() for p in points][:20],
                }
            return out

        combined = "\n\n".join(f"## File: {fn}\n{c}" for fn, c in markdowns.items())
        prompt = (
            "Extract structured knowledge from these documents (prose/reference, not "
            "API specs). For each document output a concise factual summary, the main "
            "topics, and the key takeaways someone would act on.\n\n"
            "Output ONLY valid JSON keyed by a short doc id:\n"
            '{"<doc_id>": {"title": "...", "summary": "...", "topics": ["..."], '
            '"key_points": ["..."]}}\n\n'
            f"{combined}"
        )
        return self.extract_json(await self._generate_retry(prompt, temperature=0.3))
