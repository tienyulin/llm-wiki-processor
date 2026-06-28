"""Wiki-processing pipeline: extraction, app-level merge, and CAS persistence."""

import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from models.schemas import ProcessResponse
from services.embeddings import EmbeddingClient
from services.vector_sync import VectorSyncMixin
from services.llm import LLMProvider
from repository.minio_client import MinioStorage
from repository.pg_store import PGVectorStore

logger = logging.getLogger(__name__)

_WIKI_KEY = "wiki.json"  # derived aggregate (concepts/overviews + merged view), rebuilt
_APP_PREFIX = "apps/"  # per-app source of truth: apps/<app>.json (P3 — O(1) writes)
_SNAPSHOT_KEY = "markdowns_snapshot.json"
_APP_SNAPSHOT_PREFIX = "snapshots/"
_AUDIT_PREFIX = "audit/"


def _app_key(app: str) -> str:
    return f"{_APP_PREFIX}{app}.json"


_SCHEMA_VERSION = 2
_CAS_MAX_RETRIES = 5

_SYSTEM_APP = "system"

# Endpoint signature outside fenced code — used to auto-classify a push as an
# API spec vs a prose knowledge document when doc_type isn't given.
_ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+/[\w/{}.-]*")


def _looks_like_api(markdowns: dict) -> bool:
    for content in markdowns.values():
        scan = re.sub(r"```.*?```", "", content or "", flags=re.DOTALL)
        if _ENDPOINT_RE.search(scan):
            return True
    return False


_FM_LIST_RE = re.compile(r"^\[(.*)\]$")


def _parse_frontmatter(markdowns: dict) -> dict:
    """Extract `type` / `tags` from the first markdown's YAML frontmatter.

    Tiny parser for the controlled subset (scalar + inline list) — matches the
    authoring standard (docs/guides/authoring-source-docs.md); not a full YAML
    engine. Returns {} when there's no frontmatter."""
    for content in markdowns.values():
        if not content or not content.startswith("---"):
            continue
        end = content.find("\n---", 3)
        if end == -1:
            continue
        out: dict = {}
        for line in content[3:end].splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip().strip("'\"")
            m = _FM_LIST_RE.match(val)
            if m:
                out[key] = [v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip()]
            elif val:
                out[key] = val
        return out
    return {}


def _apis_from_openapi(spec: dict, source_app: str) -> dict:
    """Deterministically build wiki API entries from an OpenAPI spec — no LLM.

    Returns {module: {"METHOD /path": {method, path, description, parameters,
    sources}}}. module = source_app (matches the LLM/mock module key)."""
    module = source_app or _SYSTEM_APP
    out: dict = {module: {}}
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            if not isinstance(op, dict):
                continue
            desc = (op.get("summary") or op.get("description") or "").strip()
            params = [
                p.get("name")
                for p in op.get("parameters", [])
                if isinstance(p, dict) and p.get("name")
            ]
            out[module][f"{method.upper()} {path}"] = {
                "method": method.upper(),
                "path": path,
                "description": desc,
                "parameters": params,
                "sources": ["openapi.json"],
            }
    return out


def _sorted_api_keys(new_apis: dict) -> list[str]:
    """Sorted list of every endpoint key across all modules in new_apis."""
    return sorted(api_key for endpoints in new_apis.values() for api_key in endpoints)


def _readme_summary(markdowns: dict) -> str:
    """First prose paragraph across the markdowns (deterministic overview for the
    OpenAPI path, so it needs no LLM call)."""
    for content in markdowns.values():
        body = content or ""
        if body.startswith("---"):
            e = body.find("\n---", 3)
            if e != -1:
                body = body[e + 4 :]
        for line in body.splitlines():
            s = line.strip()
            if s and not s.startswith(("#", "-", "*", "|")) and not _ENDPOINT_RE.match(s):
                return s[:500]
    return ""


@dataclass
class _PushContext:
    """Per-push identity shared across extraction, stamping, and persistence."""

    app: str
    version: str
    kind: str
    source_app: str | None = None
    snapshot_key: str = ""
    sync_ts: datetime | None = None


@dataclass
class _Extraction:
    """Outputs of Phase-1 extraction for a single push (one kind populated)."""

    new_apis: dict = field(default_factory=dict)
    new_knowledge: dict = field(default_factory=dict)
    overview: str | None = None
    index_rows: list[dict] = field(default_factory=list)
    knowledge_rows: list[dict] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)


@dataclass
class _Frontmatter:
    """Source-doc frontmatter fields that influence extraction."""

    type: str | None = None
    tags: list = field(default_factory=list)


@dataclass
class _ExtractInputs:
    """Phase-1 extraction inputs (frozen request context for one push)."""

    markdowns: dict
    changes: dict
    app_obj: dict
    is_first_run: bool
    openapi: dict | None
    frontmatter: _Frontmatter
    source_app: str | None


@dataclass
class _RawRequest:
    """The raw process() inputs not captured by _PushContext identity."""

    markdowns: dict
    timestamp: str
    doc_type: str | None = None
    openapi: dict | None = None


def _default_wiki() -> dict:
    """Fresh canonical (v2) wiki with a creation timestamp evaluated at call time."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "apis": {},
        "knowledge": {},
        "concepts": {},
        "overviews": {},
        "metadata": {"version": "1.0", "created_at": datetime.now().isoformat()},
    }


class WikiProcessor(VectorSyncMixin):
    """Orchestrates the wiki-processing pipeline with app-level incremental updates.

    Concurrency model (multi-replica safe): the LLM call runs unlocked and
    fully concurrent; the merge+write happens in a bounded optimistic CAS loop
    using MinIO conditional writes (ETag If-Match). See
    docs/architecture/concurrency.md.
    """

    def __init__(
        self,
        storage: MinioStorage,
        llm: LLMProvider,
        embedder: EmbeddingClient | None = None,
        vector_store: PGVectorStore | None = None,
    ):
        self.storage = storage
        self.llm = llm
        # Optional vector-index layer: both None (the default) means the
        # pipeline behaves exactly as before PG existed.
        self.embedder = embedder
        self.vector_store = vector_store
        # Serializes only Phase 2 (merge + conditional write, ~ms) within this
        # process; without it an N-way in-process burst would exhaust the CAS
        # retry budget (one winner per round). Cross-replica conflicts are
        # still handled by the CAS loop itself.
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------

    def detect_changes(self, old: dict, new: dict) -> dict:
        """
        Pure function: compare old snapshot with new markdowns.
        Returns {"added": [...], "modified": [...], "deleted": [...]}.
        """
        old_files = set(old.keys())
        new_files = set(new.keys())

        added = new_files - old_files
        deleted = old_files - new_files
        modified = {f for f in old_files & new_files if old[f] != new[f]}

        return {
            "added": sorted(added),
            "modified": sorted(modified),
            "deleted": sorted(deleted),
        }

    def _normalize_wiki(self, wiki: dict) -> dict:
        """Lazy migration to the v2 schema.

        Pre-v2 wikis mixed two shapes: structured {"apis": ...} and a flat
        file map {"doc.md": "<markdown>"}. The structured part is preserved;
        legacy file-map entries are dropped (their apps repopulate on their
        next submission) — see docs/troubleshooting.md.
        """
        if wiki.get("schema_version") == _SCHEMA_VERSION:
            return wiki

        apis = wiki.get("apis")
        metadata = wiki.get("metadata")
        dropped = [k for k, v in wiki.items() if isinstance(v, str)]
        if dropped:
            logger.warning(
                "Migrating wiki to schema v%d: dropping %d legacy file-map entries: %s...",
                _SCHEMA_VERSION,
                len(dropped),
                dropped[:5],
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "apis": apis if isinstance(apis, dict) else {},
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def _stamp(self, apis: dict, source_app: str, source_version: str) -> dict:
        """Stamp provenance onto every API entry.

        The processor owns provenance — LLM output is never trusted for
        source_app/source_version.
        """
        stamped: dict = {}
        for module, endpoints in (apis or {}).items():
            if not isinstance(endpoints, dict):
                continue
            for api_key, detail in endpoints.items():
                entry = dict(detail) if isinstance(detail, dict) else {"description": str(detail)}
                entry["source_app"] = source_app
                entry["source_version"] = source_version
                stamped.setdefault(module, {})[api_key] = entry
        return stamped

    def _app_entries(self, wiki: dict, source_app: str) -> dict:
        """Extract one app's current entries: {module: {api_key: {...}}}."""
        out: dict = {}
        for module, endpoints in wiki.get("apis", {}).items():
            if not isinstance(endpoints, dict):
                continue
            selected = {
                k: v
                for k, v in endpoints.items()
                if isinstance(v, dict) and v.get("source_app") == source_app
            }
            if selected:
                out[module] = selected
        return out

    def _merge_app_entries(
        self,
        wiki: dict,
        source_app: str,
        new_apis: dict,
        overview: str | None = None,
        new_knowledge: dict | None = None,
    ) -> dict:
        """Replace one app's entries in the wiki; other apps' entries are kept.

        Returns a new wiki dict (no mutation of the input). Existing `concepts`,
        `overviews`, and the other section (`knowledge` when this push is APIs,
        and vice-versa) are carried over so a per-app ingest never clobbers
        them; this app's overview is refreshed."""
        merged_apis: dict = {}
        for module, endpoints in wiki.get("apis", {}).items():
            if not isinstance(endpoints, dict):
                continue
            kept = {
                k: v
                for k, v in endpoints.items()
                if not (isinstance(v, dict) and v.get("source_app") == source_app)
            }
            if kept:
                merged_apis[module] = kept
        for module, endpoints in new_apis.items():
            merged_apis.setdefault(module, {}).update(endpoints)

        # Knowledge: drop this app's old docs, add the new ones; keep other apps'.
        merged_knowledge = {
            doc_id: entry
            for doc_id, entry in wiki.get("knowledge", {}).items()
            if not (isinstance(entry, dict) and entry.get("source_app") == source_app)
        }
        merged_knowledge.update(new_knowledge or {})

        overviews = dict(wiki.get("overviews", {}))
        if overview is not None:
            overviews[source_app] = {"text": overview, "updated_at": datetime.now().isoformat()}

        return {
            "schema_version": _SCHEMA_VERSION,
            "apis": merged_apis,
            "knowledge": merged_knowledge,
            "concepts": wiki.get("concepts", {}),
            "overviews": overviews,
            "metadata": {**wiki.get("metadata", {}), "updated_at": datetime.now().isoformat()},
        }

    def _build_app_object(self, app_obj: dict, ctx: _PushContext, extraction: _Extraction) -> dict:
        """This app's object: replace the section this push owns (apis OR
        knowledge), keep the other section + overview. The app pushes its full
        markdown set each time, so new_apis/new_knowledge is the complete set."""
        if ctx.kind == "knowledge":
            apis = app_obj.get("apis", {})
            knowledge = extraction.new_knowledge
            ov = app_obj.get("overview")
        else:
            apis = extraction.new_apis
            knowledge = app_obj.get("knowledge", {})
            ov = extraction.overview
        return {
            "schema_version": _SCHEMA_VERSION,
            "source_app": ctx.app,
            "source_version": ctx.version,
            "apis": apis,
            "knowledge": knowledge,
            "overview": ov,
            "updated_at": datetime.now().isoformat(),
        }

    def _stamp_knowledge(
        self,
        knowledge: dict,
        ctx: _PushContext,
        markdowns: dict,
        doc_type: str | None = None,
        tags: list | None = None,
    ) -> dict:
        """Stamp provenance onto each knowledge entry (processor owns provenance).
        `sources` lists the markdown files this push carried. doc_type/tags come
        from the source-doc frontmatter (authoring standard)."""
        app = ctx.app
        sources = sorted(markdowns.keys())
        stamped: dict = {}
        for doc_id, entry in (knowledge or {}).items():
            e = dict(entry) if isinstance(entry, dict) else {"summary": str(entry)}
            e.setdefault("topics", [])
            e.setdefault("key_points", [])
            e["source_app"] = app
            e["source_version"] = ctx.version
            e["sources"] = sources
            if doc_type:
                e["doc_type"] = doc_type
            if tags:
                e["tags"] = tags
            e["updated_at"] = datetime.now().isoformat()
            # Namespace the key by app so real-LLM output ("o.md") matches the
            # mock's "<app>:<stem>" — consistent, app-scoped, collision-free.
            stem = str(doc_id).rsplit("/", 1)[-1].rsplit(".", 1)[0]
            key = doc_id if str(doc_id).startswith(f"{app}:") else f"{app}:{stem}"
            stamped[key] = e
        return stamped

    # Vector-index sync (_entry_rows / _embed_rows / _sync_vector_index /
    # _knowledge_rows / _sync_knowledge_index / reindex) lives in
    # VectorSyncMixin — the optional, best-effort PG layer kept out of the core.

    async def aggregate_apps(self) -> dict:
        """Merge all per-app objects (apps/<app>.json) into one wiki dict.

        The whole-wiki view consumers (concepts, mcp fallback) need; built by
        reading each small app object, not by keeping one giant blob hot."""
        merged_apis: dict = {}
        merged_knowledge: dict = {}
        overviews: dict = {}
        for key in await self.storage.alist_files(_APP_PREFIX):
            if not key.endswith(".json"):
                continue
            obj = await self.storage.aget_json(key) or {}
            for module, endpoints in (obj.get("apis") or {}).items():
                merged_apis.setdefault(module, {}).update(endpoints)
            merged_knowledge.update(obj.get("knowledge") or {})
            app = obj.get("source_app")
            if app and obj.get("overview") is not None:
                overviews[app] = {"text": obj["overview"], "updated_at": obj.get("updated_at", "")}
        return {
            "schema_version": _SCHEMA_VERSION,
            "apis": merged_apis,
            "knowledge": merged_knowledge,
            "overviews": overviews,
        }

    async def rebuild_concepts(self) -> dict:
        """Rebuild the derived aggregate wiki.json from the per-app objects:
        merge all apps, synthesize concepts, write wiki.json (the view mcp reads
        for concepts/overviews + fallback). Run after a batch of pushes or on a
        schedule — the per-push hot path no longer touches this aggregate."""
        agg = await self.aggregate_apps()
        concepts = await self.llm.generate_concepts(agg["apis"], knowledge=agg["knowledge"])
        # Augment substring links with semantic ones (embedding proximity) so a
        # synonym-phrased knowledge doc still links to its concept.
        await self._link_knowledge_semantically(concepts)
        wiki = {
            **agg,
            "concepts": concepts,
            "metadata": {"version": "1.0", "updated_at": datetime.now().isoformat()},
        }
        await self.storage.aput_json(_WIKI_KEY, wiki)
        await self._notify_cache_invalidation(None)
        return {
            "concepts": len(concepts),
            "apps": len(agg["overviews"]),
            "endpoints": sum(len(e) for e in agg["apis"].values()),
        }

    async def _link_knowledge_semantically(self, concepts: dict) -> None:
        """Add knowledge↔concept links by embedding proximity (in place).

        For each knowledge doc, the nearest API entries (cosine >= threshold)
        contribute the doc to those endpoints' concepts. No-op when the vector
        index or embeddings are unavailable (links stay substring-only)."""
        if self.vector_store is None:
            return
        # Floor measured on-corpus: a synonym-only doc that *should* link scored
        # 0.656 to the recovery endpoint while an unrelated how-to scored 0.599 —
        # 0.63 sits in that gap (and in the 0.60–0.64 range reported in the entity-
        # linking literature). Substring links are kept too, so this only *adds*
        # recall; tune via CONCEPT_LINK_MIN_COSINE.
        threshold = float(os.getenv("CONCEPT_LINK_MIN_COSINE", "0.63"))
        margin = float(os.getenv("CONCEPT_LINK_MARGIN", "0.05"))
        try:
            links = await self.vector_store.knowledge_api_links(threshold=threshold, margin=margin)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Semantic linking is best-effort enrichment; never fail the rebuild.
            logger.warning("Semantic concept linking skipped: %s", e)
            return
        for doc_id, api_links in links.items():
            ref = f"knowledge::{doc_id}"
            for _module, api_key, _score in api_links:
                # _concept_token is a same-package helper on the provider base.
                token = self.llm._concept_token(api_key)  # pylint: disable=protected-access
                c = concepts.get(token)
                if c is None:
                    continue
                if ref not in c["related"]:
                    c["related"].append(ref)
                app = doc_id.split(":", 1)[0]
                if app and app not in c["apps"]:
                    c["apps"].append(app)

    async def recompile(self) -> dict:
        """Re-run extraction over stored per-app snapshots without re-ingesting (item 6).

        Use after an extraction/prompt change to refresh entries from the
        markdown already on record. Each app is reprocessed via the normal
        process() path (CAS-safe, re-embeds, refreshes its overview)."""
        keys = [
            k for k in await self.storage.alist_files(_APP_SNAPSHOT_PREFIX) if k.endswith(".json")
        ]
        apps = []
        for key in keys:
            app = key[len(_APP_SNAPSHOT_PREFIX) : -len(".json")]
            markdowns = await self.storage.aget_json(key) or {}
            if not markdowns:
                continue
            # Force a full re-extract: drop the snapshot so detect_changes sees
            # every file as added.
            await self.storage.aput_json(key, {})
            await self.process(
                markdowns, datetime.now().isoformat(), source_app=app, source_version="recompiled"
            )
            apps.append(app)
        return {"recompiled_apps": apps, "count": len(apps)}

    # ------------------------------------------------------------------
    # Side channels (audit, cache invalidation)
    # ------------------------------------------------------------------

    async def _log_audit(self, source_app: str, files_count: int, status: str, files_updated: list):
        """Write one audit entry as its own object (append-only, no contention).

        Keys sort chronologically: audit/{iso-ts}-{uuid8}.json."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source_app": source_app,
            "files_count": files_count,
            "status": status,
            "files_updated": files_updated,
        }
        key = f"{_AUDIT_PREFIX}{entry['timestamp']}-{uuid.uuid4().hex[:8]}.json"
        try:
            await self.storage.aput_json(key, entry)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Audit is a side channel; a failed write must not break the push.
            logger.error("Audit write failed (%s): %s", key, e)

    async def _notify_cache_invalidation(self, source_app: str | None = None):
        """Tell mcp-server to drop its cached wiki after a successful update.

        Best effort: a missing MCP_SERVER_URL or an unreachable server only
        logs a warning — wiki persistence already succeeded.
        """
        url = os.getenv("MCP_SERVER_URL")
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"{url.rstrip('/')}/cache/invalidate",
                    json={"source_app": source_app},
                )
            logger.info("mcp-server cache invalidated (source_app=%s)", source_app)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Cache invalidation is best-effort; persistence already succeeded.
            logger.warning("mcp-server cache invalidation failed: %s", e)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_kind(
        doc_type: str | None, openapi: dict | None, fm_type: str | None, markdowns: dict
    ) -> str:
        """Pick the push kind — always "api" or "knowledge".

        A *declared* type is authoritative: only ``api`` (or an attached OpenAPI
        spec) is an API push; every other declared type is knowledge. The
        endpoint heuristic is a last resort, used only when nothing declares a
        type, so a compliant doc is never reclassified by stray endpoint-shaped
        prose (e.g. a runbook that mentions ``POST /charge``). Type matching is
        case-insensitive. Precedence: explicit doc_type > attached openapi >
        frontmatter type > content heuristic."""
        if doc_type:
            return "api" if doc_type.strip().lower() == "api" else "knowledge"
        if openapi:
            return "api"
        if fm_type:
            return "api" if fm_type.strip().lower() == "api" else "knowledge"
        return "api" if _looks_like_api(markdowns) else "knowledge"

    async def _extract_knowledge(self, ctx: _PushContext, inputs: _ExtractInputs) -> _Extraction:
        """Phase-1 knowledge extraction (LLM prose -> stamped knowledge entries)."""
        logger.info("Knowledge ingest for %s", ctx.app)
        generated_k = await self.llm.generate_knowledge(inputs.markdowns, source_app=ctx.app)
        new_knowledge = self._stamp_knowledge(
            generated_k,
            ctx,
            inputs.markdowns,
            doc_type=inputs.frontmatter.type,
            tags=inputs.frontmatter.tags,
        )
        result = _Extraction(
            new_knowledge=new_knowledge,
            files_updated=sorted(new_knowledge.keys()),
        )
        # Embed for the hybrid (vector+keyword) knowledge index. Slow part
        # (embedding) runs here in Phase 1, before the write lock.
        if self.vector_store is not None:
            result.knowledge_rows = self._knowledge_rows(new_knowledge)
            await self._embed_rows(ctx.app, result.knowledge_rows)
        return result

    async def _extract_openapi(self, ctx: _PushContext, inputs: _ExtractInputs) -> _Extraction:
        """Phase-1 deterministic ingest from an OpenAPI spec — no LLM (no 429s)."""
        openapi = inputs.openapi or {}
        logger.info("OpenAPI ingest for %s (%d paths)", ctx.app, len(openapi.get("paths", {})))
        new_apis = self._stamp(_apis_from_openapi(openapi, ctx.app), ctx.app, ctx.version)
        self._apply_tags(new_apis, inputs.frontmatter.tags)
        result = _Extraction(
            new_apis=new_apis,
            files_updated=_sorted_api_keys(new_apis),
            # Deterministic overview from the README's first paragraph (no LLM).
            overview=_readme_summary(inputs.markdowns) or None,
        )
        if self.vector_store is not None:
            result.index_rows = self._entry_rows(new_apis)
            await self._embed_rows(ctx.app, result.index_rows)
        return result

    async def _extract_api(self, ctx: _PushContext, inputs: _ExtractInputs) -> _Extraction:
        """Phase-1 LLM API extraction (first-run full generation or incremental)."""
        if inputs.is_first_run:
            logger.info("First run detected - generating complete wiki")
            generated = await self.llm.generate_wiki(inputs.markdowns, source_app=inputs.source_app)
        else:
            logger.info("App-level update for %s", ctx.app)
            generated = await self.llm.update_wiki(
                current_apis=inputs.app_obj.get("apis", {}),
                changed_markdowns=inputs.markdowns,
                changes=inputs.changes,
                source_app=inputs.source_app,
            )

        new_apis = self._stamp(generated.get("apis", {}), ctx.app, ctx.version)
        # Keep module == source_app (the per-app model; the OpenAPI path and the
        # mock LLM already do this). A real LLM may invent its own module name
        # (e.g. "payments" for source_app "payments-svc"), which splits an app's
        # endpoints under an unexpected key and breaks grouping/filtering by app.
        # Collapse them under this app.
        if inputs.source_app:
            collapsed: dict = {}
            for endpoints in new_apis.values():
                collapsed.update(endpoints)
            new_apis = {ctx.app: collapsed}
        self._apply_tags(new_apis, inputs.frontmatter.tags)
        result = _Extraction(
            new_apis=new_apis,
            files_updated=_sorted_api_keys(new_apis),
            # Per-app overview (item 5): scoped to this app, so it folds into the
            # same CAS write — no cross-app contention, no extra round trip.
            overview=await self.llm.generate_overview(ctx.app, new_apis),
        )
        # Still Phase 1 (no lock): embedding is the slow part of index sync, and
        # new_apis is loop-invariant across CAS retries.
        if self.vector_store is not None:
            result.index_rows = self._entry_rows(new_apis)
            await self._embed_rows(ctx.app, result.index_rows)
        return result

    async def _extract(self, ctx: _PushContext, inputs: _ExtractInputs) -> _Extraction:
        """Dispatch Phase-1 extraction by push kind."""
        if ctx.kind == "knowledge":
            return await self._extract_knowledge(ctx, inputs)
        if inputs.openapi:
            return await self._extract_openapi(ctx, inputs)
        return await self._extract_api(ctx, inputs)

    @staticmethod
    def _apply_tags(new_apis: dict, fm_tags: list) -> None:
        """Stamp frontmatter tags onto every endpoint detail (in place)."""
        if not fm_tags:
            return
        for endpoints in new_apis.values():
            for detail in endpoints.values():
                detail["tags"] = fm_tags

    async def _cas_write_app(
        self, ctx: _PushContext, app_obj: dict, etag: str | None, extraction: _Extraction
    ) -> None:
        """Phase-2 optimistic CAS write of this app's object on its own key.

        Each app has a distinct key, so concurrent pushes from different apps
        never contend (no global lock). This loop only guards concurrent pushes
        of the SAME app (rare). The write is O(this app), not O(all apps)."""
        for attempt in range(_CAS_MAX_RETRIES):
            merged = self._build_app_object(app_obj, ctx, extraction)
            if etag is None:
                ok = await self.storage.aput_json_if_absent(_app_key(ctx.app), merged)
            else:
                ok = await self.storage.aput_json_if_match(_app_key(ctx.app), merged, etag)
            if ok:
                return
            logger.info("CAS conflict for %s (attempt %d), retrying", ctx.app, attempt + 1)
            await asyncio.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
            reread, etag = await self.storage.aget_json_with_etag(_app_key(ctx.app))
            app_obj = reread or {"apis": {}, "knowledge": {}}
        raise RuntimeError(f"App write failed after {_CAS_MAX_RETRIES} CAS attempts for {ctx.app}")

    @staticmethod
    def _build_snapshot(markdowns: dict, openapi: dict | None) -> dict:
        """Change-detection snapshot: the markdowns plus a canonical openapi blob
        so an openapi-only change (README untouched) still counts as a change."""
        snapshot = dict(markdowns)
        if openapi is not None:
            snapshot["__openapi__.json"] = json.dumps(openapi, sort_keys=True, ensure_ascii=False)
        return snapshot

    async def _persist_and_sync(
        self, ctx: _PushContext, snapshot: dict, markdowns: dict, extraction: _Extraction
    ) -> None:
        """Phase-3: persist the snapshot, audit, best-effort PG sync, invalidate.

        PG sync must precede cache invalidation: when mcp-server drops its
        fallback cache, PG already serves the fresh entries. Only the path
        matching this push's kind runs — api and knowledge share the per-app
        sync guard, so running both for one timestamp would let the empty one
        claim the guard and block the real sync."""
        assert ctx.sync_ts is not None  # set by process() before any sync
        await self.storage.aput_json(ctx.snapshot_key, snapshot)
        await self._log_audit(ctx.app, len(markdowns), "success", extraction.files_updated)
        if ctx.kind == "knowledge":
            await self._sync_knowledge_index(
                ctx.app, ctx.version, extraction.knowledge_rows, ctx.sync_ts
            )
        else:
            await self._sync_vector_index(ctx.app, ctx.version, extraction.index_rows, ctx.sync_ts)
        await self._notify_cache_invalidation(ctx.source_app)

    @staticmethod
    def _success_response(
        message: str,
        changes: dict,
        source_app: str | None,
        files_updated: list,
        start_time: float,
    ) -> ProcessResponse:
        """Build a success ProcessResponse with the elapsed processing time."""
        return ProcessResponse(
            status="success",
            message=message,
            wiki_url="minio://wiki-data/wiki.json",
            changes_summary=changes,
            timestamp=datetime.now().isoformat(),
            source_app=source_app,
            files_updated=files_updated,
            validation_errors=[],
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

    async def _run_pipeline(
        self, ctx: _PushContext, raw: _RawRequest, start_time: float
    ) -> ProcessResponse:
        """Happy-path pipeline (read -> extract -> CAS write -> persist/sync).

        Raises on failure; process() wraps this in the top-level error guard."""
        markdowns, source_app = raw.markdowns, ctx.source_app

        # ---- Phase 1: read this app's object (P3: O(1), per-app key) ----
        app_obj, etag = await self.storage.aget_json_with_etag(_app_key(ctx.app))
        is_first_run = app_obj is None
        app_obj = app_obj or {"apis": {}, "knowledge": {}}

        old_snapshot = await self.storage.aget_json(ctx.snapshot_key) or {}
        snapshot = self._build_snapshot(markdowns, raw.openapi)
        changes = self.detect_changes(old_snapshot, snapshot)

        if not any(changes.values()):
            logger.info("No content changes for %s, skipping LLM call", ctx.app)
            return self._success_response(
                "No changes detected, wiki unchanged", changes, source_app, [], start_time
            )

        # Source-doc metadata (authoring standard): frontmatter type/tags.
        fm = _parse_frontmatter(markdowns)
        frontmatter = _Frontmatter(type=fm.get("type"), tags=(fm.get("tags") or []))
        ctx.kind = self._resolve_kind(raw.doc_type, raw.openapi, frontmatter.type, markdowns)

        # ---- Phase 1: extraction (no lock, fully concurrent) ----
        extraction = await self._extract(
            ctx,
            _ExtractInputs(
                markdowns=markdowns,
                changes=changes,
                app_obj=app_obj,
                is_first_run=is_first_run,
                openapi=raw.openapi,
                frontmatter=frontmatter,
                source_app=source_app,
            ),
        )

        # ---- Phase 2: write this app's object (CAS on its own key) ----
        await self._cas_write_app(ctx, app_obj, etag, extraction)

        # ---- Phase 3: persist snapshot, audit, best-effort PG sync ----
        await self._persist_and_sync(ctx, snapshot, markdowns, extraction)

        logger.info(
            "Processing complete for %s in %dms",
            raw.timestamp,
            int((time.time() - start_time) * 1000),
        )
        return self._success_response(
            f"Wiki {'generated' if is_first_run else 'updated'} successfully",
            changes,
            source_app,
            extraction.files_updated,
            start_time,
        )

    # The parameter list mirrors the ProcessRequest HTTP schema 1:1; bundling
    # these named request fields into one object would only obscure the public
    # boundary. Keyword-only keeps call sites self-documenting.
    # pylint: disable=too-many-arguments
    async def process(
        self,
        markdowns: dict,
        timestamp: str,
        *,
        source_app: str | None = None,
        source_version: str | None = None,
        doc_type: str | None = None,
        openapi: dict | None = None,
    ) -> ProcessResponse:
        """
        Two-phase pipeline:
        1. (concurrent) Read wiki, call LLM to produce this app's entries
           (API endpoints, or knowledge entries for prose docs).
        2. (CAS loop) Re-read + merge + conditional write until it sticks.

        doc_type: "api" | "knowledge"; when None, auto-detected (endpoints
        present -> api, else knowledge). The parameter list mirrors the
        ProcessRequest HTTP schema; the work is delegated to _run_pipeline.
        """
        start_time = time.time()
        app = source_app or _SYSTEM_APP
        ctx = _PushContext(
            app=app,
            version=source_version or "unknown",
            kind="",  # resolved later, once frontmatter is parsed
            source_app=source_app,
            snapshot_key=(f"{_APP_SNAPSHOT_PREFIX}{app}.json" if source_app else _SNAPSHOT_KEY),
            # PG app_sync guard timestamp: request start, so an older request
            # that loses the CAS race can never clobber a newer sync.
            sync_ts=datetime.now(timezone.utc),
        )
        raw = _RawRequest(
            markdowns=markdowns, timestamp=timestamp, doc_type=doc_type, openapi=openapi
        )
        try:
            return await self._run_pipeline(ctx, raw, start_time)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Top-level guard: convert any pipeline error into a failed response.
            error_msg = f"Error processing wiki: {str(e)}"
            logger.error(error_msg)
            await self._log_audit(app, len(markdowns), "failed", [])
            return ProcessResponse(
                status="failed",
                message=error_msg,
                timestamp=datetime.now().isoformat(),
                source_app=source_app,
                validation_errors=[{"error": str(e)}],
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
