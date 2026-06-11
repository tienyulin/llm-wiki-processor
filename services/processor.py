import asyncio
import logging
import os
import random
import time
import uuid
from datetime import datetime

import httpx

from models.schemas import ProcessResponse
from services.llm import LLMProvider
from storage.minio_client import MinioStorage

logger = logging.getLogger(__name__)

_WIKI_KEY = "wiki.json"
_SNAPSHOT_KEY = "markdowns_snapshot.json"
_APP_SNAPSHOT_PREFIX = "snapshots/"
_AUDIT_PREFIX = "audit/"
_SCHEMA_VERSION = 2
_CAS_MAX_RETRIES = 5

_SYSTEM_APP = "system"


def _default_wiki() -> dict:
    """Fresh canonical (v2) wiki with a creation timestamp evaluated at call time."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "apis": {},
        "metadata": {"version": "1.0", "created_at": datetime.now().isoformat()},
    }


class WikiProcessor:
    """Orchestrates the wiki-processing pipeline with app-level incremental updates.

    Concurrency model (multi-replica safe): the LLM call runs unlocked and
    fully concurrent; the merge+write happens in a bounded optimistic CAS loop
    using MinIO conditional writes (ETag If-Match). See
    docs/architecture/concurrency.md.
    """

    def __init__(self, storage: MinioStorage, llm: LLMProvider):
        self.storage = storage
        self.llm = llm
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
                f"Migrating wiki to schema v{_SCHEMA_VERSION}: dropping "
                f"{len(dropped)} legacy file-map entries: {dropped[:5]}..."
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
                k: v for k, v in endpoints.items()
                if isinstance(v, dict) and v.get("source_app") == source_app
            }
            if selected:
                out[module] = selected
        return out

    def _merge_app_entries(self, wiki: dict, source_app: str, new_apis: dict) -> dict:
        """Replace one app's entries in the wiki; other apps' entries are kept.

        Returns a new wiki dict (no mutation of the input)."""
        merged_apis: dict = {}
        for module, endpoints in wiki.get("apis", {}).items():
            if not isinstance(endpoints, dict):
                continue
            kept = {
                k: v for k, v in endpoints.items()
                if not (isinstance(v, dict) and v.get("source_app") == source_app)
            }
            if kept:
                merged_apis[module] = kept
        for module, endpoints in new_apis.items():
            merged_apis.setdefault(module, {}).update(endpoints)

        return {
            "schema_version": _SCHEMA_VERSION,
            "apis": merged_apis,
            "metadata": {**wiki.get("metadata", {}), "updated_at": datetime.now().isoformat()},
        }

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
        except Exception as e:
            logger.error(f"Audit write failed ({key}): {e}")

    async def _notify_cache_invalidation(self, source_app: str = None):
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
            logger.info(f"mcp-server cache invalidated (source_app={source_app})")
        except Exception as e:
            logger.warning(f"mcp-server cache invalidation failed: {e}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def process(
        self,
        markdowns: dict,
        timestamp: str,
        source_app: str = None,
        source_version: str = None,
    ) -> ProcessResponse:
        """
        Two-phase pipeline:
        1. (concurrent) Read wiki, call LLM to produce this app's API entries.
        2. (CAS loop) Re-read + merge + conditional write until it sticks.
        """
        start_time = time.time()
        app = source_app or _SYSTEM_APP
        version = source_version or "unknown"
        snapshot_key = (
            f"{_APP_SNAPSHOT_PREFIX}{app}.json" if source_app else _SNAPSHOT_KEY
        )

        try:
            # ---- Phase 1: read + LLM (no lock, fully concurrent) ----
            raw, etag = await self.storage.aget_json_with_etag(_WIKI_KEY)
            is_first_run = raw is None
            wiki = _default_wiki() if is_first_run else self._normalize_wiki(raw)

            old_snapshot = await self.storage.aget_json(snapshot_key) or {}
            changes = self.detect_changes(old_snapshot, markdowns)

            if not any(changes.values()):
                logger.info(f"No content changes for {app}, skipping LLM call")
                processing_time_ms = int((time.time() - start_time) * 1000)
                return ProcessResponse(
                    status="success",
                    message="No changes detected, wiki unchanged",
                    wiki_url="minio://wiki-data/wiki.json",
                    changes_summary=changes,
                    timestamp=datetime.now().isoformat(),
                    source_app=source_app,
                    files_updated=[],
                    processing_time_ms=processing_time_ms,
                )

            if is_first_run:
                logger.info("First run detected - generating complete wiki")
                generated = await self.llm.generate_wiki(markdowns)
            else:
                logger.info(f"App-level update for {app}")
                current_entries = self._app_entries(wiki, app)
                generated = await self.llm.update_wiki(
                    current_apis=current_entries,
                    changed_markdowns=markdowns,
                    changes=changes,
                )

            new_apis = self._stamp(generated.get("apis", {}), app, version)
            files_updated = sorted(
                api_key for endpoints in new_apis.values() for api_key in endpoints
            )

            # ---- Phase 2: merge + conditional write (bounded CAS loop) ----
            async with self._write_lock:
                for attempt in range(_CAS_MAX_RETRIES):
                    merged = self._merge_app_entries(wiki, app, new_apis)
                    if etag is None:
                        ok = await self.storage.aput_json_if_absent(_WIKI_KEY, merged)
                    else:
                        ok = await self.storage.aput_json_if_match(_WIKI_KEY, merged, etag)
                    if ok:
                        break
                    logger.info(f"CAS conflict for {app} (attempt {attempt + 1}), retrying")
                    await asyncio.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
                    raw, etag = await self.storage.aget_json_with_etag(_WIKI_KEY)
                    wiki = self._normalize_wiki(raw) if raw is not None else _default_wiki()
                else:
                    raise RuntimeError(
                        f"Wiki write failed after {_CAS_MAX_RETRIES} CAS attempts for {app}"
                    )

            await self.storage.aput_json(snapshot_key, markdowns)
            await self._log_audit(app, len(markdowns), "success", files_updated)
            await self._notify_cache_invalidation(source_app)

            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Processing complete for {timestamp} in {processing_time_ms}ms")

            return ProcessResponse(
                status="success",
                message=f"Wiki {'generated' if is_first_run else 'updated'} successfully",
                wiki_url="minio://wiki-data/wiki.json",
                changes_summary=changes,
                timestamp=datetime.now().isoformat(),
                source_app=source_app,
                files_updated=files_updated,
                validation_errors=[],
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Error processing wiki: {str(e)}"
            logger.error(error_msg)
            await self._log_audit(app, len(markdowns), "failed", [])

            return ProcessResponse(
                status="failed",
                message=error_msg,
                timestamp=datetime.now().isoformat(),
                source_app=source_app,
                validation_errors=[{"error": str(e)}],
                processing_time_ms=processing_time_ms,
            )
