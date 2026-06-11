import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime

import httpx

from models.schemas import ProcessResponse
from services.llm import LLMProvider
from storage.minio_client import MinioStorage

logger = logging.getLogger(__name__)

_WIKI_KEY = "wiki.json"
_SNAPSHOT_KEY = "markdowns_snapshot.json"
_AUDIT_LOG_KEY = "wiki-audit-log.jsonl"
_METADATA_KEY = "wiki-metadata.json"
def _default_wiki() -> dict:
    """Fresh default wiki with a creation timestamp evaluated at call time."""
    return {
        "apis": {},
        "metadata": {"version": "1.0", "created_at": datetime.now().isoformat()},
    }


class WikiProcessor:
    """Orchestrates the full wiki-processing pipeline with app-level incremental updates."""

    def __init__(self, storage: MinioStorage, llm: LLMProvider):
        self.storage = storage
        self.llm = llm
        # Serializes the read-modify-write pipeline in process(). All apps share
        # a single wiki.json, so concurrent updates would lose writes across the
        # awaited LLM call. Process-local only — multi-replica deployments need
        # conditional writes (see docs/architecture/concurrency.md).
        self._lock = asyncio.Lock()

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

    def _extract_source_app(self, content: str) -> str:
        """Extract source_app from YAML frontmatter."""
        if not content.startswith("---"):
            return "unknown"

        end_idx = content.find("---", 3)
        if end_idx == -1:
            return "unknown"

        frontmatter = content[3:end_idx]
        match = re.search(r'source_app:\s*["\']*(\w+[-\w]*)', frontmatter)
        return match.group(1) if match else "unknown"

    def _get_app_files(self, wiki: dict, source_app: str) -> dict[str, str]:
        """Get all files in wiki contributed by a specific source app."""
        app_files = {}
        for path, content in wiki.items():
            if isinstance(content, str) and self._extract_source_app(content) == source_app:
                app_files[path] = content
        return app_files

    async def _update_wiki_for_app(
        self,
        current_wiki: dict,
        source_app: str,
        source_version: str,
        markdowns: dict,
    ) -> dict[str, str]:
        """
        App-level incremental update: only regenerate files for this source_app.
        Preserve files from other apps.
        """
        logger.info(f"Performing app-level update for {source_app}")

        # Identify old files from this app
        old_app_files = self._get_app_files(current_wiki, source_app)

        # Call LLM to process only this app's markdown
        app_wiki_files = await self.llm.update_wiki(
            current_files=old_app_files,
            changed_markdowns=markdowns,
            changes=f"Updated {source_app} documentation (version: {source_version})",
        )

        # Merge: keep files from other apps (and non-file entries such as
        # "apis"/"metadata" dicts), update files from this app
        merged_wiki = {
            **{path: content for path, content in current_wiki.items()
               if not (isinstance(content, str)
                       and self._extract_source_app(content) == source_app)},
            **app_wiki_files,
        }

        return merged_wiki

    def _log_audit(self, source_app: str, files_count: int, status: str, files_updated: list):
        """Append entry to audit log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_app": source_app,
            "files_count": files_count,
            "status": status,
            "files_updated": files_updated,
        }
        # Append to NDJSON audit log
        existing = self.storage.get_file(_AUDIT_LOG_KEY) or ""
        new_content = existing.rstrip() + "\n" + json.dumps(log_entry) if existing else json.dumps(log_entry)
        self.storage.put_file(_AUDIT_LOG_KEY, new_content)

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

    async def process(
        self,
        markdowns: dict,
        timestamp: str,
        source_app: str = None,
        source_version: str = None,
    ) -> ProcessResponse:
        """
        Full pipeline supporting app-level updates:
        1. Detect if first run or app-level update.
        2. Call LLM (full wiki or app-specific).
        3. Save updated wiki + snapshot.
        4. Log to audit trail.
        """
        start_time = time.time()
        processing_time_ms = 0
        validation_errors = []
        files_updated = []

        # The snapshot/wiki read and write are separated by an awaited LLM
        # call; without the lock, concurrent requests overwrite each other.
        async with self._lock:
            try:
                # Step 1: retrieve previous snapshot and wiki
                old_snapshot = self.storage.get_json(_SNAPSHOT_KEY) or {}
                current_wiki = self.storage.get_json(_WIKI_KEY) or _default_wiki()
                is_first_run = len(old_snapshot) == 0

                # Step 2: detect changes
                if is_first_run:
                    logger.info("First run detected - generating complete wiki")
                    changes = {"added": sorted(markdowns.keys()), "modified": [], "deleted": []}
                    wiki = await self.llm.generate_wiki(markdowns)
                    files_updated = list(wiki.keys())
                elif source_app:
                    # App-level incremental update
                    logger.info(f"App-level update for {source_app}")
                    changes = {"app": source_app, "version": source_version}
                    wiki = await self._update_wiki_for_app(
                        current_wiki=current_wiki,
                        source_app=source_app,
                        source_version=source_version or "unknown",
                        markdowns=markdowns,
                    )
                    files_updated = [path for path in wiki.keys()
                                     if isinstance(wiki[path], str)
                                     and self._extract_source_app(wiki[path]) == source_app]
                else:
                    # Full incremental update
                    changes = self.detect_changes(old_snapshot, markdowns)
                    logger.info(f"Changes detected: {changes}")

                    changed_files = set(changes["added"]) | set(changes["modified"])
                    changed_markdowns = {f: markdowns[f] for f in changed_files}

                    if not changed_markdowns:
                        logger.info("No content changes, skipping LLM call")
                        wiki = current_wiki
                    else:
                        wiki = await self.llm.update_wiki(current_wiki, changed_markdowns, changes)

                    files_updated = list(changed_files)

                # Step 3: persist
                wiki.setdefault("metadata", {})["updated_at"] = datetime.now().isoformat()
                self.storage.put_json(_WIKI_KEY, wiki)
                self.storage.put_json(_SNAPSHOT_KEY, markdowns)

                # Step 4: log audit + notify mcp-server cache
                self._log_audit(source_app or "system", len(markdowns), "success", files_updated)
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
                    validation_errors=validation_errors,
                    processing_time_ms=processing_time_ms,
                )

            except Exception as e:
                processing_time_ms = int((time.time() - start_time) * 1000)
                error_msg = f"Error processing wiki: {str(e)}"
                logger.error(error_msg)
                self._log_audit(source_app or "system", len(markdowns), "failed", [])

                return ProcessResponse(
                    status="failed",
                    message=error_msg,
                    timestamp=datetime.now().isoformat(),
                    source_app=source_app,
                    validation_errors=[{"error": str(e)}],
                    processing_time_ms=processing_time_ms,
                )
