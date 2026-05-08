import logging
from datetime import datetime

from models.schemas import ProcessResponse
from services.llm import MinimaxClient
from storage.minio_client import MinioStorage

logger = logging.getLogger(__name__)

_WIKI_KEY = "wiki.json"
_SNAPSHOT_KEY = "markdowns_snapshot.json"
_DEFAULT_WIKI = {
    "apis": {},
    "metadata": {"version": "1.0", "created_at": datetime.now().isoformat()},
}


class WikiProcessor:
    """Orchestrates the full wiki-processing pipeline."""

    def __init__(self, storage: MinioStorage, llm: MinimaxClient):
        self.storage = storage
        self.llm = llm

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

    async def process(self, markdowns: dict, timestamp: str) -> ProcessResponse:
        """
        Full pipeline:
        1. Get old snapshot from storage.
        2. Detect changes.
        3. Call LLM (initial or incremental).
        4. Save updated wiki + new snapshot.
        """
        # Step 1: retrieve previous snapshot
        old_snapshot = self.storage.get_json(_SNAPSHOT_KEY) or {}
        is_first_run = len(old_snapshot) == 0

        # Step 2: detect changes
        if is_first_run:
            logger.info("First run detected - generating complete wiki")
            changes = {"added": sorted(markdowns.keys()), "modified": [], "deleted": []}
        else:
            changes = self.detect_changes(old_snapshot, markdowns)
            logger.info(f"Changes detected: {changes}")

        # Step 3: call LLM
        if is_first_run:
            wiki = await self.llm.generate_wiki(markdowns)
        else:
            changed_files = set(changes["added"]) | set(changes["modified"])
            changed_markdowns = {f: markdowns[f] for f in changed_files}

            if not changed_markdowns:
                logger.info("No content changes, skipping LLM call")
                wiki = self.storage.get_json(_WIKI_KEY) or dict(_DEFAULT_WIKI)
            else:
                current_wiki = self.storage.get_json(_WIKI_KEY) or dict(_DEFAULT_WIKI)
                wiki = await self.llm.update_wiki(current_wiki, changed_markdowns, changes)

        # Step 4: ensure metadata timestamps, then persist
        wiki.setdefault("metadata", {})["updated_at"] = datetime.now().isoformat()
        self.storage.put_json(_WIKI_KEY, wiki)
        self.storage.put_json(_SNAPSHOT_KEY, markdowns)

        logger.info(f"Processing complete for {timestamp}")

        return ProcessResponse(
            status="success",
            message=f"Wiki {'generated' if is_first_run else 'updated'} successfully",
            wiki_url="minio://wiki-data/wiki.json",
            changes_summary=changes,
            timestamp=datetime.now().isoformat(),
        )
