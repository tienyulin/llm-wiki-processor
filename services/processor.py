import logging
from datetime import datetime

from models.schemas import ProcessResponse
from services.llm import MinimaxClient
from storage.minio_client import MinioStorage

logger = logging.getLogger(__name__)

_SNAPSHOT_KEY = "markdowns_snapshot.json"


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

    def _load_current_wiki_files(self) -> dict[str, str]:
        """Load all current wiki files (non-JSON) from Minio."""
        keys = self.storage.list_files()
        files = {}
        for key in keys:
            if key.endswith(".json"):
                continue
            content = self.storage.get_file(key)
            if content:
                files[key] = content
        return files

    async def process(self, markdowns: dict, timestamp: str) -> ProcessResponse:
        """
        Full pipeline:
        1. Get old snapshot from storage.
        2. Detect changes.
        3. Call LLM (initial or incremental).
        4. Save updated wiki files + new snapshot.
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
            wiki_files = await self.llm.generate_wiki(markdowns)
        else:
            changed_files = set(changes["added"]) | set(changes["modified"])
            changed_markdowns = {f: markdowns[f] for f in changed_files}

            if not changed_markdowns:
                logger.info("No content changes, skipping LLM call")
                wiki_files = self._load_current_wiki_files()
            else:
                current_files = self._load_current_wiki_files()
                wiki_files = await self.llm.update_wiki(current_files, changed_markdowns, changes)

        # Step 4: persist wiki files and snapshot
        for path, content in wiki_files.items():
            self.storage.put_file(path, content)
        self.storage.put_json(_SNAPSHOT_KEY, markdowns)

        logger.info(f"Processing complete for {timestamp}: saved {len(wiki_files)} wiki files")

        return ProcessResponse(
            status="success",
            message=f"Wiki {'generated' if is_first_run else 'updated'} successfully",
            wiki_url="minio://wiki-data/",
            changes_summary=changes,
            timestamp=datetime.now().isoformat(),
        )
