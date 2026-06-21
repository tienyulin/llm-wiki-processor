"""The optional PG/pgvector index layer, split out of WikiProcessor.

This is the "best-effort, wiki.json stays canonical" side: building rows,
embedding them, and syncing api/knowledge entries into Postgres. All methods are
no-ops (or raise, for reindex) when the vector store is disabled. Mixed into
WikiProcessor — `self.vector_store`, `self.embedder`, `self.storage`,
`self._log_audit`, `self._normalize_wiki` are provided by the host class.
"""

import logging
from datetime import datetime

from services.embeddings import entry_to_text, knowledge_to_text

logger = logging.getLogger(__name__)

_WIKI_KEY = "wiki.json"
_SYSTEM_APP = "system"


class VectorSyncMixin:
    """PG/pgvector index sync (api + knowledge). Optional; best-effort."""

    def _entry_rows(self, new_apis: dict) -> list[dict]:
        """Flatten stamped entries into api_entries rows (without vectors)."""
        rows = []
        for module, endpoints in new_apis.items():
            for api_key, detail in endpoints.items():
                rows.append({
                    "module": module,
                    "api_key": api_key,
                    "description": detail.get("description", "") if isinstance(detail, dict) else "",
                    "detail": detail,
                    "embed_text": entry_to_text(module, api_key, detail),
                    "embedding": None,
                    "embedding_model": None,
                })
        return rows

    async def _embed_rows(self, app: str, rows: list[dict]):
        """Fill rows' embedding fields in place; a failing embeddings API
        degrades to NULL vectors (rows still sync relationally)."""
        if not rows or self.embedder is None or not self.embedder.is_enabled():
            return
        try:
            vectors = await self.embedder.aembed([r["embed_text"] for r in rows])
            for row, vec in zip(rows, vectors):
                row["embedding"] = vec
                row["embedding_model"] = self.embedder.config.model
        except Exception as e:
            logger.warning(f"Embedding failed for {app}, syncing without vectors: {e}")

    async def _sync_vector_index(self, app: str, version: str, rows: list[dict], synced_at: datetime):
        """Best-effort PG sync after a successful CAS write.

        Failure never propagates — the wiki write already succeeded. It is
        flagged in the audit log and repaired by POST /admin/reindex."""
        if self.vector_store is None:
            return
        try:
            await self.vector_store.ensure_schema_once()
            applied = await self.vector_store.replace_app_entries(app, version, rows, synced_at)
            if not applied:
                logger.info(f"PG index sync for {app} superseded by a newer sync, skipped")
        except Exception as e:
            logger.warning(f"PG index sync failed for {app} (wiki write succeeded): {e}")
            await self._log_audit(app, len(rows), "success_index_sync_failed", [])

    def _knowledge_rows(self, new_knowledge: dict) -> list[dict]:
        """Flatten knowledge entries into knowledge_entries rows (no vectors yet)."""
        rows = []
        for doc_id, entry in new_knowledge.items():
            rows.append({
                "doc_id": doc_id,
                "title": entry.get("title", "") if isinstance(entry, dict) else "",
                "detail": entry,
                "embed_text": knowledge_to_text(doc_id, entry),
                "embedding": None,
                "embedding_model": None,
            })
        return rows

    async def _sync_knowledge_index(self, app: str, version: str, rows: list[dict], synced_at: datetime):
        """Best-effort PG sync of knowledge entries (hybrid search index).
        Same fail-safe contract as _sync_vector_index."""
        if self.vector_store is None:
            return
        try:
            await self.vector_store.ensure_schema_once()
            applied = await self.vector_store.replace_app_knowledge(app, version, rows, synced_at)
            if not applied:
                logger.info(f"PG knowledge sync for {app} superseded by a newer sync, skipped")
        except Exception as e:
            logger.warning(f"PG knowledge sync failed for {app} (wiki write succeeded): {e}")
            await self._log_audit(app, len(rows), "success_index_sync_failed", [])

    async def reindex(self) -> dict:
        """Rebuild the entire PG index from wiki.json (bootstrap on existing
        data, drift repair). Raises when the vector layer is disabled."""
        if self.vector_store is None:
            raise RuntimeError("Vector index disabled: PG_DSN is not configured")

        # P3: rebuild from the per-app objects (the source of truth), not the
        # derived aggregate wiki.json.
        wiki = await self.aggregate_apps()

        apps: dict[str, list[dict]] = {}
        versions: dict[str, str] = {}
        for module, endpoints in wiki.get("apis", {}).items():
            if not isinstance(endpoints, dict):
                continue
            for api_key, detail in endpoints.items():
                app = detail.get("source_app", _SYSTEM_APP) if isinstance(detail, dict) else _SYSTEM_APP
                versions.setdefault(app, detail.get("source_version", "unknown") if isinstance(detail, dict) else "unknown")
                apps.setdefault(app, []).extend(self._entry_rows({module: {api_key: detail}}))

        for app, rows in apps.items():
            await self._embed_rows(app, rows)

        await self.vector_store.ensure_schema_once()
        total = await self.vector_store.rebuild(apps, versions)
        embedded = sum(1 for rows in apps.values() for r in rows if r["embedding"] is not None)
        return {"apps": len(apps), "entries": total, "embedded": embedded}
