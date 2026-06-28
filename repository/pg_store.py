"""PGVectorStore: read-write Postgres+pgvector index for the wiki.

Postgres here is a derived, rebuildable serving index over wiki.json (MinIO
stays the source of truth). The processor syncs it best-effort after a
successful CAS write; any failure is logged, never propagated into the wiki
pipeline. PG_DSN empty => the whole layer is disabled (pg_store_from_env()
returns None), matching the PROCESSOR_API_KEY dev-mode convention.

This module owns the table DDL (ensure_schema). db/init/ only creates the
pgvector extension on first boot of the compose primary; that keeps exactly
one executable copy of the schema, usable against any PG with pgvector.

Vectors are passed as pgvector text literals ("[1,2,3]") cast with ::vector,
so no per-connection type registration is needed.
"""

import asyncio
import logging
import os
from datetime import datetime

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from services.llm.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS api_entries (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    module          TEXT        NOT NULL,
    api_key         TEXT        NOT NULL,
    source_app      TEXT        NOT NULL,
    source_version  TEXT,
    description     TEXT        NOT NULL DEFAULT '',
    detail          JSONB       NOT NULL,
    embed_text      TEXT        NOT NULL,
    embedding       vector({dim}),
    embedding_model TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (module, api_key)
);
CREATE INDEX IF NOT EXISTS idx_api_entries_source_app ON api_entries (source_app);
CREATE INDEX IF NOT EXISTS idx_api_entries_module     ON api_entries (module);
-- Knowledge documents (prose/reference). Separate table from api_entries so
-- the API search path is provably unchanged; same embeddings + pgvector machinery.
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id          TEXT        NOT NULL UNIQUE,
    source_app      TEXT        NOT NULL,
    source_version  TEXT,
    title           TEXT        NOT NULL DEFAULT '',
    detail          JSONB       NOT NULL,
    embed_text      TEXT        NOT NULL,
    embedding       vector({dim}),
    embedding_model TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_knowledge_source_app ON knowledge_entries (source_app);

CREATE TABLE IF NOT EXISTS index_state (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_sync (
    source_app     TEXT PRIMARY KEY,
    synced_at      TIMESTAMPTZ NOT NULL,
    source_version TEXT
);
"""

# Search indexes, separate from the tables: rebuild() drops and recreates
# them around its bulk insert — building HNSW once over the full data set is
# far cheaper than maintaining it row by row for tens of thousands of rows.
_SEARCH_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_api_entries_embedding
    ON api_entries USING hnsw (embedding vector_cosine_ops);
-- Trigram GIN makes ILIKE '%term%' indexed; embed_text concatenates
-- module | api_key | endpoint | description | params, so keyword search
-- over this one column covers the same haystack as the old wiki scan.
CREATE INDEX IF NOT EXISTS idx_api_entries_embed_text_trgm
    ON api_entries USING gin (embed_text gin_trgm_ops);
-- Knowledge: vector (semantic) + trigram (keyword) so reads can run a hybrid
-- (RRF) query — evidence shows fusion beats either signal alone.
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
    ON knowledge_entries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_embed_text_trgm
    ON knowledge_entries USING gin (embed_text gin_trgm_ops);
"""

_DROP_SEARCH_INDEXES_SQL = """
DROP INDEX IF EXISTS idx_api_entries_embedding;
DROP INDEX IF EXISTS idx_api_entries_embed_text_trgm;
"""

_INSERT_ENTRY_SQL = """
INSERT INTO api_entries
    (module, api_key, source_app, source_version, description, detail,
     embed_text, embedding, embedding_model, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, now())
ON CONFLICT (module, api_key) DO UPDATE SET
    source_app      = EXCLUDED.source_app,
    source_version  = EXCLUDED.source_version,
    description     = EXCLUDED.description,
    detail          = EXCLUDED.detail,
    embed_text      = EXCLUDED.embed_text,
    embedding       = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    updated_at      = now()
"""


_INSERT_KNOWLEDGE_SQL = """
INSERT INTO knowledge_entries
    (doc_id, source_app, source_version, title, detail, embed_text, embedding,
     embedding_model, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, now())
ON CONFLICT (doc_id) DO UPDATE SET
    source_app      = EXCLUDED.source_app,
    source_version  = EXCLUDED.source_version,
    title           = EXCLUDED.title,
    detail          = EXCLUDED.detail,
    embed_text      = EXCLUDED.embed_text,
    embedding       = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    updated_at      = now()
"""


def _dominant_app_links(candidates: list[tuple], threshold: float, margin: float) -> list[tuple]:
    """Pick a knowledge doc's API links, or none.

    candidates: [(module, api_key, source_app, score), ...] sorted score desc.
    A doc links only if its best match's app dominates — best score minus the
    best score of any *other* app >= margin (a generic doc similar to many apps
    fails this and links to nothing). Returns that app's matches with
    score >= threshold as [(module, api_key, round(score,4)), ...].
    """
    if not candidates:
        return []
    best_app = candidates[0][2]
    best_score = candidates[0][3]
    other_best = max((s for _, _, app, s in candidates if app != best_app), default=0.0)
    if best_score - other_best < margin:
        return []
    return [
        (m, k, round(s, 4)) for (m, k, app, s) in candidates if app == best_app and s >= threshold
    ]


def to_vector_literal(vec) -> str | None:
    """Format a vector as a pgvector text literal; None passes through as NULL."""
    if vec is None:
        return None
    return "[" + ",".join(repr(float(c)) for c in vec) + "]"


class PGVectorStore:
    """Read-write access to the api_entries index (wiki-processor side)."""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10, dim: int = 1536):
        self.dim = dim
        self._schema_ready = False
        # open=False + lazy aopen(): routes.py builds singletons at import
        # time, before any event loop exists.
        self._pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            timeout=10,
            kwargs={"connect_timeout": 5},
            check=AsyncConnectionPool.check_connection,
        )
        self._open_lock = asyncio.Lock()
        self._opened = False
        self._schema_lock = asyncio.Lock()

    async def _ensure_open(self):
        if self._opened:
            return
        async with self._open_lock:
            if not self._opened:
                # wait=False: an unreachable PG must not block startup; each
                # operation then fails fast on its own connection attempt.
                await self._pool.open(wait=False)
                self._opened = True

    async def aclose(self):
        """Close the connection pool if it was opened."""
        if self._opened:
            await self._pool.close()
            self._opened = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self, dim: int):
        """Create extension/tables idempotently; refuse on dimension mismatch.

        Changing EMBEDDING_DIM (or the embedding model's size) makes stored
        vectors incomparable — the fix is documented in
        docs/troubleshooting.md: drop the tables and POST /admin/reindex.
        """
        await self._ensure_open()
        async with self._pool.connection() as conn:
            for extension in ("vector", "pg_trgm"):
                try:
                    await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
                # CREATE EXTENSION can fail for several privilege/availability
                # reasons (non-superuser role, missing package); recover by
                # checking whether it already exists.
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # Not superuser — fine if the extension already exists.
                    await conn.rollback()
                    cur = await conn.execute(
                        "SELECT 1 FROM pg_extension WHERE extname = %s", (extension,)
                    )
                    if await cur.fetchone() is None:
                        raise ConfigurationException(
                            f"The {extension} extension is not installed and this "
                            f"role cannot create it; run db/init/01-extension.sql "
                            f"as superuser"
                        ) from exc

            existing_dim = await self._embedding_dim(conn)
            if existing_dim is not None and existing_dim != dim:
                raise ConfigurationException(
                    f"api_entries.embedding is vector({existing_dim}) but "
                    f"EMBEDDING_DIM={dim}; old and new vectors are incomparable. "
                    f"Drop the api_entries table and POST /admin/reindex."
                )
            # Always run the (fully idempotent, CREATE TABLE IF NOT EXISTS) DDL.
            # Gating it on api_entries' existence would skip creating sibling
            # tables (e.g. knowledge_entries) whenever the schema is partial —
            # e.g. a DB that an older build populated with only api_entries —
            # leaving "relation knowledge_entries does not exist" forever. The
            # dim-mismatch guard above still protects api_entries' vector width.
            await conn.execute(_TABLES_SQL.format(dim=int(dim)))
            await conn.execute(_SEARCH_INDEXES_SQL)
            await conn.commit()

    async def ensure_schema_once(self):
        """ensure_schema(self.dim), cached after the first success.

        Locked: a concurrent burst (100 simultaneous /process) must run the
        DDL exactly once, not as a thundering herd of CREATE INDEX checks."""
        if self._schema_ready:
            return
        async with self._schema_lock:
            if not self._schema_ready:
                await self.ensure_schema(self.dim)
                self._schema_ready = True

    async def _embedding_dim(self, conn) -> int | None:
        """Dimension of the existing embedding column, or None if no table.

        pgvector stores the dimension directly as the column's atttypmod."""
        cur = await conn.execute("""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'api_entries' AND a.attname = 'embedding'
            """)
        row = await cur.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def replace_app_entries(
        self,
        source_app: str,
        source_version: str,
        rows: list[dict],
        synced_at: datetime,
    ) -> bool:
        """Atomically replace one app's entries (mirrors _merge_app_entries).

        rows: [{module, api_key, description, detail, embed_text,
                embedding (list|None), embedding_model}, ...]

        Returns False when a newer sync for this app already landed (the
        app_sync guard) — protects against out-of-order replays from
        concurrent processor replicas.
        """
        await self._ensure_open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                # The index is rebuildable from wiki.json, so losing the last
                # few commits in a crash is repairable — skipping the WAL
                # fsync per app sync is a safe ~10x latency win under bursts.
                await conn.execute("SET LOCAL synchronous_commit = off")
                cur = await conn.execute(
                    """
                    INSERT INTO app_sync (source_app, synced_at, source_version)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_app) DO UPDATE SET
                        synced_at = EXCLUDED.synced_at,
                        source_version = EXCLUDED.source_version
                    WHERE app_sync.synced_at < EXCLUDED.synced_at
                    RETURNING source_app
                    """,
                    (source_app, synced_at, source_version),
                )
                if await cur.fetchone() is None:
                    logger.warning("PG sync for %s at %s is stale, skipping", source_app, synced_at)
                    return False

                await conn.execute("DELETE FROM api_entries WHERE source_app = %s", (source_app,))
                await self._insert_rows(conn, source_app, source_version, rows)
                await self._set_state(
                    conn,
                    "last_sync",
                    {
                        "source_app": source_app,
                        "synced_at": synced_at.isoformat(),
                        "entries": len(rows),
                    },
                )
        return True

    async def replace_app_knowledge(
        self,
        source_app: str,
        source_version: str,
        rows: list[dict],
        synced_at: datetime,
    ) -> bool:
        """Replace one app's knowledge entries (mirrors replace_app_entries).

        rows: [{doc_id, title, detail, embed_text, embedding, embedding_model}].
        Reuses the app_sync guard so a stale replay can't clobber a newer sync.
        """
        await self._ensure_open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL synchronous_commit = off")
                cur = await conn.execute(
                    """
                    INSERT INTO app_sync (source_app, synced_at, source_version)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_app) DO UPDATE SET
                        synced_at = EXCLUDED.synced_at,
                        source_version = EXCLUDED.source_version
                    WHERE app_sync.synced_at < EXCLUDED.synced_at
                    RETURNING source_app
                    """,
                    (source_app, synced_at, source_version),
                )
                if await cur.fetchone() is None:
                    logger.warning("PG knowledge sync for %s is stale, skipping", source_app)
                    return False

                await conn.execute(
                    "DELETE FROM knowledge_entries WHERE source_app = %s", (source_app,)
                )
                if rows:
                    async with conn.cursor() as kcur:
                        await kcur.executemany(
                            _INSERT_KNOWLEDGE_SQL,
                            [
                                (
                                    r["doc_id"],
                                    source_app,
                                    source_version,
                                    r.get("title", ""),
                                    Jsonb(r["detail"]),
                                    r["embed_text"],
                                    to_vector_literal(r.get("embedding")),
                                    r.get("embedding_model"),
                                )
                                for r in rows
                            ],
                        )
        return True

    async def knowledge_api_links(
        self, threshold: float = 0.63, margin: float = 0.05
    ) -> dict[str, list[tuple]]:
        """Semantic links between knowledge docs and API entries by embedding
        proximity (concept linking that survives synonyms — substring misses
        "roll back" ↔ "recover").

        A *specific* doc concentrates its top matches in one app; a *generic*
        doc (e.g. a FastAPI how-to) is flatly similar to endpoints across many
        apps and must NOT link. So a doc links only when its best app dominates:
        best score − best score of any *other* app >= `margin`. Then that app's
        endpoints with cosine >= `threshold` are linked. Both numbers measured
        on-corpus (see docs/architecture/semantic-concept-linking.md).

        Returns {doc_id: [(module, api_key, score), ...]} (only linked docs).
        """
        await self._ensure_open()
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT k.doc_id, a.module, a.api_key, a.source_app,
                       1 - (a.embedding <=> k.embedding) AS score
                FROM knowledge_entries k
                CROSS JOIN LATERAL (
                    SELECT module, api_key, source_app, embedding
                    FROM api_entries
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> k.embedding
                    LIMIT 10
                ) a
                WHERE k.embedding IS NOT NULL
                ORDER BY k.doc_id, score DESC
                """,
            )
            by_doc: dict[str, list[tuple]] = {}
            for doc_id, module, api_key, source_app, score in await cur.fetchall():
                by_doc.setdefault(doc_id, []).append((module, api_key, source_app, float(score)))
            return {
                doc_id: links
                for doc_id, cands in by_doc.items()
                if (links := _dominant_app_links(cands, threshold, margin))
            }

    async def rebuild(self, apps: dict[str, list[dict]], versions: dict[str, str]) -> int:
        """Full rebuild from wiki.json (bootstrap on existing data, drift repair).

        apps: {source_app: rows}; wipes everything first, in one transaction.
        """
        await self._ensure_open()
        now = datetime.now().astimezone()
        total = 0
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL synchronous_commit = off")
                # Bulk path: drop search indexes, insert everything, rebuild
                # them once at the end (row-by-row HNSW maintenance dominates
                # rebuild time past a few thousand entries).
                await conn.execute(_DROP_SEARCH_INDEXES_SQL)
                await conn.execute("TRUNCATE api_entries, app_sync")
                for source_app, rows in apps.items():
                    version = versions.get(source_app, "unknown")
                    await conn.execute(
                        "INSERT INTO app_sync (source_app, synced_at, source_version) "
                        "VALUES (%s, %s, %s)",
                        (source_app, now, version),
                    )
                    await self._insert_rows(conn, source_app, version, rows)
                    total += len(rows)
                await self._set_state(
                    conn,
                    "last_rebuild",
                    {"at": now.isoformat(), "apps": len(apps), "entries": total},
                )
                await conn.execute(_SEARCH_INDEXES_SQL)
        return total

    async def _insert_rows(self, conn, source_app: str, source_version: str, rows: list[dict]):
        if not rows:
            return
        async with conn.cursor() as cur:
            await cur.executemany(
                _INSERT_ENTRY_SQL,
                [
                    (
                        r["module"],
                        r["api_key"],
                        source_app,
                        source_version,
                        r.get("description", ""),
                        Jsonb(r["detail"]),
                        r["embed_text"],
                        to_vector_literal(r.get("embedding")),
                        r.get("embedding_model"),
                    )
                    for r in rows
                ],
            )

    async def _set_state(self, conn, key: str, value: dict):
        await conn.execute(
            """
            INSERT INTO index_state (key, value, updated_at) VALUES (%s, %s, now())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            (key, Jsonb(value)),
        )

    # ------------------------------------------------------------------
    # Reads (health / verification)
    # ------------------------------------------------------------------

    async def available(self) -> bool:
        """True if a trivial query succeeds; any failure means PG is unavailable."""
        try:
            await self._ensure_open()
            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        # Availability probe: any connection/query error simply means "not
        # available", so it must swallow every exception type.
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("PG availability probe failed: %s", e)
            return False

    async def count_entries(self) -> tuple[int, int]:
        """(total entries, entries with an embedding)."""
        await self._ensure_open()
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT count(*), count(embedding) FROM api_entries")
            row = await cur.fetchone()
            assert row is not None  # count(*) always returns exactly one row
            total, embedded = row
            return total, embedded

    async def semantic_search(self, query_vec: list[float], top_k: int = 10) -> list[dict]:
        """ANN top-k by cosine distance (used by tests and verification)."""
        await self._ensure_open()
        literal = to_vector_literal(query_vec)
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT module, api_key, description, source_app,
                       1 - (embedding <=> %s::vector) AS score
                FROM api_entries
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, top_k),
            )
            return [
                {
                    "module": m,
                    "api_key": k,
                    "description": d,
                    "source_app": s,
                    "score": float(score),
                }
                for m, k, d, s, score in await cur.fetchall()
            ]


def pg_store_from_env() -> PGVectorStore | None:
    """Build the store from PG_DSN, or None when the layer is disabled."""
    dsn = os.getenv("PG_DSN", "").strip()
    if not dsn:
        return None
    return PGVectorStore(
        dsn,
        min_size=int(os.getenv("PG_POOL_MIN", "1")),
        max_size=int(os.getenv("PG_POOL_MAX", "10")),
        dim=int(os.getenv("EMBEDDING_DIM", "1536")),
    )
