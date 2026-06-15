# db/ — Postgres + pgvector serving index

Postgres is a **derived, rebuildable index** over MinIO's `wiki.json` (the
single source of truth). If PG state is ever wrong or lost: wipe it and call
`POST /admin/reindex` on wiki-processor.

## Layout

- `init/01-extension.sql` — `CREATE EXTENSION vector / pg_trgm`, run as
  superuser on the database's first boot (mounted into the compose `pg`
  service's `/docker-entrypoint-initdb.d`).

## Where is the table DDL?

In code: `wiki-processor/storage/pg_store.py` → `PGVectorStore.ensure_schema()`.
It is idempotent and runs on startup/first use, so the schema works against
**any** PG with pgvector + pg_trgm installed. Keeping one executable copy
avoids SQL-file/code drift; this directory only handles what requires
superuser at bootstrap time.

## Topology

A single `pgvector/pgvector:pg16` instance behind compose profile `pg`:

```
PG_DSN='postgresql://wiki:wikipass@pg:5432/wiki' docker compose --profile pg up -d
```

The index is optional and rebuildable, so single-instance durability is
acceptable: if PG dies, reads fall back to the wiki.json path automatically
and `POST /admin/reindex` restores the index afterwards.

**Scaling up later:** the client code (psycopg3) already supports multi-host
failover DSNs (`host=a,b,c` + `target_session_attrs=read-write`, covered by
`test_pg_store.py::test_multihost_dsn_skips_dead_host`), so moving to an HA
cluster — repmgr, Patroni, CloudNativePG, or a managed PG — touches only
docker-compose.yml and `PG_DSN`. No application changes.

See `docs/architecture/vector-search.md` for the full design and failure
semantics.
