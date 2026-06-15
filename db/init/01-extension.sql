-- Runs as superuser on the database's first boot (mounted into
-- /docker-entrypoint-initdb.d of the compose `pg` service).
-- Table DDL is NOT here — wiki-processor's PGVectorStore.ensure_schema()
-- owns it (single source, works against any PG with these extensions).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
