"""MinIO-backed storage for wiki JSON/markdown, with ETag CAS conditional writes."""

import asyncio
import io
import json
import logging
import os

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)
logging.getLogger("minio").setLevel(logging.INFO)

_JSON_HEADERS = {"Content-Type": "application/json"}


class MinioStorage:
    """Wraps Minio operations for wiki storage.

    Conditional writes (ETag CAS) rely on `Minio._put_object`, the only
    header-capable upload path in minio-py 7.2.x (verified against 7.2.20 and
    MinIO RELEASE.2025-09). Pin minio>=7.2 in requirements; if a future
    release breaks this, the smoke assertions in tests/test_storage_cas.py
    will catch it.
    """

    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "wiki-data")

        logger.info("Initializing Minio: endpoint=%s, bucket=%s", endpoint, self.bucket)
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Ensure bucket exists, create if not.

        bucket_exists()+make_bucket() is a check-then-act with an inherent
        race, and MinIO can return a stale 404 from bucket_exists() on the
        first boot of a fresh volume — make_bucket() then fails with
        BucketAlreadyOwnedByYou. Treat "already there" as success so startup
        is idempotent.
        """
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("✓ Created bucket: %s", self.bucket)
            else:
                logger.info("✓ Bucket exists: %s", self.bucket)
        except S3Error as e:
            if e.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                logger.info("✓ Bucket already present: %s", self.bucket)
                return
            logger.error("✗ Could not ensure bucket existence: %s", e, exc_info=True)
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Log any unexpected backend error with context, then re-raise.
            logger.error("✗ Could not ensure bucket existence: %s", e, exc_info=True)
            raise

    def ping(self) -> bool:
        """Connectivity probe for health checks; never raises."""
        try:
            return bool(self.client.bucket_exists(self.bucket))
        except Exception as e:  # pylint: disable=broad-exception-caught
            # A health probe must swallow everything and report unhealthy.
            logger.error("Minio ping failed: %s", e)
            return False

    def get_json(self, key: str) -> dict | None:
        """Retrieve a JSON object from Minio. Returns None if key does not exist."""
        try:
            obj = self.client.get_object(self.bucket, key)
            content = obj.read().decode()
            logger.info("✓ Retrieved %s from Minio (%d bytes)", key, len(content))
            return json.loads(content)
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug("Key not found in Minio: %s", key)
                return None
            logger.error("✗ Minio error retrieving %s: %s", key, e, exc_info=True)
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Log any non-S3 error (decode/json) with context, then re-raise.
            logger.error("✗ Unexpected error retrieving %s: %s", key, e, exc_info=True)
            raise

    def put_json(self, key: str, data: dict) -> None:
        """Store a dict as JSON in Minio."""
        try:
            encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
            logger.info(
                "Saving %s to Minio (%d bytes, %d modules)",
                key,
                len(encoded),
                len(data.get("apis", {})),
            )
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(encoded),
                length=len(encoded),
            )
            logger.info("✓ Successfully saved %s to Minio", key)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Surface the failing key in logs, then re-raise to the caller.
            logger.error("✗ Failed to save %s to Minio: %s", key, e, exc_info=True)
            raise

    def get_json_with_etag(self, key: str) -> tuple[dict | None, str | None]:
        """Retrieve a JSON object together with its ETag for CAS writes.

        Returns (None, None) if the key does not exist.
        """
        try:
            obj = self.client.get_object(self.bucket, key)
            etag = (obj.headers.get("ETag") or "").strip('"') or None
            return json.loads(obj.read().decode()), etag
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None, None
            logger.error("✗ Minio error retrieving %s: %s", key, e, exc_info=True)
            raise

    def put_json_if_match(self, key: str, data: dict, etag: str) -> bool:
        """Conditionally store JSON only if the object's ETag still matches.

        Returns False when another writer modified the object in the meantime
        (HTTP 412); callers re-read, re-merge, and retry.
        """
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        try:
            # _put_object: only header-capable upload path in minio-py (see class docstring).
            self.client._put_object(  # pylint: disable=protected-access
                self.bucket,
                key,
                encoded,
                headers={**_JSON_HEADERS, "If-Match": etag},
            )
            return True
        except S3Error as e:
            if e.code == "PreconditionFailed":
                return False
            logger.error("✗ Conditional write failed for %s: %s", key, e, exc_info=True)
            raise

    def put_json_if_absent(self, key: str, data: dict) -> bool:
        """Store JSON only if the key does not exist yet (If-None-Match: *).

        Returns False when the object already exists — used to prevent
        concurrent double-initialization.
        """
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        try:
            # _put_object: only header-capable upload path in minio-py (see class docstring).
            self.client._put_object(  # pylint: disable=protected-access
                self.bucket,
                key,
                encoded,
                headers={**_JSON_HEADERS, "If-None-Match": "*"},
            )
            return True
        except S3Error as e:
            if e.code == "PreconditionFailed":
                return False
            logger.error("✗ Conditional create failed for %s: %s", key, e, exc_info=True)
            raise

    def get_file(self, key: str) -> str | None:
        """Retrieve a text file from Minio. Returns None if key does not exist."""
        try:
            obj = self.client.get_object(self.bucket, key)
            return obj.read().decode()
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            logger.error("✗ Minio error retrieving %s: %s", key, e, exc_info=True)
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Log any non-S3 error (decode) with context, then re-raise.
            logger.error("✗ Unexpected error retrieving %s: %s", key, e, exc_info=True)
            raise

    def put_file(self, key: str, content: str) -> None:
        """Store a text file in Minio."""
        try:
            encoded = content.encode()
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(encoded),
                length=len(encoded),
                content_type="text/markdown",
            )
            logger.info("✓ Saved %s to Minio (%d bytes)", key, len(encoded))
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Surface the failing key in logs, then re-raise to the caller.
            logger.error("✗ Failed to save %s to Minio: %s", key, e, exc_info=True)
            raise

    def list_files(self, prefix: str = "") -> list[str]:
        """List all object keys in the bucket under the given prefix."""
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    # ------------------------------------------------------------------
    # Async facade — minio-py is synchronous and would otherwise block the
    # event loop; async callers (processor, routes) go through these.
    # ------------------------------------------------------------------

    async def aget_json(self, key: str) -> dict | None:
        """Async wrapper around get_json (runs in a worker thread)."""
        return await asyncio.to_thread(self.get_json, key)

    async def aput_json(self, key: str, data: dict) -> None:
        """Async wrapper around put_json (runs in a worker thread)."""
        await asyncio.to_thread(self.put_json, key, data)

    async def aget_json_with_etag(self, key: str) -> tuple[dict | None, str | None]:
        """Async wrapper around get_json_with_etag (runs in a worker thread)."""
        return await asyncio.to_thread(self.get_json_with_etag, key)

    async def aput_json_if_match(self, key: str, data: dict, etag: str) -> bool:
        """Async wrapper around put_json_if_match (runs in a worker thread)."""
        return await asyncio.to_thread(self.put_json_if_match, key, data, etag)

    async def aput_json_if_absent(self, key: str, data: dict) -> bool:
        """Async wrapper around put_json_if_absent (runs in a worker thread)."""
        return await asyncio.to_thread(self.put_json_if_absent, key, data)

    async def aget_file(self, key: str) -> str | None:
        """Async wrapper around get_file (runs in a worker thread)."""
        return await asyncio.to_thread(self.get_file, key)

    async def aput_file(self, key: str, content: str) -> None:
        """Async wrapper around put_file (runs in a worker thread)."""
        await asyncio.to_thread(self.put_file, key, content)

    async def alist_files(self, prefix: str = "") -> list[str]:
        """Async wrapper around list_files (runs in a worker thread)."""
        return await asyncio.to_thread(self.list_files, prefix)
