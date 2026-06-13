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

        logger.info(f"Initializing Minio: endpoint={endpoint}, bucket={self.bucket}")
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
                logger.info(f"✓ Created bucket: {self.bucket}")
            else:
                logger.info(f"✓ Bucket exists: {self.bucket}")
        except S3Error as e:
            if e.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                logger.info(f"✓ Bucket already present: {self.bucket}")
                return
            logger.error(f"✗ Could not ensure bucket existence: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"✗ Could not ensure bucket existence: {e}", exc_info=True)
            raise

    def ping(self) -> bool:
        """Connectivity probe for health checks; never raises."""
        try:
            return bool(self.client.bucket_exists(self.bucket))
        except Exception as e:
            logger.error(f"Minio ping failed: {e}")
            return False

    def get_json(self, key: str) -> dict | None:
        """Retrieve a JSON object from Minio. Returns None if key does not exist."""
        try:
            obj = self.client.get_object(self.bucket, key)
            content = obj.read().decode()
            logger.info(f"✓ Retrieved {key} from Minio ({len(content)} bytes)")
            return json.loads(content)
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.debug(f"Key not found in Minio: {key}")
                return None
            logger.error(f"✗ Minio error retrieving {key}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error retrieving {key}: {e}", exc_info=True)
            raise

    def put_json(self, key: str, data: dict) -> None:
        """Store a dict as JSON in Minio."""
        try:
            encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
            logger.info(f"Saving {key} to Minio ({len(encoded)} bytes, {len(data.get('apis', {}))} modules)")
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(encoded),
                length=len(encoded),
            )
            logger.info(f"✓ Successfully saved {key} to Minio")
        except Exception as e:
            logger.error(f"✗ Failed to save {key} to Minio: {e}", exc_info=True)
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
            logger.error(f"✗ Minio error retrieving {key}: {e}", exc_info=True)
            raise

    def put_json_if_match(self, key: str, data: dict, etag: str) -> bool:
        """Conditionally store JSON only if the object's ETag still matches.

        Returns False when another writer modified the object in the meantime
        (HTTP 412); callers re-read, re-merge, and retry.
        """
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        try:
            self.client._put_object(
                self.bucket, key, encoded,
                headers={**_JSON_HEADERS, "If-Match": etag},
            )
            return True
        except S3Error as e:
            if e.code == "PreconditionFailed":
                return False
            logger.error(f"✗ Conditional write failed for {key}: {e}", exc_info=True)
            raise

    def put_json_if_absent(self, key: str, data: dict) -> bool:
        """Store JSON only if the key does not exist yet (If-None-Match: *).

        Returns False when the object already exists — used to prevent
        concurrent double-initialization.
        """
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        try:
            self.client._put_object(
                self.bucket, key, encoded,
                headers={**_JSON_HEADERS, "If-None-Match": "*"},
            )
            return True
        except S3Error as e:
            if e.code == "PreconditionFailed":
                return False
            logger.error(f"✗ Conditional create failed for {key}: {e}", exc_info=True)
            raise

    def get_file(self, key: str) -> str | None:
        """Retrieve a text file from Minio. Returns None if key does not exist."""
        try:
            obj = self.client.get_object(self.bucket, key)
            return obj.read().decode()
        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            logger.error(f"✗ Minio error retrieving {key}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error retrieving {key}: {e}", exc_info=True)
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
            logger.info(f"✓ Saved {key} to Minio ({len(encoded)} bytes)")
        except Exception as e:
            logger.error(f"✗ Failed to save {key} to Minio: {e}", exc_info=True)
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
        return await asyncio.to_thread(self.get_json, key)

    async def aput_json(self, key: str, data: dict) -> None:
        await asyncio.to_thread(self.put_json, key, data)

    async def aget_json_with_etag(self, key: str) -> tuple[dict | None, str | None]:
        return await asyncio.to_thread(self.get_json_with_etag, key)

    async def aput_json_if_match(self, key: str, data: dict, etag: str) -> bool:
        return await asyncio.to_thread(self.put_json_if_match, key, data, etag)

    async def aput_json_if_absent(self, key: str, data: dict) -> bool:
        return await asyncio.to_thread(self.put_json_if_absent, key, data)

    async def aget_file(self, key: str) -> str | None:
        return await asyncio.to_thread(self.get_file, key)

    async def aput_file(self, key: str, content: str) -> None:
        await asyncio.to_thread(self.put_file, key, content)

    async def alist_files(self, prefix: str = "") -> list[str]:
        return await asyncio.to_thread(self.list_files, prefix)
