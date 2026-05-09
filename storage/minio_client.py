import io
import json
import logging
import os

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinioStorage:
    """Wraps Minio operations for wiki storage."""

    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "wiki-data")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Ensure bucket exists, create if not."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            else:
                logger.debug(f"Bucket exists: {self.bucket}")
        except Exception as e:
            logger.warning(f"Could not ensure bucket existence: {e}")

    def get_json(self, key: str) -> dict | None:
        """Retrieve a JSON object from Minio. Returns None if key does not exist."""
        try:
            obj = self.client.get_object(self.bucket, key)
            content = obj.read().decode()
            logger.info(f"Retrieved {key} from Minio")
            return json.loads(content)
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.info(f"Key not found in Minio: {key}")
                return None
            logger.error(f"Minio error retrieving {key}: {e}")
            raise

    def put_json(self, key: str, data: dict) -> None:
        """Store a dict as JSON in Minio."""
        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(encoded),
            length=len(encoded),
        )
        logger.info(f"Saved {key} to Minio")
