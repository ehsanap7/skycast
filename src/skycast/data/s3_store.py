"""Dataset storage on any S3-compatible service."""

import os
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from skycast.errors import ConfigError, StorageError

ENDPOINT_VARIABLE = "SKYCAST_S3_ENDPOINT"
ACCESS_KEY_VARIABLE = "SKYCAST_S3_ACCESS_KEY"
SECRET_KEY_VARIABLE = "SKYCAST_S3_SECRET_KEY"
S3_REGION = "us-east-1"
S3_ERRORS = (ClientError, BotoCoreError)
HTTP_NOT_FOUND = 404
HTTP_PRECONDITION_FAILED = 412


def _status(err: Exception) -> int | None:
    """Read the HTTP status from an S3 client error."""
    if not isinstance(err, ClientError):
        return None
    return err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")


def _checked_key(key: str) -> str:
    """Reject keys that are not clean relative paths."""
    parts = key.split("/")
    if not key or "" in parts or "." in parts or ".." in parts:
        raise StorageError(f"Invalid storage key: {key!r}")
    return key


def _list_prefix(prefix: str) -> str:
    """Build a folder-style S3 prefix."""
    return f"{_checked_key(prefix)}/" if prefix else ""


def _is_visible(key: str) -> bool:
    """Skip folder markers and hidden files."""
    name = key.rsplit("/", 1)[-1]
    return bool(name) and not name.startswith(".")


class S3Store:
    """Stores datasets in an S3 bucket."""

    def __init__(self, client: Any, bucket: str) -> None:
        """Create a store using an S3 client and bucket."""
        self._client = client
        self._bucket = bucket

    def read_bytes(self, key: str) -> bytes:
        """Return the content stored at a key."""
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=_checked_key(key))
            return response["Body"].read()
        except S3_ERRORS as err:
            raise StorageError(f"Cannot read dataset {key!r}: {err}") from err

    def write_bytes(self, key: str, content: bytes) -> None:
        """Store content at a key."""
        try:
            self._client.put_object(Bucket=self._bucket, Key=_checked_key(key), Body=content)
        except S3_ERRORS as err:
            raise StorageError(f"Cannot write dataset {key!r}: {err}") from err

    def create_bytes(self, key: str, content: bytes) -> None:
        """Store content only if the key does not exist."""
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=_checked_key(key), Body=content, IfNoneMatch="*"
            )
        except S3_ERRORS as err:
            if _status(err) == HTTP_PRECONDITION_FAILED:
                raise StorageError(f"Dataset already exists: {key!r}") from err
            raise StorageError(f"Cannot create dataset {key!r}: {err}") from err

    def exists(self, key: str) -> bool:
        """Tell whether a key holds a dataset."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=_checked_key(key))
        except S3_ERRORS as err:
            if _status(err) == HTTP_NOT_FOUND:
                return False
            raise StorageError(f"Cannot check dataset {key!r}: {err}") from err
        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys under a prefix, skipping hidden files."""
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=self._bucket, Prefix=_list_prefix(prefix))
            keys = [item["Key"] for page in pages for item in page.get("Contents", [])]
        except S3_ERRORS as err:
            raise StorageError(f"Cannot list datasets under {prefix!r}: {err}") from err
        return sorted(key for key in keys if _is_visible(key))

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except S3_ERRORS as err:
            if _status(err) != HTTP_NOT_FOUND:
                raise StorageError(f"Cannot check bucket {self._bucket!r}: {err}") from err
            self._create_bucket()

    def _create_bucket(self) -> None:
        """Create the configured bucket."""
        try:
            self._client.create_bucket(Bucket=self._bucket)
        except S3_ERRORS as err:
            raise StorageError(f"Cannot create bucket {self._bucket!r}: {err}") from err


def _required_env(name: str) -> str:
    """Read a required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Missing environment variable: {name}")
    return value


def create_s3_store(bucket: str) -> S3Store:
    """Create an S3 store from environment settings."""
    client = boto3.client(
        "s3",
        endpoint_url=_required_env(ENDPOINT_VARIABLE),
        aws_access_key_id=_required_env(ACCESS_KEY_VARIABLE),
        aws_secret_access_key=_required_env(SECRET_KEY_VARIABLE),
        region_name=S3_REGION,
        config=Config(
            s3={"addressing_style": "path"},
            retries={"mode": "standard", "max_attempts": 3},
            connect_timeout=5,
            read_timeout=60,
        ),
    )
    return S3Store(client, bucket)
