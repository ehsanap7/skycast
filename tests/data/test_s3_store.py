"""Tests for S3 storage, using a fake S3 client."""

from collections.abc import Iterator
from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from skycast.config import StorageSettings
from skycast.data.s3_store import S3Store, create_s3_store
from skycast.data.storage import build_store
from skycast.errors import ConfigError, StorageError

BUCKET = "sample-bucket"


def client_error(status: int, operation: str) -> ClientError:
    """Build an S3 client error with an HTTP status."""
    response: Any = {
        "Error": {"Code": str(status), "Message": "simulated"},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, operation)


class FakeS3:
    """A small in-memory S3 client."""

    def __init__(self, page_size: int = 2) -> None:
        """Start with an empty bucket list and object map."""
        self.objects: dict[str, bytes] = {}
        self.buckets: set[str] = set()
        self.page_size = page_size
        self.failure: Exception | None = None

    def _raise_failure(self) -> None:
        """Raise a scripted failure."""
        if self.failure:
            raise self.failure

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        """Return a stored object."""
        self._raise_failure()
        if Key not in self.objects:
            raise client_error(404, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}

    def put_object(
        self,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Body: bytes,  # noqa: N803
        IfNoneMatch: str | None = None,  # noqa: N803
    ) -> None:
        """Store an object, optionally only if it is new."""
        self._raise_failure()
        if IfNoneMatch == "*" and Key in self.objects:
            raise client_error(412, "PutObject")
        self.objects[Key] = Body

    def head_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        """Check that an object exists."""
        self._raise_failure()
        if Key not in self.objects:
            raise client_error(404, "HeadObject")

    def head_bucket(self, Bucket: str) -> None:  # noqa: N803
        """Check that a bucket exists."""
        self._raise_failure()
        if Bucket not in self.buckets:
            raise client_error(404, "HeadBucket")

    def create_bucket(self, Bucket: str) -> None:  # noqa: N803
        """Create a bucket."""
        self.buckets.add(Bucket)

    def get_paginator(self, _name: str) -> FakeS3:
        """Return a paginator."""
        return self

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:  # noqa: N803
        """Yield objects in small pages."""
        self._raise_failure()
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        for start in range(0, len(keys), self.page_size):
            yield {"Contents": [{"Key": key} for key in keys[start : start + self.page_size]]}


@pytest.fixture
def fake() -> FakeS3:
    """Provide a fake S3 client."""
    return FakeS3()


@pytest.fixture
def store(fake: FakeS3) -> S3Store:
    """Provide a store using the fake client."""
    return S3Store(fake, BUCKET)


def test_write_then_read_round_trip(store: S3Store) -> None:
    """Stored content can be read back."""
    store.write_bytes("silver/city=a/file.parquet", b"content")

    assert store.exists("silver/city=a/file.parquet")
    assert store.read_bytes("silver/city=a/file.parquet") == b"content"


def test_write_replaces_content(store: S3Store) -> None:
    """A normal write replaces a dataset."""
    store.write_bytes("gold/file.parquet", b"old")
    store.write_bytes("gold/file.parquet", b"new")

    assert store.read_bytes("gold/file.parquet") == b"new"


def test_create_refuses_existing_key(store: S3Store) -> None:
    """Create-only writes keep existing content."""
    store.create_bytes("bronze/batch.json", b"original")

    with pytest.raises(StorageError, match="already exists"):
        store.create_bytes("bronze/batch.json", b"changed")

    assert store.read_bytes("bronze/batch.json") == b"original"


def test_missing_key_does_not_exist(store: S3Store) -> None:
    """A 404 means the key is absent."""
    assert not store.exists("missing.csv")


def test_missing_read_raises(store: S3Store) -> None:
    """Reading a missing key is a storage error."""
    with pytest.raises(StorageError, match="Cannot read"):
        store.read_bytes("missing.csv")


def test_exists_does_not_hide_server_errors(fake: FakeS3, store: S3Store) -> None:
    """A server failure is not treated as a missing key."""
    fake.failure = client_error(500, "HeadObject")

    with pytest.raises(StorageError, match="Cannot check"):
        store.exists("key.csv")


@pytest.mark.parametrize(
    "call",
    [
        lambda store: store.read_bytes("key"),
        lambda store: store.write_bytes("key", b"x"),
        lambda store: store.create_bytes("key", b"x"),
        lambda store: store.exists("key"),
        lambda store: store.list_keys("prefix"),
        lambda store: store.ensure_bucket(),
    ],
)
def test_connection_failure_is_wrapped(fake: FakeS3, store: S3Store, call: Any) -> None:
    """A network failure becomes a storage error."""
    fake.failure = EndpointConnectionError(endpoint_url="http://storage.test")

    with pytest.raises(StorageError):
        call(store)


@pytest.mark.parametrize("key", ["", "/absolute", "a/../b", "../outside", "a//b", "./a"])
def test_invalid_keys_are_rejected(store: S3Store, key: str) -> None:
    """Keys must be clean relative paths."""
    with pytest.raises(StorageError, match="Invalid storage key"):
        store.write_bytes(key, b"x")


def test_list_uses_folder_boundaries(store: S3Store) -> None:
    """A prefix does not match a similarly named folder."""
    store.write_bytes("bronze/a.csv", b"a")
    store.write_bytes("bronze-old/b.csv", b"b")

    assert store.list_keys("bronze") == ["bronze/a.csv"]


def test_list_reads_all_pages_and_skips_hidden(store: S3Store) -> None:
    """Listing is complete, sorted, and skips hidden files."""
    for name in ("c", "a", "e", "d", "b"):
        store.write_bytes(f"gold/{name}.parquet", b"x")
    store.write_bytes("gold/.DS_Store", b"hidden")

    assert store.list_keys("gold") == [f"gold/{name}.parquet" for name in "abcde"]


def test_list_all_keys(store: S3Store) -> None:
    """An empty prefix lists every key."""
    store.write_bytes("a/x", b"x")
    store.write_bytes("b/y", b"y")

    assert store.list_keys() == ["a/x", "b/y"]


def test_ensure_bucket_creates_once(fake: FakeS3, store: S3Store) -> None:
    """A missing bucket is created, and an existing one is kept."""
    store.ensure_bucket()
    store.ensure_bucket()

    assert fake.buckets == {BUCKET}


def test_ensure_bucket_does_not_hide_auth_errors(fake: FakeS3, store: S3Store) -> None:
    """A rejected login is reported."""
    fake.failure = client_error(403, "HeadBucket")

    with pytest.raises(StorageError, match="Cannot check bucket"):
        store.ensure_bucket()


@pytest.mark.parametrize(
    "missing", ["SKYCAST_S3_ENDPOINT", "SKYCAST_S3_ACCESS_KEY", "SKYCAST_S3_SECRET_KEY"]
)
def test_missing_environment_raises(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Connection settings come from the environment."""
    monkeypatch.setenv("SKYCAST_S3_ENDPOINT", "http://storage.test:8333")
    monkeypatch.setenv("SKYCAST_S3_ACCESS_KEY", "access")
    monkeypatch.setenv("SKYCAST_S3_SECRET_KEY", "secret")
    monkeypatch.delenv(missing)

    with pytest.raises(ConfigError, match=missing):
        create_s3_store(BUCKET)


def test_build_store_creates_s3_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The factory builds the S3 backend from settings."""
    monkeypatch.setenv("SKYCAST_S3_ENDPOINT", "http://storage.test:8333")
    monkeypatch.setenv("SKYCAST_S3_ACCESS_KEY", "access")
    monkeypatch.setenv("SKYCAST_S3_SECRET_KEY", "secret")

    built = build_store(StorageSettings(backend="s3", root=tmp_path, bucket=BUCKET))

    assert isinstance(built, S3Store)
