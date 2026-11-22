"""Copy local datasets into object storage without deleting anything."""

import hashlib
import logging
from dataclasses import dataclass

from skycast.config import Settings
from skycast.data.s3_store import create_s3_store
from skycast.data.storage import DatasetStore, Layer, LocalStore
from skycast.errors import StorageError

logger = logging.getLogger(__name__)

MIGRATED_PREFIXES: tuple[str, ...] = (*Layer, "state")


@dataclass(frozen=True, slots=True)
class MigrationSummary:
    """What a migration copied and verified."""

    files: int
    copied: int
    already_present: int
    bytes_verified: int


def _sha256(content: bytes) -> str:
    """Hash file content."""
    return hashlib.sha256(content).hexdigest()


def _require_same(target: DatasetStore, key: str, expected: str) -> None:
    """Check that a stored file matches the expected hash."""
    if _sha256(target.read_bytes(key)) != expected:
        raise StorageError(f"Stored file differs from source: {key}")


def _copy_file(source: DatasetStore, target: DatasetStore, key: str) -> tuple[bool, int]:
    """Copy one file if needed and verify its hash."""
    content = source.read_bytes(key)
    expected = _sha256(content)
    copied = not target.exists(key)
    if copied:
        target.create_bytes(key, content)
    _require_same(target, key, expected)
    return copied, len(content)


def migrate_store(source: DatasetStore, target: DatasetStore) -> MigrationSummary:
    """Copy every dataset to a target and verify each file."""
    keys = [key for prefix in MIGRATED_PREFIXES for key in source.list_keys(prefix)]
    if not keys:
        raise StorageError("No datasets found to migrate")
    results = [_copy_file(source, target, key) for key in keys]
    copied = sum(was_copied for was_copied, _ in results)
    logger.info("Verified %d files, copied %d", len(keys), copied)
    return MigrationSummary(
        files=len(keys),
        copied=copied,
        already_present=len(keys) - copied,
        bytes_verified=sum(size for _, size in results),
    )


def run_migration(settings: Settings) -> MigrationSummary:
    """Copy the local data folder into the configured bucket."""
    target = create_s3_store(settings.storage.bucket)
    target.ensure_bucket()
    return migrate_store(LocalStore(settings.storage.root), target)
