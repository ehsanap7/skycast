"""Tests for the ingestion checkpoint."""

import json
from pathlib import Path

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.storage import LocalStore
from skycast.errors import DataSourceError
from skycast.ingestion.checkpoint import (
    Checkpoint,
    SourceKind,
    checkpoint_key,
    read_checkpoint,
    write_checkpoint,
)

ARCHIVE = pd.Timestamp("2024-01-01 00:00", tz="UTC")
RECENT = pd.Timestamp("2024-01-01 02:00", tz="UTC")
KEY = checkpoint_key("sample")


def test_first_read_starts_from_seed(seeded_root: Path, city: City) -> None:
    """A new checkpoint uses the cutoff and last seed hour."""
    checkpoint = read_checkpoint(LocalStore(seeded_root), city)

    assert checkpoint == Checkpoint(archive=ARCHIVE, recent=RECENT)


def test_first_read_does_not_write(seeded_root: Path, city: City) -> None:
    """Reading never creates state."""
    store = LocalStore(seeded_root)

    read_checkpoint(store, city)

    assert not store.exists(KEY)


def test_missing_seed_raises(tmp_path: Path, city: City) -> None:
    """A checkpoint cannot start without a seed file."""
    with pytest.raises(DataSourceError, match="not found"):
        read_checkpoint(LocalStore(tmp_path), city)


def test_write_then_read_round_trip(tmp_path: Path, city: City) -> None:
    """A stored checkpoint loads back unchanged."""
    store = LocalStore(tmp_path)
    checkpoint = Checkpoint(archive=ARCHIVE, recent=RECENT)

    write_checkpoint(store, city, checkpoint)

    assert read_checkpoint(store, city) == checkpoint


def test_advanced_moves_only_one_source() -> None:
    """Each source keeps its own position."""
    moved = Checkpoint(ARCHIVE, RECENT).advanced(SourceKind.RECENT, RECENT + pd.Timedelta(hours=1))

    assert moved.archive == ARCHIVE
    assert moved.recent == RECENT + pd.Timedelta(hours=1)


@pytest.mark.parametrize("delta", [0, -1])
def test_advanced_cannot_move_back(delta: int) -> None:
    """A checkpoint only moves forward."""
    with pytest.raises(DataSourceError, match="cannot move back"):
        Checkpoint(ARCHIVE, RECENT).advanced(SourceKind.RECENT, RECENT + pd.Timedelta(hours=delta))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "Cannot parse"),
        (b"[]", "JSON object"),
        (b'{"archive": "2024-01-01T00:00:00+00:00"}', "valid recent"),
        (json.dumps({"archive": "bad", "recent": "bad"}).encode(), "Invalid checkpoint"),
        (
            json.dumps(
                {"archive": "2024-01-01T00:00:00-05:00", "recent": "2024-01-01T00:00:00+00:00"}
            ).encode(),
            "must be UTC",
        ),
        (
            json.dumps(
                {"archive": "2024-01-01T00:00:00", "recent": "2024-01-01T00:00:00+00:00"}
            ).encode(),
            "must be UTC",
        ),
    ],
)
def test_invalid_checkpoint_raises(
    tmp_path: Path, city: City, content: bytes, message: str
) -> None:
    """A damaged checkpoint stops the run."""
    store = LocalStore(tmp_path)
    store.write_bytes(KEY, content)

    with pytest.raises(DataSourceError, match=message):
        read_checkpoint(store, city)
