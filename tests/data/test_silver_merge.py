"""Tests for merging the seed file and API batches in silver."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from skycast.config import City
from skycast.data.batch_reader import BronzeBatch
from skycast.data.contracts import BRONZE_COLUMNS, MEASUREMENT_RANGES, SourceKind
from skycast.data.silver import build_silver

CITY = City(
    slug="sample",
    name="Sample",
    latitude=1.0,
    longitude=2.0,
    timezone="UTC",
    source_file="sample.csv",
    archive_source="archive-source",
    recent_source="recent-source",
    archive_cutoff=datetime(2024, 1, 1, 1, tzinfo=UTC),
)
OLDER = pd.Timestamp("2024-01-02 00:00", tz="UTC")
NEWER = pd.Timestamp("2024-01-03 00:00", tz="UTC")


def seed(hours: int = 4) -> pd.DataFrame:
    """Build raw seed text where every temperature is 1."""
    rows = [
        {
            "time": f"2024-01-01 {hour:02d}:00:00+00:00",
            **dict.fromkeys(BRONZE_COLUMNS[1:], "1"),
        }
        for hour in range(hours)
    ]
    return pd.DataFrame(rows, columns=list(BRONZE_COLUMNS), dtype="str")


def batch(
    kind: SourceKind, fetched_at: pd.Timestamp, first_hour: int, periods: int, temperature: float
) -> BronzeBatch:
    """Build a batch with one temperature value."""
    start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=first_hour)
    times = pd.date_range(start, periods=periods, freq="h", name="time")
    values = dict.fromkeys(MEASUREMENT_RANGES, 1.0) | {"temperature_2m": temperature}
    hourly = pd.DataFrame({"time": times, **values})[list(BRONZE_COLUMNS)]
    return BronzeBatch(kind=kind, fetched_at=fetched_at, hourly=hourly)


def at_hour(frame: pd.DataFrame, hour: int) -> pd.Series:
    """Return the silver row for an hour."""
    return frame.loc[frame["time"].dt.hour.eq(hour)].iloc[0]


def test_no_batches_keeps_seed() -> None:
    """Without batches, the seed labels stay the same."""
    silver = build_silver(seed(), CITY)

    assert silver.frame["source"].tolist() == [
        "archive-source",
        "archive-source",
        "recent-source",
        "recent-source",
    ]
    assert silver.report.batches_used == 0
    assert silver.report.recent_replaced_by_archive == 0


def test_archive_replaces_recent_seed_hour() -> None:
    """Archive data wins and relabels the hour."""
    silver = build_silver(seed(), CITY, [batch(SourceKind.ARCHIVE, OLDER, 2, 1, 9.0)])

    row = at_hour(silver.frame, 2)
    assert row["temperature_2m"] == 9.0
    assert row["source"] == "archive-source"
    assert silver.report.recent_replaced_by_archive == 1
    assert silver.report.rows_by_source == {"archive-source": 3, "recent-source": 1}


def test_recent_batch_replaces_recent_seed_hour() -> None:
    """A fetched recent value beats the older seed value."""
    silver = build_silver(seed(), CITY, [batch(SourceKind.RECENT, OLDER, 3, 1, 7.0)])

    assert at_hour(silver.frame, 3)["temperature_2m"] == 7.0


def test_recent_never_replaces_archive_seed_hour() -> None:
    """A recent batch cannot overwrite archive data."""
    silver = build_silver(seed(), CITY, [batch(SourceKind.RECENT, NEWER, 0, 1, 7.0)])

    row = at_hour(silver.frame, 0)
    assert row["temperature_2m"] == 1.0
    assert row["source"] == "archive-source"


def test_older_archive_beats_newer_recent() -> None:
    """Source priority comes before fetch time."""
    batches = [
        batch(SourceKind.ARCHIVE, OLDER, 3, 1, 9.0),
        batch(SourceKind.RECENT, NEWER, 3, 1, 7.0),
    ]

    silver = build_silver(seed(), CITY, batches)

    assert at_hour(silver.frame, 3)["temperature_2m"] == 9.0


@pytest.mark.parametrize("kind", list(SourceKind))
def test_newest_fetch_wins_within_source(kind: SourceKind) -> None:
    """The newest batch wins for one source."""
    batches = [batch(kind, NEWER, 3, 1, 8.0), batch(kind, OLDER, 3, 1, 6.0)]

    silver = build_silver(seed(), CITY, batches)

    assert at_hour(silver.frame, 3)["temperature_2m"] == 8.0


def test_batches_extend_the_timeline() -> None:
    """New hours join the seed in one sorted timeline."""
    silver = build_silver(seed(), CITY, [batch(SourceKind.RECENT, OLDER, 4, 2, 7.0)])

    assert len(silver.frame) == 6
    assert silver.frame["time"].is_monotonic_increasing
    assert silver.frame["time"].is_unique
    assert silver.report.batches_used == 1


def test_gap_between_sources_stays_visible() -> None:
    """A gap after the seed becomes an explicit null row."""
    silver = build_silver(seed(), CITY, [batch(SourceKind.RECENT, OLDER, 6, 1, 7.0)])

    assert silver.report.missing_hours == 2
    assert pd.isna(at_hour(silver.frame, 4)["temperature_2m"])
    assert at_hour(silver.frame, 4)["source"] == "recent-source"


def test_batch_order_does_not_change_result() -> None:
    """Rebuilding gives the same data in any file order."""
    batches = [
        batch(SourceKind.RECENT, OLDER, 3, 2, 6.0),
        batch(SourceKind.RECENT, NEWER, 3, 2, 8.0),
        batch(SourceKind.ARCHIVE, OLDER, 2, 2, 9.0),
    ]

    first = build_silver(seed(), CITY, batches)
    second = build_silver(seed(), CITY, list(reversed(batches)))

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.report == second.report


def test_merge_leaves_batches_unchanged() -> None:
    """Merging does not alter loaded batch data."""
    original = batch(SourceKind.RECENT, OLDER, 3, 1, 7.0)
    before = original.hourly.copy(deep=True)

    build_silver(seed(), CITY, [original])

    pd.testing.assert_frame_equal(original.hourly, before)
