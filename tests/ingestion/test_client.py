"""Tests for the weather API client, using fake HTTP responses."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd
import pytest

from skycast.config import City, IngestionSettings
from skycast.data.contracts import MEASUREMENT_RANGES
from skycast.errors import DataSourceError, TemporarySourceError
from skycast.ingestion.client import WeatherClient, WeatherResponse

CITY = City(
    slug="sample",
    name="Sample",
    latitude=1.5,
    longitude=-2.5,
    timezone="UTC",
    source_file="sample.csv",
    archive_source="archive-source",
    recent_source="recent-source",
    archive_cutoff=datetime(2024, 1, 1, tzinfo=UTC),
)
SETTINGS = IngestionSettings(
    archive_url="https://archive.example.test/v1/archive",
    archive_model="sample-model",
    recent_url="https://recent.example.test/v1/forecast",
    recent_forecast_days=1,
    max_past_days=92,
    timeout_seconds=5,
)
Handler = Callable[[httpx.Request], httpx.Response]


def payload() -> dict[str, Any]:
    """Build a valid two-hour API response."""
    hourly: dict[str, list[Any]] = {"time": ["2024-01-01T00:00", "2024-01-01T01:00"]}
    hourly |= {column: [1.5, 2.5] for column in MEASUREMENT_RANGES}
    return {
        "utc_offset_seconds": 0,
        "hourly_units": {"wind_speed_10m": "m/s"},
        "hourly": hourly,
    }


def client_with(handler: Handler) -> WeatherClient:
    """Create a client backed by a fake transport."""
    return WeatherClient(SETTINGS, httpx.Client(transport=httpx.MockTransport(handler)))


def replying(body: dict[str, Any] | list[Any]) -> Handler:
    """Build a handler returning a JSON body."""
    return lambda _request: httpx.Response(200, json=body)


def fetch(handler: Handler) -> WeatherResponse:
    """Fetch recent data through a handler."""
    return client_with(handler).fetch_recent(CITY, 1)


def test_archive_request_parameters() -> None:
    """Archive calls use the configured URL, model, dates, and units."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and reply."""
        requests.append(request)
        return httpx.Response(200, json=payload())

    client_with(handler).fetch_archive(CITY, date(2024, 1, 1), date(2024, 1, 2))

    params = requests[0].url.params
    assert requests[0].url.host == "archive.example.test"
    assert params["start_date"] == "2024-01-01"
    assert params["end_date"] == "2024-01-02"
    assert params["models"] == "sample-model"
    assert params["wind_speed_unit"] == "ms"
    assert params["timezone"] == "GMT"
    assert params["hourly"].split(",") == list(MEASUREMENT_RANGES)


def test_recent_request_parameters() -> None:
    """Recent calls use the configured URL and lookback."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and reply."""
        requests.append(request)
        return httpx.Response(200, json=payload())

    client_with(handler).fetch_recent(CITY, 7)

    params = requests[0].url.params
    assert requests[0].url.host == "recent.example.test"
    assert params["past_days"] == "7"
    assert params["forecast_days"] == "1"
    assert params["latitude"] == "1.5"


def test_response_keeps_raw_bytes_and_parses_rows() -> None:
    """The exact body is kept next to typed rows."""
    body = json.dumps(payload()).encode()

    response = fetch(lambda _request: httpx.Response(200, content=body))

    assert response.raw == body
    assert len(response.hourly) == 2
    assert response.hourly.loc[0, "time"] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert response.hourly.loc[1, "temperature_2m"] == 2.5


def test_null_values_are_kept() -> None:
    """Delayed hours stay null for the next layer to handle."""
    body = payload()
    body["hourly"]["temperature_2m"][1] = None

    response = fetch(replying(body))

    assert pd.isna(response.hourly.loc[1, "temperature_2m"])


def test_api_error_includes_reason() -> None:
    """A documented API error keeps its reason."""
    body = {"error": True, "reason": "Invalid variable"}

    with pytest.raises(DataSourceError, match="400: Invalid variable"):
        fetch(lambda _request: httpx.Response(400, json=body))


def test_server_error_with_text_body() -> None:
    """A non-JSON failure is still a domain error."""
    with pytest.raises(DataSourceError, match="503: Service down"):
        fetch(lambda _request: httpx.Response(503, text="Service down"))


def test_network_failure_is_wrapped() -> None:
    """Timeouts become domain errors with the cause attached."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Simulate a timeout."""
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(DataSourceError, match="request failed") as caught:
        fetch(handler)
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_is_temporary(status: int) -> None:
    """Rate limits and server errors may succeed later."""
    with pytest.raises(TemporarySourceError, match=str(status)):
        fetch(lambda _request: httpx.Response(status, text="Try later"))


@pytest.mark.parametrize("status", [400, 404])
def test_client_error_is_not_temporary(status: int) -> None:
    """A bad request cannot be fixed by retrying."""
    with pytest.raises(DataSourceError) as caught:
        fetch(lambda _request: httpx.Response(status, text="Bad request"))
    assert not isinstance(caught.value, TemporarySourceError)


def test_network_failure_is_temporary() -> None:
    """A dropped connection may succeed later."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Simulate a dropped connection."""
        raise httpx.ConnectError("connection failed")

    with pytest.raises(TemporarySourceError, match="request failed"):
        fetch(handler)


def test_invalid_json_raises() -> None:
    """A broken body is rejected."""
    with pytest.raises(DataSourceError, match="invalid JSON"):
        fetch(lambda _request: httpx.Response(200, content=b"not json"))


def test_non_object_raises() -> None:
    """The response must be a single JSON object."""
    with pytest.raises(DataSourceError, match="JSON object"):
        fetch(replying([payload()]))


def without_hourly(body: dict[str, Any]) -> None:
    """Remove the hourly block."""
    del body["hourly"]


def without_variable(body: dict[str, Any]) -> None:
    """Remove one weather variable."""
    del body["hourly"]["pressure_msl"]


def with_uneven_lengths(body: dict[str, Any]) -> None:
    """Shorten one column."""
    body["hourly"]["cloud_cover"].pop()


def with_empty_columns(body: dict[str, Any]) -> None:
    """Empty every column."""
    body["hourly"] = {column: [] for column in body["hourly"]}


def with_bad_time(body: dict[str, Any]) -> None:
    """Corrupt one timestamp."""
    body["hourly"]["time"][1] = "not-a-time"


def with_bad_value(body: dict[str, Any]) -> None:
    """Corrupt one measurement."""
    body["hourly"]["cloud_cover"][0] = "cloudy"


def with_local_time(body: dict[str, Any]) -> None:
    """Report a non-UTC response."""
    body["utc_offset_seconds"] = -18000


def with_wrong_wind_unit(body: dict[str, Any]) -> None:
    """Report wind speed in another unit."""
    body["hourly_units"]["wind_speed_10m"] = "km/h"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (without_hourly, "no hourly data"),
        (without_variable, "missing: pressure_msl"),
        (with_uneven_lengths, "empty or uneven"),
        (with_empty_columns, "empty or uneven"),
        (with_bad_time, "invalid timestamps"),
        (with_bad_value, "non-numeric values in cloud_cover"),
        (with_local_time, "not in UTC"),
        (with_wrong_wind_unit, "must be in m/s"),
    ],
)
def test_invalid_response_raises(corrupt: Callable[[dict[str, Any]], None], message: str) -> None:
    """Responses that break the contract are rejected."""
    body = payload()
    corrupt(body)

    with pytest.raises(DataSourceError, match=message):
        fetch(replying(body))


def test_reversed_date_range_makes_no_request() -> None:
    """A bad date range fails before calling the API."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Fail if a request is sent."""
        raise AssertionError("request should not be sent")

    with pytest.raises(DataSourceError, match="Invalid archive range"):
        client_with(handler).fetch_archive(CITY, date(2024, 1, 2), date(2024, 1, 1))


@pytest.mark.parametrize("past_days", [-1, 93])
def test_invalid_past_days_makes_no_request(past_days: int) -> None:
    """A lookback outside the API limit fails before calling it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        """Fail if a request is sent."""
        raise AssertionError("request should not be sent")

    with pytest.raises(DataSourceError, match="past_days"):
        client_with(handler).fetch_recent(CITY, past_days)
