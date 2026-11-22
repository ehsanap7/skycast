"""A scripted fake of the weather API for ingestion tests."""

from typing import Any

import httpx
import pandas as pd

from skycast.data.contracts import MEASUREMENT_RANGES


def _api_payload() -> dict[str, Any]:
    """Build six API hours, the last two after the fake current time."""
    times = pd.date_range("2024-01-11 00:00", periods=6, freq="h")
    hourly: dict[str, list[Any]] = {"time": [f"{time:%Y-%m-%dT%H:%M}" for time in times]}
    hourly |= {column: [15.0] * 6 for column in MEASUREMENT_RANGES}
    return {"utc_offset_seconds": 0, "hourly_units": {"wind_speed_10m": "m/s"}, "hourly": hourly}


class FakeApi:
    """A fake weather API with scripted failures."""

    def __init__(self) -> None:
        """Start with no calls and no failures."""
        self.calls = 0
        self.failures: list[int] = []

    def handle(self, _request: httpx.Request) -> httpx.Response:
        """Return the next scripted failure or a valid response."""
        self.calls += 1
        if self.failures:
            body = {"error": True, "reason": "simulated"}
            return httpx.Response(self.failures.pop(0), json=body)
        return httpx.Response(200, json=_api_payload())
