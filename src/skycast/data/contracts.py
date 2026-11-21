"""Data contracts shared by the data layers."""

TIME_COLUMN = "time"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%z"
UTC_SUFFIX = "+00:00"

BRONZE_COLUMNS: tuple[str, ...] = (
    TIME_COLUMN,
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
)
