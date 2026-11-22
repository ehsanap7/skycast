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

CITY_COLUMN = "city"
SOURCE_COLUMN = "source"
SILVER_DATA_FILE = "hourly.parquet"
QUALITY_REPORT_FILE = "quality_report.json"

MEASUREMENT_RANGES: dict[str, tuple[float, float]] = {
    "temperature_2m": (-90.0, 60.0),
    "relative_humidity_2m": (0.0, 100.0),
    "dew_point_2m": (-100.0, 60.0),
    "precipitation": (0.0, 400.0),
    "pressure_msl": (850.0, 1100.0),
    "cloud_cover": (0.0, 100.0),
    "wind_speed_10m": (0.0, 120.0),
    "wind_direction_10m": (0.0, 360.0),
}

SILVER_COLUMNS: tuple[str, ...] = (
    CITY_COLUMN,
    SOURCE_COLUMN,
    TIME_COLUMN,
    *MEASUREMENT_RANGES,
)

DATE_COLUMN = "date"
GOLD_CONTRACT_VERSION = "1.0.0"
GOLD_DATA_FILE = "training.parquet"
GOLD_METADATA_FILE = "metadata.json"

INTERPOLATED_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
)
