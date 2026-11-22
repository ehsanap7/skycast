"""Convert data frames to and from Parquet bytes."""

from io import BytesIO

import pandas as pd
import pyarrow as pa

from skycast.errors import DataSourceError


def to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a frame to Parquet bytes."""
    buffer = BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def read_parquet_bytes(content: bytes, key: str) -> pd.DataFrame:
    """Parse Parquet bytes into a frame."""
    try:
        return pd.read_parquet(BytesIO(content), engine="pyarrow")
    except pa.ArrowException as err:
        raise DataSourceError(f"Cannot parse Parquet dataset {key}: {err}") from err
