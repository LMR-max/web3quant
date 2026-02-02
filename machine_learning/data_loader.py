from __future__ import annotations

from pathlib import Path
import pandas as pd


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_integer_dtype(series):
        # assume milliseconds
        return pd.to_datetime(series, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_parquet(path: Path, time_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if time_col in df.columns:
        df[time_col] = _to_datetime(df[time_col])
        df = df.sort_values(time_col)
        df = df.set_index(time_col)
    else:
        df = df.sort_index()
    return df
