from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .json_array_stream import iter_json_objects_from_array_file


@dataclass(frozen=True)
class SpotMergedMeta:
    symbol: str
    exchange: str
    timeframe: str


def load_spot_merged_ohlcv(
    path: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load spot OHLCV data into a DataFrame.

    Supported formats:
    - *_merged.json: array-of-objects (streamed)
    - *.parquet (e.g. ohlcv_merged.parquet): DataFrame with columns including
      timestamp/open/high/low/close/volume

    Expected keys/columns per row: timestamp/open/high/low/close/volume.
    timestamp is milliseconds (if seconds are detected, it will be upscaled).

    `start_ts`/`end_ts` are inclusive bounds in milliseconds.
    """

    def _coerce_ts_to_ms(ts_val: object) -> Optional[int]:
        try:
            # pandas Timestamp
            if isinstance(ts_val, pd.Timestamp):
                if pd.isna(ts_val):
                    return None
                return int(ts_val.timestamp() * 1000)
            # python datetime-like
            if hasattr(ts_val, "timestamp"):
                return int(float(ts_val.timestamp()) * 1000)

            n = int(float(ts_val))
        except Exception:
            return None

        # Heuristic: seconds vs milliseconds
        if n < 1_000_000_000_000:
            n *= 1000
        return n

    def _finalize(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return df_in
        if "timestamp" not in df_in.columns:
            return pd.DataFrame()

        ts_ms = df_in["timestamp"].map(_coerce_ts_to_ms)
        df = df_in.copy()
        df["timestamp"] = pd.to_datetime(ts_ms, unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    p = str(path or "").strip()
    if p.lower().endswith(".parquet"):
        try:
            df = pd.read_parquet(p)
        except Exception:
            # Likely missing parquet engine (pyarrow/fastparquet) or file corrupted
            return pd.DataFrame()

        if df.empty:
            return df

        # Apply bounds before full conversion when possible
        if "timestamp" in df.columns:
            ts_ms = df["timestamp"].map(_coerce_ts_to_ms)
            if start_ts is not None:
                df = df[ts_ms >= int(start_ts)]
                ts_ms = ts_ms.loc[df.index]
            if end_ts is not None:
                df = df[ts_ms <= int(end_ts)]
            if max_rows is not None:
                df = df.iloc[: int(max_rows)]
        else:
            # If no timestamp column, can't proceed
            return pd.DataFrame()

        return _finalize(df)

    rows = []
    for obj in iter_json_objects_from_array_file(path, max_objects=None):
        ts = obj.get("timestamp")
        ts_ms = _coerce_ts_to_ms(ts)
        if ts_ms is None:
            continue
        if start_ts is not None and ts_ms < int(start_ts):
            continue
        if end_ts is not None and ts_ms > int(end_ts):
            continue

        rows.append(
            {
                "timestamp": ts_ms,
                "open": obj.get("open"),
                "high": obj.get("high"),
                "low": obj.get("low"),
                "close": obj.get("close"),
                "volume": obj.get("volume"),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break

    return _finalize(pd.DataFrame(rows))
