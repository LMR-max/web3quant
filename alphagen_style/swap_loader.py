from __future__ import annotations

import json
from typing import Optional

import pandas as pd


def load_funding_history_json(path: str) -> pd.DataFrame:
    """Load funding history saved by our fetch script.

    Expected: a JSON list of dicts with keys including:
    - funding_time (ISO string)
    - funding_rate
    - predicted_rate (optional)
    - interval_hours (optional)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "funding_time" in df.columns:
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, errors="coerce")
        df = df.sort_values("funding_time").set_index("funding_time")
    for col in ["funding_rate", "predicted_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_open_interest_history_json(path: str) -> pd.DataFrame:
    """Load open interest history saved by our fetch script.

    Expected: a JSON list of dicts with keys including:
    - timestamp (ISO string)
    - open_interest
    - open_interest_value
    - volume_24h (optional)
    - turnover_24h (optional)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp")
    for col in ["open_interest", "open_interest_value", "volume_24h", "turnover_24h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
