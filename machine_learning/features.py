from __future__ import annotations

import numpy as np
import pandas as pd
from .config import Config


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    open_ = df["open"] if "open" in df.columns else None
    high = df["high"]
    low = df["low"]
    volume = df.get("volume")

    # returns
    features["ret_1"] = close.pct_change()
    features["log_ret_1"] = np.log(close).diff()

    # momentum & rolling stats
    for lb in cfg.features_lookbacks:
        features[f"mom_{lb}"] = close.pct_change(lb)
        features[f"ma_{lb}"] = close.rolling(lb).mean()
        ma = features[f"ma_{lb}"]
        features[f"ma_dist_{lb}"] = (close - ma) / ma.replace(0, np.nan)
        features[f"vol_{lb}"] = features["ret_1"].rolling(lb).std()
        z_mean = close.rolling(lb).mean()
        z_std = close.rolling(lb).std().replace(0, np.nan)
        features[f"zscore_{lb}"] = (close - z_mean) / z_std
        features[f"ret_skew_{lb}"] = features["ret_1"].rolling(lb).skew()
        features[f"ret_kurt_{lb}"] = features["ret_1"].rolling(lb).kurt()

    # lagged returns
    for lag in cfg.return_lags:
        features[f"ret_lag_{lag}"] = features["ret_1"].shift(lag)

    # range / volatility
    features["range_amp"] = (high - low) / close.replace(0, np.nan)
    features["atr_14"] = (high - low).rolling(14).mean() / close.replace(0, np.nan)
    features["rsi_14"] = _rsi(close, 14)

    # candlestick features
    if open_ is not None:
        candle_range = (high - low).replace(0, np.nan)
        body = (close - open_)
        features["candle_body"] = body
        features["candle_range"] = candle_range
        features["candle_body_pct"] = body / candle_range
        features["upper_shadow"] = (high - close).clip(lower=0)
        features["lower_shadow"] = (close - low).clip(lower=0)
        features["shadow_ratio"] = (features["upper_shadow"] - features["lower_shadow"]) / candle_range

    # volume features
    if volume is not None:
        features["vol_chg_1"] = volume.pct_change()
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std().replace(0, np.nan)
        features["vol_z_20"] = (volume - vol_mean) / vol_std
        for lb in cfg.volume_lookbacks:
            features[f"vol_ema_{lb}"] = volume.ewm(span=lb, adjust=False).mean()
            features[f"vol_mom_{lb}"] = volume.pct_change(lb)

    # vwap-related
    if "vwap" in df.columns:
        vwap = df["vwap"].replace(0, np.nan)
        features["vwap_dist"] = (close - df["vwap"]) / vwap

    # drawdown features
    for lb in cfg.drawdown_lookbacks:
        rolling_max = close.rolling(lb).max()
        features[f"drawdown_{lb}"] = close / rolling_max - 1.0

    return features
