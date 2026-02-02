from __future__ import annotations

import numpy as np
import pandas as pd
from .config import Config


def make_labels(features: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    # regression target: future return
    future_ret = features["ret_1"].shift(-cfg.horizon_steps)

    # classification target: up / down / flat (interval)
    rolling_vol = features["ret_1"].rolling(cfg.rolling_vol_window).std()
    threshold = cfg.target_threshold_k * rolling_vol

    direction = np.where(future_ret > threshold, 1, np.where(future_ret < -threshold, -1, 0))
    direction = pd.Series(direction, index=features.index, name="direction")

    return future_ret.rename("future_ret"), direction
