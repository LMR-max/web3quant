from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    pred_scores: pd.Series,
    future_returns: pd.Series,
    n_quantiles: int = 5,
    cost_bp: float = 5.0,
) -> dict:
    data = pd.concat([pred_scores, future_returns], axis=1).dropna()
    data.columns = ["score", "ret"]

    data["group"] = pd.qcut(data["score"], q=n_quantiles, labels=False, duplicates="drop")

    long_mask = data["group"] == data["group"].max()
    short_mask = data["group"] == data["group"].min()

    long_ret = data.loc[long_mask, "ret"].mean()
    short_ret = data.loc[short_mask, "ret"].mean()

    # simple long-short return per period (cost applied on both sides)
    cost = (cost_bp / 10000.0) * 2
    ls_ret = long_ret - short_ret - cost

    # equity curve (per period)
    ls_series = data.loc[long_mask, "ret"].groupby(level=0).mean() - data.loc[short_mask, "ret"].groupby(level=0).mean()
    ls_series = ls_series.reindex(data.index).fillna(0.0) - cost
    equity = (1 + ls_series).cumprod()

    # per-quantile stability stats
    data = data.copy()
    data["ts"] = data.index
    group_ret = data.groupby(["ts", "group"])["ret"].mean().unstack("group")
    group_stats = {}
    for g in group_ret.columns:
        s = group_ret[g].dropna()
        if s.empty:
            continue
        mean = float(s.mean())
        std = float(s.std())
        n = len(s)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        group_stats[int(g)] = {
            "mean": mean,
            "std": std,
            "tstat": tstat,
            "win_rate": float((s > 0).mean()),
            "n": int(n),
        }

    ls_std = float(ls_series.std())
    ls_tstat = float(ls_series.mean() / (ls_std / np.sqrt(len(ls_series)))) if ls_std > 0 else 0.0
    ls_win_rate = float((ls_series > 0).mean())

    ann_factor = 365 * 24  # for 1h data
    ann_return = (1 + ls_ret) ** ann_factor - 1
    ann_vol = np.sqrt(ann_factor) * data["ret"].std()
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    return {
        "long_ret": long_ret,
        "short_ret": short_ret,
        "ls_ret": ls_ret,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "equity": equity,
        "group_stats": group_stats,
        "ls_tstat": ls_tstat,
        "ls_win_rate": ls_win_rate,
    }
