from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Forward log return over `horizon` bars."""
    close = close.astype(float)
    return np.log(close.shift(-horizon) / close)


def forward_simple_return(close: pd.Series, horizon: int) -> pd.Series:
    """Forward simple return over `horizon` bars."""
    close = close.astype(float)
    return close.shift(-horizon) / close - 1.0


@dataclass(frozen=True)
class PerpFundingSpec:
    """How to convert a funding rate series into a per-bar funding cost.

    Most exchanges quote fundingRate per funding interval (often 8h).

    This implementation starts with a pragmatic approximation:
    - treat funding rate as applying uniformly over its interval
    - convert to per-minute and then to per-bar

    This is good enough to start the AlphaGen loop; we can refine once we
    lock in the data granularity and exact exchange semantics.
    """

    funding_interval_minutes: int = 480  # 8h


def approx_funding_cost_per_bar(
    funding_rate: pd.Series,
    *,
    bar_minutes: int,
    spec: PerpFundingSpec = PerpFundingSpec(),
) -> pd.Series:
    """Approximate funding cost for a long position per bar.

    Positive funding_rate means longs pay shorts (cost for long).
    Returns a per-bar additive term (same unit as a simple return).
    """

    fr = pd.to_numeric(funding_rate, errors="coerce").astype(float)
    per_min = fr / float(spec.funding_interval_minutes)
    return per_min * float(bar_minutes)


def forward_perp_net_simple_return(
    close: pd.Series,
    *,
    horizon: int,
    funding_rate: Optional[pd.Series] = None,
    bar_minutes: int = 1,
    funding_spec: PerpFundingSpec = PerpFundingSpec(),
) -> pd.Series:
    """Forward net return for a long perp position.

    net_return ≈ price_forward_return - sum(funding_cost_per_bar)

    If `funding_rate` is None, this reduces to forward_simple_return.
    """

    price_ret = forward_simple_return(close, horizon=horizon)
    if funding_rate is None:
        return price_ret

    cost_per_bar = approx_funding_cost_per_bar(
        funding_rate, bar_minutes=bar_minutes, spec=funding_spec
    )
    # Sum over the forward window: t .. t+h-1
    funding_cost = cost_per_bar.rolling(window=horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    return price_ret - funding_cost
