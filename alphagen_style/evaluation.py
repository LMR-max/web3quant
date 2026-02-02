from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd

try:
    # Pandas spearman corr may emit this via scipy.stats.spearmanr
    from scipy.stats import ConstantInputWarning  # type: ignore

    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except Exception:
    # SciPy may be missing in some environments; keep evaluator functional.
    pass


CorrMethod = Literal["pearson", "spearman"]
RewardMode = Literal["ic", "trade", "hybrid"]


@dataclass(frozen=True)
class TimeSplitConfig:
    n_folds: int = 5
    embargo_bars: int = 0


@dataclass(frozen=True)
class RegimeConfig:
    vol_window: int = 60
    vol_q: Tuple[float, float] = (0.33, 0.66)
    funding_abs_q: float = 0.8


def _safe_series(x: pd.Series) -> pd.Series:
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return x.replace([np.inf, -np.inf], np.nan)


def rank_ic(factor: pd.Series, target: pd.Series) -> float:
    """Spearman correlation over time for a single instrument."""
    factor = _safe_series(factor)
    target = _safe_series(target)
    df = pd.concat([factor.rename("f"), target.rename("y")], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    if df["f"].nunique(dropna=True) < 2 or df["y"].nunique(dropna=True) < 2:
        return float("nan")
    return float(df["f"].corr(df["y"], method="spearman"))


def linear_ic(factor: pd.Series, target: pd.Series) -> float:
    factor = _safe_series(factor)
    target = _safe_series(target)
    df = pd.concat([factor.rename("f"), target.rename("y")], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    if df["f"].nunique(dropna=True) < 2 or df["y"].nunique(dropna=True) < 2:
        return float("nan")
    return float(df["f"].corr(df["y"], method="pearson"))


def ic_series(
    factor: pd.Series,
    target: pd.Series,
    *,
    freq: str = "1D",
    method: CorrMethod = "spearman",
) -> pd.Series:
    """IC time series by resampling into bins (e.g. daily IC)."""
    factor = _safe_series(factor)
    target = _safe_series(target)

    df = pd.concat([factor.rename("f"), target.rename("y")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 20:
            return float("nan")
        if g["f"].nunique(dropna=True) < 2 or g["y"].nunique(dropna=True) < 2:
            return float("nan")
        return float(g["f"].corr(g["y"], method=method))

    out = df.groupby(pd.Grouper(freq=freq)).apply(_corr)
    out.name = "ic"
    return out.dropna()


def ic_ir(ic: pd.Series) -> float:
    ic = _safe_series(ic).dropna()
    if ic.empty:
        return float("nan")
    sd = float(ic.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(ic.mean() / sd)


def time_folds(index: pd.Index, cfg: TimeSplitConfig) -> List[Tuple[pd.Index, str]]:
    """Return list of (test_index, fold_name) in chronological order.

    This evaluator is factor-only, so we treat each fold as a test window.
    `embargo_bars` drops last N bars of each fold to reduce leakage.
    """
    if cfg.n_folds <= 1:
        return [(index, "fold_0")]

    n = len(index)
    fold_size = max(1, n // cfg.n_folds)
    folds: List[Tuple[pd.Index, str]] = []
    for i in range(cfg.n_folds):
        start = i * fold_size
        end = n if i == cfg.n_folds - 1 else (i + 1) * fold_size
        sub = index[start:end]
        if cfg.embargo_bars > 0 and len(sub) > cfg.embargo_bars:
            sub = sub[:-cfg.embargo_bars]
        folds.append((sub, f"fold_{i}"))
    return folds


def label_regimes(
    df: pd.DataFrame,
    *,
    cfg: RegimeConfig,
    close_col: str = "close",
    funding_col: str = "funding_rate",
) -> pd.Series:
    """Regime label combining volatility bucket and funding state.

    - Volatility: rolling std of 1-bar returns, bucketed by quantiles.
    - Funding: sign plus an "extreme" tag by abs quantile.

    Output examples:
    - vol=low|fund=pos
    - vol=high|fund=neg_ext
    - vol=mid|fund=na
    """

    # Be robust: evaluator may be called on short/empty slices after warmup.
    if df.empty:
        out = pd.Series(dtype=str, index=df.index, name="regime")
        return out

    if close_col in df.columns:
        try:
            ret_1 = pd.to_numeric(df[close_col], errors="coerce").pct_change(1)
        except Exception:
            ret_1 = pd.Series(np.nan, index=df.index)
    else:
        ret_1 = pd.Series(np.nan, index=df.index)
    vol = ret_1.rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 3)).std()
    q1, q2 = cfg.vol_q
    v1 = float(vol.quantile(q1)) if vol.notna().any() else float("nan")
    v2 = float(vol.quantile(q2)) if vol.notna().any() else float("nan")

    def vol_bucket(x: float) -> str:
        if not np.isfinite(x):
            return "na"
        if np.isfinite(v1) and x <= v1:
            return "low"
        if np.isfinite(v2) and x <= v2:
            return "mid"
        return "high"

    vol_b = vol.apply(vol_bucket)

    if funding_col in df.columns:
        fr = pd.to_numeric(df[funding_col], errors="coerce")
        abs_thr = float(fr.abs().quantile(cfg.funding_abs_q)) if fr.notna().any() else float("nan")

        def fund_bucket(x: float) -> str:
            if not np.isfinite(x):
                return "na"
            sign = "pos" if x > 0 else ("neg" if x < 0 else "zero")
            if np.isfinite(abs_thr) and abs(x) >= abs_thr and sign in ("pos", "neg"):
                return f"{sign}_ext"
            return sign

        fund_b = fr.apply(fund_bucket)
    else:
        fund_b = pd.Series("na", index=df.index)

    label = "vol=" + vol_b.astype(str) + "|fund=" + fund_b.astype(str)
    label.name = "regime"
    return label


@dataclass(frozen=True)
class EvalConfig:
    time: TimeSplitConfig = TimeSplitConfig()
    regime: RegimeConfig = RegimeConfig()
    ic_freq: str = "1D"
    ic_method: CorrMethod = "spearman"


@dataclass(frozen=True)
class RewardConfig:
    """Single scalar reward aggregation config.

    The intent is: reward high, stable, consistent IC with low missingness.
    """

    primary_target: Optional[str] = None
    use_abs_ic: bool = True

    # Reward mode
    # - ic: original IC-based objective
    # - trade: simplified backtest score (position + costs + drawdown penalty)
    # - hybrid: combine both
    mode: RewardMode = "ic"

    w_ic: float = 1.0
    w_ic_ir: float = 0.2

    # Optional robust IC aggregation (median of per-fold ICs)
    use_fold_median_ic: bool = False
    w_fold_median_ic: float = 0.5

    # penalties
    w_fold_instability: float = 0.5
    w_regime_instability: float = 0.3
    w_inconsistency: float = 0.6
    w_missing: float = 1.0
    w_low_n: float = 0.5

    # degenerate / trivial factor penalties
    w_degenerate: float = 0.3
    min_unique_ratio: float = 0.002  # nunique / n
    min_factor_std: float = 1e-6

    # exposure / transaction-cost proxies (optional, based on columns in panel)
    w_exposure: float = 0.2
    w_autocorr: float = 0.1
    w_turnover: float = 0.1

    # trade-sim (optional)
    w_trade: float = 1.0
    trade_z_thr: float = 0.8
    trade_base_fee: float = 0.0005  # one-way fee proxy
    trade_impact_coef: float = 0.02
    trade_dd_thr: float = -0.05
    trade_dd_penalty: float = 2.0
    trade_min_activity: int = 5

    exposure_cols: Tuple[str, ...] = ("ret_1", "vol_60", "volume")

    # thresholds (scaled into penalty terms)
    exposure_corr_thr: float = 0.10
    autocorr_thr: float = 0.80
    turnover_thr: float = 0.80

    # thresholds
    max_missing_rate: float = 0.05
    min_n: int = 8000


def _missing_rate(s: pd.Series) -> float:
    s = _safe_series(s)
    if len(s) == 0:
        return 1.0
    return float(s.isna().mean())


def _finite_values(xs: Iterable[object]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if isinstance(x, (int, float)) and np.isfinite(x):
            out.append(float(x))
    return out


def _safe_corr(a: pd.Series, b: pd.Series, *, method: str = "pearson") -> float:
    a = _safe_series(pd.to_numeric(a, errors="coerce"))
    b = _safe_series(pd.to_numeric(b, errors="coerce"))
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 50:
        return float("nan")
    if df["a"].nunique(dropna=True) < 2 or df["b"].nunique(dropna=True) < 2:
        return float("nan")
    try:
        return float(df["a"].corr(df["b"], method=method))
    except Exception:
        return float("nan")


def _zscore_series(x: pd.Series) -> pd.Series:
    x = _safe_series(pd.to_numeric(x, errors="coerce"))
    if not x.notna().any():
        return x * np.nan
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    mu = float(x.mean())
    return (x - mu) / sd


def _trade_sim_metrics(
    df: pd.DataFrame,
    *,
    factor_col: str,
    target_col: str,
    volume_col: str = "volume",
    cfg: RewardConfig,
) -> Dict[str, float]:
    """A lightweight trade simulation on a single time-series.

    Inspired by AlphaGPT: turn factor into position, subtract costs, penalize drawdowns and low activity.
    This is intentionally simple and optional.
    """

    f = _safe_series(pd.to_numeric(df[factor_col], errors="coerce"))
    y = _safe_series(pd.to_numeric(df[target_col], errors="coerce"))
    dd_thr = float(cfg.trade_dd_thr)

    d = pd.concat([f.rename("f"), y.rename("y")], axis=1).dropna()
    if d.empty:
        return {
            "score": float("nan"),
            "net_ret_sum": float("nan"),
            "net_ret_mean": float("nan"),
            "net_ret_std": float("nan"),
            "sharpe": float("nan"),
            "activity": 0.0,
            "turnover_mean": float("nan"),
            "dd_count": float("nan"),
        }

    fz = _zscore_series(d["f"])
    if not fz.notna().any():
        return {
            "score": float("nan"),
            "net_ret_sum": float("nan"),
            "net_ret_mean": float("nan"),
            "net_ret_std": float("nan"),
            "sharpe": float("nan"),
            "activity": 0.0,
            "turnover_mean": float("nan"),
            "dd_count": float("nan"),
        }

    z_thr = float(cfg.trade_z_thr)
    pos = pd.Series(0.0, index=d.index)
    pos = pos.mask(fz > z_thr, 1.0)
    pos = pos.mask(fz < -z_thr, -1.0)

    turnover = (pos - pos.shift(1).fillna(0.0)).abs()

    # crude impact proxy from volume if available
    if volume_col in df.columns:
        v = _safe_series(pd.to_numeric(df.loc[d.index, volume_col], errors="coerce")).fillna(0.0)
        impact = float(cfg.trade_impact_coef) / np.sqrt(v + 1.0)
        impact = impact.clip(lower=0.0, upper=0.05)
    else:
        impact = pd.Series(0.0, index=d.index)

    fee = float(cfg.trade_base_fee)
    cost = turnover * (fee + impact)
    gross = pos * d["y"]
    net = gross - cost

    net_sum = float(net.sum())
    net_mean = float(net.mean())
    net_std = float(net.std(ddof=1)) if net.notna().sum() >= 2 else float("nan")
    sharpe = float(net_mean / (net_std + 1e-9) * np.sqrt(max(net.notna().sum(), 1))) if np.isfinite(net_mean) else float("nan")

    dd_count = float((net < dd_thr).sum())
    activity = float(pos.abs().sum())

    # score: cumulative net return minus drawdown events penalty
    # Optimization: Non-linear drawdown penalty + Turnover penalty
    dd_penalty_score = float(cfg.trade_dd_penalty) * (dd_count ** 1.2) # Non-linear penalty
    
    # Turnover penalty (discourage flickering beyond just fee costs)
    to_penalty = float(turnover.mean()) * 2.0 
    
    score = net_sum - dd_penalty_score - to_penalty
    
    if int(cfg.trade_min_activity) > 0 and activity < int(cfg.trade_min_activity):
        score = min(score, -10.0)

    return {
        "score": float(score),
        "net_ret_sum": float(net_sum),
        "net_ret_mean": float(net_mean),
        "net_ret_std": float(net_std) if np.isfinite(net_std) else float("nan"),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
        "activity": float(activity),
        "turnover_mean": float(turnover.mean()),
        "dd_count": float(dd_count),
    }


def compute_reward(
    eval_result: Dict[str, object],
    *,
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, object]:
    """Aggregate evaluation output into a single scalar reward.

    Returns dict with:
    - reward: float
    - components: dict
    """

    summary = eval_result.get("summary") or {}
    targets: List[str] = list(eval_result.get("targets") or [])

    if cfg.primary_target is not None and cfg.primary_target in targets:
        primary = cfg.primary_target
    else:
        primary = "ret_fwd_net_perp" if "ret_fwd_net_perp" in targets else (targets[0] if targets else None)

    def _get_metric(t: str, k: str) -> float:
        try:
            v = summary[t][k]  # type: ignore[index]
        except Exception:
            return float("nan")
        return float(v) if isinstance(v, (int, float)) else float("nan")

    ic = _get_metric(primary, "ic") if primary else float("nan")
    ic_ir_v = _get_metric(primary, "ic_ir") if primary else float("nan")
    n = _get_metric(primary, "n") if primary else float("nan")

    if not np.isfinite(ic):
        ic = 0.0
    if not np.isfinite(ic_ir_v):
        ic_ir_v = 0.0
    if not np.isfinite(n):
        n = 0.0

    ic_term = abs(ic) if cfg.use_abs_ic else ic
    base_ic = cfg.w_ic * ic_term + cfg.w_ic_ir * ic_ir_v

    # fold instability
    per_fold = eval_result.get("per_fold") or []
    fold_ics = _finite_values(
        (r.get(f"{primary}__ic") for r in per_fold) if primary else []  # type: ignore[union-attr]
    )
    fold_std = float(np.std(fold_ics, ddof=1)) if len(fold_ics) >= 2 else 0.0
    fold_mean = float(np.mean(fold_ics)) if fold_ics else 0.0
    fold_instability = fold_std / (abs(fold_mean) + 1e-9) if fold_ics else 1.0
    fold_instability = float(np.clip(fold_instability, 0.0, 5.0))

    fold_median_ic = float(np.median(fold_ics)) if fold_ics else float("nan")
    fold_median_term = 0.0
    if cfg.use_fold_median_ic and np.isfinite(fold_median_ic):
        fold_median_term = cfg.w_fold_median_ic * (abs(fold_median_ic) if cfg.use_abs_ic else fold_median_ic)

    # regime instability
    per_regime = eval_result.get("per_regime") or []
    reg_ics = _finite_values(
        (r.get(f"{primary}__ic") for r in per_regime) if primary else []  # type: ignore[union-attr]
    )
    reg_std = float(np.std(reg_ics, ddof=1)) if len(reg_ics) >= 2 else 0.0
    reg_mean = float(np.mean(reg_ics)) if reg_ics else 0.0
    regime_instability = reg_std / (abs(reg_mean) + 1e-9) if reg_ics else 1.0
    regime_instability = float(np.clip(regime_instability, 0.0, 5.0))

    # target consistency penalty (if present)
    incons = 0.0
    consistency = eval_result.get("consistency") or {}
    if consistency:
        fsmr = consistency.get("fold_sign_match_rate")
        rsmr = consistency.get("regime_sign_match_rate")
        parts = []
        if isinstance(fsmr, (int, float)) and np.isfinite(fsmr):
            parts.append(1.0 - float(fsmr))
        if isinstance(rsmr, (int, float)) and np.isfinite(rsmr):
            parts.append(1.0 - float(rsmr))
        incons = float(np.mean(parts)) if parts else 0.0

    # missingness penalty
    miss_factor = float(eval_result.get("missing_factor_rate") or 0.0)
    miss_target = float(eval_result.get("missing_target_rate") or 0.0)
    miss = max(miss_factor, miss_target)
    miss_pen = float(np.clip(miss / max(cfg.max_missing_rate, 1e-9), 0.0, 5.0))

    # low sample penalty
    low_n_pen = 0.0
    if cfg.min_n > 0:
        low_n_pen = float(np.clip((cfg.min_n - n) / cfg.min_n, 0.0, 1.0))

    # exposure / transaction-cost proxy penalties
    exposures = eval_result.get("exposures") or {}
    if not isinstance(exposures, dict):
        exposures = {}

    exp_abs_corrs: List[float] = []
    for col in cfg.exposure_cols:
        v = exposures.get(f"corr_{col}")
        if isinstance(v, (int, float)) and np.isfinite(v):
            exp_abs_corrs.append(abs(float(v)))

    exposure_abs_corr_mean = float(np.mean(exp_abs_corrs)) if exp_abs_corrs else float("nan")
    if np.isfinite(exposure_abs_corr_mean) and cfg.exposure_corr_thr > 0:
        exposure_pen = float(np.clip(exposure_abs_corr_mean / cfg.exposure_corr_thr, 0.0, 5.0))
    else:
        exposure_pen = 0.0

    ac1 = exposures.get("autocorr_lag1")
    autocorr_abs = float(abs(ac1)) if isinstance(ac1, (int, float)) and np.isfinite(ac1) else float("nan")
    if np.isfinite(autocorr_abs) and cfg.autocorr_thr > 0:
        autocorr_pen = float(np.clip(autocorr_abs / cfg.autocorr_thr, 0.0, 5.0))
    else:
        autocorr_pen = 0.0

    to = exposures.get("turnover")
    turnover_v = float(to) if isinstance(to, (int, float)) and np.isfinite(to) else float("nan")
    if np.isfinite(turnover_v) and cfg.turnover_thr > 0:
        turnover_pen = float(np.clip(turnover_v / cfg.turnover_thr, 0.0, 5.0))
    else:
        turnover_pen = 0.0

    # degenerate / trivial factor penalty
    # (scale-free) unique ratio + (scale-dependent) std floor
    unique_ratio = eval_result.get("factor_unique_ratio")
    factor_std = eval_result.get("factor_std")
    u = float(unique_ratio) if isinstance(unique_ratio, (int, float)) and np.isfinite(unique_ratio) else float("nan")
    sd = float(factor_std) if isinstance(factor_std, (int, float)) and np.isfinite(factor_std) else float("nan")

    unique_pen = 0.0
    if np.isfinite(u) and cfg.min_unique_ratio > 0:
        unique_pen = float(np.clip((cfg.min_unique_ratio - u) / cfg.min_unique_ratio, 0.0, 5.0))

    std_pen = 0.0
    if np.isfinite(sd) and cfg.min_factor_std > 0:
        std_pen = float(np.clip((cfg.min_factor_std - sd) / cfg.min_factor_std, 0.0, 5.0))

    degenerate_pen = max(unique_pen, std_pen)

    # trade-sim score (optional)
    trade_sim = eval_result.get("trade_sim") or {}
    trade_score = float("nan")
    if isinstance(trade_sim, dict) and primary and isinstance(trade_sim.get(primary), dict):
        ts = trade_sim.get(primary)  # type: ignore[assignment]
        if isinstance(ts, dict):
            v = ts.get("score")
            if isinstance(v, (int, float)) and np.isfinite(v):
                trade_score = float(v)

    base_trade = cfg.w_trade * float(trade_score) if np.isfinite(trade_score) else 0.0

    if cfg.mode == "trade":
        base = base_trade
    elif cfg.mode == "hybrid":
        base = base_ic + base_trade + fold_median_term
    else:
        base = base_ic + fold_median_term

    penalty = (
        cfg.w_fold_instability * fold_instability
        + cfg.w_regime_instability * regime_instability
        + cfg.w_inconsistency * incons
        + cfg.w_missing * miss_pen
        + cfg.w_low_n * low_n_pen
        + cfg.w_exposure * exposure_pen
        + cfg.w_autocorr * autocorr_pen
        + cfg.w_turnover * turnover_pen
        + cfg.w_degenerate * degenerate_pen
    )

    reward = float(base - penalty)
    return {
        "reward": reward,
        "components": {
            "primary_target": primary,
            "mode": cfg.mode,
            "base": float(base),
            "base_ic": float(base_ic),
            "base_trade": float(base_trade),
            "fold_median_ic": float(fold_median_ic) if np.isfinite(fold_median_ic) else float("nan"),
            "fold_median_term": float(fold_median_term),
            "ic": float(ic),
            "ic_ir": float(ic_ir_v),
            "n": float(n),
            "fold_std": float(fold_std),
            "fold_instability": float(fold_instability),
            "regime_std": float(reg_std),
            "regime_instability": float(regime_instability),
            "inconsistency": float(incons),
            "missing_factor_rate": float(miss_factor),
            "missing_target_rate": float(miss_target),
            "missing_pen": float(miss_pen),
            "low_n_pen": float(low_n_pen),
            "exposure_abs_corr_mean": float(exposure_abs_corr_mean)
            if np.isfinite(exposure_abs_corr_mean)
            else float("nan"),
            "exposure_pen": float(exposure_pen),
            "autocorr_lag1": float(ac1) if isinstance(ac1, (int, float)) and np.isfinite(ac1) else float("nan"),
            "autocorr_pen": float(autocorr_pen),
            "turnover": float(turnover_v) if np.isfinite(turnover_v) else float("nan"),
            "turnover_pen": float(turnover_pen),
            "factor_unique_ratio": float(u) if np.isfinite(u) else float("nan"),
            "factor_std": float(sd) if np.isfinite(sd) else float("nan"),
            "degenerate_pen": float(degenerate_pen),
            "trade_score": float(trade_score) if np.isfinite(trade_score) else float("nan"),
            "penalty": float(penalty),
        },
    }


def evaluate_factor_panel(
    panel: pd.DataFrame,
    *,
    factor_col: str,
    target_cols: Sequence[str],
    cfg: EvalConfig = EvalConfig(),
    trade_cfg: Optional[RewardConfig] = None,
) -> Dict[str, object]:
    """Evaluate a factor column on one instrument time-series panel.

    Returns a dict with:
    - summary metrics per target
    - per-fold metrics
    - per-regime metrics
    - consistency between targets
    """

    if factor_col not in panel.columns:
        raise ValueError(f"Missing factor column: {factor_col}")

    for tcol in target_cols:
        if tcol not in panel.columns:
            raise ValueError(f"Missing target column: {tcol}")

    # Ensure datetime index
    df = panel.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        else:
            raise ValueError("Panel must have DatetimeIndex or a timestamp column")

    df = df.sort_index()

    regimes = label_regimes(df, cfg=cfg.regime)

    factor_s = pd.to_numeric(df[factor_col], errors="coerce")
    n_non_nan = int(_safe_series(factor_s).notna().sum())
    nunique = int(_safe_series(factor_s).nunique(dropna=True))
    factor_unique_ratio = float(nunique / max(n_non_nan, 1))
    factor_std = float(_safe_series(factor_s).std(ddof=1)) if n_non_nan >= 2 else float("nan")
    missing_factor_rate = _missing_rate(factor_s)
    missing_target_rate = float(
        max(_missing_rate(pd.to_numeric(df[tcol], errors="coerce")) for tcol in target_cols)
    )

    # Exposure / cost proxies (optional, computed on full sample)
    exposures: Dict[str, float] = {}
    try:
        exposures["autocorr_lag1"] = float(_safe_corr(factor_s, factor_s.shift(1), method="pearson"))

        fz = _zscore_series(factor_s)
        exposures["turnover"] = float(_safe_series(fz.diff()).abs().mean()) if fz.notna().any() else float("nan")
    except Exception:
        exposures["autocorr_lag1"] = float("nan")
        exposures["turnover"] = float("nan")

    for col in ("ret_1", "vol_60", "volume"):
        if col in df.columns:
            exposures[f"corr_{col}"] = float(_safe_corr(factor_s, df[col], method="pearson"))

    # Trade-sim metrics (optional but cheap). Always computed; reward decides whether to use.
    trade_sim: Dict[str, Dict[str, float]] = {}
    tcfg = trade_cfg or RewardConfig()
    for tcol in target_cols:
        try:
            trade_sim[tcol] = _trade_sim_metrics(df, factor_col=factor_col, target_col=tcol, cfg=tcfg)
        except Exception:
            trade_sim[tcol] = {
                "score": float("nan"),
                "net_ret_sum": float("nan"),
                "net_ret_mean": float("nan"),
                "net_ret_std": float("nan"),
                "sharpe": float("nan"),
                "activity": float("nan"),
                "turnover_mean": float("nan"),
                "dd_count": float("nan"),
            }

    folds = time_folds(df.index, cfg.time)

    def eval_slice(slice_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        f = pd.to_numeric(slice_df[factor_col], errors="coerce")
        for tcol in target_cols:
            y = pd.to_numeric(slice_df[tcol], errors="coerce")
            ic_ts = ic_series(f, y, freq=cfg.ic_freq, method=cfg.ic_method)
            out[tcol] = {
                "ic": float(rank_ic(f, y) if cfg.ic_method == "spearman" else linear_ic(f, y)),
                "ic_mean": float(ic_ts.mean()) if not ic_ts.empty else float("nan"),
                "ic_ir": float(ic_ir(ic_ts)),
                "n": float(pd.concat([f, y], axis=1).dropna().shape[0]),
            }
        return out

    per_fold_rows: List[Dict[str, object]] = []
    for test_idx, name in folds:
        s = df.loc[test_idx]
        metrics = eval_slice(s)
        row: Dict[str, object] = {
            "fold": name,
            "start": str(s.index.min()) if not s.empty else None,
            "end": str(s.index.max()) if not s.empty else None,
            "rows": int(len(s)),
        }
        for tcol, m in metrics.items():
            for k, v in m.items():
                row[f"{tcol}__{k}"] = v
        per_fold_rows.append(row)

    per_regime_rows: List[Dict[str, object]] = []
    df2 = df.join(regimes)
    for reg, s in df2.groupby("regime"):
        metrics = eval_slice(s)
        row = {"regime": reg, "rows": int(len(s))}
        for tcol, m in metrics.items():
            for k, v in m.items():
                row[f"{tcol}__{k}"] = v
        per_regime_rows.append(row)

    # Summary: whole sample
    summary = eval_slice(df)

    # Consistency across targets: compare IC signs per fold & per regime
    consistency: Dict[str, object] = {}
    if len(target_cols) >= 2:
        t0, t1 = target_cols[0], target_cols[1]

        def _sign(x: float) -> int:
            if not np.isfinite(x) or x == 0:
                return 0
            return 1 if x > 0 else -1

        fold_sign_match = []
        for r in per_fold_rows:
            a = r.get(f"{t0}__ic")
            b = r.get(f"{t1}__ic")
            if isinstance(a, float) and isinstance(b, float) and np.isfinite(a) and np.isfinite(b):
                fold_sign_match.append(int(_sign(a) == _sign(b)))

        regime_sign_match = []
        for r in per_regime_rows:
            a = r.get(f"{t0}__ic")
            b = r.get(f"{t1}__ic")
            if isinstance(a, float) and isinstance(b, float) and np.isfinite(a) and np.isfinite(b):
                regime_sign_match.append(int(_sign(a) == _sign(b)))

        consistency = {
            "targets": [t0, t1],
            "fold_sign_match_rate": float(np.mean(fold_sign_match)) if fold_sign_match else float("nan"),
            "regime_sign_match_rate": float(np.mean(regime_sign_match)) if regime_sign_match else float("nan"),
            "ic_gap_abs": float(abs(summary[t0]["ic"] - summary[t1]["ic"]))
            if np.isfinite(summary[t0]["ic"]) and np.isfinite(summary[t1]["ic"])
            else float("nan"),
        }

    return {
        "factor_col": factor_col,
        "targets": list(target_cols),
        "config": {
            "n_folds": cfg.time.n_folds,
            "embargo_bars": cfg.time.embargo_bars,
            "ic_freq": cfg.ic_freq,
            "ic_method": cfg.ic_method,
            "vol_window": cfg.regime.vol_window,
            "vol_q": cfg.regime.vol_q,
            "funding_abs_q": cfg.regime.funding_abs_q,
        },
        "missing_factor_rate": missing_factor_rate,
        "missing_target_rate": missing_target_rate,
        "factor_unique_ratio": float(factor_unique_ratio),
        "factor_std": float(factor_std) if np.isfinite(factor_std) else float("nan"),
        "exposures": exposures,
        "trade_sim": trade_sim,
        "summary": summary,
        "per_fold": per_fold_rows,
        "per_regime": per_regime_rows,
        "consistency": consistency,
    }
