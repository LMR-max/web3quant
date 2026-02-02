from __future__ import annotations

import argparse
import json
import os
import sys


# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import pandas as pd

from alphagen_style.dsl import analyze_expr, eval_expr
from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig, compute_reward, evaluate_factor_panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a DSL expression directly on a panel and output reward JSON.")
    ap.add_argument("--panel", required=True, help="Panel CSV")
    ap.add_argument("--expr", required=True, help="DSL expression, e.g. zscore(ts_delta(close,1),60)")
    ap.add_argument("--factor-name", default="alpha", help="Name for computed factor column")

    ap.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target cols. Default: ret_fwd_log,ret_fwd_net_perp if present else ret_fwd_log",
    )

    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--embargo-bars", type=int, default=60)

    ap.add_argument("--ic-freq", default="1D")
    ap.add_argument("--ic-method", choices=["spearman", "pearson"], default="spearman")

    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--funding-abs-q", type=float, default=0.8)

    ap.add_argument("--reward-primary", default=None)
    ap.add_argument("--reward-mode", choices=["ic", "trade", "hybrid"], default="ic")
    ap.add_argument("--reward-max-missing", type=float, default=0.05)
    ap.add_argument("--reward-min-n", type=int, default=8000)

    ap.add_argument("--use-fold-median-ic", action="store_true")
    ap.add_argument("--w-fold-median-ic", type=float, default=0.5)

    ap.add_argument("--w-degenerate", type=float, default=0.3)
    ap.add_argument("--min-unique-ratio", type=float, default=0.002)
    ap.add_argument("--min-factor-std", type=float, default=1e-6)

    ap.add_argument("--trade-z-thr", type=float, default=0.8)
    ap.add_argument("--trade-base-fee", type=float, default=0.0005)
    ap.add_argument("--trade-impact-coef", type=float, default=0.02)
    ap.add_argument("--trade-dd-thr", type=float, default=-0.05)
    ap.add_argument("--trade-dd-penalty", type=float, default=2.0)
    ap.add_argument("--trade-min-activity", type=int, default=5)

    ap.add_argument("--out", default=None, help="Output JSON path")

    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

    info = analyze_expr(args.expr)

    # Allow running this file directly (without `-m`) from any cwd.
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    # compute factor
    df[args.factor_name] = eval_expr(args.expr, df)

    # Drop warmup bars to avoid penalizing unavoidable rolling NaNs.
    df_eval = df.iloc[info.warmup_bars :].copy()

    if args.targets:
        target_cols = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        if "ret_fwd_net_perp" in df.columns:
            target_cols = ["ret_fwd_log", "ret_fwd_net_perp"]
        else:
            target_cols = ["ret_fwd_log"]

    cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=args.n_folds, embargo_bars=args.embargo_bars),
        regime=RegimeConfig(vol_window=args.vol_window, funding_abs_q=args.funding_abs_q),
        ic_freq=args.ic_freq,
        ic_method=args.ic_method,
    )

    reward_cfg = RewardConfig(
        primary_target=args.reward_primary,
        mode=args.reward_mode,
        max_missing_rate=args.reward_max_missing,
        min_n=args.reward_min_n,
        use_fold_median_ic=bool(args.use_fold_median_ic),
        w_fold_median_ic=float(args.w_fold_median_ic),
        w_degenerate=float(args.w_degenerate),
        min_unique_ratio=float(args.min_unique_ratio),
        min_factor_std=float(args.min_factor_std),
        trade_z_thr=float(args.trade_z_thr),
        trade_base_fee=float(args.trade_base_fee),
        trade_impact_coef=float(args.trade_impact_coef),
        trade_dd_thr=float(args.trade_dd_thr),
        trade_dd_penalty=float(args.trade_dd_penalty),
        trade_min_activity=int(args.trade_min_activity),
    )

    result = evaluate_factor_panel(
        df_eval,
        factor_col=args.factor_name,
        target_cols=target_cols,
        cfg=cfg,
        trade_cfg=reward_cfg,
    )
    reward_out = compute_reward(result, cfg=reward_cfg)
    result["reward"] = reward_out
    result["expr"] = args.expr
    result["expr_info"] = {
        "columns": info.columns,
        "functions": info.functions,
        "max_window": info.max_window,
        "max_shift": info.max_shift,
        "warmup_bars": info.warmup_bars,
    }

    out_path = args.out

    # Allow running this file directly (without `-m`) from any cwd.
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    if out_path is None:
        base = os.path.splitext(os.path.basename(args.panel))[0]
        safe_name = args.factor_name
        out_path = os.path.join(os.getcwd(), f"evalexpr_{base}__{safe_name}.json")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Reward: {result['reward']['reward']:.6f}")


if __name__ == "__main__":
    main()
