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

from alphagen_style.evaluation import (
    EvalConfig,
    RegimeConfig,
    RewardConfig,
    TimeSplitConfig,
    compute_reward,
    evaluate_factor_panel,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a factor column on a panel CSV (strong eval skeleton).")
    ap.add_argument("--panel", required=True, help="Panel CSV produced by build_spot_panel")
    ap.add_argument("--factor-col", required=True, help="Column name to evaluate (factor values)")
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

    ap.add_argument("--out", default=None, help="Output JSON path")

    ap.add_argument("--reward-primary", default=None, help="Primary target for reward (default auto)")
    ap.add_argument("--reward-max-missing", type=float, default=0.05)
    ap.add_argument("--reward-min-n", type=int, default=8000)

    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    # If the panel has a timestamp column (from csv), use it.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

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

    result = evaluate_factor_panel(df, factor_col=args.factor_col, target_cols=target_cols, cfg=cfg, trade_cfg=RewardConfig())

    r_cfg = RewardConfig(
        primary_target=args.reward_primary,
        max_missing_rate=args.reward_max_missing,
        min_n=args.reward_min_n,
    )
    reward_out = compute_reward(result, cfg=r_cfg)
    result["reward"] = reward_out

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.panel))[0]
        out_path = os.path.join(os.getcwd(), f"eval_{base}__{args.factor_col}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Compact console summary
    print(f"Wrote: {out_path}")
    print(f"Reward: {result['reward']['reward']:.6f}")
    print("Summary:")
    for t, m in result["summary"].items():
        ic = m.get("ic")
        ic_ir = m.get("ic_ir")
        n = m.get("n")
        print(f"- {t}: ic={ic:.4f} ic_ir={ic_ir:.3f} n={int(n) if n is not None else 'NA'}")

    if result.get("consistency"):
        c = result["consistency"]
        print("Consistency:")
        print(
            f"- fold_sign_match_rate={c.get('fold_sign_match_rate'):.3f} regime_sign_match_rate={c.get('regime_sign_match_rate'):.3f} ic_gap_abs={c.get('ic_gap_abs'):.4f}"
        )


if __name__ == "__main__":
    main()
