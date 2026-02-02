from __future__ import annotations

import argparse
import os
import sys

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from alphagen_style.dsl import eval_expr
from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig, compute_reward, evaluate_factor_panel
from alphagen_style.masking import ActionSpace, sample_expression


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample mask-valid expressions and score them on a panel.")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--n", type=int, default=20, help="How many expressions to sample")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embargo-bars", type=int, default=60)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

    # Choose feature columns only (exclude obvious targets)
    exclude_prefix = ("ret_",)
    exclude_cols = {"timestamp"}
    cols = [c for c in df.columns if c not in exclude_cols and not c.startswith(exclude_prefix)]

    rng = np.random.default_rng(args.seed)
    space = ActionSpace(columns=cols)

    cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=args.n_folds, embargo_bars=args.embargo_bars),
        regime=RegimeConfig(),
        ic_freq="1D",
        ic_method="spearman",
    )

    targets = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        targets = ["ret_fwd_log", "ret_fwd_net_perp"]

    scored = []
    for _ in range(args.n):
        expr, _acts = sample_expression(space, max_depth=args.max_depth, rng=rng)
        try:
            factor = eval_expr(expr, df)
        except Exception as e:
            scored.append((float("-inf"), expr, f"eval_error:{e}"))
            continue

        tmp = df.copy()
        tmp["alpha"] = factor
        res = evaluate_factor_panel(tmp, factor_col="alpha", target_cols=targets, cfg=cfg, trade_cfg=RewardConfig())
        r = compute_reward(res, cfg=RewardConfig())
        scored.append((float(r["reward"]), expr, "ok"))

    scored.sort(key=lambda x: x[0], reverse=True)
    print("Top results:")
    for s, expr, status in scored[: min(10, len(scored))]:
        print(f"- reward={s:.4f} status={status} expr={expr}")


if __name__ == "__main__":
    main()
