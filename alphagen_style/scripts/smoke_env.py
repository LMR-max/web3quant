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

from alphagen_style.masking import ActionSpace, sample_actions, actions_to_expr
from alphagen_style.rl_env import EnvConfig, MaskedAlphaEnv
from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test for masked RL env (no gym dependency).")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

    exclude_prefix = ("ret_",)
    cols = [c for c in df.columns if not c.startswith(exclude_prefix) and c != "timestamp"]

    targets = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        targets = ["ret_fwd_log", "ret_fwd_net_perp"]

    space = ActionSpace(columns=cols)

    eval_cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=5, embargo_bars=60),
        regime=RegimeConfig(),
        ic_freq="1D",
        ic_method="spearman",
    )

    env = MaskedAlphaEnv(
        panel=df,
        action_space=space,
        targets=targets,
        env_cfg=EnvConfig(max_depth=args.max_depth),
        eval_cfg=eval_cfg,
        reward_cfg=RewardConfig(),
    )

    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        steps = 0
        while not done:
            mask = info["action_mask"]
            idxs = np.flatnonzero(mask)
            a = int(rng.choice(idxs))
            obs, r, done, info = env.step(a)
            steps += 1
        print(f"ep={ep} steps={steps} reward={r:.4f} expr={info.get('expr')}")


if __name__ == "__main__":
    main()
