from __future__ import annotations

import argparse
import os
import sys

# Workaround for Windows OpenMP runtime conflicts (MKL / PyTorch / SB3)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig
from alphagen_style.gym_env import GymMaskedAlphaEnv, GymObsConfig
from alphagen_style.masking import ActionSpace
from alphagen_style.rl_env import EnvConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MaskablePPO on masked alpha-expression generation (BTC/USDT).")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--timesteps", type=int, default=2000)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--panel-rows", type=int, default=8000, help="Use head N rows for faster iterations")

    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--embargo-bars", type=int, default=60)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="Model output path (.zip)")

    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device for training: auto|cpu|cuda (or cuda:0, etc.)",
    )

    # reward knobs (keep defaults aligned with RewardConfig)
    ap.add_argument("--reward-mode", choices=["ic", "trade", "hybrid"], default="ic")
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

    args = ap.parse_args()

    device = str(args.device or "auto").strip().lower()
    if device in ("gpu",):
        device = "cuda"
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                raise SystemExit(
                    "Requested CUDA device but torch.cuda.is_available() is False. "
                    "Install a CUDA-enabled PyTorch build and verify NVIDIA driver/CUDA runtime."
                )
        except ImportError:
            raise SystemExit("Requested CUDA device but PyTorch is not installed")

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")

    df = df.sort_index()
    if args.panel_rows and args.panel_rows > 0:
        df = df.iloc[: args.panel_rows].copy()

    # Feature columns for DSL
    exclude_prefix = ("ret_",)
    cols = [c for c in df.columns if c != "timestamp" and not c.startswith(exclude_prefix)]

    targets = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        targets = ["ret_fwd_log", "ret_fwd_net_perp"]

    space = ActionSpace(columns=cols)

    eval_cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=args.n_folds, embargo_bars=args.embargo_bars),
        regime=RegimeConfig(),
        ic_freq="1D",
        ic_method="spearman",
    )

    reward_cfg = RewardConfig(
        mode=str(args.reward_mode),
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

    env = GymMaskedAlphaEnv(
        panel=df,
        action_space=space,
        targets=targets,
        env_cfg=EnvConfig(max_depth=args.max_depth, n_folds=args.n_folds, embargo_bars=args.embargo_bars),
        eval_cfg=eval_cfg,
        reward_cfg=reward_cfg,
        obs_cfg=GymObsConfig(max_steps=64),
    )

    def mask_fn(_env):
        return _env.get_action_mask()

    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        device=device,
    )

    model.learn(total_timesteps=int(args.timesteps))

    out = args.out
    if out is None:
        out = os.path.join(os.getcwd(), f"maskable_ppo_btcusdt_depth{args.max_depth}_{args.timesteps}.zip")
    model.save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
