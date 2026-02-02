from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings

# Workaround for Windows OpenMP runtime conflicts (MKL / PyTorch / SB3)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reduce noise from correlation computations on near-constant series.
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

from alphagen_style.dsl import analyze_expr, eval_expr
from alphagen_style.evaluation import EvalConfig, RegimeConfig, RewardConfig, TimeSplitConfig, compute_reward, evaluate_factor_panel
from alphagen_style.gym_env import GymMaskedAlphaEnv, GymObsConfig
from alphagen_style.masking import ActionSpace
from alphagen_style.rl_env import EnvConfig


def _compute_factor_series(
    df: pd.DataFrame,
    expr: str,
    *,
    warmup_bars: int,
) -> pd.Series:
    """Compute factor series for correlation-based diversity filtering."""
    if warmup_bars >= max(len(df) - 5, 0):
        return pd.Series(dtype=float)

    tmp = df.copy()
    try:
        tmp["alpha"] = eval_expr(expr, tmp)
    except Exception:
        # Invalid DSL params or runtime issues should not crash the whole export.
        # Returning empty will cause the caller to skip this expression.
        return pd.Series(dtype=float)
    s = pd.to_numeric(tmp["alpha"], errors="coerce").iloc[warmup_bars:]
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "alpha"
    return s


def _expr_structure_key(expr: str) -> str:
    """Structure key for de-duplicating expressions by operator tree.

    - Function names and operator types are preserved.
    - Column identifiers become COL0/COL1/... in first-seen order.
    - Numeric constants are preserved (rounded for floats).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return f"PARSE_ERROR:{expr}"

    col_map: Dict[str, int] = {}

    def col_id(name: str) -> str:
        if name not in col_map:
            col_map[name] = len(col_map)
        return f"COL{col_map[name]}"

    def fmt_const(v: object) -> str:
        if isinstance(v, bool) or v is None:
            return "NA"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if np.isfinite(v):
                return f"{v:.4g}"
            return "NaN"
        return "K"

    def visit(n: ast.AST) -> str:
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Name):
            return col_id(n.id)
        if isinstance(n, ast.Constant):
            return fmt_const(n.value)
        if isinstance(n, ast.Call):
            fn = "CALL"
            if isinstance(n.func, ast.Name):
                fn = n.func.id
            args = ",".join(visit(a) for a in n.args)
            return f"{fn}({args})"
        if isinstance(n, ast.BinOp):
            op = type(n.op).__name__
            return f"{op}({visit(n.left)},{visit(n.right)})"
        if isinstance(n, ast.UnaryOp):
            op = type(n.op).__name__
            return f"{op}({visit(n.operand)})"
        return type(n).__name__

    return visit(tree)


def _family_key_from_expr_info(expr_info: Dict[str, object], *, mode: str) -> str:
    cols = expr_info.get("columns") or []
    if isinstance(cols, str):
        cols_list = [c for c in cols.split(",") if c]
    elif isinstance(cols, list):
        cols_list = [str(c) for c in cols]
    else:
        cols_list = []

    cols_list = [c.strip() for c in cols_list if c and str(c).strip()]
    if not cols_list:
        return "<no_cols>"

    if mode == "primary":
        return cols_list[0]

    # mode == "set"
    return "|".join(sorted(set(cols_list)))


def _select_diverse_topk(
    df: pd.DataFrame,
    ranked_candidates: List[Tuple[str, float]],
    *,
    details: Dict[str, Dict[str, object]],
    topk: int,
    corr_thr: float,
    corr_method: str,
    min_overlap: int,
    candidate_multiplier: int,
    drop_error: bool,
    family_mode: str,
    max_per_family: int,
    structure_dedupe: bool,
) -> List[Tuple[str, float, float, str, str]]:
    """Greedy Top-K selection with correlation-based diversity filtering.

    Returns list of (expr, reward, max_abs_corr_to_selected, struct_key, family_key).
    """
    if topk <= 0:
        return []

    max_candidates = min(
        len(ranked_candidates),
        max(topk * max(int(candidate_multiplier), 1), topk),
    )
    pool = ranked_candidates[:max_candidates]

    series_cache: Dict[str, pd.Series] = {}
    seen_struct: set[str] = set()
    family_counts: Dict[str, int] = {}
    selected: List[Tuple[str, float, float, str, str]] = []
    selected_series: List[pd.Series] = []

    for expr, r in pool:
        d = details.get(expr) or {}
        if drop_error:
            rc = d.get("reward_components") or {}
            if isinstance(rc, dict) and rc.get("error"):
                continue

        expr_info = d.get("expr_info") or {}
        if not isinstance(expr_info, dict):
            expr_info = {}

        # Structure de-duplication
        struct_key = _expr_structure_key(expr)
        if structure_dedupe and struct_key in seen_struct:
            continue

        # Column-family quota
        family_key = _family_key_from_expr_info(expr_info, mode=str(family_mode))
        if int(max_per_family) > 0:
            if family_counts.get(family_key, 0) >= int(max_per_family):
                continue

        warmup_bars = int(expr_info.get("warmup_bars") or 0)

        s = series_cache.get(expr)
        if s is None:
            s = _compute_factor_series(df, expr, warmup_bars=warmup_bars)
            series_cache[expr] = s

        if s.empty:
            continue

        max_abs_corr = 0.0
        ok = True
        for s2 in selected_series:
            common = s.index.intersection(s2.index)
            if len(common) < int(min_overlap):
                ok = False
                break
            c = float(s.loc[common].corr(s2.loc[common], method=corr_method))
            if np.isfinite(c):
                max_abs_corr = max(max_abs_corr, abs(c))
                if abs(c) >= float(corr_thr):
                    ok = False
                    break

        if not ok:
            continue

        if structure_dedupe:
            seen_struct.add(struct_key)
        family_counts[family_key] = int(family_counts.get(family_key, 0)) + 1

        selected.append((expr, float(r), float(max_abs_corr), struct_key, family_key))
        selected_series.append(s)

        if len(selected) >= topk:
            break

    return selected


def _make_env(
    df: pd.DataFrame,
    *,
    max_depth: int,
    n_folds: int,
    embargo_bars: int,
) -> Tuple[ActionMasker, List[str], List[str]]:
    exclude_prefix = ("ret_",)
    cols = [c for c in df.columns if c != "timestamp" and not c.startswith(exclude_prefix)]

    targets = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        targets = ["ret_fwd_log", "ret_fwd_net_perp"]

    space = ActionSpace(columns=cols)

    eval_cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=n_folds, embargo_bars=embargo_bars),
        regime=RegimeConfig(),
        ic_freq="1D",
        ic_method="spearman",
    )

    env = GymMaskedAlphaEnv(
        panel=df,
        action_space=space,
        targets=targets,
        env_cfg=EnvConfig(max_depth=max_depth, n_folds=n_folds, embargo_bars=embargo_bars),
        eval_cfg=eval_cfg,
        reward_cfg=RewardConfig(),
        obs_cfg=GymObsConfig(max_steps=64),
    )

    def mask_fn(_env):
        return _env.get_action_mask()

    masked_env = ActionMasker(env, mask_fn)
    return masked_env, cols, targets


def _score_expr(
    df: pd.DataFrame,
    expr: str,
    *,
    targets: List[str],
    eval_cfg: EvalConfig,
    reward_cfg: RewardConfig,
) -> Dict[str, object]:
    info = analyze_expr(expr)

    if info.warmup_bars >= max(len(df) - 5, 0):
        return {
            "expr": expr,
            "reward": -10.0,
            "reward_components": {
                "primary_target": None,
                "ic": None,
                "ic_ir": None,
                "penalty": None,
                "error": "warmup_too_large",
            },
            "missing_factor_rate": float("nan"),
            "missing_target_rate": float("nan"),
            "summary": None,
            "consistency": None,
            "expr_info": {
                "columns": info.columns,
                "functions": info.functions,
                "max_window": info.max_window,
                "max_shift": info.max_shift,
                "warmup_bars": info.warmup_bars,
            },
        }

    try:
        tmp = df.copy()
        tmp["alpha"] = eval_expr(expr, tmp)

        # Drop warmup to avoid unavoidable rolling NaNs affecting missing penalties.
        tmp_eval = tmp.iloc[info.warmup_bars :].copy()

        res = evaluate_factor_panel(tmp_eval, factor_col="alpha", target_cols=targets, cfg=eval_cfg, trade_cfg=reward_cfg)
        r = compute_reward(res, cfg=reward_cfg)
    except Exception as e:
        return {
            "expr": expr,
            "reward": -10.0,
            "reward_components": {
                "primary_target": None,
                "ic": None,
                "ic_ir": None,
                "penalty": None,
                "error": f"exception: {type(e).__name__}: {e}",
            },
            "missing_factor_rate": float("nan"),
            "missing_target_rate": float("nan"),
            "summary": None,
            "consistency": None,
            "expr_info": {
                "columns": info.columns,
                "functions": info.functions,
                "max_window": info.max_window,
                "max_shift": info.max_shift,
                "warmup_bars": info.warmup_bars,
            },
        }

    out: Dict[str, object] = {
        "expr": expr,
        "reward": float(r["reward"]),
        "reward_components": r["components"],
        "missing_factor_rate": float(res.get("missing_factor_rate", 0.0)),
        "missing_target_rate": float(res.get("missing_target_rate", 0.0)),
        "summary": res.get("summary"),
        "consistency": res.get("consistency"),
        "expr_info": {
            "columns": info.columns,
            "functions": info.functions,
            "max_window": info.max_window,
            "max_shift": info.max_shift,
            "warmup_bars": info.warmup_bars,
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample expressions from a trained MaskablePPO policy and export top-K.")
    ap.add_argument("--model", required=True, help="Path to saved MaskablePPO .zip")
    ap.add_argument("--panel", required=True, help="Panel CSV")
    ap.add_argument("--episodes", type=int, default=200, help="How many episodes to sample")
    ap.add_argument("--topk", type=int, default=50)

    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--panel-rows", type=int, default=12000)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--embargo-bars", type=int, default=60)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions")

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

    ap.add_argument(
        "--diversity-corr-thr",
        type=float,
        default=0.9,
        help="Greedy correlation threshold for Top-K diversity filtering (abs(corr) < thr)",
    )
    ap.add_argument(
        "--diversity-corr-method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method for diversity filtering",
    )
    ap.add_argument(
        "--family-mode",
        choices=["set", "primary"],
        default="set",
        help="How to define a columns family: full set of columns or primary column",
    )
    ap.add_argument(
        "--max-per-family",
        type=int,
        default=3,
        help="Max selected expressions per columns family",
    )
    ap.add_argument(
        "--no-structure-dedupe",
        action="store_true",
        help="Disable structure de-duplication (operator-tree key)",
    )
    ap.add_argument(
        "--diversity-min-overlap",
        type=int,
        default=1000,
        help="Min overlapping non-NaN samples required to compute correlation",
    )
    ap.add_argument(
        "--candidate-multiplier",
        type=int,
        default=10,
        help="How many candidates to consider before diversity filtering (topk * multiplier)",
    )
    ap.add_argument(
        "--drop-error",
        action="store_true",
        help="Drop expressions that had evaluation errors from Top-K selection",
    )

    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)

    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    df = df.sort_index()

    if args.panel_rows and args.panel_rows > 0:
        df = df.iloc[: args.panel_rows].copy()

    env, _cols, targets = _make_env(
        df,
        max_depth=args.max_depth,
        n_folds=args.n_folds,
        embargo_bars=args.embargo_bars,
    )

    model = MaskablePPO.load(args.model)

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

    rng = np.random.default_rng(args.seed)

    seen: Dict[str, float] = {}
    details: Dict[str, Dict[str, object]] = {}

    for _ in range(int(args.episodes)):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        truncated = False
        last_info: Optional[dict] = None
        last_reward = 0.0

        while not (done or truncated):
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=bool(args.deterministic), action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            last_info = info
            last_reward = float(reward)

        expr = None
        if last_info is not None:
            expr = last_info.get("expr")
        if not expr:
            continue

        if expr in seen:
            continue

        # Full scoring with breakdown
        d = _score_expr(df, expr, targets=targets, eval_cfg=eval_cfg, reward_cfg=reward_cfg)
        seen[expr] = float(d["reward"])
        details[expr] = d

        if len(seen) >= max(args.topk * 5, 100):
            # enough unique candidates
            pass

    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)

    diverse_ranked = _select_diverse_topk(
        df,
        ranked,
        details=details,
        topk=int(args.topk),
        corr_thr=float(args.diversity_corr_thr),
        corr_method=str(args.diversity_corr_method),
        min_overlap=int(args.diversity_min_overlap),
        candidate_multiplier=int(args.candidate_multiplier),
        drop_error=bool(args.drop_error),
        family_mode=str(args.family_mode),
        max_per_family=int(args.max_per_family),
        structure_dedupe=(not bool(args.no_structure_dedupe)),
    )

    rows: List[Dict[str, object]] = []
    out_details: List[Dict[str, object]] = []
    for expr, r, max_abs_corr, struct_key, family_key in diverse_ranked:
        d = details[expr]
        d["diversity"] = {
            "corr_method": str(args.diversity_corr_method),
            "corr_thr": float(args.diversity_corr_thr),
            "min_overlap": int(args.diversity_min_overlap),
            "max_abs_corr_to_selected": float(max_abs_corr),
            "structure_dedupe": (not bool(args.no_structure_dedupe)),
            "structure_key": str(struct_key),
            "family_mode": str(args.family_mode),
            "family_key": str(family_key),
            "max_per_family": int(args.max_per_family),
        }
        rows.append(
            {
                "reward": float(r),
                "expr": expr,
                "max_abs_corr_to_selected": float(max_abs_corr),
                "family_key": str(family_key),
                "structure_key": str(struct_key),
                "primary_target": d["reward_components"].get("primary_target"),
                "ic": d["reward_components"].get("ic"),
                "ic_ir": d["reward_components"].get("ic_ir"),
                "penalty": d["reward_components"].get("penalty"),
                "missing_factor_rate": d.get("missing_factor_rate"),
                "missing_target_rate": d.get("missing_target_rate"),
                "max_window": d["expr_info"].get("max_window"),
                "functions": ",".join(d["expr_info"].get("functions") or []),
                "columns": ",".join(d["expr_info"].get("columns") or []),
            }
        )
        out_details.append(d)

    out_csv = args.out_csv
    out_json = args.out_json
    if out_csv is None:
        out_csv = os.path.join(os.getcwd(), "topk_expr.csv")
    if out_json is None:
        out_json = os.path.join(os.getcwd(), "topk_expr.json")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_details, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    if rows:
        print("Top 5:")
        for r in rows[:5]:
            print(
                f"- reward={r['reward']:.4f} corr={float(r.get('max_abs_corr_to_selected') or 0.0):.3f} expr={r['expr']}"
            )


if __name__ == "__main__":
    main()
