from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Workaround for Windows OpenMP runtime conflicts (MKL / NumPy)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import ast
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from alphagen_style.dsl import WINDOW_CANDIDATES, list_allowed_functions, analyze_expr, eval_expr
from alphagen_style.evaluation import (
    EvalConfig,
    RegimeConfig,
    RewardConfig,
    TimeSplitConfig,
    compute_reward,
    evaluate_factor_panel,
)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1200
    timeout_sec: int = 60


def _post_json(url: str, payload: Dict[str, object], *, headers: Dict[str, str], timeout: int) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code}: {e.reason}\n{body}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {type(e).__name__}: {e}") from e

    try:
        obj = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response: {e}\nRaw: {raw[:5000]}") from e
    if not isinstance(obj, dict):
        raise RuntimeError("Invalid response type (expected JSON object)")
    return obj


def deepseek_chat_completion(cfg: LLMConfig, *, system: str, user: str) -> str:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload: Dict[str, object] = {
        "model": cfg.model,
        "temperature": float(cfg.temperature),
        "max_tokens": int(cfg.max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    obj = _post_json(url, payload, headers=headers, timeout=int(cfg.timeout_sec))

    # OpenAI-compatible response shape
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Missing choices in response: keys={list(obj.keys())}")

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("Missing message in response choices[0]")
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Missing message.content in response")
    return content.strip()


def _parse_llm_exprs(text: str) -> List[str]:
    """Parse expressions from model output.

    Accepts either:
    - JSON list of strings
    - newline separated expressions
    - fenced code block containing either of the above
    """

    s = text.strip()

    # Strip a single fenced block if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()

    # Try JSON list
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = [str(x).strip() for x in obj if str(x).strip()]
            return out
    except Exception:
        pass

    # Fallback: line based
    out = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading bullets
        line = line.lstrip("-*• ")
        if not line:
            continue
        out.append(line)
    return out


def _expr_structure_key(expr: str) -> str:
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


def _family_key(cols: Sequence[str], *, mode: str) -> str:
    cols_list = [c.strip() for c in cols if c and str(c).strip()]
    if not cols_list:
        return "<no_cols>"
    if mode == "primary":
        return cols_list[0]
    return "|".join(sorted(set(cols_list)))


def _compact_json(obj: object, *, max_len: int = 6000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) <= int(max_len):
        return s
    return s[: int(max_len)] + "..."


def _build_reflection_prompt(
    *,
    n: int,
    allowed_funcs: List[str],
    window_choices: List[int],
    columns: List[str],
    themes: List[str],
    seed_summaries: List[Dict[str, object]],
) -> str:
    return (
        "You will improve alpha factor expressions based on evaluation feedback.\n"
        "Return ONLY a JSON array of strings (each string is one DSL expression).\n"
        "\n"
        "Constraints:\n"
        "- Use ONLY these function names: {funcs}.\n"
        "- Use ONLY these window sizes (for rolling funcs): {wins}.\n"
        "- Use ONLY these available columns: {cols}.\n"
        "- Avoid trivial raw columns (e.g., close, volume).\n"
        "- Avoid producing the same expression as any seed.\n"
        "- Prefer expressions with lower exposure/autocorr/turnover penalties (if hinted).\n"
        "- Try to incorporate these themes: {themes}.\n"
        "\n"
        "Here are seeds with feedback (reward, penalties, errors). Use them to learn what fails and what works, and propose {n} new expressions.\n"
        "SEEDS_JSON={seeds}\n"
    ).format(
        n=int(n),
        funcs=", ".join(allowed_funcs),
        wins=", ".join(str(w) for w in window_choices),
        cols=", ".join(columns),
        themes=", ".join(themes) if themes else "(none)",
        seeds=_compact_json(seed_summaries, max_len=10000),
    )


def _make_seed_summaries(
    ranked: List[Tuple[str, float]],
    *,
    details: Dict[str, Dict[str, object]],
    max_seeds: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for expr, reward in ranked[: int(max_seeds)]:
        d = details.get(expr) or {}
        rc = d.get("reward_components") or {}
        ex = d.get("exposures")
        ei = d.get("expr_info") or {}

        item = {
            "expr": expr,
            "reward": float(reward),
            "error": rc.get("error") if isinstance(rc, dict) else None,
            "ic": rc.get("ic") if isinstance(rc, dict) else None,
            "ic_ir": rc.get("ic_ir") if isinstance(rc, dict) else None,
            "penalty": rc.get("penalty") if isinstance(rc, dict) else None,
            "exposure_abs_corr_mean": rc.get("exposure_abs_corr_mean") if isinstance(rc, dict) else None,
            "autocorr_lag1": (ex or {}).get("autocorr_lag1") if isinstance(ex, dict) else None,
            "turnover": (ex or {}).get("turnover") if isinstance(ex, dict) else None,
            "columns": ei.get("columns"),
            "functions": ei.get("functions"),
            "max_window": ei.get("max_window"),
            "warmup_bars": ei.get("warmup_bars"),
        }
        out.append(item)
    return out


def _compute_factor_series(df: pd.DataFrame, expr: str, *, warmup_bars: int) -> pd.Series:
    if warmup_bars >= max(len(df) - 5, 0):
        return pd.Series(dtype=float)
    tmp = df.copy()
    tmp["alpha"] = eval_expr(expr, tmp)
    s = pd.to_numeric(tmp["alpha"], errors="coerce").iloc[warmup_bars:]
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "alpha"
    return s


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
            "reward_components": {"error": "warmup_too_large"},
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
        tmp_eval = tmp.iloc[info.warmup_bars :].copy()
        res = evaluate_factor_panel(tmp_eval, factor_col="alpha", target_cols=targets, cfg=eval_cfg, trade_cfg=reward_cfg)
        r = compute_reward(res, cfg=reward_cfg)
    except Exception as e:
        return {
            "expr": expr,
            "reward": -10.0,
            "reward_components": {"error": f"exception: {type(e).__name__}: {e}"},
            "expr_info": {
                "columns": info.columns,
                "functions": info.functions,
                "max_window": info.max_window,
                "max_shift": info.max_shift,
                "warmup_bars": info.warmup_bars,
            },
        }

    return {
        "expr": expr,
        "reward": float(r["reward"]),
        "reward_components": r["components"],
        "exposures": res.get("exposures"),
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


def _select_diverse(
    df: pd.DataFrame,
    ranked: List[Tuple[str, float]],
    *,
    details: Dict[str, Dict[str, object]],
    topk: int,
    corr_thr: float,
    corr_method: str,
    min_overlap: int,
    candidate_multiplier: int,
    family_mode: str,
    max_per_family: int,
    structure_dedupe: bool,
    drop_error: bool,
) -> List[Tuple[str, float, float, str, str]]:
    if topk <= 0:
        return []

    max_candidates = min(len(ranked), max(topk * max(int(candidate_multiplier), 1), topk))
    pool = ranked[:max_candidates]

    seen_struct: set[str] = set()
    family_counts: Dict[str, int] = {}
    selected_series: List[pd.Series] = []
    series_cache: Dict[str, pd.Series] = {}

    out: List[Tuple[str, float, float, str, str]] = []

    for expr, reward in pool:
        d = details.get(expr) or {}
        rc = d.get("reward_components") or {}
        if drop_error and isinstance(rc, dict) and rc.get("error"):
            continue

        expr_info = d.get("expr_info") or {}
        if not isinstance(expr_info, dict):
            continue

        struct_key = _expr_structure_key(expr)
        if structure_dedupe and struct_key in seen_struct:
            continue

        cols = expr_info.get("columns") or []
        if not isinstance(cols, list):
            cols = []
        family_key = _family_key([str(c) for c in cols], mode=str(family_mode))
        if int(max_per_family) > 0 and family_counts.get(family_key, 0) >= int(max_per_family):
            continue

        warmup = int(expr_info.get("warmup_bars") or 0)
        s = series_cache.get(expr)
        if s is None:
            s = _compute_factor_series(df, expr, warmup_bars=warmup)
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
        selected_series.append(s)
        out.append((expr, float(reward), float(max_abs_corr), struct_key, family_key))

        if len(out) >= topk:
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek (OpenAI-compatible) propose DSL expressions and score them.")
    ap.add_argument("--panel", required=True, help="Panel CSV")
    ap.add_argument("--n", type=int, default=50, help="How many expressions to request from the LLM")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--panel-rows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--embargo-bars", type=int, default=60)

    # LLM
    ap.add_argument("--deepseek", action="store_true", help="Use DeepSeek OpenAI-compatible API")
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.7")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "1200")))
    ap.add_argument("--timeout-sec", type=int, default=int(os.environ.get("DEEPSEEK_TIMEOUT_SEC", "60")))
    ap.add_argument("--themes", default="funding,open_interest,volume,volatility", help="Comma-separated themes for the model")

    ap.add_argument("--exprs-file", default=None, help="Optional: read expressions (one per line) instead of calling API")
    ap.add_argument("--prompt-only", action="store_true", help="Print prompt and exit")

    # stage2 (reflection / rewrite)
    ap.add_argument("--stage2", action="store_true", help="Enable 2nd stage: reflect on scored seeds and generate improved expressions")
    ap.add_argument("--stage2-n", type=int, default=50, help="How many expressions to request for stage2")
    ap.add_argument("--stage2-seeds", type=int, default=25, help="How many top ranked expressions to provide as seeds")
    ap.add_argument(
        "--stage2-exprs-file",
        default=None,
        help="Optional: read stage2 expressions from file instead of calling API (offline stage2)",
    )

    # diversity
    ap.add_argument("--diversity-corr-thr", type=float, default=0.85)
    ap.add_argument("--diversity-corr-method", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--diversity-min-overlap", type=int, default=800)
    ap.add_argument("--candidate-multiplier", type=int, default=20)
    ap.add_argument("--family-mode", choices=["set", "primary"], default="set")
    ap.add_argument("--max-per-family", type=int, default=2)
    ap.add_argument("--no-structure-dedupe", action="store_true")
    ap.add_argument("--drop-error", action="store_true")

    # reward tuning (optional)
    ap.add_argument("--reward-mode", choices=["ic", "trade", "hybrid"], default="ic")
    ap.add_argument("--use-fold-median-ic", action="store_true")
    ap.add_argument("--w-fold-median-ic", type=float, default=None)
    ap.add_argument("--w-degenerate", type=float, default=None)
    ap.add_argument("--min-unique-ratio", type=float, default=None)
    ap.add_argument("--min-factor-std", type=float, default=None)

    ap.add_argument("--trade-z-thr", type=float, default=None)
    ap.add_argument("--trade-base-fee", type=float, default=None)
    ap.add_argument("--trade-impact-coef", type=float, default=None)
    ap.add_argument("--trade-dd-thr", type=float, default=None)
    ap.add_argument("--trade-dd-penalty", type=float, default=None)
    ap.add_argument("--trade-min-activity", type=int, default=None)

    ap.add_argument("--w-exposure", type=float, default=None)
    ap.add_argument("--w-autocorr", type=float, default=None)
    ap.add_argument("--w-turnover", type=float, default=None)
    ap.add_argument("--exposure-corr-thr", type=float, default=None)
    ap.add_argument("--autocorr-thr", type=float, default=None)
    ap.add_argument("--turnover-thr", type=float, default=None)

    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)

    args = ap.parse_args()

    df = pd.read_csv(args.panel)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    df = df.sort_index()
    if args.panel_rows and args.panel_rows > 0:
        df = df.iloc[: int(args.panel_rows)].copy()

    # derive columns and targets
    exclude_prefix = ("ret_",)
    columns = [c for c in df.columns if c != "timestamp" and not c.startswith(exclude_prefix)]

    targets = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        targets = ["ret_fwd_log", "ret_fwd_net_perp"]

    # evaluator configs
    eval_cfg = EvalConfig(
        time=TimeSplitConfig(n_folds=int(args.n_folds), embargo_bars=int(args.embargo_bars)),
        regime=RegimeConfig(),
        ic_freq="1D",
        ic_method="spearman",
    )

    reward_cfg = RewardConfig(mode=str(args.reward_mode), use_fold_median_ic=bool(args.use_fold_median_ic))

    if args.w_fold_median_ic is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "w_fold_median_ic": float(args.w_fold_median_ic)})
    if args.w_degenerate is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "w_degenerate": float(args.w_degenerate)})
    if args.min_unique_ratio is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "min_unique_ratio": float(args.min_unique_ratio)})
    if args.min_factor_std is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "min_factor_std": float(args.min_factor_std)})

    if args.trade_z_thr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_z_thr": float(args.trade_z_thr)})
    if args.trade_base_fee is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_base_fee": float(args.trade_base_fee)})
    if args.trade_impact_coef is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_impact_coef": float(args.trade_impact_coef)})
    if args.trade_dd_thr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_dd_thr": float(args.trade_dd_thr)})
    if args.trade_dd_penalty is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_dd_penalty": float(args.trade_dd_penalty)})
    if args.trade_min_activity is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "trade_min_activity": int(args.trade_min_activity)})
    if args.w_exposure is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "w_exposure": float(args.w_exposure)})
    if args.w_autocorr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "w_autocorr": float(args.w_autocorr)})
    if args.w_turnover is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "w_turnover": float(args.w_turnover)})
    if args.exposure_corr_thr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "exposure_corr_thr": float(args.exposure_corr_thr)})
    if args.autocorr_thr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "autocorr_thr": float(args.autocorr_thr)})
    if args.turnover_thr is not None:
        reward_cfg = RewardConfig(**{**reward_cfg.__dict__, "turnover_thr": float(args.turnover_thr)})

    allowed_funcs = list_allowed_functions()
    window_choices = list(WINDOW_CANDIDATES)

    system = (
        "You are an expert quantitative researcher. "
        "You propose alpha factor expressions in a strict DSL. "
        "Return ONLY valid DSL expressions (no explanations)."
    )

    themes = [t.strip() for t in str(args.themes).split(",") if t.strip()]

    user = (
        "Generate {n} diverse alpha factor expressions in the DSL.\n"
        "Constraints:\n"
        "- Output a JSON array of strings, each string is one expression.\n"
        "- Use ONLY these function names: {funcs}.\n"
        "- Use ONLY these window sizes (for rolling funcs): {wins}.\n"
        "- Use ONLY these available columns: {cols}.\n"
        "- Avoid trivial expressions that are just a raw column (e.g., open/high/low/close/volume).\n"
        "- Try to incorporate these themes: {themes}.\n"
        "Examples of valid syntax: zscore(ts_delta(close, 1), 60); ts_rank(decay_linear(volume, 60), 60).\n"
    ).format(
        n=int(args.n),
        funcs=", ".join(allowed_funcs),
        wins=", ".join(str(w) for w in window_choices),
        cols=", ".join(columns),
        themes=", ".join(themes) if themes else "(none)",
    )

    if args.prompt_only:
        print(user)
        return

    exprs: List[str] = []
    if args.exprs_file:
        with open(args.exprs_file, "r", encoding="utf-8") as f:
            exprs = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    else:
        if not args.deepseek:
            raise SystemExit("No expressions provided. Pass --exprs-file or enable --deepseek with DEEPSEEK_API_KEY.")
        if not args.api_key:
            raise SystemExit("Missing API key. Set DEEPSEEK_API_KEY or pass --api-key.")

        cfg = LLMConfig(
            base_url=str(args.base_url),
            api_key=str(args.api_key),
            model=str(args.model),
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout_sec=int(args.timeout_sec),
        )

        t0 = time.time()
        content = deepseek_chat_completion(cfg, system=system, user=user)
        dt = time.time() - t0
        print(f"LLM responded in {dt:.2f}s")
        exprs = _parse_llm_exprs(content)

    # Validate and de-dup expressions early
    valid: List[str] = []
    seen: set[str] = set()
    for e in exprs:
        e = e.strip().rstrip(";")
        if not e or e in seen:
            continue
        try:
            analyze_expr(e)
        except Exception:
            continue
        seen.add(e)
        valid.append(e)

    if not valid:
        raise SystemExit("No valid expressions parsed from input.")

    details: Dict[str, Dict[str, object]] = {}
    for e in valid:
        d = _score_expr(df, e, targets=targets, eval_cfg=eval_cfg, reward_cfg=reward_cfg)
        details[e] = d

    ranked = sorted(((e, float(details[e].get("reward", -10.0))) for e in details), key=lambda kv: kv[1], reverse=True)

    # Stage2: reflect + propose more expressions, then rescore and re-rank
    if bool(args.stage2):
        stage2_exprs: List[str] = []

        seed_summaries = _make_seed_summaries(ranked, details=details, max_seeds=int(args.stage2_seeds))
        stage2_prompt = _build_reflection_prompt(
            n=int(args.stage2_n),
            allowed_funcs=allowed_funcs,
            window_choices=window_choices,
            columns=columns,
            themes=themes,
            seed_summaries=seed_summaries,
        )

        if args.stage2_exprs_file:
            with open(args.stage2_exprs_file, "r", encoding="utf-8") as f:
                stage2_exprs = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        else:
            if not args.deepseek:
                raise SystemExit("Stage2 enabled but no LLM available. Use --deepseek or provide --stage2-exprs-file.")
            if not args.api_key:
                raise SystemExit("Missing API key for stage2. Set DEEPSEEK_API_KEY or pass --api-key.")

            cfg2 = LLMConfig(
                base_url=str(args.base_url),
                api_key=str(args.api_key),
                model=str(args.model),
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                timeout_sec=int(args.timeout_sec),
            )
            t0 = time.time()
            content2 = deepseek_chat_completion(cfg2, system=system, user=stage2_prompt)
            dt = time.time() - t0
            print(f"LLM(stage2) responded in {dt:.2f}s")
            stage2_exprs = _parse_llm_exprs(content2)

        # Validate and de-dup stage2 expressions
        new_valid: List[str] = []
        for e in stage2_exprs:
            e = e.strip().rstrip(";")
            if not e or e in seen:
                continue
            try:
                analyze_expr(e)
            except Exception:
                continue
            seen.add(e)
            new_valid.append(e)

        if new_valid:
            print(f"Stage2: scoring {len(new_valid)} new expressions")
        else:
            print("Stage2: no valid new expressions")

        for e in new_valid:
            details[e] = _score_expr(df, e, targets=targets, eval_cfg=eval_cfg, reward_cfg=reward_cfg)

        ranked = sorted(((e, float(details[e].get("reward", -10.0))) for e in details), key=lambda kv: kv[1], reverse=True)

    diverse = _select_diverse(
        df,
        ranked,
        details=details,
        topk=int(args.topk),
        corr_thr=float(args.diversity_corr_thr),
        corr_method=str(args.diversity_corr_method),
        min_overlap=int(args.diversity_min_overlap),
        candidate_multiplier=int(args.candidate_multiplier),
        family_mode=str(args.family_mode),
        max_per_family=int(args.max_per_family),
        structure_dedupe=(not bool(args.no_structure_dedupe)),
        drop_error=bool(args.drop_error),
    )

    rows: List[Dict[str, object]] = []
    out_details: List[Dict[str, object]] = []
    for expr, r, max_abs_corr, struct_key, family_key in diverse:
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
        d["pipeline"] = {
            "stage1_source": "exprs_file" if bool(args.exprs_file) else ("deepseek" if bool(args.deepseek) else "unknown"),
            "stage2_enabled": bool(args.stage2),
            "stage2_source": "exprs_file" if bool(args.stage2_exprs_file) else ("deepseek" if (bool(args.stage2) and bool(args.deepseek)) else None),
        }
        rows.append(
            {
                "reward": float(r),
                "expr": expr,
                "max_abs_corr_to_selected": float(max_abs_corr),
                "family_key": str(family_key),
                "structure_key": str(struct_key),
                "stage2_enabled": bool(args.stage2),
                "primary_target": (d.get("reward_components") or {}).get("primary_target"),
                "ic": (d.get("reward_components") or {}).get("ic"),
                "ic_ir": (d.get("reward_components") or {}).get("ic_ir"),
                "penalty": (d.get("reward_components") or {}).get("penalty"),
                "missing_factor_rate": d.get("missing_factor_rate"),
                "missing_target_rate": d.get("missing_target_rate"),
                "max_window": (d.get("expr_info") or {}).get("max_window"),
                "functions": ",".join((d.get("expr_info") or {}).get("functions") or []),
                "columns": ",".join((d.get("expr_info") or {}).get("columns") or []),
            }
        )
        out_details.append(d)

    out_csv = args.out_csv or os.path.join(os.getcwd(), "llm_topk_expr.csv")
    out_json = args.out_json or os.path.join(os.getcwd(), "llm_topk_expr.json")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_details, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    if rows:
        print("Top 5:")
        for r in rows[:5]:
            print(f"- reward={r['reward']:.4f} expr={r['expr']}")


if __name__ == "__main__":
    main()
