from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd


class DSLParseError(ValueError):
    pass


Number = Union[int, float]
Value = Union[pd.Series, Number]


MAX_WINDOW = 5000

# Discrete window choices (for action masking / controllable search)
WINDOW_CANDIDATES: List[int] = [
    1,
    2,
    3,
    5,
    10,
    20,
    30,
    60,
    120,
    240,
    480,
    720,
    1440,
    2880,
    4320,
    5000,
]


def _as_series(x: Value, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(float(x), index=index)
    return s.replace([np.inf, -np.inf], np.nan)


def _binary(df: pd.DataFrame, a: Value, b: Value, op: Callable[[pd.Series, pd.Series], pd.Series]) -> pd.Series:
    sa = _as_series(a, df.index)
    sb = _as_series(b, df.index)
    return op(sa, sb).replace([np.inf, -np.inf], np.nan)


def add(df: pd.DataFrame, a: Value, b: Value) -> pd.Series:
    return _binary(df, a, b, lambda x, y: x + y)


def sub(df: pd.DataFrame, a: Value, b: Value) -> pd.Series:
    return _binary(df, a, b, lambda x, y: x - y)


def mul(df: pd.DataFrame, a: Value, b: Value) -> pd.Series:
    return _binary(df, a, b, lambda x, y: x * y)


def div(df: pd.DataFrame, a: Value, b: Value, eps: float = 1e-12) -> pd.Series:
    def _op(x: pd.Series, y: pd.Series) -> pd.Series:
        return x / (y.where(y.abs() > eps, np.nan))

    return _binary(df, a, b, _op)


def neg(df: pd.DataFrame, x: Value) -> pd.Series:
    return -_as_series(x, df.index)


def abs_(df: pd.DataFrame, x: Value) -> pd.Series:
    return _as_series(x, df.index).abs()


def log(df: pd.DataFrame, x: Value) -> pd.Series:
    s = _as_series(x, df.index)
    return np.log(s.where(s > 0, np.nan))


def sqrt(df: pd.DataFrame, x: Value) -> pd.Series:
    s = _as_series(x, df.index)
    return np.sqrt(s.where(s >= 0, np.nan))


def sign(df: pd.DataFrame, x: Value) -> pd.Series:
    s = _as_series(x, df.index)
    return np.sign(s)


def clip(df: pd.DataFrame, x: Value, lo: Number, hi: Number) -> pd.Series:
    s = _as_series(x, df.index)
    return s.clip(lower=float(lo), upper=float(hi))


def ts_mean(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).mean()


def ts_std(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).std(ddof=1)


def ts_min(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).min()


def ts_max(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).max()


def ts_sum(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).sum()


def ts_delta(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    return s - s.shift(int(window))


def zscore(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    m = ts_mean(df, s, int(window))
    sd = ts_std(df, s, int(window))
    return (s - m) / sd.replace(0.0, np.nan)


def ts_rank(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)

    def _rank_last(a: np.ndarray) -> float:
        if len(a) < 3:
            return float("nan")
        v = pd.Series(a).rank(pct=True).iloc[-1]
        return float(v)

    mp = min(w, max(3, w // 3))
    return s.rolling(w, min_periods=mp).apply(_rank_last, raw=True)


def winsorize(df: pd.DataFrame, x: Value, q: float = 0.01) -> pd.Series:
    s = _as_series(x, df.index)
    qq = float(q)
    if not (0.0 <= qq < 0.5):
        raise DSLParseError(f"winsorize q must be in [0, 0.5), got {q}")
    lo = float(s.quantile(qq)) if s.notna().any() else float("nan")
    hi = float(s.quantile(1.0 - qq)) if s.notna().any() else float("nan")
    return s.clip(lower=lo, upper=hi)


def decay_linear(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    if w <= 1:
        return s
    weights = np.arange(1, w + 1, dtype=float)
    weights = weights / weights.sum()

    def _wma(a: np.ndarray) -> float:
        if len(a) != w:
            return float("nan")
        if np.any(~np.isfinite(a)):
            return float("nan")
        return float(np.dot(a, weights))

    mp = min(w, max(3, w // 2))
    return s.rolling(w, min_periods=mp).apply(_wma, raw=True)


def ts_cov(df: pd.DataFrame, x: Value, y: Value, window: int) -> pd.Series:
    sx = _as_series(x, df.index)
    sy = _as_series(y, df.index)
    w = int(window)
    mp = min(w, max(5, w // 2))
    return sx.rolling(w, min_periods=mp).cov(sy)


def ts_corr(df: pd.DataFrame, x: Value, y: Value, window: int) -> pd.Series:
    sx = _as_series(x, df.index)
    sy = _as_series(y, df.index)
    w = int(window)
    mp = min(w, max(5, w // 2))
    return sx.rolling(w, min_periods=mp).corr(sy)


def ts_argmax(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))

    def _idx(a: np.ndarray) -> float:
        if len(a) < 1:
            return float("nan")
        return float(np.argmax(a))

    # Normalize: 1.0 means max is the most recent bar, 0.0 means max is w-1 bars ago
    s_idx = s.rolling(w, min_periods=mp).apply(_idx, raw=True)
    return s_idx / max(1.0, float(w - 1))


def ts_argmin(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    w = int(window)
    mp = min(w, max(3, w // 3))

    def _idx(a: np.ndarray) -> float:
        if len(a) < 1:
            return float("nan")
        return float(np.argmin(a))

    s_idx = s.rolling(w, min_periods=mp).apply(_idx, raw=True)
    return s_idx / max(1.0, float(w - 1))


def if_else(df: pd.DataFrame, cond: Value, true_val: Value, false_val: Value) -> pd.Series:
    s_cond = _as_series(cond, df.index)
    s_true = _as_series(true_val, df.index)
    s_false = _as_series(false_val, df.index)
    return pd.Series(np.where(s_cond > 0, s_true, s_false), index=df.index)


def is_jump(df: pd.DataFrame, x: Value, window: int) -> pd.Series:
    s = _as_series(x, df.index)
    z = zscore(df, s, window)
    return (z.abs() > 3.0).astype(float)


def signed_power(df: pd.DataFrame, x: Value, p: Number) -> pd.Series:
    s = _as_series(x, df.index)
    pp = float(p)
    return np.sign(s) * (s.abs() ** pp)


def cs_rank(df: pd.DataFrame, x: Value) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional rank (pct)."""
    if isinstance(x, pd.Series):
        return pd.Series(0.5, index=x.index)
    return x.rank(axis=1, pct=True)


def cs_demean(df: pd.DataFrame, x: Value) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional demean."""
    if isinstance(x, pd.Series):
        return pd.Series(0.0, index=x.index)
    return x.sub(x.mean(axis=1), axis=0)


def cs_zscore(df: pd.DataFrame, x: Value) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional zscore."""
    if isinstance(x, pd.Series):
        return pd.Series(0.0, index=x.index)
    mu = x.mean(axis=1)
    sigma = x.std(axis=1)
    return x.sub(mu, axis=0).div(sigma + 1e-9, axis=0)


ALLOWED_FUNCS: Dict[str, Callable[..., Union[pd.Series, pd.DataFrame]]] = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "neg": neg,
    "abs": abs_,
    "log": log,
    "sqrt": sqrt,
    "sign": sign,
    "clip": clip,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_sum": ts_sum,
    "ts_delta": ts_delta,
    "ts_rank": ts_rank,
    "zscore": zscore,
    "winsorize": winsorize,
    "decay_linear": decay_linear,
    "ts_cov": ts_cov,
    "ts_corr": ts_corr,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "if_else": if_else,
    "is_jump": is_jump,
    "signed_power": signed_power,
    "cs_rank": cs_rank,
    "cs_demean": cs_demean,
    "cs_zscore": cs_zscore,
}


ArgKind = Literal["series", "scalar"]


@dataclass(frozen=True)
class OpSpec:
    name: str
    arg_kinds: List[ArgKind]
    window_arg: Optional[int] = None


OP_SPECS: Dict[str, OpSpec] = {
    "add": OpSpec("add", ["series", "series"]),
    "sub": OpSpec("sub", ["series", "series"]),
    "mul": OpSpec("mul", ["series", "series"]),
    "div": OpSpec("div", ["series", "series"]),
    "neg": OpSpec("neg", ["series"]),
    "abs": OpSpec("abs", ["series"]),
    "log": OpSpec("log", ["series"]),
    "sqrt": OpSpec("sqrt", ["series"]),
    "sign": OpSpec("sign", ["series"]),
    "clip": OpSpec("clip", ["series", "scalar", "scalar"]),
    "winsorize": OpSpec("winsorize", ["series", "scalar"]),
    "ts_mean": OpSpec("ts_mean", ["series", "scalar"], window_arg=1),
    "ts_std": OpSpec("ts_std", ["series", "scalar"], window_arg=1),
    "ts_min": OpSpec("ts_min", ["series", "scalar"], window_arg=1),
    "ts_max": OpSpec("ts_max", ["series", "scalar"], window_arg=1),
    "ts_sum": OpSpec("ts_sum", ["series", "scalar"], window_arg=1),
    "ts_delta": OpSpec("ts_delta", ["series", "scalar"], window_arg=1),
    "ts_rank": OpSpec("ts_rank", ["series", "scalar"], window_arg=1),
    "zscore": OpSpec("zscore", ["series", "scalar"], window_arg=1),
    "decay_linear": OpSpec("decay_linear", ["series", "scalar"], window_arg=1),
    "ts_argmax": OpSpec("ts_argmax", ["series", "scalar"], window_arg=1),
    "ts_argmin": OpSpec("ts_argmin", ["series", "scalar"], window_arg=1),
    "if_else": OpSpec("if_else", ["series", "series", "series"]),
    "is_jump": OpSpec("is_jump", ["series", "scalar"], window_arg=1),
    "signed_power": OpSpec("signed_power", ["series", "scalar"]),
    "ts_cov": OpSpec("ts_cov", ["series", "series", "scalar"], window_arg=2),
    "ts_corr": OpSpec("ts_corr", ["series", "series", "scalar"], window_arg=2),
    "cs_rank": OpSpec("cs_rank", ["series"]),
    "cs_demean": OpSpec("cs_demean", ["series"]),
    "cs_zscore": OpSpec("cs_zscore", ["series"]),
}


ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
ALLOWED_UNARYOPS = (ast.USub, ast.UAdd)


@dataclass(frozen=True)
class ExprInfo:
    expr: str
    columns: List[str]
    functions: List[str]
    max_window: int
    max_shift: int

    @property
    def warmup_bars(self) -> int:
        # Conservative warmup (rolling + shift). Keeps missingness penalties fair.
        return int(max(self.max_window, self.max_shift) + 5)


WINDOW_LASTARG_FUNCS = {
    "ts_mean",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_sum",
    "ts_delta",
    "ts_rank",
    "zscore",
    "decay_linear",
    "ts_argmax",
    "ts_argmin",
    "is_jump",
    "ts_cov",
    "ts_corr",
}


def analyze_expr(expr: str) -> ExprInfo:
    tree = parse_expr(expr)
    cols: List[str] = []
    fns: List[str] = []
    max_window = 0
    max_shift = 0

    class _Analyzer(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal max_window, max_shift
            if isinstance(node.func, ast.Name):
                fn = node.func.id
                fns.append(fn)
                if fn in WINDOW_LASTARG_FUNCS and node.args:
                    last = node.args[-1]
                    if isinstance(last, ast.Constant) and isinstance(last.value, (int, float)):
                        w = int(last.value)
                        max_window = max(max_window, w)
                        if fn == "ts_delta":
                            max_shift = max(max_shift, w)

            # Important: do NOT visit node.func (prevents counting function names as columns).
            for a in node.args:
                self.visit(a)

        def visit_Name(self, node: ast.Name) -> None:
            # Only treat non-function identifiers as columns.
            if node.id not in ALLOWED_FUNCS:
                cols.append(node.id)

    _Analyzer().visit(tree)

    # de-dup & keep order
    def _uniq(xs: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return ExprInfo(
        expr=expr.strip(),
        columns=_uniq(cols),
        functions=_uniq(fns),
        max_window=int(max_window),
        max_shift=int(max_shift),
    )


def _validate_ast(node: ast.AST) -> None:
    for n in ast.walk(node):
        if isinstance(n, (ast.Expression, ast.Load)):
            continue
        # `ast.walk` also yields operator nodes like `ast.Add()` / `ast.USub()`.
        if isinstance(n, ALLOWED_BINOPS):
            continue
        if isinstance(n, ALLOWED_UNARYOPS):
            continue
        if isinstance(n, ast.Name):
            continue
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                continue
            raise DSLParseError(f"Only numeric constants allowed, got {type(n.value)}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise DSLParseError("Only simple function calls allowed")
            if n.func.id not in ALLOWED_FUNCS:
                raise DSLParseError(f"Function not allowed: {n.func.id}")
            # For rolling-type functions, require the last arg be a numeric literal.
            if n.func.id in WINDOW_LASTARG_FUNCS:
                if not n.args:
                    raise DSLParseError(f"Missing args for function: {n.func.id}")
                last = n.args[-1]
                if not (isinstance(last, ast.Constant) and isinstance(last.value, (int, float))):
                    raise DSLParseError(f"Window arg must be a numeric literal for: {n.func.id}")
                w = int(last.value)
                if w <= 0 or w > MAX_WINDOW:
                    raise DSLParseError(f"Invalid window for {n.func.id}: {w} (max={MAX_WINDOW})")
                if w not in WINDOW_CANDIDATES:
                    raise DSLParseError(
                        f"Window must be one of {WINDOW_CANDIDATES} for {n.func.id}, got {w}"
                    )
            continue
        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, ALLOWED_BINOPS):
                raise DSLParseError(f"BinOp not allowed: {type(n.op).__name__}")
            continue
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, ALLOWED_UNARYOPS):
                raise DSLParseError(f"UnaryOp not allowed: {type(n.op).__name__}")
            continue

        # Explicitly forbid anything else (attributes, subscripts, comprehensions, etc)
        raise DSLParseError(f"AST node not allowed: {type(n).__name__}")


def parse_expr(expr: str) -> ast.AST:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise DSLParseError(f"Invalid expression syntax: {e}") from e

    _validate_ast(tree)
    return tree


def _eval_node(df: pd.DataFrame, node: ast.AST) -> Value:
    if isinstance(node, ast.Expression):
        return _eval_node(df, node.body)

    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Name):
        name = node.id
        if name not in df.columns:
            raise DSLParseError(f"Unknown variable/column: {name}")
        return pd.to_numeric(df[name], errors="coerce")

    if isinstance(node, ast.UnaryOp):
        v = _eval_node(df, node.operand)
        if isinstance(node.op, ast.USub):
            return neg(df, v)
        if isinstance(node.op, ast.UAdd):
            return _as_series(v, df.index)
        raise DSLParseError(f"Unary op not supported: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        a = _eval_node(df, node.left)
        b = _eval_node(df, node.right)
        if isinstance(node.op, ast.Add):
            return add(df, a, b)
        if isinstance(node.op, ast.Sub):
            return sub(df, a, b)
        if isinstance(node.op, ast.Mult):
            return mul(df, a, b)
        if isinstance(node.op, ast.Div):
            return div(df, a, b)
        raise DSLParseError(f"Bin op not supported: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise DSLParseError("Only simple function calls allowed")
        fn_name = node.func.id
        fn = ALLOWED_FUNCS.get(fn_name)
        if fn is None:
            raise DSLParseError(f"Function not allowed: {fn_name}")
        args = [_eval_node(df, a) for a in node.args]

        spec = OP_SPECS.get(fn_name)
        if spec is not None:
            if len(args) != len(spec.arg_kinds):
                raise DSLParseError(f"Arity mismatch for {fn_name}: expected {len(spec.arg_kinds)}, got {len(args)}")
            for i, (arg, kind) in enumerate(zip(args, spec.arg_kinds)):
                if kind == "scalar" and isinstance(arg, pd.Series):
                    raise DSLParseError(f"Arg {i} of {fn_name} must be scalar")
                if kind == "series":
                    # allow scalar -> broadcast as series
                    continue

        # Enforce int windows on rolling-type operators
        if fn_name in WINDOW_LASTARG_FUNCS and args:
            last = args[-1]
            if isinstance(last, pd.Series):
                raise DSLParseError(f"Window arg must be scalar for: {fn_name}")
            w = int(last)
            if w <= 0 or w > MAX_WINDOW:
                raise DSLParseError(f"Invalid window for {fn_name}: {w} (max={MAX_WINDOW})")
            if w not in WINDOW_CANDIDATES:
                raise DSLParseError(f"Window must be one of {WINDOW_CANDIDATES} for {fn_name}, got {w}")
            args[-1] = w

        return fn(df, *args)  # type: ignore[misc]

    raise DSLParseError(f"Cannot evaluate node: {type(node).__name__}")


def eval_expr(expr: Union[str, ast.AST], df: pd.DataFrame) -> pd.Series:
    tree = parse_expr(expr) if isinstance(expr, str) else expr
    v = _eval_node(df, tree)
    return _as_series(v, df.index)


def list_allowed_functions() -> List[str]:
    return sorted(ALLOWED_FUNCS.keys())


def list_operator_specs() -> List[OpSpec]:
    return [OP_SPECS[k] for k in sorted(OP_SPECS.keys())]
