#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Storage & cache audit tool.

目的：
- 检查 data/cache/ 与 data_manager_storage/ 下的文件是否落在标准目录
- 输出可读报告
- 可选 --fix：对“能可靠推断目标目录”的条目执行移动归位

标准目录（cache）：spot/swap/future/option/margin/onchain/social
标准目录（storage）：spot/swap/future/option/margin/onchain/social/web

使用示例：
- 只读审计：python storage_audit.py
- 自动归位：python storage_audit.py --fix
- 仅审计 cache：python storage_audit.py --only cache
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple


CACHE_STANDARD_TOPS = {
    "spot",
    "swap",
    "future",
    "option",
    "margin",
    "onchain",
    "social",
}

STORAGE_STANDARD_TOPS = set(CACHE_STANDARD_TOPS) | {"web"}

# 兼容历史/模块内部使用的 top-level 别名目录
CACHE_TOP_ALIASES: Dict[str, str] = {
    "binance_option": "option",
    "options": "option",
    "ethereum": "onchain",
    "eth": "onchain",
}

# 通过“文件名包含的 market token”推断 market（用于 cache 根目录散落文件）
# 历史上大量文件名形如：binance_spot_ohlcv_BTC_USDT_1m_<hash>.pkl.gz
# 因此不能只依赖 ^spot_ 这种前缀。
FILENAME_TOKEN_TO_MARKET: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:^|[_-])spot(?:[_-]|$)", re.IGNORECASE), "spot"),
    (re.compile(r"(?:^|[_-])swap(?:[_-]|$)", re.IGNORECASE), "swap"),
    (re.compile(r"(?:^|[_-])future(?:[_-]|$)", re.IGNORECASE), "future"),
    (re.compile(r"(?:^|[_-])margin(?:[_-]|$)", re.IGNORECASE), "margin"),
    (re.compile(r"(?:^|[_-])option(?:[_-]|$)", re.IGNORECASE), "option"),
    (re.compile(r"(?:^|[_-])onchain(?:[_-]|$)", re.IGNORECASE), "onchain"),
    (re.compile(r"(?:^|[_-])social(?:[_-]|$)", re.IGNORECASE), "social"),
]


@dataclass
class Issue:
    kind: str  # cache/storage
    path: Path
    reason: str
    suggested_target: Optional[Path] = None


def _strip_known_extensions(filename: str) -> str:
    """Strip common cache/data extensions, including compressed suffix."""
    name = filename
    # order matters: .pkl.gz
    for ext in [
        ".json.gz",
        ".pkl.gz",
        ".parquet.gz",
        ".feather.gz",
        ".csv.gz",
        ".gz",
        ".json",
        ".pkl",
        ".parquet",
        ".feather",
        ".csv",
    ]:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def infer_market_from_cache_filename(file_path: Path) -> Optional[str]:
    base = _strip_known_extensions(file_path.name)
    matches: List[str] = []
    for pat, market in FILENAME_TOKEN_TO_MARKET:
        if pat.search(base):
            matches.append(market)

    # 如果同时命中多个 market token，则认为不可靠（避免误搬移）
    uniq = sorted(set(matches))
    if len(uniq) == 1:
        return uniq[0]
    return None


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def audit_cache(cache_root: Path) -> List[Issue]:
    issues: List[Issue] = []

    if not cache_root.exists():
        return issues

    # 1) 顶层目录别名：例如 data/cache/binance_option -> option
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        top = child.name.lower()
        if top in CACHE_TOP_ALIASES:
            target_top = CACHE_TOP_ALIASES[top]
            suggested = cache_root / target_top / child.name
            issues.append(
                Issue(
                    kind="cache",
                    path=child,
                    reason=f"顶层目录 '{child.name}' 是别名，建议归一化到 '{target_top}/'",
                    suggested_target=suggested,
                )
            )

    # 2) 根目录散落文件
    for child in cache_root.iterdir():
        if child.is_file():
            market = infer_market_from_cache_filename(child)
            if market:
                suggested = cache_root / market / child.name
                issues.append(
                    Issue(
                        kind="cache",
                        path=child,
                        reason=f"文件散落在 cache 根目录，可由文件名前缀推断 market='{market}'",
                        suggested_target=suggested,
                    )
                )
            else:
                issues.append(
                    Issue(
                        kind="cache",
                        path=child,
                        reason="文件散落在 cache 根目录，且无法可靠推断 market（需要人工确认）",
                        suggested_target=None,
                    )
                )

    # 3) 非标准顶层目录（例如多层路径 top 不在标准集合里）
    for p in iter_files(cache_root):
        rel = p.relative_to(cache_root)
        if len(rel.parts) <= 1:
            continue
        top = rel.parts[0].lower()
        if top in CACHE_STANDARD_TOPS:
            continue
        if top in CACHE_TOP_ALIASES:
            # 已在 (1) 里报过整体目录；这里不重复逐文件报
            continue
        # 允许存在自定义顶层，但仍提示
        issues.append(
            Issue(
                kind="cache",
                path=p,
                reason=f"非标准 cache 顶层目录 '{rel.parts[0]}'，可能导致后续查找不一致",
                suggested_target=None,
            )
        )

    return issues


def audit_storage(storage_root: Path) -> List[Issue]:
    issues: List[Issue] = []

    if not storage_root.exists():
        return issues

    # 根目录不应出现文件
    for child in storage_root.iterdir():
        if child.is_file():
            issues.append(
                Issue(
                    kind="storage",
                    path=child,
                    reason="文件不应直接放在 data_manager_storage 根目录",
                    suggested_target=None,
                )
            )

    # 非标准顶层目录
    for child in storage_root.iterdir():
        if not child.is_dir():
            continue
        top = child.name.lower()
        if top in STORAGE_STANDARD_TOPS:
            continue
        issues.append(
            Issue(
                kind="storage",
                path=child,
                reason=f"非标准 storage 顶层目录 '{child.name}'",
                suggested_target=None,
            )
        )

    return issues


def safe_move(src: Path, dst: Path) -> Tuple[bool, str]:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return False, f"目标已存在，跳过: {dst}"
        shutil.move(str(src), str(dst))
        return True, f"已移动到: {dst}"
    except Exception as e:
        return False, f"移动失败: {e}"


def apply_fixes(issues: List[Issue]) -> List[str]:
    logs: List[str] = []
    for it in issues:
        if it.suggested_target is None:
            continue

        # 如果是目录别名归一化：我们只移动目录本身（避免巨量逐文件）
        if it.path.is_dir():
            ok, msg = safe_move(it.path, it.suggested_target)
            logs.append(f"[{it.kind}] {it.path} -> {msg}")
            continue

        # 文件移动
        ok, msg = safe_move(it.path, it.suggested_target)
        logs.append(f"[{it.kind}] {it.path} -> {msg}")

    return logs


def summarize(issues: List[Issue]) -> str:
    counts: Dict[str, int] = {}
    actionable = 0
    for it in issues:
        counts[it.kind] = counts.get(it.kind, 0) + 1
        if it.suggested_target is not None:
            actionable += 1

    parts = []
    for k in sorted(counts.keys()):
        parts.append(f"{k}: {counts[k]}")
    return f"问题总数={len(issues)}（可自动修复={actionable}）; " + ", ".join(parts)


def print_issues(issues: List[Issue], max_items: int = 200) -> None:
    if not issues:
        print("未发现目录/落盘问题 ✅")
        return

    print(summarize(issues))
    print("-")

    shown = 0
    for it in issues:
        if shown >= max_items:
            print(f"... 其余 {len(issues) - max_items} 条未展示（可用 --max-items 调大）")
            break
        print(f"[{it.kind}] {it.path}")
        print(f"  - 原因: {it.reason}")
        if it.suggested_target is not None:
            print(f"  - 建议: -> {it.suggested_target}")
        shown += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cache/storage placement and optionally fix.")
    parser.add_argument("--fix", action="store_true", help="自动移动可可靠归位的条目")
    parser.add_argument(
        "--only",
        choices=["cache", "storage", "all"],
        default="all",
        help="仅审计指定部分",
    )
    parser.add_argument("--cache-root", default="data/cache", help="cache 根目录")
    parser.add_argument("--storage-root", default="data_manager_storage", help="storage 根目录")
    parser.add_argument("--max-items", type=int, default=200, help="最多打印多少条问题")

    args = parser.parse_args()

    workdir = Path(os.getcwd())
    cache_root = (workdir / args.cache_root).resolve()
    storage_root = (workdir / args.storage_root).resolve()

    issues: List[Issue] = []

    if args.only in ("cache", "all"):
        issues.extend(audit_cache(cache_root))
    if args.only in ("storage", "all"):
        issues.extend(audit_storage(storage_root))

    print_issues(issues, max_items=args.max_items)

    if args.fix:
        logs = apply_fixes(issues)
        if logs:
            print("\n执行 --fix 结果:")
            for line in logs[:500]:
                print(line)
            if len(logs) > 500:
                print(f"... 其余 {len(logs) - 500} 条未展示")

        # fix 后再审计一次（只读）
        issues2: List[Issue] = []
        if args.only in ("cache", "all"):
            issues2.extend(audit_cache(cache_root))
        if args.only in ("storage", "all"):
            issues2.extend(audit_storage(storage_root))

        print("\nfix 后复查:")
        print_issues(issues2, max_items=args.max_items)

        # 若仍有问题，返回非 0 方便脚本化
        return 0 if not issues2 else 2

    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
