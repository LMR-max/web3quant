"""
交易结构 / MEV 分析
- sandwich / arb / liquidation 等（优先 Dune）
- 若无 Dune Query ID，返回提示与空结果
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class MEVAnalyzer:
    """MEV 分析器"""

    def __init__(
        self,
        network: str = "ethereum",
        chain: str = "mainnet",
        *,
        use_simulation: bool = False,
        config: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[Any] = None,
    ) -> None:
        self.network = (network or "ethereum").lower()
        self.chain = (chain or "mainnet").lower()
        self.logger = get_logger("onchain_mev")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    def analyze_dune(self, query_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """使用 Dune 获取 MEV 指标"""
        dune_res = self.fetcher.fetch_dune_query(query_id, parameters or {})
        rows = (dune_res.get("result") or {}).get("rows") or []
        return {
            "source": "dune",
            "network": self.network,
            "chain": self.chain,
            "rows": rows,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_placeholder(self) -> Dict[str, Any]:
        """无 Dune Query 时返回占位结构"""
        return {
            "source": "local",
            "network": self.network,
            "chain": self.chain,
            "note": "缺少 Dune Query ID，无法获取 MEV 指标。",
            "rows": [],
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }


def _default_out_path(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "mev",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="交易结构/MEV 分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = MEVAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("mev_dune")
    else:
        result = analyzer.analyze_placeholder()
        out_path = args.out or _default_out_path("mev_placeholder")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
