"""
NFT 分析
- 蓝筹交易量 / 地板价 / 鲸鱼扫货
优先 Dune，亦可 The Graph
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class NFTAnalyzer:
    """NFT 分析器"""

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
        self.logger = get_logger("onchain_nft")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    def analyze_dune(self, query_id: int, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        dune_res = self.fetcher.fetch_dune_query(query_id, parameters or {})
        rows = (dune_res.get("result") or {}).get("rows") or []
        return {
            "source": "dune",
            "network": self.network,
            "chain": self.chain,
            "rows": rows,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_the_graph(self, subgraph: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = self.fetcher.fetch_the_graph_query(subgraph, query, variables)
        return {
            "source": "the_graph",
            "network": self.network,
            "chain": self.chain,
            "subgraph": subgraph,
            "data": res,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }


def _default_out_path(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "nft",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="NFT 分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--subgraph", default="")
    ap.add_argument("--query", default="")
    ap.add_argument("--variables", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = NFTAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("nft_dune")
    elif args.subgraph and args.query:
        variables = json.loads(args.variables) if args.variables else None
        result = analyzer.analyze_the_graph(args.subgraph, args.query, variables)
        out_path = args.out or _default_out_path("nft_graph")
    else:
        raise SystemExit("需要 --dune-query 或 --subgraph + --query")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
