"""
代币层分析
- 持币集中度
- 鲸鱼占比变化
优先 Dune（可选），否则基于地址清单统计
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class TokenDistributionAnalyzer:
    """代币分布分析器"""

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
        self.logger = get_logger("onchain_token_distribution")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    @staticmethod
    def _top_concentration(counter: Dict[str, float], top_n: int = 10) -> Dict[str, Any]:
        items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        top_items = items[:top_n]
        total = sum(v for _, v in items) if items else 0.0
        top_total = sum(v for _, v in top_items) if top_items else 0.0
        return {
            "top_n": top_n,
            "total_value": total,
            "top_value": top_total,
            "top_share": (top_total / total) if total > 0 else 0.0,
            "top_items": [{"address": a, "value": v} for a, v in top_items],
        }

    def analyze_watchlist(
        self,
        token_address: str,
        addresses: List[str],
        *,
        whale_threshold: float = 100000.0,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """对地址清单统计持币集中度与鲸鱼占比"""
        balances: Dict[str, float] = {}
        whales: Dict[str, float] = {}

        for addr in addresses:
            if not self.fetcher.validate_address(addr):
                continue
            bal = self.fetcher.fetch_address_balance(addr, token_address=token_address)
            value = float(bal.get("balance_token") or 0.0)
            balances[addr.lower()] = value
            if value >= whale_threshold:
                whales[addr.lower()] = value

        total_supply = sum(balances.values())
        whale_supply = sum(whales.values())
        whale_ratio = whale_supply / total_supply if total_supply > 0 else 0.0

        return {
            "source": "etherscan",
            "network": self.network,
            "chain": self.chain,
            "token_address": token_address,
            "total_supply_sample": total_supply,
            "whale_supply_sample": whale_supply,
            "whale_ratio_sample": whale_ratio,
            "whale_threshold": whale_threshold,
            "holder_concentration": self._top_concentration(balances, top_n=top_n),
            "whale_concentration": self._top_concentration(whales, top_n=top_n),
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

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


def _default_out_path(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "token_distribution",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="代币持币集中度/鲸鱼占比")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--token", default="")
    ap.add_argument("--whale-threshold", type=float, default=100000.0)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--address-file", default="")
    ap.add_argument("--addresses", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = TokenDistributionAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("token_distribution_dune")
    else:
        if not args.token:
            raise SystemExit("未提供代币合约地址，请使用 --token")

        addresses: List[str] = []
        if args.address_file:
            with open(args.address_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            addresses.append(item)
                        elif isinstance(item, dict) and item.get("address"):
                            addresses.append(str(item.get("address")))
        if args.addresses:
            addresses.extend([a.strip() for a in args.addresses.split(",") if a.strip()])
        if not addresses:
            raise SystemExit("未提供地址清单，请使用 --address-file 或 --addresses")

        result = analyzer.analyze_watchlist(
            token_address=args.token,
            addresses=addresses,
            whale_threshold=float(args.whale_threshold),
            top_n=int(args.top_n),
        )
        out_path = args.out or _default_out_path("token_distribution_watchlist")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
