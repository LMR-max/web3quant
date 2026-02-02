"""
资金循环分析（CEX ↔ DEX ↔ Bridge）
- 优先 Dune
- 回退：基于地址清单与事件粗略统计
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class CapitalCycleAnalyzer:
    """资金循环分析器"""

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
        self.logger = get_logger("onchain_capital_cycle")
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

    def analyze_watchlist(self, addresses: List[str], hours: int = 24) -> Dict[str, Any]:
        """基于地址清单的粗略资金循环统计"""
        # 这里只做地址级别统计：
        # - CEX 热钱包 / DEX Router / Bridge 地址需要用户提供
        # - 使用转账方向推断流向
        start_block, end_block = self._block_range_hours(hours)

        flows = []
        for addr in addresses:
            if not self.fetcher.validate_address(addr):
                continue
            txs = self.fetcher.fetch_transaction_history(
                address=addr,
                start_block=start_block,
                end_block=end_block,
                limit=200,
                sort="desc",
            )
            flows.append({
                "address": addr,
                "tx_count": len(txs),
                "sample": txs[:50],
            })

        return {
            "source": "etherscan",
            "network": self.network,
            "chain": self.chain,
            "hours": hours,
            "flow_samples": flows,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

    def _avg_block_time(self) -> float:
        return 12.0 if self.network == "ethereum" else 2.0

    def _block_range_hours(self, hours: int) -> tuple[int, int]:
        current = self.fetcher.fetch_block_number()
        blocks_per_hour = int(3600 / self._avg_block_time())
        start = max(1, current - blocks_per_hour * hours)
        return start, current


def _default_out_path(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "capital_cycle",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="资金循环分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--address-file", default="")
    ap.add_argument("--addresses", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = CapitalCycleAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("capital_cycle_dune")
    else:
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

        result = analyzer.analyze_watchlist(addresses, hours=int(args.hours))
        out_path = args.out or _default_out_path("capital_cycle_watchlist")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
