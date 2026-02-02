"""
Gas 维度分析
- gas price
- gas used / gas utilization
- 合约调用成本估算
优先 Dune（可选），否则基于最近区块统计
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class GasAnalyzer:
    """Gas 维度分析器"""

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
        self.logger = get_logger("onchain_gas")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    def _avg_block_time(self) -> float:
        return 12.0 if self.network == "ethereum" else 2.0

    def _block_range_hours(self, hours: int) -> Tuple[int, int]:
        current = self.fetcher.fetch_block_number()
        blocks_per_hour = int(3600 / self._avg_block_time())
        start = max(1, current - blocks_per_hour * hours)
        return start, current

    def analyze_blocks(self, hours: int = 1, max_blocks: int = 200) -> Dict[str, Any]:
        """基于最近区块统计 gas 指标"""
        start_block, end_block = self._block_range_hours(hours)
        total_blocks = max(1, min(max_blocks, end_block - start_block + 1))
        block_numbers = list(range(end_block, max(end_block - total_blocks + 1, start_block) - 1, -1))

        gas_used_sum = 0.0
        gas_limit_sum = 0.0
        base_fee_sum = 0.0
        tx_sum = 0.0
        base_fee_count = 0

        for bn in block_numbers:
            block = self.fetcher.fetch_block(bn)
            gas_used = float(block.get("gas_used") or 0.0)
            gas_limit = float(block.get("gas_limit") or 0.0)
            txs = float(block.get("transactions") or 0.0)
            base_fee = float(block.get("base_fee_per_gas") or 0.0)

            gas_used_sum += gas_used
            gas_limit_sum += gas_limit
            tx_sum += txs
            if base_fee > 0:
                base_fee_sum += base_fee
                base_fee_count += 1

        block_count = len(block_numbers)
        avg_gas_used = gas_used_sum / block_count if block_count else 0.0
        avg_gas_limit = gas_limit_sum / block_count if block_count else 0.0
        gas_utilization = (avg_gas_used / avg_gas_limit) if avg_gas_limit > 0 else 0.0
        avg_base_fee_gwei = (base_fee_sum / base_fee_count / 1e9) if base_fee_count else 0.0
        avg_tx_per_block = tx_sum / block_count if block_count else 0.0

        gas_price = self.fetcher.fetch_gas_price()
        gas_price_gwei = float(gas_price.get("gas_price_gwei") or 0.0)

        # 常见交易/合约调用成本估算
        transfer_gas = 21000
        contract_call_gas = 200000
        est_transfer_cost_eth = gas_price_gwei * transfer_gas / 1e9
        est_contract_cost_eth = gas_price_gwei * contract_call_gas / 1e9

        return {
            "source": "web3",
            "network": self.network,
            "chain": self.chain,
            "hours": hours,
            "blocks_analyzed": block_count,
            "avg_gas_used": avg_gas_used,
            "avg_gas_limit": avg_gas_limit,
            "gas_utilization": gas_utilization,
            "avg_base_fee_gwei": avg_base_fee_gwei,
            "avg_tx_per_block": avg_tx_per_block,
            "gas_price": gas_price,
            "est_transfer_cost_eth": est_transfer_cost_eth,
            "est_contract_call_cost_eth": est_contract_cost_eth,
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
        "gas",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gas 维度分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--max-blocks", type=int, default=200)
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = GasAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("gas_dune")
    else:
        result = analyzer.analyze_blocks(int(args.hours), int(args.max_blocks))
        out_path = args.out or _default_out_path("gas_blocks")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
