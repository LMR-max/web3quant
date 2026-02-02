"""
大额异动分析
- 大额转账（原生或代币）
- 合约调用集中度
优先 Dune（可选），否则基于地址清单统计
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class LargeMoveAnalyzer:
    """大额异动分析器"""

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
        self.logger = get_logger("onchain_large_moves")
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

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime()
            except Exception:
                return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def load_addresses(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            out: List[str] = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and item.get("address"):
                    out.append(str(item.get("address")))
            return out
        return []

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

    def analyze_large_transfers(
        self,
        addresses: Sequence[str],
        *,
        hours: int = 24,
        min_value: float = 100.0,
        token_address: Optional[str] = None,
        limit: int = 200,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """对地址清单统计大额转账"""
        start_block, end_block = self._block_range_hours(hours)
        large_txs: List[Dict[str, Any]] = []
        from_counter: Dict[str, float] = {}
        to_counter: Dict[str, float] = {}

        for addr in addresses:
            if not self.fetcher.validate_address(addr):
                continue

            if token_address:
                transfers = self.fetcher.fetch_token_transfers(
                    token_address=token_address,
                    address=addr,
                    start_block=start_block,
                    end_block=end_block,
                    limit=limit,
                )
                for tx in transfers:
                    value = float(tx.get("value") or 0.0)
                    if value < min_value:
                        continue
                    large_txs.append(tx)
                    f = str(tx.get("from") or "").lower()
                    t = str(tx.get("to") or "").lower()
                    from_counter[f] = from_counter.get(f, 0.0) + value
                    to_counter[t] = to_counter.get(t, 0.0) + value
            else:
                txs = self.fetcher.fetch_transaction_history(
                    address=addr,
                    start_block=start_block,
                    end_block=end_block,
                    limit=limit,
                    sort="desc",
                )
                for tx in txs:
                    value = float(tx.get("value") or 0.0)
                    if value < min_value:
                        continue
                    large_txs.append(tx)
                    f = str(tx.get("from") or "").lower()
                    t = str(tx.get("to") or "").lower()
                    from_counter[f] = from_counter.get(f, 0.0) + value
                    to_counter[t] = to_counter.get(t, 0.0) + value

        return {
            "source": "etherscan",
            "network": self.network,
            "chain": self.chain,
            "hours": hours,
            "min_value": min_value,
            "token_address": token_address,
            "large_tx_count": len(large_txs),
            "from_concentration": self._top_concentration(from_counter, top_n=top_n),
            "to_concentration": self._top_concentration(to_counter, top_n=top_n),
            "large_transactions": large_txs[: min(200, len(large_txs))],
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_contract_call_concentration(
        self,
        addresses: Sequence[str],
        *,
        hours: int = 24,
        min_value: float = 0.0,
        limit: int = 200,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """合约调用集中度（按 to 地址统计）"""
        start_block, end_block = self._block_range_hours(hours)
        contract_counter: Dict[str, float] = {}
        calls: List[Dict[str, Any]] = []

        for addr in addresses:
            if not self.fetcher.validate_address(addr):
                continue
            txs = self.fetcher.fetch_transaction_history(
                address=addr,
                start_block=start_block,
                end_block=end_block,
                limit=limit,
                sort="desc",
            )
            for tx in txs:
                input_data = str(tx.get("input") or "")
                if input_data in ("", "0x"):
                    continue
                value = float(tx.get("value") or 0.0)
                if value < min_value:
                    continue
                to_addr = str(tx.get("to") or "").lower()
                if not to_addr:
                    continue
                calls.append(tx)
                contract_counter[to_addr] = contract_counter.get(to_addr, 0.0) + max(value, 0.0)

        return {
            "source": "etherscan",
            "network": self.network,
            "chain": self.chain,
            "hours": hours,
            "min_value": min_value,
            "call_count": len(calls),
            "contract_concentration": self._top_concentration(contract_counter, top_n=top_n),
            "sample_calls": calls[: min(200, len(calls))],
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
        "large_moves",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="大额异动分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--min-value", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--token", default="")
    ap.add_argument("--address-file", default="")
    ap.add_argument("--addresses", default="")
    ap.add_argument("--contract-calls", action="store_true")
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = LargeMoveAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("large_moves_dune")
    else:
        addresses: List[str] = []
        if args.address_file:
            addresses.extend(LargeMoveAnalyzer.load_addresses(args.address_file))
        if args.addresses:
            addresses.extend([a.strip() for a in args.addresses.split(",") if a.strip()])
        if not addresses:
            raise SystemExit("未提供地址清单，请使用 --address-file 或 --addresses")

        if args.contract_calls:
            result = analyzer.analyze_contract_call_concentration(
                addresses,
                hours=int(args.hours),
                min_value=float(args.min_value),
                limit=int(args.limit),
                top_n=int(args.top_n),
            )
            out_path = args.out or _default_out_path("contract_call_concentration")
        else:
            result = analyzer.analyze_large_transfers(
                addresses,
                hours=int(args.hours),
                min_value=float(args.min_value),
                token_address=(args.token or None),
                limit=int(args.limit),
                top_n=int(args.top_n),
            )
            out_path = args.out or _default_out_path("large_transfers")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
