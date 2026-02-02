"""
地址行为分析
- 活跃地址
- 新增地址
- 留存（两窗口）
优先使用 Dune 查询，回退为地址清单的局部统计
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


class AddressBehaviorAnalyzer:
    """地址行为分析器"""

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
        self.logger = get_logger("onchain_behavior")
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

    def _has_activity(self, address: str, start_block: int, end_block: int) -> bool:
        txs = self.fetcher.fetch_transaction_history(
            address=address,
            start_block=start_block,
            end_block=end_block,
            limit=1,
            sort="desc",
        )
        return bool(txs)

    def analyze_watchlist(
        self,
        addresses: Sequence[str],
        *,
        hours: int = 24,
        retention_hours: int = 24,
    ) -> Dict[str, Any]:
        """对地址清单做活跃/新增/留存统计"""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=hours)

        start_block, end_block = self._block_range_hours(hours)
        prev_start, prev_end = self._block_range_hours(hours + retention_hours)
        prev_window_start = prev_start
        prev_window_end = start_block

        active = []
        new_addrs = []
        retained = []

        for addr in addresses:
            if not self.fetcher.validate_address(addr):
                continue

            # 活跃：窗口内有交易
            is_active = self._has_activity(addr, start_block, end_block)
            if is_active:
                active.append(addr)

            # 新增：最早交易时间在窗口内
            earliest = self.fetcher.fetch_transaction_history(
                address=addr,
                limit=1,
                sort="asc",
            )
            if earliest:
                ts = self._to_datetime(earliest[0].get("timestamp"))
                if ts and ts >= window_start:
                    new_addrs.append(addr)

            # 留存：前一窗口与当前窗口均活跃
            if retention_hours > 0:
                was_active = self._has_activity(addr, prev_window_start, prev_window_end)
                if was_active and is_active:
                    retained.append(addr)

        retention_rate = (len(retained) / len(active)) if active else 0.0

        return {
            "source": "etherscan",
            "network": self.network,
            "chain": self.chain,
            "window_hours": hours,
            "retention_hours": retention_hours,
            "active_count": len(active),
            "new_count": len(new_addrs),
            "retained_count": len(retained),
            "retention_rate": retention_rate,
            "active_addresses": active,
            "new_addresses": new_addrs,
            "retained_addresses": retained,
            "tracked_at": now.isoformat(),
        }

    def analyze_dune(
        self,
        query_id: int,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """使用 Dune 获取全网指标（活跃/新增/留存）"""
        dune_res = self.fetcher.fetch_dune_query(query_id, parameters or {})
        rows = (dune_res.get("result") or {}).get("rows") or []

        active_sum = 0.0
        new_sum = 0.0
        retained_sum = 0.0
        for row in rows:
            for key in ("active", "active_addresses", "active_count"):
                if key in row and row[key] is not None:
                    active_sum += float(row[key])
                    break
            for key in ("new", "new_addresses", "new_count"):
                if key in row and row[key] is not None:
                    new_sum += float(row[key])
                    break
            for key in ("retained", "retained_addresses", "retained_count"):
                if key in row and row[key] is not None:
                    retained_sum += float(row[key])
                    break

        return {
            "source": "dune",
            "network": self.network,
            "chain": self.chain,
            "summary": {
                "active_sum": active_sum,
                "new_sum": new_sum,
                "retained_sum": retained_sum,
            },
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
        "behavior",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="地址行为分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--retention-hours", type=int, default=24)
    ap.add_argument("--address-file", default="")
    ap.add_argument("--addresses", default="")
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = AddressBehaviorAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.dune_query:
        params = json.loads(args.dune_params) if args.dune_params else None
        result = analyzer.analyze_dune(int(args.dune_query), params)
        out_path = args.out or _default_out_path("behavior_dune")
    else:
        addresses: List[str] = []
        if args.address_file:
            addresses.extend(AddressBehaviorAnalyzer.load_addresses(args.address_file))
        if args.addresses:
            addresses.extend([a.strip() for a in args.addresses.split(",") if a.strip()])
        if not addresses:
            raise SystemExit("未提供地址清单，请使用 --address-file 或 --addresses")

        result = analyzer.analyze_watchlist(
            addresses,
            hours=int(args.hours),
            retention_hours=int(args.retention_hours),
        )
        out_path = args.out or _default_out_path("behavior_watchlist")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
