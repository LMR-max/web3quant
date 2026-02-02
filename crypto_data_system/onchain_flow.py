"""
链上资金流分析
- 交易所流入/流出
- 稳定币流向
优先使用 Dune 查询，缺省回退到浏览器 API
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


@dataclass
class FlowSummary:
    exchange: str
    hours: int
    asset: str
    inflow: float
    outflow: float
    net_flow: float


class ExchangeFlowAnalyzer:
    """交易所资金流分析"""

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
        self.logger = get_logger("onchain_flow")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    def _stablecoin_addresses(self) -> Dict[str, str]:
        contracts = self.fetcher.common_contracts or {}
        stablecoins = {}
        for symbol in ("USDT", "USDC", "DAI"):
            addr = contracts.get(symbol)
            if addr:
                stablecoins[symbol] = addr
        return stablecoins

    @staticmethod
    def _extract_numeric(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
        for k in keys:
            if k in row and row[k] is not None:
                try:
                    return float(row[k])
                except Exception:
                    continue
        return None

    def _summarize_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        inflow = 0.0
        outflow = 0.0
        net_flow = 0.0
        counted = 0
        for row in rows:
            in_v = self._extract_numeric(row, ("inflow", "in", "in_amount", "deposit", "deposits"))
            out_v = self._extract_numeric(row, ("outflow", "out", "out_amount", "withdrawal", "withdrawals"))
            net_v = self._extract_numeric(row, ("net_flow", "net", "netflow", "net_amount"))
            if in_v is not None:
                inflow += in_v
            if out_v is not None:
                outflow += out_v
            if net_v is not None:
                net_flow += net_v
            if in_v is not None or out_v is not None or net_v is not None:
                counted += 1

        if counted and net_flow == 0.0:
            net_flow = inflow - outflow

        return {
            "rows_counted": counted,
            "inflow": inflow,
            "outflow": outflow,
            "net_flow": net_flow,
        }

    def fetch_exchange_flow(
        self,
        exchange: str = "binance",
        hours: int = 24,
        *,
        dune_query_id: Optional[int] = None,
        dune_params: Optional[Dict[str, Any]] = None,
        token_addresses: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """交易所资金流（优先 Dune，回退浏览器 API）"""
        exchange = (exchange or "binance").lower()

        if dune_query_id:
            dune_res = self.fetcher.fetch_dune_query(dune_query_id, dune_params or {})
            rows = (dune_res.get("result") or {}).get("rows") or []
            summary = self._summarize_rows(rows)
            return {
                "source": "dune",
                "exchange": exchange,
                "network": self.network,
                "chain": self.chain,
                "hours": hours,
                "summary": summary,
                "rows": rows,
                "tracked_at": datetime.now(timezone.utc).isoformat(),
            }

        # 回退到浏览器 API
        results: List[Dict[str, Any]] = []
        if token_addresses:
            for symbol, address in token_addresses.items():
                try:
                    flow = self.fetcher.fetch_exchange_flow(
                        exchange=exchange,
                        token_address=address,
                        hours=hours,
                    )
                    flow["token_symbol"] = symbol
                    results.append(flow)
                except Exception as exc:
                    self.logger.warning(f"稳定币 {symbol} 资金流失败: {exc}")
        else:
            results.append(
                self.fetcher.fetch_exchange_flow(
                    exchange=exchange,
                    token_address=None,
                    hours=hours,
                )
            )

        return {
            "source": "etherscan",
            "exchange": exchange,
            "network": self.network,
            "chain": self.chain,
            "hours": hours,
            "results": results,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_stablecoin_flow(
        self,
        exchange: str = "binance",
        hours: int = 24,
        *,
        dune_query_id: Optional[int] = None,
        dune_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """稳定币流向（USDT/USDC/DAI），优先 Dune"""
        stablecoins = self._stablecoin_addresses()
        return self.fetch_exchange_flow(
            exchange=exchange,
            hours=hours,
            dune_query_id=dune_query_id,
            dune_params=dune_params,
            token_addresses=stablecoins,
        )


def _default_out_path(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "flow",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="交易所资金流/稳定币流向")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dune-query", type=int, default=0)
    ap.add_argument("--dune-params", default="", help="JSON 字符串，可选")
    ap.add_argument("--stablecoin", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    dune_query_id = int(args.dune_query) if args.dune_query else None
    dune_params = json.loads(args.dune_params) if args.dune_params else None

    analyzer = ExchangeFlowAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    if args.stablecoin:
        result = analyzer.fetch_stablecoin_flow(
            exchange=args.exchange,
            hours=args.hours,
            dune_query_id=dune_query_id,
            dune_params=dune_params,
        )
        out_path = args.out or _default_out_path("stablecoin_flow")
    else:
        result = analyzer.fetch_exchange_flow(
            exchange=args.exchange,
            hours=args.hours,
            dune_query_id=dune_query_id,
            dune_params=dune_params,
        )
        out_path = args.out or _default_out_path("exchange_flow")

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
