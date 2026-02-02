"""
价格关联分析
- 链上活动指标 vs 价格/波动率
优先 Dune 获取链上指标，价格来自现货 OHLCV
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .fetchers.onchain_fetcher import OnChainFetcher
from .fetchers.spot_fetcher import CCXTSpotFetcher
from .utils.logger import get_logger


class PriceRelationAnalyzer:
    """链上指标与价格关系分析器"""

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
        self.logger = get_logger("onchain_price_relation")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    @staticmethod
    def _extract_timestamp(row: Dict[str, Any]) -> Optional[pd.Timestamp]:
        for key in ("timestamp", "date", "block_time", "time", "day"):
            if key in row and row[key] is not None:
                try:
                    return pd.to_datetime(row[key], utc=True)
                except Exception:
                    continue
        return None

    @staticmethod
    def _extract_metric(row: Dict[str, Any], metric_key: str) -> Optional[float]:
        if metric_key in row and row[metric_key] is not None:
            try:
                return float(row[metric_key])
            except Exception:
                return None
        return None

    def analyze_dune_vs_price(
        self,
        *,
        dune_query_id: int,
        metric_key: str,
        exchange: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 500,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """使用 Dune 指标与价格做相关分析"""
        dune_res = self.fetcher.fetch_dune_query(dune_query_id, parameters or {})
        rows = (dune_res.get("result") or {}).get("rows") or []

        metrics = []
        for row in rows:
            ts = self._extract_timestamp(row)
            val = self._extract_metric(row, metric_key)
            if ts is not None and val is not None:
                metrics.append({"timestamp": ts, "metric": val})

        if not metrics:
            return {
                "source": "dune",
                "note": "Dune 结果缺少可用时间戳或指标列",
                "rows": rows,
                "tracked_at": datetime.now(timezone.utc).isoformat(),
            }

        metric_df = pd.DataFrame(metrics).set_index("timestamp").sort_index()

        price_fetcher = CCXTSpotFetcher(exchange=exchange)
        ohlcv = price_fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        price_df = pd.DataFrame(ohlcv)
        if "timestamp" in price_df.columns:
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True, errors="coerce")
            price_df = price_df.set_index("timestamp")

        price_df = price_df.sort_index()
        price_df["ret"] = price_df["close"].pct_change()
        price_df["vol"] = price_df["ret"].rolling(24, min_periods=5).std()

        merged = metric_df.join(price_df[["close", "ret", "vol"]], how="inner")
        merged = merged.dropna()

        if merged.empty:
            return {
                "source": "dune",
                "note": "时间对齐后无有效样本",
                "metric_key": metric_key,
                "tracked_at": datetime.now(timezone.utc).isoformat(),
            }

        corr_ret = merged["metric"].corr(merged["ret"])
        corr_vol = merged["metric"].corr(merged["vol"])

        return {
            "source": "dune",
            "network": self.network,
            "chain": self.chain,
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "metric_key": metric_key,
            "sample_count": int(len(merged)),
            "corr_metric_ret": float(corr_ret) if pd.notna(corr_ret) else None,
            "corr_metric_vol": float(corr_vol) if pd.notna(corr_vol) else None,
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
        "price_relation",
        f"{name}_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="链上指标与价格关系分析")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--dune-query", type=int, required=True)
    ap.add_argument("--metric-key", required=True)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--dune-params", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")
    args = ap.parse_args()

    analyzer = PriceRelationAnalyzer(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )

    params = json.loads(args.dune_params) if args.dune_params else None
    result = analyzer.analyze_dune_vs_price(
        dune_query_id=int(args.dune_query),
        metric_key=str(args.metric_key),
        exchange=str(args.exchange),
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        limit=int(args.limit),
        parameters=params,
    )

    out_path = args.out or _default_out_path("price_relation")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
