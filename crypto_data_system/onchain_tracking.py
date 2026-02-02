"""
链上地址跟踪模块
- 地址余额与交易历史汇总
- 简单巨鲸识别
- 结果落盘为 JSON
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .fetchers.onchain_fetcher import OnChainFetcher
from .utils.logger import get_logger


@dataclass
class TrackRule:
    """巨鲸识别规则"""
    whale_balance_threshold: float = 1000.0  # 原生币余额阈值
    whale_tx_threshold: float = 100.0  # 单笔交易阈值


@dataclass
class TrackedAddress:
    """待跟踪地址配置"""
    address: str
    label: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    rule: TrackRule = field(default_factory=TrackRule)


class AddressTracker:
    """地址跟踪器"""

    DEFAULT_RULES: Dict[str, TrackRule] = {
        "ethereum": TrackRule(whale_balance_threshold=1000.0, whale_tx_threshold=100.0),
        "polygon": TrackRule(whale_balance_threshold=1_000_000.0, whale_tx_threshold=100_000.0),
        "bsc": TrackRule(whale_balance_threshold=1_000_000.0, whale_tx_threshold=100_000.0),
        "arbitrum": TrackRule(whale_balance_threshold=500_000.0, whale_tx_threshold=50_000.0),
        "optimism": TrackRule(whale_balance_threshold=500_000.0, whale_tx_threshold=50_000.0),
        "avalanche": TrackRule(whale_balance_threshold=500_000.0, whale_tx_threshold=50_000.0),
    }

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
        self.logger = get_logger("onchain_tracker")
        self.fetcher = OnChainFetcher(
            network=self.network,
            chain=self.chain,
            config=config,
            cache_manager=cache_manager,
            use_simulation=use_simulation,
        )

    def _default_rule(self) -> TrackRule:
        return self.DEFAULT_RULES.get(self.network, TrackRule())

    def _normalize_address_item(self, item: Union[str, Dict[str, Any]]) -> TrackedAddress:
        if isinstance(item, str):
            return TrackedAddress(address=item, rule=self._default_rule())

        address = str(item.get("address") or "").strip()
        label = item.get("label")
        tags = item.get("tags") or []
        rule_data = item.get("rule") or {}
        rule = TrackRule(
            whale_balance_threshold=float(rule_data.get("whale_balance_threshold", self._default_rule().whale_balance_threshold)),
            whale_tx_threshold=float(rule_data.get("whale_tx_threshold", self._default_rule().whale_tx_threshold)),
        )
        return TrackedAddress(address=address, label=label, tags=list(tags), rule=rule)

    def _compute_tx_metrics(self, address: str, txs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        incoming = 0.0
        outgoing = 0.0
        largest = 0.0
        in_count = 0
        out_count = 0
        counterparties = set()
        last_ts = None

        addr_lower = address.lower()
        for tx in txs:
            value = float(tx.get("value") or 0.0)
            largest = max(largest, value)
            from_addr = str(tx.get("from") or "").lower()
            to_addr = str(tx.get("to") or "").lower()

            ts = tx.get("timestamp")
            if ts is not None:
                last_ts = max(last_ts, ts) if last_ts is not None else ts

            if to_addr == addr_lower:
                incoming += value
                in_count += 1
                if from_addr:
                    counterparties.add(from_addr)
            if from_addr == addr_lower:
                outgoing += value
                out_count += 1
                if to_addr:
                    counterparties.add(to_addr)

        return {
            "tx_count": len(txs),
            "incoming_total": incoming,
            "outgoing_total": outgoing,
            "net_flow": incoming - outgoing,
            "largest_tx": largest,
            "in_count": in_count,
            "out_count": out_count,
            "unique_counterparties": len(counterparties),
            "last_tx_time": last_ts.isoformat() if hasattr(last_ts, "isoformat") else None,
        }

    def track_address(
        self,
        address: str,
        *,
        label: Optional[str] = None,
        tags: Optional[List[str]] = None,
        rule: Optional[TrackRule] = None,
        tx_limit: int = 200,
        token_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.fetcher.validate_address(address):
            raise ValueError(f"无效地址: {address}")

        rule = rule or self._default_rule()
        tags = tags or []

        balance_info = self.fetcher.fetch_address_balance(address)
        balance_value = float(balance_info.get("balance", 0.0))

        txs = self.fetcher.fetch_transaction_history(address, limit=tx_limit)
        metrics = self._compute_tx_metrics(address, txs)

        is_whale = bool(
            balance_value >= rule.whale_balance_threshold
            or metrics.get("largest_tx", 0.0) >= rule.whale_tx_threshold
        )

        result = {
            "address": address,
            "label": label,
            "tags": tags,
            "network": self.network,
            "chain": self.chain,
            "balance": balance_value,
            "balance_raw": balance_info,
            "rule": {
                "whale_balance_threshold": rule.whale_balance_threshold,
                "whale_tx_threshold": rule.whale_tx_threshold,
            },
            "is_whale": is_whale,
            "tx_summary": metrics,
            "sample_txs": txs[: min(50, len(txs))],
            "tracked_at": datetime.now(timezone.utc).isoformat(),
        }

        if token_address:
            token_txs = self.fetcher.fetch_token_transfers(
                token_address=token_address,
                address=address,
                limit=tx_limit,
            )
            result["token_transfers"] = token_txs[: min(50, len(token_txs))]
            result["token_transfer_count"] = len(token_txs)

        return result

    def track_addresses(
        self,
        items: Sequence[Union[str, Dict[str, Any]]],
        *,
        tx_limit: int = 200,
        token_address: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in items:
            config = self._normalize_address_item(item)
            if not config.address:
                continue
            try:
                res = self.track_address(
                    config.address,
                    label=config.label,
                    tags=config.tags,
                    rule=config.rule,
                    tx_limit=tx_limit,
                    token_address=token_address,
                )
                results.append(res)
            except Exception as exc:
                self.logger.warning(f"地址跟踪失败: {config.address} - {exc}")
        return results

    @staticmethod
    def load_addresses(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_results(results: Sequence[Dict[str, Any]], out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(list(results), f, ensure_ascii=False, indent=2)
        return out_path


def _parse_addresses(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.address_file:
        return AddressTracker.load_addresses(args.address_file)
    if args.addresses:
        return [a.strip() for a in args.addresses.split(",") if a.strip()]
    return []


def _default_out_path(network: str, chain: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data_manager_storage",
        "onchain",
        "tracking",
        network,
        chain,
        f"address_tracking_{ts}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="链上地址跟踪")
    ap.add_argument("--network", default="ethereum")
    ap.add_argument("--chain", default="mainnet")
    ap.add_argument("--addresses", default="")
    ap.add_argument("--address-file", default="")
    ap.add_argument("--tx-limit", type=int, default=200)
    ap.add_argument("--token", default="", help="可选：代币合约地址")
    ap.add_argument("--out", default="")
    ap.add_argument("--simulation", action="store_true")

    args = ap.parse_args()
    items = _parse_addresses(args)
    if not items:
        raise SystemExit("未提供地址。请使用 --addresses 或 --address-file")

    tracker = AddressTracker(
        network=args.network,
        chain=args.chain,
        use_simulation=bool(args.simulation),
    )
    results = tracker.track_addresses(
        items,
        tx_limit=int(args.tx_limit),
        token_address=(args.token or None),
    )

    out_path = args.out or _default_out_path(args.network, args.chain)
    out_path = os.path.abspath(out_path)
    AddressTracker.save_results(results, out_path)
    print(f"Saved: {out_path} (count={len(results)})")


if __name__ == "__main__":
    main()
