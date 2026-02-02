from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _serialize_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).tz_convert("UTC").isoformat()
    except Exception:
        try:
            return pd.Timestamp(value, tz="UTC").isoformat()
        except Exception:
            return None


def _funding_model_to_dict(model: Any) -> Dict[str, Any]:
    # FundingRateData fields we care about.
    return {
        "symbol": getattr(model, "symbol", None),
        "exchange": getattr(model, "exchange", None),
        "market_type": getattr(model, "market_type", None),
        "funding_time": _serialize_ts(getattr(model, "funding_time", None)),
        "funding_rate": getattr(model, "funding_rate", None),
        "predicted_rate": getattr(model, "predicted_rate", None),
        "interval_hours": getattr(model, "interval_hours", None),
    }


def _oi_model_to_dict(model: Any) -> Dict[str, Any]:
    return {
        "symbol": getattr(model, "symbol", None),
        "exchange": getattr(model, "exchange", None),
        "market_type": getattr(model, "market_type", None),
        "timestamp": _serialize_ts(getattr(model, "timestamp", None)),
        "open_interest": getattr(model, "open_interest", None),
        "open_interest_value": getattr(model, "open_interest_value", None),
        "volume_24h": getattr(model, "volume_24h", None),
        "turnover_24h": getattr(model, "turnover_24h", None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch swap funding/OI history and persist to data_manager_storage.")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--contract-type", default="linear")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT")

    ap.add_argument("--funding-limit", type=int, default=1500)
    ap.add_argument("--funding-since", default=None, help="ISO date/time or ms timestamp")

    ap.add_argument("--oi-timeframe", default="1h")
    ap.add_argument("--oi-limit", type=int, default=1000)
    ap.add_argument("--oi-since", default=None, help="ISO date/time or ms timestamp")

    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default: data_manager_storage/swap/<exchange>/<contract_type>/)",
    )

    args = ap.parse_args()

    # Import here to keep script import-light.
    from crypto_data_system.main import CryptoDataSystem

    system = CryptoDataSystem(config={"cache_enabled": True})
    manager = system.get_data_manager(
        "swap", exchange=args.exchange, contract_type=args.contract_type
    )
    manager.init_fetcher()

    symbol = args.symbol
    safe = _safe_symbol(symbol)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "data_manager_storage", "swap", args.exchange, args.contract_type)
    os.makedirs(out_dir, exist_ok=True)

    # Funding history
    funding_since = args.funding_since
    try:
        if funding_since is not None and funding_since.isdigit():
            funding_since_parsed: Any = int(funding_since)
        else:
            funding_since_parsed = funding_since
    except Exception:
        funding_since_parsed = funding_since

    funding_models = manager.fetcher.fetch_funding_rate_history(
        symbol, since=funding_since_parsed, limit=args.funding_limit
    )
    funding_records = [_funding_model_to_dict(m) for m in (funding_models or []) if m is not None]
    funding_path = os.path.join(out_dir, f"{safe}_funding_history.json")
    with open(funding_path, "w", encoding="utf-8") as f:
        json.dump(funding_records, f, ensure_ascii=False)

    # Open interest history
    oi_since = args.oi_since
    try:
        if oi_since is not None and oi_since.isdigit():
            oi_since_parsed: Any = int(oi_since)
        else:
            oi_since_parsed = oi_since
    except Exception:
        oi_since_parsed = oi_since

    oi_models = manager.fetcher.fetch_open_interest_history(
        symbol, timeframe=args.oi_timeframe, since=oi_since_parsed, limit=args.oi_limit
    )
    oi_records = [_oi_model_to_dict(m) for m in (oi_models or []) if m is not None]
    oi_path = os.path.join(out_dir, f"{safe}_open_interest_{args.oi_timeframe}.json")
    with open(oi_path, "w", encoding="utf-8") as f:
        json.dump(oi_records, f, ensure_ascii=False)

    print("Wrote:")
    print(f"- funding: {funding_path} rows={len(funding_records)}")
    print(f"- open_interest: {oi_path} rows={len(oi_records)}")


if __name__ == "__main__":
    main()
