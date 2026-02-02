"""
交易对分类工具
"""
from __future__ import annotations

from typing import Dict, Optional

from crypto_data_system.data_models import DataConverter


STABLE_USD = {
    "USDT", "USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "USDD", "PYUSD", "USTC"
}
FIAT = {"USD", "EUR", "JPY", "KRW", "GBP", "AUD", "CAD", "CHF", "CNY", "HKD", "SGD", "TRY", "RUB", "BRL", "MXN", "ZAR", "INR"}
CRYPTO_MAJOR = {"BTC", "ETH"}

TRADE_TYPE_CODE = {
    "spot": "S",
    "swap": "SW",
    "future": "F",
    "option": "O",
    "margin": "M",
}


def _fallback_parse_symbol(symbol: str) -> Dict[str, str]:
    result = {
        "original": symbol,
        "base": "",
        "quote": "",
        "settle": "",
        "market_type": "spot",
    }

    if not symbol:
        return result

    if "/" in symbol:
        parts = symbol.split("/")
        if len(parts) >= 2:
            result["base"] = parts[0]
            result["quote"] = parts[1]
            return result

    if "-" in symbol:
        # e.g. OKX: BTC-USDT
        parts = symbol.split("-")
        if len(parts) >= 2:
            result["base"] = parts[0]
            result["quote"] = parts[1]
            return result

    return result


def _get_quote_type(quote: str) -> Dict[str, str]:
    q = (quote or "").upper().strip()
    if not q:
        return {"quote_type": "unknown", "quote_type_label": "UNKNOWN"}

    if q in STABLE_USD:
        return {"quote_type": "stable_usd", "quote_type_label": "USDT/USDC"}
    if q in FIAT:
        return {"quote_type": f"fiat_{q.lower()}", "quote_type_label": q}
    if q in CRYPTO_MAJOR:
        return {"quote_type": "crypto_major", "quote_type_label": "BTC/ETH"}
    return {"quote_type": "crypto_alt", "quote_type_label": "CRYPTO"}


def classify_symbol(symbol: str, exchange: str = "binance", market_type: Optional[str] = None) -> Dict[str, str]:
    """
    交易对分类：标的资产 + 交易类型 + 计价资产类型
    """
    parsed = DataConverter.parse_symbol(symbol, exchange=exchange)
    if not parsed or not parsed.get("base"):
        parsed = _fallback_parse_symbol(symbol)

    base = (parsed.get("base") or "").upper().strip()
    quote = (parsed.get("quote") or "").upper().strip()

    trade_type = (market_type or parsed.get("market_type") or "spot").lower().strip()
    trade_code = TRADE_TYPE_CODE.get(trade_type, trade_type.upper()[:2] or "NA")

    quote_info = _get_quote_type(quote)
    quote_type = quote_info["quote_type"]
    quote_type_label = quote_info["quote_type_label"]

    group_id = f"{base}-{trade_code}-{quote_type_label}" if base else f"NA-{trade_code}-{quote_type_label}"

    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "trade_type": trade_type,
        "trade_code": trade_code,
        "quote_type": quote_type,
        "quote_type_label": quote_type_label,
        "group_id": group_id,
    }
