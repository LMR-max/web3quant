from __future__ import annotations
import argparse
import os
import sys

# Allow running this file directly (without `-m`) from any cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

from alphagen_style.spot_loader import load_spot_merged_ohlcv
from alphagen_style.swap_loader import load_funding_history_json, load_open_interest_history_json
from alphagen_style.targets import forward_log_return, forward_perp_net_simple_return


def _infer_bar_minutes(timeframe: str) -> int:
    # minimal: only what we have in storage right now
    if timeframe.endswith("m"):
        return int(timeframe[:-1])
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a simple AlphaGen panel from spot OHLCV (merged.json or parquet).")
    ap.add_argument("--path", required=True, help="Path to *_merged.json or ohlcv_merged.parquet")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--horizon", type=int, default=60, help="Forward horizon in bars")
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--swap-funding", default=None, help="Optional funding history json to merge")
    ap.add_argument("--swap-oi", default=None, help="Optional open interest history json to merge")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--drop-gaps", action="store_true", help="Drop rows with large time gaps (recommended if data has outages)")
    ap.add_argument("--max-gap-multiplier", type=float, default=2.0, help="Max gap multiplier vs expected interval before dropping")

    args = ap.parse_args()

    df = load_spot_merged_ohlcv(args.path, max_rows=args.max_rows)
    if df.empty:
        raise SystemExit("Loaded empty dataframe")
    
    original_rows = len(df)
    
    # Gap detection and cleanup
    if args.drop_gaps:
        bar_minutes = _infer_bar_minutes(args.timeframe)
        expected_delta = pd.Timedelta(minutes=bar_minutes)
        
        # Calculate actual time differences
        time_diffs = df.index.to_series().diff()
        max_allowed_gap = expected_delta * args.max_gap_multiplier
        
        # Mark rows following large gaps
        gap_mask = time_diffs > max_allowed_gap
        
        # Also mark subsequent rows within horizon to avoid contaminated forward returns
        contaminated_mask = gap_mask.copy()
        for i in range(1, args.horizon + 1):
            contaminated_mask |= gap_mask.shift(-i, fill_value=False)
        
        df = df[~contaminated_mask]
        dropped_rows = original_rows - len(df)
        
        if dropped_rows > 0:
            print(f"[Gap Filter] Dropped {dropped_rows}/{original_rows} rows due to data gaps (>{max_allowed_gap})")
    
    if df.empty:
        raise SystemExit("All data filtered out due to gaps. Try --max-gap-multiplier=5.0 or check data quality.")

    # --- AlphaGPT-style Feature Engineering ---
    # 1. ret: log returns (will be added below as ret_fwd_log, but useful as feature too?)
    #    Actually dsl handles returns via delta(log(close)). But we can add pre-computed.
    
    # 2. pressure: (buy_vol - sell_vol) / total_vol
    # Using taker_buy_base_asset_volume if available.
    # Binance spot: volume (total), taker_buy_base_asset_volume (buy). Sell = Total - Buy.
    if "volume" in df.columns and "taker_buy_base_asset_volume" in df.columns:
        buy_vol = df["taker_buy_base_asset_volume"]
        sell_vol = df["volume"] - buy_vol
        df["pressure"] = (buy_vol - sell_vol) / (df["volume"] + 1e-9)
    
    # 3. fomo: volume acceleration
    if "volume" in df.columns:
        # Simple acceleration: vol / ma(vol, 60)
        v = df["volume"]
        v_ma = v.rolling(60).mean()
        df["fomo"] = v / (v_ma + 1e-9)
        # Log volume
        df["log_vol"] = pd.Series(np.log(v + 1.0), index=df.index)
        
    # 4. dev: price deviation from trend
    # (close - ma(60)) / ma(60)
    c = df["close"]
    ma60 = c.rolling(60).mean()
    df["dev"] = (c - ma60) / (ma60 + 1e-9)
    
    # 5. hl_range: volatility proxy
    if "high" in df.columns and "low" in df.columns:
        df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

    df["ret_fwd_log"] = forward_log_return(df["close"], horizon=args.horizon)

    # Optional: merge swap state series (funding/open interest)
    bar_minutes = _infer_bar_minutes(args.timeframe)

    if args.swap_funding:
        fdf = load_funding_history_json(args.swap_funding)
        if not fdf.empty and "funding_rate" in fdf.columns:
            # asof merge: latest known funding rate at each bar
            df = pd.merge_asof(
                df.sort_index(),
                fdf[["funding_rate"]].sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )
            df["ret_fwd_net_perp"] = forward_perp_net_simple_return(
                df["close"],
                horizon=args.horizon,
                funding_rate=df["funding_rate"],
                bar_minutes=bar_minutes,
            )

    if args.swap_oi:
        odf = load_open_interest_history_json(args.swap_oi)
        if not odf.empty and "open_interest" in odf.columns:
            df = pd.merge_asof(
                df.sort_index(),
                odf[["open_interest", "open_interest_value"]].sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
            )

    # Basic sanity features (placeholder for DSL later)
    df["ret_1"] = df["close"].pct_change(1)

    # Fallback pressure proxy when taker buy volume is unavailable.
    # Approximate signed flow as sign(ret_1) * (volume / ma(volume,60) - 1)
    if "pressure" not in df.columns and "volume" in df.columns:
        v = df["volume"].astype(float)
        v_ma = v.rolling(60).mean()
        signed_flow = np.sign(df["ret_1"].fillna(0.0))
        df["pressure"] = signed_flow * (v / (v_ma + 1e-9) - 1.0)
    df["vol_60"] = df["ret_1"].rolling(60).std()
    
    # Drop rows with NaN in critical columns (forward returns)
    critical_cols = ["ret_fwd_log"]
    if "ret_fwd_net_perp" in df.columns:
        critical_cols.append("ret_fwd_net_perp")
    
    before_nan_drop = len(df)
    df = df.dropna(subset=critical_cols)
    nan_dropped = before_nan_drop - len(df)
    if nan_dropped > 0:
        print(f"[NaN Filter] Dropped {nan_dropped} rows with NaN in target columns")
    
    if df.empty:
        raise SystemExit("All data filtered out after NaN removal. Check data quality and horizon settings.")

    out_path = args.out
    if out_path is None:
        base = os.path.basename(args.path).replace("_merged.json", "")
        out_path = os.path.join(os.getcwd(), f"alphagen_panel_{base}_{args.timeframe}_h{args.horizon}.csv")

    df.to_csv(out_path)
    print(f"Wrote: {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
