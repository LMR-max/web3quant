from __future__ import annotations

import argparse
import json
import os
import sys
import glob

# Allow running this file directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor

from alphagen_style.dsl import eval_expr, list_operator_specs, DSLParseError
from alphagen_style.evaluation import (
    EvalConfig,
    TimeSplitConfig,
    RegimeConfig,
    RewardConfig,
    evaluate_factor_panel,
    compute_reward,
    rank_ic,
)

def load_panel_df(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    return df

def process_single_asset(args):
    """
    Worker function to process a single asset file:
    1. Load DF
    2. Eval Expresion -> Factor Series
    3. Return (AssetID, FactorSeries, TargetSeries(optional))
    """
    file_path, expr, target_col = args
    try:
        df = load_panel_df(file_path)
        # Infer asset id from filename if possible?
        # filename convention: alphagen_panel_{SYMBOL}_{...}.csv
        base = os.path.basename(file_path)
        # Simple heuristic: Split by _ and take 2nd/3rd token?
        # E.g. alphagen_panel_BTC_USDT_1m... -> BTC_USDT
        # Or just use filename as ID
        asset_id = base
        
        # Eval
        factor = eval_expr(expr, df)
        
        # Get Target
        target = None
        if target_col and target_col in df.columns:
            target = df[target_col]
        
        return {
            "asset_id": asset_id,
            "timestamp": df.index,
            "factor": factor,
            "target": target
        }
    except Exception as e:
        return {"error": str(e), "file": file_path}

def evaluate_multi_asset(
    file_pattern: str, 
    expr: str, 
    target_col: str = "ret_fwd_log",
    output_path: str = None
):
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found for pattern: {file_pattern}")
        return

    print(f"Found {len(files)} files. Evaluating expression on each...")
    
    # Run in parallel?
    # For simplicity, sequential first or simple loop
    results = []
    
    # Limit max files for safety if too many
    if len(files) > 50:
        print("Warning: Limiting to first 50 files for safety.")
        files = files[:50]

    all_factors = []
    all_targets = []
    
    for f in files:
        res = process_single_asset((f, expr, target_col))
        if "error" in res:
            print(f"Error processing {res['file']}: {res['error']}")
            continue
        
        # Factor
        s_fac = res['factor']
        s_fac.name = res['asset_id']
        all_factors.append(s_fac)
        
        # Target
        if res['target'] is not None:
            s_tar = res['target']
            s_tar.name = res['asset_id']
            all_targets.append(s_tar)

    if not all_factors:
        print("No valid factors generated.")
        return

    # Merge into wide DF
    print("Merging factor panel...")
    factor_panel = pd.concat(all_factors, axis=1)
    
    # Merge targets if available
    target_panel = None
    rank_ic_series = None
    mean_rank_ic = None
    
    if all_targets:
        print("Merging target panel...")
        target_panel = pd.concat(all_targets, axis=1)
        
        # Align (ensure same index)
        common_idx = factor_panel.index.intersection(target_panel.index)
        if len(common_idx) > 0:
            fp = factor_panel.loc[common_idx]
            tp = target_panel.loc[common_idx]
            
            # Compute Rank IC per timestamp
            # corrwith(method='spearman') does column-wise. We want row-wise.
            # user corr(axis=1) method='spearman'
            print("Computing Cross-Sectional Rank IC...")
            # row-wise correlation requires transpose or explicit loop or helper
            # method='spearman' in corr() computes pairwise cols? 
            # corrwith(axis=1) is for aligning cols/rows.
            
            # Correct approach:
            rank_ic_series = fp.corrwith(tp, axis=1, method='spearman')
            mean_rank_ic = rank_ic_series.mean()
            print(f"Mean Rank IC: {mean_rank_ic:.4f}")

    # 1. Basic Stats per asset
    print("Computing stats...")
    
    # 2. Cross-Sectional Rank IC
    # We also need Targets panel
    # Re-read targets?
    # Optimization: In previous loop, we got target.
    # But wait, memory might be issue.
    # Let's just create factor panel first.
    
    # For real cross-sectional IC, we need targets aligned.
    # Let's assume we want to output the "Alpha Matrix".
    
    stats = {
        "n_assets": len(all_factors),
        "start_date": str(factor_panel.index.min()),
        "end_date": str(factor_panel.index.max()),
        "columns": factor_panel.columns.tolist()
    }

    # Cross Sectional Characteristics
    # Mean correlation between assets?
    corr_matrix = factor_panel.corr()
    avg_corr = float(corr_matrix.mean().mean())
    stats["avg_pairwise_corr"] = avg_corr
    
    if mean_rank_ic is not None:
        stats["mean_rank_ic"] = float(mean_rank_ic)
        stats["rank_ic_sharpe"] = float(rank_ic_series.mean() / (rank_ic_series.std() + 1e-9))
    
    # Save results
    if output_path is None:
        output_path = "multi_asset_eval_result.json"
        
    result_pkg = {
        "expr": expr,
        "stats": stats,
        # Preview of last row ranks
        "latest_ranks": factor_panel.iloc[-1].rank(pct=True).sort_values(ascending=False).to_dict()
    }
    
    with open(output_path, "w") as f:
        json.dump(result_pkg, f, indent=2, default=str)
        
    print(f"Saved multi-asset eval to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="Glob pattern for panel CSVs")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--target", default="ret_fwd_log")
    ap.add_argument("--out", default=None)
    
    args = ap.parse_args()
    evaluate_multi_asset(args.pattern, args.expr, args.target, args.out)
