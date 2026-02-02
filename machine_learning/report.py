from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v["metrics"] for k, v in results.items()}
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(out_dir / "metrics.csv", index=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # save predictions per model
    for model_name, payload in results.items():
        preds = payload.get("preds")
        if preds is not None:
            preds.to_parquet(out_dir / f"preds_{model_name}.parquet")

            # rolling IC (Spearman)
            try:
                window = 100
                ic_series = preds["pred"].rolling(window).corr(preds["y_true"], method="spearman")
                ic_series.to_frame("ic").to_parquet(out_dir / f"ic_{model_name}.parquet")
            except Exception:
                pass
