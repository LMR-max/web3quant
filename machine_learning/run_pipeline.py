from __future__ import annotations

from pathlib import Path
from datetime import datetime

from machine_learning.config import Config, Experiment
from machine_learning.data_loader import load_parquet
from machine_learning.features import build_features
from machine_learning.labels import make_labels
from machine_learning.train import run_experiment_ml, run_experiment_deep
from machine_learning.report import save_results
from quant_backtest.simple_backtest import run_backtest


def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    if not overrides:
        return cfg

    if "data_path" in overrides:
        cfg.data_path = Path(overrides["data_path"])
    if "horizon_steps" in overrides:
        cfg.horizon_steps = int(overrides["horizon_steps"])
    if "rolling_vol_window" in overrides:
        cfg.rolling_vol_window = int(overrides["rolling_vol_window"])
    if "target_threshold_k" in overrides:
        cfg.target_threshold_k = float(overrides["target_threshold_k"])
    if "seq_len" in overrides:
        cfg.seq_len = int(overrides["seq_len"])
    if "batch_size" in overrides:
        cfg.batch_size = int(overrides["batch_size"])
    if "epochs" in overrides:
        cfg.epochs = int(overrides["epochs"])
    if "lr" in overrides:
        cfg.lr = float(overrides["lr"])
    if "pca_components" in overrides:
        cfg.pca_components = int(overrides["pca_components"])
    if "max_rows" in overrides:
        cfg.max_rows = int(overrides["max_rows"]) if overrides["max_rows"] else None
    if "max_runtime_seconds" in overrides:
        cfg.max_runtime_seconds = int(overrides["max_runtime_seconds"]) if overrides["max_runtime_seconds"] else None
    if "clip_quantiles" in overrides:
        cq = overrides["clip_quantiles"]
        if isinstance(cq, (list, tuple)) and len(cq) == 2:
            cfg.clip_quantiles = (float(cq[0]), float(cq[1]))

    # quick mode
    if overrides.get("quick_mode"):
        cfg.max_rows = cfg.max_rows or 20000
        cfg.min_train_size = 24 * 30
        cfg.test_size = 24 * 7
        cfg.step_size = 24 * 7
        cfg.epochs = min(cfg.epochs, 3)
        cfg.experiments = [Experiment(name="ml_baseline", model_type="ml")]

    experiments = overrides.get("experiments")
    if isinstance(experiments, list) and experiments:
        exp_list = []
        for e in experiments:
            if not isinstance(e, dict):
                continue
            exp_list.append(
                Experiment(
                    name=e.get("name") or e.get("deep_model") or e.get("model_type") or "exp",
                    model_type=e.get("model_type", "ml"),
                    models=e.get("models"),
                    deep_model=e.get("deep_model"),
                    use_pca=bool(e.get("use_pca", False)),
                    use_ic_weight=bool(e.get("use_ic_weight", False)),
                )
            )
        if exp_list:
            cfg.experiments = exp_list

    return cfg


def run_pipeline(cfg: Config, progress_cb=None, cancel_cb=None) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now()

    df = load_parquet(Path(cfg.data_path), cfg.time_col)
    if cfg.max_rows:
        df = df.tail(int(cfg.max_rows))
    features = build_features(df, cfg)
    y_reg, y_cls = make_labels(features, cfg)

    summary = {}

    total = len(cfg.experiments)
    for idx, exp in enumerate(cfg.experiments, start=1):
        if cancel_cb and cancel_cb():
            break
        if cfg.max_runtime_seconds:
            elapsed = (datetime.now() - start_ts).total_seconds()
            if elapsed > cfg.max_runtime_seconds:
                break
        if progress_cb:
            progress_cb({"stage": "start", "exp": exp.name, "index": idx, "total": total})
        exp_dir = cfg.out_dir / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if exp.model_type == "ml":
            results = run_experiment_ml(
                features,
                y_reg,
                y_cls,
                cfg,
                use_pca=exp.use_pca,
                use_ic_weight=exp.use_ic_weight,
                models_only=exp.models,
                cancel_cb=cancel_cb,
            )
        else:
            results = run_experiment_deep(
                features,
                y_reg,
                y_cls,
                cfg,
                deep_model=exp.deep_model or "lstm",
                cancel_cb=cancel_cb,
            )

        save_results(results, exp_dir)

        metrics_summary = {k: v.get("metrics", {}) for k, v in results.items()}

        backtest_summary = {}
        equity_paths = {}
        for model_name, payload in results.items():
            preds = payload.get("preds")
            if preds is None or preds.empty:
                continue
            bt = run_backtest(preds["pred"], preds["y_true"], n_quantiles=5)
            equity = bt.pop("equity", None)
            if equity is not None:
                try:
                    path = exp_dir / f"equity_{model_name}.parquet"
                    equity.to_frame("equity").to_parquet(path)
                    equity_paths[model_name] = str(path.as_posix())
                except Exception:
                    pass
            backtest_summary[model_name] = bt

        # save backtest summary
        import json

        with open(exp_dir / "backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(backtest_summary, f, ensure_ascii=False, indent=2)

        summary[exp.name] = {
            "models": list(results.keys()),
            "metrics": metrics_summary,
            "backtest": backtest_summary,
            "equity_paths": equity_paths,
            "metrics_path": str((exp_dir / "metrics.json").as_posix()),
            "backtest_path": str((exp_dir / "backtest_summary.json").as_posix()),
        }
        if progress_cb:
            progress_cb({"stage": "end", "exp": exp.name, "index": idx, "total": total})

    import json

    with open(cfg.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_with_overrides(overrides: dict, progress_cb=None) -> dict:
    cfg = _apply_overrides(Config(), overrides)
    return run_pipeline(cfg, progress_cb=progress_cb, cancel_cb=overrides.get("_cancel_cb"))


def main() -> None:
    run_pipeline(Config())


if __name__ == "__main__":
    main()
