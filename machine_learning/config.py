from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Experiment:
    name: str
    model_type: str = "ml"  # ml | deep
    models: Optional[List[str]] = None  # only for ml
    deep_model: Optional[str] = None  # lstm | transformer
    use_pca: bool = False
    use_ic_weight: bool = False


@dataclass
class Config:
    data_path: Path = Path("data_manager_storage/spot/binance/BTC_USDT/1h/ohlcv_merged.parquet")
    time_col: str = "timestamp"
    horizon_steps: int = 1  # 1h for 1h data
    rolling_vol_window: int = 24
    target_threshold_k: float = 0.5

    features_lookbacks: tuple = (3, 6, 12, 24, 48)

    min_train_size: int = 24 * 60  # 60 days
    test_size: int = 24 * 14  # 14 days
    step_size: int = 24 * 7  # 7 days

    pca_components: int = 10
    seed: int = 42
    out_dir: Path = Path("machine_learning/outputs")

    # deep learning settings
    seq_len: int = 24
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3

    # optional sample cap
    max_rows: int | None = None

    # optional runtime control
    max_runtime_seconds: int | None = None

    # feature clipping
    clip_quantiles: tuple = (0.01, 0.99)

    # rolling/expanding training window
    train_window: int | None = None

    # direction thresholding
    # modes: "rolling_k" | "train_vol_quantile" | "fixed"
    direction_threshold_mode: str = "train_vol_quantile"
    direction_threshold_q: float = 0.6
    fixed_threshold: float = 0.001

    # deep learning training controls
    val_ratio: float = 0.2
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    grad_clip_norm: float = 1.0

    # feature enhancement settings
    return_lags: tuple = (1, 2, 3, 6, 12, 24)
    volume_lookbacks: tuple = (5, 10, 20)
    drawdown_lookbacks: tuple = (24, 72, 168)

    experiments: List[Experiment] = field(
        default_factory=lambda: [
            Experiment(name="ml_baseline", model_type="ml"),
            Experiment(name="ml_pca", model_type="ml", use_pca=True),
            Experiment(name="ml_ic_weight", model_type="ml", use_ic_weight=True),
            Experiment(name="lstm", model_type="deep", deep_model="lstm"),
            Experiment(name="transformer", model_type="deep", deep_model="transformer"),
        ]
    )
