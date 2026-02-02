from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import Config
from .splits import walk_forward_splits
from .models import build_models
from .deep_models import get_model

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _ic_weights(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ic = x.corrwith(y, method="spearman")
    ic = ic.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if ic.abs().sum() == 0:
        return ic
    return ic / ic.abs().sum()


def _direction_from_pred(pred: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.where(pred > threshold, 1, np.where(pred < -threshold, -1, 0))


def _apply_fold_clip(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    clip_quantiles: tuple | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not clip_quantiles:
        return x_train, x_test
    ql, qh = clip_quantiles
    lower = x_train.quantile(ql)
    upper = x_train.quantile(qh)
    return (
        x_train.clip(lower=lower, upper=upper, axis=1),
        x_test.clip(lower=lower, upper=upper, axis=1),
    )


def _fold_threshold(
    x_train_ret: pd.Series,
    x_all_ret: pd.Series,
    test_slice: slice,
    cfg: Config,
    test_len: int,
) -> np.ndarray:
    mode = cfg.direction_threshold_mode
    if mode == "fixed":
        return np.full(test_len, cfg.fixed_threshold)
    if mode == "train_vol_quantile":
        train_vol = x_train_ret.rolling(cfg.rolling_vol_window).std().dropna()
        base = train_vol.quantile(cfg.direction_threshold_q) if not train_vol.empty else np.nan
        if base != base or base == 0:
            base = cfg.fixed_threshold
        return np.full(test_len, cfg.target_threshold_k * base)

    # fallback: rolling k on full series (uses only past window)
    rolling_vol = x_all_ret.rolling(cfg.rolling_vol_window).std().iloc[test_slice].values
    return cfg.target_threshold_k * rolling_vol


def run_experiment_ml(
    x: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    cfg: Config,
    use_pca: bool = False,
    use_ic_weight: bool = False,
    models_only: Optional[List[str]] = None,
    cancel_cb=None,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    data = pd.concat([x, y_reg, y_cls], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    x = data[x.columns]
    y_reg = data["future_ret"]
    y_cls = data["direction"]

    models = build_models(cfg.seed, only=models_only)

    for model_name, model in models.items():
        fold_metrics: List[Dict] = []
        preds_all = []

        for train_slice, test_slice in walk_forward_splits(
            n_samples=len(x),
            min_train_size=cfg.min_train_size,
            test_size=cfg.test_size,
            step_size=cfg.step_size,
            train_window=cfg.train_window,
        ):
            if cancel_cb and cancel_cb():
                break
            x_train, x_test = x.iloc[train_slice], x.iloc[test_slice]
            y_train, y_test = y_reg.iloc[train_slice], y_reg.iloc[test_slice]

            x_train, x_test = _apply_fold_clip(x_train, x_test, cfg.clip_quantiles)

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            if use_pca:
                pca = PCA(n_components=min(cfg.pca_components, x_train_s.shape[1]))
                x_train_s = pca.fit_transform(x_train_s)
                x_test_s = pca.transform(x_test_s)

            model.fit(x_train_s, y_train)
            pred = model.predict(x_test_s)

            # direction evaluation using volatility threshold
            threshold = _fold_threshold(
                x_train_ret=x_train["ret_1"],
                x_all_ret=x["ret_1"],
                test_slice=test_slice,
                cfg=cfg,
                test_len=len(y_test),
            )
            direction_pred = _direction_from_pred(pred, threshold)
            direction_true = _direction_from_pred(y_test.values, threshold)

            fold_metrics.append(
                {
                    "mae": mean_absolute_error(y_test, pred),
                    "rmse": mean_squared_error(y_test, pred, squared=False),
                    "r2": r2_score(y_test, pred),
                    "dir_acc": accuracy_score(direction_true, direction_pred),
                    "dir_f1": f1_score(direction_true, direction_pred, average="macro"),
                }
            )

            preds_all.append(
                pd.DataFrame(
                    {
                        "pred": pred,
                        "y_true": y_test.values,
                        "direction_true": direction_true,
                        "direction_pred": direction_pred,
                    },
                    index=y_test.index,
                )
            )

        preds_df = pd.concat(preds_all).sort_index()
        base_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        base_metrics.update(_ic_stats(preds_df))
        results[model_name] = {
            "metrics": base_metrics,
            "preds": preds_df,
        }

    if use_ic_weight:
        fold_metrics = []
        preds_all = []
        for train_slice, test_slice in walk_forward_splits(
            n_samples=len(x),
            min_train_size=cfg.min_train_size,
            test_size=cfg.test_size,
            step_size=cfg.step_size,
            train_window=cfg.train_window,
        ):
            if cancel_cb and cancel_cb():
                break
            x_train, x_test = x.iloc[train_slice], x.iloc[test_slice]
            y_train, y_test = y_reg.iloc[train_slice], y_reg.iloc[test_slice]

            x_train, x_test = _apply_fold_clip(x_train, x_test, cfg.clip_quantiles)

            scaler = StandardScaler()
            x_train_s = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
            x_test_s = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

            weights = _ic_weights(x_train_s, y_train)
            pred = (x_test_s * weights).sum(axis=1).values

            threshold = _fold_threshold(
                x_train_ret=x_train["ret_1"],
                x_all_ret=x["ret_1"],
                test_slice=test_slice,
                cfg=cfg,
                test_len=len(y_test),
            )
            direction_pred = _direction_from_pred(pred, threshold)
            direction_true = _direction_from_pred(y_test.values, threshold)

            fold_metrics.append(
                {
                    "mae": mean_absolute_error(y_test, pred),
                    "rmse": mean_squared_error(y_test, pred, squared=False),
                    "r2": r2_score(y_test, pred),
                    "dir_acc": accuracy_score(direction_true, direction_pred),
                    "dir_f1": f1_score(direction_true, direction_pred, average="macro"),
                }
            )

            preds_all.append(
                pd.DataFrame(
                    {
                        "pred": pred,
                        "y_true": y_test.values,
                        "direction_true": direction_true,
                        "direction_pred": direction_pred,
                    },
                    index=y_test.index,
                )
            )

        preds_df = pd.concat(preds_all).sort_index()
        base_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        base_metrics.update(_ic_stats(preds_df))
        results["IC_WEIGHT"] = {
            "metrics": base_metrics,
            "preds": preds_df,
        }

    return results


def _make_sequences(x: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len : i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def _make_test_sequences(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if seq_len <= 1:
        return x_test[:, None, :], y_test

    prefix_len = seq_len - 1
    prefix = x_train[-prefix_len:] if len(x_train) >= prefix_len else x_train
    x_combo = np.vstack([prefix, x_test]) if len(prefix) else x_test

    xs = []
    ys = []
    for j in range(len(y_test)):
        start = j
        end = j + seq_len
        if end > len(x_combo):
            break
        xs.append(x_combo[start:end])
        ys.append(y_test[j])
    return np.array(xs), np.array(ys)


def _ic_stats(preds: pd.DataFrame) -> Dict[str, float]:
    try:
        ic = preds["pred"].corr(preds["y_true"], method="spearman")
        rolling = preds["pred"].rolling(100).corr(preds["y_true"], method="spearman")
        icir = rolling.mean() / rolling.std() if rolling.std() not in (0, np.nan) else np.nan
        rolling = rolling.dropna()
        if len(rolling) > 1:
            ic_mean = float(rolling.mean())
            ic_std = float(rolling.std())
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(rolling))) if ic_std > 0 else np.nan
            ic_ci_low = ic_mean - 1.96 * ic_std / np.sqrt(len(rolling)) if ic_std > 0 else np.nan
            ic_ci_high = ic_mean + 1.96 * ic_std / np.sqrt(len(rolling)) if ic_std > 0 else np.nan
        else:
            ic_mean = np.nan
            ic_tstat = np.nan
            ic_ci_low = np.nan
            ic_ci_high = np.nan
        return {
            "ic": float(ic) if ic == ic else np.nan,
            "icir": float(icir) if icir == icir else np.nan,
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "ic_ci_low": ic_ci_low,
            "ic_ci_high": ic_ci_high,
            "ic_n": float(len(rolling)),
        }
    except Exception:
        return {"ic": np.nan, "icir": np.nan}


def _train_deep_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    grad_clip_norm: float,
    patience: int,
    min_delta: float,
    cancel_cb=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_state = None
    best_val = np.inf
    bad_epochs = 0

    model.train()
    for _ in range(epochs):
        if cancel_cb and cancel_cb():
            break
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_losses.append(criterion(pred, yb).item())
            val_loss = float(np.mean(val_losses)) if val_losses else np.inf
            model.train()

            if val_loss + min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_deep(model, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32).to(device)
        preds = model(xb).detach().cpu().numpy()
    return preds


def run_experiment_deep(
    x: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    cfg: Config,
    deep_model: str,
    cancel_cb=None,
) -> Dict[str, Dict]:
    if torch is None:
        return {
            deep_model: {
                "metrics": {"error": "torch_not_installed"},
                "preds": pd.DataFrame(),
            }
        }

    results: Dict[str, Dict] = {}

    data = pd.concat([x, y_reg, y_cls], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    x = data[x.columns]
    y_reg = data["future_ret"]
    y_cls = data["direction"]

    fold_metrics: List[Dict] = []
    preds_all = []

    for train_slice, test_slice in walk_forward_splits(
        n_samples=len(x),
        min_train_size=cfg.min_train_size,
        test_size=cfg.test_size,
        step_size=cfg.step_size,
        train_window=cfg.train_window,
    ):
        if cancel_cb and cancel_cb():
            break
        x_train, x_test = x.iloc[train_slice], x.iloc[test_slice]
        y_train, y_test = y_reg.iloc[train_slice], y_reg.iloc[test_slice]

        x_train, x_test = _apply_fold_clip(x_train, x_test, cfg.clip_quantiles)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        x_train_seq, y_train_seq = _make_sequences(x_train_s, y_train.values, cfg.seq_len)
        x_test_seq, y_test_seq = _make_test_sequences(x_train_s, x_test_s, y_test.values, cfg.seq_len)

        if len(x_train_seq) == 0 or len(x_test_seq) == 0:
            continue

        val_size = int(len(x_train_seq) * cfg.val_ratio)
        if val_size > 0 and len(x_train_seq) - val_size >= 1:
            train_data = TensorDataset(
                torch.tensor(x_train_seq[:-val_size], dtype=torch.float32),
                torch.tensor(y_train_seq[:-val_size], dtype=torch.float32),
            )
            val_data = TensorDataset(
                torch.tensor(x_train_seq[-val_size:], dtype=torch.float32),
                torch.tensor(y_train_seq[-val_size:], dtype=torch.float32),
            )
            val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)
        else:
            train_data = TensorDataset(
                torch.tensor(x_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
            )
            val_loader = None

        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

        model = get_model(deep_model, input_dim=x_train_seq.shape[-1])
        model = _train_deep_model(
            model,
            train_loader,
            val_loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
            grad_clip_norm=cfg.grad_clip_norm,
            patience=cfg.early_stop_patience,
            min_delta=cfg.early_stop_min_delta,
            cancel_cb=cancel_cb,
        )

        pred = _predict_deep(model, x_test_seq)

        # align test indices with sequence targets
        y_test_aligned = pd.Series(y_test_seq, index=y_test.index[: len(y_test_seq)])

        threshold = _fold_threshold(
            x_train_ret=x_train["ret_1"],
            x_all_ret=x["ret_1"],
            test_slice=test_slice,
            cfg=cfg,
            test_len=len(y_test_aligned),
        )
        direction_pred = _direction_from_pred(pred, threshold)
        direction_true = _direction_from_pred(y_test_aligned.values, threshold)

        fold_metrics.append(
            {
                "mae": mean_absolute_error(y_test_aligned, pred),
                "rmse": mean_squared_error(y_test_aligned, pred, squared=False),
                "r2": r2_score(y_test_aligned, pred),
                "dir_acc": accuracy_score(direction_true, direction_pred),
                "dir_f1": f1_score(direction_true, direction_pred, average="macro"),
            }
        )

        preds_all.append(
            pd.DataFrame(
                {
                    "pred": pred,
                    "y_true": y_test_aligned.values,
                    "direction_true": direction_true,
                    "direction_pred": direction_pred,
                },
                index=y_test_aligned.index,
            )
        )

    preds_df = pd.concat(preds_all).sort_index() if preds_all else pd.DataFrame()
    metrics = pd.DataFrame(fold_metrics).mean().to_dict() if fold_metrics else {}
    if not preds_df.empty:
        metrics.update(_ic_stats(preds_df))
    results[deep_model] = {
        "metrics": metrics,
        "preds": preds_df,
    }

    return results
