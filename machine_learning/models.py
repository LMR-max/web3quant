from __future__ import annotations

from typing import Dict, Iterable, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


def build_models(seed: int = 42, only: Optional[Iterable[str]] = None) -> Dict[str, object]:
    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "Lasso": Lasso(alpha=0.001, random_state=seed, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )
    except Exception:
        pass

    if only:
        allow = set(only)
        models = {k: v for k, v in models.items() if k in allow}

    return models
