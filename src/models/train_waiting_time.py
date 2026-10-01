"""
Train waiting-time regressors (Linear, Random Forest, XGBoost).

Target: average_waiting_time_minutes

Leakage controls:
- Target column excluded from features.
- Features that are algebraic transforms of the target are not used.
- Same-day OPD / staffing are allowed (they drive waiting time).
- Chronological train/test split (no random shuffle of time order).

Usage:
    python -m src.models.train_waiting_time
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.utils import (
    RANDOM_SEED,
    ensure_dir,
    models_dir,
    processed_dir,
    rmse,
    save_json,
)

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required. Install with: pip install xgboost") from exc

TARGET = "average_waiting_time_minutes"

# Predictors available at the same operational day without using the target.
FEATURE_COLUMNS = [
    "opd_patients",
    "emergency_patients",
    "doctors_available",
    "nurses_available",
    "is_weekend",
    "is_holiday",
    "day_of_week",
    "month",
    "bed_occupancy_percentage",
    "admissions",
]


def load_hospital_features(path=None) -> pd.DataFrame:
    path = path or (processed_dir() / "hospital_operations_features.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.features.feature_engineering"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    needed = FEATURE_COLUMNS + [TARGET, "date"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"Hospital features missing columns: {missing}")
    return df


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent fraction of days (time-safe)."""
    n = len(df)
    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    if n_train < 30:
        raise ValueError("Not enough training rows after chronological split.")
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def regression_metrics(name: str, y_true, y_pred) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "model": name,
        "mae": mae,
        "mse": mse,
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_best_regressor(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Prefer lowest RMSE (penalises large errors), then MAE, then highest R².
    Not accuracy-style 'highest score wins' alone.
    """
    ranked = sorted(
        results,
        key=lambda r: (r["rmse"], r["mae"], -r["r2"]),
    )
    best = ranked[0]
    best["selection_rationale"] = (
        "Selected by lowest RMSE on the chronological holdout, then MAE, "
        "then highest R2. RMSE emphasises large waiting-time misses."
    )
    return best


def train_waiting_time(
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    df = load_hospital_features()
    train_df, test_df = chronological_split(df, test_size=test_size)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    candidates: list[tuple[str, Pipeline]] = [
        (
            "linear_regression",
            Pipeline(
                steps=[
                    (
                        "scale",
                        ColumnTransformer(
                            [("num", StandardScaler(), FEATURE_COLUMNS)]
                        ),
                    ),
                    ("reg", LinearRegression()),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    (
                        "reg",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_leaf=2,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
        ),
        (
            "xgboost",
            Pipeline(
                steps=[
                    (
                        "reg",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=seed,
                            n_jobs=-1,
                            objective="reg:squarederror",
                        ),
                    )
                ]
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = regression_metrics(name, y_test, y_pred)
        results.append(metrics)
        fitted[name] = pipe
        print(f"\n=== {name} ===")
        print(
            f"MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
        )

    best_meta = select_best_regressor(results)
    best_name = best_meta["model"]
    best_pipe = fitted[best_name]

    out_dir = ensure_dir(models_dir() / "waiting_time")
    model_path = out_dir / "best_pipeline.joblib"
    meta_path = out_dir / "metrics.json"

    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "split": "chronological",
        "train_end_date": str(train_df["date"].max().date()),
        "test_start_date": str(test_df["date"].min().date()),
    }
    joblib.dump(artifact, model_path)

    summary = {
        "task": "waiting_time_regression",
        "leakage_note": (
            "Excluded target average_waiting_time_minutes from features. "
            "Used chronological holdout (no shuffle). Same-day OPD/staff "
            "features retained as operational drivers of waiting time."
        ),
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_end_date": artifact["train_end_date"],
        "test_start_date": artifact["test_start_date"],
        "all_models": results,
        "selected": best_meta,
        "model_path": str(model_path),
    }
    save_json(summary, meta_path)

    print(f"\nSelected model: {best_name}")
    print(best_meta["selection_rationale"])
    print(f"Saved -> {model_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train waiting-time regressors.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        train_waiting_time(test_size=args.test_size, seed=args.seed)
    except Exception as exc:
        raise SystemExit(f"Waiting-time training failed: {exc}") from exc


if __name__ == "__main__":
    main()
