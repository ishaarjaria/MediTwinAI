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
from sklearn.linear_model import HuberRegressor, LinearRegression
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


def three_way_split(
    df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test (oldest to newest)."""
    n = len(df)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    n_train = n - n_val - n_test
    if n_train < 30:
        raise ValueError("Not enough training rows after chronological split.")
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


def baseline_predictions(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Simple reference predictors fitted on the training block only."""
    y = train_df[TARGET].to_numpy(float)
    out = {"training_mean": np.full(len(eval_df), y.mean())}
    slope, intercept = np.polyfit(train_df["opd_patients"].to_numpy(float), y, 1)
    out["opd_only_linear"] = slope * eval_df["opd_patients"].to_numpy(float) + intercept
    dow_mean = train_df.groupby("day_of_week")[TARGET].mean()
    out["day_of_week_mean"] = eval_df["day_of_week"].map(dow_mean).fillna(y.mean()).to_numpy(float)
    return out


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
    Select on VALIDATION error: lowest MAE, then RMSE, then highest R2. MAE is
    used (not RMSE) because the validation block can contain an injected
    one-day spike whose squared error would dominate the ranking. The test block
    is reported but never used for selection.
    """
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("validation", r)["mae"],
            r.get("validation", r)["rmse"],
            -r.get("validation", r)["r2"],
        ),
    )
    best = ranked[0]
    best["selection_rationale"] = (
        "Selected by lowest MAE on the chronological VALIDATION block, then "
        "RMSE, then highest R2. Reported metrics are from the separate, later "
        "TEST block."
    )
    return best


def _candidate_pipelines(seed: int) -> list[tuple[str, Pipeline]]:
    """Fresh, unfitted candidate pipelines."""
    return [
        (
            "linear_regression",
            Pipeline(
                steps=[
                    ("scale", ColumnTransformer([("num", StandardScaler(), FEATURE_COLUMNS)])),
                    ("reg", LinearRegression()),
                ]
            ),
        ),
        (
            # Robust linear model: the injected one-day spikes inflate waiting
            # time multiplicatively and act as outliers for ordinary least squares.
            "huber_regression",
            Pipeline(
                steps=[
                    ("scale", ColumnTransformer([("num", StandardScaler(), FEATURE_COLUMNS)])),
                    ("reg", HuberRegressor(epsilon=1.35, max_iter=2000)),
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


def train_waiting_time(
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    df = load_hospital_features()
    train_df, val_df, test_df = three_way_split(df, val_size=test_size, test_size=test_size)
    dev_df = pd.concat([train_df, val_df])  # data the final model may see (not test)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    candidates = _candidate_pipelines(seed)
    results: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}

    for name, pipe in candidates:
        pipe.fit(X_train, y_train)  # fit on train only; judged on validation
        val_metrics = regression_metrics(name, y_val, pipe.predict(X_val))
        # Test metrics come from a model fit on train + validation (no test data).
        final = _candidate_pipelines(seed)
        final_pipe = dict(final)[name].fit(dev_df[FEATURE_COLUMNS], dev_df[TARGET])
        metrics = regression_metrics(name, y_test, final_pipe.predict(X_test))
        metrics["validation"] = val_metrics
        results.append(metrics)
        fitted[name] = final_pipe
        print(f"\n=== {name} ===")
        print(
            f"VAL RMSE={val_metrics['rmse']:.4f} | TEST MAE={metrics['mae']:.4f}  "
            f"MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
        )

    best_meta = select_best_regressor(results)
    best_name = best_meta["model"]

    baselines = {}
    base_dev = baseline_predictions(dev_df, test_df)
    for bname, pred in base_dev.items():
        baselines[bname] = regression_metrics(bname, y_test, pred)
    best_baseline = min(baselines.values(), key=lambda r: r["rmse"])
    best_meta["beats_best_baseline"] = bool(best_meta["rmse"] < best_baseline["rmse"])

    # Deploy: refit the selected pipeline on ALL data for the final artifact.
    best_pipe = dict(_candidate_pipelines(seed))[best_name].fit(df[FEATURE_COLUMNS], df[TARGET])

    out_dir = ensure_dir(models_dir() / "waiting_time")
    model_path = out_dir / "best_pipeline.joblib"
    meta_path = out_dir / "metrics.json"

    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "split": "chronological train / validation / test; final model refit on all data",
        "train_end_date": str(train_df["date"].max().date()),
        "validation_start_date": str(val_df["date"].min().date()),
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
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_end_date": artifact["train_end_date"],
        "test_start_date": artifact["test_start_date"],
        "all_models": results,
        "baselines": baselines,
        "best_baseline": best_baseline["model"],
        "selected": best_meta,
        "model_path": model_path.name,
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
