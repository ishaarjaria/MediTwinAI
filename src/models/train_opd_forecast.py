"""
OPD time-series forecasting with Prophet.

Chronological train/holdout only — never shuffle time series.

Usage:
    python -m src.models.train_opd_forecast
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.utils import (
    RANDOM_SEED,
    ensure_dir,
    models_dir,
    processed_dir,
    rmse,
    save_json,
)

try:
    from prophet import Prophet
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "prophet is required. Install with: pip install prophet"
    ) from exc

from sklearn.metrics import mean_absolute_error


def load_opd_series(path=None) -> pd.DataFrame:
    path = path or (processed_dir() / "hospital_operations_features.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.features.feature_engineering"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "ds": df["date"],
            "y": df["opd_patients"].astype(float),
            "is_weekend": df["is_weekend"].astype(float),
            "is_holiday": df["is_holiday"].astype(float),
        }
    )
    return out


def chronological_holdout(
    df: pd.DataFrame, holdout_days: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_days < 7:
        raise ValueError("holdout_days should be at least 7.")
    if len(df) <= holdout_days + 60:
        raise ValueError("Series too short for the requested holdout.")
    train = df.iloc[:-holdout_days].copy()
    test = df.iloc[-holdout_days:].copy()
    return train, test


def build_prophet_model(seed: int = RANDOM_SEED) -> Prophet:
    # Prophet uses its own RNG; seed numpy for any auxiliary sampling.
    np.random.seed(seed)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.8,
    )
    model.add_regressor("is_weekend")
    model.add_regressor("is_holiday")
    return model


def _future_calendar(last_date: pd.Timestamp, periods: int) -> pd.DataFrame:
    """Build future frame with weekend/holiday regressors."""
    # Educational holiday set aligned with Phase 1 generator (extend lightly).
    holiday_dates = {
        "2024-01-26",
        "2024-03-08",
        "2024-03-29",
        "2024-04-11",
        "2024-08-15",
        "2024-10-02",
        "2024-10-12",
        "2024-11-01",
        "2024-12-25",
        "2025-01-26",
    }
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq="D")
    rows = []
    for d in future_dates:
        rows.append(
            {
                "ds": d,
                "is_weekend": float(d.dayofweek >= 5),
                "is_holiday": float(d.strftime("%Y-%m-%d") in holiday_dates),
            }
        )
    return pd.DataFrame(rows)


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
    }


def train_opd_forecast(
    holdout_days: int = 30,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    series = load_opd_series()
    train, test = chronological_holdout(series, holdout_days=holdout_days)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_prophet_model(seed=seed)
        model.fit(train[["ds", "y", "is_weekend", "is_holiday"]])

        future_test = test[["ds", "is_weekend", "is_holiday"]].copy()
        forecast_test = model.predict(future_test)

    y_true = test["y"].to_numpy()
    y_pred = forecast_test["yhat"].to_numpy()
    metrics = evaluate_forecast(y_true, y_pred)

    # Refit on full series for deployment forecasting.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_model = build_prophet_model(seed=seed)
        full_model.fit(series[["ds", "y", "is_weekend", "is_holiday"]])

    out_dir = ensure_dir(models_dir() / "opd_forecast")
    model_path = out_dir / "prophet_model.joblib"
    meta_path = out_dir / "metrics.json"

    artifact = {
        "model": full_model,
        "model_name": "prophet",
        "last_train_date": str(series["ds"].max().date()),
        "holdout_days": holdout_days,
        "regressors": ["is_weekend", "is_holiday"],
        "target": "opd_patients",
    }
    joblib.dump(artifact, model_path)

    # Also store holdout predictions for transparency.
    holdout_path = out_dir / "holdout_predictions.csv"
    holdout_df = pd.DataFrame(
        {
            "ds": test["ds"].values,
            "y_true": y_true,
            "yhat": y_pred,
            "yhat_lower": forecast_test["yhat_lower"].to_numpy(),
            "yhat_upper": forecast_test["yhat_upper"].to_numpy(),
        }
    )
    holdout_df.to_csv(holdout_path, index=False)

    summary = {
        "task": "opd_forecasting",
        "model": "prophet",
        "split": "chronological",
        "holdout_days": holdout_days,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_end_date": str(train["ds"].max().date()),
        "test_start_date": str(test["ds"].min().date()),
        "test_end_date": str(test["ds"].max().date()),
        "metrics_holdout": metrics,
        "model_path": str(model_path),
        "holdout_predictions_path": str(holdout_path),
        "note": (
            "Holdout metrics are from a model fit on data before the holdout. "
            "Saved artifact is refit on the full series for future 7-day forecasts."
        ),
    }
    save_json(summary, meta_path)

    print("\n=== Prophet OPD forecast ===")
    print(f"Train days={len(train)}  Holdout days={len(test)}")
    print(f"Holdout MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")
    print(f"Saved model -> {model_path}")
    return summary


def forecast_opd_from_artifact(artifact: dict[str, Any], days: int = 7) -> pd.DataFrame:
    """Produce a multi-day OPD forecast from a saved Prophet artifact."""
    if days < 1:
        raise ValueError("days must be >= 1")
    model: Prophet = artifact["model"]
    last_date = pd.Timestamp(artifact["last_train_date"])
    future = _future_calendar(last_date, periods=days)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fcst = model.predict(future)
    out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["yhat"] = out["yhat"].clip(lower=0).round(1)
    out["yhat_lower"] = out["yhat_lower"].clip(lower=0).round(1)
    out["yhat_upper"] = out["yhat_upper"].clip(lower=0).round(1)
    out = out.rename(columns={"ds": "date", "yhat": "opd_forecast"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prophet OPD forecast model.")
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        train_opd_forecast(holdout_days=args.holdout_days, seed=args.seed)
    except Exception as exc:
        raise SystemExit(f"OPD forecast training failed: {exc}") from exc


if __name__ == "__main__":
    main()
