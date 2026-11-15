"""
Feature builder and simple, picklable OPD forecasting models.

All features use lags of at least 7 days, so a forecast for day ``t`` made on
day ``t - h`` (``h <= 7``) uses only values that were already observed. For
horizons beyond 7 days the forecaster feeds its own predictions back in
(recursive forecasting), which is documented in the model card.

The classes live in this module (not in the training script) so that Joblib
artifacts can be loaded no matter how the training script was launched.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

LAGS = (7, 14, 21, 28)
MIN_HISTORY_DAYS = max(LAGS)

# Educational holiday calendar aligned with the Phase 1 generator.
_HOLIDAYS_2024 = {
    "2024-01-26", "2024-03-08", "2024-03-29", "2024-04-11", "2024-08-15",
    "2024-10-02", "2024-10-12", "2024-11-01", "2024-12-25",
}


def is_holiday(day: pd.Timestamp) -> int:
    """1 if ``day`` is in the project's synthetic holiday calendar.

    Fixed-date holidays (Republic Day, Independence Day, Gandhi Jayanti,
    Christmas) repeat every year; the movable 2024 ones are listed explicitly.
    """
    day = pd.Timestamp(day)
    key = day.strftime("%Y-%m-%d")
    if key in _HOLIDAYS_2024:
        return 1
    return int((day.month, day.day) in {(1, 26), (8, 15), (10, 2), (12, 25)})


FEATURE_COLUMNS = (
    [f"dow_{i}" for i in range(7)]
    + ["is_weekend", "is_holiday", "month_sin", "month_cos"]
    + [f"lag_{k}" for k in LAGS]
    + ["lag_avg_4"]
)


def build_feature_row(day: pd.Timestamp, history: dict) -> dict:
    """Features for ``day`` from ``history`` (mapping Timestamp -> OPD value)."""
    day = pd.Timestamp(day)
    row: dict = {f"dow_{i}": float(day.dayofweek == i) for i in range(7)}
    row["is_weekend"] = float(day.dayofweek >= 5)
    row["is_holiday"] = float(is_holiday(day))
    row["month_sin"] = float(np.sin(2 * np.pi * day.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * day.month / 12))
    lags = []
    for k in LAGS:
        value = history.get(day - pd.Timedelta(days=k), np.nan)
        row[f"lag_{k}"] = float(value)
        lags.append(value)
    row["lag_avg_4"] = float(np.nanmean(lags)) if not np.all(np.isnan(lags)) else np.nan
    return row


def build_feature_frame(dates, history: dict) -> pd.DataFrame:
    return pd.DataFrame([build_feature_row(d, history) for d in dates], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Baselines (same fit/predict interface as scikit-learn estimators)
# ---------------------------------------------------------------------------
class MeanBaseline:
    """Predict the training mean for every day."""

    name = "Training mean"

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class WeekdayMeanBaseline:
    """Predict the training mean of the same day of the week."""

    name = "Day-of-week mean"

    def fit(self, X, y):
        y = np.asarray(y, float)
        dow = np.argmax(X[[f"dow_{i}" for i in range(7)]].to_numpy(), axis=1)
        self.means_ = {d: float(y[dow == d].mean()) for d in range(7) if (dow == d).any()}
        self.fallback_ = float(y.mean())
        return self

    def predict(self, X):
        dow = np.argmax(X[[f"dow_{i}" for i in range(7)]].to_numpy(), axis=1)
        return np.array([self.means_.get(int(d), self.fallback_) for d in dow])


class SeasonalNaiveBaseline:
    """Predict the value observed 7 days earlier (previous week, same weekday)."""

    name = "Previous week, same day"

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.where(X["lag_7"].isna(), self.fallback_, X["lag_7"]).astype(float)


class SeasonalAverageBaseline:
    """Predict the mean of the same weekday over the previous 4 weeks."""

    name = "Same weekday, last 4 weeks average"

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.where(X["lag_avg_4"].isna(), self.fallback_, X["lag_avg_4"]).astype(float)


def make_candidates(seed: int = 42) -> dict:
    """Return the trainable weekday-aware models (fresh, unfitted)."""
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    return {
        "Ridge (calendar + seasonal lags)": Ridge(alpha=1.0),
        "XGBoost (calendar + seasonal lags)": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=seed,
            n_jobs=1,
        ),
    }


def make_baselines() -> dict:
    return {
        cls.name: cls()
        for cls in (
            MeanBaseline,
            WeekdayMeanBaseline,
            SeasonalNaiveBaseline,
            SeasonalAverageBaseline,
        )
    }


def recursive_forecast(model, history: dict, last_date: pd.Timestamp, days: int) -> pd.DataFrame:
    """Forecast ``days`` days after ``last_date`` (predictions feed later lags)."""
    hist = dict(history)
    rows = []
    for step in range(1, days + 1):
        day = pd.Timestamp(last_date) + pd.Timedelta(days=step)
        X = build_feature_frame([day], hist)
        yhat = float(max(0.0, np.asarray(model.predict(X)).ravel()[0]))
        hist[day] = yhat
        rows.append({"date": day, "yhat": yhat})
    return pd.DataFrame(rows)


__all__ = [
    "FEATURE_COLUMNS",
    "LAGS",
    "MIN_HISTORY_DAYS",
    "MeanBaseline",
    "SeasonalAverageBaseline",
    "SeasonalNaiveBaseline",
    "WeekdayMeanBaseline",
    "build_feature_frame",
    "build_feature_row",
    "is_holiday",
    "make_baselines",
    "make_candidates",
    "recursive_forecast",
]
