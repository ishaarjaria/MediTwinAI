"""
Inference helpers for MediTwin AI.

Phase 4: thin wrappers around the shared ``PredictionService`` so models
are loaded only once per process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.prediction_service import get_prediction_service
from src.models.train_anomaly_detection import FEATURE_COLUMNS as ANOMALY_FEATURES
from src.models.train_patient_risk import (
    CATEGORICAL_FEATURES as RISK_CAT,
)
from src.models.train_patient_risk import (
    NUMERIC_FEATURES as RISK_NUM,
)
from src.models.train_waiting_time import FEATURE_COLUMNS as WAIT_FEATURES


def predict_patient_risk(
    patients: dict | list[dict] | pd.DataFrame,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Predict Low/Medium/High patient risk."""
    if model_path is not None:
        # Custom artifact path: build a one-off service rooted at parent/parent.
        # Prefer the shared service for normal use.
        from src.models.prediction_service import PredictionService

        root = Path(model_path).resolve().parents[1]
        service = PredictionService(models_root=root)
        return service.predict_patient_risk(patients)
    return get_prediction_service().predict_patient_risk(patients)


def predict_waiting_time(
    days: dict | list[dict] | pd.DataFrame,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Predict average waiting time in minutes."""
    if model_path is not None:
        from src.models.prediction_service import PredictionService

        root = Path(model_path).resolve().parents[1]
        service = PredictionService(models_root=root)
        return service.predict_waiting_time(days)
    return get_prediction_service().predict_waiting_time(days)


def forecast_opd(
    days: int = 7,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Return a multi-day OPD forecast."""
    if model_path is not None:
        from src.models.prediction_service import PredictionService

        root = Path(model_path).resolve().parents[1]
        service = PredictionService(models_root=root)
        return service.forecast_opd(days=days)
    return get_prediction_service().forecast_opd(days=days)


def detect_hospital_anomaly(
    days: dict | list[dict] | pd.DataFrame,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Detect unusual hospital operational days."""
    if model_path is not None:
        from src.models.prediction_service import PredictionService

        root = Path(model_path).resolve().parents[1]
        service = PredictionService(models_root=root)
        return service.detect_hospital_anomaly(days)
    return get_prediction_service().detect_hospital_anomaly(days)


__all__ = [
    "predict_patient_risk",
    "predict_waiting_time",
    "forecast_opd",
    "detect_hospital_anomaly",
    "RISK_NUM",
    "RISK_CAT",
    "WAIT_FEATURES",
    "ANOMALY_FEATURES",
]
