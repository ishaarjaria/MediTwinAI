"""
Central MediTwin AI prediction service (Phase 4).

Loads trained model artifacts **once** and exposes validated, reusable
prediction helpers for:

- patient risk classification
- waiting-time regression
- OPD forecasting (Prophet)
- hospital anomaly detection

This is a software service over saved Joblib artifacts — not a live
IoT / EMR integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.train_anomaly_detection import explain_anomaly
from src.models.train_opd_forecast import forecast_opd_from_artifact
from src.models.train_patient_risk import (
    CATEGORICAL_FEATURES as RISK_CAT_FEATURES,
)
from src.models.train_patient_risk import (
    NUMERIC_FEATURES as RISK_NUM_FEATURES,
)
from src.models.train_waiting_time import FEATURE_COLUMNS as WAIT_FEATURES
from src.models.utils import models_dir

# Plausible educational bounds for input validation (synthetic project ranges).
_PATIENT_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "heart_rate": (30, 220),
    "systolic_bp": (60, 250),
    "diastolic_bp": (30, 150),
    "temperature": (30.0, 43.0),
    "oxygen_saturation": (50, 100),
    "previous_admissions": (0, 50),
    "pulse_pressure": (-50, 200),
}

_HOSPITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "opd_patients": (0, 5000),
    "emergency_patients": (0, 2000),
    "doctors_available": (0, 500),
    "nurses_available": (0, 2000),
    "is_weekend": (0, 1),
    "is_holiday": (0, 1),
    "day_of_week": (0, 6),
    "month": (1, 12),
    "bed_occupancy_percentage": (0, 100),
    "icu_occupancy_percentage": (0, 100),
    "admissions": (0, 2000),
    "average_waiting_time_minutes": (0, 600),
    "occupied_beds": (0, 5000),
    "total_beds": (1, 5000),
    "occupied_icu_beds": (0, 1000),
    "total_icu_beds": (1, 1000),
}

_VALID_GENDERS = {"Male", "Female", "Other"}


class PredictionServiceError(ValueError):
    """Raised when prediction inputs fail validation."""


class PredictionService:
    """
    Load-once facade over Phase 3 model artifacts.

    Prefer a shared instance via ``get_prediction_service()`` so models are
    not reloaded on every call.
    """

    def __init__(self, models_root: Path | None = None) -> None:
        self.models_root = Path(models_root) if models_root else models_dir()
        self._patient_risk: dict[str, Any] | None = None
        self._waiting_time: dict[str, Any] | None = None
        self._opd_forecast: dict[str, Any] | None = None
        self._anomaly: dict[str, Any] | None = None
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_joblib(self, relative: str) -> Any:
        path = self.models_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. "
                "Train models with: python -m src.models.train_all"
            )
        return joblib.load(path)

    def _load_all(self) -> None:
        self._patient_risk = self._load_joblib("patient_risk/best_pipeline.joblib")
        self._waiting_time = self._load_joblib("waiting_time/best_pipeline.joblib")
        self._opd_forecast = self._load_joblib("opd_forecast/prophet_model.joblib")
        self._anomaly = self._load_joblib("anomaly_detection/isolation_forest.joblib")

    @property
    def is_ready(self) -> bool:
        return all(
            x is not None
            for x in (
                self._patient_risk,
                self._waiting_time,
                self._opd_forecast,
                self._anomaly,
            )
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_dataframe(records: dict | list[dict] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy()
        if isinstance(records, dict):
            return pd.DataFrame([records])
        if isinstance(records, list):
            return pd.DataFrame(list(records))
        raise PredictionServiceError(
            "Input must be a dict, list of dicts, or pandas DataFrame."
        )

    @staticmethod
    def _check_bounds(
        df: pd.DataFrame,
        bounds: dict[str, tuple[float, float]],
        context: str,
    ) -> None:
        for col, (low, high) in bounds.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.isna().any():
                raise PredictionServiceError(
                    f"{context}: column '{col}' contains non-numeric or missing values."
                )
            if ((series < low) | (series > high)).any():
                bad = series[(series < low) | (series > high)]
                raise PredictionServiceError(
                    f"{context}: '{col}' out of allowed range [{low}, {high}] "
                    f"(example bad value: {bad.iloc[0]})."
                )

    def _prepare_patient_frame(
        self, patients: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        df = self._as_dataframe(patients)

        if "pulse_pressure" not in df.columns and {
            "systolic_bp",
            "diastolic_bp",
        }.issubset(df.columns):
            df["pulse_pressure"] = (
                pd.to_numeric(df["systolic_bp"], errors="coerce")
                - pd.to_numeric(df["diastolic_bp"], errors="coerce")
            )

        if "chronic_disease_flag" not in df.columns and "chronic_disease" in df.columns:
            df["chronic_disease_flag"] = (
                df["chronic_disease"].astype(str).str.title().eq("Yes").astype(int)
            )
        if "emergency_case_flag" not in df.columns and "emergency_case" in df.columns:
            df["emergency_case_flag"] = (
                df["emergency_case"].astype(str).str.title().eq("Yes").astype(int)
            )

        required = list(RISK_NUM_FEATURES) + list(RISK_CAT_FEATURES)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise PredictionServiceError(
                f"Patient risk input missing required fields: {missing}"
            )

        if "gender" in df.columns:
            genders = df["gender"].astype(str).str.strip().str.title()
            invalid = ~genders.isin(_VALID_GENDERS)
            if invalid.any():
                raise PredictionServiceError(
                    f"Invalid gender values (allowed: {sorted(_VALID_GENDERS)})."
                )
            df["gender"] = genders

        for flag in ("chronic_disease_flag", "emergency_case_flag"):
            if flag in df.columns:
                vals = pd.to_numeric(df[flag], errors="coerce")
                if vals.isna().any() or ((vals != 0) & (vals != 1)).any():
                    raise PredictionServiceError(
                        f"Patient risk: '{flag}' must be 0 or 1."
                    )

        self._check_bounds(df, _PATIENT_BOUNDS, "Patient risk")
        return df

    def _prepare_waiting_frame(
        self, days: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        df = self._as_dataframe(days)
        artifact = self._waiting_time
        assert artifact is not None
        cols = artifact["feature_columns"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise PredictionServiceError(
                f"Waiting-time input missing required fields: {missing}"
            )
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[cols].isna().any().any():
            raise PredictionServiceError(
                "Waiting-time input contains non-numeric or missing feature values."
            )
        self._check_bounds(df, _HOSPITAL_BOUNDS, "Waiting-time")
        return df

    def _prepare_anomaly_frame(
        self, days: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        df = self._as_dataframe(days)
        artifact = self._anomaly
        assert artifact is not None
        cols = artifact["feature_columns"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise PredictionServiceError(
                f"Anomaly detection input missing required fields: {missing}"
            )
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[cols].isna().any().any():
            raise PredictionServiceError(
                "Anomaly input contains non-numeric or missing feature values."
            )
        self._check_bounds(df, _HOSPITAL_BOUNDS, "Anomaly detection")
        return df

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------
    def predict_patient_risk(
        self, patients: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        """Predict Low/Medium/High risk with optional class probabilities."""
        assert self._patient_risk is not None
        artifact = self._patient_risk
        pipe = artifact["pipeline"]
        df = self._prepare_patient_frame(patients)
        feature_cols = artifact["feature_columns"]
        X = df[feature_cols]

        raw_pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None

        if artifact.get("uses_encoded_labels"):
            idx_to_label = {int(k): v for k, v in artifact["idx_to_label"].items()}
            labels = [idx_to_label[int(i)] for i in raw_pred]
            class_order = artifact.get("class_order", ["Low", "Medium", "High"])
        else:
            labels = list(raw_pred)
            class_order = list(pipe.named_steps["clf"].classes_)

        out = df.copy()
        out["predicted_risk_level"] = labels
        if proba is not None:
            for i, cls in enumerate(class_order):
                out[f"proba_{cls}"] = proba[:, i]
        return out

    def predict_waiting_time(
        self, days: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        """Predict average waiting time (minutes) for hospital-day rows."""
        assert self._waiting_time is not None
        artifact = self._waiting_time
        pipe = artifact["pipeline"]
        df = self._prepare_waiting_frame(days)
        preds = pipe.predict(df[artifact["feature_columns"]])
        out = df.copy()
        out["predicted_waiting_time_minutes"] = np.clip(np.round(preds, 2), 0, None)
        return out

    def forecast_opd(self, days: int = 7) -> pd.DataFrame:
        """Return a multi-day OPD demand forecast from the saved Prophet model."""
        if not isinstance(days, int) or isinstance(days, bool):
            raise PredictionServiceError("forecast_opd: 'days' must be an integer.")
        if days < 1 or days > 90:
            raise PredictionServiceError(
                "forecast_opd: 'days' must be between 1 and 90 (inclusive)."
            )
        assert self._opd_forecast is not None
        return forecast_opd_from_artifact(self._opd_forecast, days=days)

    def detect_hospital_anomaly(
        self, days: dict | list[dict] | pd.DataFrame
    ) -> pd.DataFrame:
        """
        Flag unusual operational days.

        Returns anomaly_flag (0/1), anomaly_score (higher = more anomalous),
        and anomaly_reason when flagged.
        """
        assert self._anomaly is not None
        artifact = self._anomaly
        model = artifact["model"]
        scaler = artifact["scaler"]
        cols = artifact["feature_columns"]

        df = self._prepare_anomaly_frame(days)
        X = df[cols].astype(float)
        X_scaled = scaler.transform(X)
        raw = model.decision_function(X_scaled)
        scores = (-raw).astype(float)
        flags = (model.predict(X_scaled) == -1).astype(int)

        means = pd.Series(artifact["feature_means"])
        stds = pd.Series(artifact["feature_stds"])
        reasons = [
            explain_anomaly(X.iloc[i], means, stds) if flags[i] == 1 else ""
            for i in range(len(X))
        ]

        out = df.copy()
        out["anomaly_flag"] = flags
        out["anomaly_score"] = np.round(scores, 4)
        out["anomaly_reason"] = reasons
        return out


# Module-level singleton — models load at most once per process.
_SERVICE: PredictionService | None = None


def get_prediction_service(
    models_root: Path | None = None,
    *,
    reload: bool = False,
) -> PredictionService:
    """Return the shared PredictionService (loads artifacts on first use)."""
    global _SERVICE
    if _SERVICE is None or reload:
        _SERVICE = PredictionService(models_root=models_root)
    return _SERVICE


__all__ = [
    "PredictionService",
    "PredictionServiceError",
    "get_prediction_service",
    "WAIT_FEATURES",
    "RISK_NUM_FEATURES",
    "RISK_CAT_FEATURES",
]
