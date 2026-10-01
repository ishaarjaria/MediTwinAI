"""
Phase 2 — Feature engineering for MediTwin AI hospital operations.

Creates modelling-ready features while avoiding obvious target leakage:
- Lag and rolling OPD features use only *past* values (shift before rolling).
- Same-day operational ratios are fine for digital-twin / same-day state views;
  when forecasting future OPD, exclude same-day OPD-derived targets from
  the feature matrix (documented below).

Usage (from meditwin-ai project root):
    python -m src.features.feature_engineering
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import (
    get_project_root,
    preprocess_hospital_data,
    preprocess_patient_data,
    load_hospital_data,
    load_patient_data,
    validate_hospital_processed,
    validate_patient_processed,
)

# Features that use only historical OPD (safe for next-day forecasting).
HISTORICAL_OPD_FEATURES = [
    "lag_1_opd",
    "lag_7_opd",
    "rolling_7_day_opd_mean",
    "rolling_14_day_opd_mean",
]

# Same-day features derived partly from today's patient counts.
# Safe for twin dashboards; drop or shift if predicting today's OPD itself.
SAME_DAY_LOAD_FEATURES = [
    "daily_patient_count",
    "total_patient_load",
    "admission_rate",
    "discharge_rate",
    "medicine_consumption_rate",
    "staff_to_patient_ratio",
    "emergency_ratio",
    "peak_load_indicator",
]


def engineer_hospital_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Phase 2 hospital features to a *cleaned* operations DataFrame.

    Expects a sorted datetime ``date`` column and core operational fields.
    """
    if "date" not in df.columns:
        raise ValueError("engineer_hospital_features requires a 'date' column.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Unparseable dates found - run preprocessing first.")

    out = out.sort_values("date").reset_index(drop=True)

    # --- Calendar features ---
    out["month"] = out["date"].dt.month.astype(int)
    out["quarter"] = out["date"].dt.quarter.astype(int)
    out["is_weekday"] = (out["day_of_week"] < 5).astype(int)
    # Keep is_weekend if present; otherwise derive.
    if "is_weekend" not in out.columns:
        out["is_weekend"] = (1 - out["is_weekday"]).astype(int)

    # --- Load aggregates ---
    out["daily_patient_count"] = (
        out["opd_patients"] + out["emergency_patients"]
    ).astype(int)
    # total_patient_load also includes admissions pressure (inpatients flow).
    out["total_patient_load"] = (
        out["opd_patients"] + out["emergency_patients"] + out["admissions"]
    ).astype(int)

    # Occupancy percentages (reaffirm; preprocessing already recomputed).
    out["bed_occupancy_percentage"] = (
        100.0 * out["occupied_beds"] / out["total_beds"]
    ).round(2)
    out["icu_occupancy_percentage"] = (
        100.0 * out["occupied_icu_beds"] / out["total_icu_beds"]
    ).round(2)

    # --- Rates (guard against divide-by-zero) ---
    daily = out["daily_patient_count"].replace(0, np.nan)
    load = out["total_patient_load"].replace(0, np.nan)
    occupied = out["occupied_beds"].replace(0, np.nan)

    out["admission_rate"] = (out["admissions"] / daily).fillna(0).round(4)
    out["discharge_rate"] = (out["discharges"] / occupied).fillna(0).round(4)
    out["medicine_consumption_rate"] = (
        out["medicine_consumption_units"] / load
    ).fillna(0).round(4)
    out["staff_to_patient_ratio"] = (
        (out["doctors_available"] + out["nurses_available"]) / load
    ).fillna(0).round(4)
    out["emergency_ratio"] = (out["emergency_patients"] / daily).fillna(0).round(4)

    # --- Leakage-safe lag / rolling OPD (past only) ---
    # shift(1) then rolling → window ends yesterday.
    out["lag_1_opd"] = out["opd_patients"].shift(1)
    out["lag_7_opd"] = out["opd_patients"].shift(7)
    out["rolling_7_day_opd_mean"] = (
        out["opd_patients"].shift(1).rolling(window=7, min_periods=1).mean().round(2)
    )
    out["rolling_14_day_opd_mean"] = (
        out["opd_patients"].shift(1).rolling(window=14, min_periods=1).mean().round(2)
    )

    # Peak load: high same-day combined demand vs recent past median load.
    # Uses today's load vs past-only median → limited leakage for twin view.
    past_median_load = (
        out["total_patient_load"].shift(1).rolling(window=14, min_periods=3).median()
    )
    out["peak_load_indicator"] = (
        (out["total_patient_load"] >= 1.25 * past_median_load)
        | (out["bed_occupancy_percentage"] >= 90)
        | (out["icu_occupancy_percentage"] >= 90)
    ).astype(int)
    # First few days without history → 0.
    out["peak_load_indicator"] = out["peak_load_indicator"].fillna(0).astype(int)

    return out


def engineer_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light patient-side features for EDA / later classification.

    Does NOT encode the target into features (no leakage).
    risk_level is preserved as-is for supervised learning later.
    """
    out = df.copy()
    if "risk_level" not in out.columns:
        raise ValueError("patient features require preserved 'risk_level' target.")

    # Pulse pressure and simple vital flags (inputs only — not from risk_level).
    if {"systolic_bp", "diastolic_bp"}.issubset(out.columns):
        out["pulse_pressure"] = (out["systolic_bp"] - out["diastolic_bp"]).astype(int)

    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"],
            bins=[-1, 17, 39, 64, 120],
            labels=["Child", "Young_Adult", "Adult", "Senior"],
        ).astype(str)

    # Binary encodings useful later (still not derived from risk_level).
    if "chronic_disease" in out.columns:
        out["chronic_disease_flag"] = (
            out["chronic_disease"].astype(str).str.title().eq("Yes").astype(int)
        )
    if "emergency_case" in out.columns:
        out["emergency_case_flag"] = (
            out["emergency_case"].astype(str).str.title().eq("Yes").astype(int)
        )

    return out


def validate_no_obvious_leakage(hospital_df: pd.DataFrame) -> list[str]:
    """
    Heuristic checks that lag/rolling features do not equal raw same-day OPD.

    This does not prove absence of all leakage; it catches common mistakes.
    """
    errors: list[str] = []
    if "lag_1_opd" in hospital_df.columns:
        # After row 0, lag_1 should equal previous day's opd — not today's.
        today = hospital_df["opd_patients"].iloc[1:]
        lag = hospital_df["lag_1_opd"].iloc[1:]
        if (lag == today).all():
            errors.append(
                "lag_1_opd equals today's opd_patients for all rows - likely leakage."
            )
        prev = hospital_df["opd_patients"].shift(1).iloc[1:]
        if not np.allclose(lag.to_numpy(dtype=float), prev.to_numpy(dtype=float), equal_nan=True):
            errors.append("lag_1_opd does not match shift(1) of opd_patients.")

    # Rolling mean should not be identical to today's OPD across the board.
    if "rolling_7_day_opd_mean" in hospital_df.columns:
        same = (
            hospital_df["rolling_7_day_opd_mean"] == hospital_df["opd_patients"]
        ).mean()
        if same > 0.9:
            errors.append(
                "rolling_7_day_opd_mean matches opd_patients too often - check shift."
            )
    return errors


def validate_engineered_hospital(df: pd.DataFrame) -> list[str]:
    """Combine capacity checks with leakage heuristics and feature presence."""
    errors = validate_hospital_processed(df)
    required_features = [
        "daily_patient_count",
        "total_patient_load",
        "bed_occupancy_percentage",
        "icu_occupancy_percentage",
        "admission_rate",
        "discharge_rate",
        "medicine_consumption_rate",
        "staff_to_patient_ratio",
        "emergency_ratio",
        "lag_1_opd",
        "lag_7_opd",
        "rolling_7_day_opd_mean",
        "rolling_14_day_opd_mean",
        "peak_load_indicator",
        "is_weekday",
        "is_weekend",
        "month",
        "quarter",
    ]
    for col in required_features:
        if col not in df.columns:
            errors.append(f"Missing engineered feature: {col}")

    for col in ("daily_patient_count", "total_patient_load", "opd_patients", "emergency_patients"):
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative patient-related values in {col}.")

    errors.extend(validate_no_obvious_leakage(df))
    return errors


def run_feature_pipeline(
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Preprocess → engineer features → validate → save processed CSVs.
    """
    output_dir = output_dir or (get_project_root() / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    hospital_raw = load_hospital_data()
    patient_raw = load_patient_data()

    hospital_clean, hospital_report = preprocess_hospital_data(hospital_raw)
    patient_clean, patient_report = preprocess_patient_data(patient_raw)

    hospital_report.print_report()
    patient_report.print_report()

    hospital_feat = engineer_hospital_features(hospital_clean)
    patient_feat = engineer_patient_features(patient_clean)

    hosp_errors = validate_engineered_hospital(hospital_feat)
    pat_errors = validate_patient_processed(patient_feat)
    if "risk_level" not in patient_feat.columns:
        pat_errors.append("risk_level was not preserved.")

    if hosp_errors or pat_errors:
        raise ValueError(
            "Validation failed after feature engineering:\n"
            + "\n".join(f"  hospital: {e}" for e in hosp_errors)
            + "\n".join(f"  patient: {e}" for e in pat_errors)
        )

    # Save clean + feature-rich outputs.
    hospital_clean_path = output_dir / "hospital_operations_clean.csv"
    patient_clean_path = output_dir / "patient_data_clean.csv"
    hospital_feat_path = output_dir / "hospital_operations_features.csv"
    patient_feat_path = output_dir / "patient_data_features.csv"

    hospital_clean.to_csv(hospital_clean_path, index=False)
    patient_clean.to_csv(patient_clean_path, index=False)
    hospital_feat.to_csv(hospital_feat_path, index=False)
    patient_feat.to_csv(patient_feat_path, index=False)

    print("Validation passed:")
    print("  - no impossible bed / ICU occupancy")
    print("  - no negative patient counts")
    print("  - dates sorted ascending")
    print("  - patient risk_level preserved")
    print("  - no obvious lag/rolling leakage")
    print(f"\nSaved:\n  {hospital_clean_path}\n  {patient_clean_path}")
    print(f"  {hospital_feat_path}\n  {patient_feat_path}")
    print(
        "\nLeakage note: lag/rolling OPD features are past-only. "
        "Same-day load features are for twin/EDA; exclude them when the "
        "forecast target is same-day opd_patients."
    )

    return {
        "hospital_clean": hospital_clean,
        "patient_clean": patient_clean,
        "hospital_features": hospital_feat,
        "patient_features": patient_feat,
        "hospital_report": hospital_report.to_dict(),
        "patient_report": patient_report.to_dict(),
        "paths": {
            "hospital_clean": hospital_clean_path,
            "patient_clean": patient_clean_path,
            "hospital_features": hospital_feat_path,
            "patient_features": patient_feat_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediTwin AI Phase 2 feature engineering pipeline."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        run_feature_pipeline(args.output_dir)
    except Exception as exc:
        raise SystemExit(f"Feature engineering failed: {exc}") from exc


if __name__ == "__main__":
    main()
