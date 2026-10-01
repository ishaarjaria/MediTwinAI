"""
Phase 2 — Data preprocessing for MediTwin AI.

Cleans hospital operational and patient datasets, prints a data-quality
report, and saves cleaned frames under data/processed/.

IMPORTANT: Never silently drops a large share of rows. Severe cleaning
issues raise errors or are reported clearly.

Usage (from meditwin-ai project root):
    python -m src.data.preprocess
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Maximum fraction of rows that may be removed before we abort.
MAX_DROP_FRACTION = 0.05

HOSPITAL_NUMERIC_NONNEG = [
    "opd_patients",
    "emergency_patients",
    "admissions",
    "discharges",
    "total_beds",
    "occupied_beds",
    "available_beds",
    "bed_occupancy_percentage",
    "total_icu_beds",
    "occupied_icu_beds",
    "icu_occupancy_percentage",
    "doctors_available",
    "nurses_available",
    "average_waiting_time_minutes",
    "medicine_stock_units",
    "medicine_consumption_units",
    "equipment_availability_percentage",
]

PATIENT_NUMERIC = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "oxygen_saturation",
    "previous_admissions",
]

RISK_LEVELS = {"Low", "Medium", "High"}
GENDERS = {"Male", "Female", "Other"}
YES_NO = {"Yes", "No"}
SEASONS = {"Winter", "Spring", "Monsoon", "Autumn"}


def get_project_root() -> Path:
    """Return the meditwin-ai project root."""
    return Path(__file__).resolve().parents[2]


def _default_paths() -> dict[str, Path]:
    root = get_project_root()
    return {
        "hospital_raw": root / "data" / "synthetic" / "hospital_operations.csv",
        "patient_raw": root / "data" / "synthetic" / "patient_data.csv",
        "processed_dir": root / "data" / "processed",
    }


class DataQualityReport:
    """Collects and prints preprocessing diagnostics."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.rows_before = 0
        self.rows_after = 0
        self.notes: list[str] = []
        self.fixes: list[str] = []
        self.warnings: list[str] = []

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def add_fix(self, message: str) -> None:
        self.fixes.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        dropped = self.rows_before - self.rows_after
        drop_pct = (
            100.0 * dropped / self.rows_before if self.rows_before else 0.0
        )
        return {
            "dataset": self.dataset_name,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": dropped,
            "drop_percentage": round(drop_pct, 2),
            "fixes": self.fixes,
            "warnings": self.warnings,
            "notes": self.notes,
        }

    def print_report(self) -> None:
        info = self.to_dict()
        print("\n" + "=" * 60)
        print(f"DATA QUALITY REPORT - {info['dataset']}")
        print("=" * 60)
        print(f"Rows before : {info['rows_before']}")
        print(f"Rows after  : {info['rows_after']}")
        print(
            f"Rows dropped: {info['rows_dropped']} "
            f"({info['drop_percentage']}%)"
        )
        if self.fixes:
            print("\nFixes applied:")
            for item in self.fixes:
                print(f"  - {item}")
        if self.warnings:
            print("\nWarnings:")
            for item in self.warnings:
                print(f"  - {item}")
        if self.notes:
            print("\nNotes:")
            for item in self.notes:
                print(f"  - {item}")
        print("=" * 60 + "\n")


def _assert_drop_budget(report: DataQualityReport) -> None:
    """Abort if cleaning would silently remove too much data."""
    if report.rows_before == 0:
        raise ValueError(f"{report.dataset_name}: empty dataset.")
    dropped = report.rows_before - report.rows_after
    fraction = dropped / report.rows_before
    if fraction > MAX_DROP_FRACTION:
        raise ValueError(
            f"{report.dataset_name}: refusing to drop {fraction:.1%} of rows "
            f"(limit {MAX_DROP_FRACTION:.0%}). Investigate source data."
        )


def load_hospital_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw hospital operations CSV."""
    path = path or _default_paths()["hospital_raw"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Hospital data not found at {path}. "
            "Run: python -m src.data.generate_synthetic_data"
        )
    return pd.read_csv(path)


def load_patient_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw patient CSV."""
    path = path or _default_paths()["patient_raw"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Patient data not found at {path}. "
            "Run: python -m src.data.generate_synthetic_data"
        )
    return pd.read_csv(path)


def preprocess_hospital_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean hospital operational data.

    Handles missing values, duplicates, invalid numbers, date parsing,
    categoricals, impossible occupancy, and inconsistent bed arithmetic.
    """
    report = DataQualityReport("hospital_operations")
    report.rows_before = len(df)
    out = df.copy()

    # --- Date parsing ---
    if "date" not in out.columns:
        raise ValueError("Hospital data missing required 'date' column.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(out["date"].isna().sum())
    if bad_dates:
        # Do not silently drop many rows; fail loudly if dates are broken.
        if bad_dates / len(out) > MAX_DROP_FRACTION:
            raise ValueError(
                f"Unparseable dates in {bad_dates} rows - investigate source."
            )
        out = out.dropna(subset=["date"])
        report.add_fix(f"Dropped {bad_dates} rows with unparseable dates.")

    # --- Duplicates ---
    dup_count = int(out.duplicated().sum())
    date_dups = int(out.duplicated(subset=["date"]).sum())
    if dup_count:
        out = out.drop_duplicates()
        report.add_fix(f"Removed {dup_count} fully duplicate rows.")
    if date_dups:
        # Keep first occurrence per date after sorting.
        before = len(out)
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="first")
        removed = before - len(out)
        if removed:
            report.add_fix(f"Removed {removed} extra rows sharing the same date.")

    # --- Categorical normalisation ---
    if "season" in out.columns:
        out["season"] = out["season"].astype(str).str.strip().str.title()
        # Map common aliases if any appear.
        season_map = {
            "Fall": "Autumn",
            "Summer": "Monsoon",  # educational mapping for this dataset
        }
        out["season"] = out["season"].replace(season_map)
        invalid_season = ~out["season"].isin(SEASONS)
        n_invalid_season = int(invalid_season.sum())
        if n_invalid_season:
            # Impute from month rather than dropping.
            month = out["date"].dt.month
            imputed = month.map(
                lambda m: (
                    "Winter"
                    if m in (12, 1, 2)
                    else "Spring"
                    if m in (3, 4, 5)
                    else "Monsoon"
                    if m in (6, 7, 8, 9)
                    else "Autumn"
                )
            )
            out.loc[invalid_season, "season"] = imputed[invalid_season]
            report.add_fix(
                f"Imputed season from month for {n_invalid_season} invalid values."
            )

    for flag in ("is_weekend", "is_holiday"):
        if flag in out.columns:
            out[flag] = (
                out[flag]
                .map({True: 1, False: 0, "1": 1, "0": 0, 1: 1, 0: 0})
                .fillna(0)
                .astype(int)
            )

    # --- Missing values (numeric): median impute, never mass-drop ---
    for col in HOSPITAL_NUMERIC_NONNEG:
        if col not in out.columns:
            report.add_warning(f"Expected column missing: {col}")
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        missing = int(out[col].isna().sum())
        if missing:
            median_val = out[col].median()
            if pd.isna(median_val):
                median_val = 0
            out[col] = out[col].fillna(median_val)
            report.add_fix(
                f"Imputed {missing} missing values in '{col}' with median={median_val}."
            )

    # Ensure day_of_week aligns with date if present / missing.
    if "day_of_week" not in out.columns or out["day_of_week"].isna().any():
        out["day_of_week"] = out["date"].dt.dayofweek
        report.add_fix("Recomputed day_of_week from date.")
    else:
        out["day_of_week"] = out["day_of_week"].astype(int)

    # Sync weekend flag with day_of_week for consistency.
    recomputed_weekend = (out["day_of_week"] >= 5).astype(int)
    if "is_weekend" in out.columns:
        mismatch = int((out["is_weekend"] != recomputed_weekend).sum())
        if mismatch:
            out["is_weekend"] = recomputed_weekend
            report.add_fix(
                f"Corrected is_weekend for {mismatch} rows to match day_of_week."
            )
    else:
        out["is_weekend"] = recomputed_weekend
        report.add_fix("Created is_weekend from day_of_week.")

    # --- Invalid / impossible numerical values ---
    # Negatives → clip to 0 (counts) with reporting.
    count_cols = [
        "opd_patients",
        "emergency_patients",
        "admissions",
        "discharges",
        "occupied_beds",
        "available_beds",
        "occupied_icu_beds",
        "doctors_available",
        "nurses_available",
        "medicine_stock_units",
        "medicine_consumption_units",
        "average_waiting_time_minutes",
    ]
    for col in count_cols:
        if col not in out.columns:
            continue
        n_neg = int((out[col] < 0).sum())
        if n_neg:
            out[col] = out[col].clip(lower=0)
            report.add_fix(f"Clipped {n_neg} negative values in '{col}' to 0.")

    # Capacity floors.
    if "total_beds" in out.columns:
        n_bad_beds = int((out["total_beds"] <= 0).sum())
        if n_bad_beds:
            mode_beds = out.loc[out["total_beds"] > 0, "total_beds"].mode()
            fill_beds = int(mode_beds.iloc[0]) if len(mode_beds) else 250
            out.loc[out["total_beds"] <= 0, "total_beds"] = fill_beds
            report.add_fix(
                f"Replaced {n_bad_beds} non-positive total_beds with {fill_beds}."
            )

    if "total_icu_beds" in out.columns:
        n_bad_icu = int((out["total_icu_beds"] <= 0).sum())
        if n_bad_icu:
            mode_icu = out.loc[out["total_icu_beds"] > 0, "total_icu_beds"].mode()
            fill_icu = int(mode_icu.iloc[0]) if len(mode_icu) else 30
            out.loc[out["total_icu_beds"] <= 0, "total_icu_beds"] = fill_icu
            report.add_fix(
                f"Replaced {n_bad_icu} non-positive total_icu_beds with {fill_icu}."
            )

    # Occupied cannot exceed capacity.
    if {"occupied_beds", "total_beds"}.issubset(out.columns):
        over = out["occupied_beds"] > out["total_beds"]
        n_over = int(over.sum())
        if n_over:
            out.loc[over, "occupied_beds"] = out.loc[over, "total_beds"]
            report.add_fix(
                f"Capped occupied_beds to total_beds for {n_over} impossible rows."
            )

    if {"occupied_icu_beds", "total_icu_beds"}.issubset(out.columns):
        over_icu = out["occupied_icu_beds"] > out["total_icu_beds"]
        n_over_icu = int(over_icu.sum())
        if n_over_icu:
            out.loc[over_icu, "occupied_icu_beds"] = out.loc[
                over_icu, "total_icu_beds"
            ]
            report.add_fix(
                f"Capped occupied_icu_beds to total_icu_beds for {n_over_icu} rows."
            )

    # Recompute available beds and occupancy % for consistency.
    if {"total_beds", "occupied_beds"}.issubset(out.columns):
        inconsistent = (
            out["occupied_beds"] + out.get("available_beds", 0) != out["total_beds"]
        )
        # Always recompute available + percentages from occupied/capacity.
        out["available_beds"] = out["total_beds"] - out["occupied_beds"]
        out["bed_occupancy_percentage"] = (
            100.0 * out["occupied_beds"] / out["total_beds"]
        ).round(2)
        n_inconsistent = int(inconsistent.sum()) if "available_beds" in df.columns else 0
        report.add_fix(
            "Recomputed available_beds and bed_occupancy_percentage from capacities"
            + (f" ({n_inconsistent} previously inconsistent)." if n_inconsistent else ".")
        )

    if {"total_icu_beds", "occupied_icu_beds"}.issubset(out.columns):
        out["icu_occupancy_percentage"] = (
            100.0 * out["occupied_icu_beds"] / out["total_icu_beds"]
        ).round(2)
        report.add_fix("Recomputed icu_occupancy_percentage from ICU capacities.")

    # Percentage bounds.
    for pct_col in (
        "bed_occupancy_percentage",
        "icu_occupancy_percentage",
        "equipment_availability_percentage",
    ):
        if pct_col in out.columns:
            n_oob = int(((out[pct_col] < 0) | (out[pct_col] > 100)).sum())
            if n_oob:
                out[pct_col] = out[pct_col].clip(0, 100)
                report.add_fix(f"Clipped {n_oob} out-of-range values in '{pct_col}'.")

    # Integer-ish count columns.
    int_cols = [
        "opd_patients",
        "emergency_patients",
        "admissions",
        "discharges",
        "total_beds",
        "occupied_beds",
        "available_beds",
        "total_icu_beds",
        "occupied_icu_beds",
        "doctors_available",
        "nurses_available",
        "medicine_stock_units",
        "medicine_consumption_units",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ]
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].round().astype(int)

    # Sort by date.
    out = out.sort_values("date").reset_index(drop=True)
    report.add_note("Rows sorted by date ascending.")
    report.add_note(
        f"Date range: {out['date'].min().date()} to {out['date'].max().date()}"
    )

    report.rows_after = len(out)
    _assert_drop_budget(report)
    return out, report


def preprocess_patient_data(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean patient-level data for future risk classification.

    Preserves risk_level labels. Uses imputation over mass deletion.
    """
    report = DataQualityReport("patient_data")
    report.rows_before = len(df)
    out = df.copy()

    required = {"patient_id", "risk_level"}
    missing_required = required - set(out.columns)
    if missing_required:
        raise ValueError(f"Patient data missing columns: {missing_required}")

    # Duplicates
    full_dups = int(out.duplicated().sum())
    if full_dups:
        out = out.drop_duplicates()
        report.add_fix(f"Removed {full_dups} fully duplicate rows.")

    id_dups = int(out.duplicated(subset=["patient_id"]).sum())
    if id_dups:
        before = len(out)
        out = out.drop_duplicates(subset=["patient_id"], keep="first")
        report.add_fix(
            f"Removed {before - len(out)} duplicate patient_id rows (kept first)."
        )

    # Categoricals
    if "gender" in out.columns:
        out["gender"] = out["gender"].astype(str).str.strip().str.title()
        out.loc[~out["gender"].isin(GENDERS), "gender"] = "Other"
        report.add_note("Normalised gender to Male/Female/Other.")

    for col in ("chronic_disease", "emergency_case"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.title()
            # Map common variants.
            out[col] = out[col].replace(
                {"Y": "Yes", "N": "No", "True": "Yes", "False": "No"}
            )
            invalid = ~out[col].isin(YES_NO)
            n_invalid = int(invalid.sum())
            if n_invalid:
                out.loc[invalid, col] = "No"
                report.add_fix(
                    f"Set {n_invalid} invalid '{col}' values to 'No' (conservative)."
                )

    # Preserve and normalise target.
    out["risk_level"] = out["risk_level"].astype(str).str.strip().str.title()
    invalid_risk = ~out["risk_level"].isin(RISK_LEVELS)
    n_invalid_risk = int(invalid_risk.sum())
    if n_invalid_risk:
        # Do not invent labels silently for many rows.
        if n_invalid_risk / len(out) > MAX_DROP_FRACTION:
            raise ValueError(
                f"Invalid risk_level in {n_invalid_risk} rows - investigate labels."
            )
        out = out.loc[~invalid_risk].copy()
        report.add_fix(f"Dropped {n_invalid_risk} rows with invalid risk_level.")
    report.add_note("Target column 'risk_level' preserved (Low/Medium/High).")

    # Numeric cleaning + imputation
    for col in PATIENT_NUMERIC:
        if col not in out.columns:
            report.add_warning(f"Expected column missing: {col}")
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        missing = int(out[col].isna().sum())
        if missing:
            median_val = out[col].median()
            out[col] = out[col].fillna(median_val)
            report.add_fix(
                f"Imputed {missing} missing values in '{col}' with median={median_val}."
            )

    # Plausible clinical-style bounds (synthetic educational ranges).
    bounds = {
        "age": (0, 120),
        "heart_rate": (30, 220),
        "systolic_bp": (60, 250),
        "diastolic_bp": (30, 150),
        "temperature": (30.0, 43.0),
        "oxygen_saturation": (50, 100),
        "previous_admissions": (0, 50),
    }
    for col, (low, high) in bounds.items():
        if col not in out.columns:
            continue
        n_oob = int(((out[col] < low) | (out[col] > high)).sum())
        if n_oob:
            out[col] = out[col].clip(low, high)
            report.add_fix(
                f"Clipped {n_oob} out-of-range values in '{col}' to [{low}, {high}]."
            )

    # Types
    for col in ("age", "heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation", "previous_admissions"):
        if col in out.columns:
            out[col] = out[col].round().astype(int)
    if "temperature" in out.columns:
        out["temperature"] = out["temperature"].astype(float).round(1)

    out = out.reset_index(drop=True)
    report.rows_after = len(out)
    _assert_drop_budget(report)

    # Class balance note
    counts = out["risk_level"].value_counts().to_dict()
    report.add_note(f"risk_level distribution: {counts}")
    return out, report


def validate_hospital_processed(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error strings (empty if OK)."""
    errors: list[str] = []
    if not df["date"].is_monotonic_increasing:
        errors.append("Dates are not sorted ascending.")
    if (df["occupied_beds"] > df["total_beds"]).any():
        errors.append("occupied_beds exceed total_beds.")
    if (df["occupied_icu_beds"] > df["total_icu_beds"]).any():
        errors.append("occupied_icu_beds exceed total_icu_beds.")
    if (df["available_beds"] < 0).any():
        errors.append("available_beds is negative.")
    if (
        df["occupied_beds"] + df["available_beds"] != df["total_beds"]
    ).any():
        errors.append("occupied + available beds != total_beds.")
    for col in ("opd_patients", "emergency_patients", "admissions", "discharges"):
        if (df[col] < 0).any():
            errors.append(f"Negative values in {col}.")
    if ((df["bed_occupancy_percentage"] < 0) | (df["bed_occupancy_percentage"] > 100)).any():
        errors.append("Impossible bed_occupancy_percentage.")
    if ((df["icu_occupancy_percentage"] < 0) | (df["icu_occupancy_percentage"] > 100)).any():
        errors.append("Impossible icu_occupancy_percentage.")
    return errors


def validate_patient_processed(df: pd.DataFrame) -> list[str]:
    """Return validation errors for patient data."""
    errors: list[str] = []
    if "risk_level" not in df.columns:
        errors.append("Target column risk_level missing.")
    elif not set(df["risk_level"].unique()).issubset(RISK_LEVELS):
        errors.append("Unexpected risk_level values present.")
    for col in ("age", "heart_rate", "previous_admissions"):
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative values in {col}.")
    if df["patient_id"].duplicated().any():
        errors.append("Duplicate patient_id values remain.")
    return errors


def save_processed(
    hospital_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save cleaned datasets (before heavy feature engineering)."""
    output_dir = output_dir or _default_paths()["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    hospital_path = output_dir / "hospital_operations_clean.csv"
    patient_path = output_dir / "patient_data_clean.csv"
    hospital_df.to_csv(hospital_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    return hospital_path, patient_path


def run_preprocessing(
    hospital_path: Path | None = None,
    patient_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full Phase 2 preprocessing entry: load → clean → validate → save.
    Feature engineering is applied separately via src.features.
    """
    hospital_raw = load_hospital_data(hospital_path)
    patient_raw = load_patient_data(patient_path)

    hospital_clean, hospital_report = preprocess_hospital_data(hospital_raw)
    patient_clean, patient_report = preprocess_patient_data(patient_raw)

    hospital_report.print_report()
    patient_report.print_report()

    hosp_errors = validate_hospital_processed(hospital_clean)
    pat_errors = validate_patient_processed(patient_clean)
    if hosp_errors or pat_errors:
        raise ValueError(
            "Validation failed after preprocessing:\n"
            + "\n".join(f"  hospital: {e}" for e in hosp_errors)
            + "\n".join(f"  patient: {e}" for e in pat_errors)
        )

    h_path, p_path = save_processed(hospital_clean, patient_clean, output_dir)
    print(f"Saved cleaned hospital data -> {h_path}")
    print(f"Saved cleaned patient data  -> {p_path}")

    return {
        "hospital_clean": hospital_clean,
        "patient_clean": patient_clean,
        "hospital_report": hospital_report.to_dict(),
        "patient_report": patient_report.to_dict(),
        "hospital_path": h_path,
        "patient_path": p_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MediTwin AI Phase 2 preprocessing.")
    parser.add_argument("--hospital-csv", type=Path, default=None)
    parser.add_argument("--patient-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        run_preprocessing(args.hospital_csv, args.patient_csv, args.output_dir)
    except Exception as exc:
        raise SystemExit(f"Preprocessing failed: {exc}") from exc


if __name__ == "__main__":
    main()
