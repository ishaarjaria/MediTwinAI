"""
Generate synthetic hospital operational and patient datasets for MediTwin AI.

IMPORTANT DISCLAIMER
--------------------
All data produced by this script is SYNTHETIC and created only for educational
and university project purposes. It is NOT real clinical data and must NEVER
be used for real medical decisions, diagnosis, triage, or hospital operations.

Usage (from the meditwin-ai project root):
    python -m src.data.generate_synthetic_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed for reproducibility (required for viva / grading).
RANDOM_SEED = 42

# Hospital capacity constants (synthetic mid-size hospital).
TOTAL_BEDS = 250
TOTAL_ICU_BEDS = 30
BASE_DOCTORS = 45
BASE_NURSES = 120
BASE_MEDICINE_STOCK = 50000

# Default date range: one full calendar year of daily records.
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Approximate Indian public holidays (synthetic educational calendar for 2024).
HOLIDAY_DATES = {
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Holi (approx.)
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Eid (approx.)
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Gandhi Jayanti
    "2024-10-12",  # Dussehra (approx.)
    "2024-11-01",  # Diwali (approx.)
    "2024-12-25",  # Christmas
}


def get_project_root() -> Path:
    """Return the meditwin-ai project root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def get_season(month: int) -> str:
    """Map calendar month to a simple season label."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Autumn"


def generate_hospital_operations(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate at least one year of daily synthetic hospital operational data.

    Logical relationships encoded:
    - Higher OPD generally increases waiting time.
    - Higher emergency volume increases admissions and ICU usage.
    - Occupied beds never exceed total beds; ICU occupied never exceeds ICU capacity.
    - Medicine consumption broadly rises with patient load.
    - Weekend and seasonal variation are visible.
    - A few realistic unusual spikes are injected for future anomaly detection.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    if n_days < 365:
        raise ValueError(
            f"Date range must cover at least 365 days; got {n_days} days "
            f"({start_date} to {end_date})."
        )

    records: list[dict] = []
    medicine_stock = float(BASE_MEDICINE_STOCK)
    # Carry a mild occupancy memory so bed usage looks continuous day-to-day.
    prev_occupied = 160.0

    for i, current_date in enumerate(dates):
        day_of_week = int(current_date.dayofweek)  # Monday=0 ... Sunday=6
        is_weekend = int(day_of_week >= 5)
        date_str = current_date.strftime("%Y-%m-%d")
        is_holiday = int(date_str in HOLIDAY_DATES)
        season = get_season(int(current_date.month))

        # --- Seasonal base load ---
        # Winter / monsoon often see higher respiratory and infection-related demand.
        if season == "Winter":
            seasonal_opd = 320
            seasonal_emergency = 55
        elif season == "Monsoon":
            seasonal_opd = 300
            seasonal_emergency = 60
        elif season == "Spring":
            seasonal_opd = 260
            seasonal_emergency = 45
        else:  # Autumn
            seasonal_opd = 270
            seasonal_emergency = 48

        # Weekend: lower OPD (clinics quieter), slightly higher emergency.
        weekend_opd_factor = 0.55 if is_weekend else 1.0
        weekend_emergency_factor = 1.15 if is_weekend else 1.0

        # Holidays: OPD drops sharply; emergency rises.
        holiday_opd_factor = 0.35 if is_holiday else 1.0
        holiday_emergency_factor = 1.25 if is_holiday else 1.0

        # Mild weekly / yearly smooth variation (sinusoidal).
        weekly_wave = 1.0 + 0.08 * np.sin(2 * np.pi * day_of_week / 7)
        yearly_wave = 1.0 + 0.05 * np.sin(2 * np.pi * i / 365)

        opd_patients = int(
            rng.normal(
                loc=seasonal_opd * weekend_opd_factor * holiday_opd_factor * weekly_wave,
                scale=25,
            )
        )
        opd_patients = max(40, opd_patients)

        emergency_patients = int(
            rng.normal(
                loc=seasonal_emergency
                * weekend_emergency_factor
                * holiday_emergency_factor
                * yearly_wave,
                scale=8,
            )
        )
        emergency_patients = max(10, emergency_patients)

        # Admissions rise with emergency volume (and a fraction of OPD).
        admissions = int(
            0.35 * emergency_patients
            + 0.04 * opd_patients
            + rng.normal(8, 3)
        )
        admissions = max(5, admissions)

        # Discharges roughly track previous occupancy pressure.
        discharges = int(0.85 * admissions + rng.normal(5, 4) + 0.02 * prev_occupied)
        discharges = max(3, discharges)

        occupied_beds = int(
            np.clip(
                prev_occupied + admissions - discharges + rng.normal(0, 4),
                40,
                TOTAL_BEDS,
            )
        )
        available_beds = TOTAL_BEDS - occupied_beds
        bed_occupancy_percentage = round(100.0 * occupied_beds / TOTAL_BEDS, 2)

        # ICU usage increases with emergency load and overall occupancy.
        icu_pressure = (
            0.25 * emergency_patients
            + 0.02 * occupied_beds
            + rng.normal(2, 1.5)
        )
        occupied_icu_beds = int(np.clip(icu_pressure, 3, TOTAL_ICU_BEDS))
        icu_occupancy_percentage = round(
            100.0 * occupied_icu_beds / TOTAL_ICU_BEDS, 2
        )

        # Staffing: slightly lower on weekends/holidays.
        staff_factor = 0.85 if (is_weekend or is_holiday) else 1.0
        doctors_available = int(
            np.clip(rng.normal(BASE_DOCTORS * staff_factor, 3), 25, BASE_DOCTORS + 5)
        )
        nurses_available = int(
            np.clip(rng.normal(BASE_NURSES * staff_factor, 8), 70, BASE_NURSES + 15)
        )

        # Waiting time rises with OPD and falls when more doctors are available.
        average_waiting_time_minutes = round(
            float(
                np.clip(
                    8
                    + 0.12 * opd_patients
                    - 0.35 * doctors_available
                    + rng.normal(0, 4),
                    5,
                    180,
                )
            ),
            1,
        )

        # Medicine: consumption tracks patient load; stock depletes and is restocked.
        medicine_consumption_units = int(
            max(
                50,
                0.8 * opd_patients
                + 2.5 * emergency_patients
                + 1.2 * admissions
                + rng.normal(20, 10),
            )
        )
        medicine_stock = max(500.0, medicine_stock - medicine_consumption_units)
        # Weekly restock roughly every Monday.
        if day_of_week == 0:
            medicine_stock = min(
                BASE_MEDICINE_STOCK * 1.2,
                medicine_stock + rng.uniform(8000, 12000),
            )
        medicine_stock_units = int(medicine_stock)

        # Equipment availability dips when occupancy/ICU pressure is high.
        equipment_availability_percentage = round(
            float(
                np.clip(
                    98
                    - 0.05 * bed_occupancy_percentage
                    - 0.08 * icu_occupancy_percentage
                    + rng.normal(0, 1.5),
                    70,
                    100,
                )
            ),
            2,
        )

        records.append(
            {
                "date": date_str,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "season": season,
                "opd_patients": opd_patients,
                "emergency_patients": emergency_patients,
                "admissions": admissions,
                "discharges": discharges,
                "total_beds": TOTAL_BEDS,
                "occupied_beds": occupied_beds,
                "available_beds": available_beds,
                "bed_occupancy_percentage": bed_occupancy_percentage,
                "total_icu_beds": TOTAL_ICU_BEDS,
                "occupied_icu_beds": occupied_icu_beds,
                "icu_occupancy_percentage": icu_occupancy_percentage,
                "doctors_available": doctors_available,
                "nurses_available": nurses_available,
                "average_waiting_time_minutes": average_waiting_time_minutes,
                "medicine_stock_units": medicine_stock_units,
                "medicine_consumption_units": medicine_consumption_units,
                "equipment_availability_percentage": equipment_availability_percentage,
            }
        )
        prev_occupied = float(occupied_beds)

    df = pd.DataFrame(records)

    # Inject a small number of realistic unusual spikes for future anomaly detection.
    df = _inject_anomaly_spikes(df, rng)
    _validate_hospital_constraints(df)
    return df


def _inject_anomaly_spikes(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add a few unusual but realistic spike days (e.g., mass-casualty style surge,
    festival rush, or infection outbreak pressure). Still respects capacity caps.
    """
    df = df.copy()
    n = len(df)
    # Choose 5 distinct spike days away from the very start/end.
    spike_indices = sorted(
        rng.choice(np.arange(20, n - 20), size=5, replace=False).tolist()
    )

    for idx in spike_indices:
        spike_type = rng.choice(["emergency_surge", "opd_rush", "icu_pressure"])

        if spike_type == "emergency_surge":
            df.loc[idx, "emergency_patients"] = int(
                df.loc[idx, "emergency_patients"] * 2.2 + 25
            )
            df.loc[idx, "admissions"] = int(df.loc[idx, "admissions"] * 1.8 + 10)
            new_icu = min(
                TOTAL_ICU_BEDS,
                int(df.loc[idx, "occupied_icu_beds"] + rng.integers(8, 15)),
            )
            df.loc[idx, "occupied_icu_beds"] = new_icu
            df.loc[idx, "icu_occupancy_percentage"] = round(
                100.0 * new_icu / TOTAL_ICU_BEDS, 2
            )
            new_occupied = min(
                TOTAL_BEDS,
                int(df.loc[idx, "occupied_beds"] + rng.integers(20, 40)),
            )
            df.loc[idx, "occupied_beds"] = new_occupied
            df.loc[idx, "available_beds"] = TOTAL_BEDS - new_occupied
            df.loc[idx, "bed_occupancy_percentage"] = round(
                100.0 * new_occupied / TOTAL_BEDS, 2
            )
            df.loc[idx, "medicine_consumption_units"] = int(
                df.loc[idx, "medicine_consumption_units"] * 1.7
            )
            df.loc[idx, "average_waiting_time_minutes"] = round(
                min(180.0, float(df.loc[idx, "average_waiting_time_minutes"]) * 1.6),
                1,
            )

        elif spike_type == "opd_rush":
            df.loc[idx, "opd_patients"] = int(df.loc[idx, "opd_patients"] * 1.9 + 80)
            df.loc[idx, "average_waiting_time_minutes"] = round(
                min(180.0, float(df.loc[idx, "average_waiting_time_minutes"]) * 1.9),
                1,
            )
            df.loc[idx, "medicine_consumption_units"] = int(
                df.loc[idx, "medicine_consumption_units"] * 1.5
            )

        else:  # icu_pressure
            new_icu = TOTAL_ICU_BEDS  # near/full ICU capacity spike
            df.loc[idx, "occupied_icu_beds"] = new_icu
            df.loc[idx, "icu_occupancy_percentage"] = 100.0
            df.loc[idx, "emergency_patients"] = int(
                df.loc[idx, "emergency_patients"] * 1.5 + 15
            )
            df.loc[idx, "equipment_availability_percentage"] = round(
                max(70.0, float(df.loc[idx, "equipment_availability_percentage"]) - 12),
                2,
            )

    return df


def _validate_hospital_constraints(df: pd.DataFrame) -> None:
    """Raise if capacity or occupancy constraints are violated."""
    if (df["occupied_beds"] > df["total_beds"]).any():
        raise ValueError("Constraint violated: occupied_beds exceed total_beds.")
    if (df["occupied_icu_beds"] > df["total_icu_beds"]).any():
        raise ValueError(
            "Constraint violated: occupied_icu_beds exceed total_icu_beds."
        )
    if (df["available_beds"] < 0).any():
        raise ValueError("Constraint violated: available_beds is negative.")
    if (df["occupied_beds"] + df["available_beds"] != df["total_beds"]).any():
        raise ValueError(
            "Constraint violated: occupied_beds + available_beds != total_beds."
        )


def assign_risk_level(row: pd.Series) -> str:
    """
    Assign Low / Medium / High risk using transparent synthetic clinical-style rules.

    These rules are for educational labelling only. They are NOT validated medical
    guidelines and must not be used for real patient care.
    """
    score = 0

    # Age contributes to risk in this synthetic scheme.
    if row["age"] >= 65:
        score += 2
    elif row["age"] >= 45:
        score += 1

    # Vital-sign style heuristics (synthetic thresholds).
    if row["heart_rate"] < 50 or row["heart_rate"] > 110:
        score += 2
    elif row["heart_rate"] < 60 or row["heart_rate"] > 100:
        score += 1

    if row["systolic_bp"] >= 160 or row["systolic_bp"] <= 90:
        score += 2
    elif row["systolic_bp"] >= 140 or row["systolic_bp"] <= 100:
        score += 1

    if row["diastolic_bp"] >= 100 or row["diastolic_bp"] <= 50:
        score += 1

    if row["temperature"] >= 39.0 or row["temperature"] <= 35.5:
        score += 2
    elif row["temperature"] >= 38.0:
        score += 1

    if row["oxygen_saturation"] < 90:
        score += 3
    elif row["oxygen_saturation"] < 94:
        score += 2
    elif row["oxygen_saturation"] < 96:
        score += 1

    if row["chronic_disease"] == "Yes":
        score += 2

    if row["emergency_case"] == "Yes":
        score += 2

    if row["previous_admissions"] >= 3:
        score += 2
    elif row["previous_admissions"] >= 1:
        score += 1

    if score >= 7:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def generate_patient_data(
    n_patients: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic patient-level dataset for future risk classification.

    Labels (Low / Medium / High) come from ``assign_risk_level`` using explicit
    synthetic rules — educational only, not a clinical decision system.
    """
    if n_patients < 1:
        raise ValueError("n_patients must be at least 1.")

    rng = np.random.default_rng(seed)

    ages = rng.integers(1, 90, size=n_patients)
    genders = rng.choice(["Male", "Female", "Other"], size=n_patients, p=[0.48, 0.48, 0.04])

    # Vitals centred on typical adult ranges with noise.
    heart_rate = np.clip(rng.normal(78, 14, size=n_patients), 40, 160).astype(int)
    systolic_bp = np.clip(rng.normal(125, 18, size=n_patients), 80, 200).astype(int)
    diastolic_bp = np.clip(rng.normal(80, 12, size=n_patients), 40, 120).astype(int)
    temperature = np.round(
        np.clip(rng.normal(36.8, 0.6, size=n_patients), 34.5, 41.0), 1
    )
    oxygen_saturation = np.clip(rng.normal(97, 2.5, size=n_patients), 80, 100).astype(
        int
    )

    chronic_disease = rng.choice(["Yes", "No"], size=n_patients, p=[0.30, 0.70])
    emergency_case = rng.choice(["Yes", "No"], size=n_patients, p=[0.25, 0.75])
    previous_admissions = rng.integers(0, 6, size=n_patients)

    df = pd.DataFrame(
        {
            "patient_id": [f"P{str(i + 1).zfill(5)}" for i in range(n_patients)],
            "age": ages,
            "gender": genders,
            "heart_rate": heart_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "temperature": temperature,
            "oxygen_saturation": oxygen_saturation,
            "chronic_disease": chronic_disease,
            "emergency_case": emergency_case,
            "previous_admissions": previous_admissions,
        }
    )
    df["risk_level"] = df.apply(assign_risk_level, axis=1)
    return df


def save_datasets(
    hospital_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Save CSVs under data/synthetic/ and return their paths."""
    if output_dir is None:
        output_dir = get_project_root() / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    hospital_path = output_dir / "hospital_operations.csv"
    patient_path = output_dir / "patient_data.csv"

    hospital_df.to_csv(hospital_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    return hospital_path, patient_path


def main() -> None:
    """CLI entry point for Phase 1 synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic MediTwin AI hospital and patient datasets."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End date (YYYY-MM-DD). Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=1000,
        help="Number of synthetic patients. Default: 1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}",
    )
    args = parser.parse_args()

    try:
        print("Generating synthetic hospital operations data...")
        hospital_df = generate_hospital_operations(
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
        )
        print("Generating synthetic patient data...")
        patient_df = generate_patient_data(
            n_patients=args.n_patients,
            seed=args.seed,
        )
        hospital_path, patient_path = save_datasets(hospital_df, patient_df)
    except Exception as exc:  # beginner-friendly surface error
        raise SystemExit(f"Data generation failed: {exc}") from exc

    print("\n=== MediTwin AI — Phase 1 data generation complete ===")
    print(f"Hospital operations: {hospital_path}")
    print(f"  rows={len(hospital_df)}, cols={len(hospital_df.columns)}")
    print(f"Patient data:        {patient_path}")
    print(f"  rows={len(patient_df)}, cols={len(patient_df.columns)}")
    print(f"Risk level counts:\n{patient_df['risk_level'].value_counts().to_string()}")
    print(
        "\nDISCLAIMER: Synthetic educational data only. "
        "Not for real clinical use."
    )


if __name__ == "__main__":
    main()
