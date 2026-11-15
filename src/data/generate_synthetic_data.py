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


# ---------------------------------------------------------------------------
# Operational dynamics (all values synthetic; each has an operational meaning)
# ---------------------------------------------------------------------------
# Bed occupancy is a stock-and-flow process:
#   occupied[t] = occupied[t-1] + admissions[t] - discharges[t]
# where discharges are a fraction of the beds currently occupied (the inverse of
# the average length of stay, ~4 days). Weekends discharge fewer patients.
DISCHARGE_RATE_WEEKDAY = 0.27
DISCHARGE_RATE_WEEKEND = 0.19
INITIAL_OCCUPIED_BEDS = 150  # ~steady state for normal demand (no warm-up drift)

# Medicine stock follows a reorder-point policy with delivery lead time.
BASE_MEDICINE_STOCK = 16000  # opening stock (units)
REORDER_POINT = 10500  # place an order when stock falls to/below this
ORDER_QUANTITY = 10000
LEAD_TIME_DAYS = (3, 5)  # inclusive range of normal delivery delay

# Documented, controlled high-demand / resource-pressure periods.
# Multipliers apply on top of the seasonal baseline.
PRESSURE_PERIODS = [
    {
        "name": "Winter respiratory wave",
        "start": "2024-01-15",
        "end": "2024-02-12",
        "opd": 1.10,
        "emergency": 1.30,
        "admission": 1.08,
    },
    {
        "name": "Monsoon dengue outbreak",
        "start": "2024-08-12",
        "end": "2024-09-08",
        "opd": 1.15,
        "emergency": 1.35,
        "admission": 1.10,
    },
    {
        "name": "Festival accident / burn surge",
        "start": "2024-10-30",
        "end": "2024-11-04",
        "opd": 1.00,
        "emergency": 1.40,
        "admission": 1.10,
    },
]
# Staff sickness during the monsoon outbreak (documented resource pressure).
STAFF_SHORTAGE_PERIODS = [
    {"name": "Outbreak staff sickness", "start": "2024-08-19", "end": "2024-08-30", "factor": 0.80},
]
# Supplier disruption: orders placed in this window arrive late by extra days.
SUPPLY_DISRUPTION_PERIODS = [
    {"name": "Supplier delay", "start": "2024-08-25", "end": "2024-09-10", "extra_days": 9},
]
N_INJECTED_SPIKES = 5
SPIKE_TYPES = ("emergency_surge", "opd_rush", "icu_pressure")


def _in_period(date_str: str, start: str, end: str) -> bool:
    return start <= date_str <= end


def _period_factor(date_str: str, key: str) -> float:
    factor = 1.0
    for p in PRESSURE_PERIODS:
        if _in_period(date_str, p["start"], p["end"]):
            factor *= p[key]
    return factor


def _choose_spikes(n_days: int, seed: int) -> dict[int, str]:
    """Pick spike day indices and types using an independent RNG stream."""
    rng = np.random.default_rng(seed + 1000)
    idx = sorted(
        rng.choice(np.arange(20, n_days - 20), size=N_INJECTED_SPIKES, replace=False).tolist()
    )
    return {int(i): str(rng.choice(SPIKE_TYPES)) for i in idx}


def generate_hospital_operations_with_truth(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate daily synthetic hospital operations plus the injected-spike ground truth.

    Returns ``(hospital_df, truth_df)``; ``truth_df`` has columns ``date``,
    ``spike_type`` and is used only to *evaluate* anomaly detection.

    Logical relationships encoded:
    - Waiting time rises with OPD, emergency volume and bed occupancy, and falls
      with more doctors.
    - OPD and emergency demand follow season, weekday, holidays and documented
      pressure periods (``PRESSURE_PERIODS``); demand has day-to-day persistence.
    - Admissions rise with emergency volume; discharges are a fraction of the
      beds occupied (length of stay), so occupancy evolves continuously and can
      rise in pressure periods and recover afterwards.
    - ICU occupancy follows emergency load and overall bed occupancy.
    - Medicine stock is consumed daily and replenished by a reorder-point policy
      with lead time, including a documented supplier-disruption window.
    - A few unusual spikes are injected (applied inside the daily flow so the
      beds / stock stay internally consistent).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    if n_days < 365:
        raise ValueError(
            f"Date range must cover at least 365 days; got {n_days} days "
            f"({start_date} to {end_date})."
        )

    spikes = _choose_spikes(n_days, seed)
    records: list[dict] = []
    truth: list[dict] = []
    medicine_stock = float(BASE_MEDICINE_STOCK)
    pending_orders: list[tuple[int, float]] = []  # (arrival day index, quantity)
    prev_occupied = float(INITIAL_OCCUPIED_BEDS)
    demand_memory = 0.0  # AR(1) persistence of demand (outbreaks last several days)

    for i, current_date in enumerate(dates):
        day_of_week = int(current_date.dayofweek)  # Monday=0 ... Sunday=6
        is_weekend = int(day_of_week >= 5)
        date_str = current_date.strftime("%Y-%m-%d")
        is_holiday = int(date_str in HOLIDAY_DATES)
        season = get_season(int(current_date.month))
        spike = spikes.get(i)

        if season == "Winter":
            seasonal_opd, seasonal_emergency = 320, 55
        elif season == "Monsoon":
            seasonal_opd, seasonal_emergency = 300, 60
        elif season == "Spring":
            seasonal_opd, seasonal_emergency = 260, 45
        else:
            seasonal_opd, seasonal_emergency = 270, 48

        weekend_opd_factor = 0.55 if is_weekend else 1.0
        weekend_emergency_factor = 1.15 if is_weekend else 1.0
        holiday_opd_factor = 0.35 if is_holiday else 1.0
        holiday_emergency_factor = 1.25 if is_holiday else 1.0
        weekly_wave = 1.0 + 0.08 * np.sin(2 * np.pi * day_of_week / 7)
        yearly_wave = 1.0 + 0.05 * np.sin(2 * np.pi * i / 365)

        # Persistent demand shock (random, but correlated across days).
        demand_memory = 0.6 * demand_memory + rng.normal(0, 0.04)
        demand_shock = 1.0 + demand_memory

        opd_patients = max(
            40,
            int(
                rng.normal(
                    loc=seasonal_opd
                    * weekend_opd_factor
                    * holiday_opd_factor
                    * weekly_wave
                    * _period_factor(date_str, "opd")
                    * demand_shock,
                    scale=25,
                )
            ),
        )
        emergency_patients = max(
            10,
            int(
                rng.normal(
                    loc=seasonal_emergency
                    * weekend_emergency_factor
                    * holiday_emergency_factor
                    * yearly_wave
                    * _period_factor(date_str, "emergency")
                    * demand_shock,
                    scale=8,
                )
            ),
        )

        admissions = max(
            5,
            int(
                (0.35 * emergency_patients + 0.04 * opd_patients + rng.normal(8, 3))
                * _period_factor(date_str, "admission")
            ),
        )

        # --- injected spikes: applied to the daily inputs so flows stay consistent ---
        if spike == "emergency_surge":
            emergency_patients = int(emergency_patients * 2.2 + 25)
            admissions = int(admissions * 1.8 + 10)
        elif spike == "opd_rush":
            opd_patients = int(opd_patients * 1.9 + 80)
        elif spike == "icu_pressure":
            emergency_patients = int(emergency_patients * 1.5 + 15)

        # --- beds: stock-and-flow with length-of-stay discharges ---
        rate = DISCHARGE_RATE_WEEKEND if is_weekend else DISCHARGE_RATE_WEEKDAY
        discharges = max(3, int(round(prev_occupied * rate + rng.normal(0, 3))))
        flow_occupied = float(
            np.clip(prev_occupied + admissions - discharges, 20, TOTAL_BEDS)
        )
        occupied_beds = int(flow_occupied)
        available_beds = TOTAL_BEDS - occupied_beds
        bed_occupancy_percentage = round(100.0 * occupied_beds / TOTAL_BEDS, 2)

        # --- ICU ---
        icu_pressure = (
            0.17 * emergency_patients + 0.045 * occupied_beds + rng.normal(1, 1.2)
        )
        if spike == "emergency_surge":
            icu_pressure += rng.integers(6, 12)
        if spike == "icu_pressure":
            icu_pressure = TOTAL_ICU_BEDS
        occupied_icu_beds = int(np.clip(icu_pressure, 3, TOTAL_ICU_BEDS))
        icu_occupancy_percentage = round(100.0 * occupied_icu_beds / TOTAL_ICU_BEDS, 2)

        # --- staffing ---
        staff_factor = 0.85 if (is_weekend or is_holiday) else 1.0
        for p in STAFF_SHORTAGE_PERIODS:
            if _in_period(date_str, p["start"], p["end"]):
                staff_factor *= p["factor"]
        doctors_available = int(
            np.clip(rng.normal(BASE_DOCTORS * staff_factor, 3), 25, BASE_DOCTORS + 5)
        )
        nurses_available = int(
            np.clip(rng.normal(BASE_NURSES * staff_factor, 8), 70, BASE_NURSES + 15)
        )

        # --- waiting time ---
        # Waiting time rises with outpatient load, emergency volume (shared triage and
        # staff) and bed occupancy (crowding); it falls with more doctors.
        waiting = (
            0.12 * opd_patients
            + 0.12 * emergency_patients
            + 0.05 * bed_occupancy_percentage
            - 0.35 * doctors_available
            + 2.0
            + rng.normal(0, 4)
        )
        if spike == "emergency_surge":
            waiting *= 1.6
        elif spike == "opd_rush":
            waiting *= 1.9
        average_waiting_time_minutes = round(float(np.clip(waiting, 5, 180)), 1)

        # --- medicine: consumption, deliveries, reorder policy ---
        medicine_consumption_units = int(
            max(
                50,
                0.8 * opd_patients
                + 2.5 * emergency_patients
                + 1.2 * admissions
                + rng.normal(20, 10),
            )
        )
        if spike == "emergency_surge":
            medicine_consumption_units = int(medicine_consumption_units * 1.7)
        elif spike == "opd_rush":
            medicine_consumption_units = int(medicine_consumption_units * 1.5)
        arrived = sum(q for d, q in pending_orders if d <= i)
        pending_orders = [(d, q) for d, q in pending_orders if d > i]
        medicine_stock = max(0.0, medicine_stock + arrived - medicine_consumption_units)
        if medicine_stock <= REORDER_POINT and not pending_orders:
            lead = int(rng.integers(LEAD_TIME_DAYS[0], LEAD_TIME_DAYS[1] + 1))
            for p in SUPPLY_DISRUPTION_PERIODS:
                if _in_period(date_str, p["start"], p["end"]):
                    lead += p["extra_days"]
            pending_orders.append((i + lead, float(ORDER_QUANTITY)))
        medicine_stock_units = int(medicine_stock)

        equipment = (
            98
            - 0.05 * bed_occupancy_percentage
            - 0.08 * icu_occupancy_percentage
            + rng.normal(0, 1.5)
        )
        if spike == "icu_pressure":
            equipment -= 12
        equipment_availability_percentage = round(float(np.clip(equipment, 70, 100)), 2)

        if spike is not None:
            truth.append({"date": date_str, "spike_type": spike})

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
        prev_occupied = flow_occupied

    df = pd.DataFrame(records)
    _validate_hospital_constraints(df)
    return df, pd.DataFrame(truth, columns=["date", "spike_type"])


def generate_hospital_operations(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate daily synthetic hospital operations (see the ``..._with_truth`` variant)."""
    return generate_hospital_operations_with_truth(start_date, end_date, seed)[0]



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


def risk_score_breakdown(row: pd.Series) -> list[tuple[str, int]]:
    """
    Itemised synthetic risk score: a list of ``(factor, points)`` for every rule
    that contributes. ``assign_risk_level`` sums these points.

    These rules are for educational labelling only. They are NOT validated medical
    guidelines and must not be used for real patient care.
    """
    items: list[tuple[str, int]] = []

    def add(text: str, points: int) -> None:
        items.append((text, points))

    # Age contributes to risk in this synthetic scheme.
    if row["age"] >= 65:
        add("Age 65 or older", 2)
    elif row["age"] >= 45:
        add("Age 45 to 64", 1)

    # Vital-sign style heuristics (synthetic thresholds).
    if row["heart_rate"] < 50 or row["heart_rate"] > 110:
        add("Heart rate far outside 50-110", 2)
    elif row["heart_rate"] < 60 or row["heart_rate"] > 100:
        add("Heart rate outside 60-100", 1)

    if row["systolic_bp"] >= 160 or row["systolic_bp"] <= 90:
        add("Systolic blood pressure far outside 90-160", 2)
    elif row["systolic_bp"] >= 140 or row["systolic_bp"] <= 100:
        add("Systolic blood pressure outside 100-140", 1)

    if row["diastolic_bp"] >= 100 or row["diastolic_bp"] <= 50:
        add("Diastolic blood pressure outside 50-100", 1)

    if row["temperature"] >= 39.0 or row["temperature"] <= 35.5:
        add("Temperature far outside 35.5-39.0", 2)
    elif row["temperature"] >= 38.0:
        add("Fever (38.0 or higher)", 1)

    if row["oxygen_saturation"] < 90:
        add("Oxygen saturation below 90", 3)
    elif row["oxygen_saturation"] < 94:
        add("Oxygen saturation below 94", 2)
    elif row["oxygen_saturation"] < 96:
        add("Oxygen saturation below 96", 1)

    if row["chronic_disease"] == "Yes":
        add("Chronic disease", 2)

    if row["emergency_case"] == "Yes":
        add("Emergency case", 2)

    if row["previous_admissions"] >= 3:
        add("Three or more previous admissions", 2)
    elif row["previous_admissions"] >= 1:
        add("Previous admission", 1)

    return items


def assign_risk_level(row: pd.Series) -> str:
    """
    Assign Low / Medium / High risk using transparent synthetic clinical-style rules.

    These rules are for educational labelling only. They are NOT validated medical
    guidelines and must not be used for real patient care.
    """
    score = sum(points for _, points in risk_score_breakdown(row))
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
    truth_df: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Save CSVs under data/synthetic/ and return their paths.

    If ``truth_df`` (injected-spike ground truth) is given it is saved as
    ``injected_anomalies.csv`` for anomaly-detection evaluation only.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    hospital_path = output_dir / "hospital_operations.csv"
    patient_path = output_dir / "patient_data.csv"

    hospital_df.to_csv(hospital_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    if truth_df is not None:
        truth_df.to_csv(output_dir / "injected_anomalies.csv", index=False)
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
        hospital_df, truth_df = generate_hospital_operations_with_truth(
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
        )
        print("Generating synthetic patient data...")
        patient_df = generate_patient_data(
            n_patients=args.n_patients,
            seed=args.seed,
        )
        hospital_path, patient_path = save_datasets(
            hospital_df, patient_df, truth_df=truth_df
        )
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
