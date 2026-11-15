"""
MediTwin AI — Hospital Digital Twin (Phase 4).

A **data-driven software representation** of hospital operational state based
on the synthetic/processed dataset and ML outputs.

This is NOT a physical IoT-connected hospital twin and does not claim live
sensor / EMR connectivity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.data.preprocess import get_project_root
from src.models.prediction_service import (
    PredictionService,
    get_prediction_service,
)

CapacityStatus = Literal["Normal", "Moderate", "High", "Critical"]

# Occupancy thresholds for capacity_status (educational policy, not clinical).
_BED_ICU_THRESHOLDS = (
    (95.0, "Critical"),
    (85.0, "High"),
    (70.0, "Moderate"),
)
# Waiting-time escalation (minutes) layered on top of occupancy.
_WAIT_HIGH = 60.0
_WAIT_CRITICAL = 120.0


@dataclass
class HospitalState:
    """Structured snapshot of hospital operational state."""

    as_of_date: str
    total_beds: int
    occupied_beds: int
    available_beds: int
    bed_occupancy_percentage: float
    total_icu_beds: int
    occupied_icu_beds: int
    icu_occupancy_percentage: float
    current_opd: int
    emergency_cases: int
    doctors_available: int
    nurses_available: int
    average_waiting_time: float
    medicine_stock: int
    # Helpful optional context for ML / updates
    admissions: int = 0
    discharges: int = 0
    is_weekend: int = 0
    is_holiday: int = 0
    day_of_week: int = 0
    month: int = 1
    capacity_status: CapacityStatus = "Normal"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_occupancy_percentage(occupied: int | float, total: int | float) -> float:
    """Return occupancy %; raises if capacity is invalid."""
    total_f = float(total)
    occupied_f = float(occupied)
    if total_f <= 0:
        raise ValueError("total capacity must be > 0")
    if occupied_f < 0:
        raise ValueError("occupied count cannot be negative")
    if occupied_f > total_f:
        raise ValueError("occupied count cannot exceed total capacity")
    return round(100.0 * occupied_f / total_f, 2)


def compute_available_beds(total_beds: int, occupied_beds: int) -> int:
    """Derive available beds from totals (never negative)."""
    if total_beds < 0 or occupied_beds < 0:
        raise ValueError("bed counts cannot be negative")
    if occupied_beds > total_beds:
        raise ValueError("occupied_beds cannot exceed total_beds")
    return int(total_beds - occupied_beds)


def _validate_state_values(state: HospitalState) -> None:
    """Reject invalid operational inputs before they can enter the twin."""
    count_fields = (
        "total_beds",
        "occupied_beds",
        "total_icu_beds",
        "occupied_icu_beds",
        "current_opd",
        "emergency_cases",
        "doctors_available",
        "nurses_available",
        "medicine_stock",
        "admissions",
        "discharges",
    )
    for name in count_fields:
        value = getattr(state, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a non-negative integer.")
        if not math.isfinite(float(value)) or float(value) < 0 or not float(value).is_integer():
            raise ValueError(f"{name} must be a non-negative integer.")
        setattr(state, name, int(value))

    if state.total_beds == 0 or state.total_icu_beds == 0:
        raise ValueError("Bed and ICU capacities must be greater than zero.")

    wait = state.average_waiting_time
    if isinstance(wait, bool) or not isinstance(wait, (int, float)):
        raise ValueError("average_waiting_time must be a finite non-negative number.")
    if not math.isfinite(float(wait)) or float(wait) < 0:
        raise ValueError("average_waiting_time must be a finite non-negative number.")
    state.average_waiting_time = float(wait)

    for name in ("is_weekend", "is_holiday"):
        value = getattr(state, name)
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1.")
        setattr(state, name, int(value))

    if not isinstance(state.day_of_week, int) or not 0 <= state.day_of_week <= 6:
        raise ValueError("day_of_week must be an integer between 0 and 6.")
    if not isinstance(state.month, int) or not 1 <= state.month <= 12:
        raise ValueError("month must be an integer between 1 and 12.")
    if not isinstance(state.notes, list) or not all(
        isinstance(note, str) for note in state.notes
    ):
        raise ValueError("notes must be a list of strings.")

    try:
        parsed_date = pd.Timestamp(state.as_of_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of_date must be a valid date.") from exc
    if pd.isna(parsed_date):
        raise ValueError("as_of_date must be a valid date.")
    state.as_of_date = parsed_date.strftime("%Y-%m-%d")


def waiting_time_features(
    *,
    opd: int,
    emergency: int,
    doctors: int,
    nurses: int,
    is_weekend: int,
    is_holiday: int,
    day_of_week: int,
    month: int,
    bed_occupancy: float,
    admissions: int,
) -> dict[str, Any]:
    """The single place that turns operational values into waiting-time model inputs."""
    return {
        "opd_patients": opd,
        "emergency_patients": emergency,
        "doctors_available": doctors,
        "nurses_available": nurses,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "day_of_week": day_of_week,
        "month": month,
        "bed_occupancy_percentage": bed_occupancy,
        "admissions": admissions,
    }


# Provenance of every twin value. "observed" = a recorded value in the synthetic
# daily dataset; "ml_prediction" = output of a trained model; "rule_derived" =
# arithmetic or a transparent heuristic (including values carried forward).
CURRENT_FIELD_SOURCES: dict[str, str] = {
    "occupied_beds": "observed",
    "available_beds": "observed",
    "bed_occupancy_percentage": "observed",
    "occupied_icu_beds": "observed",
    "icu_occupancy_percentage": "observed",
    "current_opd": "observed",
    "emergency_cases": "observed",
    "doctors_available": "observed",
    "nurses_available": "observed",
    "average_waiting_time": "observed",
    "medicine_stock": "observed",
}
PREDICTED_FIELD_SOURCES: dict[str, str] = {
    "occupied_beds": "rule_derived",  # heuristic: 2% of forecast OPD change
    "available_beds": "rule_derived",
    "bed_occupancy_percentage": "rule_derived",
    "occupied_icu_beds": "rule_derived",  # heuristic: 15% of emergency change
    "icu_occupancy_percentage": "rule_derived",
    "current_opd": "ml_prediction",  # trained OPD forecaster
    "emergency_cases": "rule_derived",  # held at today's level, weekend uplift
    "doctors_available": "rule_derived",  # carried forward from today
    "nurses_available": "rule_derived",  # carried forward from today
    "average_waiting_time": "ml_prediction",  # trained waiting-time model
    "medicine_stock": "rule_derived",  # drawdown only if OPD rises
}


def compute_capacity_status(
    bed_occupancy_percentage: float,
    icu_occupancy_percentage: float,
    average_waiting_time: float | None = None,
) -> CapacityStatus:
    """
    Map occupancy (and optional waiting time) to Normal/Moderate/High/Critical.

    Rule (educational):
    - Use the worse of bed vs ICU occupancy against fixed thresholds.
    - Waiting time can escalate status by at most one tier
      (>=60 min -> at least High; >=120 min -> Critical).
    """
    for key, val in (
        ("bed_occupancy_percentage", bed_occupancy_percentage),
        ("icu_occupancy_percentage", icu_occupancy_percentage),
    ):
        if not math.isfinite(float(val)) or val < 0 or val > 100:
            raise ValueError(f"{key} must be between 0 and 100 (got {val})")

    worst = max(float(bed_occupancy_percentage), float(icu_occupancy_percentage))
    status: CapacityStatus = "Normal"
    for threshold, label in _BED_ICU_THRESHOLDS:
        if worst >= threshold:
            status = label  # type: ignore[assignment]
            break

    if average_waiting_time is not None:
        wait = float(average_waiting_time)
        if not math.isfinite(wait) or wait < 0:
            raise ValueError("average_waiting_time must be finite and non-negative")
        order = ["Normal", "Moderate", "High", "Critical"]
        idx = order.index(status)
        if wait >= _WAIT_CRITICAL:
            idx = max(idx, order.index("Critical"))
        elif wait >= _WAIT_HIGH:
            idx = max(idx, order.index("High"))
        status = order[idx]  # type: ignore[assignment]

    return status


def _recompute_derived_fields(state: HospitalState) -> HospitalState:
    """Keep available beds, occupancy %, and capacity_status consistent."""
    _validate_state_values(state)
    state.available_beds = compute_available_beds(
        state.total_beds, state.occupied_beds
    )
    state.bed_occupancy_percentage = compute_occupancy_percentage(
        state.occupied_beds, state.total_beds
    )
    # ICU available is implicit; enforce occupancy consistency.
    if state.occupied_icu_beds > state.total_icu_beds:
        raise ValueError("occupied_icu_beds cannot exceed total_icu_beds")
    state.icu_occupancy_percentage = compute_occupancy_percentage(
        state.occupied_icu_beds, state.total_icu_beds
    )
    state.capacity_status = compute_capacity_status(
        state.bed_occupancy_percentage,
        state.icu_occupancy_percentage,
        state.average_waiting_time,
    )
    return state


def hospital_state_from_row(row: pd.Series | dict[str, Any]) -> HospitalState:
    """Build a HospitalState from a processed hospital-operations row."""
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    as_of = data.get("date") or data.get("as_of_date")
    if hasattr(as_of, "strftime"):
        as_of_date = as_of.strftime("%Y-%m-%d")
    else:
        as_of_date = str(as_of)[:10]

    total_beds = int(data["total_beds"])
    occupied_beds = int(data["occupied_beds"])
    total_icu = int(data["total_icu_beds"])
    occupied_icu = int(data["occupied_icu_beds"])

    state = HospitalState(
        as_of_date=as_of_date,
        total_beds=total_beds,
        occupied_beds=occupied_beds,
        available_beds=int(
            data.get(
                "available_beds",
                compute_available_beds(total_beds, occupied_beds),
            )
        ),
        bed_occupancy_percentage=float(
            data.get(
                "bed_occupancy_percentage",
                compute_occupancy_percentage(occupied_beds, total_beds),
            )
        ),
        total_icu_beds=total_icu,
        occupied_icu_beds=occupied_icu,
        icu_occupancy_percentage=float(
            data.get(
                "icu_occupancy_percentage",
                compute_occupancy_percentage(occupied_icu, total_icu),
            )
        ),
        current_opd=int(data.get("opd_patients", data.get("current_opd", 0))),
        emergency_cases=int(
            data.get("emergency_patients", data.get("emergency_cases", 0))
        ),
        doctors_available=int(data.get("doctors_available", 0)),
        nurses_available=int(data.get("nurses_available", 0)),
        average_waiting_time=float(
            data.get(
                "average_waiting_time_minutes",
                data.get("average_waiting_time", 0.0),
            )
        ),
        medicine_stock=int(
            data.get("medicine_stock_units", data.get("medicine_stock", 0))
        ),
        admissions=int(data.get("admissions", 0)),
        discharges=int(data.get("discharges", 0)),
        is_weekend=int(data.get("is_weekend", 0)),
        is_holiday=int(data.get("is_holiday", 0)),
        day_of_week=int(data.get("day_of_week", 0)),
        month=int(data.get("month", 1)),
    )
    return _recompute_derived_fields(state)


class HospitalDigitalTwin:
    """
    Software digital twin for MediTwin AI.

    Holds the current operational state, can update it, and can produce a
    short-horizon *predicted* state using the PredictionService (OPD forecast
    + waiting-time model + anomaly check).
    """

    def __init__(
        self,
        initial_state: HospitalState | None = None,
        prediction_service: PredictionService | None = None,
        operations_csv: Path | None = None,
    ) -> None:
        self.service = prediction_service or get_prediction_service()
        self._operations_csv = operations_csv or (
            get_project_root()
            / "data"
            / "processed"
            / "hospital_operations_features.csv"
        )
        if initial_state is None:
            initial_state = self._load_latest_state_from_csv()
        self._state = _recompute_derived_fields(deepcopy(initial_state))
        self._history: list[HospitalState] = [deepcopy(self._state)]

    def _load_latest_state_from_csv(self) -> HospitalState:
        if not self._operations_csv.is_file():
            raise FileNotFoundError(
                f"Hospital operations file not found: {self._operations_csv}. "
                "Run Phase 2 feature engineering first."
            )
        df = pd.read_csv(self._operations_csv, parse_dates=["date"])
        df = df.sort_values("date")
        return hospital_state_from_row(df.iloc[-1])

    # ------------------------------------------------------------------
    # Core API required by Phase 4
    # ------------------------------------------------------------------
    def get_current_state(self) -> HospitalState:
        """Return a copy of the current twin state."""
        return deepcopy(self._state)

    def get_capacity_status(self) -> CapacityStatus:
        """Return Normal / Moderate / High / Critical for the current state."""
        return compute_capacity_status(
            self._state.bed_occupancy_percentage,
            self._state.icu_occupancy_percentage,
            self._state.average_waiting_time,
        )

    def update_state(self, **updates: Any) -> HospitalState:
        """
        Update one or more state fields, then recompute derived values.

        Example:
            twin.update_state(occupied_beds=200, current_opd=310)
        """
        allowed = {f.name for f in fields(HospitalState)} - {
            "capacity_status",
            "notes",
            "available_beds",
            "bed_occupancy_percentage",
            "icu_occupancy_percentage",
        }
        allowed.add("notes")

        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(
                "Unknown or derived state fields cannot be updated: "
                f"{sorted(unknown)}"
            )

        candidate = deepcopy(self._state)
        for key, value in updates.items():
            setattr(candidate, key, value)

        # Validate and recompute before committing, so failed updates are atomic.
        candidate = _recompute_derived_fields(candidate)
        self._state = candidate
        self._history.append(deepcopy(self._state))
        return self.get_current_state()

    def model_waiting_time(self, state: HospitalState | None = None) -> float:
        """
        Waiting time (minutes) the trained model assigns to ``state``'s own inputs.

        Use this - not the observed ``average_waiting_time`` - when comparing
        today with a predicted day, so both sides come from the same model and
        the same feature builder. Defaults to the twin's current state.
        """
        state = state or self._state
        features = waiting_time_features(
            opd=state.current_opd,
            emergency=state.emergency_cases,
            doctors=state.doctors_available,
            nurses=state.nurses_available,
            is_weekend=state.is_weekend,
            is_holiday=state.is_holiday,
            day_of_week=state.day_of_week,
            month=state.month,
            bed_occupancy=state.bed_occupancy_percentage,
            admissions=state.admissions,
        )
        out = self.service.predict_waiting_time(features)
        return float(out.iloc[0]["predicted_waiting_time_minutes"])

    def get_predicted_state(self, days_ahead: int = 1) -> HospitalState:
        """
        Build a predicted near-future state using ML outputs.

        Uses:
        - ML OPD forecast for horizon ``days_ahead``
        - Waiting-time model with forecasted OPD and current staffing/calendar
        - Isolation Forest anomaly flag on the projected operational vector

        Bed/ICU projection is a transparent heuristic (not IoT): small pressure
        from forecasted OPD vs current OPD, clipped to capacity.
        """
        if days_ahead < 1:
            raise ValueError("days_ahead must be >= 1")

        current = self.get_current_state()
        forecast = self.service.forecast_opd(days=days_ahead)
        # Use the horizon day (last row of the returned forecast window).
        row = forecast.iloc[days_ahead - 1]
        forecast_date = pd.Timestamp(row["date"])
        predicted_opd = int(round(float(row["opd_forecast"])))

        # Calendar flags for the forecast date.
        is_weekend = int(forecast_date.dayofweek >= 5)
        day_of_week = int(forecast_date.dayofweek)
        month = int(forecast_date.month)

        # Heuristic bed pressure: if OPD rises vs today, nudge occupancy up slightly.
        opd_delta = predicted_opd - current.current_opd
        bed_nudge = int(round(opd_delta * 0.02))  # educational sensitivity
        projected_occupied = int(
            min(
                current.total_beds,
                max(0, current.occupied_beds + bed_nudge),
            )
        )
        # Emergency held near current with mild weekend uplift (synthetic policy).
        projected_emergency = int(
            round(current.emergency_cases * (1.1 if is_weekend else 1.0))
        )
        projected_icu = int(
            min(
                current.total_icu_beds,
                max(
                    0,
                    current.occupied_icu_beds
                    + int(round((projected_emergency - current.emergency_cases) * 0.15)),
                ),
            )
        )

        bed_pct = compute_occupancy_percentage(
            projected_occupied, current.total_beds
        )
        icu_pct = compute_occupancy_percentage(projected_icu, current.total_icu_beds)

        # Same feature builder and model as ``model_waiting_time`` (equivalent inputs).
        wait_input = waiting_time_features(
            opd=predicted_opd,
            emergency=projected_emergency,
            doctors=current.doctors_available,
            nurses=current.nurses_available,
            is_weekend=is_weekend,
            is_holiday=current.is_holiday,  # unknown future holidays -> carry flag
            day_of_week=day_of_week,
            month=month,
            bed_occupancy=bed_pct,
            admissions=current.admissions,
        )
        wait_pred = self.service.predict_waiting_time(wait_input)
        predicted_wait = float(wait_pred.iloc[0]["predicted_waiting_time_minutes"])

        anomaly_input = {
            "opd_patients": predicted_opd,
            "emergency_patients": projected_emergency,
            "bed_occupancy_percentage": bed_pct,
            "icu_occupancy_percentage": icu_pct,
            "average_waiting_time_minutes": predicted_wait,
        }
        anomaly = self.service.detect_hospital_anomaly(anomaly_input)
        anom_flag = int(anomaly.iloc[0]["anomaly_flag"])
        anom_reason = str(anomaly.iloc[0]["anomaly_reason"])

        # Simple medicine stock drawdown vs OPD change.
        med_delta = int(round((predicted_opd - current.current_opd) * 0.8))
        projected_stock = max(0, current.medicine_stock - max(0, med_delta))

        notes = [
            "Predicted state is a data-driven software estimate (not IoT live data).",
            f"OPD forecast ({self.service.opd_model_name}) for {forecast_date.date()}: {predicted_opd}",
            f"Predicted waiting time (ML): {predicted_wait:.1f} min",
        ]
        if anom_flag:
            notes.append(f"Anomaly flag on projected day: {anom_reason or 'flagged'}")

        predicted = HospitalState(
            as_of_date=str(forecast_date.date()),
            total_beds=current.total_beds,
            occupied_beds=projected_occupied,
            available_beds=0,  # recomputed
            bed_occupancy_percentage=0.0,
            total_icu_beds=current.total_icu_beds,
            occupied_icu_beds=projected_icu,
            icu_occupancy_percentage=0.0,
            current_opd=predicted_opd,
            emergency_cases=projected_emergency,
            doctors_available=current.doctors_available,
            nurses_available=current.nurses_available,
            average_waiting_time=predicted_wait,
            medicine_stock=projected_stock,
            admissions=current.admissions,
            discharges=current.discharges,
            is_weekend=is_weekend,
            is_holiday=int(current.is_holiday),
            day_of_week=day_of_week,
            month=month,
            notes=notes,
        )
        return _recompute_derived_fields(predicted)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Compact dict for dashboards / viva demos."""
        state = self.get_current_state()
        return {
            "disclaimer": (
                "Software digital twin from dataset + ML — not a physical "
                "IoT-connected hospital."
            ),
            "current": state.to_dict(),
            "capacity_status": self.get_capacity_status(),
        }


# Alias matching the prompt naming preference.
HospitalTwin = HospitalDigitalTwin


def build_twin_from_latest(
    prediction_service: PredictionService | None = None,
) -> HospitalDigitalTwin:
    """Factory: twin initialised from the latest processed operations row."""
    return HospitalDigitalTwin(prediction_service=prediction_service)


__all__ = [
    "HospitalState",
    "HospitalDigitalTwin",
    "HospitalTwin",
    "CapacityStatus",
    "compute_occupancy_percentage",
    "compute_available_beds",
    "compute_capacity_status",
    "waiting_time_features",
    "CURRENT_FIELD_SOURCES",
    "PREDICTED_FIELD_SOURCES",
    "hospital_state_from_row",
    "build_twin_from_latest",
]
