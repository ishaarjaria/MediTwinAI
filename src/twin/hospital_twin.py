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
        if val < 0 or val > 100:
            raise ValueError(f"{key} must be between 0 and 100 (got {val})")

    worst = max(float(bed_occupancy_percentage), float(icu_occupancy_percentage))
    status: CapacityStatus = "Normal"
    for threshold, label in _BED_ICU_THRESHOLDS:
        if worst >= threshold:
            status = label  # type: ignore[assignment]
            break

    if average_waiting_time is not None:
        wait = float(average_waiting_time)
        if wait < 0:
            raise ValueError("average_waiting_time cannot be negative")
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
        # Allow direct occupancy updates via occupied_* ; derived fields recomputed.
        allowed |= {
            "available_beds",
            "bed_occupancy_percentage",
            "icu_occupancy_percentage",
            "notes",
        }

        unknown = set(updates) - {f.name for f in fields(HospitalState)}
        if unknown:
            raise ValueError(f"Unknown state fields: {sorted(unknown)}")

        for key, value in updates.items():
            setattr(self._state, key, value)

        # If caller set occupancy % directly without occupied counts, trust counts first.
        self._state = _recompute_derived_fields(self._state)
        self._history.append(deepcopy(self._state))
        return self.get_current_state()

    def get_predicted_state(self, days_ahead: int = 1) -> HospitalState:
        """
        Build a predicted near-future state using ML outputs.

        Uses:
        - Prophet OPD forecast for horizon ``days_ahead``
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

        wait_input = {
            "opd_patients": predicted_opd,
            "emergency_patients": projected_emergency,
            "doctors_available": current.doctors_available,
            "nurses_available": current.nurses_available,
            "is_weekend": is_weekend,
            "is_holiday": current.is_holiday,  # unknown future holidays -> carry flag
            "day_of_week": day_of_week,
            "month": month,
            "bed_occupancy_percentage": bed_pct,
            "admissions": current.admissions,
        }
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
            f"OPD forecast (Prophet) for {forecast_date.date()}: {predicted_opd}",
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
    "hospital_state_from_row",
    "build_twin_from_latest",
]
