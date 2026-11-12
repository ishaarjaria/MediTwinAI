"""
What-if simulation for MediTwin AI (Phase 5).

Compares a baseline hospital state with a user-defined scenario.

Two kinds of estimates are kept separate:

* ML-derived prediction
    Waiting time comes from the trained waiting-time model when a
    prediction service (or a test double with the same method) is available.
    If no model can be loaded, the baseline waiting time is kept and the
    result says the model was unavailable. A hand-written formula is never
    labelled as AI.

* rule-derived simulation
    Patient load, bed requirement, ICU pressure, medicine consumption,
    and staff pressure use the formulas in this file. There is no trained
    model for those outputs.

This is an educational software scenario tool. It is not a live hospital
control system.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.twin.hospital_twin import (
    HospitalState,
    compute_available_beds,
    compute_capacity_status,
    compute_occupancy_percentage,
)

# Labels used in every comparison so a viva can tell the two sources apart.
SOURCE_ML = "ML-derived prediction"
SOURCE_RULE = "rule-derived simulation"

# Admission mix copied from the Phase 1 synthetic generator:
# admissions rise with emergency volume and a smaller share of OPD.
# Used only as a documented rule, not as a fitted model.
_ADMISSION_PER_OPD = 0.04
_ADMISSION_PER_EMERGENCY = 0.35

# Medicine draw copied from the Phase 1 generator.
# consumption ~= 0.8*OPD + 2.5*emergency + 1.2*admissions.
_MED_PER_OPD = 0.8
_MED_PER_EMERGENCY = 2.5
_MED_PER_ADMISSION = 1.2


class ScenarioError(ValueError):
    """Raised when a what-if scenario is impossible or inconsistent."""


@dataclass
class ScenarioChanges:
    """
    Knobs for one what-if run.

    Percent fields are relative to the baseline (+20 means +20%).
    Capacity and staffing fields, when set, replace the baseline absolute value.
    """

    patient_inflow_percent: float = 0.0
    total_beds: int | None = None
    icu_capacity: int | None = None
    doctors_available: int | None = None
    nurses_available: int | None = None
    emergency_cases_percent: float = 0.0

    def describe(self) -> list[str]:
        lines: list[str] = []
        if self.patient_inflow_percent:
            lines.append(
                f"Patient inflow {self.patient_inflow_percent:+.1f}% versus baseline OPD."
            )
        if self.emergency_cases_percent:
            lines.append(
                f"Emergency cases {self.emergency_cases_percent:+.1f}% versus baseline."
            )
        if self.total_beds is not None:
            lines.append(f"Total bed capacity set to {self.total_beds}.")
        if self.icu_capacity is not None:
            lines.append(f"ICU capacity set to {self.icu_capacity}.")
        if self.doctors_available is not None:
            lines.append(f"Doctors available set to {self.doctors_available}.")
        if self.nurses_available is not None:
            lines.append(f"Nurses available set to {self.nurses_available}.")
        if not lines:
            lines.append("No scenario changes; simulated volumes match the baseline.")
        return lines


@dataclass
class MetricComparison:
    """One current-versus-simulated metric, with its source labelled."""

    name: str
    current: float
    simulated: float
    source: str
    explanation: str

    @property
    def absolute_change(self) -> float:
        return round(self.simulated - self.current, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current": self.current,
            "simulated": self.simulated,
            "absolute_change": self.absolute_change,
            "source": self.source,
            "explanation": self.explanation,
        }


@dataclass
class WhatIfResult:
    """Full comparison of the current scenario and the simulated scenario."""

    baseline: HospitalState
    simulated_state: HospitalState
    changes: ScenarioChanges
    comparisons: list[MetricComparison]
    explanations: list[str]
    bed_shortfall: int = 0
    icu_shortfall: int = 0
    ml_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "simulated_state": self.simulated_state.to_dict(),
            "changes": self.changes.describe(),
            "comparisons": [item.to_dict() for item in self.comparisons],
            "explanations": self.explanations,
            "bed_shortfall": self.bed_shortfall,
            "icu_shortfall": self.icu_shortfall,
            "ml_notes": self.ml_notes,
        }

    def summary_text(self) -> str:
        lines = ["Current scenario vs simulated scenario", ""]
        lines.extend(f"- {line}" for line in self.changes.describe())
        lines.append("")
        for item in self.comparisons:
            lines.append(item.explanation)
        if self.bed_shortfall:
            lines.append(
                f"Bed shortfall: demand exceeds capacity by {self.bed_shortfall} beds."
            )
        if self.icu_shortfall:
            lines.append(
                f"ICU shortfall: demand exceeds capacity by {self.icu_shortfall} beds."
            )
        lines.extend(self.ml_notes)
        return "\n".join(lines)


def _clamp_count(value: float, low: int, high: int) -> int:
    """Round and clamp a count into [low, high]."""
    if high < low:
        raise ScenarioError(f"Invalid bounds [{low}, {high}].")
    rounded = int(round(value))
    return max(low, min(high, rounded))


def _scale_percent(base: float, percent: float) -> float:
    """Apply a percentage change. A change of -100% reaches zero, not below."""
    if percent < -100:
        raise ScenarioError(
            f"Percentage change cannot be below -100 (got {percent})."
        )
    return base * (1.0 + percent / 100.0)


def medicine_consumption(opd: float, emergency: float, admissions: float) -> float:
    """
    Rule-derived daily medicine consumption (units).

    Coefficients match the Phase 1 synthetic generator. This is not a
    trained model.
    """
    value = (
        _MED_PER_OPD * opd
        + _MED_PER_EMERGENCY * emergency
        + _MED_PER_ADMISSION * admissions
    )
    return round(max(0.0, value), 1)


def staff_pressure(patient_load: float, doctors: int, nurses: int) -> float:
    """
    Rule-derived staff pressure: patients per clinical staff member.

    Higher means more pressure. Not a trained model.
    """
    staff = int(doctors) + int(nurses)
    if staff < 0:
        raise ScenarioError("Staff counts cannot be negative.")
    if staff == 0:
        return float("inf") if patient_load > 0 else 0.0
    return round(patient_load / staff, 4)


def admission_driver(opd: float, emergency: float) -> float:
    """Weighted inflow used only to scale admissions and bed demand."""
    return _ADMISSION_PER_OPD * opd + _ADMISSION_PER_EMERGENCY * emergency


def _explain_change(name: str, current: float, simulated: float, source: str) -> str:
    delta = simulated - current
    if abs(delta) < 1e-9:
        movement = "stays the same"
    elif delta > 0:
        movement = "increases"
    else:
        movement = "decreases"
    return (
        f"{name} {movement} from {current:.2f} to {simulated:.2f} "
        f"(change {delta:+.2f}). Source: {source}."
    )


def _validate_changes(changes: ScenarioChanges, baseline: HospitalState) -> None:
    if changes.total_beds is not None and changes.total_beds <= 0:
        raise ScenarioError("Total bed capacity must be greater than 0.")
    if changes.icu_capacity is not None and changes.icu_capacity <= 0:
        raise ScenarioError("ICU capacity must be greater than 0.")
    if changes.doctors_available is not None and changes.doctors_available < 0:
        raise ScenarioError("Doctors available cannot be negative.")
    if changes.nurses_available is not None and changes.nurses_available < 0:
        raise ScenarioError("Nurses available cannot be negative.")
    _scale_percent(baseline.current_opd, changes.patient_inflow_percent)
    _scale_percent(baseline.emergency_cases, changes.emergency_cases_percent)


class WhatIfSimulator:
    """
    Apply scenario knobs to a baseline HospitalState.

    ``prediction_service`` is optional. When it exposes ``predict_waiting_time``,
    waiting time is reported as an ML-derived prediction. Pass the shared
    Phase 4 service in the application. Tests may pass a stub with the same
    method. If no service can be loaded, waiting time stays at the baseline
    and is not replaced by a fake model score.
    """

    def __init__(self, prediction_service: Any | None = None) -> None:
        self.prediction_service = prediction_service

    def _resolve_service(self) -> Any | None:
        if self.prediction_service is not None:
            return self.prediction_service
        try:
            from src.models.prediction_service import get_prediction_service

            return get_prediction_service()
        except Exception:
            return None

    def simulate(
        self,
        baseline: HospitalState,
        changes: ScenarioChanges | None = None,
        **change_kwargs: Any,
    ) -> WhatIfResult:
        """
        Run one what-if comparison.

        Pass either a ScenarioChanges object or keyword arguments such as
        ``patient_inflow_percent=20``.
        """
        if changes is None:
            changes = ScenarioChanges(**change_kwargs)
        elif change_kwargs:
            raise ScenarioError("Pass ScenarioChanges or keyword changes, not both.")

        _validate_changes(changes, baseline)

        sim_opd = max(
            0, int(round(_scale_percent(baseline.current_opd, changes.patient_inflow_percent)))
        )
        sim_emergency = max(
            0,
            int(round(_scale_percent(baseline.emergency_cases, changes.emergency_cases_percent))),
        )

        base_driver = admission_driver(baseline.current_opd, baseline.emergency_cases)
        sim_driver = admission_driver(sim_opd, sim_emergency)
        volume_ratio = 1.0 if base_driver <= 0 else sim_driver / base_driver
        sim_admissions = max(0, int(round(baseline.admissions * volume_ratio)))

        # Bed demand follows the same volume ratio, then capacity caps occupancy.
        bed_requirement = max(0, int(round(baseline.occupied_beds * volume_ratio)))
        total_beds = (
            changes.total_beds if changes.total_beds is not None else baseline.total_beds
        )
        occupied_beds = _clamp_count(bed_requirement, 0, total_beds)
        bed_shortfall = max(0, bed_requirement - total_beds)
        available_beds = compute_available_beds(total_beds, occupied_beds)
        bed_occupancy = compute_occupancy_percentage(occupied_beds, total_beds)

        # ICU demand follows emergency volume. Capacity is a separate knob.
        if baseline.emergency_cases <= 0:
            icu_ratio = 1.0 if sim_emergency == 0 else float(sim_emergency)
        else:
            icu_ratio = sim_emergency / baseline.emergency_cases
        icu_requirement = max(0, int(round(baseline.occupied_icu_beds * icu_ratio)))
        icu_capacity = (
            changes.icu_capacity
            if changes.icu_capacity is not None
            else baseline.total_icu_beds
        )
        occupied_icu = _clamp_count(icu_requirement, 0, icu_capacity)
        icu_shortfall = max(0, icu_requirement - icu_capacity)
        icu_occupancy = compute_occupancy_percentage(occupied_icu, icu_capacity)

        doctors = (
            changes.doctors_available
            if changes.doctors_available is not None
            else baseline.doctors_available
        )
        nurses = (
            changes.nurses_available
            if changes.nurses_available is not None
            else baseline.nurses_available
        )

        base_load = baseline.current_opd + baseline.emergency_cases + baseline.admissions
        sim_load = sim_opd + sim_emergency + sim_admissions

        base_medicine = medicine_consumption(
            baseline.current_opd, baseline.emergency_cases, baseline.admissions
        )
        sim_medicine = medicine_consumption(sim_opd, sim_emergency, sim_admissions)
        extra_medicine = max(0.0, sim_medicine - base_medicine)
        projected_stock = max(0, int(round(baseline.medicine_stock - extra_medicine)))

        base_pressure = staff_pressure(
            base_load, baseline.doctors_available, baseline.nurses_available
        )
        sim_pressure = staff_pressure(sim_load, doctors, nurses)

        waiting, wait_source, ml_notes = self._waiting_time(
            baseline=baseline,
            opd=sim_opd,
            emergency=sim_emergency,
            doctors=doctors,
            nurses=nurses,
            bed_occupancy=bed_occupancy,
            admissions=sim_admissions,
        )

        if bed_occupancy < 0 or icu_occupancy < 0:
            raise ScenarioError("Occupancy became negative.")
        if available_beds > total_beds or available_beds < 0:
            raise ScenarioError("Available beds are outside 0..total beds.")
        if not 0.0 <= icu_occupancy <= 100.0:
            raise ScenarioError("ICU occupancy is outside 0-100.")

        simulated = HospitalState(
            as_of_date=baseline.as_of_date,
            total_beds=total_beds,
            occupied_beds=occupied_beds,
            available_beds=available_beds,
            bed_occupancy_percentage=bed_occupancy,
            total_icu_beds=icu_capacity,
            occupied_icu_beds=occupied_icu,
            icu_occupancy_percentage=icu_occupancy,
            current_opd=sim_opd,
            emergency_cases=sim_emergency,
            doctors_available=doctors,
            nurses_available=nurses,
            average_waiting_time=waiting,
            medicine_stock=projected_stock,
            admissions=sim_admissions,
            discharges=baseline.discharges,
            is_weekend=baseline.is_weekend,
            is_holiday=baseline.is_holiday,
            day_of_week=baseline.day_of_week,
            month=baseline.month,
            notes=[
                "Simulated scenario. Volumes, beds, ICU, medicine, and staffing "
                "are rule-derived.",
                f"Waiting time source: {wait_source}.",
            ]
            + changes.describe(),
        )
        simulated.capacity_status = compute_capacity_status(
            simulated.bed_occupancy_percentage,
            simulated.icu_occupancy_percentage,
            simulated.average_waiting_time,
        )

        comparisons = [
            self._metric("patient_load", base_load, sim_load, SOURCE_RULE),
            self._metric(
                "bed_requirement", baseline.occupied_beds, bed_requirement, SOURCE_RULE
            ),
            self._metric(
                "bed_occupancy_percentage",
                baseline.bed_occupancy_percentage,
                bed_occupancy,
                SOURCE_RULE,
            ),
            self._metric(
                "icu_pressure_percentage",
                baseline.icu_occupancy_percentage,
                icu_occupancy,
                SOURCE_RULE,
            ),
            self._metric(
                "expected_waiting_time_minutes",
                baseline.average_waiting_time,
                waiting,
                wait_source,
            ),
            self._metric(
                "medicine_consumption_units",
                base_medicine,
                sim_medicine,
                SOURCE_RULE,
            ),
            self._metric(
                "staff_pressure_patients_per_staff",
                base_pressure if base_pressure != float("inf") else -1,
                sim_pressure if sim_pressure != float("inf") else -1,
                SOURCE_RULE,
            ),
        ]

        explanations = changes.describe() + [item.explanation for item in comparisons]
        if bed_shortfall:
            explanations.append(
                f"Bed requirement is {bed_requirement}, but only {total_beds} beds "
                f"exist, so occupancy is capped at 100% and {bed_shortfall} beds "
                "of demand are unmet. Source: rule-derived simulation."
            )
        if icu_shortfall:
            explanations.append(
                f"ICU requirement is {icu_requirement}, but capacity is {icu_capacity}, "
                f"so ICU occupancy is capped at 100% and {icu_shortfall} ICU beds "
                "of demand are unmet. Source: rule-derived simulation."
            )

        return WhatIfResult(
            baseline=deepcopy(baseline),
            simulated_state=simulated,
            changes=changes,
            comparisons=comparisons,
            explanations=explanations,
            bed_shortfall=bed_shortfall,
            icu_shortfall=icu_shortfall,
            ml_notes=ml_notes,
        )

    @staticmethod
    def _metric(
        name: str, current: float, simulated: float, source: str
    ) -> MetricComparison:
        current_r = round(float(current), 4)
        simulated_r = round(float(simulated), 4)
        return MetricComparison(
            name=name,
            current=current_r,
            simulated=simulated_r,
            source=source,
            explanation=_explain_change(name, current_r, simulated_r, source),
        )

    def _waiting_time(
        self,
        baseline: HospitalState,
        opd: int,
        emergency: int,
        doctors: int,
        nurses: int,
        bed_occupancy: float,
        admissions: int,
    ) -> tuple[float, str, list[str]]:
        """Ask the waiting-time model. Do not invent a score if it is missing."""
        service = self._resolve_service()
        features = {
            "opd_patients": opd,
            "emergency_patients": emergency,
            "doctors_available": doctors,
            "nurses_available": nurses,
            "is_weekend": baseline.is_weekend,
            "is_holiday": baseline.is_holiday,
            "day_of_week": baseline.day_of_week,
            "month": baseline.month,
            "bed_occupancy_percentage": bed_occupancy,
            "admissions": admissions,
        }
        if service is None or not hasattr(service, "predict_waiting_time"):
            return (
                float(baseline.average_waiting_time),
                "unavailable (no waiting-time model loaded; baseline value kept)",
                [
                    "Waiting time was not re-estimated. No model was loaded, "
                    "and no rule was labelled as an ML prediction."
                ],
            )
        try:
            predicted = service.predict_waiting_time(features)
            minutes = float(predicted.iloc[0]["predicted_waiting_time_minutes"])
        except Exception as exc:
            return (
                float(baseline.average_waiting_time),
                "unavailable (waiting-time model rejected the scenario inputs)",
                [
                    "Waiting time was not re-estimated because the model "
                    f"could not score this scenario ({exc}). Baseline value kept."
                ],
            )
        if minutes < 0:
            minutes = 0.0
        return minutes, SOURCE_ML, []


def simulate_what_if(
    baseline: HospitalState,
    changes: ScenarioChanges | None = None,
    prediction_service: Any | None = None,
    **change_kwargs: Any,
) -> WhatIfResult:
    """Convenience wrapper around WhatIfSimulator.simulate."""
    simulator = WhatIfSimulator(prediction_service=prediction_service)
    return simulator.simulate(baseline, changes=changes, **change_kwargs)
