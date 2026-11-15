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

Fair comparison (Phase 5 fix)
    The "current" and "simulated" waiting times are BOTH produced by the same
    waiting-time model, with the same feature builder. The observed waiting
    time from the dataset is reported separately
    (``WhatIfResult.observed_waiting_time``) and is never compared with a
    model output. A scenario with no changes therefore gives identical
    current and simulated predictions.

Assumptions and limitations
    * Bed demand scales with a weighted OPD + emergency volume ratio
      (a simplification of the generator's admissions/discharge flow).
    * ICU demand changes by 0.25 per extra emergency case and 0.02 per extra
      occupied bed (the Phase 1 generator's ICU formula), applied as a delta
      to the baseline ICU occupancy.
    * Medicine consumption and admissions use the Phase 1 generator's
      coefficients. All of these are rules, not fitted models.
    * Waiting time depends only on the 10 features of the saved model. It is
      trained on one year of synthetic data, so its response to a single
      knob (for example nurses, whose counts correlate with OPD volume in
      that data) is a statistical association, not a causal staffing effect.
    * The model rejects inputs outside its validated ranges; such scenarios
      raise ScenarioError instead of returning a guess.

This is an educational software scenario tool. It is not a live hospital
control system.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.models.prediction_service import PredictionServiceError
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

# ICU demand from the Phase 1 generator: 0.17*emergency + 0.045*occupied_beds.
_ICU_PER_EMERGENCY = 0.17
_ICU_PER_OCCUPIED_BED = 0.045


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
    # Change in the medicine stock on hand at the start of the scenario day
    # (e.g. -60 models a late or short delivery). Relative to the baseline.
    medicine_stock_percent: float = 0.0

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
        if self.medicine_stock_percent:
            lines.append(
                f"Medicine stock on hand {self.medicine_stock_percent:+.1f}% versus baseline."
            )
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
    # Model prediction for the unchanged baseline (None if no model was used).
    baseline_predicted_waiting_time: float | None = None
    # Observed dataset value. Informational only; never compared to the model.
    observed_waiting_time: float = 0.0
    # Capacity status of the baseline using the same waiting-time source as the
    # simulated state, so the two statuses are comparable.
    baseline_capacity_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_predicted_waiting_time": self.baseline_predicted_waiting_time,
            "observed_waiting_time": self.observed_waiting_time,
            "baseline_capacity_status": self.baseline_capacity_status,
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


def _require_number(name: str, value: Any) -> float:
    """Return value as a finite float, or raise a clear ScenarioError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{name} must be a number (got {value!r}).")
    if not math.isfinite(float(value)):
        raise ScenarioError(f"{name} must be finite (got {value!r}).")
    return float(value)


def _require_count(name: str, value: Any, *, minimum: int) -> int:
    """Return value as an integer count >= minimum, or raise ScenarioError."""
    number = _require_number(name, value)
    if not number.is_integer():
        raise ScenarioError(f"{name} must be a whole number (got {value!r}).")
    if number < minimum:
        qualifier = "greater than 0" if minimum == 1 else "non-negative"
        raise ScenarioError(f"{name} must be {qualifier} (got {value!r}).")
    return int(number)


def _validate_baseline(baseline: HospitalState) -> None:
    """Reject impossible baseline states before they can be simulated."""
    for name in (
        "current_opd",
        "emergency_cases",
        "admissions",
        "discharges",
        "occupied_beds",
        "occupied_icu_beds",
        "doctors_available",
        "nurses_available",
        "medicine_stock",
    ):
        _require_count(f"Baseline {name}", getattr(baseline, name), minimum=0)
    total_beds = _require_count("Baseline total_beds", baseline.total_beds, minimum=1)
    total_icu = _require_count(
        "Baseline total_icu_beds", baseline.total_icu_beds, minimum=1
    )
    if baseline.occupied_beds > total_beds:
        raise ScenarioError("Baseline occupied beds exceed total beds.")
    if baseline.occupied_icu_beds > total_icu:
        raise ScenarioError("Baseline occupied ICU beds exceed ICU capacity.")
    _require_number("Baseline average_waiting_time", baseline.average_waiting_time)
    if baseline.average_waiting_time < 0:
        raise ScenarioError("Baseline average waiting time cannot be negative.")


def _validate_changes(changes: ScenarioChanges, baseline: HospitalState) -> None:
    _validate_baseline(baseline)
    if changes.total_beds is not None:
        _require_count("Total bed capacity", changes.total_beds, minimum=1)
    if changes.icu_capacity is not None:
        _require_count("ICU capacity", changes.icu_capacity, minimum=1)
    if changes.doctors_available is not None:
        _require_count("Doctors available", changes.doctors_available, minimum=0)
    if changes.nurses_available is not None:
        _require_count("Nurses available", changes.nurses_available, minimum=0)
    for label, pct, base in (
        ("Patient inflow percent", changes.patient_inflow_percent, baseline.current_opd),
        (
            "Emergency cases percent",
            changes.emergency_cases_percent,
            baseline.emergency_cases,
        ),
    ):
        _require_number(label, pct)
        _scale_percent(base, pct)
    _require_number("Medicine stock percent", changes.medicine_stock_percent)
    _scale_percent(baseline.medicine_stock, changes.medicine_stock_percent)


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

        # ICU demand: baseline plus the generator's marginal effects of extra
        # emergencies and extra occupied beds. Capacity is a separate knob.
        icu_delta = _ICU_PER_EMERGENCY * (
            sim_emergency - baseline.emergency_cases
        ) + _ICU_PER_OCCUPIED_BED * (bed_requirement - baseline.occupied_beds)
        icu_requirement = max(0, int(round(baseline.occupied_icu_beds + icu_delta)))
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
        start_stock = _scale_percent(baseline.medicine_stock, changes.medicine_stock_percent)
        projected_stock = max(0, int(round(start_stock - extra_medicine)))

        base_pressure = staff_pressure(
            base_load, baseline.doctors_available, baseline.nurses_available
        )
        sim_pressure = staff_pressure(sim_load, doctors, nurses)

        # Baseline and scenario are scored by the same model with the same
        # feature builder. Bed occupancy for the baseline is derived from its
        # counts exactly as it is for the scenario.
        base_bed_occupancy = compute_occupancy_percentage(
            baseline.occupied_beds, baseline.total_beds
        )
        base_wait, waiting, wait_source, ml_notes = self._waiting_times(
            baseline=baseline,
            base_bed_occupancy=base_bed_occupancy,
            opd=sim_opd,
            emergency=sim_emergency,
            doctors=doctors,
            nurses=nurses,
            bed_occupancy=bed_occupancy,
            admissions=sim_admissions,
            changes=changes,
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
                base_wait,
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
                "medicine_stock_units",
                baseline.medicine_stock,
                projected_stock,
                SOURCE_RULE,
            ),
            self._metric(
                "medicine_days_of_cover",
                baseline.medicine_stock / base_medicine if base_medicine > 0 else 0.0,
                projected_stock / sim_medicine if sim_medicine > 0 else 0.0,
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
            baseline_predicted_waiting_time=(
                base_wait if wait_source == SOURCE_ML else None
            ),
            observed_waiting_time=float(baseline.average_waiting_time),
            baseline_capacity_status=compute_capacity_status(
                baseline.bed_occupancy_percentage,
                baseline.icu_occupancy_percentage,
                base_wait,
            ),
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

    @staticmethod
    def _wait_features(
        baseline: HospitalState,
        *,
        opd: int,
        emergency: int,
        doctors: int,
        nurses: int,
        bed_occupancy: float,
        admissions: int,
    ) -> dict[str, Any]:
        """Single feature builder shared by the baseline and the scenario."""
        return {
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

    @staticmethod
    def _score(service: Any, features: dict[str, Any]) -> float:
        predicted = service.predict_waiting_time(features)
        return max(0.0, float(predicted.iloc[0]["predicted_waiting_time_minutes"]))

    def _waiting_times(
        self,
        baseline: HospitalState,
        base_bed_occupancy: float,
        opd: int,
        emergency: int,
        doctors: int,
        nurses: int,
        bed_occupancy: float,
        admissions: int,
        changes: ScenarioChanges,
    ) -> tuple[float, float, str, list[str]]:
        """
        Return (baseline_wait, simulated_wait, source, notes).

        Both values come from the same model. If no model is available the
        observed baseline value is returned for BOTH sides (zero difference,
        clearly labelled unavailable) so no fake score is ever shown.
        """
        observed = float(baseline.average_waiting_time)
        service = self._resolve_service()
        if service is None or not hasattr(service, "predict_waiting_time"):
            return (
                observed,
                observed,
                "unavailable (no waiting-time model loaded; observed value shown for both)",
                [
                    "Waiting time was not re-estimated. No model was loaded, "
                    "and no rule was labelled as an ML prediction."
                ],
            )

        base_features = self._wait_features(
            baseline,
            opd=baseline.current_opd,
            emergency=baseline.emergency_cases,
            doctors=baseline.doctors_available,
            nurses=baseline.nurses_available,
            bed_occupancy=base_bed_occupancy,
            admissions=baseline.admissions,
        )
        sim_features = self._wait_features(
            baseline,
            opd=opd,
            emergency=emergency,
            doctors=doctors,
            nurses=nurses,
            bed_occupancy=bed_occupancy,
            admissions=admissions,
        )
        try:
            base_wait = self._score(service, base_features)
            sim_wait = self._score(service, sim_features)
        except PredictionServiceError as exc:
            raise ScenarioError(
                "The scenario or baseline is outside the waiting-time model's "
                f"supported input range: {exc}"
            ) from exc
        except Exception as exc:
            return (
                observed,
                observed,
                "unavailable (waiting-time model could not score the inputs)",
                [
                    "Waiting time was not re-estimated because the model "
                    f"failed ({exc}). Observed value shown for both sides."
                ],
            )

        notes = [
            "Both current and simulated waiting times are model predictions "
            f"(baseline predicted {base_wait:.2f} min; observed value in the "
            f"dataset was {observed:.2f} min and is not used in the comparison)."
        ]
        if changes.nurses_available is not None:
            notes.append(
                "Nurse count is correlated with OPD volume in the synthetic "
                "training data, so the model may show little or no waiting-time "
                "benefit from more nurses. This is a statistical association, "
                "not a staffing-effect claim."
            )
        return base_wait, sim_wait, SOURCE_ML, notes


def simulate_what_if(
    baseline: HospitalState,
    changes: ScenarioChanges | None = None,
    prediction_service: Any | None = None,
    **change_kwargs: Any,
) -> WhatIfResult:
    """Convenience wrapper around WhatIfSimulator.simulate."""
    simulator = WhatIfSimulator(prediction_service=prediction_service)
    return simulator.simulate(baseline, changes=changes, **change_kwargs)
