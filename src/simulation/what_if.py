"""
MediTwin AI — What-If Simulation (Phase 5).

Accepts a baseline ``HospitalState`` and applies scenario changes such as
patient inflow, bed/ICU capacity, staffing, and emergency volume.

Outputs compare **Current Scenario vs Simulated Scenario** with:
- ``rule_derived`` — transparent deterministic formulas
- ``ml_derived`` — outputs from trained models (e.g. waiting-time regressor)

This is an educational simulation over the digital twin state — not a live
hospital control system and not fabricated AI.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from src.twin.hospital_twin import (
    HospitalState,
    compute_available_beds,
    compute_capacity_status,
    compute_occupancy_percentage,
)

# ---------------------------------------------------------------------------
# Transparent rule coefficients (educational; aligned with Phase 1 generator)
# ---------------------------------------------------------------------------
# Extra admissions pressure from OPD / emergency deltas.
_ADMISSION_FROM_OPD = 0.04
_ADMISSION_FROM_EMERGENCY = 0.35
# Fraction of new admissions that need a ward bed the same day (net of discharges).
_BED_FROM_ADMISSION = 0.55
# Fraction of emergency increase that pressures ICU beds.
_ICU_FROM_EMERGENCY = 0.25
# Medicine consumption ≈ Phase 1 synthetic rule.
_MED_OPD = 0.8
_MED_EMERGENCY = 2.5
_MED_ADMISSION = 1.2
# Rule-based waiting time fallback (same spirit as Phase 1 generator).
_WAIT_INTERCEPT = 8.0
_WAIT_OPD_COEF = 0.12
_WAIT_DOC_COEF = 0.35


@dataclass
class ScenarioChanges:
    """
    Relative / absolute knobs for a what-if scenario.

    Percentage fields are relative changes, e.g. ``patient_inflow_pct=20``
    means +20% OPD (and linked admission pressure). Use negative values to cut.
    Absolute fields replace capacities / staffing when not None.
    """

    patient_inflow_pct: float = 0.0
    emergency_cases_pct: float = 0.0
    total_bed_capacity: int | None = None
    icu_capacity: int | None = None
    doctors_available: int | None = None
    nurses_available: int | None = None

    def describe(self) -> list[str]:
        bits: list[str] = []
        if self.patient_inflow_pct:
            sign = "+" if self.patient_inflow_pct > 0 else ""
            bits.append(f"Patient inflow {sign}{self.patient_inflow_pct:.1f}%")
        if self.emergency_cases_pct:
            sign = "+" if self.emergency_cases_pct > 0 else ""
            bits.append(f"Emergency cases {sign}{self.emergency_cases_pct:.1f}%")
        if self.total_bed_capacity is not None:
            bits.append(f"Total bed capacity -> {self.total_bed_capacity}")
        if self.icu_capacity is not None:
            bits.append(f"ICU capacity -> {self.icu_capacity}")
        if self.doctors_available is not None:
            bits.append(f"Doctors available -> {self.doctors_available}")
        if self.nurses_available is not None:
            bits.append(f"Nurses available -> {self.nurses_available}")
        return bits or ["No changes (identity scenario)"]


@dataclass
class ScenarioMetrics:
    """Comparable KPIs for baseline or simulated scenario."""

    opd_patients: int
    emergency_cases: int
    admissions: int
    patient_load: int
    total_beds: int
    occupied_beds: int
    available_beds: int
    bed_requirement: int
    bed_occupancy_percentage: float
    total_icu_beds: int
    occupied_icu_beds: int
    icu_occupancy_percentage: float
    icu_pressure: float  # alias emphasis on ICU occupancy %
    doctors_available: int
    nurses_available: int
    staff_to_patient_ratio: float
    staff_pressure: float  # patients per staff (higher = more pressure)
    medicine_consumption_units: float
    expected_waiting_time_minutes: float
    capacity_status: str
    waiting_time_source: str  # "ml_derived" | "rule_derived"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WhatIfResult:
    """Full current-vs-simulated comparison with provenance."""

    baseline: ScenarioMetrics
    simulated: ScenarioMetrics
    changes: ScenarioChanges
    deltas: dict[str, float]
    rule_derived: dict[str, Any]
    ml_derived: dict[str, Any]
    explanations: list[str] = field(default_factory=list)
    constraints_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "simulated": self.simulated.to_dict(),
            "changes": asdict(self.changes),
            "deltas": self.deltas,
            "rule_derived": self.rule_derived,
            "ml_derived": self.ml_derived,
            "explanations": self.explanations,
            "constraints_applied": self.constraints_applied,
            "comparison_label": "Current Scenario vs Simulated Scenario",
        }


def _clip_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def _medicine_consumption(opd: int, emergency: int, admissions: int) -> float:
    """Rule-derived medicine use from patient mix."""
    return round(
        _MED_OPD * opd + _MED_EMERGENCY * emergency + _MED_ADMISSION * admissions,
        1,
    )


def _rule_waiting_time(opd: int, doctors: int) -> float:
    """Transparent fallback if no waiting-time model is available."""
    raw = _WAIT_INTERCEPT + _WAIT_OPD_COEF * opd - _WAIT_DOC_COEF * doctors
    return float(max(5.0, min(180.0, round(raw, 1))))


def _patient_load(opd: int, emergency: int, admissions: int) -> int:
    return int(opd + emergency + admissions)


def _staff_metrics(patient_load: int, doctors: int, nurses: int) -> tuple[float, float]:
    staff = max(1, int(doctors) + int(nurses))
    ratio = round(staff / patient_load, 4) if patient_load > 0 else 0.0
    pressure = round(patient_load / staff, 4)
    return ratio, pressure


def metrics_from_state(
    state: HospitalState,
    *,
    waiting_time: float | None = None,
    waiting_time_source: str = "rule_derived",
    medicine_consumption: float | None = None,
    bed_requirement: int | None = None,
) -> ScenarioMetrics:
    """Build comparable metrics from a HospitalState (+ optional overrides)."""
    opd = int(state.current_opd)
    emergency = int(state.emergency_cases)
    admissions = int(state.admissions)
    load = _patient_load(opd, emergency, admissions)
    doctors = int(state.doctors_available)
    nurses = int(state.nurses_available)
    ratio, pressure = _staff_metrics(load, doctors, nurses)
    wait = (
        float(waiting_time)
        if waiting_time is not None
        else float(state.average_waiting_time)
    )
    med = (
        float(medicine_consumption)
        if medicine_consumption is not None
        else _medicine_consumption(opd, emergency, admissions)
    )
    bed_req = int(bed_requirement) if bed_requirement is not None else int(state.occupied_beds)

    return ScenarioMetrics(
        opd_patients=opd,
        emergency_cases=emergency,
        admissions=admissions,
        patient_load=load,
        total_beds=int(state.total_beds),
        occupied_beds=int(state.occupied_beds),
        available_beds=int(state.available_beds),
        bed_requirement=bed_req,
        bed_occupancy_percentage=float(state.bed_occupancy_percentage),
        total_icu_beds=int(state.total_icu_beds),
        occupied_icu_beds=int(state.occupied_icu_beds),
        icu_occupancy_percentage=float(state.icu_occupancy_percentage),
        icu_pressure=float(state.icu_occupancy_percentage),
        doctors_available=doctors,
        nurses_available=nurses,
        staff_to_patient_ratio=ratio,
        staff_pressure=pressure,
        medicine_consumption_units=med,
        expected_waiting_time_minutes=wait,
        capacity_status=str(state.capacity_status),
        waiting_time_source=waiting_time_source,
    )


class WhatIfSimulator:
    """
    Run what-if scenarios against a baseline hospital state.

    Parameters
    ----------
    prediction_service:
        Optional Phase 4 ``PredictionService`` (or compatible object with
        ``predict_waiting_time``). When provided, expected waiting time is
        **ML-derived**. Otherwise a transparent rule fallback is used and
        labelled ``rule_derived``.
    """

    def __init__(self, prediction_service: Any | None = None) -> None:
        self.prediction_service = prediction_service

    def simulate(
        self,
        baseline: HospitalState,
        changes: ScenarioChanges | None = None,
    ) -> WhatIfResult:
        changes = changes or ScenarioChanges()
        constraints: list[str] = []
        rule_derived: dict[str, Any] = {}
        ml_derived: dict[str, Any] = {}

        base = deepcopy(baseline)

        # ----- Apply percentage / absolute changes (rule layer) -----
        inflow_factor = 1.0 + float(changes.patient_inflow_pct) / 100.0
        emergency_factor = 1.0 + float(changes.emergency_cases_pct) / 100.0
        if inflow_factor < 0 or emergency_factor < 0:
            raise ValueError("Percentage changes cannot reduce volumes below -100%.")

        sim_opd = max(0, int(round(base.current_opd * inflow_factor)))
        sim_emergency = max(0, int(round(base.emergency_cases * emergency_factor)))

        # Admissions respond to OPD + emergency deltas (deterministic).
        delta_opd = sim_opd - base.current_opd
        delta_em = sim_emergency - base.emergency_cases
        admission_delta = (
            _ADMISSION_FROM_OPD * delta_opd + _ADMISSION_FROM_EMERGENCY * delta_em
        )
        sim_admissions = max(0, int(round(base.admissions + admission_delta)))

        rule_derived["patient_inflow_factor"] = round(inflow_factor, 4)
        rule_derived["emergency_factor"] = round(emergency_factor, 4)
        rule_derived["simulated_opd"] = sim_opd
        rule_derived["simulated_emergency"] = sim_emergency
        rule_derived["simulated_admissions"] = sim_admissions
        rule_derived["admission_delta_rule"] = (
            f"admissions += {_ADMISSION_FROM_OPD}*delta_opd + "
            f"{_ADMISSION_FROM_EMERGENCY}*delta_emergency"
        )

        # Capacities / staffing
        sim_total_beds = (
            int(changes.total_bed_capacity)
            if changes.total_bed_capacity is not None
            else int(base.total_beds)
        )
        sim_icu_cap = (
            int(changes.icu_capacity)
            if changes.icu_capacity is not None
            else int(base.total_icu_beds)
        )
        if sim_total_beds < 1:
            raise ValueError("total_bed_capacity must be >= 1")
        if sim_icu_cap < 1:
            raise ValueError("icu_capacity must be >= 1")

        sim_doctors = (
            int(changes.doctors_available)
            if changes.doctors_available is not None
            else int(base.doctors_available)
        )
        sim_nurses = (
            int(changes.nurses_available)
            if changes.nurses_available is not None
            else int(base.nurses_available)
        )
        if sim_doctors < 0 or sim_nurses < 0:
            raise ValueError("Staff counts cannot be negative.")

        # Bed requirement / occupancy (rule)
        extra_beds = int(round(_BED_FROM_ADMISSION * admission_delta))
        bed_requirement = max(0, int(base.occupied_beds + extra_beds))
        occupied_beds = min(bed_requirement, sim_total_beds)
        if bed_requirement > sim_total_beds:
            constraints.append(
                f"Bed requirement {bed_requirement} clipped to capacity {sim_total_beds}."
            )
            bed_requirement = sim_total_beds
            occupied_beds = sim_total_beds
        occupied_beds = max(0, occupied_beds)
        available_beds = compute_available_beds(sim_total_beds, occupied_beds)
        # available cannot exceed total (enforced by compute_available_beds)
        if available_beds > sim_total_beds:
            available_beds = sim_total_beds
            constraints.append("Available beds capped at total bed capacity.")
        bed_occ = compute_occupancy_percentage(occupied_beds, sim_total_beds)

        rule_derived["bed_requirement_rule"] = (
            f"occupied_baseline + {_BED_FROM_ADMISSION}*admission_delta"
        )
        rule_derived["bed_requirement"] = bed_requirement
        rule_derived["occupied_beds"] = occupied_beds
        rule_derived["available_beds"] = available_beds
        rule_derived["bed_occupancy_percentage"] = bed_occ

        # ICU pressure (rule)
        icu_delta = _ICU_FROM_EMERGENCY * delta_em
        raw_icu = base.occupied_icu_beds + icu_delta
        occupied_icu = _clip_int(raw_icu, 0, sim_icu_cap)
        if raw_icu > sim_icu_cap:
            constraints.append(
                f"ICU demand {raw_icu:.1f} clipped to ICU capacity {sim_icu_cap}."
            )
        elif (
            changes.icu_capacity is not None
            and base.occupied_icu_beds > sim_icu_cap
        ):
            constraints.append(
                "Occupied ICU beds reduced to new ICU capacity (cannot exceed capacity)."
            )

        icu_occ = compute_occupancy_percentage(occupied_icu, sim_icu_cap)
        rule_derived["icu_pressure_rule"] = (
            f"occupied_icu += {_ICU_FROM_EMERGENCY}*delta_emergency, clipped to capacity"
        )
        rule_derived["occupied_icu_beds"] = occupied_icu
        rule_derived["icu_occupancy_percentage"] = icu_occ
        rule_derived["icu_pressure"] = icu_occ

        # Patient load, medicine, staff (rule)
        patient_load = _patient_load(sim_opd, sim_emergency, sim_admissions)
        medicine = _medicine_consumption(sim_opd, sim_emergency, sim_admissions)
        staff_ratio, staff_pressure = _staff_metrics(patient_load, sim_doctors, sim_nurses)
        rule_derived["patient_load"] = patient_load
        rule_derived["medicine_consumption_units"] = medicine
        rule_derived["medicine_rule"] = (
            f"{_MED_OPD}*opd + {_MED_EMERGENCY}*emergency + {_MED_ADMISSION}*admissions"
        )
        rule_derived["staff_to_patient_ratio"] = staff_ratio
        rule_derived["staff_pressure"] = staff_pressure
        rule_derived["staff_pressure_rule"] = "patient_load / (doctors + nurses)"

        # ----- Waiting time: ML if service available, else rule -----
        wait_source = "rule_derived"
        expected_wait: float
        if self.prediction_service is not None:
            wait_input = {
                "opd_patients": sim_opd,
                "emergency_patients": sim_emergency,
                "doctors_available": sim_doctors,
                "nurses_available": sim_nurses,
                "is_weekend": int(base.is_weekend),
                "is_holiday": int(base.is_holiday),
                "day_of_week": int(base.day_of_week),
                "month": int(base.month),
                "bed_occupancy_percentage": bed_occ,
                "admissions": sim_admissions,
            }
            try:
                pred = self.prediction_service.predict_waiting_time(wait_input)
                expected_wait = float(pred.iloc[0]["predicted_waiting_time_minutes"])
                expected_wait = float(max(0.0, expected_wait))
                wait_source = "ml_derived"
                ml_derived["expected_waiting_time_minutes"] = expected_wait
                ml_derived["waiting_time_model"] = "Phase 3 waiting-time regressor"
                ml_derived["waiting_time_features_used"] = list(wait_input.keys())
            except Exception as exc:  # keep simulation usable if model fails
                expected_wait = _rule_waiting_time(sim_opd, sim_doctors)
                wait_source = "rule_derived"
                rule_derived["expected_waiting_time_minutes"] = expected_wait
                rule_derived["waiting_time_fallback_reason"] = str(exc)
        else:
            expected_wait = _rule_waiting_time(sim_opd, sim_doctors)
            rule_derived["expected_waiting_time_minutes"] = expected_wait
            rule_derived["waiting_time_rule"] = (
                f"clip({_WAIT_INTERCEPT} + {_WAIT_OPD_COEF}*opd - "
                f"{_WAIT_DOC_COEF}*doctors, 5, 180)"
            )

        capacity_status = compute_capacity_status(bed_occ, icu_occ, expected_wait)
        rule_derived["capacity_status"] = capacity_status

        # Baseline metrics (medicine/staff recomputed for fair comparison)
        base_load = _patient_load(base.current_opd, base.emergency_cases, base.admissions)
        base_med = _medicine_consumption(
            base.current_opd, base.emergency_cases, base.admissions
        )
        base_wait_source = "baseline_state"
        baseline_metrics = metrics_from_state(
            base,
            waiting_time=float(base.average_waiting_time),
            waiting_time_source=base_wait_source,
            medicine_consumption=base_med,
            bed_requirement=int(base.occupied_beds),
        )

        sim_state = HospitalState(
            as_of_date=base.as_of_date,
            total_beds=sim_total_beds,
            occupied_beds=occupied_beds,
            available_beds=available_beds,
            bed_occupancy_percentage=bed_occ,
            total_icu_beds=sim_icu_cap,
            occupied_icu_beds=occupied_icu,
            icu_occupancy_percentage=icu_occ,
            current_opd=sim_opd,
            emergency_cases=sim_emergency,
            doctors_available=sim_doctors,
            nurses_available=sim_nurses,
            average_waiting_time=expected_wait,
            medicine_stock=max(0, int(base.medicine_stock - max(0, medicine - base_med))),
            admissions=sim_admissions,
            discharges=base.discharges,
            is_weekend=base.is_weekend,
            is_holiday=base.is_holiday,
            day_of_week=base.day_of_week,
            month=base.month,
            capacity_status=capacity_status,  # type: ignore[arg-type]
        )

        simulated_metrics = metrics_from_state(
            sim_state,
            waiting_time=expected_wait,
            waiting_time_source=wait_source,
            medicine_consumption=medicine,
            bed_requirement=bed_requirement,
        )

        deltas = _compute_deltas(baseline_metrics, simulated_metrics)
        explanations = _build_explanations(
            changes=changes,
            baseline=baseline_metrics,
            simulated=simulated_metrics,
            deltas=deltas,
            wait_source=wait_source,
            constraints=constraints,
        )

        return WhatIfResult(
            baseline=baseline_metrics,
            simulated=simulated_metrics,
            changes=changes,
            deltas=deltas,
            rule_derived=rule_derived,
            ml_derived=ml_derived,
            explanations=explanations,
            constraints_applied=constraints,
        )


def _compute_deltas(
    baseline: ScenarioMetrics, simulated: ScenarioMetrics
) -> dict[str, float]:
    keys = [
        "opd_patients",
        "emergency_cases",
        "admissions",
        "patient_load",
        "bed_requirement",
        "bed_occupancy_percentage",
        "icu_pressure",
        "expected_waiting_time_minutes",
        "medicine_consumption_units",
        "staff_pressure",
        "doctors_available",
        "nurses_available",
        "total_beds",
        "total_icu_beds",
        "available_beds",
        "occupied_beds",
        "occupied_icu_beds",
    ]
    out: dict[str, float] = {}
    for key in keys:
        b = float(getattr(baseline, key))
        s = float(getattr(simulated, key))
        out[key] = round(s - b, 4)
    return out


def _build_explanations(
    *,
    changes: ScenarioChanges,
    baseline: ScenarioMetrics,
    simulated: ScenarioMetrics,
    deltas: dict[str, float],
    wait_source: str,
    constraints: list[str],
) -> list[str]:
    lines = [
        "Comparison: Current Scenario vs Simulated Scenario",
        "Scenario changes: " + "; ".join(changes.describe()),
    ]

    def _fmt(delta: float, unit: str = "") -> str:
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f}{unit}"

    lines.append(
        f"Patient load {baseline.patient_load} -> {simulated.patient_load} "
        f"({_fmt(deltas['patient_load'])})."
    )
    lines.append(
        f"Bed occupancy {baseline.bed_occupancy_percentage:.1f}% -> "
        f"{simulated.bed_occupancy_percentage:.1f}% "
        f"({_fmt(deltas['bed_occupancy_percentage'], ' pp')}); "
        f"bed requirement {baseline.bed_requirement} -> {simulated.bed_requirement}."
    )
    lines.append(
        f"ICU pressure {baseline.icu_pressure:.1f}% -> {simulated.icu_pressure:.1f}% "
        f"({_fmt(deltas['icu_pressure'], ' pp')})."
    )
    src_label = (
        "ML-derived prediction (waiting-time model)"
        if wait_source == "ml_derived"
        else "rule-derived simulation (transparent waiting-time formula)"
    )
    lines.append(
        f"Expected waiting time {baseline.expected_waiting_time_minutes:.1f} -> "
        f"{simulated.expected_waiting_time_minutes:.1f} min "
        f"({_fmt(deltas['expected_waiting_time_minutes'], ' min')}); source: {src_label}."
    )
    lines.append(
        f"Medicine consumption {baseline.medicine_consumption_units:.1f} -> "
        f"{simulated.medicine_consumption_units:.1f} units "
        f"({_fmt(deltas['medicine_consumption_units'])}); rule-derived."
    )
    lines.append(
        f"Staff pressure {baseline.staff_pressure:.3f} -> {simulated.staff_pressure:.3f} "
        f"patients/staff ({_fmt(deltas['staff_pressure'])}); rule-derived."
    )
    lines.append(
        f"Capacity status {baseline.capacity_status} -> {simulated.capacity_status}."
    )
    for c in constraints:
        lines.append(f"Constraint applied: {c}")
    lines.append(
        "Note: Rule-derived values use documented formulas; ML-derived values "
        "come only from trained Joblib models when a prediction service is attached."
    )
    return lines


def run_what_if(
    baseline: HospitalState,
    changes: ScenarioChanges,
    prediction_service: Any | None = None,
) -> WhatIfResult:
    """Convenience wrapper around ``WhatIfSimulator.simulate``."""
    return WhatIfSimulator(prediction_service=prediction_service).simulate(
        baseline, changes
    )


__all__ = [
    "ScenarioChanges",
    "ScenarioMetrics",
    "WhatIfResult",
    "WhatIfSimulator",
    "run_what_if",
    "metrics_from_state",
]
