"""
Alerts engine (Phase 6).

Turns a hospital state (current or predicted) plus optional model outputs
into operational alerts. Pure functions over existing project interfaces:

* ``HospitalState`` from ``src.twin.hospital_twin`` (or a mapping with the
  same field names),
* the anomaly dict/row produced by ``PredictionService.detect_hospital_anomaly``,
* reference statistics computed from the processed hospital dataset.

If a needed metric or reference value is missing, the check is skipped and
listed in ``AlertReport.unavailable``. No value is invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.alerts.models import (
    BASIS_ML,
    BASIS_OBSERVED,
    BASIS_RULE,
    Alert,
    AlertCategory,
    AlertReport,
    AlertThresholds,
    Severity,
)
from src.data.preprocess import get_project_root
from src.simulation.what_if import medicine_consumption, staff_pressure

SCOPE_CURRENT = "current"
SCOPE_PREDICTED = "predicted"


# ----------------------------------------------------------------------
# Reference statistics (what "unusual" is measured against)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceStats:
    """Historical mean/std used for z-score alerts. All fields optional."""

    opd_mean: float | None = None
    opd_std: float | None = None
    medicine_mean: float | None = None
    medicine_std: float | None = None
    staff_pressure_mean: float | None = None
    staff_pressure_std: float | None = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ReferenceStats":
        """Build from hospital-operations rows (same columns as the dataset)."""
        needed = {
            "opd_patients",
            "emergency_patients",
            "admissions",
            "doctors_available",
            "nurses_available",
        }
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Reference data missing columns: {sorted(missing)}")
        if len(df) < 30:
            raise ValueError("Reference data needs at least 30 rows.")
        opd = df["opd_patients"].astype(float)
        med = df.apply(
            lambda r: medicine_consumption(
                r["opd_patients"], r["emergency_patients"], r["admissions"]
            ),
            axis=1,
        )
        pressure = df.apply(
            lambda r: staff_pressure(
                r["opd_patients"] + r["emergency_patients"] + r["admissions"],
                int(r["doctors_available"]),
                int(r["nurses_available"]),
            ),
            axis=1,
        )
        pressure = pressure[pressure.apply(math.isfinite)]
        return cls(
            opd_mean=float(opd.mean()),
            opd_std=float(opd.std(ddof=0)),
            medicine_mean=float(med.mean()),
            medicine_std=float(med.std(ddof=0)),
            staff_pressure_mean=float(pressure.mean()),
            staff_pressure_std=float(pressure.std(ddof=0)),
        )


def load_reference_stats(path: Path | None = None) -> ReferenceStats:
    """Reference statistics from the processed hospital features CSV."""
    path = path or (
        get_project_root() / "data" / "processed" / "hospital_operations_features.csv"
    )
    if not Path(path).is_file():
        raise FileNotFoundError(f"Reference hospital data not found: {path}")
    return ReferenceStats.from_dataframe(pd.read_csv(path))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _get(state: Any, name: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _number(value: Any) -> float | None:
    """Finite non-negative float, else None (treated as unavailable)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _severity_ascending(
    value: float, ladder: tuple[float, float, float, float]
) -> tuple[Severity, float] | None:
    """Highest severity whose threshold is <= value (boundary inclusive)."""
    hit: tuple[Severity, float] | None = None
    for severity, threshold in zip(Severity, ladder):
        if value >= threshold:
            hit = (severity, threshold)
    return hit


def _severity_descending(
    value: float, ladder: tuple[float, float, float, float]
) -> tuple[Severity, float] | None:
    """Highest severity whose threshold is > value (boundary exclusive)."""
    hit: tuple[Severity, float] | None = None
    for severity, threshold in zip(Severity, ladder):
        if value < threshold:
            hit = (severity, threshold)
    return hit


def _zscore(value: float, mean: float | None, std: float | None) -> float | None:
    if mean is None or std is None or not math.isfinite(mean) or std <= 1e-9:
        return None
    return (value - mean) / std


def deduplicate_alerts(alerts: list[Alert]) -> list[Alert]:
    """Keep one alert per (category, scope, date): the most severe one."""
    best: dict[tuple, Alert] = {}
    for alert in alerts:
        current = best.get(alert.dedup_key)
        if current is None or alert.severity > current.severity:
            best[alert.dedup_key] = alert
    return sorted(
        best.values(), key=lambda a: (-int(a.severity), a.category.value, a.scope)
    )


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class AlertEngine:
    """Evaluate a hospital state against configurable thresholds."""

    def __init__(
        self,
        thresholds: AlertThresholds | None = None,
        reference: ReferenceStats | None = None,
    ) -> None:
        self.thresholds = thresholds or AlertThresholds()
        self.reference = reference

    # -- public ---------------------------------------------------------
    def evaluate(
        self,
        state: Any,
        *,
        scope: str = SCOPE_CURRENT,
        anomaly: Mapping[str, Any] | None = None,
        compare_to: Any | None = None,
        compare_kind: str = "forecast",
    ) -> AlertReport:
        """
        Evaluate ``state`` (HospitalState or mapping).

        ``scope`` is ``"current"`` or ``"predicted"`` and controls how each
        alert's source is labelled. ``anomaly`` is one row from
        ``detect_hospital_anomaly`` (``anomaly_flag``, ``anomaly_score``,
        ``anomaly_reason``). ``compare_to`` is the current state; with a
        predicted ``state`` it enables the rising-demand check.
        ``compare_kind`` is ``"forecast"`` (``state`` is an ML forecast) or
        ``"scenario"`` (``state`` is a user-defined what-if scenario, so the
        alert is labelled rule-derived and worded as a projection).
        """
        if scope not in (SCOPE_CURRENT, SCOPE_PREDICTED):
            raise ValueError("scope must be 'current' or 'predicted'.")
        date = _get(state, "as_of_date")
        date = str(date)[:10] if date else None
        report = AlertReport(generated_for=date)
        raw: list[Alert] = []

        for check in (
            self._bed,
            self._icu,
            self._opd_inflow,
            self._medicine_stock,
            self._medicine_consumption,
            self._staff,
        ):
            alert = check(state, scope, date, report.unavailable)
            if alert:
                raw.append(alert)

        alert = self._anomaly(anomaly, scope, date, report.unavailable)
        if alert:
            raw.append(alert)

        if compare_to is not None:
            alert = self._opd_rising(
                state, compare_to, scope, date, report.unavailable, compare_kind
            )
            if alert:
                raw.append(alert)

        report.alerts = deduplicate_alerts(raw)
        return report

    # -- builders -------------------------------------------------------
    @staticmethod
    def _make(
        category: AlertCategory,
        severity: Severity,
        *,
        title: str,
        explanation: str,
        metric: str,
        value: float,
        threshold: str,
        threshold_value: float | None,
        date: str | None,
        basis: str,
        scope: str,
        details: dict[str, Any] | None = None,
    ) -> Alert:
        return Alert(
            alert_id=f"{category.value}:{scope}:{date or 'undated'}",
            category=category,
            severity=severity,
            title=title,
            explanation=explanation,
            metric=metric,
            observed_value=round(float(value), 2),
            threshold=threshold,
            threshold_value=threshold_value,
            date=date,
            basis=basis,
            scope=scope,
            details=details or {},
        )

    @staticmethod
    def _capacity_basis(scope: str) -> str:
        # Projected beds/ICU are heuristics in the twin, not ML outputs.
        return BASIS_OBSERVED if scope == SCOPE_CURRENT else BASIS_RULE

    def _occupancy(
        self,
        state: Any,
        scope: str,
        date: str | None,
        unavailable: list[str],
        *,
        field: str,
        ladder: tuple[float, float, float, float],
        category: AlertCategory,
        label: str,
    ) -> Alert | None:
        value = _number(_get(state, field))
        if value is None or value > 100:
            unavailable.append(f"{label}: '{field}' missing or invalid.")
            return None
        hit = _severity_ascending(value, ladder)
        if hit is None:
            return None
        severity, threshold = hit
        when = "projected" if scope == SCOPE_PREDICTED else "current"
        return self._make(
            category,
            severity,
            title=f"{severity.label}: {label} at {value:.1f}%",
            explanation=(
                f"{when.capitalize()} {label.lower()} is {value:.1f}%, at or above "
                f"the {severity.label} threshold of {threshold:.0f}%."
            ),
            metric=field,
            value=value,
            threshold=f"{field} >= {threshold:.0f}",
            threshold_value=threshold,
            date=date,
            basis=self._capacity_basis(scope),
            scope=scope,
        )

    def _bed(self, state, scope, date, unavailable):
        return self._occupancy(
            state,
            scope,
            date,
            unavailable,
            field="bed_occupancy_percentage",
            ladder=self.thresholds.bed_occupancy,
            category=AlertCategory.BED_OCCUPANCY,
            label="Bed occupancy",
        )

    def _icu(self, state, scope, date, unavailable):
        return self._occupancy(
            state,
            scope,
            date,
            unavailable,
            field="icu_occupancy_percentage",
            ladder=self.thresholds.icu_occupancy,
            category=AlertCategory.ICU_PRESSURE,
            label="ICU occupancy",
        )

    def _opd_inflow(self, state, scope, date, unavailable):
        value = _number(_get(state, "current_opd"))
        if value is None:
            unavailable.append("OPD inflow: 'current_opd' missing or invalid.")
            return None
        ref = self.reference
        z = _zscore(value, ref.opd_mean if ref else None, ref.opd_std if ref else None)
        if z is None:
            unavailable.append("OPD inflow: no reference statistics available.")
            return None
        hit = _severity_ascending(z, self.thresholds.opd_zscore)
        if hit is None:
            return None
        severity, threshold = hit
        return self._make(
            AlertCategory.OPD_INFLOW,
            severity,
            title=f"{severity.label}: unusually high OPD load ({value:.0f})",
            explanation=(
                f"OPD load of {value:.0f} is {z:.1f} standard deviations above the "
                f"historical mean of {ref.opd_mean:.0f}."
            ),
            metric="current_opd",
            value=value,
            threshold=f"OPD z-score >= {threshold:g}",
            threshold_value=threshold,
            date=date,
            basis=BASIS_OBSERVED if scope == SCOPE_CURRENT else BASIS_ML,
            scope=scope,
            details={"z_score": round(z, 2), "historical_mean": round(ref.opd_mean, 1)},
        )

    def _daily_consumption(self, state) -> float | None:
        opd = _number(_get(state, "current_opd"))
        emergency = _number(_get(state, "emergency_cases"))
        admissions = _number(_get(state, "admissions"))
        if opd is None or emergency is None or admissions is None:
            return None
        return medicine_consumption(opd, emergency, admissions)

    def _medicine_stock(self, state, scope, date, unavailable):
        stock = _number(_get(state, "medicine_stock"))
        consumption = self._daily_consumption(state)
        if stock is None or consumption is None:
            unavailable.append("Medicine stock: stock or load inputs missing.")
            return None
        if consumption <= 0:
            return None  # no consumption, no depletion risk to report
        cover = stock / consumption
        hit = _severity_descending(cover, self.thresholds.medicine_days_cover)
        if hit is None:
            return None
        severity, threshold = hit
        return self._make(
            AlertCategory.MEDICINE_STOCK,
            severity,
            title=f"{severity.label}: medicine stock covers about {cover:.1f} days",
            explanation=(
                f"Stock of {stock:.0f} units at an estimated {consumption:.0f} "
                f"units/day gives {cover:.1f} days of cover, below {threshold:.0f} days. "
                "Consumption is a rule-derived estimate, not a trained model."
            ),
            metric="medicine_days_of_cover",
            value=cover,
            threshold=f"days of cover < {threshold:g}",
            threshold_value=threshold,
            date=date,
            basis=BASIS_RULE,
            scope=scope,
            details={"stock_units": stock, "est_daily_consumption": consumption},
        )

    def _medicine_consumption(self, state, scope, date, unavailable):
        consumption = self._daily_consumption(state)
        if consumption is None:
            unavailable.append("Medicine consumption: load inputs missing.")
            return None
        ref = self.reference
        z = _zscore(
            consumption,
            ref.medicine_mean if ref else None,
            ref.medicine_std if ref else None,
        )
        if z is None:
            unavailable.append("Medicine consumption: no reference statistics available.")
            return None
        hit = _severity_ascending(z, self.thresholds.medicine_consumption_zscore)
        if hit is None:
            return None
        severity, threshold = hit
        return self._make(
            AlertCategory.MEDICINE_CONSUMPTION,
            severity,
            title=f"{severity.label}: unusually high medicine consumption",
            explanation=(
                f"Estimated consumption of {consumption:.0f} units/day is {z:.1f} "
                f"standard deviations above the historical mean of "
                f"{ref.medicine_mean:.0f}. Rule-derived estimate from OPD, "
                "emergency and admission counts."
            ),
            metric="estimated_medicine_consumption",
            value=consumption,
            threshold=f"consumption z-score >= {threshold:g}",
            threshold_value=threshold,
            date=date,
            basis=BASIS_RULE,
            scope=scope,
            details={"z_score": round(z, 2)},
        )

    def _staff(self, state, scope, date, unavailable):
        opd = _number(_get(state, "current_opd"))
        emergency = _number(_get(state, "emergency_cases"))
        admissions = _number(_get(state, "admissions"))
        doctors = _number(_get(state, "doctors_available"))
        nurses = _number(_get(state, "nurses_available"))
        if None in (opd, emergency, admissions, doctors, nurses):
            unavailable.append("Staff workload: load or staffing inputs missing.")
            return None
        load = opd + emergency + admissions
        pressure = staff_pressure(load, int(doctors), int(nurses))
        ref = self.reference
        if not math.isfinite(pressure):
            # Patients but no clinical staff at all.
            return self._make(
                AlertCategory.STAFF_WORKLOAD,
                Severity.CRITICAL,
                title="Critical: patient load with no clinical staff available",
                explanation="Doctors and nurses available are both zero.",
                metric="patient_load",
                value=load,
                threshold="doctors + nurses == 0",
                threshold_value=0.0,
                date=date,
                basis=BASIS_RULE,
                scope=scope,
                details={"patient_load": load},
            )
        z = _zscore(
            pressure,
            ref.staff_pressure_mean if ref else None,
            ref.staff_pressure_std if ref else None,
        )
        if z is None:
            unavailable.append("Staff workload: no reference statistics available.")
            return None
        hit = _severity_ascending(z, self.thresholds.staff_pressure_zscore)
        if hit is None:
            return None
        severity, threshold = hit
        return self._make(
            AlertCategory.STAFF_WORKLOAD,
            severity,
            title=f"{severity.label}: high staff workload ({pressure:.2f} patients/staff)",
            explanation=(
                f"Patient load per clinical staff member is {pressure:.2f}, "
                f"{z:.1f} standard deviations above the historical mean of "
                f"{ref.staff_pressure_mean:.2f}. Rule-derived ratio."
            ),
            metric="staff_pressure_patients_per_staff",
            value=pressure,
            threshold=f"staff-pressure z-score >= {threshold:g}",
            threshold_value=threshold,
            date=date,
            basis=BASIS_RULE,
            scope=scope,
            details={"z_score": round(z, 2), "patient_load": load},
        )

    def _anomaly(self, anomaly, scope, date, unavailable):
        if anomaly is None:
            unavailable.append("Anomaly: no anomaly-detection output supplied.")
            return None
        try:
            flag = int(anomaly["anomaly_flag"])
            score = float(anomaly["anomaly_score"])
        except (KeyError, TypeError, ValueError):
            unavailable.append("Anomaly: output missing 'anomaly_flag'/'anomaly_score'.")
            return None
        if not math.isfinite(score) or flag not in (0, 1):
            unavailable.append("Anomaly: output contained invalid values.")
            return None
        if flag == 0:
            return None
        high = self.thresholds.anomaly_high_score
        severity = Severity.HIGH if score >= high else Severity.MEDIUM
        reason = str(anomaly.get("anomaly_reason") or "").strip()
        return self._make(
            AlertCategory.ANOMALY,
            severity,
            title=f"{severity.label}: unusual hospital activity detected",
            explanation=(
                "The Isolation Forest model flagged this day as unusual compared "
                "with normal activity"
                + (f" ({reason})." if reason else ".")
                + " A flag means 'different from past data', not 'harmful'."
            ),
            metric="anomaly_score",
            value=score,
            threshold=f"anomaly_flag == 1; High if score >= {high:g}",
            threshold_value=high,
            date=date,
            basis=BASIS_ML,
            scope=scope,
            details={"reason": reason},
        )

    def _opd_rising(self, state, compare_to, scope, date, unavailable, kind="forecast"):
        future = _number(_get(state, "current_opd"))
        now = _number(_get(compare_to, "current_opd"))
        if future is None or now is None or now <= 0:
            unavailable.append("Rising OPD: current or forecast OPD missing.")
            return None
        pct = 100.0 * (future - now) / now
        hit = _severity_ascending(pct, self.thresholds.rising_opd_percent)
        if hit is None:
            return None
        severity, threshold = hit
        scenario = kind == "scenario"
        return self._make(
            AlertCategory.OPD_RISING,
            severity,
            title=(
                f"{severity.label}: OPD demand projected to rise {pct:.0f}%"
                if scenario
                else f"{severity.label}: OPD demand forecast to rise {pct:.0f}%"
            ),
            explanation=(
                f"The scenario sets OPD to {future:.0f}, {pct:.0f}% above the baseline "
                f"{now:.0f}. This is a user-defined what-if value (rule-derived), not a forecast."
                if scenario
                else f"The OPD forecast for {date} is {future:.0f}, {pct:.0f}% above the "
                f"current {now:.0f}. The forecast is an ML prediction (trained forecaster) "
                "and carries uncertainty."
            ),
            metric="forecast_opd_change_percent",
            value=pct,
            threshold=(
                f"scenario OPD >= baseline OPD + {threshold:g}%"
                if scenario
                else f"forecast OPD >= current OPD + {threshold:g}%"
            ),
            threshold_value=threshold,
            date=date,
            basis=BASIS_RULE if scenario else BASIS_ML,
            scope=scope,
            details={"forecast_opd": future, "current_opd": now},
        )
