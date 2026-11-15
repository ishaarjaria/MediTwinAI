"""
Glue between the digital twin / prediction service and the alerts engine.

Uses only existing interfaces:
``twin.get_current_state()``, ``twin.get_predicted_state(days_ahead)`` and
``twin.service.detect_hospital_anomaly(...)``. Nothing here changes the twin,
the simulator or the models.
"""

from __future__ import annotations

from typing import Any

from src.alerts.engine import (
    SCOPE_CURRENT,
    SCOPE_PREDICTED,
    AlertEngine,
    ReferenceStats,
    deduplicate_alerts,
    load_reference_stats,
)
from src.alerts.models import AlertReport, AlertThresholds


def anomaly_row_for_state(service: Any, state: Any) -> tuple[dict | None, str | None]:
    """
    Run the saved anomaly model on a state's operational vector.

    Returns ``(row, None)`` or ``(None, reason)`` when the model is unavailable
    or rejects the input, so callers can report it instead of guessing.
    """
    if service is None or not hasattr(service, "detect_hospital_anomaly"):
        return None, "Anomaly: no prediction service available."
    try:
        out = service.detect_hospital_anomaly(
            {
                "opd_patients": state.current_opd,
                "emergency_patients": state.emergency_cases,
                "bed_occupancy_percentage": state.bed_occupancy_percentage,
                "icu_occupancy_percentage": state.icu_occupancy_percentage,
                "average_waiting_time_minutes": state.average_waiting_time,
            }
        )
        return out.iloc[0].to_dict(), None
    except Exception as exc:  # unavailable model or out-of-range input
        return None, f"Anomaly: model could not score this state ({exc})."


def generate_alerts(
    twin: Any,
    *,
    days_ahead: int | None = 1,
    engine: AlertEngine | None = None,
    thresholds: AlertThresholds | None = None,
    reference: ReferenceStats | None = None,
) -> AlertReport:
    """
    Alerts for the twin's current state and, if ``days_ahead`` is set, its
    predicted state. The result is one de-duplicated report.
    """
    if engine is None:
        if reference is None:
            try:
                reference = load_reference_stats()
            except FileNotFoundError:
                reference = None  # z-score checks will report 'unavailable'
        engine = AlertEngine(thresholds=thresholds, reference=reference)

    current = twin.get_current_state()
    service = getattr(twin, "service", None)

    row, why = anomaly_row_for_state(service, current)
    report = engine.evaluate(current, scope=SCOPE_CURRENT, anomaly=row)
    if why:
        report.unavailable = [u for u in report.unavailable if not u.startswith("Anomaly")]
        report.unavailable.append(why)

    alerts = list(report.alerts)
    unavailable = list(report.unavailable)

    if days_ahead is not None:
        predicted = twin.get_predicted_state(days_ahead=days_ahead)
        prow, pwhy = anomaly_row_for_state(service, predicted)
        future = engine.evaluate(
            predicted, scope=SCOPE_PREDICTED, anomaly=prow, compare_to=current
        )
        alerts.extend(future.alerts)
        unavailable.extend(
            u for u in future.unavailable if not (pwhy and u.startswith("Anomaly"))
        )
        if pwhy:
            unavailable.append(f"{pwhy} (predicted state)")

    return AlertReport(
        alerts=deduplicate_alerts(alerts),
        unavailable=unavailable,
        generated_for=report.generated_for,
    )
