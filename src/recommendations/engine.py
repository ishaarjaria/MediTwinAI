"""
Recommendation engine (Phase 6).

Maps alerts to practical, operational suggestions for hospital administrators.

* Every recommendation is **rule-based**: it comes from the fixed table in this
  file, never from a trained model. ``alert_basis`` records whether the alert
  that triggered it was an observed value, a rule-derived value, or an ML
  prediction, so the two are never confused.
* Each recommendation links to its triggering alert(s) by ``alert_id``.
* Recommendations are advisory. They do not guarantee any outcome and contain
  no medical or patient-specific advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.alerts.models import (
    BASIS_ML,
    Alert,
    AlertCategory,
    AlertReport,
    Severity,
)

SOURCE_RULE_BASED = "rule-based"
COMBINED_CAPACITY = "capacity_surge"

DISCLAIMER = (
    "Advisory only: a rule-based operational suggestion for administrators. "
    "It does not guarantee any outcome and is not medical advice."
)

_URGENCY = {
    Severity.LOW: "Monitor and plan",
    Severity.MEDIUM: "Review during the current planning cycle",
    Severity.HIGH: "Prioritise today",
    Severity.CRITICAL: "Escalate to the duty administrator now",
}


@dataclass
class Recommendation:
    recommendation_id: str
    alert_id: str  # primary triggering alert
    related_alert_ids: list[str]
    category: str  # alert category value, or a combined-rule name
    severity: Severity
    title: str
    action: str
    rationale: str
    source: str = SOURCE_RULE_BASED
    alert_basis: str = ""  # observed | ml_prediction | rule_derived
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.label
        return data


# Each rule returns (title, action) for an alert.
def _bed(a: Alert) -> tuple[str, str]:
    return (
        "Review bed allocation",
        "Review bed allocation and discharge planning, and identify beds that "
        "could be made available or reallocated across wards.",
    )


def _icu(a: Alert) -> tuple[str, str]:
    return (
        "Review ICU capacity",
        "Review ICU capacity and step-down options, and confirm the escalation "
        "plan if ICU occupancy keeps rising.",
    )


def _opd_inflow(a: Alert) -> tuple[str, str]:
    return (
        "Prepare for high OPD load",
        "Check OPD staffing and registration capacity and consider opening "
        "additional consultation slots while the load stays unusually high.",
    )


def _opd_rising(a: Alert) -> tuple[str, str]:
    return (
        "Prepare for increased patient inflow",
        "Plan ahead for higher OPD demand: review staffing rosters, consultation "
        "capacity and supplies for the forecast day. The forecast is uncertain, "
        "so re-check it as the day approaches.",
    )


def _med_stock(a: Alert) -> tuple[str, str]:
    return (
        "Check medicine replenishment",
        "Check stock of critical medicines against the pharmacy's reorder lead "
        "times and place or expedite replenishment orders if needed.",
    )


def _med_use(a: Alert) -> tuple[str, str]:
    return (
        "Review medicine usage and supply",
        "Review medicine usage with pharmacy, confirm replenishment lead times, "
        "and check that stock levels cover the higher expected demand.",
    )


def _staff(a: Alert) -> tuple[str, str]:
    return (
        "Consider additional staff",
        "Consider additional doctor or nurse cover, or rebalancing staff "
        "between departments, while workload per staff member is high.",
    )


def _anomaly(a: Alert) -> tuple[str, str]:
    return (
        "Investigate unusual activity",
        "Review the flagged indicators with the operations team to confirm "
        "whether the change is real, a data issue, or a one-off event.",
    )


_RULES: dict[AlertCategory, Callable[[Alert], tuple[str, str]]] = {
    AlertCategory.BED_OCCUPANCY: _bed,
    AlertCategory.ICU_PRESSURE: _icu,
    AlertCategory.OPD_INFLOW: _opd_inflow,
    AlertCategory.OPD_RISING: _opd_rising,
    AlertCategory.MEDICINE_STOCK: _med_stock,
    AlertCategory.MEDICINE_CONSUMPTION: _med_use,
    AlertCategory.STAFF_WORKLOAD: _staff,
    AlertCategory.ANOMALY: _anomaly,
}


def _why(alert: Alert) -> str:
    source = {
        "observed": "an observed value",
        "rule_derived": "a rule-derived value",
        "ml_prediction": "an ML prediction",
    }.get(alert.basis, alert.basis)
    return (
        f"Generated because alert '{alert.title}' fired "
        f"({alert.metric} = {alert.observed_value:g}; condition: {alert.threshold}). "
        f"The alert is based on {source}; the recommendation itself is a fixed rule."
    )


def recommend_for_alert(alert: Alert) -> Recommendation:
    """Build the single recommendation for one alert."""
    rule = _RULES.get(alert.category)
    if rule is None:  # defensive: unknown category
        raise ValueError(f"No recommendation rule for category {alert.category!r}")
    title, action = rule(alert)
    when = "predicted" if alert.scope == "predicted" else "current"
    return Recommendation(
        recommendation_id=f"rec:{alert.alert_id}",
        alert_id=alert.alert_id,
        related_alert_ids=[alert.alert_id],
        category=alert.category.value,
        severity=alert.severity,
        title=f"{title} ({when}, {alert.severity.label})",
        action=f"{_URGENCY[alert.severity]}: {action}",
        rationale=_why(alert),
        alert_basis=alert.basis,
    )


def _combined_capacity(alerts: list[Alert]) -> Recommendation | None:
    """Beds and ICU both High+ on the same scope: suggest a surge-plan review."""
    for scope in ("current", "predicted"):
        bed = next(
            (a for a in alerts if a.category == AlertCategory.BED_OCCUPANCY and a.scope == scope),
            None,
        )
        icu = next(
            (a for a in alerts if a.category == AlertCategory.ICU_PRESSURE and a.scope == scope),
            None,
        )
        if bed and icu and bed.severity >= Severity.HIGH and icu.severity >= Severity.HIGH:
            top = max(bed.severity, icu.severity)
            return Recommendation(
                recommendation_id=f"rec:{COMBINED_CAPACITY}:{scope}:{bed.date or 'undated'}",
                alert_id=bed.alert_id,
                related_alert_ids=[bed.alert_id, icu.alert_id],
                category=COMBINED_CAPACITY,
                severity=top,
                title=f"Review hospital surge plan ({scope}, {top.label})",
                action=(
                    f"{_URGENCY[top]}: bed and ICU occupancy are both high. Review "
                    "the surge/overflow plan, elective-admission scheduling and "
                    "inter-ward transfer options."
                ),
                rationale=(
                    "Generated because both bed-occupancy and ICU-pressure alerts "
                    "are High or Critical in the same scope. Fixed combined rule."
                ),
                alert_basis=bed.basis if bed.basis == icu.basis else "mixed",
            )
    return None


def generate_recommendations(
    alerts: list[Alert] | AlertReport,
) -> list[Recommendation]:
    """
    One recommendation per alert, plus a combined rule where applicable.
    Sorted most severe first. Empty input (normal conditions) gives ``[]``.
    """
    items = alerts.alerts if isinstance(alerts, AlertReport) else list(alerts)
    recs = [recommend_for_alert(a) for a in items]
    combined = _combined_capacity(items)
    if combined:
        recs.append(combined)
    return sorted(recs, key=lambda r: (-int(r.severity), r.category, r.recommendation_id))


__all__ = [
    "BASIS_ML",
    "COMBINED_CAPACITY",
    "DISCLAIMER",
    "Recommendation",
    "SOURCE_RULE_BASED",
    "generate_recommendations",
    "recommend_for_alert",
]
