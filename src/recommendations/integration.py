"""Twin -> alerts -> recommendations in one call (Phase 6)."""

from __future__ import annotations

from typing import Any

from src.alerts.integration import generate_alerts
from src.alerts.models import AlertReport
from src.recommendations.engine import Recommendation, generate_recommendations


def generate_operational_report(
    twin: Any, **alert_kwargs: Any
) -> tuple[AlertReport, list[Recommendation]]:
    """
    Alerts (current + optional predicted state) and the linked recommendations.

    ``alert_kwargs`` are passed to ``generate_alerts`` (for example
    ``days_ahead``, ``thresholds`` or ``reference``).
    """
    report = generate_alerts(twin, **alert_kwargs)
    return report, generate_recommendations(report)
