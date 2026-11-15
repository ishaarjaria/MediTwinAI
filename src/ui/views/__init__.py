"""View registry and navigation helpers (Phase 7)."""

from __future__ import annotations

import importlib
from typing import Callable

# id, sidebar label, Material icon
VIEWS: list[tuple[str, str, str]] = [
    ("overview", "Overview", ":material/space_dashboard:"),
    ("alerts", "Alerts", ":material/notifications_active:"),
    ("analytics", "Trends", ":material/monitoring:"),
    ("predictions", "Forecasts", ":material/query_stats:"),
    ("patient_risk", "Patient Risk", ":material/stethoscope:"),
    ("twin", "Hospital Twin", ":material/hub:"),
    ("whatif", "What-If", ":material/tune:"),
]
# Sidebar sections: group label -> view ids (every view appears exactly once).
NAV_GROUPS: list[tuple[str, list[str]]] = [
    ("Today", ["overview", "alerts", "analytics"]),
    ("Look ahead", ["predictions", "patient_risk"]),
    ("Plan", ["twin", "whatif"]),
]
# One-line description of each page, shown as the tooltip on its navigation icon.
VIEW_HINTS: dict[str, str] = {
    "overview": "Hospital status today",
    "alerts": "What needs attention, and what to do",
    "analytics": "Demand, capacity and staffing over time",
    "predictions": "What to expect in the coming days",
    "patient_risk": "Risk group for a synthetic patient record",
    "twin": "Today beside the expected state",
    "whatif": "Try a scenario before deciding",
}
VIEW_IDS = [v[0] for v in VIEWS]
DEFAULT_VIEW = "overview"


def normalize_view(value: str | None) -> str:
    """Valid view id for a (possibly missing or bad) query-parameter value."""
    return value if value in VIEW_IDS else DEFAULT_VIEW


def go_to(view_id: str) -> None:
    """Button callback: switch the active view."""
    import streamlit as st

    st.session_state["view"] = normalize_view(view_id)


def get_renderer(view_id: str) -> Callable[[], None]:
    module = importlib.import_module(f"src.ui.views.{normalize_view(view_id)}")
    return module.render
