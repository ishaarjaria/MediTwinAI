"""
Cached access to the existing MediTwin backend for the dashboard (Phase 7).

Thin adapters only. Heavy objects (prediction service with the OPD forecaster / Isolation
Forest, the twin) are created once per server process with ``cache_resource``.
Derived results (forecast, predicted state, alert report, what-if runs) are
cached by their inputs with ``cache_data``. Every public function returns an
``Outcome`` so views can show an unavailable/error state instead of crashing.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import pandas as pd
import streamlit as st

from src.alerts.engine import SCOPE_PREDICTED, AlertEngine, ReferenceStats, load_reference_stats
from src.alerts.integration import anomaly_row_for_state
from src.models.prediction_service import get_prediction_service
from src.recommendations.engine import generate_recommendations
from src.recommendations.integration import generate_operational_report
from src.simulation.what_if import WhatIfSimulator
from src.twin.hospital_twin import (
    HospitalDigitalTwin,
    HospitalState,
    build_twin_from_latest,
    hospital_state_from_row,
)
from src.ui import data as ui_data


class Outcome(NamedTuple):
    """A value or a human-readable error. Exactly one is set."""

    value: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _safe(fn: Callable[[], Any]) -> Outcome:
    try:
        return Outcome(value=fn())
    except Exception as exc:  # surfaced in the UI as an unavailable state
        return Outcome(error=f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------
# Load-once resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _service():
    return get_prediction_service()


@st.cache_resource(show_spinner=False)
def _twin() -> HospitalDigitalTwin:
    return build_twin_from_latest(prediction_service=_service())


@st.cache_resource(show_spinner=False)
def _reference() -> ReferenceStats | None:
    try:
        return load_reference_stats()
    except FileNotFoundError:
        return None


def warm_up() -> Outcome:
    """Load models and the twin once (shown behind a spinner at start-up)."""
    return _safe(lambda: (_service(), _twin()) and True)


def reference_stats() -> ReferenceStats | None:
    return _reference()


@st.cache_resource(show_spinner=False)
def _prefetch_state() -> dict:
    return {"done": False}


def is_ready() -> bool:
    """True once ``prefetch`` has completed in this server process."""
    return bool(_prefetch_state()["done"])


def prefetch() -> None:
    """Fill the data caches and import every view once per server process.

    Without this the first visit to each page pays for loading its data and importing its
    modules (several seconds on a cold start). It only calls the existing cached accessors,
    so no result changes; it just moves the cost to the first workspace load.
    """
    state = _prefetch_state()
    if state["done"]:
        return
    try:
        import importlib

        from src.ui.views import VIEW_IDS

        for view_id in VIEW_IDS:
            importlib.import_module(f"src.ui.views.{view_id}")
        ops = operations()
        anomaly_scores(); patients(); risk_metrics(); model_cards(); opd_model_info()
        current_state(); current_model_wait()
        for horizon in range(1, 8):
            predicted_state(horizon)
        forecast(7)
        operational_report(1)
        if ops.ok:
            baseline_state(str(ops.value["date"].max().date()))
        state["done"] = True
    except Exception:  # prefetch is an optimisation; any failure surfaces on the page itself
        state["done"] = False


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _operations():
    return ui_data.load_operations()


@st.cache_data(show_spinner=False)
def _anomaly_scores():
    return ui_data.load_anomaly_scores()


@st.cache_data(show_spinner=False)
def _patients():
    return ui_data.load_patients()


@st.cache_data(show_spinner=False)
def _risk_metrics():
    return ui_data.load_risk_metrics()


def operations() -> Outcome:
    return _safe(_operations)


def anomaly_scores() -> Outcome:
    return _safe(_anomaly_scores)


def patients() -> Outcome:
    return _safe(_patients)


def risk_metrics() -> Outcome:
    return _safe(_risk_metrics)


@st.cache_data(show_spinner=False)
def _model_cards():
    return ui_data.load_model_cards()


def model_cards() -> Outcome:
    """Saved metrics (and baseline comparisons) for every trained model."""
    return _safe(_model_cards)


def assess_patient(record: dict) -> Outcome:
    """Score one synthetic patient record with the trained risk model.

    Returns ``Outcome(value={"level", "probabilities", "model"})``. Invalid input
    comes back as an ``Outcome`` whose error starts with ``PredictionServiceError``
    so the view can tell "check your inputs" apart from "model unavailable".
    """
    def run() -> dict:
        out = _service().predict_patient_risk(record).iloc[0]
        probs = {k: float(out[f"proba_{k}"]) for k in ("Low", "Medium", "High") if f"proba_{k}" in out.index}
        return {"level": str(out["predicted_risk_level"]), "probabilities": probs,
                "model": _service().patient_risk_model_name}

    return _safe(run)


def opd_model_info() -> Outcome:
    """(model name, interval note) of the deployed OPD forecaster."""
    return _safe(lambda: (_service().opd_model_name, _service().opd_interval_note))


# ----------------------------------------------------------------------
# Twin and predictions
# ----------------------------------------------------------------------
def current_state() -> Outcome:
    return _safe(lambda: _twin().get_current_state())


@st.cache_data(show_spinner=False)
def _current_model_wait() -> float:
    return _twin().model_waiting_time()


def current_model_wait() -> Outcome:
    """Waiting time the trained model gives for today's own inputs.

    The like-for-like baseline for predicted days (same model, same inputs
    builder). The observed value stays available on the current state.
    """
    return _safe(_current_model_wait)


@st.cache_data(show_spinner=False)
def _predicted_state(days_ahead: int) -> HospitalState:
    return _twin().get_predicted_state(days_ahead=days_ahead)


def predicted_state(days_ahead: int) -> Outcome:
    return _safe(lambda: _predicted_state(int(days_ahead)))


@st.cache_data(show_spinner=False)
def _forecast(days: int):
    return _service().forecast_opd(days=days)


def forecast(days: int = 7) -> Outcome:
    return _safe(lambda: _forecast(int(days)))


@st.cache_data(show_spinner=False)
def _operational_report(days_ahead: int | None):
    return generate_operational_report(
        _twin(), days_ahead=days_ahead, reference=_reference()
    )


def operational_report(days_ahead: int | None = 1) -> Outcome:
    """(AlertReport, [Recommendation]) from the Phase 6 integration.

    ``days_ahead=None`` evaluates the current state only.
    """
    return _safe(
        lambda: _operational_report(None if days_ahead is None else int(days_ahead))
    )


# ----------------------------------------------------------------------
# What-if
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _baseline_state(date_iso: str) -> HospitalState:
    df = _operations()
    rows = df[df["date"] == pd.Timestamp(date_iso)]
    if rows.empty:
        raise ValueError(f"No hospital data for {date_iso}.")
    return hospital_state_from_row(rows.iloc[0])


def baseline_state(date_iso: str) -> Outcome:
    return _safe(lambda: _baseline_state(date_iso))


@st.cache_data(show_spinner=False)
def _whatif(
    date_iso: str,
    inflow: float,
    emergency: float,
    total_beds: int,
    icu: int,
    doctors: int,
    nurses: int,
    medicine: float = 0.0,
):
    baseline = _baseline_state(date_iso)
    changes = ui_data.scenario_from_controls(
        baseline,
        ui_data.WhatIfControls(
            patient_inflow_percent=inflow,
            emergency_cases_percent=emergency,
            total_beds=total_beds,
            icu_capacity=icu,
            doctors_available=doctors,
            nurses_available=nurses,
            medicine_stock_percent=medicine,
        ),
    )
    # The existing Phase 5 simulator does all of the work.
    return WhatIfSimulator(prediction_service=_service()).simulate(baseline, changes)


@st.cache_data(show_spinner=False)
def _scenario_alerts(
    date_iso: str, inflow: float, emergency: float,
    total_beds: int, icu: int, doctors: int, nurses: int, medicine: float = 0.0,
):
    """Phase 6 engines applied to the Phase 5 simulated state (no new rules)."""
    result = _whatif(date_iso, inflow, emergency, total_beds, icu, doctors, nurses, medicine)
    simulated = result.simulated_state
    row, _why = anomaly_row_for_state(_service(), simulated)
    # The rising-demand check compares the scenario with its baseline day and is
    # labelled as a rule-derived scenario alert (not an ML forecast).
    report = AlertEngine(reference=_reference()).evaluate(
        simulated, scope=SCOPE_PREDICTED, anomaly=row,
        compare_to=result.baseline, compare_kind="scenario",
    )
    return report, generate_recommendations(report)


def scenario_alerts(date_iso: str, controls: ui_data.WhatIfControls) -> Outcome:
    """(AlertReport, [Recommendation]) for a what-if scenario's simulated state."""
    return _safe(
        lambda: _scenario_alerts(
            date_iso,
            float(controls.patient_inflow_percent),
            float(controls.emergency_cases_percent),
            int(controls.total_beds),
            int(controls.icu_capacity),
            int(controls.doctors_available),
            int(controls.nurses_available),
            float(controls.medicine_stock_percent),
        )
    )


def run_whatif(
    date_iso: str,
    controls: ui_data.WhatIfControls,
) -> Outcome:
    """Run the Phase 5 simulator for a baseline day and the UI's controls."""
    return _safe(
        lambda: _whatif(
            date_iso,
            float(controls.patient_inflow_percent),
            float(controls.emergency_cases_percent),
            int(controls.total_beds),
            int(controls.icu_capacity),
            int(controls.doctors_available),
            int(controls.nurses_available),
            float(controls.medicine_stock_percent),
        )
    )
