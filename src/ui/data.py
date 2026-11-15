"""
Dashboard data preparation (Phase 7). Pure functions, no Streamlit imports.

Everything here either reads existing project artifacts (processed CSVs, model
metrics) or reshapes objects produced by the existing twin, simulator, alert
and recommendation modules for display. No business logic is re-implemented:
simulation, alerting and inference stay in their own packages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.alerts.models import Alert, AlertCategory, Severity
from src.data.preprocess import get_project_root
from src.recommendations.engine import Recommendation
from src.simulation.what_if import (
    SOURCE_ML,
    SOURCE_RULE,
    ScenarioChanges,
    WhatIfResult,
    medicine_consumption,
    staff_pressure,
)
from src.twin.hospital_twin import HospitalState

# Capacity-status cut-offs drawn on occupancy bars. A test asserts these equal
# the twin's own policy so the two cannot drift apart.
CAPACITY_MARKS = (70.0, 85.0, 95.0)

WINDOWS = ("Last 30 days", "Last 90 days", "Full period", "Custom")


# ----------------------------------------------------------------------
# Loading (read-only)
# ----------------------------------------------------------------------
def _root() -> Path:
    return get_project_root()


def load_operations(path: Path | None = None) -> pd.DataFrame:
    """Processed hospital-operations features, sorted by date."""
    path = Path(path) if path else _root() / "data" / "processed" / "hospital_operations_features.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Hospital operations data not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Hospital operations data is empty.")
    return df


def load_anomaly_scores(path: Path | None = None) -> pd.DataFrame | None:
    """Saved Isolation Forest scoring of the history, or None if absent."""
    path = Path(path) if path else _root() / "models" / "anomaly_detection" / "scored_operations.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_patients(path: Path | None = None) -> pd.DataFrame | None:
    path = Path(path) if path else _root() / "data" / "processed" / "patient_data_features.csv"
    return pd.read_csv(path) if path.is_file() else None


_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression",
}


def model_label(name: Any) -> str:
    """Display name for a saved model id (falls back to a tidied id)."""
    key = str(name or "").strip()
    return _MODEL_LABELS.get(key, key.replace("_", " ").title() or "—")


def load_risk_metrics(path: Path | None = None) -> dict[str, Any] | None:
    """Selected patient-risk model metrics from the saved training run."""
    path = Path(path) if path else _root() / "models" / "patient_risk" / "metrics.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf8") as fh:
        meta = json.load(fh)
    sel = meta.get("selected")
    if not sel:
        return None
    return {
        "model": sel.get("model"),
        "accuracy": sel.get("accuracy"),
        "f1_macro": sel.get("f1_macro"),
        "recall_high": sel.get("recall_high"),
        "n_test": meta.get("n_test"),
        "rationale": sel.get("selection_rationale"),
    }


# ----------------------------------------------------------------------
# Date windows and aggregates
# ----------------------------------------------------------------------
def window_dates(
    df: pd.DataFrame,
    label: str,
    custom: tuple[Any, Any] | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a window label to (start, end) clamped to the data's range."""
    lo, hi = df["date"].min(), df["date"].max()
    if label == "Last 30 days":
        start = hi - pd.Timedelta(days=29)
    elif label == "Last 90 days":
        start = hi - pd.Timedelta(days=89)
    elif label == "Custom" and custom and custom[0] and custom[1]:
        start, hi = pd.Timestamp(custom[0]), min(pd.Timestamp(custom[1]), hi)
    else:
        start = lo
    start = max(pd.Timestamp(start), lo)
    if start > hi:
        start = hi
    return start, hi


def filter_dates(df: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def summarize_window(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Headline aggregates for a filtered window (None when empty)."""
    if df.empty:
        return {k: None for k in ("days", "avg_opd", "peak_opd", "avg_bed", "avg_icu", "avg_wait")}
    return {
        "days": int(len(df)),
        "avg_opd": float(df["opd_patients"].mean()),
        "peak_opd": float(df["opd_patients"].max()),
        "avg_bed": float(df["bed_occupancy_percentage"].mean()),
        "avg_icu": float(df["icu_occupancy_percentage"].mean()),
        "avg_wait": float(df["average_waiting_time_minutes"].mean()),
    }


_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean OPD and emergency load by weekday (Mon..Sun)."""
    if df.empty:
        return pd.DataFrame(columns=["weekday", "opd_patients", "emergency_patients"])
    g = df.groupby(df["date"].dt.dayofweek)[["opd_patients", "emergency_patients"]].mean()
    g = g.reindex(range(7))
    g.insert(0, "weekday", _WEEKDAYS)
    return g.reset_index(drop=True)


def monthly_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "opd_patients"])
    g = df.groupby(df["date"].dt.to_period("M"))["opd_patients"].mean().reset_index()
    g["month"] = g["date"].dt.to_timestamp()
    return g[["month", "opd_patients"]]


def to_weekly(df: pd.DataFrame, columns: Iterable[str], max_daily_points: int = 120) -> tuple[pd.DataFrame, str]:
    """Daily data, or weekly means when the window is long. Returns (frame, grain)."""
    cols = list(columns)
    if len(df) <= max_daily_points:
        return df[["date", *cols]].copy(), "daily"
    weekly = df.set_index("date")[cols].resample("W").mean().reset_index()
    return weekly, "weekly mean"


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def fmt_num(value: float | int | None, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:,.{decimals}f}"


def fmt_signed(value: float | None, decimals: int = 0, unit: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if round(abs(value), decimals) == 0:
        return f"0{unit}"
    sign = "+" if value > 0 else "−"
    return f"{sign}{abs(value):,.{decimals}f}{unit}"


def delta_tone(delta: float | None, worse_when_higher: bool | None, decimals: int = 2) -> str:
    """'good' / 'bad' / 'neutral' for a change, given which direction is worse."""
    if delta is None or not math.isfinite(delta) or round(abs(delta), decimals) == 0:
        return "neutral"
    if worse_when_higher is None:
        return "neutral"
    return "bad" if (delta > 0) == worse_when_higher else "good"


def delta_arrow(delta: float | None, decimals: int = 2) -> str:
    if delta is None or not math.isfinite(delta) or round(abs(delta), decimals) == 0:
        return "•"
    return "▲" if delta > 0 else "▼"


def format_date(value: Any, with_year: bool = True) -> str:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "—"
    return ts.strftime("%d %b %Y" if with_year else "%d %b")


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------
def provenance_from_source(source: str) -> str:
    """Map the simulator's source label to a provenance key."""
    if source == SOURCE_ML:
        return "ml"
    if source == SOURCE_RULE:
        return "rule"
    return "unavailable"


def provenance_from_basis(basis: str) -> str:
    """Map a Phase 6 alert ``basis`` to a provenance key."""
    return {"observed": "observed", "ml_prediction": "ml", "rule_derived": "rule"}.get(
        basis, "unavailable"
    )


def state_staff_pressure(state: HospitalState) -> float | None:
    """Patients per clinical staff member, using the Phase 5 rule."""
    load = state.current_opd + state.emergency_cases + state.admissions
    value = staff_pressure(load, state.doctors_available, state.nurses_available)
    return value if math.isfinite(value) else None


def estimated_medicine_consumption(state: HospitalState) -> float:
    """Rule-derived daily medicine use (Phase 5 formula)."""
    return medicine_consumption(state.current_opd, state.emergency_cases, state.admissions)


# ----------------------------------------------------------------------
# Digital twin: current vs predicted
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Field:
    label: str
    attr: str
    unit: str
    decimals: int
    worse_when_higher: bool | None
    group: str
    prov_current: str
    prov_predicted: str


_FIELDS = (
    _Field("Bed occupancy", "bed_occupancy_percentage", "%", 1, True, "Capacity", "observed", "rule"),
    _Field("Occupied beds", "occupied_beds", "", 0, True, "Capacity", "observed", "rule"),
    _Field("Available beds", "available_beds", "", 0, False, "Capacity", "observed", "rule"),
    _Field("ICU occupancy", "icu_occupancy_percentage", "%", 1, True, "Capacity", "observed", "rule"),
    _Field("Occupied ICU beds", "occupied_icu_beds", "", 0, True, "Capacity", "observed", "rule"),
    _Field("OPD patients", "current_opd", "", 0, True, "Demand", "observed", "ml"),
    _Field("Emergency cases", "emergency_cases", "", 0, True, "Demand", "observed", "rule"),
    _Field("Average waiting time", "average_waiting_time", " min", 1, True, "Demand", "observed", "ml"),
    _Field("Doctors available", "doctors_available", "", 0, False, "Resources", "observed", "rule"),
    _Field("Nurses available", "nurses_available", "", 0, False, "Resources", "observed", "rule"),
    _Field("Medicine stock", "medicine_stock", " units", 0, False, "Resources", "observed", "rule"),
)


def compare_states(
    current: HospitalState,
    predicted: HospitalState,
    current_model_wait: float | None = None,
) -> list[dict[str, Any]]:
    """Rows comparing the twin's current and predicted state, with provenance.

    Waiting time is compared like for like: when ``current_model_wait`` (the
    trained model's estimate for today's own inputs) is given it is used as
    today's value, so both sides come from the same model. Without it the
    observed value would be compared with a model estimate, so that row is
    left without a change figure.
    """
    rows: list[dict[str, Any]] = []
    for f in _FIELDS:
        cur = float(getattr(current, f.attr))
        pred = float(getattr(predicted, f.attr))
        prov_cur = f.prov_current
        delta: float | None = pred - cur
        if f.attr == "average_waiting_time":
            if current_model_wait is not None:
                cur, prov_cur = float(current_model_wait), "ml"
                delta = pred - cur
            else:
                delta = None  # observed vs model estimate is not a fair change
        rows.append(
            {
                "label": f.label, "unit": f.unit, "decimals": f.decimals, "group": f.group,
                "current": cur, "predicted": pred, "delta": delta,
                "tone": delta_tone(delta, f.worse_when_higher, f.decimals),
                "prov_current": prov_cur, "prov_predicted": f.prov_predicted,
            }
        )
    cur_p, pred_p = state_staff_pressure(current), state_staff_pressure(predicted)
    delta_p = None if cur_p is None or pred_p is None else pred_p - cur_p
    rows.append(
        {
            "label": "Staff pressure", "unit": " pts/staff", "decimals": 2, "group": "Resources",
            "current": cur_p, "predicted": pred_p, "delta": delta_p,
            "tone": delta_tone(delta_p, True, 2),
            "prov_current": "rule", "prov_predicted": "rule",
        }
    )
    return rows


# ----------------------------------------------------------------------
# What-if
# ----------------------------------------------------------------------
# name -> (label, unit, decimals, worse_when_higher, group)
WHATIF_META: dict[str, tuple[str, str, int, bool | None, str]] = {
    "patient_load": ("Patient load", " patients", 0, None, "Demand"),
    "bed_requirement": ("Beds needed", " beds", 0, True, "Beds"),
    "bed_occupancy_percentage": ("Beds in use", "%", 1, True, "Beds"),
    "icu_pressure_percentage": ("ICU in use", "%", 1, True, "ICU"),
    "expected_waiting_time_minutes": ("Estimated waiting time", " min", 1, True, "Waiting"),
    "medicine_consumption_units": ("Medicine used per day", " units", 0, None, "Supplies"),
    "medicine_stock_units": ("Medicine stock at day end", " units", 0, False, "Supplies"),
    "medicine_days_of_cover": ("Days of medicine cover", " days", 1, False, "Supplies"),
    "staff_pressure_patients_per_staff": ("Staff workload", " per staff", 2, True, "Staffing"),
}

WHATIF_EXPLAIN = {
    "patient_load": "OPD + emergency + admissions for the day.",
    "bed_requirement": "Beds needed if demand scales with admissions volume (before capacity cap).",
    "bed_occupancy_percentage": "Occupied beds as a share of total beds, capped at 100%.",
    "icu_pressure_percentage": "ICU occupancy after emergency and bed-demand changes, capped at capacity.",
    "expected_waiting_time_minutes": "Average OPD waiting time from the trained waiting-time model, for both sides.",
    "medicine_consumption_units": "Daily medicine use estimated from OPD, emergency and admissions.",
    "medicine_stock_units": "Stock on hand after the day's estimated use (starting stock changed by the medicine control).",
    "medicine_days_of_cover": "How many days the end-of-day stock would last at that day's estimated use. Alerts start below 21 days.",
    "staff_pressure_patients_per_staff": "Patient load per doctor or nurse; higher means more pressure.",
}


def whatif_rows(result: WhatIfResult) -> list[dict[str, Any]]:
    """Display rows from the simulator's own comparisons. No recalculation."""
    rows = []
    for c in result.comparisons:
        label, unit, decimals, worse, group = WHATIF_META.get(
            c.name, (c.name.replace("_", " ").title(), "", 2, None, "Other")
        )
        # The simulator encodes "no clinical staff" as -1; show it as unbounded.
        unbounded = c.name == "staff_pressure_patients_per_staff" and (c.current < 0 or c.simulated < 0)
        delta = None if unbounded else c.simulated - c.current
        rows.append(
            {
                "name": c.name, "label": label, "unit": unit, "decimals": decimals, "group": group,
                "baseline": None if c.current < 0 and unbounded else c.current,
                "simulated": None if c.simulated < 0 and unbounded else c.simulated,
                "unbounded": unbounded,
                "delta": delta, "tone": delta_tone(delta, worse, decimals),
                "source": provenance_from_source(c.source), "source_text": c.source,
                "explanation": WHATIF_EXPLAIN.get(c.name, c.explanation),
            }
        )
    return rows


@dataclass(frozen=True)
class WhatIfControls:
    """Values from the UI controls (absolute capacities, percent changes)."""

    patient_inflow_percent: float = 0.0
    emergency_cases_percent: float = 0.0
    total_beds: int | None = None
    icu_capacity: int | None = None
    doctors_available: int | None = None
    nurses_available: int | None = None
    medicine_stock_percent: float = 0.0


def scenario_from_controls(baseline: HospitalState, c: WhatIfControls) -> ScenarioChanges:
    """
    Build the simulator's ScenarioChanges. A capacity control left at the
    baseline value is passed as None, so an untouched UI is a true no-change run.
    """
    def changed(value: int | None, base: int) -> int | None:
        return None if value is None or int(value) == int(base) else int(value)

    return ScenarioChanges(
        patient_inflow_percent=float(c.patient_inflow_percent),
        emergency_cases_percent=float(c.emergency_cases_percent),
        total_beds=changed(c.total_beds, baseline.total_beds),
        icu_capacity=changed(c.icu_capacity, baseline.total_icu_beds),
        doctors_available=changed(c.doctors_available, baseline.doctors_available),
        nurses_available=changed(c.nurses_available, baseline.nurses_available),
        medicine_stock_percent=float(c.medicine_stock_percent),
    )


# Upper bounds the prediction service accepts (mirrors its input validation),
# so a slider can never offer a value the model would reject.
_MAX_BEDS, _MAX_ICU, _MAX_DOCTORS, _MAX_NURSES = 5000, 1000, 500, 2000


def control_ranges(b: HospitalState) -> dict[str, tuple[int, int, int]]:
    """(min, max, default) for each capacity slider, derived from the baseline."""
    def rng(base: int, lo: int, hi_cap: int, mult: int, floor_max: int) -> tuple[int, int, int]:
        hi = min(hi_cap, max(base * mult, floor_max, base))
        return (min(lo, base), hi, base)

    return {
        "total_beds": rng(b.total_beds, max(1, b.total_beds // 2), _MAX_BEDS, 2, 10),
        "icu_capacity": rng(b.total_icu_beds, 1, _MAX_ICU, 2, 5),
        "doctors_available": rng(b.doctors_available, 0, _MAX_DOCTORS, 2, 10),
        "nurses_available": rng(b.nurses_available, 0, _MAX_NURSES, 2, 10),
    }


# Quick scenarios: name -> overrides applied on top of an all-baseline reset.
PRESETS: dict[str, dict[str, float | int]] = {
    "Inflow +20%": {"patient_inflow_percent": 20},
    "Delivery delay (stock −60%)": {"medicine_stock_percent": -60},
    "Emergency surge +50%": {"emergency_cases_percent": 50},
    "ICU −10 beds": {"icu_delta": -10},
    "+10 doctors": {"doctors_delta": 10},
    "+50 beds": {"beds_delta": 50},
}


def apply_preset(b: HospitalState, name: str | None) -> WhatIfControls:
    """Controls for a named quick scenario (None or unknown = baseline)."""
    p = PRESETS.get(name or "", {})
    ranges = control_ranges(b)

    def clamp(key: str, value: int) -> int:
        lo, hi, _ = ranges[key]
        return max(lo, min(hi, value))

    return WhatIfControls(
        patient_inflow_percent=float(p.get("patient_inflow_percent", 0)),
        emergency_cases_percent=float(p.get("emergency_cases_percent", 0)),
        total_beds=clamp("total_beds", b.total_beds + int(p.get("beds_delta", 0))),
        icu_capacity=clamp("icu_capacity", b.total_icu_beds + int(p.get("icu_delta", 0))),
        doctors_available=clamp("doctors_available", b.doctors_available + int(p.get("doctors_delta", 0))),
        nurses_available=b.nurses_available,
        medicine_stock_percent=float(p.get("medicine_stock_percent", 0)),
    )


def is_no_change(changes: ScenarioChanges) -> bool:
    return (
        not changes.patient_inflow_percent
        and not changes.emergency_cases_percent
        and changes.total_beds is None
        and changes.icu_capacity is None
        and changes.doctors_available is None
        and changes.nurses_available is None
        and not changes.medicine_stock_percent
    )


# ----------------------------------------------------------------------
# Plain-language layer (text only; no calculation of results)
# ----------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "beds": "Share of the hospital's beds that currently have a patient. The tick marks on the bar show 70%, 85% and 95%, where capacity gets tight.",
    "icu": "How much of the intensive-care capacity is being used right now.",
    "opd": "Outpatients are patients who visit for consultations without staying overnight (OPD). The forecast is a statistical estimate learned from past daily counts.",
    "wait": "Estimated average time patients wait. It comes from a model trained on past days using patient load, staffing and occupancy.",
    "staff": "Approximate patient load per doctor or nurse on duty. Higher means each staff member is covering more patients.",
    "status": "A simple traffic-light summary of how close beds, ICU and waiting time are to their limits. It is a rule, not a prediction.",
    "observed": "Recorded in the hospital dataset for that day.",
    "ml": "Expected: predicted by a trained model. It carries uncertainty and can be wrong.",
    "rule": "Estimated: worked out with a fixed formula or simple assumption, not a trained model.",
    "anomaly": "Unusual activity compared with historical patterns. A flag means 'different from normal', not 'harmful'.",
    "forecast_band": "The shaded band is the likely range, built from the forecast's past errors. It is a rough guide, not a guarantee; the Forecasts page shows how often it held on unseen days.",
}

CATEGORY_HELP: dict[AlertCategory, str] = {
    AlertCategory.BED_OCCUPANCY: "How full the hospital's beds are.",
    AlertCategory.ICU_PRESSURE: "How much ICU capacity is being used.",
    AlertCategory.OPD_INFLOW: "Outpatient numbers far above the usual level.",
    AlertCategory.OPD_RISING: "Outpatient demand expected to jump compared with today.",
    AlertCategory.MEDICINE_STOCK: "Days of medicine supply left at the current usage rate.",
    AlertCategory.MEDICINE_CONSUMPTION: "Medicine use far above the usual level.",
    AlertCategory.STAFF_WORKLOAD: "Patient load per staff member well above usual.",
    AlertCategory.ANOMALY: "Activity unlike the hospital's normal patterns.",
}

_STATUS_HEADLINE = {
    "Normal": "Operating within normal limits",
    "Moderate": "Capacity is getting tighter",
    "High": "Under high pressure",
    "Critical": "At critical capacity",
}


def status_headline(status: str) -> str:
    """Plain-language one-liner for a capacity status."""
    return _STATUS_HEADLINE.get(status, "Status unavailable")


def status_summary(state: HospitalState, n_alerts: int | None, highest: str | None = None) -> str:
    """One plain sentence about beds, ICU and alerts, from real values only."""
    beds = (
        f"{state.occupied_beds:,} of {state.total_beds:,} beds are in use "
        f"({state.bed_occupancy_percentage:.0f}%) and {state.occupied_icu_beds:,} of "
        f"{state.total_icu_beds:,} ICU beds ({state.icu_occupancy_percentage:.0f}%)."
    )
    if n_alerts is None:
        return beds + " Alert checks are unavailable."
    if n_alerts == 0:
        return beds + " No alerts need attention."
    tail = f" Highest severity: {highest}." if highest else ""
    return beds + f" {n_alerts} alert{'s' if n_alerts != 1 else ''} need attention.{tail}"


def trailing_context(df: pd.DataFrame, column: str, days: int = 30) -> float | None:
    """Mean of ``column`` over the ``days`` rows before the latest one (None if too short)."""
    if column not in df.columns or len(df) < days + 1:
        return None
    value = df[column].iloc[-(days + 1):-1].mean()
    return float(value) if math.isfinite(value) else None


_PLAIN_ROWS = {
    "OPD patients": "Expected outpatients",
    "Average waiting time": "Estimated waiting time",
    "Bed occupancy": "Beds in use",
    "ICU occupancy": "ICU in use",
    "Emergency cases": "Emergency cases",
    "Medicine stock": "Medicine stock",
    "Staff pressure": "Patients per staff member",
}


def twin_change_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Plain sentences for what differs between the current and predicted state."""
    out: list[dict[str, str]] = []
    for r in rows:
        name = _PLAIN_ROWS.get(r["label"])
        delta = r["delta"]
        if name is None or delta is None or r["current"] is None or r["predicted"] is None:
            continue
        if round(abs(delta), r["decimals"]) == 0:
            continue
        unit = r["unit"]
        out.append({
            "text": (
                f"{name}: {fmt_num(r['current'], r['decimals'])}{unit} → "
                f"{fmt_num(r['predicted'], r['decimals'])}{unit} ({fmt_signed(delta, r['decimals'], unit)})"
            ),
            "tone": r["tone"],
            "source": r["prov_predicted"],
        })
    return out


def meter_geometry(value: float, alert_from: float, direction: str = "≥") -> tuple[float, float]:
    """(fill %, alert-mark %) for a headroom meter. Scale leaves room beyond the alert level."""
    if direction == "<":
        scale = max(value * 1.05, alert_from * 1.25, 1e-9)
    else:
        scale = max(alert_from * 1.25, value * 1.05, 1e-9)
    return (max(0.0, min(100.0, 100.0 * value / scale)), max(0.0, min(100.0, 100.0 * alert_from / scale)))


def control_caption(baseline: float, value: float, unit: str = "", percent: bool = False) -> str:
    """Caption under a what-if slider: the control's baseline against its scenario value."""
    if percent:
        return "Unchanged from baseline" if int(value) == 0 else f"{int(value):+d}% versus baseline".replace("-", "−")
    if int(value) == int(baseline):
        return f"Same as baseline ({int(baseline):,}{unit})"
    return f"Baseline {int(baseline):,} → scenario {int(value):,} ({fmt_signed(value - baseline, 0, unit)})"


# ----------------------------------------------------------------------
# Alerts and recommendations (display helpers over Phase 6 objects)
# ----------------------------------------------------------------------
CATEGORY_LABELS = {
    AlertCategory.BED_OCCUPANCY: "Bed occupancy",
    AlertCategory.ICU_PRESSURE: "ICU pressure",
    AlertCategory.OPD_INFLOW: "OPD load",
    AlertCategory.OPD_RISING: "Rising demand",
    AlertCategory.MEDICINE_STOCK: "Medicine stock",
    AlertCategory.MEDICINE_CONSUMPTION: "Medicine use",
    AlertCategory.STAFF_WORKLOAD: "Staff workload",
    AlertCategory.ANOMALY: "Unusual activity",
}


def severity_counts(alerts: Iterable[Alert]) -> dict[str, int]:
    counts = {s.label: 0 for s in reversed(list(Severity))}
    for a in alerts:
        counts[a.severity.label] += 1
    return counts


def filter_alerts(
    alerts: Iterable[Alert],
    severities: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    scope: str | None = None,
) -> list[Alert]:
    """Filter by severity label, category label and scope ('current'/'predicted')."""
    sev = set(severities) if severities else None
    cat = set(categories) if categories else None
    out = []
    for a in alerts:
        if sev is not None and a.severity.label not in sev:
            continue
        if cat is not None and CATEGORY_LABELS[a.category] not in cat:
            continue
        if scope in ("current", "predicted") and a.scope != scope:
            continue
        out.append(a)
    return out


def recs_for_alert(recs: Iterable[Recommendation], alert_id: str) -> list[Recommendation]:
    return [r for r in recs if alert_id in r.related_alert_ids]


def monitoring_rows(state: HospitalState, thresholds: Any, reference: Any | None) -> list[dict[str, Any]]:
    """
    Headroom table for the 'all clear' state: today's value next to the level
    at which the Phase 6 engine would start alerting. Display only; severity
    is never decided here.
    """
    rows = [
        {"label": "Bed occupancy", "value": state.bed_occupancy_percentage, "unit": "%",
         "alert_from": thresholds.bed_occupancy[0], "direction": "≥"},
        {"label": "ICU occupancy", "value": state.icu_occupancy_percentage, "unit": "%",
         "alert_from": thresholds.icu_occupancy[0], "direction": "≥"},
    ]
    if reference is not None and reference.opd_mean is not None and reference.opd_std:
        rows.append(
            {"label": "OPD load", "value": float(state.current_opd), "unit": "",
             "alert_from": reference.opd_mean + thresholds.opd_zscore[0] * reference.opd_std,
             "direction": "≥"}
        )
    consumption = estimated_medicine_consumption(state)
    if consumption > 0:
        rows.append(
            {"label": "Medicine cover", "value": state.medicine_stock / consumption, "unit": " days",
             "alert_from": thresholds.medicine_days_cover[0], "direction": "<"}
        )
    return rows


# ----------------------------------------------------------------------
# Patient risk form (synthetic educational tool)
# ----------------------------------------------------------------------
# key, label, minimum, maximum, default, step, help
PATIENT_INPUTS: list[tuple[str, str, float, float, float, float, str]] = [
    ("age", "Age (years)", 0, 120, 45, 1, "Whole years."),
    ("heart_rate", "Heart rate (beats/min)", 30, 220, 78, 1, "Resting pulse."),
    ("systolic_bp", "Systolic blood pressure (mmHg)", 60, 250, 125, 1, "The top number."),
    ("diastolic_bp", "Diastolic blood pressure (mmHg)", 30, 150, 80, 1, "The bottom number. Must be lower than systolic."),
    ("temperature", "Body temperature (°C)", 30.0, 43.0, 36.8, 0.1, "Normal is about 36.5 to 37.5."),
    ("oxygen_saturation", "Oxygen saturation (%)", 50, 100, 97, 1, "Normal is 95 to 100."),
    ("previous_admissions", "Previous hospital admissions", 0, 50, 0, 1, "Number of earlier admissions."),
]
PATIENT_GENDERS = ["Female", "Male", "Other"]

PATIENT_PRESETS: dict[str, dict[str, Any]] = {
    "Typical, healthy": dict(age=30, gender="Female", heart_rate=72, systolic_bp=118, diastolic_bp=76,
                             temperature=36.7, oxygen_saturation=98, previous_admissions=0,
                             chronic_disease="No", emergency_case="No"),
    "In-between": dict(age=55, gender="Male", heart_rate=104, systolic_bp=142, diastolic_bp=88,
                       temperature=38.1, oxygen_saturation=95, previous_admissions=1,
                       chronic_disease="No", emergency_case="No"),
    "Several warning signs": dict(age=78, gender="Male", heart_rate=118, systolic_bp=165, diastolic_bp=102,
                                  temperature=39.2, oxygen_saturation=88, previous_admissions=4,
                                  chronic_disease="Yes", emergency_case="Yes"),
}


def patient_record(values: dict[str, Any]) -> dict[str, Any]:
    """Form values -> the record the prediction service validates and scores."""
    rec = dict(values)
    for key in ("age", "heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation", "previous_admissions"):
        if key in rec and rec[key] is not None:
            rec[key] = int(rec[key])
    if rec.get("systolic_bp") is not None and rec.get("diastolic_bp") is not None:
        rec["pulse_pressure"] = rec["systolic_bp"] - rec["diastolic_bp"]
    rec["chronic_disease_flag"] = int(str(rec.get("chronic_disease", "No")) == "Yes")
    rec["emergency_case_flag"] = int(str(rec.get("emergency_case", "No")) == "Yes")
    return rec


def risk_factors(values: dict[str, Any]) -> list[tuple[str, int]]:
    """Itemised points from the synthetic scoring rule behind the training labels."""
    import pandas as pd

    from src.data.generate_synthetic_data import risk_score_breakdown

    row = pd.Series({
        **values,
        "chronic_disease": str(values.get("chronic_disease", "No")),
        "emergency_case": str(values.get("emergency_case", "No")),
    })
    return risk_score_breakdown(row)


RISK_PLAIN = {
    "Low": "few of the warning signs the synthetic scoring rules look for are present.",
    "Medium": "some warning signs are present, so this record sits between the low and high groups.",
    "High": "several warning signs are present together, which the synthetic scoring rules treat as high risk.",
}


def risk_explanation(level: str, probabilities: dict[str, float]) -> str:
    """One plain-language sentence for a predicted risk level and its confidence."""
    top = probabilities.get(level, 0.0)
    if top >= 0.8:
        confidence = "The model is fairly sure"
    elif top >= 0.55:
        confidence = "The model leans this way but is not certain"
    else:
        confidence = "The model is unsure"
    return f"Predicted {level.lower()} risk: {RISK_PLAIN.get(level, '')} {confidence} ({top:.0%} for {level.lower()})."


# ----------------------------------------------------------------------
# Model cards (saved metrics) for the Predictions page
# ----------------------------------------------------------------------
def load_model_cards(models_root: Path | None = None) -> dict[str, Any]:
    """Saved training metrics for every model; missing files give ``None``."""
    root = Path(models_root) if models_root else _root() / "models"
    cards: dict[str, Any] = {}
    for key, rel in (
        ("opd_forecast", "opd_forecast/metrics.json"),
        ("waiting_time", "waiting_time/metrics.json"),
        ("patient_risk", "patient_risk/metrics.json"),
        ("anomaly", "anomaly_detection/metrics.json"),
    ):
        path = root / rel
        cards[key] = json.loads(path.read_text(encoding="utf8")) if path.is_file() else None
    return cards


def forecast_comparison_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Table rows (model vs baselines) for the OPD forecast, selected model flagged."""
    rows = []
    for c in meta.get("comparison", []):
        base = {"model": c["model"], "type": c.get("type", "model"),
                "selected": c["model"] == meta.get("selected_model")}
        if "test" in c:
            rows.append({**base, "val_mae": c["validation"]["mae"], "test_mae": c["test"]["mae"],
                         "test_rmse": c["test"]["rmse"], "test_r2": c["test"]["r2"], "note": ""})
        else:
            rows.append({**base, "val_mae": None, "test_mae": None, "test_rmse": None, "test_r2": None,
                         "note": "Not run on this machine (the Windows security policy blocked its solver). Not compared."
                         if "4551" in str(c.get("skipped_reason")) or "Application Control" in str(c.get("skipped_reason"))
                         else "Not evaluated: " + str(c.get("skipped_reason") or "unavailable")[:90]})
    return rows


_BASELINE_LABELS = {
    "training_mean": "Training mean",
    "opd_only_linear": "Straight line on OPD only",
    "day_of_week_mean": "Day-of-week mean",
}


def waiting_comparison_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Waiting-time models and simple baselines on the held-out test block (readable names)."""
    selected = (meta.get("selected") or {}).get("model")
    rows = []
    for m in meta.get("all_models", []):
        rows.append({"model": model_label(m["model"]), "type": "model", "selected": m["model"] == selected,
                     "test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"]})
    for name, m in (meta.get("baselines") or {}).items():
        rows.append({"model": _BASELINE_LABELS.get(name, name.replace("_", " ").capitalize()), "type": "baseline",
                     "selected": False, "test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"]})
    return rows


def resource_projection_rows(states: list[HospitalState]) -> list[dict[str, Any]]:
    """Per-day projected resource values for the Predictions page (rule-derived)."""
    return [{
        "date": st_.as_of_date, "beds": st_.bed_occupancy_percentage, "icu": st_.icu_occupancy_percentage,
        "emergency": st_.emergency_cases, "medicine": st_.medicine_stock,
        "staff_pressure": state_staff_pressure(st_),
    } for st_ in states]


# Plain-language definitions for the sidebar guide (administrator-friendly wording).
GLOSSARY_TERMS: list[tuple[str, str]] = [
    ("OPD (outpatients)", "Patients who visit for a consultation and go home the same day, without a hospital bed."),
    ("ICU pressure", "How full the intensive-care unit is. Above about 70% the unit has little room for sudden emergencies."),
    ("Patient risk", "A Low / Medium / High group a model assigns to a synthetic patient record. It is a demonstration, not a diagnosis."),
    ("Waiting-time prediction", "The average outpatient wait a model expects, based on patient load, emergencies, staffing and bed use."),
    ("Unusual activity", "A day whose pattern differs from normal activity. It means \"unusual\", not necessarily \"bad\"."),
    ("Forecast interval", "The shaded band around a forecast. It is built from past forecast errors; on held-out days it held less often than its nominal 80% (see Forecasts), so treat it as a rough guide."),
    ("Recorded", "A value stored in the synthetic dataset."),
    ("Expected", "A value the system expects, estimated by a trained model, so it carries uncertainty."),
    ("Estimated", "A value worked out with a simple formula or assumption, not learned from data."),
]


def spark_series(df: pd.DataFrame, key: str, n: int = 30) -> list[float]:
    """Last ``n`` daily values of a dataset column, or of the derived ``staff_pressure``."""
    tail = df.tail(n)
    if key == "staff_pressure":
        return [
            staff_pressure(r.opd_patients + r.emergency_patients + r.admissions, r.doctors_available, r.nurses_available)
            for r in tail.itertuples()
        ]
    return [] if key not in tail.columns else [float(v) for v in tail[key]]
