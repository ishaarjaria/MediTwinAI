"""Hospital Digital Twin: current versus predicted state, with provenance."""

from __future__ import annotations

import streamlit as st

from src.ui import backend, charts, components as ui, theme
from src.ui.data import (
    GLOSSARY,
    compare_states,
    fmt_num,
    format_date,
    state_staff_pressure,
    twin_change_summary,
)

_HORIZONS = [1, 2, 3, 5, 7]


def _stat_row(label: str, value: str, badge_key: str, delta: str = "", tip: str | None = None) -> str:
    tip_html = f" {ui.info(tip)}" if tip else ""
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:.6rem;'
        f'padding:.5rem 0;border-bottom:1px solid {theme.BORDER_SOFT}">'
        f'<span class="mt-note" style="font-size:.84rem">{ui.esc(label)}{tip_html}</span>'
        f'<span style="font-weight:600;font-variant-numeric:tabular-nums;display:inline-flex;gap:.55rem;align-items:center">'
        f'{delta}{ui.esc(value)} {ui.badge(badge_key)}</span></div>'
    )


def _occupancy_block(label: str, occupied: int, total: int, pct: float, badge_key: str, tip: str) -> str:
    return (
        '<div style="margin:.15rem 0 1rem">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<span class="mt-note" style="font-size:.84rem">{ui.esc(label)} {ui.info(tip)}</span>'
        f'<span style="font-weight:600;font-variant-numeric:tabular-nums">{occupied:,} / {total:,} '
        f'<span class="mt-note">({pct:.1f}%)</span> {ui.badge(badge_key)}</span></div>'
        f"{ui.occupancy_bar(pct, 10)}</div>"
    )


def _panel_html(state, predicted: bool, deltas: dict | None = None, model_wait: float | None = None) -> str:
    cap = "rule" if predicted else "observed"
    opd_key = "ml" if predicted else "observed"
    other = "rule" if predicted else "observed"
    pressure = state_staff_pressure(state)
    deltas = deltas or {}

    def d(label: str) -> str:
        r = deltas.get(label)
        if r is None or r["delta"] is None or round(abs(r["delta"]), r["decimals"]) == 0:
            return ""  # only show a change marker when something actually changed
        return ui.delta_html(r["delta"], r["tone"], r["decimals"], r["unit"])

    rows = "".join([
        _stat_row("Outpatients", fmt_num(state.current_opd), opd_key, d("OPD patients"), GLOSSARY["opd"]),
        _stat_row("Emergency cases", fmt_num(state.emergency_cases), other, d("Emergency cases")),
        _stat_row("Waiting time (recorded)" if not predicted else "Estimated waiting time",
                  f"{state.average_waiting_time:.1f} min",
                  "ml" if predicted else "observed", d("Average waiting time"), GLOSSARY["wait"]),
        *([_stat_row("Waiting time, model estimate", f"{model_wait:.1f} min", "ml")]
          if (not predicted and model_wait is not None) else []),
        _stat_row("Doctors · nurses", f"{state.doctors_available} · {state.nurses_available}", other),
        _stat_row("Medicine stock", f"{fmt_num(state.medicine_stock)} units", other, d("Medicine stock")),
        _stat_row("Staff workload", "—" if pressure is None else f"{pressure:.2f} per staff", "rule",
                  d("Staff pressure"), GLOSSARY["staff"]),
    ])
    return (
        _occupancy_block("Beds in use", state.occupied_beds, state.total_beds, state.bed_occupancy_percentage, cap, GLOSSARY["beds"])
        + _occupancy_block("ICU in use", state.occupied_icu_beds, state.total_icu_beds, state.icu_occupancy_percentage, cap, GLOSSARY["icu"])
        + rows
    )


def render() -> None:
    cur = backend.current_state()
    ui.md(ui.page_header(
        "Hospital twin",
        "Today beside the expected state, built from the dataset. Not connected to live sensors or patient records.",
        ui.date_chip(f"Latest data: {format_date(cur.value.as_of_date)}") if cur.ok else "",
        eyebrow="Plan",
    ))
    if not cur.ok:
        ui.md(ui.unavailable("Digital twin", cur.error))
        return
    state = cur.value

    ui.md(ui.section("Look ahead"))
    horizon = st.segmented_control(
        "Prediction horizon", _HORIZONS, default=1, key="twin_horizon",
        format_func=lambda d: f"{d} day" if d == 1 else f"{d} days", label_visibility="collapsed",
    ) or 1
    pred_out = backend.predicted_state(horizon)
    mw = backend.current_model_wait()
    model_wait = mw.value if mw.ok else None
    ui.md(ui.provenance_legend())
    with st.expander("What do Recorded, Expected and Estimated mean?"):
        for key, name in (("observed", "Recorded"), ("ml", "Expected"), ("rule", "Estimated")):
            ui.md(f'<div style="margin:.25rem 0">{ui.badge(key)} <span class="mt-note">{ui.esc(GLOSSARY[key])}</span></div>')

    st.write("")
    left, right = st.columns(2, gap="medium")
    with left, ui.card("twin-current"):
        ui.md(ui.section("Today", ui.status_pill(state.capacity_status)))
        ui.md(f'<div class="mt-note" style="margin:-.3rem 0 .9rem">{format_date(state.as_of_date)} · latest day in the dataset</div>')
        ui.md(_panel_html(state, predicted=False, model_wait=model_wait))
    pred = pred_out.value if pred_out.ok else None
    rows = compare_states(state, pred, model_wait) if pred is not None else []
    with right, ui.card("twin-predicted"):
        if pred is not None:
            ui.md(ui.section("Predicted", ui.status_pill(pred.capacity_status)))
            ui.md(f'<div class="mt-note" style="margin:-.3rem 0 .9rem">{format_date(pred.as_of_date)} · '
                  f"{horizon} day{'s' if horizon > 1 else ''} ahead · arrows show the change from today</div>")
            ui.md(_panel_html(pred, predicted=True, deltas={r["label"]: r for r in rows}))
        else:
            ui.md(ui.section("Predicted"))
            ui.md(ui.unavailable("Predicted state", pred_out.error))

    if pred is None:
        return

    st.write("")
    with ui.card("twin-summary"):
        ui.md(ui.section("What changes, in plain words"))
        lines = twin_change_summary(rows)
        if not lines:
            ui.md(ui.callout("No meaningful change is expected over this horizon."))
        for line in lines:
            ui.md(f'<div style="display:flex;gap:.6rem;align-items:center;margin:.3rem 0">'
                  f'{ui.badge(line["source"])}<span style="font-size:.9rem">{ui.esc(line["text"])}</span></div>')
        ui.md(ui.how_to_read(
            "Only the outpatient count and the waiting time come from trained models. The bed, ICU, emergency "
            "and medicine figures are a simple projection from the outpatient change, so small or zero "
            "differences there are expected."
        ))

    st.write("")
    with ui.card("twin-compare"):
        ui.md(ui.section("Full comparison"))
        body, last_group = [], None
        for r in rows:
            if r["group"] != last_group:
                body.append(["__group__", r["group"]])
                last_group = r["group"]
            body.append([
                ui.esc(r["label"]),
                f'{ui.value_text(r["current"], r["decimals"], r["unit"])} {ui.badge(r["prov_current"])}',
                f'{ui.value_text(r["predicted"], r["decimals"], r["unit"])} {ui.badge(r["prov_predicted"])}',
                ui.delta_html(r["delta"], r["tone"], r["decimals"], r["unit"]),
            ])
        ui.md(ui.table([("Indicator", False), ("Today", True), ("Predicted", True), ("Change", True)], body))

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("twin-occ"):
        ui.md(ui.section("Beds and ICU in use"))
        ui.plot(charts.paired_occupancy(
            ["Beds", "ICU"],
            [state.bed_occupancy_percentage, state.icu_occupancy_percentage],
            [pred.bed_occupancy_percentage, pred.icu_occupancy_percentage],
            "Today", "Predicted",
        ), key="twin_occ")
        ui.md(ui.how_to_read("Gridlines at 70%, 85% and 95% mark where the hospital's capacity status steps up."))
    with b, ui.card("twin-how"):
        ui.md(ui.section("What is expected and what is estimated"))
        ui.md('<div class="mt-note">Outpatient numbers and waiting time are <b>expected</b> values from trained forecasting '
              "models. Beds, ICU, emergency cases and medicine stock are simple <b>estimates</b> based on the expected "
              "change in outpatients, so they stay close to today's values. Staffing is carried forward from today.</div>")
        with st.expander("How this is calculated"):
            ui.md('<div class="mt-note">Outpatients come from the trained OPD forecasting model and waiting time from the '
                  "waiting-time model, both machine learning. Beds, ICU, emergency cases and medicine stock are a "
                  "transparent heuristic projection from the outpatient change; staffing is carried forward. Those "
                  "are rule-derived, not model predictions.</div>")
            for note in pred.notes:
                st.markdown(f"- {note}")
