"""Alerts & Recommendations: the Phase 6 engines, presented for administrators."""

from __future__ import annotations

import streamlit as st

from src.alerts.models import AlertThresholds
from src.recommendations.engine import DISCLAIMER
from src.ui import backend, components as ui
from src.ui.views import go_to
from src.ui.data import (
    CATEGORY_HELP,
    scenario_from_controls,
    CATEGORY_LABELS,
    GLOSSARY,
    filter_alerts,
    format_date,
    meter_geometry,
    monitoring_rows,
    provenance_from_basis,
    recs_for_alert,
    severity_counts,
)

_SEVERITIES = ["Critical", "High", "Medium", "Low"]
_HORIZONS = {"Current only": None, "Current + 1 day": 1, "Current + 3 days": 3, "Current + 7 days": 7}


def render() -> None:
    cur = backend.current_state()
    ui.md(ui.page_header(
        "Alerts",
        "What needs attention, why, and what to consider doing.",
        ui.date_chip(f"As of {format_date(cur.value.as_of_date)}") if cur.ok else "",
        eyebrow="Today",
    ))
    ui.md(ui.callout(
        f"{DISCLAIMER} Alerts describe hospital operations only, never individual patients. Data is synthetic."
    ))
    st.write("")

    ui.md(ui.section("What to check"))
    horizon_label = st.segmented_control(
        "Evaluate", list(_HORIZONS), default="Current + 1 day", key="al_horizon", label_visibility="collapsed",
    ) or "Current + 1 day"
    out = backend.operational_report(_HORIZONS[horizon_label])
    if not out.ok:
        ui.md(ui.unavailable("Alerts", out.error))
        return
    report, recs = out.value

    counts = severity_counts(report.alerts)
    cols = st.columns(4, gap="small")
    for col, label in zip(cols, _SEVERITIES):
        with col:
            ui.md(ui.severity_tile(label, counts[label]))
    ui.md(f'<div class="mt-note" style="margin:.7rem 0 .9rem">{len(report.alerts)} active · '
          f"{len(recs)} recommendation(s)</div>")

    tab_alerts, tab_recs = st.tabs(["Alerts", "Recommendations"])
    with tab_alerts:
        _alerts_tab(report, recs)
    with tab_recs:
        _recs_tab(report, recs)

    _scenario_section()

    if report.unavailable:
        with st.expander(f"Checks that could not run ({len(report.unavailable)})"):
            for item in report.unavailable:
                st.markdown(f"- {item}")


def _clear_scenario() -> None:
    st.session_state.pop("wi_last_scenario", None)


def _scenario_section() -> None:
    """Alerts for the scenario last set in the What-If simulator, kept apart from real-state alerts."""
    last = st.session_state.get("wi_last_scenario")
    st.write("")
    ui.md(ui.section("What-if scenario alerts", ui.badge("rule")))
    if not last:
        ui.md('<div class="mt-note">No scenario is active. Change a condition in the What-If Simulator and its '
              "alerts will also appear here, labelled as hypothetical.</div>")
        return
    date_iso, controls = last
    base = backend.baseline_state(date_iso)
    out = backend.scenario_alerts(date_iso, controls)
    with ui.card("al-scenario"):
        ui.md(ui.callout(
            "These alerts describe a hypothetical scenario you set in the What-If Simulator. They are not about the "
            "real hospital today and are not counted above.", "warn"))
        if base.ok:
            changes = scenario_from_controls(base.value, controls).describe()
            ui.md(f'<div class="mt-note" style="margin:.6rem 0">Scenario from {ui.esc(format_date(date_iso))}: '
                  + " · ".join(ui.esc(c) for c in changes) + "</div>")
        if not out.ok:
            ui.md(ui.unavailable("Scenario alerts", out.error))
            return
        s_report, s_recs = out.value
        if not s_report.alerts:
            ui.md(ui.all_clear("This scenario would not raise any alerts", "All monitored indicators stay below their alert levels."))
        for a in s_report.alerts:
            ui.md(ui.alert_card(
                a.severity.label, a.title, CATEGORY_LABELS[a.category], a.scope, "", None,
                f"{a.metric.replace('_', ' ')} {a.observed_value:g}", scope_label="Scenario",
            ))
            for rec in recs_for_alert(s_recs, a.alert_id):
                ui.md(f'<div class="mt-note" style="margin:0 0 .7rem .5rem">{ui.badge("rule", compact=True)} {ui.esc(rec.action)}</div>')
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.button("Edit in What-If Simulator", key="al_edit_scenario", on_click=go_to, args=("whatif",), width="stretch")
        with c2:
            st.button("Clear scenario", key="al_clear_scenario", on_click=_clear_scenario, width="stretch")


def _alerts_tab(report, recs) -> None:
    if not report.alerts:
        _all_clear()
        return
    present = sorted({CATEGORY_LABELS[a.category] for a in report.alerts})
    st.write("")
    f1, f2, f3 = st.columns([3, 3, 2], gap="medium")
    with f1:
        sev = st.pills("Severity", _SEVERITIES, selection_mode="multi", key="al_sev")
    with f2:
        cats = st.multiselect("Category", present, key="al_cat", placeholder="All categories")
    with f3:
        scope_label = st.segmented_control("Timing", ["All", "Current", "Predicted"], default="All", key="al_scope") or "All"
    scope = None if scope_label == "All" else scope_label.lower()
    shown = filter_alerts(report.alerts, sev or None, cats or None, scope)

    if not shown:
        ui.md(ui.empty_state("No alerts match these filters", "Clear a filter to see all active alerts."))
        return
    ui.md(f'<div class="mt-note" style="margin:.3rem 0 .7rem">Showing {len(shown)} of {len(report.alerts)}</div>')
    for a in shown:
        ui.md(ui.alert_card(
            a.severity.label, a.title, CATEGORY_LABELS[a.category], a.scope,
            format_date(a.date, False), provenance_from_basis(a.basis),
            f"{a.metric.replace('_', ' ')} {a.observed_value:g}",
        ))
        with st.expander("Why this alert and what to consider"):
            d, r = st.columns([1, 1], gap="large")
            with d:
                ui.md(ui.section("What triggered it"))
                st.markdown(a.explanation)
                ui.md(ui.table([("Detail", False), ("Value", True)], [
                    ["Measure", ui.esc(a.metric.replace("_", " "))],
                    ["Observed value", ui.esc(f"{a.observed_value:g}")],
                    ["Alert condition", ui.esc(a.threshold)],
                    ["Date", ui.esc(format_date(a.date))],
                    ["Timing", ui.esc("Predicted state" if a.scope == "predicted" else "Current state")],
                    ["Based on", ui.badge(provenance_from_basis(a.basis))],
                    ["Alert ID", f'<span class="mt-note">{ui.esc(a.alert_id)}</span>'],
                ]))
            with r:
                ui.md(ui.section("Suggested actions"))
                linked = recs_for_alert(recs, a.alert_id)
                if not linked:
                    ui.md('<div class="mt-note">No recommendation is linked to this alert.</div>')
                for rec in linked:
                    st.markdown(f"**{rec.title}**")
                    st.markdown(rec.action)
                    ui.md(f'<div class="mt-note">{ui.esc(rec.rationale)}</div>')
                    ui.md(f'<div style="margin:.45rem 0 .9rem">{ui.badge("rule")} '
                          f'<span class="mt-note">Rule-based recommendation. Alert is '
                          f'{ui.esc(ui.theme.PROVENANCE[provenance_from_basis(rec.alert_basis)][0].lower())}.</span></div>')


def _all_clear() -> None:
    st.write("")
    with ui.card("al-clear"):
        ui.md(ui.all_clear(
            "No active alerts",
            "Every monitored indicator is below the level where an alert would start. Nothing needs attention right now.",
        ))
        st.write("")
        left, right = st.columns(2, gap="large")
        cur = backend.current_state()
        with left:
            ui.md(ui.section("Headroom today"))
            if cur.ok:
                for r in monitoring_rows(cur.value, AlertThresholds(), backend.reference_stats()):
                    fill, mark = meter_geometry(r["value"], r["alert_from"], r["direction"])
                    ui.md(ui.meter(
                        r["label"], f'{r["value"]:,.1f}{r["unit"]}', fill, mark,
                        f'Alert {"at" if r["direction"] == "≥" else "if below"} {r["alert_from"]:,.1f}{r["unit"]}',
                    ))
            else:
                ui.md(ui.unavailable("Headroom", cur.error))
        with right:
            ui.md(ui.section("What is being watched"))
            for cat, label in CATEGORY_LABELS.items():
                ui.md(f'<div style="margin:.05rem 0 .65rem"><div style="font-weight:600;font-size:.88rem">{ui.esc(label)}</div>'
                      f'<div class="mt-note">{ui.esc(CATEGORY_HELP[cat])}</div></div>')
        ui.md(ui.how_to_read(
            "The amber mark on each meter is where an alert would start. " + GLOSSARY["anomaly"]
        ))
        ui.md('<div class="mt-note" style="margin-top:.6rem">Want to see alerts in action? Open the What-If Simulator '
              "and choose a stressful scenario.</div>")


def _recs_tab(report, recs) -> None:
    if not recs:
        ui.md(ui.empty_state("No recommendations", "Recommendations appear when an alert is active."))
        return
    titles = {a.alert_id: a.title for a in report.alerts}
    st.write("")
    for rec in recs:
        sev = rec.severity.label
        with st.expander(f"{sev} · {rec.title}"):
            ui.md(f'<div style="margin-bottom:.5rem">{ui.severity_pill(sev)} {ui.badge("rule")}</div>')
            st.markdown(rec.action)
            ui.md(f'<div class="mt-note">{ui.esc(rec.rationale)}</div>')
            ui.md(ui.section("Linked alerts"))
            for aid in rec.related_alert_ids:
                st.markdown(f"- {titles.get(aid, aid)}")
            ui.md(f'<div class="mt-note" style="margin-top:.4rem">{ui.esc(rec.disclaimer)}</div>')
