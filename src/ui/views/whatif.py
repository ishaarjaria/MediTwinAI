"""What-If Simulator: interactive controls over the existing Phase 5 simulator."""

from __future__ import annotations

import streamlit as st

from src.ui import backend, charts, components as ui, theme
from src.ui.data import (
    CATEGORY_LABELS,
    GLOSSARY,
    PRESETS,
    WhatIfControls,
    apply_preset,
    control_caption,
    control_ranges,
    format_date,
    is_no_change,
    recs_for_alert,
    scenario_from_controls,
    whatif_rows,
)

_FIELDS = (
    "patient_inflow_percent", "emergency_cases_percent",
    "total_beds", "icu_capacity", "doctors_available", "nurses_available",
    "medicine_stock_percent",
)


def _key(field: str, date_iso: str) -> str:
    return f"wi_{field}_{date_iso}"


def _write_controls(date_iso: str, ctl: WhatIfControls) -> None:
    for f in _FIELDS:
        st.session_state[_key(f, date_iso)] = getattr(ctl, f)


def _on_preset(date_iso: str, baseline) -> None:
    name = st.session_state.get(f"wi_preset_{date_iso}")
    _write_controls(date_iso, apply_preset(baseline, name))


def _on_manual(date_iso: str) -> None:
    st.session_state[f"wi_preset_{date_iso}"] = None


def _on_reset(date_iso: str, baseline) -> None:
    st.session_state[f"wi_preset_{date_iso}"] = None
    _write_controls(date_iso, apply_preset(baseline, None))


def render() -> None:
    ops = backend.operations()
    ui.md(ui.page_header(
        "What-if scenarios",
        "Change conditions and compare the simulated day with the baseline.",
        eyebrow="Plan",
    ))
    if not ops.ok:
        ui.md(ui.unavailable("Hospital data", ops.error))
        return
    dates = [d.strftime("%Y-%m-%d") for d in reversed(list(ops.value["date"]))]
    latest = dates[0]

    top, _ = st.columns([2, 3])
    with top:
        date_iso = st.selectbox(
            "Baseline day (the day the scenario starts from)", dates, key="wi_date",
            format_func=lambda d: format_date(d) + ("  ·  latest" if d == latest else ""),
        )
    base_out = backend.baseline_state(date_iso)
    if not base_out.ok:
        ui.md(ui.unavailable("Baseline state", base_out.error))
        return
    base = base_out.value
    ranges = control_ranges(base)
    defaults = apply_preset(base, None)
    for f in _FIELDS:  # initialise once per baseline day; widgets own the value afterwards
        st.session_state.setdefault(_key(f, date_iso), getattr(defaults, f))

    controls_col, results_col = st.columns([5, 8], gap="medium")
    with controls_col, ui.card("wi-controls"):
        ui.md(ui.section("Scenario controls"))
        st.segmented_control(
            "Quick scenarios", list(PRESETS), key=f"wi_preset_{date_iso}",
            on_change=_on_preset, args=(date_iso, base),
            help="One-click starting points. You can still adjust any slider afterwards.",
        )
        ui.md(f'<div class="mt-note" style="margin:.7rem 0 .1rem;color:{theme.TEAL};font-weight:600">Demand</div>')
        inflow = st.slider(
            "Patient inflow (outpatients)", -50, 100, step=5, format="%+d%%",
            key=_key("patient_inflow_percent", date_iso), on_change=_on_manual, args=(date_iso,),
            help="Change in outpatient arrivals compared with the baseline day. 0% means no change.",
        )
        ui.md(f'<div class="mt-note" style="margin:-1.35rem 0 1.3rem">{ui.esc(control_caption(0, inflow, percent=True))}</div>')
        emergency = st.slider(
            "Emergency cases", -50, 200, step=5, format="%+d%%",
            key=_key("emergency_cases_percent", date_iso), on_change=_on_manual, args=(date_iso,),
            help="Change in emergency arrivals compared with the baseline day. 0% means no change.",
        )
        ui.md(f'<div class="mt-note" style="margin:-1.35rem 0 1.3rem">{ui.esc(control_caption(0, emergency, percent=True))}</div>')
        ui.md(f'<div class="mt-note" style="margin:.9rem 0 .1rem;color:{theme.TEAL};font-weight:600">Supplies</div>')
        medicine = st.slider(
            "Medicine stock on hand", -80, 100, step=5, format="%+d%%",
            key=_key("medicine_stock_percent", date_iso), on_change=_on_manual, args=(date_iso,),
            help="Change in the stock available at the start of the day, for example a late or short delivery (-60%). "
                 "0% means no change.",
        )
        ui.md(f'<div class="mt-note" style="margin:-1.35rem 0 1.3rem">{ui.esc(control_caption(0, medicine, percent=True))}</div>')
        ui.md(f'<div class="mt-note" style="margin:.9rem 0 .1rem;color:{theme.TEAL};font-weight:600">Capacity and staffing</div>')
        sliders = {}
        for field, label, unit, help_text in (
            ("total_beds", "Total beds", " beds", "Number of beds the hospital has in this scenario."),
            ("icu_capacity", "ICU capacity", " beds", "Number of ICU beds in this scenario."),
            ("doctors_available", "Doctors available", "", "Doctors on duty in this scenario."),
            ("nurses_available", "Nurses available", "", "Nurses on duty in this scenario."),
        ):
            lo, hi, default = ranges[field]
            sliders[field] = st.slider(
                label, lo, hi, step=1, key=_key(field, date_iso), on_change=_on_manual, args=(date_iso,),
                help=help_text,
            )
            ui.md(f'<div class="mt-note" style="margin:-1.35rem 0 1.3rem">{ui.esc(control_caption(default, sliders[field], unit))}</div>')
        st.button("Reset to baseline", on_click=_on_reset, args=(date_iso, base),
                  key=f"wi_reset_{date_iso}", width="stretch")

    controls = WhatIfControls(
        patient_inflow_percent=inflow, emergency_cases_percent=emergency,
        total_beds=sliders["total_beds"], icu_capacity=sliders["icu_capacity"],
        doctors_available=sliders["doctors_available"], nurses_available=sliders["nurses_available"],
        medicine_stock_percent=medicine,
    )
    # Remember the last non-trivial scenario so the Alerts page can show its alerts separately.
    if is_no_change(scenario_from_controls(base, controls)):
        st.session_state.pop("wi_last_scenario", None)
    else:
        st.session_state["wi_last_scenario"] = (date_iso, controls)
    with results_col:
        with st.spinner("Simulating…"):
            out = backend.run_whatif(date_iso, controls)
        _results(base, controls, out)
        if out.ok:
            st.write("")
            _scenario_alerts(date_iso, controls)


def _scenario_alerts(date_iso: str, controls: WhatIfControls) -> None:
    """Phase 6 alerts and recommendations for the simulated state."""
    with ui.card("wi-alerts"):
        ui.md(ui.section("Alerts under this scenario"))
        out = backend.scenario_alerts(date_iso, controls)
        if not out.ok:
            ui.md(ui.unavailable("Scenario alerts", out.error))
            return
        report, recs = out.value
        if not report.alerts:
            ui.md(ui.all_clear("This scenario would not raise any alerts", "All monitored indicators stay below their alert levels."))
        for a in report.alerts:
            ui.md(ui.alert_card(
                a.severity.label, a.title, CATEGORY_LABELS[a.category], a.scope, "", None,
                f"{a.metric.replace('_', ' ')} {a.observed_value:g}", scope_label="Scenario",
            ))
            for rec in recs_for_alert(recs, a.alert_id):
                ui.md(f'<div class="mt-note" style="margin:0 0 .7rem .5rem">'
                      f"{ui.badge('rule', compact=True)} {ui.esc(rec.action)}</div>")
        ui.md('<div class="mt-note" style="margin-top:.5rem">Evaluated by the same alert and recommendation '
              "engines as the Alerts view, on the simulated state. Recommendations are rule-based and "
              "advisory; they guarantee no outcome.</div>")


def _results(base, controls: WhatIfControls, out) -> None:
    if not out.ok:
        with ui.card("wi-error"):
            ui.md(ui.section("Scenario"))
            ui.md(ui.callout(f"This scenario could not be simulated. {out.error}", "warn"))
        return
    result = out.value
    sim = result.simulated_state
    no_change = is_no_change(scenario_from_controls(base, controls))

    with ui.card("wi-summary"):
        ui.md(ui.section("Outcome"))
        ui.md(
            '<div style="display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;margin-bottom:.6rem">'
            f'<span class="mt-note">Baseline</span>{ui.status_pill(result.baseline_capacity_status or base.capacity_status)}'
            f'<span style="color:{theme.FAINT}">→</span>'
            f'<span class="mt-note">Simulated</span>{ui.status_pill(sim.capacity_status)}'
            f'{ui.info(GLOSSARY["status"])}</div>'
        )
        if no_change:
            ui.md(ui.callout("No changes selected. The simulated day equals the baseline.", "ok"))
        else:
            ui.md('<div class="mt-note">' + " · ".join(ui.esc(c) for c in result.changes.describe()) + "</div>")
            ui.md(f'<div class="mt-note" style="margin-top:.25rem">Outpatients {base.current_opd:,} → {sim.current_opd:,} · '
                  f"emergency {base.emergency_cases:,} → {sim.emergency_cases:,}</div>")
            chips = [
                f'<span class="mt-dchip"><b>{ui.esc(r["label"])}</b>{ui.delta_html(r["delta"], r["tone"], r["decimals"], r["unit"])}</span>'
                for r in whatif_rows(result)
                if r["delta"] is not None and round(abs(r["delta"]), r["decimals"]) != 0
            ]
            if chips:
                ui.md('<div class="mt-chips">' + "".join(chips) + "</div>")

    st.write("")
    with ui.card("wi-table"):
        ui.md(ui.section("Baseline vs simulated"))
        rows, last = [], None
        for r in whatif_rows(result):
            if r["group"] != last:
                rows.append(["__group__", r["group"]])
                last = r["group"]
            base_txt = "∞" if r["unbounded"] else ui.value_text(r["baseline"], r["decimals"], r["unit"])
            sim_txt = "∞" if r["unbounded"] else ui.value_text(r["simulated"], r["decimals"], r["unit"])
            rows.append([
                f'{ui.esc(r["label"])} {ui.info(r["explanation"])} {ui.badge(r["source"], compact=True)}',
                base_txt, sim_txt,
                ui.delta_html(r["delta"], r["tone"], r["decimals"], r["unit"]),
            ])
        ui.md(ui.table(
            [("Metric and source", False), ("Baseline", True), ("Simulated", True), ("Change", True)], rows,
        ))
        ui.md(ui.how_to_read(
            "Green is a favourable change and orange an unfavourable one. Waiting time is the only value "
            "estimated by a trained model; the rest follow fixed rules. Hover the i icons for definitions."
        ))

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("wi-chart"):
        ui.md(ui.section("Beds and ICU in use"))
        ui.plot(charts.paired_occupancy(
            ["Beds", "ICU"],
            [base.bed_occupancy_percentage, base.icu_occupancy_percentage],
            [sim.bed_occupancy_percentage, sim.icu_occupancy_percentage],
            "Baseline", "Simulated", height=200,
        ), key="wi_occ")
    with b, ui.card("wi-notes"):
        ui.md(ui.section("Notes"))
        shown = False
        if result.bed_shortfall:
            ui.md(ui.callout(f"Bed demand exceeds capacity by {result.bed_shortfall} beds; occupancy is capped at 100%.", "warn"))
            shown = True
        if result.icu_shortfall:
            ui.md(ui.callout(f"ICU demand exceeds capacity by {result.icu_shortfall} beds; occupancy is capped at 100%.", "warn"))
            shown = True
        for note in result.ml_notes:
            ui.md(ui.callout(note))
            shown = True
        if not shown:
            ui.md('<div class="mt-note">No capacity shortfalls in this scenario.</div>')
        with st.expander("How this is calculated"):
            st.markdown(
                "- **Waiting time** is predicted by the trained waiting-time model for both the baseline and "
                "the scenario, using the same inputs, so an unchanged scenario shows no difference.\n"
                f"- The observed waiting time on this day was **{result.observed_waiting_time:.1f} min**; it is "
                "not used in the comparison.\n"
                "- **Patient load, beds, ICU, medicine and staff workload** use fixed rules from the Phase 5 "
                "simulator. They are not model predictions.\n"
                "- The model has no ICU-capacity input, so ICU changes do not alter waiting time."
            )
