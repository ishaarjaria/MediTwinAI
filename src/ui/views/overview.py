"""Hospital Overview: status, key indicators, outlook, active alerts and trends."""

from __future__ import annotations

import streamlit as st

from src.alerts.models import AlertThresholds
from src.ui import backend, charts, components as ui, theme
from src.ui.data import (
    CATEGORY_LABELS,
    GLOSSARY,
    WINDOWS,
    filter_dates,
    fmt_num,
    format_date,
    meter_geometry,
    monitoring_rows,
    provenance_from_basis,
    severity_counts,
    spark_series,
    state_staff_pressure,
    status_headline,
    status_summary,
    trailing_context,
    window_dates,
)
from src.ui.views import go_to

_WINDOWS = WINDOWS[:3]


def render() -> None:
    ops, cur = backend.operations(), backend.current_state()
    if not ops.ok or not cur.ok:
        ui.md(ui.page_header("Hospital overview", eyebrow="Today"))
        ui.md(ui.unavailable("Hospital data", ops.error or cur.error))
        return
    df, state = ops.value, cur.value
    pred = backend.predicted_state(1)
    fc = backend.forecast(7)
    report = backend.operational_report(1)
    pred_state = pred.value if pred.ok else None

    ui.md(ui.page_header(
        "Hospital overview",
        "Today's status, tomorrow's outlook and what needs attention.",
        ui.date_chip(f"As of {format_date(state.as_of_date)}") + ui.status_pill(state.capacity_status),
        eyebrow="Today",
    ))

    _hero(state, pred_state, report)
    st.write("")
    _kpis(state, pred_state, df)

    st.write("")
    left, right = st.columns([5, 3], gap="medium")
    with left, ui.card("overview-opd"):
        win = st.segmented_control(
            "Window", _WINDOWS, default="Last 30 days", key="ov_window", label_visibility="collapsed"
        ) or "Last 30 days"
        start, end = window_dates(df, win)
        view = filter_dates(df, start, end)
        ui.md(ui.section("Expected outpatients", ui.info(GLOSSARY["opd"])))
        ui.plot(charts.opd_forecast_chart(view, fc.value if fc.ok else None), key="ov_opd")
        if fc.ok:
            ui.md(ui.how_to_read(
                "The blue line is the number of outpatients seen each day. The teal dotted line, inside the "
                "shaded forecast zone, is the model's expectation for the next 7 days. " + GLOSSARY["forecast_band"]
            ))
        else:
            ui.md(ui.unavailable("OPD forecast", fc.error))
    with right, ui.card("overview-alerts"):
        _alerts_panel(state, report)

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("overview-occ"):
        ui.md(ui.section("Bed and ICU use over time"))
        ui.plot(charts.occupancy_trend(view), key="ov_occ")
        ui.md(ui.how_to_read("Each line is the share of capacity in use. Higher means fuller."))
    with b, ui.card("overview-wait"):
        ui.md(ui.section("Average waiting time", ui.info(GLOSSARY["wait"])))
        ui.plot(charts.wait_trend(view), key="ov_wait")
        ui.md(ui.how_to_read("Thin line: each day. Bold line: the 7-day average, which smooths out weekday and weekend swings."))


def _hero(state, pred, report) -> None:
    n = highest = None
    if report.ok:
        alerts, _recs = report.value
        n = len(alerts.alerts)
        highest = alerts.highest_severity.label if alerts.highest_severity else None
    facts = [(
        "Active alerts",
        "—" if n is None else str(n),
        "Alert checks unavailable" if n is None else (f"Highest: {highest}" if highest else "Nothing needs attention"),
    )]
    if pred is not None:
        facts.append((
            "Tomorrow's outlook",
            f"{fmt_num(pred.current_opd)} outpatients",
            f"Estimated wait {pred.average_waiting_time:.0f} min",
        ))
    else:
        facts.append(("Tomorrow's outlook", "—", "Forecast unavailable"))
    facts.append(("Data as of", format_date(state.as_of_date, False), "Synthetic dataset"))
    ui.md(ui.hero(
        state.capacity_status,
        status_headline(state.capacity_status),
        status_summary(state, n, highest),
        facts,
        tip=GLOSSARY["status"],
        rings=[
            ("Beds", state.bed_occupancy_percentage, f"{fmt_num(state.occupied_beds)} of {fmt_num(state.total_beds)}"),
            ("ICU", state.icu_occupancy_percentage, f"{fmt_num(state.occupied_icu_beds)} of {fmt_num(state.total_icu_beds)}"),
        ],
    ))


def _ctx(df, column: str, fmt: str, suffix: str = "") -> str:
    avg = trailing_context(df, column, 30)
    return "" if avg is None else f'<br><span style="color:{theme.FAINT}">30-day average {avg:{fmt}}{suffix}</span>'


def _kpis(state, pred, df) -> None:
    b = lambda k: ui.badge(k, compact=True)  # noqa: E731 - tiny local helper
    sp = lambda key, color=None: ui.sparkline(spark_series(df, key), color)  # noqa: E731
    items = [dict(
        label="Beds in use", value=fmt_num(state.occupied_beds), unit=f"/ {fmt_num(state.total_beds)}",
        sub=f"{fmt_num(state.available_beds)} free · {state.bed_occupancy_percentage:.1f}%"
        + _ctx(df, "bed_occupancy_percentage", ".1f", "%"),
        bar_pct=state.bed_occupancy_percentage, badge=b("observed"), tip=GLOSSARY["beds"],
        spark=sp("bed_occupancy_percentage"),
    ), dict(
        label="ICU in use", value=fmt_num(state.occupied_icu_beds), unit=f"/ {fmt_num(state.total_icu_beds)}",
        sub=f"{state.icu_occupancy_percentage:.1f}% of ICU capacity" + _ctx(df, "icu_occupancy_percentage", ".1f", "%"),
        bar_pct=state.icu_occupancy_percentage, badge=b("observed"), tip=GLOSSARY["icu"],
        spark=sp("icu_occupancy_percentage", theme.CORAL),
    )]
    if pred is not None:
        opd_sub = f"Expected {format_date(pred.as_of_date, False)}: {fmt_num(pred.current_opd)}"
        opd_badge = b("observed") + " " + b("ml")
    else:
        opd_sub, opd_badge = "Forecast unavailable", b("observed")
    items.append(dict(
        label="Outpatients", value=fmt_num(state.current_opd), unit="today",
        sub=opd_sub + _ctx(df, "opd_patients", ",.0f"), badge=opd_badge, tip=GLOSSARY["opd"],
        spark=sp("opd_patients", theme.BLUE),
    ))
    if pred is not None:
        mw = backend.current_model_wait()
        like = (
            f"Expected today {mw.value:.1f} min · recorded {state.average_waiting_time:.1f} min"
            if mw.ok else f"Recorded today {state.average_waiting_time:.1f} min"
        )
        items.append(dict(
            label="Waiting time", value=f"{pred.average_waiting_time:.1f}", unit="min",
            sub=f"Estimate for next day<br>{like}", badge=b("ml"), tip=GLOSSARY["wait"],
            spark=sp("average_waiting_time_minutes", theme.AMBER),
        ))
    else:
        items.append(dict(
            label="Waiting time", value=f"{state.average_waiting_time:.1f}", unit="min",
            sub="Today · estimate unavailable", badge=b("observed"), tip=GLOSSARY["wait"],
            spark=sp("average_waiting_time_minutes", theme.AMBER),
        ))
    pressure = state_staff_pressure(state)
    items.append(dict(
        label="Staff workload", value="—" if pressure is None else f"{pressure:.2f}", unit="per staff",
        sub=f"{fmt_num(state.doctors_available)} doctors · {fmt_num(state.nurses_available)} nurses",
        badge=b("rule"), tip=GLOSSARY["staff"], spark=sp("staff_pressure", theme.MUTED),
    ))
    ui.md(ui.stat_strip(items))


def _alerts_panel(state, report) -> None:
    ui.md(ui.section("Needs attention"))
    if not report.ok:
        ui.md(ui.unavailable("Alerts", report.error))
        return
    alerts, _recs = report.value
    if alerts.alerts:
        counts = severity_counts(alerts.alerts)
        ui.md('<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.75rem">'
              + "".join(ui.chip(label, n, label) for label, n in counts.items() if n) + "</div>")
        for a in alerts.alerts[:4]:
            ui.md(ui.alert_card(
                a.severity.label, a.title, CATEGORY_LABELS[a.category], a.scope,
                format_date(a.date, False), provenance_from_basis(a.basis), a.threshold,
            ))
        if len(alerts.alerts) > 4:
            ui.md(f'<div class="mt-note">+{len(alerts.alerts) - 4} more in Alerts &amp; Recommendations.</div>')
    else:
        ui.md(ui.all_clear("No active alerts", "Every monitored indicator is below its alert level."))
        st.write("")
        ui.md(ui.section("Headroom today"))
        for r in monitoring_rows(state, AlertThresholds(), backend.reference_stats()):
            fill, mark = meter_geometry(r["value"], r["alert_from"], r["direction"])
            ui.md(ui.meter(
                r["label"], f'{r["value"]:,.1f}{r["unit"]}', fill, mark,
                f'Alert {"at" if r["direction"] == "≥" else "if below"} {r["alert_from"]:,.1f}{r["unit"]}',
            ))
    st.button("Open alerts & recommendations", on_click=go_to, args=("alerts",),
              key="ov_open_alerts", width="stretch")
    if alerts.unavailable:
        ui.md(f'<div class="mt-note" style="margin-top:.5rem">{len(alerts.unavailable)} check(s) '
              "could not run. Details are in Alerts &amp; Recommendations.</div>")
