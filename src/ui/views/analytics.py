"""Operations & Analytics: filterable demand, flow, staffing, supplies and EDA."""

from __future__ import annotations

import streamlit as st

from src.ui import backend, charts, components as ui
from src.ui.data import (
    GLOSSARY,
    WINDOWS,
    filter_dates,
    fmt_num,
    format_date,
    model_label,
    monthly_profile,
    summarize_window,
    to_weekly,
    weekday_profile,
    window_dates,
)


def render() -> None:
    ops = backend.operations()
    if not ops.ok:
        ui.md(ui.page_header("Trends", eyebrow="Today"))
        ui.md(ui.unavailable("Hospital data", ops.error))
        return
    df = ops.value
    header_slot = st.empty()  # filled once the selected period is known

    # ---- Period selector ----------------------------------------------
    ui.md(ui.section("Time period"))
    c1, c2 = st.columns([2, 3], gap="medium")
    with c1:
        label = st.segmented_control(
            "Period", WINDOWS, default="Full period", key="an_window", label_visibility="collapsed"
        ) or "Full period"
    custom = None
    with c2:
        if label == "Custom":
            lo, hi = df["date"].min().date(), df["date"].max().date()
            custom = st.date_input("Custom range", value=(lo, hi), min_value=lo, max_value=hi, key="an_range")
            if not (isinstance(custom, (tuple, list)) and len(custom) == 2):
                st.caption("Pick both a start and an end date.")
                custom = None
    start, end = window_dates(df, label, custom)
    view = filter_dates(df, start, end)

    header_slot.markdown(ui.page_header(
        "Trends",
        "Demand, capacity, staffing and supplies over time.",
        ui.date_chip(f"{format_date(start, False)} – {format_date(end)} · {len(view)} days") + ui.badge("observed"),
        eyebrow="Today",
    ), unsafe_allow_html=True)
    if view.empty:
        ui.md(ui.empty_state("No data in this period", "Choose a wider date range."))
        return

    s = summarize_window(view)
    cols = st.columns(4, gap="small")
    stats = [
        ("Outpatients per day", fmt_num(s["avg_opd"]), "average", f"Busiest day: {fmt_num(s['peak_opd'])}", GLOSSARY["opd"]),
        ("Beds in use", f"{s['avg_bed']:.1f}", "% average", "Share of beds with a patient", GLOSSARY["beds"]),
        ("ICU in use", f"{s['avg_icu']:.1f}", "% average", "Share of ICU capacity used", GLOSSARY["icu"]),
        ("Waiting time", f"{s['avg_wait']:.1f}", "min average", "Average across days", GLOSSARY["wait"]),
    ]
    for col, (lab, val, unit, sub, tip) in zip(cols, stats):
        with col:
            ui.md(ui.tile(lab, val, unit, sub, compact=True, tip=tip))
    st.write("")

    # Lazy tabs: only the open tab's charts are built, which keeps page changes fast.
    t_demand, t_flow, t_staff, t_anom, t_risk = st.tabs(
        ["Demand", "Capacity & flow", "Staffing & supplies", "Unusual activity", "Patient risk"],
        key="trends_tab", on_change="rerun",
    )
    for tab, draw in ((t_demand, lambda: _demand(view)), (t_flow, lambda: _flow(view)),
                      (t_staff, lambda: _staffing(view)), (t_anom, lambda: _anomalies(view)), (t_risk, _risk)):
        with tab:
            if tab.open:
                draw()


def _panel(name: str, title: str, fig, how: str = "", tip: str | None = None) -> None:
    with ui.card(name):
        ui.md(ui.section(title, ui.info(tip) if tip else ""))
        ui.plot(fig, key=f"an_{name}")
        if how:
            ui.md(ui.how_to_read(how))


def _demand(view) -> None:
    st.write("")
    _panel("demand", "Patient demand", charts.demand_trend(view),
           "Each thin line is a daily count. The bold teal line is the 7-day average of outpatients, "
           "which makes the underlying trend easier to see.", GLOSSARY["opd"])
    st.write("")
    a, b = st.columns(2, gap="medium")
    with a:
        _panel("weekday", "Average load by weekday", charts.weekday_profile_chart(weekday_profile(view)),
               "Taller bars are busier days on average. Weekends are typically quieter for outpatients.")
    with b:
        _panel("monthly", "Average outpatients by month", charts.monthly_chart(monthly_profile(view)),
               "Each bar is the average outpatients per day in that month, which shows seasonal patterns.")
    st.write("")
    _panel("scatter", "Waiting time against outpatient load", charts.wait_scatter(view),
           "Each dot is one day. Dots rising to the right mean busier days tend to have longer waits.")


def _flow(view) -> None:
    st.write("")
    flow, grain = to_weekly(view, ["admissions", "discharges"])
    _panel("flow", "Admissions and discharges", charts.flow_chart(flow, grain),
           ("Daily counts. " if grain == "daily" else "Weekly averages of daily counts, because the period is long. ")
           + "When admissions stay above discharges, beds fill up.")
    st.write("")
    _panel("occupancy", "Bed and ICU use", charts.occupancy_trend(view, 280),
           "The share of beds and ICU capacity in use each day. Closer to 100% means less room to absorb extra patients.",
           GLOSSARY["beds"])
    st.write("")
    with ui.card("coverage"):
        ui.md(ui.section("Data coverage"))
        ui.md(ui.callout(
            "Refused requests and bed availability by service are not recorded in this dataset, "
            "so they are not shown. The figures above use hospital-wide bed and ICU counts."
        ))


def _staffing(view) -> None:
    st.write("")
    a, b = st.columns(2, gap="medium")
    with a:
        _panel("staff", "Doctors and nurses available", charts.staffing_chart(view),
               "Two separate scales: doctors on the left, nurses on the right.", GLOSSARY["staff"])
    with b:
        _panel("wait", "Waiting time", charts.wait_trend(view, 280),
               "Thin line: each day. Bold line: the 7-day average.", GLOSSARY["wait"])
    st.write("")
    c, d = st.columns(2, gap="medium")
    with c:
        _panel("stock", "Medicine stock", charts.stock_chart(view),
               "Units of medicine in stock. Sudden rises are restocking; slow falls are use.")
    with d:
        _panel("consumption", "Medicine used per day", charts.consumption_chart(view),
               "Daily consumption. Taller bars are heavier-use days.")


def _anomalies(view) -> None:
    st.write("")
    scores = backend.anomaly_scores()
    sc = scores.value if scores.ok else None
    if sc is None:
        ui.md(ui.unavailable("Anomaly scores", scores.error or "saved scoring file not found"))
    _panel("anomaly", "Days flagged as unusual", charts.anomaly_chart(view, sc),
           "Red dots are days the model considers unusual compared with the rest of the history. "
           "A flag means different from normal, not harmful.", GLOSSARY["anomaly"])
    if sc is not None:
        flagged = sc[(sc["anomaly_flag"] == 1) & sc["date"].between(view["date"].min(), view["date"].max())]
        st.write("")
        with ui.card("anomaly-table"):
            ui.md(ui.section(f"Flagged days ({len(flagged)})"))
            if flagged.empty:
                ui.md(ui.empty_state("No flagged days in this period"))
            else:
                ui.md(ui.table(
                    [("Date", False), ("Outpatients", True), ("Emergency", True), ("Why it stood out", False), ("Score", True)],
                    [[ui.esc(format_date(r.date)), fmt_num(r.opd_patients), fmt_num(r.emergency_patients),
                      ui.esc(r.anomaly_reason), f"{r.anomaly_score:.3f}"]
                     for r in flagged.sort_values("anomaly_score", ascending=False).head(12).itertuples()],
                ))
                ui.md(ui.how_to_read("A higher score means a stronger difference from normal. z is how many standard deviations a value is from its usual level."))


def _risk() -> None:
    st.write("")
    patients, metrics = backend.patients(), backend.risk_metrics()
    ui.md(ui.callout(
        "Synthetic academic labels, not clinical validation. Every patient record here is synthetic, and each "
        "risk label was produced by fixed scoring rules, not by clinicians. The model results below only show "
        "how well it learned those rules.", "warn",
    ))
    st.write("")
    if not patients.ok or patients.value is None:
        ui.md(ui.unavailable("Patient risk data", patients.error))
        return
    p = patients.value
    a, b = st.columns(2, gap="medium")
    with a:
        _panel("risk-dist", "Risk class distribution", charts.risk_distribution(p),
               "How many of the synthetic patients fall into each risk group.")
    with b:
        _panel("risk-age", "Age by risk class", charts.risk_age_box(p),
               "Each box spans the middle half of ages in that group; the line inside is the median.")
    st.write("")
    with ui.card("risk-model"):
        ui.md(ui.section("How accurate is the risk model? (held-out test set)"))
        if metrics.ok and metrics.value:
            m = metrics.value
            cols = st.columns(3, gap="small")
            items = [("Accuracy", f"{m['accuracy']:.1%}", "Share of patients labelled correctly"),
                     ("Macro F1", f"{m['f1_macro']:.2f}", "Balanced score across the three classes"),
                     ("High-risk recall", f"{m['recall_high']:.0%}", "Share of high-risk cases it caught")]
            for col, (lab, val, sub) in zip(cols, items):
                with col:
                    ui.md(ui.tile(lab, val, "", sub, compact=True))
            with st.expander("How this is calculated"):
                ui.md(f'<div class="mt-note">Model: {ui.esc(model_label(m["model"]))}, the best of three compared on a held-out '
                      "test set. The risk groups come from fixed synthetic scoring rules, so these scores show how well the rules "
                      "were relearned, not clinical accuracy.</div>")
            ui.md(f'<div class="mt-note" style="margin-top:.7rem">Evaluated on {m["n_test"]} synthetic records. '
                  "These figures say nothing about real clinical accuracy.</div>")
        else:
            ui.md(ui.unavailable("Model metrics", metrics.error))
