"""Forecasts: what to expect, with the model details under "How this is calculated"."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import backend, charts, components as ui, theme
from src.ui.data import (
    GLOSSARY,
    forecast_comparison_rows,
    fmt_num,
    format_date,
    model_label,
    resource_projection_rows,
    waiting_comparison_rows,
)

_HORIZONS = (7, 14, 30)
_BASELINE_NAMES = {"training_mean": "training mean", "opd_only_linear": "straight line on OPD only", "day_of_week_mean": "day-of-week mean"}


def _metric_cell(value: float | None, decimals: int = 2) -> str:
    return "—" if value is None else f"{value:.{decimals}f}"


def _comparison_table(rows: list[dict], with_validation: bool) -> str:
    headers = [("Model", False), ("Kind", False)]
    if with_validation:
        headers.append(("Validation MAE", True))
    headers += [("Test MAE", True), ("Test RMSE", True), ("Test R²", True)]
    body = []
    for r in rows:
        name = ui.esc(r["model"]) + (" " + ui.pill("Selected", theme.TEAL, theme.TEAL_SOFT) if r["selected"] else "")
        kind = "Baseline" if r["type"] == "baseline" else "Model"
        cells = [name, kind]
        if with_validation:
            cells.append(_metric_cell(r.get("val_mae")))
        cells += [_metric_cell(r["test_mae"]), _metric_cell(r["test_rmse"]), _metric_cell(r["test_r2"], 3)]
        if r.get("note"):
            cells[0] += f'<div class="mt-note">{ui.esc(r["note"])}</div>'
        body.append(cells)
    return ui.table(headers, body)


# ----------------------------------------------------------------------
def _opd_tab(cards: dict) -> None:
    meta = cards.get("opd_forecast")
    ops = backend.operations()
    if meta:
        iv, rows = meta["interval"], {r["model"]: r for r in forecast_comparison_rows(meta)}
        sel, base = meta["selected_model"], meta["best_baseline_by_test"]
        beats = meta.get("selected_beats_best_baseline_on_test")
        ui.md(ui.stat_strip([
            dict(label="Typical error", value=f"{rows[sel]['test_mae']:.0f}", unit="patients/day",
                 sub="Average miss on 30 unseen test days (lower is better)"),
            dict(label="Simple comparison", value=f"{rows[base]['test_mae']:.0f}", unit="patients/day",
                 sub=f"{base}: the best no-AI comparison"),
            dict(label="Verdict", value="About equal" if not beats else "Better",
                 sub="The model " + ("beat" if beats else "did not clearly beat") + " the best simple baseline"),
            dict(label="Likely range held", value=f"{iv['coverage_test']:.0%}", unit="of test days",
                 sub="Aimed at 80%, so use it as a rough guide", badge=ui.badge("ml")),
        ]))
    st.write("")
    ui.md(ui.section("Outpatient (OPD) forecast", ui.badge("ml")))
    horizon = st.segmented_control(
        "Forecast length", _HORIZONS, default=7, key="pred_horizon", label_visibility="collapsed",
        format_func=lambda d: f"{d} days",
    ) or 7
    fc = backend.forecast(int(horizon))
    info = backend.opd_model_info()
    if not fc.ok or not ops.ok:
        ui.md(ui.unavailable("OPD forecast", fc.error or ops.error))
    else:
        with ui.card("pred-opd-chart"):
            ui.plot(charts.opd_forecast_chart(ops.value.tail(60), fc.value, height=340), key="pred_opd_chart")
            ui.md(ui.how_to_read(GLOSSARY["forecast_band"]))
            if info.ok:
                with st.expander("How this is calculated"):
                    ui.md(f'<div class="mt-note">Model: {ui.esc(info.value[0])}. {ui.esc(info.value[1])}</div>')
            if horizon > 7:
                ui.md(ui.callout(
                    "Beyond 7 days each day is predicted from earlier predictions, so errors can build up. "
                    "Accuracy below was measured for forecasts of up to 7 days ahead only.", "warn"))
        f = fc.value
        body = [[format_date(r.date), fmt_num(r.opd_forecast), f"{fmt_num(r.yhat_lower)} – {fmt_num(r.yhat_upper)}"]
                for r in f.itertuples()]
        with ui.card("pred-opd-table"):
            ui.md(ui.table([("Date", False), ("Expected patients", True), ("Likely range", True)], body))
    st.write("")
    with st.expander("Compare with simple baselines"):
        _opd_comparison(meta)


def _opd_comparison(meta: dict | None) -> None:
    with ui.card("pred-opd-compare"):
        ui.md(ui.section("How good is it? Model versus simple baselines"))
        if not meta:
            ui.md(ui.unavailable("Forecast metrics", "models/opd_forecast/metrics.json not found"))
            return
        rows = forecast_comparison_rows(meta)
        ui.plot(charts.model_comparison_chart(rows, "test_mae", "Test MAE in patients per day (lower is better)"),
                key="pred_opd_cmp")
        ui.md(_comparison_table(rows, with_validation=True))
        sel = meta["selected_model"]
        base = meta["best_baseline_by_test"]
        by_name = {r["model"]: r for r in rows}
        beats = meta.get("selected_beats_best_baseline_on_test")
        verdict = (
            f"{sel} reached a test MAE of {by_name[sel]['test_mae']:.1f} patients per day. The best simple baseline on the "
            f"same days, {base}, reached {by_name[base]['test_mae']:.1f}. "
        )
        verdict += (
            "The trained model beat every baseline on the test block." if beats
            else "The trained model did NOT clearly beat the best baseline on the test block, so on this synthetic "
                 "data a simple weekday average is about as good."
        )
        ui.md(ui.callout(verdict, "" if beats else "warn"))
        iv = meta["interval"]
        ui.md(
            f'<div class="mt-note" style="margin-top:.6rem"><b>Likely range.</b> Built from validation errors and '
            f'nominally 80%; it contained the actual value on {iv["coverage_test"]:.0%} of test days. '
            + ("That is close to 80%." if iv["validated_80pct"] else "That is not close enough to call it a validated 80% interval.")
            + "</div>"
        )
        ui.md(
            f'<div class="mt-note">Selected on the validation block ({meta["validation_start_date"]} onward) by lowest MAE, '
            f'then reported on the later test block ({meta["test_start_date"]} to {meta["test_end_date"]}); '
            "each day is scored as a forecast of at most 7 days ahead. Only 30 test days, so small gaps are within noise.</div>"
        )
        if meta.get("prophet_note"):
            ui.md(ui.callout(
                "Prophet (a well-known forecasting tool) could not be run on this computer because a Windows security "
                "policy blocked its solver, so it was not compared. An earlier run on a previous version of the data "
                "scored worse than a simple average, so it is not used.", "warn"))


# ----------------------------------------------------------------------
def _state_rows(horizons=range(1, 8)):
    states, errors = [], []
    for h in horizons:
        out = backend.predicted_state(h)
        if out.ok:
            states.append(out.value)
        else:
            errors.append(out.error)
    return states, errors


def _waiting_tab(cards: dict) -> None:
    st.write("")
    ui.md(ui.section("Waiting-time prediction", ui.badge("ml")))
    states, errors = _state_rows()
    cur = backend.current_state()
    mw = backend.current_model_wait()
    if not states:
        ui.md(ui.unavailable("Predicted waiting time", errors[0] if errors else None))
    else:
        body = []
        if cur.ok and mw.ok:
            body.append([f"Today · {ui.esc(format_date(cur.value.as_of_date, False))}", fmt_num(cur.value.current_opd),
                         f"{mw.value:.1f} min", f"{cur.value.average_waiting_time:.1f} min"])
        for s in states:
            body.append([ui.esc(format_date(s.as_of_date, False)), fmt_num(s.current_opd),
                         f"{s.average_waiting_time:.1f} min", "—"])
        with ui.card("pred-wait-table"):
            ui.md(ui.table([("Day", False), ("Expected patients", True), ("Expected waiting time", True),
                            ("Recorded", True)], body))
            ui.md(ui.how_to_read(
                "Today's row uses the same model as the future days, so the two are comparable. The observed value "
                "is shown for reference only; a model estimate always differs a little from what was recorded."))
    st.write("")
    meta = cards.get("waiting_time")
    with st.expander("Compare with simple baselines"):
        ui.md(ui.section("How good is it? Model versus simple baselines"))
        if not meta:
            ui.md(ui.unavailable("Waiting-time metrics", "models/waiting_time/metrics.json not found"))
            return
        rows = waiting_comparison_rows(meta)
        ui.plot(charts.model_comparison_chart(rows, "test_rmse", "Test RMSE in minutes (lower is better)"), key="pred_wait_cmp")
        ui.md(_comparison_table(rows, with_validation=False))
        sel = meta["selected"]
        beats = sel.get("beats_best_baseline")
        best = meta.get("best_baseline")
        ui.md(ui.callout(
            f"{model_label(sel['model'])} reached a test RMSE of {sel['rmse']:.2f} minutes (R² {sel['r2']:.2f}). "
            + (f"That beats the best simple baseline ({_BASELINE_NAMES.get(best, best)})." if beats
               else f"That does not beat the best simple baseline ({_BASELINE_NAMES.get(best, str(best))}).")
            , "" if beats else "warn"))
        ui.md(
            f'<div class="mt-note">Selected on the validation block by lowest MAE, then evaluated on the later test block '
            f'from {meta["test_start_date"]} ({meta["n_test"]} days). On this synthetic data waiting time is largely '
            "a formula of outpatient load, emergency volume, occupancy and doctors, so a good score shows the model "
            "learned that formula, not that it would work in a real hospital.</div>")


def _resource_tab() -> None:
    st.write("")
    ui.md(ui.section("Resource projections", ui.badge("rule")))
    ui.md(ui.callout(
        "No trained model exists for beds, ICU, emergency cases or medicine stock. These figures come from "
        "transparent formulas applied to the forecast outpatient count, so they stay close to today's values by "
        "design. Do not read them as machine-learning predictions.", "warn"))
    states, errors = _state_rows()
    st.write("")
    if not states:
        ui.md(ui.unavailable("Resource projections", errors[0] if errors else None))
        return
    body = []
    for r in resource_projection_rows(states):
        sp = r["staff_pressure"]
        body.append([ui.esc(format_date(r["date"], False)), f'{r["beds"]:.1f}%', f'{r["icu"]:.1f}%',
                     fmt_num(r["emergency"]), fmt_num(r["medicine"]), "—" if sp is None else f"{sp:.2f}"])
    with ui.card("pred-resources"):
        ui.md(ui.table([("Day", False), ("Beds in use", True), ("ICU in use", True), ("Emergency cases", True),
                        ("Medicine stock", True), ("Staff workload", True)], body))
        ui.md(ui.how_to_read(
            "Doctors and nurses are carried forward from today. Medicine stock only falls if outpatients are expected to "
            "rise, and it never receives a delivery. Use the What-If simulator to test how these respond to changes."))


def _performance_tab(cards: dict) -> None:
    st.write("")
    ui.md(ui.section("All models at a glance"))
    risk = cards.get("patient_risk")
    anomaly = cards.get("anomaly")
    items = []
    if risk and risk.get("selected"):
        s = risk["selected"]
        items.append(("Patient risk classifier", model_label(s["model"]),
                      f'Accuracy {s["accuracy"]:.1%} · macro F1 {s["f1_macro"]:.2f} · High recall {s["recall_high"]:.0%}',
                      "Labels come from fixed synthetic rules, so this measures how well the rules were relearned. "
                      "A majority-class guess scores about 51%."))
    if anomaly and anomaly.get("evaluation"):
        ev = anomaly["evaluation"]
        chosen = next((r for r in ev["isolation_forest"] if r["contamination"] == anomaly["contamination"]), ev["isolation_forest"][0])
        lof = max(ev["local_outlier_factor"], key=lambda r: r["f1"])
        items.append(("Anomaly detector", "Isolation Forest",
                      f'Precision {chosen["precision"]:.0%} · recall {chosen["recall"]:.0%} · F1 {chosen["f1"]:.2f}',
                      f'Scored on {chosen["n_injected_in_scored"]} injected spike days using only earlier data. '
                      f'{ev["small_sample_warning"]} A Local Outlier Factor comparison reached F1 {lof["f1"]:.2f} '
                      f'at its best setting. Unusual-but-real pressure periods are also flagged, which counts as a miss here.'))
    opd, wait = cards.get("opd_forecast"), cards.get("waiting_time")
    if opd:
        m = opd["metrics_holdout"]
        items.insert(0, ("Outpatient forecaster", opd["selected_model"].split(" (")[0],
                         f'Test MAE {m["mae"]:.1f} patients/day · R² {m["r2"]:.2f}',
                         "Compared with four simple baselines and Ridge on 30 unseen days; about equal to a weekday average. "
                         "The likely range held on " + f'{opd["interval"]["coverage_test"]:.0%} of test days.'))
    if wait:
        w = wait["selected"]
        items.insert(1, ("Waiting-time model", model_label(w["model"]),
                         f'Test RMSE {w["rmse"]:.1f} min · R² {w["r2"]:.2f}',
                         "Beats the best simple baseline on the later test block. On synthetic data this mostly shows it learned the "
                         "generator's formula."))
    cols = st.columns(2, gap="medium")
    for i, (title, model, line, note) in enumerate(items):
        with cols[i % 2], ui.card(f"pred-perf-{i}"):
            ui.md(ui.section(title, ui.badge("ml")))
            ui.md(f'<div style="font-weight:600">{ui.esc(model)}</div><div style="margin:.2rem 0 .5rem">{ui.esc(line)}</div>'
                  f'<div class="mt-note">{ui.esc(note)}</div>')
    if not items:
        ui.md(ui.unavailable("Model metrics", "metrics files not found"))
    st.write("")
    with ui.card("pred-limits"):
        ui.md(ui.section("Limitations to keep in mind"))
        for text in (
            "All data is synthetic and generated by this project. Scores say how well models learn the generator, not how "
            "they would do in a real hospital.",
            "Waiting time and OPD are the only quantities predicted by a trained model for the twin. Everything else is a "
            "formula and is labelled Estimated.",
            "The OPD test block is only 30 days, so differences of a few patients per day between models are within noise.",
            "Not a medical device or clinical decision system. No real patient data, sensors or hospital systems are involved.",
        ):
            ui.md(f'<div class="mt-list">{ui.esc(text)}</div>')


def render() -> None:
    ui.md(ui.page_header(
        "Forecasts",
        "What to expect in the coming days, and how reliable it is.",
        eyebrow="Look ahead",
    ))
    cards = backend.model_cards()
    if not cards.ok:
        ui.md(ui.unavailable("Model metrics", cards.error))
        cards_value: dict = {}
    else:
        cards_value = cards.value
    # Lazy tabs: only the open tab is computed, which keeps page changes fast.
    t1, t2, t3, t4 = st.tabs(["Outpatients", "Waiting time", "Resources", "How this is calculated"],
                             key="forecasts_tab", on_change="rerun")
    for tab, draw in ((t1, lambda: _opd_tab(cards_value)), (t2, lambda: _waiting_tab(cards_value)),
                      (t3, _resource_tab), (t4, lambda: _performance_tab(cards_value))):
        with tab:
            if tab.open:
                draw()
