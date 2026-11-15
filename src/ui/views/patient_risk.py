"""Patient Risk: score one synthetic patient record with the trained classifier."""

from __future__ import annotations

import streamlit as st

from src.ui import backend, charts, components as ui, theme
from src.ui.data import (
    PATIENT_GENDERS,
    PATIENT_INPUTS,
    PATIENT_PRESETS,
    model_label,
    patient_record,
    risk_explanation,
    risk_factors,
)

DISCLAIMER = (
    "Educational prototype. This is not a medical diagnosis and must not be used for real patient care. "
    "The model was trained on synthetic patients whose risk labels came from fixed scoring rules, "
    "not from clinicians. It shows how a risk model behaves, nothing more."
)
_PRESET_NONE = "Choose an example…"
_BOOL_FIELDS = (("chronic_disease", "Chronic disease"), ("emergency_case", "Arrived as an emergency"))


def _on_example() -> None:
    name = st.session_state.get("pr_example")
    preset = PATIENT_PRESETS.get(name)
    if not preset:
        return
    for key, value in preset.items():
        st.session_state[f"pr_{key}"] = value


def _init_defaults() -> None:
    for key, _label, _lo, _hi, default, _step, _help in PATIENT_INPUTS:
        st.session_state.setdefault(f"pr_{key}", default)
    st.session_state.setdefault("pr_gender", PATIENT_GENDERS[0])
    for key, _ in _BOOL_FIELDS:
        st.session_state.setdefault(f"pr_{key}", "No")


def _form() -> dict | None:
    """Render the form; return the entered values when it was submitted."""
    st.selectbox(
        "Start from an example", [_PRESET_NONE, *PATIENT_PRESETS], key="pr_example", on_change=_on_example,
        help="Fills the form with a synthetic example you can then change.",
    )
    with st.form("patient_risk_form", border=False):
        spec = {row[0]: row for row in PATIENT_INPUTS}

        def number(key: str):
            _k, label, lo, hi, _default, step, help_text = spec[key]
            is_float = isinstance(step, float)
            return st.number_input(
                label, min_value=float(lo) if is_float else int(lo), max_value=float(hi) if is_float else int(hi),
                step=step if is_float else int(step), key=f"pr_{key}", help=help_text,
                format="%.1f" if is_float else "%d",
            )

        values: dict = {}
        ui.md('<div class="mt-subhead">Patient background</div>')
        c = st.columns(5, gap="medium")
        with c[0]:
            values["age"] = number("age")
        with c[1]:
            values["gender"] = st.selectbox("Gender", PATIENT_GENDERS, key="pr_gender")
        with c[2]:
            values["previous_admissions"] = number("previous_admissions")
        with c[3]:
            values["chronic_disease"] = st.selectbox(
                "Chronic disease", ["No", "Yes"], key="pr_chronic_disease",
                help="Any long-term condition such as diabetes or heart disease.")
        with c[4]:
            values["emergency_case"] = st.selectbox(
                "Arrived as an emergency", ["No", "Yes"], key="pr_emergency_case")
        ui.md('<div class="mt-subhead">Vital signs</div>')
        c = st.columns(5, gap="medium")
        for col, key in zip(c, ("heart_rate", "systolic_bp", "diastolic_bp", "temperature", "oxygen_saturation")):
            with col:
                values[key] = number(key)
        submitted = st.form_submit_button("Assess risk", key="pr_submit", type="primary", icon=":material/stethoscope:")
    return values if submitted else None


def _result(record_values: dict, outcome) -> None:
    if not outcome.ok:
        if outcome.error.startswith("PredictionServiceError"):
            detail = outcome.error.split(":", 1)[1].strip()
            ui.md(ui.callout(f"Please check the inputs: {detail}", "warn"))
        else:
            ui.md(ui.unavailable("Risk model", outcome.error))
        return
    res = outcome.value
    level, probs = res["level"], res["probabilities"]
    st.write("")
    left, right = st.columns([2, 3], gap="medium")
    with left, ui.card("risk-result"):
        ui.md(ui.section("Predicted risk level", ui.badge("ml")))
        ui.md(
            f'<div style="margin:.4rem 0 .6rem;display:flex;align-items:center;gap:.6rem">{ui.severity_pill(level)}'
            f'<span class="mt-note">risk group</span></div>'
            f'<div style="font-size:.95rem">{ui.esc(risk_explanation(level, probs))}</div>'
        )
        ui.md('<div class="mt-note" style="margin-top:.8rem">The probabilities show how confident the system is in each of the '
              "three groups. This is not a medical diagnosis.</div>")
        with st.expander("How this is calculated"):
            ui.md(f'<div class="mt-note">Model: {ui.esc(model_label(res["model"]))}, a classifier trained on synthetic patient '
                  "records whose risk groups come from fixed scoring rules.</div>")
    with right, ui.card("risk-probs"):
        ui.md(ui.section("Class probabilities"))
        ui.plot(charts.risk_probability_chart(probs), key="pr_probs")
    st.write("")
    with ui.card("risk-factors"):
        ui.md(ui.section("Warning signs in this record", ui.badge("rule")))
        factors = risk_factors(record_values)
        if factors:
            body = [[ui.esc(text), f"+{pts}"] for text, pts in factors]
            body.append([f"<b>Total score</b>", f"<b>{sum(p for _, p in factors)}</b>"])
            ui.md(ui.table([("Sign", False), ("Points", True)], body))
        else:
            ui.md('<div class="mt-note">None of the scoring rules fired for this record.</div>')
        ui.md(ui.how_to_read(
            "These points come from the fixed synthetic scoring rules that were used to create the training "
            "labels (7 or more is High, 4 to 6 is Medium). They explain the labels, not the model's inner workings; "
            "the model can differ from the rules, especially near the borders between groups."
        ))


def _model_card() -> None:
    m = backend.risk_metrics()
    with ui.card("risk-model-card"):
        ui.md(ui.section("How reliable is this model?"))
        if m.ok and m.value:
            v = m.value
            ui.md(
                '<div class="mt-note">On synthetic test records this model misses nearly half of the High-risk ones, so a '
                "Low or Medium result must never be read as safe. Its scores only describe agreement with synthetic rules.</div>"
            )
            with st.expander("How this is calculated"):
                ui.md(
                    f'<div class="mt-note">{ui.esc(model_label(v["model"]))} scored {v["accuracy"]:.1%} accuracy, '
                    f'macro F1 {v["f1_macro"]:.2f} and caught {v["recall_high"]:.0%} of High-risk records on '
                    f'{v["n_test"]} held-out synthetic records. Labels come from fixed scoring rules, so these figures '
                    "measure how well the rules were relearned, not clinical accuracy.</div>"
                )
        else:
            ui.md(ui.unavailable("Model metrics", m.error))


def render() -> None:
    ui.md(ui.page_header(
        "Patient risk",
        "Enter a synthetic patient record to see its expected risk group.",
        ui.badge("ml"), eyebrow="Look ahead",
    ))
    ui.md(ui.callout(DISCLAIMER, "warn"))
    st.write("")
    _init_defaults()
    with ui.card("risk-form"):
        ui.md(ui.section("Patient record"))
        values = _form()
    if values is not None:
        record = patient_record(values)
        st.session_state["pr_last"] = (values, backend.assess_patient(record))
    last = st.session_state.get("pr_last")
    st.write("")
    if last is not None:
        _result(*last)
    else:
        ui.md(ui.empty_state("No record assessed yet", "Fill in the form and press Assess risk."))
    st.write("")
    _model_card()
