"""Public landing page: what MediTwin AI is, what it does, how it works and a look inside."""

from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui import landing_assets as art
from src.ui import router
from src.ui.pages import common

VALUES = [
    ("overview", "See the whole hospital at a glance",
     "Beds, ICU, outpatients, waiting time and staff workload in one calm view."),
    ("forecast", "Prepare for what is coming",
     "Expected outpatient demand and waiting times for the days ahead, with a likely range."),
    ("whatif", "Try decisions before making them",
     "Test a busier day or a late medicine delivery and see the effect side by side."),
]

FEATURES = [
    ("overview", "Hospital Operations Overview",
     "One status view of bed and ICU use, outpatients, waiting time and staff workload, with the last 30 days of trend."),
    ("risk", "Patient Risk Assessment",
     "Enter a synthetic patient record and see a Low, Medium or High risk group with a plain-language explanation. For learning only, never a diagnosis."),
    ("forecast", "OPD & Waiting Time Forecasts",
     "Expected outpatient numbers and waiting times for up to 30 days ahead, shown with a likely range."),
    ("twin", "Hospital Digital Twin",
     "A working copy of today's hospital that you can step forward one to seven days to see what is expected."),
    ("whatif", "What-If Scenario Planning",
     "Change patient inflow, emergency cases, beds, staff or medicine stock and compare the outcome with a normal day."),
    ("alerts", "Alerts & Recommendations",
     "Clear “needs attention” alerts with suggested actions, from bed pressure to low medicine cover to unusual activity."),
]

STEPS = [
    ("Understand hospital data",
     "Daily records of admissions, beds, staff and supplies are brought together and summarised in plain language."),
    ("Predict operational trends",
     "Trained models estimate outpatient demand, waiting times and unusual days. Simple, transparent rules estimate beds, ICU and supplies."),
    ("Plan better decisions",
     "Use alerts, recommendations and what-if scenarios to decide what to prepare for, before the day arrives."),
]

PREVIEWS = [
    ("Overview", "preview-overview.png", "Overview page of the MediTwin AI workspace", "overview"),
    ("Forecasts", "preview-forecasts.png", "Forecasts page showing expected outpatients with a likely range", "forecasts"),
    ("Hospital Twin", "preview-twin.png", "Hospital Twin page comparing today with the expected state", "hospital-twin"),
    ("What-If", "preview-whatif.png", "What-If page comparing a scenario with a normal day", "what-if"),
]


def _hero() -> None:
    left, right = st.columns([1.02, 1.0], gap="large", vertical_alignment="center")
    with left:
        ui.md(
            '<span class="lp-eyebrow">Educational prototype · Built on synthetic data</span>'
            '<h1 class="lp-h1">Hospital Operations, <em>Simplified</em> with AI</h1>'
            '<p class="lp-lead">MediTwin AI brings daily hospital data into one calm workspace. See how busy the hospital '
            'is today, what to expect in the coming days, and test decisions before making them.</p>'
        )
        primary, ghost, _ = st.columns([1.15, 1.15, 1.2])
        with primary, st.container(key="cta-primary"):
            if st.button("Get Started", key="hero_start", type="primary", width="stretch"):
                router.go(router.SIGNIN)
        with ghost:
            ui.md('<a class="lp-btn-ghost" href="#lp-preview" style="width:100%">View Demo</a>')
        ui.md('<div class="lp-trust"><span>Synthetic data only</span><span>No real patient information</span>'
              '<span>Plain-language explanations</span></div>')
    with right:
        ui.md(art.hospital_illustration())

    items = "".join(
        f'<div class="lp-value"><span class="ic">{art.icon(i, 26)}</span><div><b>{t}</b><span>{d}</span></div></div>'
        for i, t, d in VALUES
    )
    ui.md(f'<div id="lp-platform" class="lp-values">{items}</div>')


def _features() -> None:
    cards = "".join(
        f'<div class="lp-feature"><div class="ic">{art.icon(i)}</div><h3>{t}</h3><p>{d}</p></div>'
        for i, t, d in FEATURES
    )
    ui.md(
        '<div id="lp-features" class="lp-section"><div class="lp-kicker">Platform</div>'
        '<h2 class="lp-h2">Everything an administrator needs to plan the day</h2>'
        '<p class="lp-sub">Six connected tools, each written in plain language, so the numbers lead to a decision.</p>'
        f'<div class="lp-features">{cards}</div></div>'
    )


def _how_it_works() -> None:
    steps = "".join(
        f'<div class="lp-step"><div class="n">{n}</div><h3>{t}</h3><p>{d}</p></div>'
        for n, (t, d) in enumerate(STEPS, start=1)
    )
    with st.container(key="lp-howband"):  # a soft sage band to separate this section
        ui.md(
            '<div id="lp-how" class="lp-section"><div class="lp-kicker">How it works</div>'
            '<h2 class="lp-h2">From daily records to a clear plan</h2>'
            f'<div class="lp-steps">{steps}</div></div>'
        )


def _preview() -> None:
    with st.container(key="lp-band"):
        ui.md(
            '<div id="lp-preview" class="lp-kicker">Product preview</div>'
            '<h2 class="lp-h2">A look inside the workspace</h2>'
            '<p class="lp-sub">Screenshots of the working prototype. Everything shown runs on synthetic data.</p>'
        )
        tabs = st.tabs([label for label, *_ in PREVIEWS])
        for tab, (label, filename, alt, slug) in zip(tabs, PREVIEWS):
            with tab:
                src = common.preview_src(filename)
                if src:
                    ui.md(art.browser_frame(src, alt, f"meditwin · {slug}"))
                else:
                    ui.md(ui.empty_state("Preview not available", "Open the workspace to see this page."))
        ui.md('<div class="lp-caption">These are screenshots of the actual prototype. The values come from a generated '
              'dataset, not from a real hospital.</div>')
        _, middle, _ = st.columns([1, 1.4, 1])
        with middle:
            if st.button("Continue as demo user", key="preview_demo", type="primary", width="stretch"):
                common.demo_sign_in()


def _about() -> None:
    facts = "".join(
        f'<div class="lp-fact"><span class="ic">{art.icon(i, 22)}</span><div><b>{t}</b><span>{d}</span></div></div>'
        for i, t, d in (
            ("data", "Synthetic data only",
             "366 days of generated hospital operations and 1,000 generated patient records. No real patient is represented."),
            ("trend", "Numbers that explain themselves",
             "Each value is marked Recorded, Expected or Estimated, and “How this is calculated” explains the method and its limits."),
            ("shield", "Not for clinical use",
             "It supports learning about hospital operations. It does not diagnose, treat or triage anyone."),
        )
    )
    ui.md(
        '<div id="lp-about" class="lp-section"><div class="lp-about"><div>'
        '<div class="lp-kicker">About</div><h2 class="lp-h2">A university project on hospital operations</h2>'
        '<p class="lp-sub">MediTwin AI is an educational data science and AI prototype. It explores how a digital twin of a '
        'hospital could help administrators understand today’s workload and prepare for tomorrow.</p></div>'
        f'<div>{facts}</div></div>'
        '<div class="lp-disclaimer"><span class="ic">' + art.icon("shield", 22) + '</span><div><b>Educational prototype.</b> '
        'All data is synthetic. MediTwin AI is not a clinical decision system, does not diagnose or treat, and is not connected '
        'to any real hospital, sensor or patient record. Sign-in is a demonstration for presentations and is not production '
        'authentication.</div></div></div>'
    )


def _final_cta() -> None:
    with st.container(key="lp-final"):
        ui.md('<h2 class="lp-h2">Explore the MediTwin AI prototype</h2>'
              '<p class="lp-sub">Create a demo account or jump straight in. No real data or passwords are needed.</p>')
        _, a, b, _ = st.columns([1, 1.2, 1.5, 1])
        with a, st.container(key="cta-final"):
            if st.button("Get Started", key="final_start", type="primary", width="stretch"):
                router.go(router.SIGNIN)
        with b:
            if st.button("Continue as demo user", key="final_demo", width="stretch"):
                common.demo_sign_in()


def render() -> None:
    common.inject_styles()
    ui.md(f'<div class="lp-bg">{art.network_background()}</div>')
    common.nav()
    _hero()
    _features()
    _how_it_works()
    _preview()
    _about()
    _final_cta()
    common.footer()
