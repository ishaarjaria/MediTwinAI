"""
MediTwin AI: hospital operations platform (educational prototype, synthetic data).

Run from the project root:

    streamlit run app.py

This file is only the router. The public landing page and the demo sign-in live in
``src/ui/pages``; the authenticated dashboard is ``src/ui/workspace.py``; all
calculation stays in the existing twin, simulation, alerts, recommendations and
prediction-service packages behind ``src/ui/backend.py``.
"""

from __future__ import annotations

import streamlit as st

from src.ui import router

st.set_page_config(
    page_title="MediTwin AI",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="auto",
)

router.run()
