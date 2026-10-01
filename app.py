"""
MediTwin AI — Streamlit landing page (Phase 1).

This is a minimal educational prototype entry point.
The full operational dashboard is intentionally NOT implemented yet.
"""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
HOSPITAL_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "hospital_operations.csv"
PATIENT_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "patient_data.csv"


def data_files_exist() -> bool:
    """Return True if both Phase 1 synthetic CSV files are present."""
    return HOSPITAL_DATA_PATH.is_file() and PATIENT_DATA_PATH.is_file()


def main() -> None:
    st.set_page_config(
        page_title="MediTwin AI",
        page_icon="🏥",
        layout="centered",
    )

    st.title("MediTwin AI")
    st.subheader("Intelligent Hospital Digital Twin & Predictive Healthcare Platform")

    st.markdown(
        """
        **MediTwin AI** is a university Data Science + AI/ML project that aims to:

        - Represent the current operational state of a hospital (digital twin)
        - Predict future hospital conditions using machine learning
        - Detect unusual activity (anomaly detection)
        - Support what-if simulations
        - Generate risk alerts and recommendations
        - Visualise insights in a Streamlit dashboard

        This application is being built **phase by phase**. Phase 1 focuses on
        project setup and synthetic dataset generation only.
        """
    )

    st.warning(
        "**Educational prototype disclaimer:** MediTwin AI uses synthetic data "
        "for learning and demonstration only. It is **not** a real clinical "
        "decision system and must **not** be used for diagnosis, treatment, "
        "triage, or hospital operations."
    )

    st.markdown("---")
    st.header("Phase 1 status")

    if data_files_exist():
        st.success(
            "Phase 1 data has been generated.\n\n"
            f"- `{HOSPITAL_DATA_PATH.relative_to(PROJECT_ROOT)}`\n"
            f"- `{PATIENT_DATA_PATH.relative_to(PROJECT_ROOT)}`"
        )
    else:
        st.error(
            "Phase 1 synthetic data files were not found.\n\n"
            "From the `meditwin-ai` folder, run:\n\n"
            "`python -m src.data.generate_synthetic_data`"
        )

    st.info(
        "The full dashboard (EDA, forecasting, anomaly detection, twin view, "
        "and simulation) will be added in later phases."
    )


if __name__ == "__main__":
    main()
