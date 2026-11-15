"""
End-to-end dashboard tests using Streamlit's AppTest (Phase 7).

The real app.py runs against the real twin, prediction service, simulator,
alert and recommendation engines. Backend failures are injected by patching
the dashboard's backend adapter, never the business logic.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from html import escape
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils import models_dir

_NEEDED = [
    models_dir() / "waiting_time" / "best_pipeline.joblib",
    models_dir() / "patient_risk" / "best_pipeline.joblib",
    models_dir() / "opd_forecast" / "forecaster.joblib",
    models_dir() / "anomaly_detection" / "isolation_forest.joblib",
    ROOT / "data" / "processed" / "hospital_operations_features.csv",
]

APP = str(ROOT / "app.py")
# Titles as they appear in rendered HTML (text is escaped, so "&" becomes "&amp;").
VIEWS = {
    "overview": escape("Hospital overview"),
    "analytics": escape("Trends"),
    "predictions": escape("Forecasts"),
    "patient_risk": escape("Patient risk"),
    "twin": escape("Hospital twin"),
    "whatif": escape("What-if scenarios"),
    "alerts": escape("Alerts"),
}


def run_app(view: str, session: dict | None = None):
    from streamlit.testing.v1 import AppTest

    from src.ui.auth import DEMO_USER

    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["view"] = view
    at.session_state["auth_user"] = DEMO_USER  # the workspace is a protected route
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


@unittest.skipUnless(all(p.is_file() for p in _NEEDED), "model artifacts or data missing")
class TestDashboardViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import streamlit as st

        from src.ui import backend

        st.cache_data.clear()
        cls.backend = backend
        cls.state = backend.current_state().value
        cls.latest = cls.state.as_of_date

    # ---- rendering ----------------------------------------------------
    def test_every_view_renders_without_errors_or_unavailable_states(self):
        for view, title in VIEWS.items():
            with self.subTest(view=view):
                at = run_app(view)
                self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
                text = page_text(at)
                self.assertIn(title, text)
                self.assertNotIn("is unavailable", text)

    def test_unknown_view_falls_back_to_the_overview(self):
        at = run_app("does-not-exist")
        self.assertEqual(at.session_state["view"], "overview")
        self.assertIn("Hospital overview", page_text(at))

    def test_disclaimer_is_visible_on_every_view(self):
        for view in VIEWS:
            with self.subTest(view=view):
                text = page_text(run_app(view))
                self.assertIn("Not a clinical decision system", text)
                self.assertIn("not a live feed", text.lower())

    # ---- redesigned content -------------------------------------------
    def test_overview_hero_explains_status_in_plain_language(self):
        from src.ui.data import status_headline

        text = page_text(run_app("overview"))
        self.assertIn(status_headline(self.state.capacity_status), text)
        self.assertIn("beds are in use", text)
        self.assertIn("Tomorrow&#x27;s outlook", text.replace("'", "&#x27;"))
        self.assertIn("30-day average", text)  # genuine trailing context, not an invented trend
        self.assertIn("How to read this.", text)

    def test_kpi_tiles_carry_source_labels_and_tooltips(self):
        text = page_text(run_app("overview"))
        for label in ("Recorded", "Expected", "Estimated"):
            self.assertIn(f">{label}<", text)
        self.assertGreaterEqual(text.count("mt-info"), 5)

    def test_analytics_risk_tab_states_the_labels_are_synthetic(self):
        text = page_text(run_app("analytics", {"trends_tab": "Patient risk"}))  # tabs run lazily: open it
        self.assertIn("not clinical validation", text)
        self.assertIn("Synthetic academic labels", text)

    def test_twin_explains_what_changed_in_plain_words(self):
        at = run_app("twin")
        text = page_text(at)
        self.assertIn("What changes, in plain words", text)
        self.assertIn("Estimated waiting time", text)
        labels = [e.label for e in at.expander]
        self.assertTrue(any(label.startswith("What do Recorded, Expected and Estimated mean") for label in labels), labels)

    def test_whatif_sliders_show_baseline_against_scenario(self):
        text = page_text(run_app("whatif"))
        self.assertIn(f"Same as baseline ({self.state.total_beds:,} beds)", text)
        self.assertIn("Unchanged from baseline", text)
        at = run_app("whatif", {self._wi("total_beds"): self.state.total_beds + 50})
        self.assertIn(f"Baseline {self.state.total_beds:,} → scenario {self.state.total_beds + 50:,} (+50 beds)", page_text(at))

    def test_alert_view_has_severity_summary_and_designed_all_clear(self):
        report, _recs = self.backend.operational_report(1).value
        text = page_text(run_app("alerts"))
        for label in ("Critical", "High", "Medium", "Low"):
            self.assertIn(f'class="l">{label}<', text)
        if not report.alerts:
            self.assertIn("What is being watched", text)
            self.assertIn("Headroom today", text)

    # ---- navigation ---------------------------------------------------
    def test_sidebar_navigation_switches_views(self):
        at = run_app("overview")
        self.assertEqual(len([b for b in at.sidebar.button if str(b.key).startswith("nav-")]), 7)
        for view, title in VIEWS.items():
            with self.subTest(view=view):
                at.sidebar.button(key=f"nav-{view}").click().run()
                self.assertEqual(at.session_state["view"], view)
                self.assertIn(title, page_text(at))
                self.assertEqual(len(at.exception), 0)

    def test_active_nav_item_is_marked_primary(self):
        at = run_app("twin")
        kinds = {b.key: b.proto.type for b in at.sidebar.button}
        self.assertEqual(kinds["nav-twin"], "primary")
        self.assertEqual(kinds["nav-overview"], "secondary")

    def test_overview_shortcut_opens_alerts(self):
        at = run_app("overview")
        at.button(key="ov_open_alerts").click().run()
        self.assertEqual(at.session_state["view"], "alerts")
        self.assertIn(VIEWS["alerts"], page_text(at))

    # ---- real backend values -----------------------------------------
    def test_overview_shows_the_twins_actual_numbers(self):
        text = page_text(run_app("overview"))
        s = self.state
        self.assertIn(f"{s.occupied_beds:,}", text)
        self.assertIn(f"/ {s.total_beds:,}", text)
        self.assertIn(f"{s.occupied_icu_beds:,}", text)
        self.assertIn(f"{s.current_opd:,}", text)
        self.assertIn(s.capacity_status, text)
        pred = self.backend.predicted_state(1).value
        self.assertIn(f"{pred.average_waiting_time:.1f}", text)  # next-day ML waiting time
        self.assertIn(f"{pred.current_opd:,}", text)  # forecast model

    def test_twin_view_matches_twin_and_labels_every_source(self):
        at = run_app("twin")
        text = page_text(at)
        pred = self.backend.predicted_state(1).value
        self.assertIn(pred.as_of_date[:4], text)
        for label in ("Recorded", "Expected", "Estimated"):
            self.assertIn(label, text)
        self.assertIn(f"{pred.average_waiting_time:.1f} min", text)

    def test_twin_horizon_changes_the_predicted_date(self):
        at = run_app("twin", {"twin_horizon": 3})
        pred3 = self.backend.predicted_state(3).value
        self.assertIn("3 days ahead", page_text(at))
        from src.ui.data import format_date

        self.assertIn(format_date(pred3.as_of_date), page_text(at))

    def test_untouched_data_shows_the_all_clear_state(self):
        report, _recs = self.backend.operational_report(1).value
        text = page_text(run_app("alerts"))
        if not report.alerts:
            self.assertIn("No active alerts", text)
            self.assertIn("Headroom today", text)
        self.assertIn("Advisory only", text)
        self.assertIn("not medical advice", text)

    # ---- what-if ------------------------------------------------------
    def _wi(self, field):
        return f"wi_{field}_{self.latest}"

    def test_whatif_no_change_shows_no_differences(self):
        at = run_app("whatif")
        text = page_text(at)
        self.assertIn("No changes selected", text)
        self.assertNotIn("▲", text)
        self.assertNotIn("▼", text)

    def test_whatif_controls_drive_the_existing_simulator(self):
        import streamlit as st

        from src.simulation.what_if import ScenarioChanges, WhatIfSimulator
        from src.ui.data import fmt_num

        st.cache_data.clear()
        real = WhatIfSimulator.simulate
        with mock.patch.object(WhatIfSimulator, "simulate", autospec=True, side_effect=real) as spy:
            at = run_app("whatif", {self._wi("emergency_cases_percent"): 55})
            self.assertEqual(len(at.exception), 0)
        self.assertTrue(spy.called, "UI must call the Phase 5 simulator")
        changes = next(c.args[2] for c in spy.call_args_list if isinstance(c.args[2], ScenarioChanges))
        self.assertEqual(changes.emergency_cases_percent, 55.0)
        self.assertIsNone(changes.total_beds)  # untouched capacity sliders stay None

        expected = WhatIfSimulator(self.backend._service()).simulate(
            self.state, ScenarioChanges(emergency_cases_percent=55)
        )
        text = page_text(at)
        load = next(c for c in expected.comparisons if c.name == "patient_load")
        wait = next(c for c in expected.comparisons if c.name == "expected_waiting_time_minutes")
        self.assertIn(f"{fmt_num(load.simulated)} patients", text)
        self.assertIn(f"{wait.simulated:.1f} min", text)
        self.assertIn("▲", text)  # an unfavourable increase is marked

    def test_whatif_slider_changes_recalculate_on_rerun(self):
        at = run_app("whatif")
        self.assertIn("No changes selected", page_text(at))
        at.slider(key=self._wi("doctors_available")).set_value(self.state.doctors_available + 10).run()
        text = page_text(at)
        self.assertNotIn("No changes selected", text)
        self.assertIn("Doctors available set to", text)
        self.assertIn("▼", text)  # staff pressure and waiting time fall

    def test_whatif_reset_returns_to_the_baseline(self):
        at = run_app("whatif")
        at.slider(key=self._wi("patient_inflow_percent")).set_value(40).run()
        self.assertNotIn("No changes selected", page_text(at))
        at.button(key=f"wi_reset_{self.latest}").click().run()
        self.assertIn("No changes selected", page_text(at))

    def test_whatif_shows_scenario_alerts_from_the_phase6_engine(self):
        at = run_app("whatif", {self._wi("icu_capacity"): 4})
        text = page_text(at)
        self.assertIn("Alerts under this scenario", text)
        self.assertIn("ICU pressure", text)  # 14+ occupied in a 4-bed ICU is capped and critical
        self.assertIn("Critical", text)
        self.assertIn("ICU demand exceeds capacity", text)

    def test_whatif_baseline_day_can_be_changed(self):
        at = run_app("whatif")
        iso_dates = [d.strftime("%Y-%m-%d") for d in reversed(list(self.backend.operations().value["date"]))]
        at.selectbox(key="wi_date").select_index(40).run()  # options are newest-first
        self.assertEqual(len(at.exception), 0)
        self.assertIn("No changes selected", page_text(at))
        self.assertEqual(at.slider(key=f"wi_total_beds_{iso_dates[40]}").value, self.state.total_beds)

    # ---- alerts view with real alerts --------------------------------
    def _stressed_report(self):
        from src.recommendations.integration import generate_operational_report
        from src.twin.hospital_twin import build_twin_from_latest

        twin = build_twin_from_latest(prediction_service=self.backend._service())
        twin.update_state(occupied_beds=240, occupied_icu_beds=29, current_opd=520)
        return generate_operational_report(twin, days_ahead=1, reference=self.backend.reference_stats())

    def test_alerts_view_lists_real_alerts_with_linked_recommendations(self):
        report, recs = self._stressed_report()
        self.assertTrue(report.alerts)
        out = self.backend.Outcome(value=(report, recs))
        with mock.patch.object(self.backend, "operational_report", return_value=out):
            at = run_app("alerts")
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        for alert in report.alerts:
            self.assertIn(alert.title, text)
        self.assertIn(f"{len(report.alerts)} active", text)
        self.assertIn("Review bed allocation", " ".join(e.label for e in at.expander) + text)
        self.assertGreaterEqual(len(at.expander), len(report.alerts))

    def test_alert_severity_filter_narrows_the_list(self):
        report, recs = self._stressed_report()
        out = self.backend.Outcome(value=(report, recs))
        with mock.patch.object(self.backend, "operational_report", return_value=out):
            at = run_app("alerts", {"al_sev": ["Critical"]})
        n_crit = sum(1 for a in report.alerts if a.severity.label == "Critical")
        text = page_text(at)
        self.assertIn(f"Showing {n_crit} of {len(report.alerts)}", text)
        # Alert cards only: the Recommendations tab lists every alert by design.
        cards = [f'mt-alert-title">{escape(a.title)}' for a in report.alerts]
        for alert, marker in zip(report.alerts, cards):
            if alert.severity.label == "Critical":
                self.assertIn(marker, text)
            else:
                self.assertNotIn(marker, text)

    def test_alert_filters_with_no_match_show_an_empty_state(self):
        from src.alerts.models import AlertReport

        report, recs = self._stressed_report()
        critical_only = AlertReport(
            alerts=[a for a in report.alerts if a.severity.label == "Critical"],
            unavailable=list(report.unavailable),
            generated_for=report.generated_for,
        )
        self.assertTrue(critical_only.alerts)
        out = self.backend.Outcome(value=(critical_only, recs))
        with mock.patch.object(self.backend, "operational_report", return_value=out):
            at = run_app("alerts", {"al_sev": ["Low"]})  # none of the alerts are Low
        self.assertIn("No alerts match these filters", page_text(at))
        self.assertEqual(len(at.exception), 0)

    # ---- failure handling --------------------------------------------
    def test_backend_failures_show_unavailable_states_not_crashes(self):
        bad = self.backend.Outcome(error="RuntimeError: injected failure")
        names = [
            "operations", "current_state", "predicted_state", "forecast", "operational_report",
            "anomaly_scores", "patients", "risk_metrics", "baseline_state", "run_whatif", "scenario_alerts",
            "model_cards", "assess_patient", "opd_model_info", "current_model_wait",
        ]
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(self.backend, name, return_value=bad))
            for view, title in VIEWS.items():
                with self.subTest(view=view):
                    at = run_app(view)
                    self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
                    self.assertIn("unavailable", page_text(at).lower())
                    self.assertIn(title, page_text(at))

    def test_partial_failures_degrade_gracefully(self):
        bad = self.backend.Outcome(error="RuntimeError: injected failure")
        # Forecast and alerts down: overview must still show the twin's observed numbers.
        with mock.patch.object(self.backend, "forecast", return_value=bad), \
             mock.patch.object(self.backend, "predicted_state", return_value=bad), \
             mock.patch.object(self.backend, "operational_report", return_value=bad):
            at = run_app("overview")
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn(f"{self.state.occupied_beds:,}", text)
        self.assertIn("Forecast unavailable", text)
        self.assertIn("Alerts is unavailable", text)

    def test_whatif_simulator_error_is_explained(self):
        bad = self.backend.Outcome(error="ScenarioError: Total bed capacity must be greater than 0.")
        with mock.patch.object(self.backend, "run_whatif", return_value=bad):
            at = run_app("whatif")
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("could not be simulated", text)
        self.assertIn("Total bed capacity must be greater than 0", text)

    def test_a_crashing_view_is_contained(self):
        with mock.patch("src.ui.views.overview.render", side_effect=RuntimeError("boom")):
            at = run_app("overview")
        self.assertEqual(len(at.exception), 0)
        self.assertIn("This view is unavailable", page_text(at))
        self.assertEqual(len([b for b in at.sidebar.button if str(b.key).startswith("nav-")]), 7)  # navigation still works


class TestBackendAdapter(unittest.TestCase):
    def test_outcome_semantics(self):
        from src.ui.backend import Outcome, _safe

        self.assertTrue(Outcome(value=1).ok)
        self.assertFalse(Outcome(error="x").ok)
        self.assertEqual(_safe(lambda: 5).value, 5)
        failed = _safe(lambda: 1 / 0)
        self.assertFalse(failed.ok)
        self.assertIn("ZeroDivisionError", failed.error)


if __name__ == "__main__":
    unittest.main()
