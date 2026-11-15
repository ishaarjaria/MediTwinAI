"""Tests for the Patient Risk and Predictions views and their data helpers."""

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_synthetic_data import assign_risk_level, risk_score_breakdown
from src.ui import data as d

APP = str(ROOT / "app.py")


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


def _assess(at):
    at.button(key="pr_submit").click()
    at.run()
    return at


class TestPatientHelpers(unittest.TestCase):
    def test_record_building_adds_derived_fields(self):
        rec = d.patient_record(dict(PRESET := d.PATIENT_PRESETS["Typical, healthy"]))
        self.assertEqual(rec["pulse_pressure"], rec["systolic_bp"] - rec["diastolic_bp"])
        self.assertEqual((rec["chronic_disease_flag"], rec["emergency_case_flag"]), (0, 0))
        sick = d.patient_record(d.PATIENT_PRESETS["Several warning signs"])
        self.assertEqual((sick["chronic_disease_flag"], sick["emergency_case_flag"]), (1, 1))

    def test_form_defaults_lie_inside_the_allowed_ranges(self):
        for key, _label, lo, hi, default, step, _h in d.PATIENT_INPUTS:
            self.assertLessEqual(lo, default, key)
            self.assertLessEqual(default, hi, key)
            self.assertGreater(step, 0)

    def test_factor_points_sum_to_the_rule_label(self):
        for name, preset in d.PATIENT_PRESETS.items():
            total = sum(p for _, p in d.risk_factors(preset))
            expected = "High" if total >= 7 else "Medium" if total >= 4 else "Low"
            row = pd.Series({**preset})
            self.assertEqual(assign_risk_level(row), expected, name)

    def test_breakdown_is_empty_for_a_healthy_record(self):
        row = pd.Series(d.PATIENT_PRESETS["Typical, healthy"])
        self.assertEqual(risk_score_breakdown(row), [])

    def test_explanation_wording_tracks_confidence(self):
        self.assertIn("fairly sure", d.risk_explanation("Low", {"Low": 0.95, "Medium": 0.04, "High": 0.01}))
        self.assertIn("unsure", d.risk_explanation("Medium", {"Low": 0.3, "Medium": 0.4, "High": 0.3}))


class TestPatientRiskView(unittest.TestCase):
    def test_page_shows_disclaimer_form_and_model_card(self):
        at = run_app("patient_risk")
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("not a medical diagnosis", text)
        self.assertIn("Assess risk", " ".join(b.label for b in at.button))
        self.assertIn("How reliable is this model?", text)
        self.assertIn("No record assessed yet", text)

    def test_typical_and_severe_examples_give_low_and_high(self):
        for example, expected in (("Typical, healthy", "Low"), ("Several warning signs", "High")):
            with self.subTest(example=example):
                at = run_app("patient_risk")
                at.selectbox(key="pr_example").select(example).run()
                _assess(at)
                text = page_text(at)
                self.assertEqual(len(at.exception), 0)
                self.assertIn(f"Predicted {expected.lower()} risk", text)
                self.assertIn("Class probabilities", text)
                self.assertIn("not a medical diagnosis", text)

    def test_probabilities_are_shown_and_sum_to_one(self):
        from src.ui import backend

        res = backend.assess_patient(d.patient_record(d.PATIENT_PRESETS["In-between"]))
        self.assertTrue(res.ok)
        self.assertAlmostEqual(sum(res.value["probabilities"].values()), 1.0, places=5)
        self.assertEqual(set(res.value["probabilities"]), {"Low", "Medium", "High"})
        self.assertIn(res.value["level"], res.value["probabilities"])

    def test_invalid_blood_pressure_is_reported_politely(self):
        at = run_app("patient_risk")
        at.number_input(key="pr_systolic_bp").set_value(80)
        at.number_input(key="pr_diastolic_bp").set_value(90)
        _assess(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Please check the inputs", page_text(at))
        self.assertIn("systolic", page_text(at).lower())

    def test_backend_validation_rejects_bad_records(self):
        from src.ui import backend

        good = d.patient_record(d.PATIENT_PRESETS["Typical, healthy"])
        for bad in ({"age": 400}, {"gender": "Robot"}, {"oxygen_saturation": float("nan")}, {"temperature": 99}):
            with self.subTest(bad=bad):
                res = backend.assess_patient({**good, **bad})
                self.assertFalse(res.ok)
                self.assertTrue(res.error.startswith("PredictionServiceError"), res.error)
        missing = dict(good)
        missing.pop("heart_rate")
        self.assertFalse(backend.assess_patient(missing).ok)

    def test_model_failure_shows_unavailable_not_a_crash(self):
        from src.ui import backend

        bad = backend.Outcome(error="RuntimeError: injected failure")
        with mock.patch.object(backend, "assess_patient", return_value=bad):
            at = run_app("patient_risk")
            _assess(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("unavailable", page_text(at).lower())


class TestPredictionsView(unittest.TestCase):
    TABS = ("Outpatients", "Waiting time", "Resources", "How this is calculated")

    def all_tabs_text(self) -> str:
        """Tabs run lazily (only the open one), so read every tab by opening each in turn."""
        parts = []
        for tab in self.TABS:
            at = run_app("predictions", {"forecasts_tab": tab})
            self.assertEqual(len(at.exception), 0, tab)
            parts.append(page_text(at))
        return " ".join(parts)

    def test_sections_labels_and_baselines_are_present(self):
        text = self.all_tabs_text()
        for expected in (
            "Outpatient (OPD) forecast", "Waiting-time prediction", "Resource projections",
            "Model versus simple baselines", "Training mean", "Day-of-week mean", "Previous week, same day",
            "Estimated", "No trained model exists for beds",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("validated 80% interval.", text.replace("not close enough to call it a validated 80% interval.", ""))

    def test_interval_claim_matches_saved_coverage(self):
        import json

        meta = json.loads((ROOT / "models" / "opd_forecast" / "metrics.json").read_text(encoding="utf8"))
        at = run_app("predictions")
        text = page_text(at)
        self.assertIn(f"{meta['interval']['coverage_test']:.0%} of test days", text)

    def test_longer_horizon_adds_a_caution_and_more_rows(self):
        at = run_app("predictions", {"pred_horizon": 14})
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("Beyond 7 days", text)

    def test_waiting_time_table_has_a_like_for_like_today_row(self):
        text = page_text(run_app("predictions", {"forecasts_tab": "Waiting time"}))
        self.assertIn("same model as the future days", text)

    def test_failures_show_unavailable_states(self):
        from src.ui import backend

        bad = backend.Outcome(error="RuntimeError: injected failure")
        names = ["model_cards", "forecast", "predicted_state", "operations", "current_state",
                 "current_model_wait", "opd_model_info"]
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(backend, name, return_value=bad))
            at = run_app("predictions")
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("unavailable", page_text(at).lower())


class TestComparisonHelpers(unittest.TestCase):
    def test_forecast_rows_flag_the_selected_model_and_skips(self):
        meta = {
            "selected_model": "A",
            "comparison": [
                {"model": "A", "type": "model", "validation": {"mae": 1.0}, "test": {"mae": 2.0, "rmse": 3.0, "r2": 0.5}},
                {"model": "B", "type": "baseline", "validation": {"mae": 2.0}, "test": {"mae": 4.0, "rmse": 5.0, "r2": 0.1}},
                {"model": "Prophet", "type": "model", "skipped_reason": "blocked"},
            ],
        }
        rows = d.forecast_comparison_rows(meta)
        self.assertEqual([r["selected"] for r in rows], [True, False, False])
        self.assertIsNone(rows[2]["test_mae"])
        self.assertIn("blocked", rows[2]["note"])

    def test_waiting_rows_include_models_and_baselines(self):
        meta = {
            "selected": {"model": "m1"},
            "all_models": [{"model": "m1", "mae": 1, "rmse": 2, "r2": 0.9}],
            "baselines": {"training_mean": {"mae": 3, "rmse": 4, "r2": 0.0}},
        }
        rows = d.waiting_comparison_rows(meta)
        self.assertEqual({r["type"] for r in rows}, {"model", "baseline"})
        self.assertTrue(rows[0]["selected"])

    def test_model_cards_load_every_saved_file(self):
        cards = d.load_model_cards()
        for key in ("opd_forecast", "waiting_time", "patient_risk", "anomaly"):
            self.assertIsNotNone(cards[key], key)
        self.assertIsNone(d.load_model_cards(Path("does-not-exist"))["opd_forecast"])


if __name__ == "__main__":
    unittest.main()
