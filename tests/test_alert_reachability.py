"""Alert reachability, medicine-stock scenarios and scenario-alert separation (real models)."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.alerts import AlertCategory, AlertEngine, SCOPE_PREDICTED, load_reference_stats
from src.alerts.integration import anomaly_row_for_state, generate_alerts
from src.models.prediction_service import get_prediction_service
from src.recommendations.engine import generate_recommendations
from src.simulation.what_if import ScenarioError, WhatIfSimulator
from src.twin.hospital_twin import build_twin_from_latest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def run_app(view, session=None):
    from streamlit.testing.v1 import AppTest

    from src.ui.auth import DEMO_USER

    at = AppTest.from_file(APP, default_timeout=240)
    at.query_params["view"] = view
    at.session_state["auth_user"] = DEMO_USER  # the workspace is a protected route
    for key, value in (session or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def page_text(at):
    return " ".join(m.value for m in at.markdown)


class ScenarioAlertBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = get_prediction_service()
        cls.twin = build_twin_from_latest(prediction_service=cls.svc)
        cls.base = cls.twin.get_current_state()
        cls.engine = AlertEngine(reference=load_reference_stats())
        cls.sim = WhatIfSimulator(prediction_service=cls.svc)

    def scenario(self, **changes):
        result = self.sim.simulate(self.base, **changes)
        state = result.simulated_state
        row, _ = anomaly_row_for_state(self.svc, state)
        report = self.engine.evaluate(
            state, scope=SCOPE_PREDICTED, anomaly=row, compare_to=result.baseline, compare_kind="scenario"
        )
        return result, report

    @staticmethod
    def cats(report):
        return {a.category for a in report.alerts}


class TestPressureLevels(ScenarioAlertBase):
    def test_normal_scenario_raises_nothing(self):
        result, report = self.scenario()
        self.assertEqual(report.alerts, [])
        self.assertEqual(report.unavailable, [])
        self.assertEqual(max(abs(c.simulated - c.current) for c in result.comparisons), 0.0)

    def test_baseline_day_itself_is_not_in_alert(self):
        self.assertEqual(self.base.capacity_status, "Normal")
        self.assertEqual([a for a in generate_alerts(self.twin, days_ahead=None).alerts], [])

    def test_medium_pressure_scenario(self):
        _, report = self.scenario(emergency_cases_percent=50)
        top = max(SEVERITY_RANK[a.severity.label] for a in report.alerts)
        self.assertTrue(report.alerts)
        self.assertLessEqual(top, SEVERITY_RANK["High"])
        self.assertIn(AlertCategory.BED_OCCUPANCY, self.cats(report))

    def test_critical_pressure_scenario(self):
        _, report = self.scenario(patient_inflow_percent=100, emergency_cases_percent=100, total_beds=150)
        self.assertEqual(max(a.severity.label for a in report.alerts if a.severity.label == "Critical"), "Critical")
        self.assertGreaterEqual(len(report.alerts), 6)


class TestMedicineStock(ScenarioAlertBase):
    def test_no_change_keeps_stock_and_cover_identical(self):
        result, _ = self.scenario()
        by = {c.name: c for c in result.comparisons}
        self.assertEqual(by["medicine_stock_units"].current, by["medicine_stock_units"].simulated)
        self.assertEqual(by["medicine_days_of_cover"].current, by["medicine_days_of_cover"].simulated)

    def test_stock_cut_raises_medicine_stock_alert_with_rising_severity(self):
        levels = []
        for pct in (-40, -60, -80):
            result, report = self.scenario(medicine_stock_percent=pct)
            alerts = [a for a in report.alerts if a.category == AlertCategory.MEDICINE_STOCK]
            self.assertEqual(len(alerts), 1, pct)
            levels.append(SEVERITY_RANK[alerts[0].severity.label])
            self.assertEqual(alerts[0].basis, "rule_derived")
            self.assertEqual(self.cats(report), {AlertCategory.MEDICINE_STOCK})  # nothing else disturbed
        self.assertEqual(levels, sorted(levels))
        self.assertGreater(levels[-1], levels[0])

    def test_stock_increase_never_raises_a_stock_alert(self):
        _, report = self.scenario(medicine_stock_percent=50)
        self.assertNotIn(AlertCategory.MEDICINE_STOCK, self.cats(report))

    def test_stock_value_scales_the_starting_stock(self):
        result, _ = self.scenario(medicine_stock_percent=-50)
        stock = {c.name: c for c in result.comparisons}["medicine_stock_units"]
        self.assertAlmostEqual(stock.simulated, self.base.medicine_stock * 0.5, delta=1)

    def test_invalid_stock_changes_are_rejected(self):
        for bad in (-101, float("nan"), float("inf"), True, "10"):
            with self.subTest(bad=bad):
                with self.assertRaises(ScenarioError):
                    self.sim.simulate(self.base, medicine_stock_percent=bad)

    def test_recommendation_links_to_the_stock_alert(self):
        _, report = self.scenario(medicine_stock_percent=-80)
        recs = generate_recommendations(report)
        stock = next(a for a in report.alerts if a.category == AlertCategory.MEDICINE_STOCK)
        linked = [r for r in recs if stock.alert_id in r.related_alert_ids]
        self.assertTrue(linked)
        self.assertTrue(any("stock" in r.action.lower() for r in linked))


class TestRisingDemand(ScenarioAlertBase):
    def test_inflow_increase_raises_rising_demand_labelled_as_scenario(self):
        _, report = self.scenario(patient_inflow_percent=50)
        rising = [a for a in report.alerts if a.category == AlertCategory.OPD_RISING]
        self.assertEqual(len(rising), 1)
        a = rising[0]
        self.assertEqual(a.basis, "rule_derived")
        self.assertIn("scenario", a.explanation.lower())
        self.assertNotIn("forecast is an ML prediction", a.explanation)
        self.assertGreater(a.observed_value, 25)

    def test_no_rise_means_no_rising_alert(self):
        _, report = self.scenario(patient_inflow_percent=-20)
        self.assertNotIn(AlertCategory.OPD_RISING, self.cats(report))

    def test_real_forecast_rising_alert_still_ml_based(self):
        twin = build_twin_from_latest(prediction_service=self.svc)
        twin.update_state(current_opd=100)
        report = generate_alerts(twin, days_ahead=1)
        rising = [a for a in report.alerts if a.category == AlertCategory.OPD_RISING]
        self.assertTrue(rising)
        self.assertEqual(rising[0].basis, "ml_prediction")


class TestAllCategoriesReachable(ScenarioAlertBase):
    def test_every_alert_category_can_be_raised_from_what_if_controls(self):
        seen = set()
        for changes in (
            dict(patient_inflow_percent=100),
            dict(emergency_cases_percent=100),
            dict(emergency_cases_percent=200),
            dict(medicine_stock_percent=-60),
            dict(total_beds=150),
            dict(icu_capacity=10),
            dict(doctors_available=10, nurses_available=30),
            dict(patient_inflow_percent=100, emergency_cases_percent=100, total_beds=150),
        ):
            _, report = self.scenario(**changes)
            seen |= self.cats(report)
        self.assertEqual(seen, set(AlertCategory))

    def test_dedup_and_recommendation_linking_hold_under_stress(self):
        _, report = self.scenario(patient_inflow_percent=100, emergency_cases_percent=100, total_beds=150)
        keys = [a.dedup_key for a in report.alerts]
        self.assertEqual(len(keys), len(set(keys)))
        ids = {a.alert_id for a in report.alerts}
        recs = generate_recommendations(report)
        self.assertTrue(recs)
        for r in recs:
            self.assertTrue(set(r.related_alert_ids) <= ids)
        for a in report.alerts:
            self.assertTrue(math.isfinite(a.observed_value))


class TestScenarioAlertsInTheApp(unittest.TestCase):
    def _controls(self, **kw):
        from src.ui.data import WhatIfControls
        from src.ui import backend

        base = backend.baseline_state(backend.current_state().value.as_of_date).value
        defaults = dict(total_beds=base.total_beds, icu_capacity=base.total_icu_beds,
                        doctors_available=base.doctors_available, nurses_available=base.nurses_available)
        return base.as_of_date, WhatIfControls(**{**defaults, **kw})

    def test_whatif_has_a_medicine_stock_control_that_creates_a_stock_alert(self):
        from src.ui import backend

        date_iso, _ = self._controls()
        at = run_app("whatif")
        slider = at.slider(key=f"wi_medicine_stock_percent_{date_iso}")
        slider.set_value(-60).run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("medicine stock covers", page_text(at))
        self.assertIn("Days of medicine cover", page_text(at))
        self.assertIsNotNone(at.session_state["wi_last_scenario"])

    def test_quick_scenario_for_delivery_delay(self):
        at = run_app("whatif")
        date_iso = at.session_state["wi_date"]
        at.session_state[f"wi_preset_{date_iso}"] = "Delivery delay (stock −60%)"
        at.run()
        self.assertEqual(len(at.exception), 0)

    def test_alerts_page_separates_scenario_alerts_from_real_alerts(self):
        from src.ui import backend

        date_iso, controls = self._controls(medicine_stock_percent=-60.0)
        at = run_app("alerts", {"wi_last_scenario": (date_iso, controls)})
        text = page_text(at)
        self.assertEqual(len(at.exception), 0)
        self.assertIn("What-if scenario alerts", text)
        self.assertIn("hypothetical scenario", text)
        self.assertIn("medicine stock covers", text)
        # The real-state counters do not include scenario alerts.
        real = backend.operational_report(1).value[0]
        self.assertIn(f"{len(real.alerts)} active", text)

    def test_alerts_page_without_scenario_explains_how_to_create_one(self):
        text = page_text(run_app("alerts"))
        self.assertIn("No scenario is active", text)

    def test_clearing_the_scenario_removes_it(self):
        date_iso, controls = self._controls(medicine_stock_percent=-60.0)
        at = run_app("alerts", {"wi_last_scenario": (date_iso, controls)})
        at.button(key="al_clear_scenario").click()
        at.run()
        self.assertIn("No scenario is active", page_text(at))

    def test_unchanged_whatif_does_not_leave_a_scenario_behind(self):
        at = run_app("whatif")
        self.assertNotIn("wi_last_scenario", at.session_state)


if __name__ == "__main__":
    unittest.main()
