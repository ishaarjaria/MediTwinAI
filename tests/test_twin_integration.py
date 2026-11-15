"""Digital-twin integration tests (real service, real saved models and data)."""

from __future__ import annotations

import unittest
from dataclasses import fields

import pandas as pd

from src.models.prediction_service import get_prediction_service
from src.simulation.what_if import WhatIfSimulator
from src.twin.hospital_twin import (
    CURRENT_FIELD_SOURCES,
    PREDICTED_FIELD_SOURCES,
    HospitalState,
    build_twin_from_latest,
    waiting_time_features,
)
from src.ui import data as ui_data

HORIZONS = (1, 2, 3, 5, 7)
_SOURCE_TO_UI = {"observed": "observed", "ml_prediction": "ml", "rule_derived": "rule"}


class TestTwinPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = get_prediction_service()
        cls.twin = build_twin_from_latest(prediction_service=cls.service)
        cls.cur = cls.twin.get_current_state()

    def test_current_state_is_the_latest_dataset_day_and_consistent(self):
        ops = pd.read_csv("data/processed/hospital_operations_features.csv", parse_dates=["date"])
        last = ops.sort_values("date").iloc[-1]
        self.assertEqual(self.cur.as_of_date, str(last["date"].date()))
        self.assertEqual(self.cur.occupied_beds, int(last["occupied_beds"]))
        self.assertEqual(self.cur.medicine_stock, int(last["medicine_stock_units"]))
        self.assertEqual(self.cur.occupied_beds + self.cur.available_beds, self.cur.total_beds)

    def test_predicted_states_for_required_horizons(self):
        forecast = self.service.forecast_opd(days=7)
        previous_date = pd.Timestamp(self.cur.as_of_date)
        for h in HORIZONS:
            with self.subTest(horizon=h):
                p = self.twin.get_predicted_state(days_ahead=h)
                self.assertEqual(pd.Timestamp(p.as_of_date), previous_date + pd.Timedelta(days=h))
                self.assertEqual(p.current_opd, int(round(float(forecast["opd_forecast"].iloc[h - 1]))))
                self.assertLessEqual(p.occupied_beds, p.total_beds)
                self.assertLessEqual(p.occupied_icu_beds, p.total_icu_beds)
                self.assertGreaterEqual(p.available_beds, 0)
                self.assertGreaterEqual(p.medicine_stock, 0)
                self.assertGreater(p.average_waiting_time, 0)
                self.assertEqual(p.doctors_available, self.cur.doctors_available)  # carried forward
                self.assertEqual(p.nurses_available, self.cur.nurses_available)
                self.assertIn(p.capacity_status, ("Normal", "Moderate", "High", "Critical"))

    def test_weekday_pattern_is_reflected_in_the_forecast(self):
        week = {h: self.twin.get_predicted_state(days_ahead=h) for h in range(1, 8)}
        weekend = [p.current_opd for p in week.values() if p.is_weekend]
        weekday = [p.current_opd for p in week.values() if not p.is_weekend]
        self.assertTrue(weekend and weekday)
        self.assertLess(max(weekend), min(weekday))

    def test_waiting_time_baseline_and_prediction_use_the_same_model(self):
        # Today's model estimate equals the model applied to today's own inputs...
        c = self.cur
        direct = self.service.predict_waiting_time(
            waiting_time_features(
                opd=c.current_opd, emergency=c.emergency_cases, doctors=c.doctors_available,
                nurses=c.nurses_available, is_weekend=c.is_weekend, is_holiday=c.is_holiday,
                day_of_week=c.day_of_week, month=c.month,
                bed_occupancy=c.bed_occupancy_percentage, admissions=c.admissions,
            )
        ).iloc[0]["predicted_waiting_time_minutes"]
        self.assertAlmostEqual(self.twin.model_waiting_time(), float(direct), places=9)
        # ...and agrees with the What-If baseline (same like-for-like rule).
        base = WhatIfSimulator(prediction_service=self.service).simulate(c).baseline_predicted_waiting_time
        self.assertAlmostEqual(self.twin.model_waiting_time(), base, places=6)

    def test_identical_inputs_give_identical_waiting_time_regardless_of_path(self):
        # A twin whose state equals a predicted day's inputs must score identically.
        p = self.twin.get_predicted_state(days_ahead=2)
        self.assertAlmostEqual(self.twin.model_waiting_time(p), p.average_waiting_time, places=6)

    def test_compare_states_uses_model_baseline_not_observed_value(self):
        p = self.twin.get_predicted_state(days_ahead=1)
        model_today = self.twin.model_waiting_time()
        rows = {r["label"]: r for r in ui_data.compare_states(self.cur, p, model_today)}
        wait = rows["Average waiting time"]
        self.assertAlmostEqual(wait["current"], model_today)
        self.assertEqual(wait["prov_current"], "ml")
        self.assertAlmostEqual(wait["delta"], p.average_waiting_time - model_today)

    def test_compare_states_without_model_baseline_shows_no_unfair_change(self):
        p = self.twin.get_predicted_state(days_ahead=1)
        wait = {r["label"]: r for r in ui_data.compare_states(self.cur, p)}["Average waiting time"]
        self.assertIsNone(wait["delta"])

    def test_staff_pressure_is_finite_for_current_and_predicted(self):
        p = self.twin.get_predicted_state(days_ahead=3)
        for s in (self.cur, p):
            value = ui_data.state_staff_pressure(s)
            self.assertIsNotNone(value)
            self.assertGreater(value, 0)


class TestProvenanceDeclarations(unittest.TestCase):
    def test_every_state_value_has_a_declared_source(self):
        state_fields = {f.name for f in fields(HospitalState)}
        for name in CURRENT_FIELD_SOURCES:
            self.assertIn(name, state_fields)
        self.assertEqual(set(CURRENT_FIELD_SOURCES), set(PREDICTED_FIELD_SOURCES))

    def test_only_trained_models_are_labelled_ml(self):
        ml = {k for k, v in PREDICTED_FIELD_SOURCES.items() if v == "ml_prediction"}
        self.assertEqual(ml, {"current_opd", "average_waiting_time"})
        self.assertTrue(all(v == "observed" for v in CURRENT_FIELD_SOURCES.values()))

    def test_ui_labels_match_backend_declaration(self):
        for f in ui_data._FIELDS:
            self.assertEqual(f.prov_current, _SOURCE_TO_UI[CURRENT_FIELD_SOURCES[f.attr]], f.attr)
            self.assertEqual(f.prov_predicted, _SOURCE_TO_UI[PREDICTED_FIELD_SOURCES[f.attr]], f.attr)


if __name__ == "__main__":
    unittest.main()
