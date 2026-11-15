"""
What-if integration tests (Phase 5) using the REAL saved prediction service.

No stub is used here. The tests load the Joblib artifacts and the latest
processed hospital row, so they check that current and simulated scenarios are
compared fairly with the actual waiting-time model.

If the artifacts or processed data are missing the module is skipped and the
skip reason is printed by unittest.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.utils import models_dir
from src.simulation.what_if import (
    SOURCE_ML,
    SOURCE_RULE,
    ScenarioError,
    WhatIfSimulator,
)

# Documented numerical tolerance for "no change" (minutes). The model is
# deterministic and both sides use identical inputs, so the real difference is 0.
NO_CHANGE_TOLERANCE = 1e-6

_ARTIFACTS = [
    models_dir() / "waiting_time" / "best_pipeline.joblib",
    models_dir() / "patient_risk" / "best_pipeline.joblib",
    models_dir() / "opd_forecast" / "forecaster.joblib",
    models_dir() / "anomaly_detection" / "isolation_forest.joblib",
    ROOT / "data" / "processed" / "hospital_operations_features.csv",
]


@unittest.skipUnless(
    all(p.is_file() for p in _ARTIFACTS),
    "Phase 3 model artifacts or processed hospital data not found",
)
class TestWhatIfRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models.prediction_service import get_prediction_service
        from src.twin.hospital_twin import build_twin_from_latest

        cls.service = get_prediction_service()
        cls.base = build_twin_from_latest(
            prediction_service=cls.service
        ).get_current_state()
        cls.sim = WhatIfSimulator(prediction_service=cls.service)

    def _metric(self, result, name):
        matches = [m for m in result.comparisons if m.name == name]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def _wait(self, result):
        m = self._metric(result, "expected_waiting_time_minutes")
        self.assertEqual(m.source, SOURCE_ML)
        return m

    def _assert_sane(self, result):
        s = result.simulated_state
        self.assertGreaterEqual(s.available_beds, 0)
        self.assertLessEqual(s.occupied_beds, s.total_beds)
        self.assertLessEqual(s.occupied_icu_beds, s.total_icu_beds)
        self.assertTrue(0.0 <= s.bed_occupancy_percentage <= 100.0)
        self.assertTrue(0.0 <= s.icu_occupancy_percentage <= 100.0)
        for m in result.comparisons:
            self.assertTrue(math.isfinite(m.current), m.name)
            self.assertTrue(math.isfinite(m.simulated), m.name)

    # 1 -----------------------------------------------------------------
    def test_no_change_has_zero_difference_in_every_metric(self):
        result = self.sim.simulate(self.base)
        self._assert_sane(result)
        for m in result.comparisons:
            self.assertAlmostEqual(
                m.absolute_change, 0.0, delta=NO_CHANGE_TOLERANCE, msg=m.name
            )
        wait = self._wait(result)
        self.assertAlmostEqual(wait.current, wait.simulated, delta=NO_CHANGE_TOLERANCE)
        self.assertEqual(result.baseline_capacity_status, result.simulated_state.capacity_status)

    def test_baseline_wait_is_model_prediction_not_observed_value(self):
        result = self.sim.simulate(self.base)
        direct = self.service.predict_waiting_time(
            {
                "opd_patients": self.base.current_opd,
                "emergency_patients": self.base.emergency_cases,
                "doctors_available": self.base.doctors_available,
                "nurses_available": self.base.nurses_available,
                "is_weekend": self.base.is_weekend,
                "is_holiday": self.base.is_holiday,
                "day_of_week": self.base.day_of_week,
                "month": self.base.month,
                "bed_occupancy_percentage": self.base.bed_occupancy_percentage,
                "admissions": self.base.admissions,
            }
        ).iloc[0]["predicted_waiting_time_minutes"]
        self.assertAlmostEqual(
            result.baseline_predicted_waiting_time, float(direct), delta=NO_CHANGE_TOLERANCE
        )
        self.assertEqual(result.observed_waiting_time, self.base.average_waiting_time)

    # 2 -----------------------------------------------------------------
    def test_plus_20_percent_inflow(self):
        result = self.sim.simulate(self.base, patient_inflow_percent=20)
        self._assert_sane(result)
        self.assertEqual(
            result.simulated_state.current_opd, round(self.base.current_opd * 1.2)
        )
        load = self._metric(result, "patient_load")
        self.assertEqual(load.source, SOURCE_RULE)
        self.assertGreater(load.simulated, load.current)
        self.assertGreater(
            self._metric(result, "medicine_consumption_units").simulated,
            self._metric(result, "medicine_consumption_units").current,
        )
        self.assertGreater(
            self._metric(result, "staff_pressure_patients_per_staff").simulated,
            self._metric(result, "staff_pressure_patients_per_staff").current,
        )
        self.assertGreaterEqual(
            self._metric(result, "bed_requirement").simulated,
            self._metric(result, "bed_requirement").current,
        )
        wait = self._wait(result)
        self.assertGreater(wait.simulated, wait.current)

    # 3 -----------------------------------------------------------------
    def test_reduced_icu_capacity_raises_icu_pressure(self):
        smaller = max(1, self.base.total_icu_beds - 10)
        result = self.sim.simulate(self.base, icu_capacity=smaller)
        self._assert_sane(result)
        pressure = self._metric(result, "icu_pressure_percentage")
        self.assertEqual(pressure.source, SOURCE_RULE)
        self.assertGreater(pressure.simulated, pressure.current)
        # The waiting-time model has no ICU feature: no invented effect.
        self.assertAlmostEqual(
            self._wait(result).absolute_change, 0.0, delta=NO_CHANGE_TOLERANCE
        )

    def test_icu_capacity_below_demand_reports_shortfall_and_caps_at_100(self):
        result = self.sim.simulate(self.base, icu_capacity=1)
        self._assert_sane(result)
        self.assertEqual(result.simulated_state.icu_occupancy_percentage, 100.0)
        self.assertEqual(
            result.icu_shortfall, self.base.occupied_icu_beds - 1
        )

    # 4 -----------------------------------------------------------------
    def test_additional_doctors(self):
        result = self.sim.simulate(
            self.base, doctors_available=self.base.doctors_available + 10
        )
        self._assert_sane(result)
        pressure = self._metric(result, "staff_pressure_patients_per_staff")
        self.assertLess(pressure.simulated, pressure.current)
        wait = self._wait(result)
        # Fair comparison: more doctors must not look worse than no change.
        self.assertLess(wait.simulated, wait.current)

    # 5 -----------------------------------------------------------------
    def test_emergency_surge(self):
        result = self.sim.simulate(self.base, emergency_cases_percent=50)
        self._assert_sane(result)
        self.assertGreater(
            result.simulated_state.emergency_cases, self.base.emergency_cases
        )
        for name in (
            "patient_load",
            "icu_pressure_percentage",
            "medicine_consumption_units",
            "staff_pressure_patients_per_staff",
            "bed_requirement",
        ):
            m = self._metric(result, name)
            self.assertEqual(m.source, SOURCE_RULE, name)
            self.assertGreater(m.simulated, m.current, name)
        wait = self._wait(result)
        self.assertGreaterEqual(wait.simulated, wait.current)

    # 6 -----------------------------------------------------------------
    def test_increased_bed_capacity_never_creates_impossible_occupancy(self):
        result = self.sim.simulate(self.base, total_beds=self.base.total_beds + 50)
        self._assert_sane(result)
        self.assertEqual(result.simulated_state.total_beds, self.base.total_beds + 50)
        occ = self._metric(result, "bed_occupancy_percentage")
        self.assertLess(occ.simulated, occ.current)
        self.assertEqual(result.bed_shortfall, 0)
        self.assertEqual(result.simulated_state.occupied_beds, self.base.occupied_beds)
        # Demand unchanged by capacity alone.
        self.assertEqual(
            self._metric(result, "bed_requirement").absolute_change, 0.0
        )

    # 7 -----------------------------------------------------------------
    def test_increased_nurse_availability(self):
        result = self.sim.simulate(
            self.base, nurses_available=self.base.nurses_available + 20
        )
        self._assert_sane(result)
        self.assertEqual(
            result.simulated_state.nurses_available, self.base.nurses_available + 20
        )
        pressure = self._metric(result, "staff_pressure_patients_per_staff")
        self.assertLess(pressure.simulated, pressure.current)
        wait = self._wait(result)
        # The model's nurse response is an association, not a promise: only
        # require a finite, in-range prediction and the documented caveat.
        self.assertTrue(math.isfinite(wait.simulated))
        self.assertGreaterEqual(wait.simulated, 0.0)
        self.assertTrue(any("Nurse count" in note for note in result.ml_notes))

    # validation with the real model ------------------------------------
    def test_scenario_outside_model_range_raises_meaningful_error(self):
        with self.assertRaises(ScenarioError) as ctx:
            self.sim.simulate(self.base, patient_inflow_percent=5000)
        self.assertIn("supported input range", str(ctx.exception))

    def test_non_finite_and_invalid_inputs_rejected(self):
        for kwargs in (
            {"patient_inflow_percent": float("nan")},
            {"patient_inflow_percent": float("inf")},
            {"emergency_cases_percent": -101},
            {"total_beds": 0},
            {"icu_capacity": -3},
            {"doctors_available": -1},
            {"nurses_available": 2.5},
            {"total_beds": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ScenarioError):
                    self.sim.simulate(self.base, **kwargs)


if __name__ == "__main__":
    unittest.main()
