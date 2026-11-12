"""
What-if simulation tests (Phase 5).

Waiting time is supplied by a tiny stub that copies the synthetic generator
slope (higher OPD and emergency raise wait; more doctors lower it). The stub
is not a trained model. The simulator still labels that call as
"ML-derived prediction" only because it came through predict_waiting_time.
Rule metrics are checked directly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.what_if import (
    SOURCE_ML,
    SOURCE_RULE,
    WhatIfSimulator,
    medicine_consumption,
)
from src.twin.hospital_twin import HospitalState


class StubWaitingModel:
    """Deterministic stand-in. Not the saved Phase 3 model."""

    def predict_waiting_time(self, days):
        row = days if isinstance(days, dict) else dict(days)
        wait = (
            8
            + 0.12 * float(row["opd_patients"])
            - 0.35 * float(row["doctors_available"])
            + 0.05 * float(row["emergency_patients"])
        )
        frame = pd.DataFrame(
            [{**row, "predicted_waiting_time_minutes": round(max(0.0, wait), 2)}]
        )
        return frame


def baseline_state() -> HospitalState:
    """A comfortable mid-size day: OPD 400, as in the phase brief."""
    return HospitalState(
        as_of_date="2024-06-15",
        total_beds=250,
        occupied_beds=150,
        available_beds=100,
        bed_occupancy_percentage=60.0,
        total_icu_beds=30,
        occupied_icu_beds=12,
        icu_occupancy_percentage=40.0,
        current_opd=400,
        emergency_cases=40,
        doctors_available=40,
        nurses_available=100,
        # Matches StubWaitingModel on this unchanged day, so ML deltas are comparable.
        average_waiting_time=44.0,
        medicine_stock=20000,
        admissions=30,
        discharges=28,
        is_weekend=0,
        is_holiday=0,
        day_of_week=5,
        month=6,
    )


class TestWhatIfScenarios(unittest.TestCase):
    def setUp(self):
        self.sim = WhatIfSimulator(prediction_service=StubWaitingModel())
        self.base = baseline_state()

    def _metric(self, result, name):
        matches = [item for item in result.comparisons if item.name == name]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_plus_20_percent_patient_inflow(self):
        result = self.sim.simulate(self.base, patient_inflow_percent=20)

        self.assertEqual(result.simulated_state.current_opd, 480)
        self.assertEqual(result.simulated_state.emergency_cases, 40)

        load = self._metric(result, "patient_load")
        self.assertEqual(load.source, SOURCE_RULE)
        self.assertGreater(load.simulated, load.current)

        beds = self._metric(result, "bed_requirement")
        self.assertEqual(beds.source, SOURCE_RULE)
        self.assertGreater(beds.simulated, beds.current)

        occupancy = self._metric(result, "bed_occupancy_percentage")
        self.assertGreaterEqual(occupancy.simulated, 0)
        self.assertLessEqual(occupancy.simulated, 100)
        self.assertLessEqual(
            result.simulated_state.available_beds, result.simulated_state.total_beds
        )
        self.assertGreaterEqual(result.simulated_state.available_beds, 0)

        wait = self._metric(result, "expected_waiting_time_minutes")
        self.assertEqual(wait.source, SOURCE_ML)
        self.assertGreater(wait.simulated, wait.current)

        medicine = self._metric(result, "medicine_consumption_units")
        self.assertEqual(medicine.source, SOURCE_RULE)
        self.assertGreater(medicine.simulated, medicine.current)
        self.assertIn("rule-derived simulation", medicine.explanation)
        self.assertIn("ML-derived prediction", wait.explanation)

        # OPD 400 -> 480 is exactly +20%.
        self.assertAlmostEqual(result.simulated_state.current_opd, 400 * 1.2)

    def test_reduced_icu_capacity(self):
        # 12 occupied ICU beds into a 10-bed unit: occupancy hits the cap.
        result = self.sim.simulate(self.base, icu_capacity=10)

        state = result.simulated_state
        self.assertEqual(state.total_icu_beds, 10)
        self.assertEqual(state.occupied_icu_beds, 10)
        self.assertEqual(state.icu_occupancy_percentage, 100.0)
        self.assertEqual(result.icu_shortfall, 2)
        self.assertGreaterEqual(state.icu_occupancy_percentage, 0)
        self.assertLessEqual(state.icu_occupancy_percentage, 100)

        pressure = self._metric(result, "icu_pressure_percentage")
        self.assertEqual(pressure.source, SOURCE_RULE)
        self.assertGreater(pressure.simulated, pressure.current)
        self.assertEqual(state.current_opd, self.base.current_opd)

    def test_additional_doctors(self):
        result = self.sim.simulate(self.base, doctors_available=60)

        self.assertEqual(result.simulated_state.doctors_available, 60)
        load = self._metric(result, "patient_load")
        self.assertEqual(load.simulated, load.current)

        pressure = self._metric(result, "staff_pressure_patients_per_staff")
        self.assertEqual(pressure.source, SOURCE_RULE)
        self.assertLess(pressure.simulated, pressure.current)

        wait = self._metric(result, "expected_waiting_time_minutes")
        self.assertEqual(wait.source, SOURCE_ML)
        self.assertLess(wait.simulated, wait.current)

    def test_emergency_surge(self):
        result = self.sim.simulate(self.base, emergency_cases_percent=100)

        self.assertEqual(result.simulated_state.emergency_cases, 80)
        self.assertEqual(result.simulated_state.current_opd, 400)

        load = self._metric(result, "patient_load")
        self.assertGreater(load.simulated, load.current)

        icu = self._metric(result, "icu_pressure_percentage")
        self.assertEqual(icu.source, SOURCE_RULE)
        self.assertGreater(icu.simulated, icu.current)
        self.assertLessEqual(result.simulated_state.occupied_icu_beds, 30)
        self.assertGreaterEqual(result.simulated_state.occupied_icu_beds, 0)
        self.assertLessEqual(result.simulated_state.icu_occupancy_percentage, 100)

        medicine = self._metric(result, "medicine_consumption_units")
        expected = medicine_consumption(400, 80, result.simulated_state.admissions)
        self.assertAlmostEqual(medicine.simulated, expected)

    def test_constraints_reject_bad_capacity(self):
        with self.assertRaises(ValueError):
            self.sim.simulate(self.base, total_beds=0)
        with self.assertRaises(ValueError):
            self.sim.simulate(self.base, patient_inflow_percent=-150)

    def test_summary_mentions_both_scenarios(self):
        result = self.sim.simulate(self.base, patient_inflow_percent=20)
        text = result.summary_text()
        self.assertIn("Current scenario vs simulated scenario", text)
        self.assertIn("Patient inflow", text)


if __name__ == "__main__":
    unittest.main()
