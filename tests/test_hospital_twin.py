"""
Unit tests for MediTwin digital-twin core state calculations (Phase 4).

These tests do not require trained ML models.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow `python -m unittest` from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.hospital_twin import (
    HospitalDigitalTwin,
    HospitalState,
    compute_available_beds,
    compute_capacity_status,
    compute_occupancy_percentage,
    hospital_state_from_row,
)


class TestOccupancyCalculations(unittest.TestCase):
    def test_occupancy_percentage_basic(self):
        self.assertEqual(compute_occupancy_percentage(125, 250), 50.0)

    def test_occupancy_full(self):
        self.assertEqual(compute_occupancy_percentage(30, 30), 100.0)

    def test_occupancy_rejects_over_capacity(self):
        with self.assertRaises(ValueError):
            compute_occupancy_percentage(31, 30)

    def test_occupancy_rejects_zero_total(self):
        with self.assertRaises(ValueError):
            compute_occupancy_percentage(0, 0)

    def test_available_beds(self):
        self.assertEqual(compute_available_beds(250, 180), 70)

    def test_available_beds_rejects_inconsistent(self):
        with self.assertRaises(ValueError):
            compute_available_beds(100, 101)


class TestCapacityStatus(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(compute_capacity_status(50, 40), "Normal")

    def test_moderate(self):
        self.assertEqual(compute_capacity_status(75, 40), "Moderate")

    def test_high(self):
        self.assertEqual(compute_capacity_status(90, 40), "High")

    def test_critical_from_icu(self):
        self.assertEqual(compute_capacity_status(50, 96), "Critical")

    def test_waiting_time_escalates_to_high(self):
        # Occupancy alone would be Normal; long wait escalates.
        self.assertEqual(
            compute_capacity_status(50, 40, average_waiting_time=65),
            "High",
        )

    def test_waiting_time_escalates_to_critical(self):
        self.assertEqual(
            compute_capacity_status(50, 40, average_waiting_time=130),
            "Critical",
        )

    def test_invalid_percentage(self):
        with self.assertRaises(ValueError):
            compute_capacity_status(-1, 50)


class TestHospitalStateFromRow(unittest.TestCase):
    def test_from_row_recomputes_derived(self):
        row = {
            "date": "2024-06-01",
            "total_beds": 250,
            "occupied_beds": 200,
            "available_beds": 999,  # intentionally wrong; must be recomputed
            "bed_occupancy_percentage": 0.0,
            "total_icu_beds": 30,
            "occupied_icu_beds": 15,
            "icu_occupancy_percentage": 0.0,
            "opd_patients": 280,
            "emergency_patients": 40,
            "doctors_available": 40,
            "nurses_available": 100,
            "average_waiting_time_minutes": 25.0,
            "medicine_stock_units": 40000,
            "admissions": 20,
            "discharges": 18,
            "is_weekend": 0,
            "is_holiday": 0,
            "day_of_week": 5,
            "month": 6,
        }
        state = hospital_state_from_row(row)
        self.assertEqual(state.available_beds, 50)
        self.assertEqual(state.bed_occupancy_percentage, 80.0)
        self.assertEqual(state.icu_occupancy_percentage, 50.0)
        self.assertEqual(state.current_opd, 280)
        self.assertEqual(state.emergency_cases, 40)
        self.assertEqual(state.capacity_status, "Moderate")


class _DummyService:
    """Minimal stub so twin unit tests do not load Joblib models."""

    is_ready = True


class TestTwinUpdateState(unittest.TestCase):
    def _make_twin(self) -> HospitalDigitalTwin:
        initial = HospitalState(
            as_of_date="2024-12-31",
            total_beds=250,
            occupied_beds=100,
            available_beds=150,
            bed_occupancy_percentage=40.0,
            total_icu_beds=30,
            occupied_icu_beds=10,
            icu_occupancy_percentage=33.33,
            current_opd=200,
            emergency_cases=40,
            doctors_available=45,
            nurses_available=120,
            average_waiting_time=20.0,
            medicine_stock=50000,
        )
        return HospitalDigitalTwin(
            initial_state=initial,
            prediction_service=_DummyService(),  # type: ignore[arg-type]
        )

    def test_get_current_state_copy(self):
        twin = self._make_twin()
        s1 = twin.get_current_state()
        s1.occupied_beds = 249
        self.assertEqual(twin.get_current_state().occupied_beds, 100)

    def test_update_state_recomputes_capacity(self):
        twin = self._make_twin()
        updated = twin.update_state(occupied_beds=240, occupied_icu_beds=29)
        self.assertEqual(updated.available_beds, 10)
        self.assertEqual(updated.bed_occupancy_percentage, 96.0)
        self.assertAlmostEqual(updated.icu_occupancy_percentage, 96.67, places=2)
        self.assertEqual(updated.capacity_status, "Critical")
        self.assertEqual(twin.get_capacity_status(), "Critical")

    def test_update_state_rejects_unknown_field(self):
        twin = self._make_twin()
        with self.assertRaises(ValueError):
            twin.update_state(not_a_real_field=1)


if __name__ == "__main__":
    unittest.main()
