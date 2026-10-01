"""
Light validation tests for PredictionService input checks (no full model I/O required
beyond construction — skipped gracefully if artifacts are missing).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prediction_service import PredictionServiceError
from src.models.utils import models_dir


def _artifacts_present() -> bool:
    root = models_dir()
    required = [
        root / "patient_risk" / "best_pipeline.joblib",
        root / "waiting_time" / "best_pipeline.joblib",
        root / "opd_forecast" / "prophet_model.joblib",
        root / "anomaly_detection" / "isolation_forest.joblib",
    ]
    return all(p.is_file() for p in required)


@unittest.skipUnless(_artifacts_present(), "Phase 3 model artifacts not found")
class TestPredictionServiceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models.prediction_service import get_prediction_service

        cls.service = get_prediction_service(reload=True)

    def test_rejects_bad_age(self):
        patient = {
            "age": 200,
            "gender": "Female",
            "heart_rate": 80,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "temperature": 36.8,
            "oxygen_saturation": 98,
            "previous_admissions": 0,
            "pulse_pressure": 40,
            "chronic_disease_flag": 0,
            "emergency_case_flag": 0,
        }
        with self.assertRaises(PredictionServiceError):
            self.service.predict_patient_risk(patient)

    def test_rejects_bad_forecast_horizon(self):
        with self.assertRaises(PredictionServiceError):
            self.service.forecast_opd(days=0)
        with self.assertRaises(PredictionServiceError):
            self.service.forecast_opd(days=100)

    def test_forecast_opd_ok(self):
        out = self.service.forecast_opd(days=7)
        self.assertEqual(len(out), 7)
        self.assertIn("opd_forecast", out.columns)


if __name__ == "__main__":
    unittest.main()
