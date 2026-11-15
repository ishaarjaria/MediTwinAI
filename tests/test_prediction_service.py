"""
Light validation tests for PredictionService input checks (no full model I/O required
beyond construction — skipped gracefully if artifacts are missing).
"""

from __future__ import annotations

import sys
import unittest
import numpy as np
import pandas as pd
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
        root / "opd_forecast" / "forecaster.joblib",
        root / "anomaly_detection" / "isolation_forest.joblib",
    ]
    return all(p.is_file() for p in required)


@unittest.skipUnless(_artifacts_present(), "Phase 3 model artifacts not found")
class TestPredictionServiceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from src.models.prediction_service import get_prediction_service

        cls.service = get_prediction_service(reload=True)
        cls.patient = pd.read_csv(ROOT / "data" / "processed" / "patient_data_features.csv").iloc[[0]]
        cls.operation = pd.read_csv(ROOT / "data" / "processed" / "hospital_operations_features.csv").iloc[[-1]]

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

    def test_predict_patient_risk_returns_real_class_and_probabilities(self):
        out = self.service.predict_patient_risk(self.patient)
        self.assertEqual(len(out), 1)
        self.assertIn(out.iloc[0]["predicted_risk_level"], {"Low", "Medium", "High"})
        probabilities = out[["proba_Low", "proba_Medium", "proba_High"]].to_numpy()
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_predict_waiting_time_returns_finite_nonnegative_value(self):
        out = self.service.predict_waiting_time(self.operation)
        value = float(out.iloc[0]["predicted_waiting_time_minutes"])
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0)

    def test_detect_hospital_anomaly_returns_model_output(self):
        out = self.service.detect_hospital_anomaly(self.operation)
        self.assertEqual(len(out), 1)
        self.assertIn(int(out.iloc[0]["anomaly_flag"]), {0, 1})
        self.assertTrue(np.isfinite(float(out.iloc[0]["anomaly_score"])))

    def test_twin_predicted_state_uses_saved_model_outputs(self):
        from src.twin.hospital_twin import build_twin_from_latest

        twin = build_twin_from_latest(prediction_service=self.service)
        predicted = twin.get_predicted_state(days_ahead=1)
        self.assertEqual(predicted.as_of_date, "2025-01-01")
        self.assertTrue(any("OPD forecast (" in note for note in predicted.notes))
        self.assertTrue(any("Predicted waiting time (ML)" in note for note in predicted.notes))
        self.assertGreaterEqual(predicted.current_opd, 0)
        self.assertGreaterEqual(predicted.average_waiting_time, 0)

    def test_rejects_impossible_blood_pressure(self):
        patient = self.patient.to_dict("records")[0]
        patient["systolic_bp"], patient["diastolic_bp"] = 90, 100
        patient["pulse_pressure"] = -10
        with self.assertRaises(PredictionServiceError):
            self.service.predict_patient_risk(patient)

        patient = self.patient.to_dict("records")[0]
        patient["pulse_pressure"] = patient["systolic_bp"] - patient["diastolic_bp"] + 5
        with self.assertRaises(PredictionServiceError):
            self.service.predict_patient_risk(patient)

    def test_forecast_opd_is_valid_and_reproducible(self):
        a = self.service.forecast_opd(days=7)
        b = self.service.forecast_opd(days=7)
        self.assertEqual(len(a), 7)
        self.assertTrue(np.isfinite(a.select_dtypes("number").to_numpy()).all())
        self.assertTrue((a.select_dtypes("number") >= 0).all().all())
        pd.testing.assert_frame_equal(a, b)

    def test_anomaly_flags_extreme_day_but_not_typical_day(self):
        # Last dataset day (2024-12-31) is not flagged in the saved scoring.
        typical = self.operation.iloc[0][
            self.service._anomaly["feature_columns"]
        ].to_dict()
        extreme = {
            "opd_patients": 900, "emergency_patients": 300,
            "bed_occupancy_percentage": 99, "icu_occupancy_percentage": 100,
            "average_waiting_time_minutes": 300,
        }
        out = self.service.detect_hospital_anomaly([typical, extreme])
        self.assertEqual(out["anomaly_flag"].tolist(), [0, 1])
        self.assertGreater(out["anomaly_score"].iloc[1], out["anomaly_score"].iloc[0])
        self.assertNotEqual(out["anomaly_reason"].iloc[1], "")

    def test_rejects_nonfinite_and_out_of_range_inputs(self):
        patient = self.patient.to_dict("records")[0]
        patient["age"] = float("nan")
        with self.assertRaises(PredictionServiceError):
            self.service.predict_patient_risk(patient)

        operation = self.operation.to_dict("records")[0]
        operation["opd_patients"] = 6000
        with self.assertRaises(PredictionServiceError):
            self.service.detect_hospital_anomaly(operation)

        operation = self.operation.to_dict("records")[0]
        operation["opd_patients"] = float("nan")
        with self.assertRaises(PredictionServiceError):
            self.service.predict_waiting_time(pd.DataFrame([operation]))


if __name__ == "__main__":
    unittest.main()
