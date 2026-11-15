"""
Phase 6 integration tests with the real digital twin, saved prediction
service and processed dataset. Skipped if the artifacts are missing.

Nothing is written to disk: twin state changes are in-memory only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


@unittest.skipUnless(all(p.is_file() for p in _NEEDED), "model artifacts or data missing")
class TestPhase6RealTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        from src.alerts import AlertEngine, load_reference_stats
        from src.models.prediction_service import get_prediction_service
        from src.twin.hospital_twin import build_twin_from_latest

        cls.service = get_prediction_service()
        cls.reference = load_reference_stats()
        cls.build = staticmethod(lambda: build_twin_from_latest(prediction_service=cls.service))
        cls.engine = AlertEngine(reference=cls.reference)
        cls.df = pd.read_csv(ROOT / "data" / "processed" / "hospital_operations_features.csv")

    def test_reference_stats_come_from_the_dataset(self):
        ref = self.reference
        self.assertAlmostEqual(ref.opd_mean, float(self.df["opd_patients"].mean()), places=6)
        self.assertGreater(ref.opd_std, 0)
        self.assertGreater(ref.medicine_std, 0)
        self.assertGreater(ref.staff_pressure_std, 0)

    def test_report_for_untouched_twin_is_consistent(self):
        from src.recommendations import generate_operational_report

        report, recs = generate_operational_report(self.build(), days_ahead=1)
        ids = [a.alert_id for a in report.alerts]
        self.assertEqual(len(ids), len(set(ids)))
        keys = [a.dedup_key for a in report.alerts]
        self.assertEqual(len(keys), len(set(keys)))
        alert_ids = set(ids)
        for rec in recs:
            self.assertIn(rec.alert_id, alert_ids)
        for alert in report.alerts:
            self.assertTrue(alert.title and alert.explanation and alert.threshold)
            self.assertIn(alert.scope, ("current", "predicted"))
            self.assertIsNotNone(alert.date)

    def test_stressed_twin_raises_capacity_alerts_and_surge_recommendation(self):
        from src.alerts import AlertCategory, Severity, generate_alerts
        from src.recommendations import COMBINED_CAPACITY, generate_recommendations

        twin = self.build()
        state = twin.get_current_state()
        twin.update_state(
            occupied_beds=int(state.total_beds * 0.96),
            occupied_icu_beds=state.total_icu_beds - 1,
        )
        report = generate_alerts(twin, days_ahead=None, reference=self.reference)
        bed = report.by_category(AlertCategory.BED_OCCUPANCY)
        icu = report.by_category(AlertCategory.ICU_PRESSURE)
        self.assertEqual(bed[0].severity, Severity.CRITICAL)
        self.assertEqual(icu[0].severity, Severity.CRITICAL)
        recs = generate_recommendations(report)
        self.assertTrue([r for r in recs if r.category == COMBINED_CAPACITY])
        self.assertEqual(report.highest_severity, Severity.CRITICAL)

    def test_extreme_day_triggers_ml_anomaly_and_opd_alerts(self):
        from src.alerts import AlertCategory, generate_alerts
        from src.alerts.models import BASIS_ML

        twin = self.build()
        twin.update_state(
            current_opd=900, emergency_cases=300,
            occupied_beds=int(twin.get_current_state().total_beds * 0.99),
            occupied_icu_beds=twin.get_current_state().total_icu_beds,
            average_waiting_time=300.0,
        )
        report = generate_alerts(twin, days_ahead=None, reference=self.reference)
        anomaly = report.by_category(AlertCategory.ANOMALY)
        self.assertEqual(len(anomaly), 1)
        self.assertEqual(anomaly[0].basis, BASIS_ML)
        self.assertTrue(report.by_category(AlertCategory.OPD_INFLOW))
        self.assertTrue(report.by_category(AlertCategory.MEDICINE_CONSUMPTION))

    def test_service_unavailable_is_reported_not_hidden(self):
        from src.alerts import generate_alerts

        class NoServiceTwin:
            service = None

            def __init__(self, state):
                self._state = state

            def get_current_state(self):
                return self._state

        twin = NoServiceTwin(self.build().get_current_state())
        report = generate_alerts(twin, days_ahead=None, reference=self.reference)
        self.assertTrue(any("Anomaly" in u for u in report.unavailable))

    def test_predicted_state_alerts_are_scoped_and_labelled(self):
        from src.alerts import generate_alerts

        twin = self.build()
        twin.update_state(occupied_beds=int(twin.get_current_state().total_beds * 0.96))
        report = generate_alerts(twin, days_ahead=1, reference=self.reference)
        scopes = {a.scope for a in report.alerts}
        self.assertEqual(scopes, {"current", "predicted"})
        predicted_beds = [a for a in report.alerts if a.category.value == "bed_occupancy" and a.scope == "predicted"]
        self.assertEqual(predicted_beds[0].basis, "rule_derived")  # twin projection is a heuristic

    def test_whole_dataset_sweep_is_sane(self):
        """Evaluate every historical day: no crashes, valid output, rare OPD alerts."""
        from src.alerts import AlertCategory
        from src.twin.hospital_twin import hospital_state_from_row

        opd_alert_days = 0
        for _, row in self.df.iterrows():
            report = self.engine.evaluate(hospital_state_from_row(row), anomaly=None)
            keys = [a.dedup_key for a in report.alerts]
            self.assertEqual(len(keys), len(set(keys)))
            for alert in report.alerts:
                self.assertTrue(1 <= int(alert.severity) <= 4)
            if report.by_category(AlertCategory.OPD_INFLOW):
                opd_alert_days += 1
        self.assertLess(opd_alert_days / len(self.df), 0.05)


if __name__ == "__main__":
    unittest.main()
