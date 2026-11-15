"""Unit tests for the Phase 6 alerts engine (no model files needed)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts import (
    Alert,
    AlertCategory,
    AlertEngine,
    AlertThresholds,
    ReferenceStats,
    Severity,
    deduplicate_alerts,
)
from src.alerts.models import BASIS_ML, BASIS_OBSERVED, BASIS_RULE

C = AlertCategory

# Explicit reference statistics so the boundary arithmetic is exact.
REF = ReferenceStats(
    opd_mean=250.0,
    opd_std=80.0,
    medicine_mean=300.0,
    medicine_std=50.0,
    staff_pressure_mean=2.0,
    staff_pressure_std=0.5,
)


def normal_state(**overrides) -> dict:
    """A comfortable day. Plain dict: the engine accepts mappings."""
    state = {
        "as_of_date": "2024-06-15",
        "current_opd": 250,
        "emergency_cases": 50,
        "admissions": 36,
        "bed_occupancy_percentage": 40.0,
        "icu_occupancy_percentage": 50.0,
        "doctors_available": 42,
        "nurses_available": 115,
        "medicine_stock": 58000,
        "average_waiting_time": 25.0,
    }
    state.update(overrides)
    return state


def only(report, category):
    return [a for a in report.alerts if a.category == category]


class AlertTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine(reference=REF)

    def severity_of(self, report, category):
        found = only(report, category)
        self.assertLessEqual(len(found), 1, "duplicate alerts for one category")
        return found[0].severity if found else None


class TestNormalConditions(AlertTestCase):
    def test_normal_state_has_no_alerts(self):
        report = self.engine.evaluate(
            normal_state(), anomaly={"anomaly_flag": 0, "anomaly_score": -0.1}
        )
        self.assertEqual(report.alerts, [])
        self.assertIsNone(report.highest_severity)
        self.assertEqual(report.unavailable, [])

    def test_report_to_dict_for_empty_report(self):
        data = self.engine.evaluate(normal_state(), anomaly=None).to_dict()
        self.assertEqual(data["alerts"], [])
        self.assertIsNone(data["highest_severity"])


class TestOccupancyBoundaries(AlertTestCase):
    LADDER = [
        (69.99, None),
        (70.0, Severity.LOW),
        (79.99, Severity.LOW),
        (80.0, Severity.MEDIUM),
        (90.0, Severity.HIGH),
        (94.99, Severity.HIGH),
        (95.0, Severity.CRITICAL),
        (100.0, Severity.CRITICAL),
    ]

    def test_bed_occupancy_boundaries(self):
        for value, expected in self.LADDER:
            with self.subTest(value=value):
                report = self.engine.evaluate(
                    normal_state(bed_occupancy_percentage=value), anomaly={"anomaly_flag": 0, "anomaly_score": 0}
                )
                self.assertEqual(self.severity_of(report, C.BED_OCCUPANCY), expected)

    def test_icu_occupancy_boundaries(self):
        for value, expected in self.LADDER:
            with self.subTest(value=value):
                report = self.engine.evaluate(
                    normal_state(icu_occupancy_percentage=value), anomaly={"anomaly_flag": 0, "anomaly_score": 0}
                )
                self.assertEqual(self.severity_of(report, C.ICU_PRESSURE), expected)

    def test_alert_fields_are_complete(self):
        report = self.engine.evaluate(
            normal_state(bed_occupancy_percentage=92.0), anomaly=None
        )
        alert = only(report, C.BED_OCCUPANCY)[0]
        self.assertEqual(alert.severity, Severity.HIGH)
        self.assertTrue(alert.alert_id)
        self.assertEqual(alert.metric, "bed_occupancy_percentage")
        self.assertEqual(alert.observed_value, 92.0)
        self.assertIn(">= 90", alert.threshold)
        self.assertEqual(alert.threshold_value, 90.0)
        self.assertEqual(alert.date, "2024-06-15")
        self.assertEqual(alert.basis, BASIS_OBSERVED)
        self.assertTrue(alert.title and alert.explanation)
        data = alert.to_dict()
        self.assertEqual(data["severity"], "High")
        self.assertEqual(data["category"], "bed_occupancy")

    def test_predicted_scope_capacity_is_rule_derived_not_observed(self):
        report = self.engine.evaluate(
            normal_state(bed_occupancy_percentage=96.0),
            scope="predicted",
            anomaly=None,
        )
        self.assertEqual(only(report, C.BED_OCCUPANCY)[0].basis, BASIS_RULE)


class TestOpdInflow(AlertTestCase):
    def test_zscore_boundaries(self):
        # mean 250, std 80 -> z=2.0 at 410, 2.5 at 450, 3.0 at 490, 4.0 at 570.
        cases = [
            (409, None),
            (410, Severity.LOW),
            (450, Severity.MEDIUM),
            (490, Severity.HIGH),
            (569, Severity.HIGH),
            (570, Severity.CRITICAL),
        ]
        for opd, expected in cases:
            with self.subTest(opd=opd):
                report = self.engine.evaluate(normal_state(current_opd=opd), anomaly=None)
                self.assertEqual(self.severity_of(report, C.OPD_INFLOW), expected)

    def test_predicted_opd_is_labelled_ml(self):
        report = self.engine.evaluate(
            normal_state(current_opd=600), scope="predicted", anomaly=None
        )
        self.assertEqual(only(report, C.OPD_INFLOW)[0].basis, BASIS_ML)


class TestRisingDemand(AlertTestCase):
    def test_percent_increase_boundaries(self):
        current = normal_state(current_opd=200)
        cases = [
            (229, None),
            (230, Severity.LOW),  # +15%
            (250, Severity.MEDIUM),  # +25%
            (280, Severity.HIGH),  # +40%
            (320, Severity.CRITICAL),  # +60%
            (150, None),  # falling demand never alerts
        ]
        for forecast, expected in cases:
            with self.subTest(forecast=forecast):
                report = self.engine.evaluate(
                    normal_state(current_opd=forecast),
                    scope="predicted",
                    anomaly=None,
                    compare_to=current,
                )
                self.assertEqual(self.severity_of(report, C.OPD_RISING), expected)

    def test_rising_alert_is_ml_basis_and_needs_comparison_state(self):
        report = self.engine.evaluate(
            normal_state(current_opd=320),
            scope="predicted",
            anomaly=None,
            compare_to=normal_state(current_opd=200),
        )
        self.assertEqual(only(report, C.OPD_RISING)[0].basis, BASIS_ML)
        without = self.engine.evaluate(
            normal_state(current_opd=320), scope="predicted", anomaly=None
        )
        self.assertEqual(only(without, C.OPD_RISING), [])

    def test_zero_current_opd_is_unavailable_not_divide_by_zero(self):
        report = self.engine.evaluate(
            normal_state(current_opd=100),
            scope="predicted",
            anomaly=None,
            compare_to=normal_state(current_opd=0),
        )
        self.assertTrue(any("Rising OPD" in u for u in report.unavailable))


class TestMedicine(AlertTestCase):
    def state_with_cover(self, stock):
        # 100 OPD, no emergencies/admissions -> exactly 80 units/day.
        return normal_state(
            current_opd=100, emergency_cases=0, admissions=0, medicine_stock=stock
        )

    def test_days_of_cover_boundaries(self):
        # cover = stock / 80. Triggers strictly below 21/14/7/3 days.
        cases = [
            (1680, None),  # exactly 21.0 days
            (1679, Severity.LOW),
            (1120, Severity.LOW),  # exactly 14.0 days is not < 14
            (1119, Severity.MEDIUM),
            (560, Severity.MEDIUM),  # exactly 7.0
            (559, Severity.HIGH),
            (240, Severity.HIGH),  # exactly 3.0
            (239, Severity.CRITICAL),
            (0, Severity.CRITICAL),
        ]
        for stock, expected in cases:
            with self.subTest(stock=stock):
                report = self.engine.evaluate(self.state_with_cover(stock), anomaly=None)
                self.assertEqual(self.severity_of(report, C.MEDICINE_STOCK), expected)

    def test_stock_alert_is_rule_derived_and_says_so(self):
        alert = only(
            self.engine.evaluate(self.state_with_cover(100), anomaly=None),
            C.MEDICINE_STOCK,
        )[0]
        self.assertEqual(alert.basis, BASIS_RULE)
        self.assertIn("rule-derived", alert.explanation)
        self.assertEqual(alert.metric, "medicine_days_of_cover")

    def test_zero_consumption_does_not_alert_or_crash(self):
        state = normal_state(current_opd=0, emergency_cases=0, admissions=0, medicine_stock=0)
        report = self.engine.evaluate(state, anomaly=None)
        self.assertEqual(only(report, C.MEDICINE_STOCK), [])

    def test_high_consumption_zscore_boundaries(self):
        # consumption = 0.8 * opd. mean 300, std 50 -> z=2 at 400 (opd 500).
        cases = [(499, None), (500, Severity.LOW), (550, Severity.MEDIUM),
                 (600, Severity.HIGH), (700, Severity.CRITICAL)]
        for opd, expected in cases:
            with self.subTest(opd=opd):
                state = normal_state(current_opd=opd, emergency_cases=0, admissions=0)
                report = self.engine.evaluate(state, anomaly=None)
                self.assertEqual(self.severity_of(report, C.MEDICINE_CONSUMPTION), expected)


class TestStaffWorkload(AlertTestCase):
    def state_with_pressure(self, load):
        # load / (20 + 80) patients per staff member.
        return normal_state(
            current_opd=load, emergency_cases=0, admissions=0,
            doctors_available=20, nurses_available=80,
        )

    def test_staff_pressure_boundaries(self):
        # mean 2.0 std 0.5 -> z=1.5 at 2.75, 2.0 at 3.0, 2.5 at 3.25, 3.0 at 3.5.
        cases = [(274, None), (275, Severity.LOW), (300, Severity.MEDIUM),
                 (325, Severity.HIGH), (350, Severity.CRITICAL)]
        for load, expected in cases:
            with self.subTest(load=load):
                report = self.engine.evaluate(self.state_with_pressure(load), anomaly=None)
                self.assertEqual(self.severity_of(report, C.STAFF_WORKLOAD), expected)

    def test_no_clinical_staff_is_critical(self):
        state = normal_state(doctors_available=0, nurses_available=0)
        alert = only(self.engine.evaluate(state, anomaly=None), C.STAFF_WORKLOAD)[0]
        self.assertEqual(alert.severity, Severity.CRITICAL)
        self.assertTrue(math.isfinite(alert.observed_value))


class TestAnomaly(AlertTestCase):
    def test_unflagged_gives_no_alert(self):
        report = self.engine.evaluate(normal_state(), anomaly={"anomaly_flag": 0, "anomaly_score": 0.5})
        self.assertEqual(only(report, C.ANOMALY), [])

    def test_flag_severity_by_score(self):
        cases = [(0.05, Severity.MEDIUM), (0.0999, Severity.MEDIUM),
                 (0.10, Severity.HIGH), (0.19, Severity.HIGH)]
        for score, expected in cases:
            with self.subTest(score=score):
                report = self.engine.evaluate(
                    normal_state(),
                    anomaly={"anomaly_flag": 1, "anomaly_score": score,
                             "anomaly_reason": "opd_patients high"},
                )
                alert = only(report, C.ANOMALY)[0]
                self.assertEqual(alert.severity, expected)
                self.assertEqual(alert.basis, BASIS_ML)
                self.assertIn("opd_patients high", alert.explanation)

    def test_missing_or_invalid_anomaly_output_is_unavailable(self):
        for bad in (None, {}, {"anomaly_flag": 1}, {"anomaly_flag": 1, "anomaly_score": float("nan")},
                    {"anomaly_flag": 7, "anomaly_score": 0.1}):
            with self.subTest(bad=bad):
                report = self.engine.evaluate(normal_state(), anomaly=bad)
                self.assertEqual(only(report, C.ANOMALY), [])
                self.assertTrue(any("Anomaly" in u for u in report.unavailable))


class TestMissingMetrics(AlertTestCase):
    def test_missing_metric_is_reported_not_guessed(self):
        state = normal_state()
        del state["bed_occupancy_percentage"]
        state["icu_occupancy_percentage"] = None
        report = self.engine.evaluate(state, anomaly=None)
        self.assertEqual(only(report, C.BED_OCCUPANCY), [])
        self.assertEqual(only(report, C.ICU_PRESSURE), [])
        self.assertTrue(any("Bed occupancy" in u for u in report.unavailable))
        self.assertTrue(any("ICU occupancy" in u for u in report.unavailable))

    def test_invalid_values_are_unavailable(self):
        for bad in (float("nan"), float("inf"), -5, "abc", True, 150):
            with self.subTest(bad=bad):
                report = self.engine.evaluate(
                    normal_state(bed_occupancy_percentage=bad), anomaly=None
                )
                self.assertEqual(only(report, C.BED_OCCUPANCY), [])
                self.assertTrue(any("Bed occupancy" in u for u in report.unavailable))

    def test_no_reference_statistics_skips_zscore_checks_only(self):
        engine = AlertEngine(reference=None)
        report = engine.evaluate(
            normal_state(current_opd=900, bed_occupancy_percentage=96), anomaly=None
        )
        self.assertEqual(only(report, C.OPD_INFLOW), [])
        self.assertEqual(self.severity_of(report, C.BED_OCCUPANCY), Severity.CRITICAL)
        self.assertTrue(any("OPD inflow" in u for u in report.unavailable))

    def test_zero_std_reference_is_treated_as_unavailable(self):
        engine = AlertEngine(reference=ReferenceStats(opd_mean=250, opd_std=0.0))
        report = engine.evaluate(normal_state(current_opd=900), anomaly=None)
        self.assertEqual(only(report, C.OPD_INFLOW), [])

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate(normal_state(), scope="tomorrow")


class TestDuplicatesAndOrdering(AlertTestCase):
    def test_one_alert_per_category_and_unique_ids(self):
        state = normal_state(
            bed_occupancy_percentage=97, icu_occupancy_percentage=97,
            current_opd=700, medicine_stock=10,
        )
        report = self.engine.evaluate(
            state, anomaly={"anomaly_flag": 1, "anomaly_score": 0.2}
        )
        categories = [a.category for a in report.alerts]
        self.assertEqual(len(categories), len(set(categories)))
        ids = [a.alert_id for a in report.alerts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deduplicate_keeps_most_severe_for_same_key(self):
        def mk(severity, scope="current"):
            return Alert(
                alert_id=f"bed:{scope}", category=C.BED_OCCUPANCY, severity=severity,
                title="t", explanation="e", metric="m", observed_value=1.0,
                threshold="x", threshold_value=1.0, date="2024-01-01",
                basis=BASIS_OBSERVED, scope=scope,
            )
        merged = deduplicate_alerts(
            [mk(Severity.LOW), mk(Severity.CRITICAL), mk(Severity.HIGH), mk(Severity.LOW, "predicted")]
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].severity, Severity.CRITICAL)
        self.assertEqual({a.scope for a in merged}, {"current", "predicted"})

    def test_current_and_predicted_scopes_are_distinct_alerts(self):
        now = self.engine.evaluate(normal_state(bed_occupancy_percentage=96), anomaly=None)
        later = self.engine.evaluate(
            normal_state(bed_occupancy_percentage=96), scope="predicted", anomaly=None
        )
        merged = deduplicate_alerts(now.alerts + later.alerts)
        self.assertEqual(len(merged), 2)
        self.assertNotEqual(merged[0].alert_id, merged[1].alert_id)

    def test_alerts_sorted_most_severe_first(self):
        report = self.engine.evaluate(
            normal_state(bed_occupancy_percentage=72, icu_occupancy_percentage=99),
            anomaly=None,
        )
        severities = [int(a.severity) for a in report.alerts]
        self.assertEqual(severities, sorted(severities, reverse=True))
        self.assertEqual(report.highest_severity, Severity.CRITICAL)


class TestThresholdConfiguration(unittest.TestCase):
    def test_custom_thresholds_change_behaviour(self):
        strict = AlertThresholds(bed_occupancy=(30.0, 40.0, 50.0, 60.0))
        engine = AlertEngine(thresholds=strict, reference=REF)
        report = engine.evaluate(normal_state(bed_occupancy_percentage=45), anomaly=None)
        self.assertEqual(only(report, C.BED_OCCUPANCY)[0].severity, Severity.MEDIUM)

    def test_non_monotonic_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            AlertThresholds(bed_occupancy=(90.0, 80.0, 70.0, 60.0))
        with self.assertRaises(ValueError):
            AlertThresholds(medicine_days_cover=(3.0, 7.0, 14.0, 21.0))
        with self.assertRaises(ValueError):
            AlertThresholds(anomaly_high_score=0)

    def test_reference_stats_from_dataframe_validates_input(self):
        import pandas as pd

        with self.assertRaises(ValueError):
            ReferenceStats.from_dataframe(pd.DataFrame({"opd_patients": range(40)}))
        with self.assertRaises(ValueError):
            ReferenceStats.from_dataframe(
                pd.DataFrame(
                    {c: [1] * 5 for c in (
                        "opd_patients", "emergency_patients", "admissions",
                        "doctors_available", "nurses_available")}
                )
            )


if __name__ == "__main__":
    unittest.main()
