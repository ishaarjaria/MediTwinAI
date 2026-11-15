"""Unit tests for the Phase 6 recommendation engine (no model files needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts import AlertCategory, AlertEngine, AlertReport, ReferenceStats, Severity
from src.recommendations import (
    COMBINED_CAPACITY,
    DISCLAIMER,
    SOURCE_RULE_BASED,
    generate_recommendations,
    recommend_for_alert,
)

C = AlertCategory
REF = ReferenceStats(
    opd_mean=250.0, opd_std=80.0, medicine_mean=300.0, medicine_std=50.0,
    staff_pressure_mean=2.0, staff_pressure_std=0.5,
)


def state(**overrides) -> dict:
    base = {
        "as_of_date": "2024-06-15", "current_opd": 250, "emergency_cases": 50,
        "admissions": 36, "bed_occupancy_percentage": 40.0,
        "icu_occupancy_percentage": 50.0, "doctors_available": 42,
        "nurses_available": 115, "medicine_stock": 58000,
    }
    base.update(overrides)
    return base


ENGINE = AlertEngine(reference=REF)


def alerts_for(**overrides):
    return ENGINE.evaluate(state(**overrides), anomaly=None)


class TestRecommendationPerCategory(unittest.TestCase):
    CASES = {
        C.BED_OCCUPANCY: dict(bed_occupancy_percentage=88.0),
        C.ICU_PRESSURE: dict(icu_occupancy_percentage=88.0),
        C.OPD_INFLOW: dict(current_opd=500),
        C.MEDICINE_STOCK: dict(current_opd=100, emergency_cases=0, admissions=0, medicine_stock=300),
        C.MEDICINE_CONSUMPTION: dict(current_opd=600, emergency_cases=0, admissions=0),
        C.STAFF_WORKLOAD: dict(current_opd=325, emergency_cases=0, admissions=0,
                               doctors_available=20, nurses_available=80),
    }

    def test_each_alert_category_yields_a_linked_rule_based_recommendation(self):
        for category, overrides in self.CASES.items():
            with self.subTest(category=category.value):
                report = alerts_for(**overrides)
                alert = next(a for a in report.alerts if a.category == category)
                recs = [r for r in generate_recommendations(report) if r.alert_id == alert.alert_id]
                self.assertEqual(len(recs), 1)
                rec = recs[0]
                self.assertEqual(rec.category, category.value)
                self.assertEqual(rec.source, SOURCE_RULE_BASED)
                self.assertEqual(rec.severity, alert.severity)
                self.assertEqual(rec.alert_basis, alert.basis)
                self.assertIn(alert.alert_id, rec.related_alert_ids)
                self.assertIn(alert.metric, rec.rationale)
                self.assertTrue(rec.action.strip())
                self.assertEqual(rec.disclaimer, DISCLAIMER)

    def test_anomaly_and_rising_demand_recommendations(self):
        report = ENGINE.evaluate(
            state(current_opd=320), scope="predicted",
            anomaly={"anomaly_flag": 1, "anomaly_score": 0.15},
            compare_to=state(current_opd=200),
        )
        recs = {r.category: r for r in generate_recommendations(report)}
        self.assertIn("anomaly", recs)
        self.assertIn("opd_rising", recs)
        self.assertEqual(recs["opd_rising"].alert_basis, "ml_prediction")
        self.assertIn("Prepare for increased patient inflow", recs["opd_rising"].title)
        self.assertEqual(recs["opd_rising"].source, SOURCE_RULE_BASED)

    def test_action_text_matches_the_brief(self):
        wanted = {
            C.BED_OCCUPANCY: "bed allocation",
            C.ICU_PRESSURE: "ICU capacity",
            C.MEDICINE_STOCK: "replenishment",
            C.STAFF_WORKLOAD: "additional doctor or nurse",
        }
        for category, phrase in wanted.items():
            with self.subTest(category=category.value):
                report = alerts_for(**self.CASES[category])
                alert = next(a for a in report.alerts if a.category == category)
                self.assertIn(phrase, recommend_for_alert(alert).action)

    def test_text_never_promises_outcomes_or_gives_medical_advice(self):
        report = ENGINE.evaluate(
            state(bed_occupancy_percentage=97, icu_occupancy_percentage=97, current_opd=700,
                  medicine_stock=5),
            anomaly={"anomaly_flag": 1, "anomaly_score": 0.2},
        )
        banned = ("guarantee", "will resolve", "diagnos", "treat ", "prescrib", "dose")
        for rec in generate_recommendations(report):
            text = (rec.title + rec.action + rec.rationale).lower()
            for word in banned:
                self.assertNotIn(word, text, rec.recommendation_id)
            self.assertIn("not medical advice", rec.disclaimer)


class TestSeverityAndOrdering(unittest.TestCase):
    def test_urgency_wording_follows_severity(self):
        low = generate_recommendations(alerts_for(bed_occupancy_percentage=72))[0]
        crit = generate_recommendations(alerts_for(bed_occupancy_percentage=98))[0]
        self.assertEqual(low.severity, Severity.LOW)
        self.assertEqual(crit.severity, Severity.CRITICAL)
        self.assertTrue(low.action.startswith("Monitor and plan"))
        self.assertTrue(crit.action.startswith("Escalate"))

    def test_sorted_most_severe_first(self):
        recs = generate_recommendations(
            alerts_for(bed_occupancy_percentage=72, icu_occupancy_percentage=99)
        )
        severities = [int(r.severity) for r in recs]
        self.assertEqual(severities, sorted(severities, reverse=True))

    def test_predicted_scope_is_stated_in_title(self):
        report = ENGINE.evaluate(state(bed_occupancy_percentage=92), scope="predicted", anomaly=None)
        self.assertIn("predicted", generate_recommendations(report)[0].title)


class TestLinkingAndCombinedRule(unittest.TestCase):
    def test_every_recommendation_links_to_an_existing_alert(self):
        report = ENGINE.evaluate(
            state(bed_occupancy_percentage=97, icu_occupancy_percentage=97, current_opd=700,
                  medicine_stock=5),
            anomaly={"anomaly_flag": 1, "anomaly_score": 0.2},
        )
        ids = {a.alert_id for a in report.alerts}
        recs = generate_recommendations(report)
        self.assertTrue(recs)
        for rec in recs:
            self.assertIn(rec.alert_id, ids)
            self.assertTrue(set(rec.related_alert_ids) <= ids)
        self.assertEqual(len({r.recommendation_id for r in recs}), len(recs))

    def test_combined_surge_rule_needs_both_high(self):
        both = generate_recommendations(
            alerts_for(bed_occupancy_percentage=91, icu_occupancy_percentage=96)
        )
        combined = [r for r in both if r.category == COMBINED_CAPACITY]
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0].severity, Severity.CRITICAL)
        self.assertEqual(len(combined[0].related_alert_ids), 2)
        self.assertEqual(combined[0].source, SOURCE_RULE_BASED)

        medium = generate_recommendations(
            alerts_for(bed_occupancy_percentage=91, icu_occupancy_percentage=85)
        )
        self.assertFalse([r for r in medium if r.category == COMBINED_CAPACITY])

    def test_combined_rule_does_not_mix_current_and_predicted(self):
        current = ENGINE.evaluate(state(bed_occupancy_percentage=92), anomaly=None)
        future = ENGINE.evaluate(state(icu_occupancy_percentage=92), scope="predicted", anomaly=None)
        merged = AlertReport(alerts=current.alerts + future.alerts)
        self.assertFalse(
            [r for r in generate_recommendations(merged) if r.category == COMBINED_CAPACITY]
        )


class TestEmptyAndShapes(unittest.TestCase):
    def test_normal_conditions_give_no_recommendations(self):
        self.assertEqual(generate_recommendations(alerts_for()), [])
        self.assertEqual(generate_recommendations([]), [])
        self.assertEqual(generate_recommendations(AlertReport()), [])

    def test_accepts_report_or_plain_list(self):
        report = alerts_for(bed_occupancy_percentage=92)
        self.assertEqual(
            [r.recommendation_id for r in generate_recommendations(report)],
            [r.recommendation_id for r in generate_recommendations(report.alerts)],
        )

    def test_to_dict_is_serialisable(self):
        import json

        recs = generate_recommendations(alerts_for(bed_occupancy_percentage=92))
        json.dumps([r.to_dict() for r in recs])
        self.assertEqual(recs[0].to_dict()["severity"], "High")


if __name__ == "__main__":
    unittest.main()
