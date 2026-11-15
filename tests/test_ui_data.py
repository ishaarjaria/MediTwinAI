"""Tests for the dashboard's pure data-preparation helpers (Phase 7)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerts import AlertCategory, AlertEngine, AlertThresholds, ReferenceStats
from src.recommendations import generate_recommendations
from src.simulation.what_if import SOURCE_ML, SOURCE_RULE, ScenarioChanges, WhatIfSimulator
from src.twin import hospital_twin
from src.twin.hospital_twin import HospitalState
from src.ui import data as d


def make_state(**kw) -> HospitalState:
    base = dict(
        as_of_date="2024-06-15", total_beds=250, occupied_beds=150, available_beds=100,
        bed_occupancy_percentage=60.0, total_icu_beds=30, occupied_icu_beds=12,
        icu_occupancy_percentage=40.0, current_opd=400, emergency_cases=40,
        doctors_available=40, nurses_available=100, average_waiting_time=44.0,
        medicine_stock=20000, admissions=30, discharges=28, is_weekend=0, is_holiday=0,
        day_of_week=5, month=6,
    )
    base.update(kw)
    return HospitalState(**base)


class StubWait:
    def predict_waiting_time(self, days):
        row = dict(days)
        wait = 8 + 0.12 * row["opd_patients"] - 0.35 * row["doctors_available"] + 0.05 * row["emergency_patients"]
        return pd.DataFrame([{**row, "predicted_waiting_time_minutes": round(max(0.0, wait), 2)}])


class TestFormatting(unittest.TestCase):
    def test_fmt_num_handles_missing_and_non_finite(self):
        self.assertEqual(d.fmt_num(1234.5, 1), "1,234.5")
        self.assertEqual(d.fmt_num(None), "—")
        self.assertEqual(d.fmt_num(float("nan")), "—")
        self.assertEqual(d.fmt_num(float("inf")), "—")

    def test_fmt_signed(self):
        self.assertEqual(d.fmt_signed(3.14159, 1, "%"), "+3.1%")
        self.assertEqual(d.fmt_signed(-2, 0), "−2")
        self.assertEqual(d.fmt_signed(0.001, 1), "0")
        self.assertEqual(d.fmt_signed(None), "—")

    def test_delta_tone_follows_direction_of_harm(self):
        self.assertEqual(d.delta_tone(5, True), "bad")
        self.assertEqual(d.delta_tone(-5, True), "good")
        self.assertEqual(d.delta_tone(5, False), "good")
        self.assertEqual(d.delta_tone(-5, False), "bad")
        self.assertEqual(d.delta_tone(5, None), "neutral")
        self.assertEqual(d.delta_tone(0.0, True), "neutral")
        self.assertEqual(d.delta_tone(0.004, True, decimals=2), "neutral")
        self.assertEqual(d.delta_tone(None, True), "neutral")
        self.assertEqual(d.delta_tone(float("nan"), True), "neutral")

    def test_delta_arrow(self):
        self.assertEqual(d.delta_arrow(2), "▲")
        self.assertEqual(d.delta_arrow(-2), "▼")
        self.assertEqual(d.delta_arrow(0), "•")
        self.assertEqual(d.delta_arrow(None), "•")

    def test_format_date_and_model_label(self):
        self.assertEqual(d.format_date("2024-12-31"), "31 Dec 2024")
        self.assertEqual(d.format_date("2024-12-31", with_year=False), "31 Dec")
        self.assertEqual(d.format_date("not a date"), "—")
        self.assertEqual(d.model_label("xgboost"), "XGBoost")
        self.assertEqual(d.model_label("random_forest"), "Random Forest")
        self.assertEqual(d.model_label(None), "—")

    def test_provenance_mapping(self):
        self.assertEqual(d.provenance_from_source(SOURCE_ML), "ml")
        self.assertEqual(d.provenance_from_source(SOURCE_RULE), "rule")
        self.assertEqual(d.provenance_from_source("unavailable (no model)"), "unavailable")
        self.assertEqual(d.provenance_from_basis("ml_prediction"), "ml")
        self.assertEqual(d.provenance_from_basis("observed"), "observed")
        self.assertEqual(d.provenance_from_basis("rule_derived"), "rule")
        self.assertEqual(d.provenance_from_basis("???"), "unavailable")


class TestWindowsAndAggregates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = d.load_operations()

    def test_loaded_data_is_sorted_and_non_empty(self):
        self.assertTrue(self.df["date"].is_monotonic_increasing)
        self.assertGreater(len(self.df), 300)

    def test_window_dates(self):
        lo, hi = self.df["date"].min(), self.df["date"].max()
        s30, e30 = d.window_dates(self.df, "Last 30 days")
        self.assertEqual(e30, hi)
        self.assertEqual((e30 - s30).days, 29)
        self.assertEqual(d.window_dates(self.df, "Full period"), (lo, hi))

    def test_custom_window_is_clamped_to_data(self):
        lo, hi = self.df["date"].min(), self.df["date"].max()
        start, end = d.window_dates(self.df, "Custom", ("1990-01-01", "2999-01-01"))
        self.assertEqual((start, end), (lo, hi))
        start, end = d.window_dates(self.df, "Custom", (None, None))
        self.assertEqual((start, end), (lo, hi))

    def test_filter_and_summary(self):
        s, e = d.window_dates(self.df, "Last 30 days")
        view = d.filter_dates(self.df, s, e)
        self.assertEqual(len(view), 30)
        summary = d.summarize_window(view)
        self.assertEqual(summary["days"], 30)
        self.assertAlmostEqual(summary["avg_opd"], view["opd_patients"].mean())
        self.assertEqual(summary["peak_opd"], view["opd_patients"].max())

    def test_empty_window_summary_has_no_numbers(self):
        empty = d.filter_dates(self.df, "2030-01-01", "2030-02-01")
        self.assertTrue(empty.empty)
        self.assertTrue(all(v is None for v in d.summarize_window(empty).values()))
        self.assertTrue(d.weekday_profile(empty).empty)
        self.assertTrue(d.monthly_profile(empty).empty)

    def test_weekday_profile_matches_data(self):
        prof = d.weekday_profile(self.df)
        self.assertEqual(list(prof["weekday"]), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        monday = self.df[self.df["date"].dt.dayofweek == 0]["opd_patients"].mean()
        self.assertAlmostEqual(prof.loc[0, "opd_patients"], monday)

    def test_weekly_resample_only_for_long_windows(self):
        short, grain = d.to_weekly(self.df.head(60), ["admissions"])
        self.assertEqual((len(short), grain), (60, "daily"))
        long, grain = d.to_weekly(self.df, ["admissions"])
        self.assertEqual(grain, "weekly mean")
        self.assertLess(len(long), len(self.df))

    def test_optional_artifacts_load_or_return_none(self):
        scores = d.load_anomaly_scores()
        self.assertTrue(scores is None or {"anomaly_flag", "anomaly_reason"} <= set(scores.columns))
        self.assertIsNone(d.load_anomaly_scores(ROOT / "nope.csv"))
        self.assertIsNone(d.load_patients(ROOT / "nope.csv"))
        self.assertIsNone(d.load_risk_metrics(ROOT / "nope.json"))
        with self.assertRaises(FileNotFoundError):
            d.load_operations(ROOT / "nope.csv")

    def test_risk_metrics_come_from_the_saved_file(self):
        m = d.load_risk_metrics()
        self.assertIsNotNone(m)
        self.assertTrue(0 < m["accuracy"] <= 1)
        self.assertTrue(0 <= m["recall_high"] <= 1)


class TestTwinComparison(unittest.TestCase):
    def test_capacity_marks_match_the_twins_policy(self):
        policy = tuple(sorted(t for t, _ in hospital_twin._BED_ICU_THRESHOLDS))
        self.assertEqual(d.CAPACITY_MARKS, policy)

    def test_compare_states_rows_and_provenance(self):
        cur = make_state()
        pred = make_state(current_opd=380, average_waiting_time=40.0, occupied_beds=140,
                          bed_occupancy_percentage=56.0, available_beds=110)
        rows = {r["label"]: r for r in d.compare_states(cur, pred)}
        self.assertEqual(rows["OPD patients"]["delta"], -20)
        self.assertEqual(rows["OPD patients"]["tone"], "good")  # lower load is favourable
        self.assertEqual(rows["OPD patients"]["prov_current"], "observed")
        self.assertEqual(rows["OPD patients"]["prov_predicted"], "ml")
        self.assertEqual(rows["Average waiting time"]["prov_predicted"], "ml")
        self.assertEqual(rows["Bed occupancy"]["prov_predicted"], "rule")
        self.assertEqual(rows["Medicine stock"]["prov_predicted"], "rule")
        self.assertEqual(rows["Staff pressure"]["prov_current"], "rule")
        self.assertEqual(rows["Available beds"]["delta"], 10)
        self.assertEqual(rows["Available beds"]["tone"], "good")

    def test_staff_pressure_matches_phase5_rule_and_handles_no_staff(self):
        s = make_state()
        self.assertAlmostEqual(d.state_staff_pressure(s), (400 + 40 + 30) / (40 + 100), places=4)
        self.assertIsNone(d.state_staff_pressure(make_state(doctors_available=0, nurses_available=0)))


class TestWhatIfAdapters(unittest.TestCase):
    def setUp(self):
        self.base = make_state()
        self.sim = WhatIfSimulator(prediction_service=StubWait())

    def test_untouched_controls_are_a_true_no_change(self):
        controls = d.apply_preset(self.base, None)
        changes = d.scenario_from_controls(self.base, controls)
        self.assertTrue(d.is_no_change(changes))
        self.assertIsNone(changes.total_beds)
        self.assertIsNone(changes.icu_capacity)
        result = self.sim.simulate(self.base, changes)
        for row in d.whatif_rows(result):
            self.assertEqual(row["delta"], 0, row["name"])
            self.assertEqual(row["tone"], "neutral", row["name"])

    def test_changed_controls_reach_the_simulator(self):
        controls = d.WhatIfControls(patient_inflow_percent=20, total_beds=300, icu_capacity=30,
                                    doctors_available=50, nurses_available=100, emergency_cases_percent=0)
        changes = d.scenario_from_controls(self.base, controls)
        self.assertEqual(changes.patient_inflow_percent, 20.0)
        self.assertEqual(changes.total_beds, 300)
        self.assertEqual(changes.doctors_available, 50)
        self.assertIsNone(changes.icu_capacity)  # equal to baseline
        self.assertIsNone(changes.nurses_available)
        self.assertFalse(d.is_no_change(changes))

    def test_rows_mirror_simulator_output_and_label_sources(self):
        result = self.sim.simulate(self.base, ScenarioChanges(patient_inflow_percent=20))
        rows = {r["name"]: r for r in d.whatif_rows(result)}
        self.assertEqual(len(rows), len(result.comparisons))
        for comp in result.comparisons:
            self.assertEqual(rows[comp.name]["baseline"], comp.current)
            self.assertEqual(rows[comp.name]["simulated"], comp.simulated)
        self.assertEqual(rows["expected_waiting_time_minutes"]["source"], "ml")
        self.assertEqual(rows["patient_load"]["source"], "rule")
        self.assertEqual(rows["patient_load"]["tone"], "neutral")
        self.assertEqual(rows["bed_occupancy_percentage"]["tone"], "bad")  # more occupancy is worse

    def test_more_doctors_reads_as_favourable(self):
        result = self.sim.simulate(self.base, ScenarioChanges(doctors_available=60))
        rows = {r["name"]: r for r in d.whatif_rows(result)}
        self.assertEqual(rows["staff_pressure_patients_per_staff"]["tone"], "good")
        self.assertEqual(rows["expected_waiting_time_minutes"]["tone"], "good")

    def test_no_staff_is_shown_as_unbounded_not_minus_one(self):
        result = self.sim.simulate(self.base, ScenarioChanges(doctors_available=0, nurses_available=0))
        row = next(r for r in d.whatif_rows(result) if r["name"] == "staff_pressure_patients_per_staff")
        self.assertTrue(row["unbounded"])
        self.assertIsNone(row["simulated"])
        self.assertIsNone(row["delta"])

    def test_control_ranges_contain_the_default_and_respect_model_bounds(self):
        for state in (self.base, make_state(total_beds=5000, doctors_available=500, nurses_available=2000,
                                            total_icu_beds=1000, occupied_beds=10, occupied_icu_beds=1)):
            for field, (lo, hi, default) in d.control_ranges(state).items():
                self.assertLessEqual(lo, default, field)
                self.assertLessEqual(default, hi, field)
            r = d.control_ranges(state)
            self.assertLessEqual(r["total_beds"][1], 5000)
            self.assertLessEqual(r["icu_capacity"][1], 1000)
            self.assertLessEqual(r["doctors_available"][1], 500)
            self.assertLessEqual(r["nurses_available"][1], 2000)

    def test_presets_apply_a_single_clamped_change(self):
        self.assertEqual(d.apply_preset(self.base, "Inflow +20%").patient_inflow_percent, 20)
        self.assertEqual(d.apply_preset(self.base, "ICU −10 beds").icu_capacity, 20)
        self.assertEqual(d.apply_preset(self.base, "+10 doctors").doctors_available, 50)
        self.assertEqual(d.apply_preset(self.base, "+50 beds").total_beds, 300)
        self.assertEqual(d.apply_preset(self.base, "Emergency surge +50%").emergency_cases_percent, 50)
        tiny = make_state(total_icu_beds=5, occupied_icu_beds=2, icu_occupancy_percentage=40.0)
        self.assertEqual(d.apply_preset(tiny, "ICU −10 beds").icu_capacity, 1)  # clamped, never below 1
        self.assertTrue(d.is_no_change(d.scenario_from_controls(self.base, d.apply_preset(self.base, "unknown"))))
        for name in d.PRESETS:
            changes = d.scenario_from_controls(self.base, d.apply_preset(self.base, name))
            self.assertFalse(d.is_no_change(changes), name)
            self.sim.simulate(self.base, changes)  # every preset is a valid scenario


class TestPlainLanguageLayer(unittest.TestCase):
    def test_glossary_covers_every_term_the_ui_explains(self):
        for key in ("beds", "icu", "opd", "wait", "staff", "status", "observed", "ml", "rule", "anomaly", "forecast_band"):
            self.assertTrue(len(d.GLOSSARY[key]) > 20, key)
        self.assertEqual(set(d.CATEGORY_HELP), set(AlertCategory))

    def test_status_headlines(self):
        self.assertEqual(d.status_headline("Normal"), "Operating within normal limits")
        self.assertEqual(len({d.status_headline(s) for s in ("Normal", "Moderate", "High", "Critical")}), 4)
        self.assertEqual(d.status_headline("???"), "Status unavailable")

    def test_status_summary_uses_real_numbers_and_alert_state(self):
        s = make_state()
        text = d.status_summary(s, 0)
        self.assertIn("150 of 250 beds", text)
        self.assertIn("12 of 30 ICU", text)
        self.assertIn("No alerts need attention", text)
        self.assertIn("1 alert need attention", d.status_summary(s, 1, "High"))
        self.assertIn("Highest severity: High", d.status_summary(s, 3, "High"))
        self.assertIn("unavailable", d.status_summary(s, None))

    def test_trailing_context_excludes_today_and_handles_short_data(self):
        df = pd.DataFrame({"x": list(range(1, 41))})  # 40 rows, latest = 40
        self.assertAlmostEqual(d.trailing_context(df, "x", 30), sum(range(10, 40)) / 30)  # rows 10..39
        self.assertIsNone(d.trailing_context(df.head(30), "x", 30))
        self.assertIsNone(d.trailing_context(df, "missing", 30))
        real = d.load_operations()
        avg = d.trailing_context(real, "opd_patients", 30)
        self.assertAlmostEqual(avg, real["opd_patients"].iloc[-31:-1].mean())

    def test_twin_change_summary_only_lists_real_changes(self):
        cur = make_state()
        same = d.twin_change_summary(d.compare_states(cur, make_state(), current_model_wait=cur.average_waiting_time))
        self.assertEqual(same, [])
        pred = make_state(current_opd=380, average_waiting_time=51.5)
        # Without a like-for-like model baseline no waiting-time change is claimed.
        unfair = " | ".join(l["text"] for l in d.twin_change_summary(d.compare_states(cur, pred)))
        self.assertNotIn("waiting time", unfair.lower())
        lines = d.twin_change_summary(d.compare_states(cur, pred, current_model_wait=44.0))
        texts = " | ".join(line["text"] for line in lines)
        self.assertIn("Expected outpatients: 400 → 380", texts)
        self.assertIn("Estimated waiting time: 44.0 min → 51.5 min", texts)
        by_text = {line["text"].split(":")[0]: line for line in lines}
        self.assertEqual(by_text["Expected outpatients"]["source"], "ml")
        self.assertEqual(by_text["Expected outpatients"]["tone"], "good")
        self.assertEqual(by_text["Estimated waiting time"]["tone"], "bad")

    def test_control_caption(self):
        self.assertEqual(d.control_caption(0, 0, percent=True), "Unchanged from baseline")
        self.assertEqual(d.control_caption(0, 20, percent=True), "+20% versus baseline")
        self.assertEqual(d.control_caption(0, -30, percent=True), "−30% versus baseline")
        self.assertEqual(d.control_caption(250, 250, " beds"), "Same as baseline (250 beds)")
        self.assertEqual(d.control_caption(250, 300, " beds"), "Baseline 250 → scenario 300 (+50 beds)")
        self.assertEqual(d.control_caption(41, 31), "Baseline 41 → scenario 31 (−10)")

    def test_meter_geometry_places_the_alert_mark(self):
        fill, mark = d.meter_geometry(24.4, 70.0, "≥")
        self.assertLess(fill, mark)  # comfortably below the alert level
        fill, mark = d.meter_geometry(90.0, 70.0, "≥")
        self.assertGreater(fill, mark)  # past the alert level
        fill, mark = d.meter_geometry(148.3, 21.0, "<")  # plenty of cover
        self.assertGreater(fill, mark)
        fill, mark = d.meter_geometry(10.0, 21.0, "<")  # cover below the alert level
        self.assertLess(fill, mark)
        for args in ((0, 70, "≥"), (500, 70, "≥"), (0, 0, "≥"), (5, 0, "<")):
            f, m = d.meter_geometry(*args)
            self.assertTrue(0 <= f <= 100 and 0 <= m <= 100, args)

    def test_whatif_labels_are_plain_language(self):
        labels = {meta[0] for meta in d.WHATIF_META.values()}
        for plain in ("Beds in use", "ICU in use", "Estimated waiting time", "Staff workload", "Beds needed"):
            self.assertIn(plain, labels)
        self.assertEqual(set(d.WHATIF_META), set(d.WHATIF_EXPLAIN))  # every metric has a definition


class TestAlertDisplayHelpers(unittest.TestCase):
    REF = ReferenceStats(opd_mean=250, opd_std=80, medicine_mean=300, medicine_std=50,
                         staff_pressure_mean=2.0, staff_pressure_std=0.5)

    def report(self):
        state = {
            "as_of_date": "2024-06-15", "current_opd": 520, "emergency_cases": 50, "admissions": 36,
            "bed_occupancy_percentage": 96.0, "icu_occupancy_percentage": 75.0,
            "doctors_available": 42, "nurses_available": 115, "medicine_stock": 58000,
        }
        return AlertEngine(reference=self.REF).evaluate(state, anomaly=None)

    def test_counts_and_filters(self):
        rep = self.report()
        counts = d.severity_counts(rep.alerts)
        self.assertEqual(list(counts), ["Critical", "High", "Medium", "Low"])
        self.assertEqual(sum(counts.values()), len(rep.alerts))
        crit = d.filter_alerts(rep.alerts, severities=["Critical"])
        self.assertTrue(crit and all(a.severity.label == "Critical" for a in crit))
        beds = d.filter_alerts(rep.alerts, categories=["Bed occupancy"])
        self.assertEqual([a.category for a in beds], [AlertCategory.BED_OCCUPANCY])
        self.assertEqual(d.filter_alerts(rep.alerts, scope="predicted"), [])
        self.assertEqual(len(d.filter_alerts(rep.alerts)), len(rep.alerts))  # no filters = all
        self.assertEqual(d.filter_alerts(rep.alerts, severities=["Low"], categories=["Bed occupancy"]), [])

    def test_recommendations_link_back_to_alerts(self):
        rep = self.report()
        recs = generate_recommendations(rep)
        for alert in rep.alerts:
            linked = d.recs_for_alert(recs, alert.alert_id)
            self.assertTrue(linked, alert.alert_id)
        self.assertEqual(d.recs_for_alert(recs, "nonexistent"), [])

    def test_every_category_has_a_display_label(self):
        self.assertEqual(set(d.CATEGORY_LABELS), set(AlertCategory))

    def test_monitoring_rows_show_headroom_without_deciding_severity(self):
        rows = d.monitoring_rows(make_state(bed_occupancy_percentage=60.0), AlertThresholds(), self.REF)
        labels = [r["label"] for r in rows]
        self.assertEqual(labels, ["Bed occupancy", "ICU occupancy", "OPD load", "Medicine cover"])
        opd = next(r for r in rows if r["label"] == "OPD load")
        self.assertAlmostEqual(opd["alert_from"], 250 + 2.0 * 80)
        cover = next(r for r in rows if r["label"] == "Medicine cover")
        self.assertEqual(cover["direction"], "<")
        self.assertTrue(math.isfinite(cover["value"]))

    def test_monitoring_rows_skip_what_cannot_be_computed(self):
        rows = d.monitoring_rows(make_state(), AlertThresholds(), None)
        self.assertNotIn("OPD load", [r["label"] for r in rows])
        idle = make_state(current_opd=0, emergency_cases=0, admissions=0)
        self.assertNotIn("Medicine cover", [r["label"] for r in d.monitoring_rows(idle, AlertThresholds(), None)])


if __name__ == "__main__":
    unittest.main()
