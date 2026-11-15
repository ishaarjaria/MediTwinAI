"""Tests for Phase 1-2 components: data generation, preprocessing, feature engineering."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import generate_synthetic_data as gen
from src.data.preprocess import (
    preprocess_hospital_data,
    preprocess_patient_data,
    validate_hospital_processed,
)
from src.features.feature_engineering import (
    engineer_hospital_features,
    validate_engineered_hospital,
    validate_no_obvious_leakage,
)

ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth = gen.generate_hospital_operations_with_truth()

    def test_reproducible_with_fixed_seed(self):
        again, truth2 = gen.generate_hospital_operations_with_truth()
        pd.testing.assert_frame_equal(self.df, again)
        pd.testing.assert_frame_equal(self.truth, truth2)

    def test_different_seed_changes_data(self):
        other = gen.generate_hospital_operations(seed=7)
        self.assertFalse(self.df["opd_patients"].equals(other["opd_patients"]))

    def test_shape_schema_and_no_missing(self):
        self.assertEqual(len(self.df), 366)
        self.assertEqual(self.df.shape[1], 22)
        self.assertEqual(int(self.df.isna().sum().sum()), 0)
        self.assertTrue(self.df["date"].is_unique)

    def test_capacity_limits(self):
        d = self.df
        self.assertTrue((d.occupied_beds <= d.total_beds).all())
        self.assertTrue((d.occupied_icu_beds <= d.total_icu_beds).all())
        self.assertTrue((d.available_beds >= 0).all())
        self.assertTrue(((d.occupied_beds + d.available_beds) == d.total_beds).all())
        self.assertTrue((d.medicine_stock_units >= 0).all())

    def test_bed_occupancy_has_no_floor_drift(self):
        occ = self.df.bed_occupancy_percentage
        # The old generator pinned 16% of days on a fixed lower clip floor.
        self.assertLess((self.df.occupied_beds == self.df.occupied_beds.min()).mean(), 0.05)
        self.assertGreater(occ.iloc[:14].mean(), 50.0)  # no warm-up collapse
        self.assertGreater(occ.min(), 35.0)

    def test_normal_moderate_and_high_pressure_all_exist(self):
        occ = self.df.bed_occupancy_percentage
        self.assertGreater((occ >= 70).sum(), 10)  # reaches the 70% alert threshold
        self.assertGreater((occ >= 90).sum(), 0)  # at least one critical-pressure spell
        self.assertGreater((occ < 70).mean(), 0.5)  # but most days are not alerting
        self.assertLess((occ >= 70).mean(), 0.5)

    def test_pressure_periods_raise_occupancy(self):
        d = self.df.set_index("date")
        p = gen.PRESSURE_PERIODS[1]  # monsoon outbreak
        inside = d.loc[p["start"]:p["end"], "bed_occupancy_percentage"].mean()
        outside = d.loc["2024-04-01":"2024-05-31", "bed_occupancy_percentage"].mean()
        self.assertGreater(inside, outside + 10)

    def test_occupancy_recovers_after_pressure(self):
        d = self.df.set_index("date")
        peak = d.loc["2024-08-12":"2024-09-08", "bed_occupancy_percentage"].max()
        later = d.loc["2024-10-01":"2024-10-25", "bed_occupancy_percentage"].mean()
        self.assertLess(later, peak - 15)

    def test_medicine_stock_depletes_and_recovers(self):
        stock = self.df.medicine_stock_units
        self.assertGreater(stock.max() - stock.min(), 5000)
        diffs = stock.diff().dropna()
        self.assertTrue((diffs < 0).any())  # consumption
        self.assertTrue((diffs > 0).any())  # replenishment
        cover = stock / self.df.medicine_consumption_units
        self.assertTrue((cover < 21).any())  # approaches the shortage ladder
        self.assertGreater((cover >= 21).mean(), 0.6)  # but is mostly healthy

    def test_supplier_disruption_lowers_stock(self):
        d = self.df.set_index("date")
        p = gen.SUPPLY_DISRUPTION_PERIODS[0]
        self.assertLess(d.loc[p["start"]:"2024-09-20", "medicine_stock_units"].min(), 7000)

    def test_stock_is_consistent_with_consumption(self):
        # Without a delivery, stock falls by exactly the day's consumption.
        d = self.df
        drop = -d.medicine_stock_units.diff()
        no_delivery = drop[(drop > 0)]
        diff = (no_delivery - d.medicine_consumption_units.loc[no_delivery.index]).abs()
        self.assertLessEqual(int(diff.max()), 1)  # int() truncation only

    def test_injected_spikes_ground_truth(self):
        self.assertEqual(len(self.truth), gen.N_INJECTED_SPIKES)
        self.assertTrue(set(self.truth.date) <= set(self.df.date))
        self.assertTrue(set(self.truth.spike_type) <= set(gen.SPIKE_TYPES))
        # Spikes are away from the first and last 20 days.
        idx = self.df.index[self.df.date.isin(self.truth.date)]
        self.assertTrue(((idx >= 20) & (idx < len(self.df) - 20)).all())

    def test_spike_days_are_visibly_unusual(self):
        d = self.df.set_index("date")
        spike = d.loc[self.truth.date]
        z_opd = (spike.opd_patients - d.opd_patients.mean()) / d.opd_patients.std()
        z_em = (spike.emergency_patients - d.emergency_patients.mean()) / d.emergency_patients.std()
        icu = spike.icu_occupancy_percentage
        self.assertTrue(((z_opd > 1.5) | (z_em > 1.5) | (icu >= 99)).all())

    def test_icu_and_staff_ranges(self):
        d = self.df
        self.assertTrue(d.icu_occupancy_percentage.between(0, 100).all())
        self.assertTrue(d.doctors_available.between(25, 50).all())
        self.assertTrue(d.nurses_available.between(70, 135).all())

    def test_waiting_time_correlates_with_opd(self):
        self.assertGreater(self.df.opd_patients.corr(self.df.average_waiting_time_minutes), 0.6)

    def test_date_range_validation(self):
        with self.assertRaises(ValueError):
            gen.generate_hospital_operations("2024-01-01", "2024-03-01")

    def test_saved_csv_matches_generator(self):
        saved = pd.read_csv(ROOT / "data" / "synthetic" / "hospital_operations.csv")
        pd.testing.assert_frame_equal(saved, self.df, check_dtype=False)
        truth = pd.read_csv(ROOT / "data" / "synthetic" / "injected_anomalies.csv")
        pd.testing.assert_frame_equal(truth, self.truth, check_dtype=False)


class PatientGeneratorTests(unittest.TestCase):
    def test_labels_follow_rule_and_are_reproducible(self):
        a = gen.generate_patient_data(200)
        b = gen.generate_patient_data(200)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.risk_level), {"Low", "Medium", "High"})
        self.assertTrue((a.apply(gen.assign_risk_level, axis=1) == a.risk_level).all())

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            gen.generate_patient_data(0)


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.raw = gen.generate_hospital_operations()

    def test_clean_data_passes_through_unchanged(self):
        clean, report = preprocess_hospital_data(self.raw.copy())
        self.assertEqual(len(clean), len(self.raw))
        self.assertEqual(validate_hospital_processed(clean), [])

    def test_dirty_input_is_repaired(self):
        dirty = self.raw.copy().astype(object)
        dirty.loc[3, "opd_patients"] = None
        dirty.loc[5, "average_waiting_time_minutes"] = -10
        dirty.loc[7, "date"] = "not-a-date"
        dirty = pd.concat([dirty, dirty.iloc[[10]]], ignore_index=True)  # duplicate row
        clean, report = preprocess_hospital_data(dirty)
        self.assertEqual(int(clean.isna().sum().sum()), 0)
        self.assertTrue(clean["date"].is_unique)
        self.assertGreaterEqual(clean["average_waiting_time_minutes"].min(), 0)
        self.assertEqual(validate_hospital_processed(clean), [])
        self.assertLess(len(clean), len(dirty))

    def test_occupancy_never_exceeds_capacity_after_cleaning(self):
        dirty = self.raw.copy()
        dirty.loc[2, "occupied_beds"] = 999
        clean, _ = preprocess_hospital_data(dirty)
        self.assertTrue((clean.occupied_beds <= clean.total_beds).all())

    def test_patient_cleaning_normalises_and_clips(self):
        p = gen.generate_patient_data(50).astype(object)
        p.loc[0, "gender"] = "MALE "
        p.loc[1, "age"] = 400
        clean, _ = preprocess_patient_data(p)
        self.assertEqual(clean.loc[clean.index[0], "gender"], "Male")
        self.assertLessEqual(int(clean["age"].max()), 120)
        self.assertEqual(int(clean.isna().sum().sum()), 0)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = gen.generate_hospital_operations()
        cls.clean, _ = preprocess_hospital_data(cls.raw.copy())
        cls.feat = engineer_hospital_features(cls.clean)

    def test_features_present_and_validated(self):
        for col in ("lag_1", "lag_7", "rolling_7", "rolling_14", "month", "quarter", "is_weekday"):
            self.assertTrue(any(c.startswith(col) or c == col for c in self.feat.columns), col)
        self.assertEqual(validate_engineered_hospital(self.feat), [])

    def test_lags_are_past_only(self):
        opd_lag = [c for c in self.feat.columns if "lag_1" in c and "opd" in c]
        self.assertTrue(opd_lag)
        col = opd_lag[0]
        valid = self.feat[col].notna()
        shifted = self.feat["opd_patients"].shift(1)
        np.testing.assert_allclose(self.feat.loc[valid, col], shifted[valid])

    def test_no_leakage(self):
        self.assertEqual(validate_no_obvious_leakage(self.feat), [])

    def test_saved_features_match_regeneration(self):
        saved = pd.read_csv(ROOT / "data" / "processed" / "hospital_operations_features.csv")
        self.assertEqual(len(saved), len(self.feat))
        self.assertEqual(set(saved.columns), set(self.feat.columns))


class NotebookTests(unittest.TestCase):
    """The EDA notebooks must be executed (saved outputs) and carry written findings."""

    def _load(self, name):
        import json

        return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf8"))

    def test_notebooks_have_outputs_and_no_errors(self):
        for name in ("01_hospital_eda.ipynb", "02_patient_eda.ipynb"):
            nb = self._load(name)
            code = [c for c in nb["cells"] if c["cell_type"] == "code"]
            self.assertTrue(code)
            self.assertTrue(all(c.get("outputs") for c in code), name)
            for c in code:
                self.assertFalse([o for o in c["outputs"] if o["output_type"] == "error"], name)

    def test_notebooks_have_findings_and_name_the_dataset(self):
        text = "\n".join("".join(c["source"]) for c in self._load("01_hospital_eda.ipynb")["cells"])
        self.assertNotIn("fill after running", text)
        self.assertIn("Findings", text)
        self.assertIn("synthetic daily", text)
        self.assertIn("not** the earlier four-CSV", text)

    def test_notebook_numbers_match_the_data(self):
        df = pd.read_csv(ROOT / "data" / "processed" / "hospital_operations_features.csv")
        text = "\n".join("".join(c["source"]) for c in self._load("01_hospital_eda.ipynb")["cells"])
        self.assertIn(f"{int((df.bed_occupancy_percentage >= 70).sum())} days", text)
        self.assertIn(f"{int(df.medicine_stock_units.min()):,}", text)
        self.assertIn(f"{int(df.medicine_stock_units.max()):,}", text)
        self.assertIn(f"{df.loc[df.is_weekend == 1, 'opd_patients'].mean():.0f}", text)


if __name__ == "__main__":
    unittest.main()
