"""Tests for Phase 3 training/evaluation logic (forecast, waiting time, anomaly)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.generate_synthetic_data import generate_hospital_operations_with_truth
from src.models import forecast_models as fm
from src.models import train_anomaly_detection as tad
from src.models import train_opd_forecast as tof
from src.models import train_waiting_time as twt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _series(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(df["date"]),
            "y": df["opd_patients"].astype(float),
            "is_weekend": df["is_weekend"].astype(float),
            "is_holiday": df["is_holiday"].astype(float),
        }
    )


class ForecastSplitAndFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth = generate_hospital_operations_with_truth()
        cls.series = _series(cls.df)

    def test_chronological_split_is_ordered_and_disjoint(self):
        split = tof.chronological_split(len(self.series), 30, 30)
        tr, va, te = split["train"], split["validation"], split["test"]
        self.assertLess(tr.max(), va.min())
        self.assertLess(va.max(), te.min())
        self.assertEqual(len(va), 30)
        self.assertEqual(len(te), 30)
        self.assertEqual(te.max(), len(self.series) - 1)

    def test_split_rejects_short_series_and_tiny_blocks(self):
        with self.assertRaises(ValueError):
            tof.chronological_split(100, 30, 30)
        with self.assertRaises(ValueError):
            tof.chronological_split(366, 3, 30)

    def test_features_only_use_lags_of_seven_days_or_more(self):
        history = dict(zip(self.series["ds"], self.series["y"]))
        day = self.series["ds"].iloc[200]
        row = fm.build_feature_row(day, history)
        self.assertEqual(row["lag_7"], history[day - pd.Timedelta(days=7)])
        self.assertEqual(row["lag_28"], history[day - pd.Timedelta(days=28)])
        # Poisoning the target day and the previous six days must not change features.
        poisoned = dict(history)
        for k in range(0, 7):
            poisoned[day - pd.Timedelta(days=k)] = 1e9
        self.assertEqual(fm.build_feature_row(day, poisoned), row)

    def test_baselines_behave_as_documented(self):
        history = dict(zip(self.series["ds"], self.series["y"]))
        idx = np.arange(40, 120)
        X = fm.build_feature_frame(self.series["ds"].iloc[idx], history)
        y = self.series["y"].to_numpy()[idx]
        base = fm.make_baselines()
        mean_pred = base["Training mean"].fit(X, y).predict(X)
        self.assertTrue(np.allclose(mean_pred, y.mean()))
        naive = base["Previous week, same day"].fit(X, y).predict(X)
        self.assertTrue(np.allclose(naive, X["lag_7"]))
        wd = base["Day-of-week mean"].fit(X, y)
        pred = wd.predict(X)
        self.assertLess(np.mean(np.abs(pred - y)), np.mean(np.abs(mean_pred - y)))

    def test_recursive_forecast_extends_beyond_seven_days(self):
        history = {pd.Timestamp(d): float(v) for d, v in zip(self.series["ds"], self.series["y"])}
        idx = np.arange(28, len(self.series))
        X = fm.build_feature_frame(self.series["ds"].iloc[idx], history)
        model = fm.make_baselines()["Day-of-week mean"].fit(X, self.series["y"].to_numpy()[idx])
        out = fm.recursive_forecast(model, history, self.series["ds"].iloc[-1], 21)
        self.assertEqual(len(out), 21)
        self.assertTrue((out["yhat"] >= 0).all())
        self.assertEqual(out["date"].iloc[0], self.series["ds"].iloc[-1] + pd.Timedelta(days=1))

    def test_metrics_and_interval_helpers(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        p = np.array([12.0, 18.0, 33.0, 36.0])
        m = tof.evaluate_forecast(y, p)
        self.assertAlmostEqual(m["mae"], 2.75)
        self.assertAlmostEqual(m["mse"], 8.25)
        self.assertAlmostEqual(m["rmse"], 8.25 ** 0.5)
        self.assertLess(m["r2"], 1.0)
        half = tof.empirical_interval_halfwidth(y, p, 0.5)
        self.assertGreater(half, 0)
        self.assertAlmostEqual(tof.interval_coverage(y, p, 100.0), 1.0)
        self.assertAlmostEqual(tof.interval_coverage(y, p, 0.0), 0.0)

    def test_deployed_forecast_shape_and_band(self):
        artifact = joblib.load(MODELS / "opd_forecast" / "forecaster.joblib")
        out = tof.forecast_opd_from_artifact(artifact, 7)
        self.assertEqual(list(out.columns), ["date", "opd_forecast", "yhat_lower", "yhat_upper"])
        self.assertTrue((out["yhat_lower"] <= out["opd_forecast"]).all())
        self.assertTrue((out["opd_forecast"] <= out["yhat_upper"]).all())
        pd.testing.assert_frame_equal(out, tof.forecast_opd_from_artifact(artifact, 7))  # deterministic
        with self.assertRaises(ValueError):
            tof.forecast_opd_from_artifact(artifact, 0)


class SavedMetricsTests(unittest.TestCase):
    def test_forecast_metrics_are_complete_and_honest(self):
        meta = json.loads((MODELS / "opd_forecast" / "metrics.json").read_text(encoding="utf8"))
        names = {c["model"] for c in meta["comparison"]}
        for required in ("Training mean", "Day-of-week mean", "Previous week, same day"):
            self.assertIn(required, names)
        evaluated = [c for c in meta["comparison"] if "test" in c]
        for c in evaluated:
            for key in ("mae", "mse", "rmse", "r2"):
                self.assertIn(key, c["test"])
        self.assertEqual(
            meta["selected_model"],
            min(evaluated, key=lambda c: c["validation"]["mae"])["model"],
        )  # selected on validation only
        # The interval is only called validated if measured test coverage supports it.
        iv = meta["interval"]
        self.assertEqual(iv["validated_80pct"], abs(iv["coverage_test"] - 0.8) <= 0.05)

    def test_forecast_metrics_reproduce_from_holdout_file(self):
        meta = json.loads((MODELS / "opd_forecast" / "metrics.json").read_text(encoding="utf8"))
        hold = pd.read_csv(MODELS / "opd_forecast" / "holdout_predictions.csv")
        m = tof.evaluate_forecast(hold["y_true"], hold["yhat"])
        self.assertAlmostEqual(m["mae"], meta["metrics_holdout"]["mae"], places=6)
        self.assertAlmostEqual(m["r2"], meta["metrics_holdout"]["r2"], places=6)

    def test_waiting_time_split_baselines_and_selection(self):
        df = twt.load_hospital_features()
        tr, va, te = twt.three_way_split(df)
        self.assertLess(tr["date"].max(), va["date"].min())
        self.assertLess(va["date"].max(), te["date"].min())
        self.assertEqual(len(tr) + len(va) + len(te), len(df))
        base = twt.baseline_predictions(pd.concat([tr, va]), te)
        self.assertEqual(set(base), {"training_mean", "opd_only_linear", "day_of_week_mean"})
        meta = json.loads((MODELS / "waiting_time" / "metrics.json").read_text(encoding="utf8"))
        best_val = min(meta["all_models"], key=lambda r: (r["validation"]["mae"], r["validation"]["rmse"]))
        self.assertEqual(meta["selected"]["model"], best_val["model"])
        self.assertIn("baselines", meta)
        self.assertEqual(
            meta["selected"]["beats_best_baseline"],
            meta["selected"]["rmse"] < min(b["rmse"] for b in meta["baselines"].values()),
        )

    def test_waiting_time_metrics_reproduce(self):
        df = twt.load_hospital_features()
        tr, va, te = twt.three_way_split(df)
        art = joblib.load(MODELS / "waiting_time" / "best_pipeline.joblib")
        # The deployed model is refit on all data (so it has seen the test block);
        # check the saved test metrics by refitting the selected model on train+val.
        name = art["model_name"]
        pipe = dict(twt._candidate_pipelines(42))[name]
        dev = pd.concat([tr, va])
        pipe.fit(dev[twt.FEATURE_COLUMNS], dev[twt.TARGET])
        m = twt.regression_metrics(name, te[twt.TARGET], pipe.predict(te[twt.FEATURE_COLUMNS]))
        meta = json.loads((MODELS / "waiting_time" / "metrics.json").read_text(encoding="utf8"))
        self.assertAlmostEqual(m["rmse"], meta["selected"]["rmse"], places=4)

    def test_waiting_time_direction_of_effects(self):
        art = joblib.load(MODELS / "waiting_time" / "best_pipeline.joblib")
        cols = art["feature_columns"]
        base = pd.DataFrame([{c: 0.0 for c in cols}])
        df = twt.load_hospital_features()
        base = df[cols].median().to_frame().T
        p0 = float(art["pipeline"].predict(base)[0])
        for col, delta, sign in (("opd_patients", 60, 1), ("doctors_available", 10, -1)):
            hi = base.copy()
            hi[col] = hi[col] + delta
            self.assertGreater(sign * (float(art["pipeline"].predict(hi)[0]) - p0), 0, col)


class AnomalyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = tad.load_hospital_features()
        cls.truth = tad.load_injected_truth()

    def test_ground_truth_available(self):
        self.assertIsNotNone(self.truth)
        self.assertEqual(len(self.truth), 5)

    def test_forward_chaining_folds_never_train_on_the_future(self):
        folds = tad.forward_chaining_folds(len(self.df))
        self.assertEqual(folds[0][0], 60)
        self.assertEqual(folds[-1][1], len(self.df))
        for (s1, e1), (s2, e2) in zip(folds, folds[1:]):
            self.assertEqual(e1, s2)

    def test_precision_recall_f1_arithmetic(self):
        r = tad._prf(np.array([1, 1, 0, 0, 0, 1]), np.array([1, 0, 1, 0, 0, 1]))
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (2, 1, 1))
        self.assertAlmostEqual(r["precision"], 2 / 3)
        self.assertAlmostEqual(r["recall"], 2 / 3)
        self.assertAlmostEqual(r["f1"], 2 / 3)
        self.assertEqual(tad._prf(np.zeros(3), np.zeros(3))["f1"], 0.0)

    def test_evaluation_reports_and_excludes_unscorable_spikes(self):
        res = tad.evaluate_against_injected(self.df, self.truth, 0.02)
        self.assertEqual(res["n_injected_total"], 5)
        self.assertEqual(res["n_injected_in_scored"] + len(res["excluded_dates_no_history"]), 5)
        self.assertGreaterEqual(res["recall"], 0.5)
        self.assertLessEqual(res["n_flagged"], 0.2 * res["n_days_scored"])

    def test_detector_flags_an_obvious_synthetic_spike(self):
        art = joblib.load(MODELS / "anomaly_detection" / "isolation_forest.joblib")
        day = self.df[tad.FEATURE_COLUMNS].median().to_frame().T
        day["opd_patients"] *= 3
        day["emergency_patients"] *= 4
        day["icu_occupancy_percentage"] = 100.0
        day["average_waiting_time_minutes"] *= 4
        flag = art["model"].predict(art["scaler"].transform(day.astype(float)))[0]
        self.assertEqual(flag, -1)
        typical = self.df[tad.FEATURE_COLUMNS].median().to_frame().T
        self.assertEqual(art["model"].predict(art["scaler"].transform(typical.astype(float)))[0], 1)

    def test_saved_metrics_record_evaluation_and_warning(self):
        meta = json.loads((MODELS / "anomaly_detection" / "metrics.json").read_text(encoding="utf8"))
        self.assertIn("isolation_forest", meta["evaluation"])
        self.assertIn("local_outlier_factor", meta["evaluation"])
        self.assertIn("small_sample_warning", meta["evaluation"])

    def test_no_january_warmup_cluster_of_flags(self):
        scored = pd.read_csv(MODELS / "anomaly_detection" / "scored_operations.csv", parse_dates=["date"])
        jan_first_week = scored[scored["date"] <= "2024-01-07"]["anomaly_flag"].sum()
        self.assertLessEqual(int(jan_first_week), 1)


class TrainAllSkipTests(unittest.TestCase):
    """`train_all --skip X` must keep model X untouched and reuse its saved metrics."""

    def test_skipped_models_are_not_retrained(self):
        import tempfile
        from unittest import mock

        from src.models import train_all as ta

        names = ("patient_risk", "waiting_time", "opd_forecast", "anomaly_detection")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in names:
                (root / n).mkdir()
                (root / n / "metrics.json").write_text(json.dumps({"marker": n}), encoding="utf8")
            boom = mock.Mock(side_effect=AssertionError("a skipped model was retrained"))
            with mock.patch.object(ta, "models_dir", return_value=root), \
                 mock.patch.object(ta, "train_patient_risk", boom), \
                 mock.patch.object(ta, "train_waiting_time", boom), \
                 mock.patch.object(ta, "train_opd_forecast", boom), \
                 mock.patch.object(ta, "train_anomaly_detection", boom), \
                 mock.patch.object(ta, "_print_comparison"):
                summary = ta.train_all(skip=names)
            boom.assert_not_called()
            self.assertEqual({k: v["marker"] for k, v in summary["components"].items()}, {n: n for n in names})
            self.assertTrue((root / "phase3_model_comparison.json").is_file())

    def test_unskipped_models_are_trained(self):
        import tempfile
        from unittest import mock

        from src.models import train_all as ta

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "patient_risk").mkdir()
            (root / "patient_risk" / "metrics.json").write_text(json.dumps({"marker": "kept"}), encoding="utf8")
            called = []
            fake = lambda tag: (lambda: called.append(tag) or {"marker": tag})  # noqa: E731
            with mock.patch.object(ta, "models_dir", return_value=root), \
                 mock.patch.object(ta, "train_patient_risk", fake("patient_risk")), \
                 mock.patch.object(ta, "train_waiting_time", fake("waiting_time")), \
                 mock.patch.object(ta, "train_opd_forecast", fake("opd_forecast")), \
                 mock.patch.object(ta, "train_anomaly_detection", fake("anomaly_detection")), \
                 mock.patch.object(ta, "_print_comparison"):
                summary = ta.train_all(skip=("patient_risk",))
            self.assertEqual(called, ["waiting_time", "opd_forecast", "anomaly_detection"])
            self.assertEqual(summary["components"]["patient_risk"]["marker"], "kept")


if __name__ == "__main__":
    unittest.main()
