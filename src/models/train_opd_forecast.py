"""
OPD demand forecasting: honest model comparison on chronological splits.

Candidates
----------
Baselines : training mean, day-of-week mean, previous-week same-day,
            same-weekday 4-week average.
Models    : Ridge and XGBoost on weekday/calendar features plus seasonal lags,
            and Prophet (optional; skipped with a recorded reason if its Stan
            backend cannot run on this machine).

Protocol
--------
Chronological split: train -> validation (30 days) -> test (30 days). The
model is **selected on validation MAE only**; the test block is reported once.
Every evaluated day is scored as a <=7-day-ahead forecast (features use lags
of >= 7 days; Prophet is refit at each weekly origin). The deployed model is
refit on all data. Intervals are empirical (absolute-residual quantile on the
validation block) and their *measured* test coverage is stored - they are not
claimed to be a validated 80 % interval unless the test coverage supports it.

Usage:
    python -m src.models.train_opd_forecast
"""

from __future__ import annotations

import argparse
import logging
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.models.forecast_models import (
    MIN_HISTORY_DAYS,
    build_feature_frame,
    is_holiday,
    make_baselines,
    make_candidates,
    recursive_forecast,
)
from src.models.utils import (
    RANDOM_SEED,
    ensure_dir,
    models_dir,
    processed_dir,
    rmse,
    save_json,
)

NOMINAL_INTERVAL = 0.8
ARTIFACT_NAME = "forecaster.joblib"


def load_opd_series(path=None) -> pd.DataFrame:
    path = path or (processed_dir() / "hospital_operations_features.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.features.feature_engineering"
        )
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return pd.DataFrame(
        {
            "ds": df["date"],
            "y": df["opd_patients"].astype(float),
            "is_weekend": df["is_weekend"].astype(float),
            "is_holiday": df["is_holiday"].astype(float),
        }
    )


def chronological_split(n: int, val_days: int = 30, test_days: int = 30) -> dict[str, np.ndarray]:
    """Index blocks for train / validation / test in time order."""
    if val_days < 7 or test_days < 7:
        raise ValueError("val_days and test_days should each be at least 7.")
    if n <= val_days + test_days + MIN_HISTORY_DAYS + 60:
        raise ValueError("Series too short for the requested validation/test blocks.")
    i_test = n - test_days
    i_val = i_test - val_days
    return {
        "train": np.arange(MIN_HISTORY_DAYS, i_val),  # first 28 days only supply lags
        "validation": np.arange(i_val, i_test),
        "test": np.arange(i_test, n),
    }


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def empirical_interval_halfwidth(y_true, y_pred, coverage: float = NOMINAL_INTERVAL) -> float:
    """Absolute-residual quantile; the band is ``yhat +/- halfwidth``."""
    return float(np.quantile(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)), coverage))


def interval_coverage(y_true, y_pred, halfwidth: float) -> float:
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    return float(np.mean(err <= halfwidth))


def _prophet_rolling(series: pd.DataFrame, block: np.ndarray, step: int = 7):
    """Refit Prophet at each weekly origin; return predictions or raise."""
    from prophet import Prophet  # local import: optional dependency path

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    preds: list[float] = []
    for s in range(0, len(block), step):
        origin = int(block[0]) + s
        chunk = block[s : s + step]
        train = series.iloc[:origin]
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=NOMINAL_INTERVAL,
        )
        model.add_regressor("is_weekend")
        model.add_regressor("is_holiday")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[["ds", "y", "is_weekend", "is_holiday"]])
            fut = series.iloc[chunk][["ds", "is_weekend", "is_holiday"]]
            preds.extend(model.predict(fut)["yhat"].tolist())
    return np.asarray(preds, float)


def _score_blocks(series: pd.DataFrame, split: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    """Fit every candidate on data before each block and score the block."""
    y = series["y"].to_numpy()
    history_all = dict(zip(series["ds"], y))
    results: dict[str, dict[str, Any]] = {}
    preds: dict[str, dict[str, np.ndarray]] = {}

    for block_name in ("validation", "test"):
        block = split[block_name]
        # Only data strictly before the block may be used for fitting.
        fit_idx = np.arange(MIN_HISTORY_DAYS, int(block[0]))
        # Features use lags >= 7 days, so block rows only need *observed* history.
        X_fit = build_feature_frame(series["ds"].iloc[fit_idx], history_all)
        X_eval = build_feature_frame(series["ds"].iloc[block], history_all)
        y_fit = y[fit_idx]
        y_eval = y[block]

        models = {**make_baselines(), **make_candidates(seed)}
        for name, model in models.items():
            model.fit(X_fit, y_fit)
            p = np.asarray(model.predict(X_eval), float)
            preds.setdefault(name, {})[block_name] = p
            results.setdefault(name, {})[block_name] = evaluate_forecast(y_eval, p)

        try:
            p = _prophet_rolling(series, block)
            preds.setdefault("Prophet (weekly + calendar)", {})[block_name] = p
            results.setdefault("Prophet (weekly + calendar)", {})[block_name] = evaluate_forecast(y_eval, p)
        except Exception as exc:  # Stan backend blocked / unavailable
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            results.setdefault("Prophet (weekly + calendar)", {})["skipped_reason"] = reason
    return {"results": results, "predictions": preds}


def train_opd_forecast(
    val_days: int = 30,
    test_days: int = 30,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    series = load_opd_series()
    split = chronological_split(len(series), val_days, test_days)
    y = series["y"].to_numpy()

    scored = _score_blocks(series, split, seed)
    results = scored["results"]
    preds = scored["predictions"]

    baseline_names = list(make_baselines().keys())
    evaluated = [n for n, r in results.items() if "validation" in r and "test" in r]
    candidate_names = [n for n in evaluated if n not in baseline_names]
    # Selection uses validation MAE only, across everything that could be evaluated.
    selected = min(evaluated, key=lambda n: results[n]["validation"]["mae"])
    best_baseline = min(baseline_names, key=lambda n: results[n]["validation"]["mae"])
    sel_val = results[selected]["validation"]
    sel_test = results[selected]["test"]

    # Empirical interval: calibrated on validation, coverage measured on test.
    halfwidth = empirical_interval_halfwidth(y[split["validation"]], preds[selected]["validation"])
    coverage_test = interval_coverage(y[split["test"]], preds[selected]["test"], halfwidth)
    coverage_val = interval_coverage(y[split["validation"]], preds[selected]["validation"], halfwidth)

    # Deploy: refit the selected model on all data. (Prophet is only deployed
    # through the same generic path when it was selected and evaluated.)
    history = {pd.Timestamp(d): float(v) for d, v in zip(series["ds"], y)}
    fit_idx = np.arange(MIN_HISTORY_DAYS, len(series))
    X_all = build_feature_frame(series["ds"].iloc[fit_idx], history)
    models_now = {**make_baselines(), **make_candidates(seed)}
    if selected in models_now:
        deployed = models_now[selected].fit(X_all, y[fit_idx])
        kind = "feature_model"
    else:  # Prophet selected: refit on all data
        from prophet import Prophet

        deployed = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        deployed.add_regressor("is_weekend")
        deployed.add_regressor("is_holiday")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deployed.fit(series[["ds", "y", "is_weekend", "is_holiday"]])
        kind = "prophet"

    out_dir = ensure_dir(models_dir() / "opd_forecast")
    model_path = out_dir / ARTIFACT_NAME
    artifact = {
        "kind": kind,
        "model": deployed,
        "model_name": selected,
        "last_train_date": str(series["ds"].max().date()),
        "history": history,
        "interval_halfwidth": halfwidth,
        "interval_note": (
            f"Empirical +/- band (absolute-residual {int(NOMINAL_INTERVAL * 100)}th percentile on "
            f"validation). Measured coverage on the test block: {coverage_test:.0%}."
        ),
        "target": "opd_patients",
    }
    joblib.dump(artifact, model_path)

    # Per-day held-out predictions for transparency (test block, selected model).
    holdout = pd.DataFrame(
        {
            "ds": series["ds"].iloc[split["test"]].to_numpy(),
            "y_true": y[split["test"]],
            "yhat": preds[selected]["test"],
        }
    )
    holdout["yhat_lower"] = (holdout["yhat"] - halfwidth).clip(lower=0)
    holdout["yhat_upper"] = holdout["yhat"] + halfwidth
    holdout_path = out_dir / "holdout_predictions.csv"
    holdout.to_csv(holdout_path, index=False)

    comparison = []
    for name, r in results.items():
        if "validation" in r and "test" in r:
            comparison.append(
                {
                    "model": name,
                    "type": "baseline" if name in baseline_names else "model",
                    "validation": r["validation"],
                    "test": r["test"],
                }
            )
        else:
            comparison.append({"model": name, "type": "model", "skipped_reason": r.get("skipped_reason")})
    best_test_baseline = min(baseline_names, key=lambda n: results[n]["test"]["mae"])

    summary = {
        "task": "opd_forecasting",
        "model": selected,
        "selected_model": selected,
        "selection_rule": "lowest validation MAE among all evaluated candidates",
        "split": "chronological train / validation / test",
        "validation_days": val_days,
        "test_days": test_days,
        "n_train": int(len(split["train"])),
        "train_end_date": str(series["ds"].iloc[split["train"][-1]].date()),
        "validation_start_date": str(series["ds"].iloc[split["validation"][0]].date()),
        "test_start_date": str(series["ds"].iloc[split["test"][0]].date()),
        "test_end_date": str(series["ds"].iloc[split["test"][-1]].date()),
        "evaluation_horizon": "each day scored as a forecast of at most 7 days ahead",
        "metrics_validation": sel_val,
        "metrics_holdout": sel_test,
        "comparison": comparison,
        "best_baseline_by_validation": best_baseline,
        "best_baseline_by_test": best_test_baseline,
        "selected_beats_best_baseline_on_test": bool(
            sel_test["mae"] < results[best_test_baseline]["test"]["mae"]
        ),
        "interval": {
            "method": "empirical absolute-residual quantile (validation)",
            "nominal": NOMINAL_INTERVAL,
            "halfwidth": halfwidth,
            "coverage_validation": coverage_val,
            "coverage_test": coverage_test,
            "validated_80pct": bool(abs(coverage_test - NOMINAL_INTERVAL) <= 0.05),
        },
        "prophet_note": results.get("Prophet (weekly + calendar)", {}).get("skipped_reason"),
        "model_path": str(model_path.name),
        "holdout_predictions_path": str(holdout_path.name),
        "note": (
            "Metrics come from models fit only on data before each evaluated block. "
            "The saved artifact is refit on all data for future forecasts."
        ),
    }
    save_json(summary, out_dir / "metrics.json")

    print("\n=== OPD forecast comparison (MAE on validation / test) ===")
    for c in comparison:
        if "validation" in c:
            print(f"  {c['model']:42s} val={c['validation']['mae']:7.2f}  test={c['test']['mae']:7.2f}  R2(test)={c['test']['r2']:.3f}")
        else:
            print(f"  {c['model']:42s} SKIPPED ({c['skipped_reason']})")
    print(f"Selected: {selected}   interval test coverage={coverage_test:.0%}")
    print(f"Saved model -> {model_path}")
    return summary


def forecast_opd_from_artifact(artifact: dict[str, Any], days: int = 7) -> pd.DataFrame:
    """Produce a multi-day OPD forecast from a saved forecaster artifact."""
    if days < 1:
        raise ValueError("days must be >= 1")
    last_date = pd.Timestamp(artifact["last_train_date"])
    if artifact.get("kind") == "prophet":  # pragma: no cover - needs Stan to train
        model = artifact["model"]
        future = pd.DataFrame(
            {"ds": pd.date_range(last_date + pd.Timedelta(days=1), periods=days, freq="D")}
        )
        future["is_weekend"] = (future["ds"].dt.dayofweek >= 5).astype(float)
        future["is_holiday"] = [float(is_holiday(d)) for d in future["ds"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = model.predict(future)
        frame = pd.DataFrame({"date": future["ds"], "yhat": fc["yhat"].to_numpy()})
    else:
        frame = recursive_forecast(artifact["model"], artifact["history"], last_date, days)
    half = float(artifact.get("interval_halfwidth", 0.0))
    out = pd.DataFrame(
        {
            "date": frame["date"],
            "opd_forecast": frame["yhat"].clip(lower=0).round(1),
            "yhat_lower": (frame["yhat"] - half).clip(lower=0).round(1),
            "yhat_upper": (frame["yhat"] + half).clip(lower=0).round(1),
        }
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare OPD forecast models.")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        train_opd_forecast(val_days=args.val_days, test_days=args.test_days, seed=args.seed)
    except Exception as exc:
        raise SystemExit(f"OPD forecast training failed: {exc}") from exc


if __name__ == "__main__":
    main()
