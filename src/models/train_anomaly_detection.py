"""
Hospital anomaly detection with Isolation Forest.

Usage:
    python -m src.models.train_anomaly_detection
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.models.utils import (
    RANDOM_SEED,
    ensure_dir,
    get_project_root,
    models_dir,
    processed_dir,
    save_json,
)

FEATURE_COLUMNS = [
    "opd_patients",
    "emergency_patients",
    "bed_occupancy_percentage",
    "icu_occupancy_percentage",
    "average_waiting_time_minutes",
]


def load_hospital_features(path=None) -> pd.DataFrame:
    path = path or (processed_dir() / "hospital_operations_features.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.features.feature_engineering"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    missing = set(FEATURE_COLUMNS + ["date"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for anomaly detection: {missing}")
    return df


def explain_anomaly(
    row: pd.Series,
    feature_means: pd.Series,
    feature_stds: pd.Series,
    z_threshold: float = 2.0,
) -> str:
    """
    Build a short human-readable reason from z-score deviations.

    Educational heuristic only — not a clinical explanation.
    """
    reasons: list[str] = []
    labels = {
        "opd_patients": "OPD volume",
        "emergency_patients": "emergency volume",
        "bed_occupancy_percentage": "bed occupancy",
        "icu_occupancy_percentage": "ICU occupancy",
        "average_waiting_time_minutes": "waiting time",
    }
    for col in FEATURE_COLUMNS:
        std = float(feature_stds[col])
        if std < 1e-8:
            continue
        z = (float(row[col]) - float(feature_means[col])) / std
        if abs(z) >= z_threshold:
            direction = "unusually high" if z > 0 else "unusually low"
            reasons.append(f"{labels[col]} {direction} (z={z:.1f})")
    if not reasons:
        return "Flagged by Isolation Forest multivariate pattern (no single-feature z>=2)."
    return "; ".join(reasons)


def load_injected_truth(path=None) -> pd.DataFrame | None:
    """Ground-truth injected spike days from the generator (None if absent)."""
    path = path or (get_project_root() / "data" / "synthetic" / "injected_anomalies.csv")
    if not path.is_file():
        return None
    return pd.read_csv(path)


def forward_chaining_folds(n: int, first_start: int = 60, n_folds: int = 4) -> list[tuple[int, int]]:
    """(start, end) of each scored block; training uses only days before ``start``."""
    if n < first_start + n_folds * 7:
        raise ValueError("Series too short for forward-chaining evaluation.")
    edges = np.linspace(first_start, n, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]


def _prf(y_true: np.ndarray, y_flag: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_flag == 1)).sum())
    fp = int(((y_true == 0) & (y_flag == 1)).sum())
    fn = int(((y_true == 1) & (y_flag == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def evaluate_against_injected(
    df: pd.DataFrame,
    truth: pd.DataFrame,
    contamination: float,
    seed: int = RANDOM_SEED,
    method: str = "isolation_forest",
) -> dict[str, Any]:
    """
    Score each block using a detector fitted ONLY on earlier days.

    Labels are never used for fitting (the detectors are unsupervised); they
    are used only to count hits. Spikes before the first block (no history to
    fit on) are excluded and reported.
    """
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    y_true_all = np.isin(dates, truth["date"].to_numpy()).astype(int)
    folds = forward_chaining_folds(len(df))
    flags = np.zeros(len(df), dtype=int)
    scored = np.zeros(len(df), dtype=bool)
    for start, end in folds:
        scaler = StandardScaler().fit(X[:start])
        if method == "isolation_forest":
            model = IsolationForest(
                n_estimators=200, contamination=contamination, random_state=seed, n_jobs=1
            ).fit(scaler.transform(X[:start]))
        else:
            from sklearn.neighbors import LocalOutlierFactor

            model = LocalOutlierFactor(
                n_neighbors=20, contamination=contamination, novelty=True
            ).fit(scaler.transform(X[:start]))
        flags[start:end] = (model.predict(scaler.transform(X[start:end])) == -1).astype(int)
        scored[start:end] = True
    res = _prf(y_true_all[scored], flags[scored])
    res.update(
        {
            "method": method,
            "contamination": contamination,
            "n_days_scored": int(scored.sum()),
            "n_injected_in_scored": int(y_true_all[scored].sum()),
            "n_injected_total": int(y_true_all.sum()),
            "n_flagged": int(flags[scored].sum()),
            "missed_dates": [str(d) for d in dates[(y_true_all == 1) & scored & (flags == 0)]],
            "excluded_dates_no_history": [str(d) for d in dates[(y_true_all == 1) & ~scored]],
        }
    )
    return res


def train_anomaly_detection(
    contamination: float | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    df = load_hospital_features()
    truth = load_injected_truth()
    evaluation: dict[str, Any] | None = None
    if truth is not None:
        grid = [0.01, 0.02, 0.03, 0.05]
        iforest = [evaluate_against_injected(df, truth, c, seed) for c in grid]
        lof = [evaluate_against_injected(df, truth, c, seed, method="lof") for c in grid]
        if contamination is None:
            contamination = max(iforest, key=lambda r: (r["f1"], -r["contamination"]))["contamination"]
        evaluation = {
            "protocol": (
                "Forward chaining: each block is scored by a detector fitted only on "
                "earlier days; the injected-spike labels are never used for fitting."
            ),
            "isolation_forest": iforest,
            "local_outlier_factor": lof,
            "small_sample_warning": (
                f"Only {iforest[0]['n_injected_in_scored']} injected spikes are scored; "
                "precision/recall are indicative, not statistically stable."
            ),
        }
    if contamination is None:
        contamination = 0.05
    X = df[FEATURE_COLUMNS].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal. Convert to anomaly score
    # where higher means more anomalous for easier reading.
    raw_scores = model.decision_function(X_scaled)
    anomaly_scores = (-raw_scores).astype(float)
    flags = (model.predict(X_scaled) == -1).astype(int)

    means = X.mean()
    stds = X.std(ddof=0)

    reasons = [
        explain_anomaly(X.iloc[i], means, stds) if flags[i] == 1 else ""
        for i in range(len(X))
    ]

    scored = df[["date"]].copy()
    scored["anomaly_flag"] = flags
    scored["anomaly_score"] = np.round(anomaly_scores, 4)
    scored["anomaly_reason"] = reasons
    for col in FEATURE_COLUMNS:
        scored[col] = df[col].values

    out_dir = ensure_dir(models_dir() / "anomaly_detection")
    model_path = out_dir / "isolation_forest.joblib"
    scored_path = out_dir / "scored_operations.csv"
    meta_path = out_dir / "metrics.json"

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "feature_means": means.to_dict(),
        "feature_stds": stds.to_dict(),
        "contamination": contamination,
        "model_name": "isolation_forest",
    }
    joblib.dump(artifact, model_path)
    scored.to_csv(scored_path, index=False)

    n_anom = int(flags.sum())
    summary = {
        "task": "hospital_anomaly_detection",
        "model": "isolation_forest",
        "contamination": contamination,
        "n_rows": int(len(df)),
        "n_anomalies_flagged": n_anom,
        "anomaly_rate": float(n_anom / len(df)),
        "feature_columns": FEATURE_COLUMNS,
        "score_note": (
            "anomaly_score is -decision_function (higher = more anomalous). "
            "anomaly_flag=1 means IsolationForest predict==-1."
        ),
        "model_path": str(model_path.name),
        "scored_path": str(scored_path.name),
        "evaluation": evaluation,
        "contamination_selection": (
            "chosen by pooled F1 on the forward-chaining evaluation"
            if evaluation
            else "default (no ground-truth file found)"
        ),
        "in_sample_note": (
            "scored_operations.csv flags are an in-sample scoring of the final model "
            "(fit on all days) used only for display; use `evaluation` for accuracy."
        ),
        "example_anomaly_dates": [
            str(d.date())
            for d in scored.loc[scored["anomaly_flag"] == 1, "date"].head(10)
        ],
    }
    save_json(summary, meta_path)

    print("\n=== Isolation Forest anomaly detection ===")
    print(f"Rows={len(df)}  Flagged anomalies={n_anom} ({100 * n_anom / len(df):.1f}%)")
    print(f"Saved model -> {model_path}")
    print(f"Scored table -> {scored_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomalies.")
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        train_anomaly_detection(
            contamination=args.contamination, seed=args.seed
        )
    except Exception as exc:
        raise SystemExit(f"Anomaly training failed: {exc}") from exc


if __name__ == "__main__":
    main()
