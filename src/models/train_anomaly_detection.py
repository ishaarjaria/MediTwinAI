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


def train_anomaly_detection(
    contamination: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    df = load_hospital_features()
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
        "model_path": str(model_path),
        "scored_path": str(scored_path),
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
    parser.add_argument("--contamination", type=float, default=0.05)
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
