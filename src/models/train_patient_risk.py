"""
Train patient risk classifiers (Logistic Regression, Random Forest, XGBoost).

Target: risk_level (Low / Medium / High)

Usage (from meditwin-ai root):
    python -m src.models.train_patient_risk
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.utils import (
    RANDOM_SEED,
    ensure_dir,
    models_dir,
    processed_dir,
    save_json,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc

# Features that do NOT include the target or patient_id (no leakage).
NUMERIC_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "oxygen_saturation",
    "previous_admissions",
    "pulse_pressure",
    "chronic_disease_flag",
    "emergency_case_flag",
]
CATEGORICAL_FEATURES = ["gender"]
TARGET = "risk_level"
CLASS_ORDER = ["Low", "Medium", "High"]


def load_patient_features(path=None) -> pd.DataFrame:
    path = path or (processed_dir() / "patient_data_features.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.features.feature_engineering"
        )
    df = pd.read_csv(path)
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Patient features missing columns: {missing}")
    return df


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode gender; optionally scale numerics (needed for LR)."""
    numeric_pipe = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_classifier(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Multiclass metrics use *macro* averaging (unweighted mean across classes)
    so the minority High class is not drowned by Medium. Weighted averages
    are also reported for reference.
    """
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_high": float(
            recall_score(
                y_true, y_pred, labels=labels, average=None, zero_division=0
            )[labels.index("High")]
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Selection rule (justified, not max-accuracy alone):
    1. Highest macro-F1 (balanced across Low/Medium/High).
    2. Tie-break: higher recall on High (minority / higher-stakes class).
    3. Final tie-break: higher accuracy.
    """
    ranked = sorted(
        results,
        key=lambda r: (r["f1_macro"], r["recall_high"], r["accuracy"]),
        reverse=True,
    )
    best = ranked[0]
    best["selection_rationale"] = (
        "Selected by highest macro-F1 (multiclass, unweighted across classes), "
        "then High-class recall, then accuracy — not accuracy alone, because "
        "the Medium class dominates and can inflate accuracy."
    )
    return best


def train_patient_risk(
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    df = load_patient_features()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    candidates: list[tuple[str, Pipeline]] = [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("prep", build_preprocessor(scale_numeric=True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("prep", build_preprocessor(scale_numeric=False)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            min_samples_leaf=3,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "xgboost",
            Pipeline(
                steps=[
                    ("prep", build_preprocessor(scale_numeric=False)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            random_state=seed,
                            n_jobs=-1,
                            # sklearn API uses string labels via enable_categorical
                            # so we map labels for XGB only inside a thin wrapper —
                            # instead encode y for this model.
                        ),
                    ),
                ]
            ),
        ),
    ]

    # XGBoost needs integer class labels; wrap training carefully.
    label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, pipe in candidates:
        if name == "xgboost":
            y_train_enc = y_train.map(label_to_idx)
            y_test_enc = y_test.map(label_to_idx)
            pipe.fit(X_train, y_train_enc)
            pred_idx = pipe.predict(X_test)
            y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
            # Store label maps on the pipeline for inference.
            pipe.label_to_idx_ = label_to_idx
            pipe.idx_to_label_ = idx_to_label
            pipe.uses_encoded_labels_ = True
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            pipe.uses_encoded_labels_ = False

        metrics = evaluate_classifier(name, y_test.to_numpy(), y_pred, CLASS_ORDER)
        results.append(metrics)
        fitted[name] = pipe

        print(f"\n=== {name} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(
            f"Macro P/R/F1: {metrics['precision_macro']:.4f} / "
            f"{metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}"
        )
        print(
            f"Weighted P/R/F1: {metrics['precision_weighted']:.4f} / "
            f"{metrics['recall_weighted']:.4f} / {metrics['f1_weighted']:.4f}"
        )
        print(f"High-class recall: {metrics['recall_high']:.4f}")
        print("Confusion matrix (rows=true, cols=pred)", CLASS_ORDER)
        print(np.array(metrics["confusion_matrix"]))
        print(metrics["classification_report"])

    best_meta = select_best_model(results)
    best_name = best_meta["model"]
    best_pipe = fitted[best_name]

    out_dir = ensure_dir(models_dir() / "patient_risk")
    model_path = out_dir / "best_pipeline.joblib"
    meta_path = out_dir / "metrics.json"

    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_order": CLASS_ORDER,
        "target": TARGET,
        "uses_encoded_labels": bool(getattr(best_pipe, "uses_encoded_labels_", False)),
        "label_to_idx": getattr(best_pipe, "label_to_idx_", None),
        "idx_to_label": getattr(best_pipe, "idx_to_label_", None),
    }
    joblib.dump(artifact, model_path)

    summary = {
        "task": "patient_risk_classification",
        "averaging_note": (
            "Primary multiclass metrics use macro averaging "
            "(unweighted mean over Low/Medium/High). "
            "Weighted averages are also stored."
        ),
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "all_models": results,
        "selected": best_meta,
        "model_path": str(model_path),
    }
    # Drop bulky text from nested all_models copies for JSON? keep report.
    save_json(summary, meta_path)

    print(f"\nSelected model: {best_name}")
    print(best_meta["selection_rationale"])
    print(f"Saved -> {model_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train patient risk classifiers.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    try:
        train_patient_risk(test_size=args.test_size, seed=args.seed)
    except Exception as exc:
        raise SystemExit(f"Patient risk training failed: {exc}") from exc


if __name__ == "__main__":
    main()
