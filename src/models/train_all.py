"""
Train all Phase 3 MediTwin AI models and write a comparison summary.

Usage (from meditwin-ai root):
    python -m src.models.train_all
"""

from __future__ import annotations

import traceback
from typing import Any

from src.models.train_anomaly_detection import train_anomaly_detection
from src.models.train_opd_forecast import train_opd_forecast
from src.models.train_patient_risk import train_patient_risk
from src.models.train_waiting_time import train_waiting_time
from src.models.utils import ensure_dir, load_json, models_dir, save_json


def train_all(skip: tuple[str, ...] = ()) -> dict[str, Any]:
    summary: dict[str, Any] = {"phase": 3, "components": {}}
    errors: dict[str, str] = {}

    steps = [
        ("patient_risk", train_patient_risk),
        ("waiting_time", train_waiting_time),
        ("opd_forecast", train_opd_forecast),
        ("anomaly_detection", train_anomaly_detection),
    ]

    for name, fn in steps:
        if name in skip:
            # Keep the existing trained model and reuse its saved metrics.
            metrics_path = models_dir() / name / "metrics.json"
            if metrics_path.is_file():
                summary["components"][name] = load_json(metrics_path)
                print(f"Skipping {name} (reusing {metrics_path.parent.name}/{metrics_path.name})")
            else:
                errors[name] = f"Skipped, but {metrics_path} does not exist."
            continue
        print("\n" + "#" * 64)
        print(f"# Training: {name}")
        print("#" * 64)
        try:
            summary["components"][name] = fn()
        except Exception as exc:
            errors[name] = f"{exc}\n{traceback.format_exc()}"
            print(f"FAILED {name}: {exc}")

    summary["errors"] = errors
    out_path = ensure_dir(models_dir()) / "phase3_model_comparison.json"
    save_json(summary, out_path)

    print("\n" + "=" * 64)
    print("PHASE 3 MODEL COMPARISON SUMMARY")
    print("=" * 64)
    _print_comparison(summary)
    print(f"\nFull summary JSON -> {out_path}")

    if errors:
        raise SystemExit(
            "One or more model training steps failed: " + ", ".join(errors.keys())
        )
    return summary


def _print_comparison(summary: dict[str, Any]) -> None:
    comps = summary.get("components", {})

    if "patient_risk" in comps:
        pr = comps["patient_risk"]
        print("\n[A] Patient Risk Classification")
        print("  Multiclass averaging: MACRO (primary), weighted also reported.")
        for m in pr.get("all_models", []):
            print(
                f"  - {m['model']}: acc={m['accuracy']:.4f}  "
                f"macroF1={m['f1_macro']:.4f}  highRecall={m['recall_high']:.4f}"
            )
        sel = pr.get("selected", {})
        print(f"  SELECTED: {sel.get('model')}")
        print(f"  Reason: {sel.get('selection_rationale')}")

    if "waiting_time" in comps:
        wt = comps["waiting_time"]
        print("\n[B] Waiting-Time Regression (chronological holdout)")
        for m in wt.get("all_models", []):
            print(
                f"  - {m['model']}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                f"R2={m['r2']:.4f}"
            )
        sel = wt.get("selected", {})
        print(f"  SELECTED: {sel.get('model')}")
        print(f"  Reason: {sel.get('selection_rationale')}")

    if "opd_forecast" in comps:
        fc = comps["opd_forecast"]
        print("\n[C] OPD Forecasting (chronological validation/test comparison)")
        for c in fc.get("comparison", []):
            if "test" in c:
                print(
                    f"  - {c['model']}: test MAE={c['test']['mae']:.2f}  "
                    f"RMSE={c['test']['rmse']:.2f}  R2={c['test']['r2']:.3f}"
                )
            else:
                print(f"  - {c['model']}: skipped ({c.get('skipped_reason')})")
        print(f"  SELECTED: {fc.get('selected_model')}")

    if "anomaly_detection" in comps:
        an = comps["anomaly_detection"]
        print("\n[D] Anomaly Detection (Isolation Forest)")
        print(
            f"  Flagged {an.get('n_anomalies_flagged')} / {an.get('n_rows')} "
            f"(contamination={an.get('contamination')})"
        )

    if summary.get("errors"):
        print("\nErrors:")
        for k, v in summary["errors"].items():
            print(f"  - {k}: {v.splitlines()[0]}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train all MediTwin AI models.")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["patient_risk", "waiting_time", "opd_forecast", "anomaly_detection"],
        help="Models to keep as they are (their saved metrics are reused).",
    )
    train_all(skip=tuple(parser.parse_args().skip))


if __name__ == "__main__":
    main()
