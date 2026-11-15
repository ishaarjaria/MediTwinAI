"""
Regression snapshot: the UI redesign must not change any backend output.

``snapshot()`` evaluates the twin, the What-If simulator, the alert engine and the
patient-risk model on fixed inputs. ``tests/fixtures/backend_snapshot.json`` holds the
values recorded *before* the redesign started. Run this module directly
(``python -m tests.test_backend_snapshot --write``) only when a backend change is
intended; the redesign itself never needs that.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = Path(__file__).parent / "fixtures" / "backend_snapshot.json"


def _round(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return value


def snapshot() -> dict:
    from src.alerts.integration import generate_alerts
    from src.models.prediction_service import get_prediction_service
    from src.simulation.what_if import WhatIfSimulator
    from src.twin.hospital_twin import build_twin_from_latest
    from src.ui import data as ui_data

    twin = build_twin_from_latest()
    service = get_prediction_service()
    current = twin.get_current_state()
    out: dict = {
        "current": _round(current.to_dict()),
        "predicted_day1": _round(twin.get_predicted_state(1).to_dict()),
        "predicted_day7": _round(twin.get_predicted_state(7).to_dict()),
        "model_waiting_time": _round(float(twin.model_waiting_time())),
    }
    sim = WhatIfSimulator(prediction_service=service)
    for name, kwargs in {
        "no_change": {},
        "emergency_surge": {"emergency_cases_percent": 50},
        "medicine_delay": {"medicine_stock_percent": -60},
    }.items():
        res = sim.simulate(current, **kwargs)
        out[f"whatif_{name}"] = {
            "simulated": _round(res.simulated_state.to_dict()),
            "comparisons": _round([c.__dict__ if hasattr(c, "__dict__") else str(c) for c in res.comparisons]),
            "shortfalls": [res.bed_shortfall, res.icu_shortfall],
        }
    report = generate_alerts(twin, days_ahead=1)
    out["alerts"] = sorted(f"{a.category.name}|{a.severity.name}|{a.scope}" for a in report.alerts)
    out["alerts_unavailable"] = sorted(report.unavailable)
    risk = {}
    for label, values in ui_data.PATIENT_PRESETS.items():
        row = service.predict_patient_risk(ui_data.patient_record(values)).iloc[0]
        risk[label] = {
            "level": str(row["predicted_risk_level"]),
            **{k: round(float(row[k]), 6) for k in row.index if str(k).startswith("proba_")},
        }
    out["patient_risk"] = risk
    return out


class TestBackendSnapshot(unittest.TestCase):
    def test_backend_outputs_match_the_pre_redesign_snapshot(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf8"))
        actual = json.loads(json.dumps(snapshot(), default=str))
        for key in expected:
            self.assertEqual(actual[key], expected[key], f"backend output changed: {key}")
        self.assertEqual(set(actual), set(expected))


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(json.loads(json.dumps(snapshot(), default=str)), indent=1, sort_keys=True), encoding="utf8")
        print("wrote", FIXTURE)
    else:
        unittest.main()
