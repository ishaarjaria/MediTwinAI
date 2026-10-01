"""
Shared helpers for MediTwin AI Phase 3 modelling.

Keeps paths, seeds, and metric formatting consistent across train scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RANDOM_SEED = 42


def get_project_root() -> Path:
    """Return meditwin-ai project root (src/models -> parents[2])."""
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    path = get_project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir() -> Path:
    return get_project_root() / "data" / "processed"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
