"""
Alert data models and configurable thresholds (Phase 6).

Alerts are operational (beds, ICU, OPD load, medicine, staffing, unusual
activity). They never describe a patient or give medical advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Severity(IntEnum):
    """Ordered severity so alerts can be sorted and compared."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()


class AlertCategory(str, Enum):
    BED_OCCUPANCY = "bed_occupancy"
    ICU_PRESSURE = "icu_pressure"
    OPD_INFLOW = "opd_inflow"  # today's OPD unusually high vs history
    OPD_RISING = "opd_rising"  # forecast OPD well above current OPD
    MEDICINE_STOCK = "medicine_stock"
    MEDICINE_CONSUMPTION = "medicine_consumption"
    STAFF_WORKLOAD = "staff_workload"
    ANOMALY = "anomaly"


# Where a value comes from. Kept explicit so ML output is never confused
# with an observed value or a hand-written rule.
BASIS_OBSERVED = "observed"  # value read from the twin's current state
BASIS_ML = "ml_prediction"  # output of a trained model (OPD forecaster / Isolation Forest)
BASIS_RULE = "rule_derived"  # deterministic formula or heuristic


def _ladder(values: tuple[float, float, float, float], *, ascending: bool) -> None:
    low, med, high, crit = values
    ok = (low < med < high < crit) if ascending else (low > med > high > crit)
    if not ok:
        order = "increasing" if ascending else "decreasing"
        raise ValueError(
            f"Thresholds must be strictly {order} from Low to Critical: {values}"
        )


@dataclass(frozen=True)
class AlertThresholds:
    """
    Configurable alert thresholds. Each 4-tuple is (Low, Medium, High, Critical).

    Rationale (educational policy, not clinical guidance):

    * ``bed_occupancy`` / ``icu_occupancy`` (percent, trigger when value >= t):
      70 and 95 mirror the Moderate and Critical cut-offs already used by
      ``compute_capacity_status`` in the digital twin; 80 and 90 are even
      steps in between so four severities exist.
    * ``opd_zscore`` / ``medicine_consumption_zscore`` / ``staff_pressure_zscore``
      (trigger when value >= t standard deviations above the historical mean
      of the project's own processed data). 2 sigma is a common "unusual"
      cut-off; on the project dataset 2 sigma flags well under 2% of days.
    * ``medicine_days_cover`` (days, trigger when cover < t): stock divided by
      the estimated daily consumption. 21/14/7/3 days are typical reorder
      lead-time bands and are meant to be tuned per hospital.
    * ``rising_opd_percent`` (trigger when forecast OPD >= current OPD by t%):
      15/25/40/60 percent step-ups chosen as visible jumps well above normal
      day-to-day variation.
    * ``anomaly_high_score``: an Isolation Forest flag is Medium; a score at
      or above this value is High. The saved model's flagged days score about
      0.00-0.20, so 0.10 separates the stronger half. Anomalies are never
      Critical because the model does not measure capacity risk.
    """

    bed_occupancy: tuple[float, float, float, float] = (70.0, 80.0, 90.0, 95.0)
    icu_occupancy: tuple[float, float, float, float] = (70.0, 80.0, 90.0, 95.0)
    opd_zscore: tuple[float, float, float, float] = (2.0, 2.5, 3.0, 4.0)
    medicine_consumption_zscore: tuple[float, float, float, float] = (
        2.0,
        2.5,
        3.0,
        4.0,
    )
    staff_pressure_zscore: tuple[float, float, float, float] = (1.5, 2.0, 2.5, 3.0)
    medicine_days_cover: tuple[float, float, float, float] = (21.0, 14.0, 7.0, 3.0)
    rising_opd_percent: tuple[float, float, float, float] = (15.0, 25.0, 40.0, 60.0)
    anomaly_high_score: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "bed_occupancy",
            "icu_occupancy",
            "opd_zscore",
            "medicine_consumption_zscore",
            "staff_pressure_zscore",
            "rising_opd_percent",
        ):
            _ladder(getattr(self, name), ascending=True)
        _ladder(self.medicine_days_cover, ascending=False)
        if self.anomaly_high_score <= 0:
            raise ValueError("anomaly_high_score must be positive.")


@dataclass
class Alert:
    """One operational alert."""

    alert_id: str
    category: AlertCategory
    severity: Severity
    title: str
    explanation: str
    metric: str
    observed_value: float
    threshold: str  # human-readable condition that fired
    threshold_value: float | None
    date: str | None  # state date (as_of_date) when known
    basis: str  # observed | ml_prediction | rule_derived
    scope: str  # "current" or "predicted"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.label
        return data

    @property
    def dedup_key(self) -> tuple[str, str, str | None]:
        return (self.category.value, self.scope, self.date)


@dataclass
class AlertReport:
    """Alerts plus the checks that could not be performed."""

    alerts: list[Alert] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    generated_for: str | None = None

    @property
    def highest_severity(self) -> Severity | None:
        return max((a.severity for a in self.alerts), default=None)

    def by_category(self, category: AlertCategory) -> list[Alert]:
        return [a for a in self.alerts if a.category == category]

    def to_dict(self) -> dict[str, Any]:
        top = self.highest_severity
        return {
            "generated_for": self.generated_for,
            "highest_severity": top.label if top else None,
            "alerts": [a.to_dict() for a in self.alerts],
            "unavailable": list(self.unavailable),
        }
