"""
MediTwin AI digital twin package (Phase 4).

Example:
    from src.twin.hospital_twin import HospitalTwin, build_twin_from_latest

    twin = build_twin_from_latest()
    print(twin.get_current_state())
    print(twin.get_capacity_status())
    print(twin.get_predicted_state(days_ahead=1))
"""

from src.twin.hospital_twin import (
    HospitalDigitalTwin,
    HospitalState,
    HospitalTwin,
    build_twin_from_latest,
    compute_available_beds,
    compute_capacity_status,
    compute_occupancy_percentage,
)

__all__ = [
    "HospitalState",
    "HospitalDigitalTwin",
    "HospitalTwin",
    "build_twin_from_latest",
    "compute_available_beds",
    "compute_capacity_status",
    "compute_occupancy_percentage",
]
