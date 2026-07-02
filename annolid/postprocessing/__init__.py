"""Lazy postprocessing exports.

GUI startup imports lightweight zone helpers from this package; pandas-backed
analysis policies should load only when those workflows are used.
"""

from __future__ import annotations

from typing import Any


_EXPORTS = {
    "EvidenceRule": ("annolid.postprocessing.identity_governor", "EvidenceRule"),
    "GovernorPolicy": ("annolid.postprocessing.identity_governor", "GovernorPolicy"),
    "IdentityCorrection": (
        "annolid.postprocessing.identity_governor",
        "IdentityCorrection",
    ),
    "IdentityGovernor": (
        "annolid.postprocessing.identity_governor",
        "IdentityGovernor",
    ),
    "IdentityGovernorResult": (
        "annolid.postprocessing.identity_governor",
        "IdentityGovernorResult",
    ),
    "MetricCondition": ("annolid.postprocessing.identity_governor", "MetricCondition"),
    "run_identity_governor": (
        "annolid.postprocessing.identity_governor",
        "run_identity_governor",
    ),
    "run_temporal_identity_repair": (
        "annolid.postprocessing.temporal_identity_repair",
        "run_temporal_identity_repair",
    ),
    "ZoneOccupancyPolicyResult": (
        "annolid.postprocessing.zone_occupancy_policy",
        "ZoneOccupancyPolicyResult",
    ),
    "apply_zone_occupancy_policy": (
        "annolid.postprocessing.zone_occupancy_policy",
        "apply_zone_occupancy_policy",
    ),
    "apply_zone_occupancy_policy_file": (
        "annolid.postprocessing.zone_occupancy_policy",
        "apply_zone_occupancy_policy_file",
    ),
    "load_zone_occupancy_policy": (
        "annolid.postprocessing.zone_occupancy_policy",
        "load_zone_occupancy_policy",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'annolid.postprocessing' has no attribute {name!r}"
        ) from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
