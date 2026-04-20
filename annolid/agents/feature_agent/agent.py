"""Feature planning agent for assay-specific target selection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturePlanningResult:
    objectives: list[str]
    target_features: list[str]
    evidence: list[dict[str, object]] = field(default_factory=list)


class FeaturePlanningAgent:
    """Map assay type to goal-oriented objectives and feature targets."""

    _MAP: dict[str, tuple[list[str], list[str]]] = {
        "novel_object_recognition": (
            ["measure investigation time", "compare novel vs familiar object"],
            [
                "nose_tip",
                "object_center_left",
                "object_center_right",
                "investigation_bouts",
            ],
        ),
        "open_field": (
            [
                "measure locomotion and occupancy",
                "quantify center vs periphery behavior",
            ],
            ["centroid", "speed", "center_occupancy", "periphery_occupancy"],
        ),
        "social_interaction": (
            [
                "measure reciprocal interaction",
                "quantify social proximity and orientation",
            ],
            ["nose_to_nose", "body_distance", "orientation", "contact_bouts"],
        ),
        "courtship": (
            ["measure pursuit and contact", "quantify courtship transition sequence"],
            ["distance", "mount_events", "follow_events", "interaction_bouts"],
        ),
        "aggression": (
            ["count aggression bouts", "identify initiator and retreats"],
            ["slap_face", "run_away", "fight_initiation", "bout_boundaries"],
        ),
    }

    def plan(
        self, assay_type: str, context: dict | None = None
    ) -> FeaturePlanningResult:
        normalized = str(assay_type or "unknown").strip().lower()
        objectives, features = self._MAP.get(
            normalized,
            (
                ["summarize dominant behaviors"],
                ["track_presence", "motion_energy"],
            ),
        )
        evidence = [
            {
                "stage": "feature_planning",
                "assay_type": normalized,
                "objective_count": len(objectives),
                "feature_count": len(features),
                "context_keys": sorted(list((context or {}).keys())),
            }
        ]
        return FeaturePlanningResult(
            objectives=list(objectives),
            target_features=list(features),
            evidence=evidence,
        )


__all__ = ["FeaturePlanningAgent", "FeaturePlanningResult"]
