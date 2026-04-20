"""Behavior-agent subagent profiles for Annolid agent delegation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BehaviorSubagentProfile:
    """Typed profile that configures a specialized behavior subagent."""

    name: str
    description: str
    system_instructions: str
    default_skill_names: tuple[str, ...]


_PROFILES: dict[str, BehaviorSubagentProfile] = {
    "behavior_assay_inference": BehaviorSubagentProfile(
        name="behavior_assay_inference",
        description="Infer assay type from video/task context and evidence.",
        system_instructions=(
            "You are the Assay Inference subagent. Infer the most likely assay type "
            "from available metadata and evidence, report confidence, and provide "
            "concise rationale with ambiguity notes."
        ),
        default_skill_names=("vision-language-analysis", "behavior-assay-taxonomy"),
    ),
    "behavior_feature_planning": BehaviorSubagentProfile(
        name="behavior_feature_planning",
        description="Plan target features and objectives per inferred assay.",
        system_instructions=(
            "You are the Feature Planning subagent. Translate assay context into "
            "target features and measurable objectives suitable for downstream "
            "tracking and segmentation."
        ),
        default_skill_names=("behavior-feature-selection", "experimental-design"),
    ),
    "behavior_perception_routing": BehaviorSubagentProfile(
        name="behavior_perception_routing",
        description="Select perception backend strategy for the assay/task.",
        system_instructions=(
            "You are the Perception Routing subagent. Choose an execution backend "
            "policy (tracker continuity, open-set grounding, or propagation) and "
            "justify tradeoffs."
        ),
        default_skill_names=("perception-routing", "model-policy"),
    ),
    "behavior_segmentation": BehaviorSubagentProfile(
        name="behavior_segmentation",
        description="Convert tracks/signals into behavior segments with rationale.",
        system_instructions=(
            "You are the Behavior Segmentation subagent. Build deterministic "
            "timeline segments from artifacts and include concise rationale."
        ),
        default_skill_names=("behavior-segmentation", "timeline-reasoning"),
    ),
    "behavior_analysis_coding": BehaviorSubagentProfile(
        name="behavior_analysis_coding",
        description="Generate and execute constrained analysis code over artifacts.",
        system_instructions=(
            "You are the Analysis Coding subagent. Produce minimal deterministic "
            "analysis code and execution outputs suitable for reproducible runs."
        ),
        default_skill_names=("sandboxed-analysis", "metrics-derivation"),
    ),
    "behavior_reporting": BehaviorSubagentProfile(
        name="behavior_reporting",
        description="Produce report summaries with provenance links.",
        system_instructions=(
            "You are the Report subagent. Summarize findings, cite provenance "
            "artifacts, and preserve typed output compatibility."
        ),
        default_skill_names=("scientific-reporting", "provenance"),
    ),
}


def resolve_behavior_subagent_profile(
    profile_name: str | None,
) -> BehaviorSubagentProfile | None:
    key = str(profile_name or "").strip().lower()
    if not key:
        return None
    return _PROFILES.get(key)


def list_behavior_subagent_profiles() -> Sequence[BehaviorSubagentProfile]:
    return tuple(_PROFILES[name] for name in sorted(_PROFILES))


__all__ = [
    "BehaviorSubagentProfile",
    "list_behavior_subagent_profiles",
    "resolve_behavior_subagent_profile",
]
