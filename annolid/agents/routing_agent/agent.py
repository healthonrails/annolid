"""Perception routing agent for backend selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annolid.services.behavior_agent.model_policy import BehaviorModelPolicy


@dataclass(frozen=True)
class PerceptionRoute:
    backend: str
    rationale: str
    evidence: list[dict[str, object]] = field(default_factory=list)


class PerceptionRoutingAgent:
    """Select perception backend using assay + model policy."""

    def route(self, *, assay_type: str, policy: BehaviorModelPolicy) -> PerceptionRoute:
        assay = str(assay_type or "unknown").strip().lower()
        override = policy.route_overrides.get(assay)
        if override:
            backend = str(override)
            rationale = f"policy override for assay '{assay}'"
        elif (
            assay in {"aggression", "social_interaction"}
            and policy.allow_open_set_grounding
        ):
            backend = "grounding_dino"
            rationale = "open-set social/aggression cues benefit from grounding"
        elif assay in {"courtship"} and policy.allow_server_scale_models:
            backend = "sam2_server"
            rationale = "optional server-scale propagation enabled by model policy"
        else:
            backend = str(policy.perception_default_backend)
            rationale = "default continuity route to Annolid trackers"

        evidence = [
            {
                "stage": "perception_routing",
                "assay_type": assay,
                "policy": policy.name,
                "backend": backend,
                "rationale": rationale,
            }
        ]
        return PerceptionRoute(backend=backend, rationale=rationale, evidence=evidence)


__all__ = ["PerceptionRoutingAgent", "PerceptionRoute"]
