"""Explicit, swappable model policy for behavior-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


InferenceMode = Literal["hosted", "local"]


@dataclass(frozen=True)
class BehaviorModelPolicy:
    name: str
    task_inference_mode: InferenceMode = "hosted"
    perception_default_backend: str = "annolid_tracking"
    allow_open_set_grounding: bool = False
    allow_server_scale_models: bool = False
    privacy_sensitive: bool = False
    route_overrides: dict[str, str] = field(default_factory=dict)


DEFAULT_POLICY = BehaviorModelPolicy(
    name="hosted_reasoning_local_tracking_v1",
    task_inference_mode="hosted",
    perception_default_backend="annolid_tracking",
    allow_open_set_grounding=True,
    allow_server_scale_models=False,
    privacy_sensitive=False,
)

PRIVACY_POLICY = BehaviorModelPolicy(
    name="local_reasoning_local_tracking_v1",
    task_inference_mode="local",
    perception_default_backend="annolid_tracking",
    allow_open_set_grounding=False,
    allow_server_scale_models=False,
    privacy_sensitive=True,
)


def resolve_behavior_model_policy(name: str | None) -> BehaviorModelPolicy:
    normalized = str(name or "").strip().lower()
    if not normalized or normalized == DEFAULT_POLICY.name.lower():
        return DEFAULT_POLICY
    if normalized == PRIVACY_POLICY.name.lower():
        return PRIVACY_POLICY
    return BehaviorModelPolicy(
        name=str(name or DEFAULT_POLICY.name).strip() or DEFAULT_POLICY.name
    )


__all__ = [
    "BehaviorModelPolicy",
    "DEFAULT_POLICY",
    "PRIVACY_POLICY",
    "resolve_behavior_model_policy",
]
