"""Behavior specification (ProjectSchema) utilities.

Phase-1 treats :class:`annolid.behavior.project_schema.ProjectSchema` as the
canonical, GUI-free "BehaviorSpec" consumed by both GUI and headless tools.
"""

from .spec import (
    BehaviorSpec,
    ProjectSchema,
    default_behavior_spec,
    load_behavior_spec,
    save_behavior_spec,
    validate_behavior_spec,
)

__all__ = [
    "BehaviorSpec",
    "ProjectSchema",
    "default_behavior_spec",
    "load_behavior_spec",
    "save_behavior_spec",
    "validate_behavior_spec",
]
