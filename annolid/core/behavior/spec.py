from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from annolid.behavior.project_schema import (
    BehaviorDefinition,
    CategoryDefinition,
    DEFAULT_SCHEMA_FILENAME,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    _dict_to_schema,
    default_schema,
    find_schema_near_video,
    is_workspace_only_project_payload,
    read_schema_payload,
    save_schema,
    validate_schema,
)

BehaviorSpec = ProjectSchema


def _load_spec_from_resolved_path(
    resolved: Path,
    fallback: Optional[BehaviorSpec],
) -> BehaviorSpec:
    payload = read_schema_payload(resolved)
    loaded = _dict_to_schema(payload)
    if is_workspace_only_project_payload(payload):
        seed = fallback if fallback is not None else default_behavior_spec()
        return ProjectSchema(
            behaviors=list(seed.behaviors),
            categories=list(seed.categories),
            modifiers=list(seed.modifiers),
            subjects=list(seed.subjects),
            pose_schema_path=loaded.pose_schema_path or seed.pose_schema_path,
            pose_schema=loaded.pose_schema or seed.pose_schema,
            version=loaded.version,
        )
    return loaded


def load_behavior_spec(
    *,
    path: Optional[Union[str, Path]] = None,
    video_path: Optional[Union[str, Path]] = None,
    fallback: Optional[BehaviorSpec] = None,
) -> Tuple[BehaviorSpec, Optional[Path]]:
    """Load the canonical behavior spec (ProjectSchema) from JSON/YAML.

    Resolution order:
    1) explicit `path`
    2) auto-discovery near `video_path` (looks for `project.annolid.json`)
    3) `fallback` (or `default_behavior_spec()`)
    """

    if path is not None:
        resolved = Path(path)
        return _load_spec_from_resolved_path(resolved, fallback), resolved

    if video_path is not None:
        discovered = find_schema_near_video(Path(video_path))
        if discovered is not None:
            return _load_spec_from_resolved_path(discovered, fallback), discovered

    return fallback if fallback is not None else default_behavior_spec(), None


def default_behavior_spec() -> BehaviorSpec:
    """Return a minimal default behavior spec."""

    return default_schema()


def validate_behavior_spec(schema: BehaviorSpec) -> List[str]:
    """Return non-fatal schema validation warnings."""

    return validate_schema(schema)


def save_behavior_spec(schema: BehaviorSpec, path: Union[str, Path]) -> None:
    """Save the behavior spec to JSON/YAML based on file suffix."""

    save_schema(schema, Path(path))


__all__ = [
    "DEFAULT_SCHEMA_FILENAME",
    "BehaviorDefinition",
    "BehaviorSpec",
    "CategoryDefinition",
    "ModifierDefinition",
    "ProjectSchema",
    "SubjectDefinition",
    "default_behavior_spec",
    "load_behavior_spec",
    "save_behavior_spec",
    "validate_behavior_spec",
]
