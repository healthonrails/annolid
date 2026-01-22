from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from annolid.behavior.project_schema import (
    DEFAULT_SCHEMA_FILENAME,
    ProjectSchema,
    default_schema,
    find_schema_near_video,
    load_schema,
    save_schema,
    validate_schema,
)

BehaviorSpec = ProjectSchema


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
        return load_schema(resolved), resolved

    if video_path is not None:
        discovered = find_schema_near_video(Path(video_path))
        if discovered is not None:
            return load_schema(discovered), discovered

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
    "BehaviorSpec",
    "ProjectSchema",
    "default_behavior_spec",
    "load_behavior_spec",
    "save_behavior_spec",
    "validate_behavior_spec",
]
