from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from annolid.annotation.pose_schema import PoseSchema
from annolid.simulation.mapping import simulation_mapping_from_dict
from annolid.simulation.types import SimulationMapping


def generate_flybody_mapping_template(
    *,
    keypoints: Sequence[str],
    pose_schema: PoseSchema | None = None,
) -> SimulationMapping:
    ordered_keypoints = _normalize_keypoints(keypoints, pose_schema=pose_schema)
    keypoint_to_site = {
        name: f"{_sanitize_name(name)}_site" for name in ordered_keypoints if name
    }
    metadata = {
        "template": "flybody",
        "notes": [
            "Replace generated site names with the actual FlyBody site names.",
            "Fill site_to_joint once the target MuJoCo model is finalized.",
            "Set camera_intrinsics when using depth-assisted 2D-to-3D lifting.",
        ],
    }
    return simulation_mapping_from_dict(
        {
            "backend": "flybody",
            "keypoint_to_site": keypoint_to_site,
            "site_to_joint": {},
            "coordinate_system": {
                "units": "meters",
                "camera_intrinsics": {
                    "fx": None,
                    "fy": None,
                    "cx": None,
                    "cy": None,
                },
            },
            "metadata": metadata,
        }
    )


def save_simulation_mapping_template(
    mapping: SimulationMapping,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": mapping.backend,
        "keypoint_to_site": dict(mapping.keypoint_to_site),
        "site_to_joint": dict(mapping.site_to_joint),
        "coordinate_system": dict(mapping.coordinate_system),
        "metadata": dict(mapping.metadata),
    }
    if output_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to save simulation mapping YAML files."
            ) from exc
        output_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        return output_path
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _normalize_keypoints(
    keypoints: Sequence[str],
    *,
    pose_schema: PoseSchema | None,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    if pose_schema is not None:
        for name in pose_schema.keypoints:
            label = str(name or "").strip()
            if label and label not in seen:
                seen.add(label)
                ordered.append(label)
    for name in keypoints:
        label = str(name or "").strip()
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _sanitize_name(value: str) -> str:
    out = []
    for char in str(value or "").strip().lower():
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")
