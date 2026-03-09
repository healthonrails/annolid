from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from annolid.simulation.types import SimulationMapping


def _as_dict(value: Any, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def simulation_mapping_from_dict(data: Dict[str, Any]) -> SimulationMapping:
    if not isinstance(data, dict):
        raise ValueError("Simulation mapping config must be an object")
    backend = str(data.get("backend") or "").strip()
    if not backend:
        raise ValueError("Simulation mapping config requires a non-empty 'backend'")

    keypoint_to_site = {
        str(key).strip(): str(value).strip()
        for key, value in _as_dict(
            data.get("keypoint_to_site"), field_name="keypoint_to_site"
        ).items()
        if str(key).strip() and str(value).strip()
    }
    site_to_joint = {
        str(key).strip(): str(value).strip()
        for key, value in _as_dict(
            data.get("site_to_joint"), field_name="site_to_joint"
        ).items()
        if str(key).strip() and str(value).strip()
    }
    coordinate_system = _as_dict(
        data.get("coordinate_system"), field_name="coordinate_system"
    )
    metadata = _as_dict(data.get("metadata"), field_name="metadata")
    return SimulationMapping(
        backend=backend,
        keypoint_to_site=keypoint_to_site,
        site_to_joint=site_to_joint,
        coordinate_system=coordinate_system,
        metadata=metadata,
    )


def load_simulation_mapping(path: str | Path) -> SimulationMapping:
    mapping_path = Path(path).expanduser()
    if not mapping_path.exists():
        raise FileNotFoundError(str(mapping_path))

    suffix = mapping_path.suffix.lower()
    raw_text = mapping_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to load simulation mapping YAML files."
            ) from exc
        data = yaml.safe_load(raw_text) or {}
    else:
        data = json.loads(raw_text)
    return simulation_mapping_from_dict(data)
