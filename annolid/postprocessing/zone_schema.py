from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shapely.geometry import Point, Polygon


ZONE_SCHEMA_VERSION = 1
LEGACY_ZONE_HINTS = frozenset(
    {
        "zone",
        "zones",
        "chamber",
        "chambers",
        "door",
        "doors",
        "doorway",
        "doorways",
        "passage",
        "passages",
        "barrier",
        "barriers",
        "mesh",
        "interaction",
        "interactions",
    }
)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _tokenize_text(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _normalize_text(value)) if token
    }


def _normalize_points(points: Sequence[Sequence[Any]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def _rectangle_to_polygon(points: Sequence[Sequence[Any]]) -> list[list[float]]:
    if len(points) < 2:
        return _normalize_points(points)
    top_left = points[0]
    bottom_right = points[1]
    top_right = [bottom_right[0], top_left[1]]
    bottom_left = [top_left[0], bottom_right[1]]
    return _normalize_points([top_left, top_right, bottom_right, bottom_left, top_left])


def _infer_zone_kind(label: str, description: str, flags: Mapping[str, Any]) -> str:
    explicit = _normalize_text(
        flags.get("zone_kind") or flags.get("zone_type") or flags.get("kind")
    )
    if explicit:
        return explicit
    tokens = _tokenize_text(f"{label} {description}")
    if "chamber" in tokens:
        return "chamber"
    if {"door", "doorway", "passage", "opening"} & tokens:
        return "doorway"
    if {"barrier", "mesh"} & tokens:
        return "barrier_edge"
    if {"interaction", "contact"} & tokens:
        return "interaction_zone"
    return "custom"


def _infer_phase(description: str, flags: Mapping[str, Any]) -> str:
    explicit = _normalize_text(flags.get("phase") or flags.get("assay_phase"))
    if explicit:
        return explicit
    tokens = _tokenize_text(description)
    if {"phase", "1"} <= tokens or "phase1" in tokens:
        return "phase_1"
    if {"phase", "2"} <= tokens or "phase2" in tokens:
        return "phase_2"
    return "custom"


def _infer_occupant_role(label: str, description: str, flags: Mapping[str, Any]) -> str:
    explicit = _normalize_text(
        flags.get("occupant_role") or flags.get("role") or flags.get("animal_role")
    )
    if explicit:
        return explicit
    tokens = _tokenize_text(f"{label} {description}")
    if "stim" in tokens:
        return "stim"
    if "rover" in tokens:
        return "rover"
    if "neutral" in tokens:
        return "neutral"
    return "unknown"


def _infer_access_state(description: str, flags: Mapping[str, Any]) -> str:
    explicit = _normalize_text(
        flags.get("access_state") or flags.get("passage_state") or flags.get("state")
    )
    if explicit:
        return explicit
    tokens = _tokenize_text(description)
    if {"blocked", "mesh"} & tokens:
        return "blocked"
    if {"open", "accessible"} & tokens:
        return "open"
    return "unknown"


def _infer_semantic_type(flags: Mapping[str, Any], label: str, description: str) -> str:
    explicit = _normalize_text(
        flags.get("semantic_type") or flags.get("shape_category") or flags.get("type")
    )
    if explicit:
        return explicit
    tokens = _tokenize_text(f"{label} {description}")
    if tokens & LEGACY_ZONE_HINTS:
        return "zone"
    return "shape"


def _has_legacy_zone_hint(shape: Mapping[str, Any]) -> bool:
    label = _normalize_text(shape.get("label"))
    description = _normalize_text(shape.get("description"))
    flags = shape.get("flags") or {}
    if not isinstance(flags, Mapping):
        flags = {}
    tokens = _tokenize_text(f"{label} {description}")
    if tokens & LEGACY_ZONE_HINTS:
        return True
    if _normalize_text(flags.get("legacy_zone")) in {"1", "true", "yes"}:
        return True
    return False


@dataclass
class ZoneShapeSpec:
    """Normalized zone shape with explicit semantics.

    The original shape dict remains compatible with LabelMe-style storage.
    New zone shapes should set explicit semantic flags. Legacy shapes are
    still accepted through label/description inference.
    """

    label: str
    points: list[list[float]]
    shape_type: str = "polygon"
    description: str = ""
    group_id: Any = None
    flags: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    semantic_type: str = "zone"
    zone_kind: str = "custom"
    phase: str = "custom"
    occupant_role: str = "unknown"
    access_state: str = "unknown"
    source_shape_type: str = "polygon"
    inferred_from_legacy: bool = False
    compatibility_mode: str = "explicit"

    @property
    def is_zone(self) -> bool:
        return self.semantic_type == "zone"

    @property
    def analysis_points(self) -> list[list[float]]:
        if self.source_shape_type == "rectangle" and len(self.points) == 2:
            return _rectangle_to_polygon(self.points)
        return _normalize_points(self.points)

    @property
    def display_label(self) -> str:
        return self.label or self.zone_kind

    def to_shape_dict(self) -> dict[str, Any]:
        flags = dict(self.flags)
        flags.update(
            {
                "semantic_type": self.semantic_type,
                "shape_category": "zone"
                if self.is_zone
                else flags.get("shape_category", "shape"),
                "zone_kind": self.zone_kind,
                "phase": self.phase,
                "occupant_role": self.occupant_role,
                "access_state": self.access_state,
                "schema_version": ZONE_SCHEMA_VERSION,
            }
        )
        if self.inferred_from_legacy:
            flags["inferred_from_legacy"] = True
        flags["compatibility_mode"] = self.compatibility_mode
        return {
            "label": self.label,
            "points": [list(point) for point in self.points],
            "shape_type": self.shape_type,
            "description": self.description,
            "group_id": self.group_id,
            "flags": flags,
            "visible": self.visible,
        }

    @classmethod
    def from_shape_dict(cls, shape: Mapping[str, Any]) -> "ZoneShapeSpec":
        if not isinstance(shape, Mapping):
            raise TypeError("shape must be a mapping")
        label = str(shape.get("label") or "").strip()
        description = str(shape.get("description") or "").strip()
        flags = dict(shape.get("flags") or {})
        shape_type = str(shape.get("shape_type") or "polygon")
        source_shape_type = shape_type
        points = shape.get("points") or []
        explicit_semantic = _normalize_text(
            flags.get("semantic_type")
            or flags.get("shape_category")
            or flags.get("type")
        )
        semantic_type = _infer_semantic_type(flags, label, description)
        zone_candidate = _has_legacy_zone_hint(shape)
        inferred_from_legacy = False
        compatibility_mode = "explicit"
        if not explicit_semantic and zone_candidate:
            semantic_type = "zone"
            inferred_from_legacy = True
            compatibility_mode = "legacy_compat"
        elif explicit_semantic:
            compatibility_mode = "explicit"
        zone_kind = _infer_zone_kind(label, description, flags)
        phase = _infer_phase(description, flags)
        occupant_role = _infer_occupant_role(label, description, flags)
        access_state = _infer_access_state(description, flags)
        return cls(
            label=label,
            points=_normalize_points(points),
            shape_type=shape_type,
            description=description,
            group_id=shape.get("group_id"),
            flags=flags,
            visible=bool(shape.get("visible", True)),
            semantic_type=semantic_type,
            zone_kind=zone_kind,
            phase=phase,
            occupant_role=occupant_role,
            access_state=access_state,
            source_shape_type=source_shape_type,
            inferred_from_legacy=inferred_from_legacy,
            compatibility_mode=compatibility_mode,
        )


def build_zone_shape(
    label: str,
    points: Sequence[Sequence[Any]],
    *,
    shape_type: str = "polygon",
    zone_kind: str = "custom",
    phase: str = "custom",
    occupant_role: str = "unknown",
    access_state: str = "unknown",
    description: str = "",
    group_id: Any = None,
    visible: bool = True,
    extra_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a LabelMe-compatible shape dict with explicit zone metadata."""

    flags = dict(extra_flags or {})
    flags.update(
        {
            "semantic_type": "zone",
            "shape_category": "zone",
            "zone_kind": zone_kind,
            "phase": phase,
            "occupant_role": occupant_role,
            "access_state": access_state,
            "schema_version": ZONE_SCHEMA_VERSION,
        }
    )
    spec = ZoneShapeSpec(
        label=label,
        points=_normalize_points(points),
        shape_type=shape_type,
        description=description,
        group_id=group_id,
        flags=flags,
        visible=visible,
        semantic_type="zone",
        zone_kind=zone_kind,
        phase=phase,
        occupant_role=occupant_role,
        access_state=access_state,
    )
    return spec.to_shape_dict()


def load_zone_shapes(zone_data: Mapping[str, Any]) -> list[ZoneShapeSpec]:
    """Return normalized zone shapes, preserving legacy JSON compatibility."""

    shapes = zone_data.get("shapes", [])
    zone_shapes: list[ZoneShapeSpec] = []
    for shape in shapes:
        try:
            spec = ZoneShapeSpec.from_shape_dict(shape)
        except Exception:
            continue
        if spec.is_zone or spec.inferred_from_legacy:
            zone_shapes.append(spec)
    return zone_shapes


def normalize_zone_shape(shape: Mapping[str, Any]) -> ZoneShapeSpec:
    """Normalize a single shape without mutating the original dict."""

    return ZoneShapeSpec.from_shape_dict(shape)


def zone_shape_covers_point(shape: ZoneShapeSpec, point: Sequence[float]) -> bool:
    if len(shape.analysis_points) < 3:
        return False
    polygon = Polygon(shape.analysis_points)
    return polygon.covers(Point(float(point[0]), float(point[1])))


def zone_shape_bounds(shape: ZoneShapeSpec) -> tuple[float, float, float, float] | None:
    if len(shape.analysis_points) < 3:
        return None
    polygon = Polygon(shape.analysis_points)
    min_x, min_y, max_x, max_y = polygon.bounds
    return min_x, min_y, max_x, max_y


def zone_shape_centroid(shape: ZoneShapeSpec) -> tuple[float, float] | None:
    if len(shape.analysis_points) < 3:
        return None
    polygon = Polygon(shape.analysis_points)
    centroid = polygon.centroid
    return float(centroid.x), float(centroid.y)


def zone_shape_distance_to_point(
    shape: ZoneShapeSpec,
    point: Sequence[float],
) -> float | None:
    if len(shape.analysis_points) < 3:
        return None
    try:
        polygon = Polygon(shape.analysis_points)
        return float(polygon.distance(Point(float(point[0]), float(point[1]))))
    except Exception:
        return None
