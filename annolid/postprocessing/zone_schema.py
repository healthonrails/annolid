from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid


ZONE_SCHEMA_VERSION = 1
LEGACY_ZONE_HINTS = frozenset(
    {
        "zone",
        "zones",
        "chamber",
        "chambers",
        "tube",
        "tubes",
        "connector",
        "connectors",
        "tunnel",
        "tunnels",
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
    if {"tube", "tunnel"} & tokens:
        return "connector_tube"
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


class _CompiledZoneGeometry(NamedTuple):
    geometry: Any
    bounds: tuple[float, float, float, float]
    centroid: tuple[float, float]
    area: float


@lru_cache(maxsize=512)
def _compile_zone_geometry_cached(
    source_shape_type: str,
    raw_points: tuple[tuple[float, float], ...],
) -> _CompiledZoneGeometry | None:
    if source_shape_type == "rectangle" and len(raw_points) == 2:
        pts: list[list[float]] | tuple[tuple[float, float], ...] = (
            _rectangle_to_polygon(raw_points)
        )
    else:
        pts = raw_points
    if len(pts) < 3:
        return None
    for x_val, y_val in pts:
        if not (math.isfinite(x_val) and math.isfinite(y_val)):
            return None
    try:
        geom = Polygon(pts)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty:
            return None
        shapely.prepare(geom)
        min_x, min_y, max_x, max_y = geom.bounds
        centroid_pt = geom.centroid
        bounds = (float(min_x), float(min_y), float(max_x), float(max_y))
        centroid = (float(centroid_pt.x), float(centroid_pt.y))
        area = float(geom.area)
        if not all(math.isfinite(v) for v in (*bounds, *centroid, area)):
            return None
        return _CompiledZoneGeometry(
            geometry=geom,
            bounds=bounds,
            centroid=centroid,
            area=area,
        )
    except Exception:
        return None


def _get_compiled_zone_geometry(shape: ZoneShapeSpec) -> _CompiledZoneGeometry | None:
    try:
        raw_points = tuple((float(pt[0]), float(pt[1])) for pt in shape.points)
    except Exception:
        return None
    return _compile_zone_geometry_cached(str(shape.source_shape_type or ""), raw_points)


def zone_shape_area(shape: ZoneShapeSpec) -> float | None:
    compiled = _get_compiled_zone_geometry(shape)
    return compiled.area if compiled is not None else None


def zone_shape_covers_point(shape: ZoneShapeSpec, point: Sequence[float]) -> bool:
    compiled = _get_compiled_zone_geometry(shape)
    if compiled is None or len(point) < 2:
        return False
    try:
        px = float(point[0])
        py = float(point[1])
    except Exception:
        return False
    if not (math.isfinite(px) and math.isfinite(py)):
        return False
    min_x, min_y, max_x, max_y = compiled.bounds
    if px < min_x or px > max_x or py < min_y or py > max_y:
        return False
    return bool(compiled.geometry.covers(Point(px, py)))


def zone_shape_covers_points_xy(
    shape: ZoneShapeSpec,
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
) -> np.ndarray:
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    n = len(xs_arr)
    out = np.zeros(n, dtype=bool)
    if n == 0 or len(ys_arr) != n:
        return out
    compiled = _get_compiled_zone_geometry(shape)
    if compiled is None:
        return out
    min_x, min_y, max_x, max_y = compiled.bounds
    candidates = (
        np.isfinite(xs_arr)
        & np.isfinite(ys_arr)
        & (xs_arr >= min_x)
        & (xs_arr <= max_x)
        & (ys_arr >= min_y)
        & (ys_arr <= max_y)
    )
    if np.any(candidates):
        pts = shapely.points(xs_arr[candidates], ys_arr[candidates])
        out[candidates] = np.asarray(shapely.covers(compiled.geometry, pts), dtype=bool)
    return out


def zone_shape_bounds(shape: ZoneShapeSpec) -> tuple[float, float, float, float] | None:
    compiled = _get_compiled_zone_geometry(shape)
    return compiled.bounds if compiled is not None else None


def zone_shape_centroid(shape: ZoneShapeSpec) -> tuple[float, float] | None:
    compiled = _get_compiled_zone_geometry(shape)
    return compiled.centroid if compiled is not None else None


def zone_shape_distance_to_point(
    shape: ZoneShapeSpec,
    point: Sequence[float],
) -> float | None:
    compiled = _get_compiled_zone_geometry(shape)
    if compiled is None or len(point) < 2:
        return None
    try:
        px = float(point[0])
        py = float(point[1])
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return float(compiled.geometry.distance(Point(px, py)))
    except Exception:
        return None


def zone_shape_distances_to_points_xy(
    shape: ZoneShapeSpec,
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
) -> np.ndarray:
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    n = len(xs_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or len(ys_arr) != n:
        return out
    compiled = _get_compiled_zone_geometry(shape)
    if compiled is None:
        return out
    valid = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    if np.any(valid):
        pts = shapely.points(xs_arr[valid], ys_arr[valid])
        out[valid] = np.asarray(
            shapely.distance(compiled.geometry, pts), dtype=np.float64
        )
    return out
