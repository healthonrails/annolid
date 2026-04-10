from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from qtpy import QtCore

from annolid.gui.polygon_tools import interpolate_closed_polygon_points
from annolid.gui.polygon_tools import polygon_identity_key


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
PRESENCE_STATES = {"present", "hidden", "created", "zero_area"}


def _as_point2_list(points: Sequence[Any] | None) -> list[Point2]:
    result: list[Point2] = []
    for point in list(points or []):
        try:
            qpoint = QtCore.QPointF(point)
            result.append((float(qpoint.x()), float(qpoint.y())))
            continue
        except Exception:
            pass
        try:
            result.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    return result


def _resample_polygon_points(points: Sequence[Point2], count: int) -> list[Point2]:
    if not points:
        return []
    qpoints = [QtCore.QPointF(float(x), float(y)) for x, y in list(points or [])]
    resampled = interpolate_closed_polygon_points(
        qpoints,
        qpoints,
        0.0,
        point_count=max(3, int(count or 0)),
    )
    return [(float(point.x()), float(point.y())) for point in resampled]


def _interpolate_point_lists(
    points_a: Sequence[Point2],
    points_b: Sequence[Point2],
    ratio: float,
    *,
    point_count: int,
) -> list[Point2]:
    qa = [QtCore.QPointF(float(x), float(y)) for x, y in list(points_a or [])]
    qb = [QtCore.QPointF(float(x), float(y)) for x, y in list(points_b or [])]
    blended = interpolate_closed_polygon_points(
        qa,
        qb,
        max(0.0, min(1.0, float(ratio))),
        point_count=max(3, int(point_count or 0)),
    )
    return [(float(point.x()), float(point.y())) for point in blended]


def _polygon_signed_area(points: Sequence[Point2]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    pts = list(points)
    for index, (x2, y2) in enumerate(pts):
        x1, y1 = pts[index - 1]
        area2 += (float(x1) * float(y2)) - (float(x2) * float(y1))
    return 0.5 * area2


def _rotate_points(points: Sequence[Point2], offset: int) -> list[Point2]:
    pts = list(points or [])
    if not pts:
        return []
    n = len(pts)
    shift = int(offset) % n
    return pts[shift:] + pts[:shift]


def _alignment_cost(a: Sequence[Point2], b: Sequence[Point2]) -> float:
    if len(a) != len(b):
        return float("inf")
    cost = 0.0
    for (ax, ay), (bx, by) in zip(a, b):
        dx = float(ax) - float(bx)
        dy = float(ay) - float(by)
        cost += (dx * dx) + (dy * dy)
    return cost


def _align_points_to_reference(
    reference: Sequence[Point2],
    candidate: Sequence[Point2],
) -> list[Point2]:
    ref = list(reference or [])
    cand = list(candidate or [])
    if len(ref) < 3 or len(cand) < 3 or len(ref) != len(cand):
        return cand

    ref_area = _polygon_signed_area(ref)
    cand_area = _polygon_signed_area(cand)
    forward = list(cand)
    if (ref_area > 0.0 and cand_area < 0.0) or (ref_area < 0.0 and cand_area > 0.0):
        forward = list(reversed(forward))

    best = forward
    best_cost = float("inf")
    for offset in range(len(forward)):
        rotated = _rotate_points(forward, offset)
        cost = _alignment_cost(ref, rotated)
        if cost < best_cost:
            best_cost = cost
            best = rotated
    return best


def _smooth_closed_polygon(points: Sequence[Point2], alpha: float) -> list[Point2]:
    pts = list(points or [])
    if len(pts) < 3:
        return pts
    strength = max(0.0, min(1.0, float(alpha)))
    if strength <= 1e-6:
        return pts
    smoothed: list[Point2] = []
    n = len(pts)
    for index, (x, y) in enumerate(pts):
        px, py = pts[index - 1]
        nx, ny = pts[(index + 1) % n]
        avg_x = (float(px) + float(x) + float(nx)) / 3.0
        avg_y = (float(py) + float(y) + float(ny)) / 3.0
        new_x = (float(x) * (1.0 - strength)) + (avg_x * strength)
        new_y = (float(y) * (1.0 - strength)) + (avg_y * strength)
        smoothed.append((new_x, new_y))
    return smoothed


def _smooth_longitudinal_sections(
    sections: dict[int, list[Point2]],
    alpha: float,
) -> dict[int, list[Point2]]:
    if not sections:
        return {}
    strength = max(0.0, min(1.0, float(alpha)))
    if strength <= 1e-6:
        return {int(k): list(v or []) for k, v in dict(sections).items()}

    ordered_sections = sorted(int(section) for section in sections.keys())
    point_count = len(list(sections.get(ordered_sections[0]) or []))
    if point_count < 3:
        return {int(k): list(v or []) for k, v in dict(sections).items()}

    smoothed: dict[int, list[Point2]] = {
        section: [tuple(point) for point in list(sections.get(section) or [])]
        for section in ordered_sections
    }
    for interior_idx in range(1, len(ordered_sections) - 1):
        left_key = ordered_sections[interior_idx - 1]
        key = ordered_sections[interior_idx]
        right_key = ordered_sections[interior_idx + 1]
        left = list(sections.get(left_key) or [])
        cur = list(sections.get(key) or [])
        right = list(sections.get(right_key) or [])
        if (
            len(left) != point_count
            or len(cur) != point_count
            or len(right) != point_count
        ):
            continue
        next_points: list[Point2] = []
        for idx in range(point_count):
            lx, ly = left[idx]
            cx, cy = cur[idx]
            rx, ry = right[idx]
            avg_x = (float(lx) + float(cx) + float(rx)) / 3.0
            avg_y = (float(ly) + float(cy) + float(ry)) / 3.0
            nx = (float(cx) * (1.0 - strength)) + (avg_x * strength)
            ny = (float(cy) * (1.0 - strength)) + (avg_y * strength)
            next_points.append((nx, ny))
        smoothed[key] = next_points
    return smoothed


def _snap_points_to_guides(
    points: Sequence[Point2],
    guide_points: Sequence[Point2],
    *,
    strength: float,
    max_distance: float,
) -> list[Point2]:
    pts = list(points or [])
    guides = list(guide_points or [])
    if not pts or not guides:
        return pts
    snap_strength = max(0.0, min(1.0, float(strength)))
    threshold = max(0.0, float(max_distance))
    if snap_strength <= 1e-6 or threshold <= 1e-6:
        return pts
    threshold2 = threshold * threshold
    snapped: list[Point2] = []
    for x, y in pts:
        best = None
        best_dist2 = float("inf")
        for gx, gy in guides:
            dx = float(gx) - float(x)
            dy = float(gy) - float(y)
            dist2 = (dx * dx) + (dy * dy)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best = (float(gx), float(gy))
        if best is None or best_dist2 > threshold2:
            snapped.append((float(x), float(y)))
            continue
        bx, by = best
        nx = (float(x) * (1.0 - snap_strength)) + (float(bx) * snap_strength)
        ny = (float(y) * (1.0 - snap_strength)) + (float(by) * snap_strength)
        snapped.append((nx, ny))
    return snapped


def _region_id_from_shape(shape: Any) -> str:
    if isinstance(shape, dict):
        label = str(shape.get("label", "") or "")
        group_id = str(shape.get("group_id", "") or "")
        description = str(shape.get("description", "") or "")
        return f"{label}|{group_id}|{description}"
    label, group_id, description = polygon_identity_key(shape)
    return f"{label}|{group_id}|{description}"


def _region_label_from_id(region_id: str) -> str:
    raw = str(region_id or "")
    if "|" in raw:
        return raw.split("|", 1)[0]
    return raw


def _shape_shape_type(shape: Any) -> str:
    if isinstance(shape, dict):
        return str(shape.get("shape_type", "") or "").lower()
    return str(getattr(shape, "shape_type", "") or "").lower()


def _shape_points(shape: Any) -> list[Point2]:
    if isinstance(shape, dict):
        return _as_point2_list(shape.get("points"))
    return _as_point2_list(getattr(shape, "points", []) or [])


def _shape_label(shape: Any) -> str:
    if isinstance(shape, dict):
        return str(shape.get("label", "") or "")
    return str(getattr(shape, "label", "") or "")


def _shape_group_id(shape: Any) -> str:
    if isinstance(shape, dict):
        value = shape.get("group_id", "")
    else:
        value = getattr(shape, "group_id", "")
    return str(value if value is not None else "")


def _shape_description(shape: Any) -> str:
    if isinstance(shape, dict):
        return str(shape.get("description", "") or "")
    return str(getattr(shape, "description", "") or "")


def _page_polygon_signature(shapes: Sequence[Any] | None) -> str:
    entries: list[dict[str, Any]] = []
    for shape in list(shapes or []):
        if _shape_shape_type(shape) != "polygon":
            continue
        points = _shape_points(shape)
        if len(points) < 3:
            continue
        entries.append(
            {
                "label": _shape_label(shape),
                "group_id": _shape_group_id(shape),
                "description": _shape_description(shape),
                "points": [[round(float(x), 4), round(float(y), 4)] for x, y in points],
            }
        )
    entries.sort(
        key=lambda item: (
            str(item.get("label") or ""),
            str(item.get("group_id") or ""),
            str(item.get("description") or ""),
            len(list(item.get("points") or [])),
        )
    )
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Brain3DConfig:
    source_orientation: str = "sagittal"
    point_count: int = 64
    interpolation_density: int = 1
    section_positions: list[float] | None = None
    coronal_spacing: float | None = 1.0
    coronal_plane_count: int | None = None
    smoothing_longitudinal: float = 0.0
    smoothing_inplane: float = 0.0
    snapping_enabled: bool = False
    snapping_strength: float = 0.0
    snapping_max_distance: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_orientation": str(self.source_orientation or "sagittal"),
            "point_count": int(self.point_count or 64),
            "interpolation_density": max(1, int(self.interpolation_density or 1)),
            "section_positions": (
                [float(value) for value in list(self.section_positions or [])]
                if self.section_positions is not None
                else None
            ),
            "coronal_spacing": (
                None if self.coronal_spacing is None else float(self.coronal_spacing)
            ),
            "coronal_plane_count": (
                None
                if self.coronal_plane_count is None
                else int(self.coronal_plane_count)
            ),
            "smoothing_longitudinal": float(self.smoothing_longitudinal or 0.0),
            "smoothing_inplane": float(self.smoothing_inplane or 0.0),
            "snapping_enabled": bool(self.snapping_enabled),
            "snapping_strength": float(self.snapping_strength or 0.0),
            "snapping_max_distance": float(self.snapping_max_distance or 8.0),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None):
        data = dict(payload or {})
        section_positions = data.get("section_positions")
        return cls(
            source_orientation=str(data.get("source_orientation") or "sagittal"),
            point_count=max(3, int(data.get("point_count") or 64)),
            interpolation_density=max(1, int(data.get("interpolation_density") or 1)),
            section_positions=(
                [float(value) for value in list(section_positions or [])]
                if section_positions is not None
                else None
            ),
            coronal_spacing=(
                None
                if data.get("coronal_spacing") is None
                else float(data.get("coronal_spacing"))
            ),
            coronal_plane_count=(
                None
                if data.get("coronal_plane_count") is None
                else int(data.get("coronal_plane_count"))
            ),
            smoothing_longitudinal=float(data.get("smoothing_longitudinal") or 0.0),
            smoothing_inplane=float(data.get("smoothing_inplane") or 0.0),
            snapping_enabled=bool(data.get("snapping_enabled", False)),
            snapping_strength=float(data.get("snapping_strength") or 0.0),
            snapping_max_distance=float(data.get("snapping_max_distance") or 8.0),
        )


@dataclass(slots=True)
class RegionTrack:
    region_id: str
    label: str
    observed_sections: list[int] = field(default_factory=list)
    reconstructed_sections: dict[int, list[Point2]] = field(default_factory=dict)
    contour_nodes_3d: dict[int, list[Point3]] = field(default_factory=dict)
    presence_interval: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": str(self.region_id or ""),
            "label": str(self.label or ""),
            "observed_sections": [int(value) for value in list(self.observed_sections)],
            "presence_interval": (
                None
                if self.presence_interval is None
                else [int(self.presence_interval[0]), int(self.presence_interval[1])]
            ),
            "reconstructed_sections": {
                str(int(section)): [[float(x), float(y)] for x, y in points]
                for section, points in dict(self.reconstructed_sections or {}).items()
            },
            "contour_nodes_3d": {
                str(int(section)): [
                    [float(x), float(y), float(z)] for x, y, z in points
                ]
                for section, points in dict(self.contour_nodes_3d or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None):
        data = dict(payload or {})
        reconstructed_raw = dict(data.get("reconstructed_sections") or {})
        reconstructed_sections: dict[int, list[Point2]] = {}
        for key, points in reconstructed_raw.items():
            try:
                section = int(key)
            except Exception:
                continue
            reconstructed_sections[section] = _as_point2_list(points)
        contour_nodes_raw = dict(data.get("contour_nodes_3d") or {})
        contour_nodes_3d: dict[int, list[Point3]] = {}
        for key, points in contour_nodes_raw.items():
            try:
                section = int(key)
            except Exception:
                continue
            section_nodes: list[Point3] = []
            for point in list(points or []):
                try:
                    section_nodes.append(
                        (float(point[0]), float(point[1]), float(point[2]))
                    )
                except Exception:
                    continue
            contour_nodes_3d[section] = section_nodes
        presence_interval = data.get("presence_interval")
        interval_tuple = None
        if isinstance(presence_interval, (list, tuple)) and len(presence_interval) >= 2:
            interval_tuple = (int(presence_interval[0]), int(presence_interval[1]))
        return cls(
            region_id=str(data.get("region_id") or ""),
            label=str(data.get("label") or ""),
            observed_sections=[
                int(value) for value in list(data.get("observed_sections") or [])
            ],
            reconstructed_sections=reconstructed_sections,
            contour_nodes_3d=contour_nodes_3d,
            presence_interval=interval_tuple,
        )


@dataclass(slots=True)
class RegionPlanePolygon:
    region_id: str
    label: str
    state: str
    points: list[Point2]
    source: str = "model"


@dataclass(slots=True)
class PlanePolygonSet:
    orientation: str
    plane_index: int
    plane_position: float
    regions: list[RegionPlanePolygon] = field(default_factory=list)


@dataclass(slots=True)
class Brain3DModel:
    version: int
    source_orientation: str
    section_indices: list[int]
    section_positions: list[float]
    image_shape: tuple[int, int] | None
    config: Brain3DConfig
    regions: dict[str, RegionTrack] = field(default_factory=dict)
    coronal_overrides: dict[int, dict[str, list[Point2]]] = field(default_factory=dict)
    plane_presence: dict[int, dict[str, str]] = field(default_factory=dict)
    generated_coronal_planes: list[float] = field(default_factory=list)
    mesh_cache_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "source_orientation": str(self.source_orientation or "sagittal"),
            "section_indices": [int(value) for value in list(self.section_indices)],
            "section_positions": [
                float(value) for value in list(self.section_positions)
            ],
            "image_shape": (
                None
                if self.image_shape is None
                else [int(self.image_shape[0]), int(self.image_shape[1])]
            ),
            "config": self.config.to_dict(),
            "regions": {
                str(region_id): track.to_dict()
                for region_id, track in dict(self.regions or {}).items()
            },
            "coronal_overrides": {
                str(int(plane_index)): {
                    str(region_id): [[float(x), float(y)] for x, y in points]
                    for region_id, points in dict(region_points or {}).items()
                }
                for plane_index, region_points in dict(
                    self.coronal_overrides or {}
                ).items()
            },
            "plane_presence": {
                str(int(plane_index)): {
                    str(region_id): str(state or "present")
                    for region_id, state in dict(region_states or {}).items()
                }
                for plane_index, region_states in dict(
                    self.plane_presence or {}
                ).items()
            },
            "generated_coronal_planes": [
                float(position)
                for position in list(self.generated_coronal_planes or [])
            ],
            "mesh_cache_metadata": dict(self.mesh_cache_metadata or {}),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None):
        data = dict(payload or {})
        region_tracks = {
            str(region_id): RegionTrack.from_dict(track_payload)
            for region_id, track_payload in dict(data.get("regions") or {}).items()
        }
        overrides: dict[int, dict[str, list[Point2]]] = {}
        for plane_key, region_points in dict(
            data.get("coronal_overrides") or {}
        ).items():
            try:
                plane_index = int(plane_key)
            except Exception:
                continue
            overrides[plane_index] = {
                str(region_id): _as_point2_list(points)
                for region_id, points in dict(region_points or {}).items()
            }
        plane_presence: dict[int, dict[str, str]] = {}
        for plane_key, region_states in dict(data.get("plane_presence") or {}).items():
            try:
                plane_index = int(plane_key)
            except Exception:
                continue
            plane_presence[plane_index] = {
                str(region_id): str(state or "present")
                for region_id, state in dict(region_states or {}).items()
            }
        image_shape_raw = data.get("image_shape")
        image_shape = None
        if isinstance(image_shape_raw, (list, tuple)) and len(image_shape_raw) >= 2:
            image_shape = (int(image_shape_raw[0]), int(image_shape_raw[1]))
        model = cls(
            version=int(data.get("version") or 1),
            source_orientation=str(data.get("source_orientation") or "sagittal"),
            section_indices=[
                int(value) for value in list(data.get("section_indices") or [])
            ],
            section_positions=[
                float(value) for value in list(data.get("section_positions") or [])
            ],
            image_shape=image_shape,
            config=Brain3DConfig.from_dict(dict(data.get("config") or {})),
            regions=region_tracks,
            coronal_overrides=overrides,
            plane_presence=plane_presence,
            generated_coronal_planes=[
                float(position)
                for position in list(data.get("generated_coronal_planes") or [])
            ],
            mesh_cache_metadata=dict(data.get("mesh_cache_metadata") or {}),
            metadata=dict(data.get("metadata") or {}),
        )
        for track in list(model.regions.values()):
            _ensure_region_track_contour_nodes(model, track)
        return model


@dataclass(slots=True)
class MeshPayload:
    type: str
    regions: dict[str, dict[str, Any]]


def _iter_sagittal_pages(pages: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, page in enumerate(list(pages or [])):
        if isinstance(page, dict):
            page_index = int(page.get("page_index", index) or index)
            shapes = list(page.get("shapes") or [])
            normalized.append(
                {
                    "page_index": page_index,
                    "shapes": shapes,
                    "image_shape": page.get("image_shape"),
                }
            )
            continue
        if isinstance(page, (list, tuple)) and len(page) >= 2:
            page_index = int(page[0] or index)
            shapes = list(page[1] or [])
            normalized.append({"page_index": page_index, "shapes": shapes})
            continue
        normalized.append({"page_index": int(index), "shapes": list(page or [])})
    normalized.sort(key=lambda item: int(item.get("page_index") or 0))
    return normalized


def _build_interpolated_section_axis(
    source_section_indices: Sequence[int],
    interpolation_density: int,
    section_positions: Sequence[float] | None,
) -> tuple[list[int], list[float], dict[int, float], list[int]]:
    density = max(1, int(interpolation_density or 1))
    source_indices = [int(value) for value in list(source_section_indices or [])]
    if not source_indices:
        return [], [], {}, []

    scaled_source_indices = [int(value) * density for value in source_indices]
    min_section = int(min(scaled_source_indices))
    max_section = int(max(scaled_source_indices))
    section_indices = list(range(min_section, max_section + 1))

    if section_positions is not None and len(section_positions) == len(source_indices):
        source_positions = {
            int(section): float(position)
            for section, position in zip(scaled_source_indices, section_positions)
        }
        resolved_positions: list[float] = []
        for section in section_indices:
            if section in source_positions:
                resolved_positions.append(float(source_positions[section]))
                continue
            left_sections = [idx for idx in scaled_source_indices if idx < section]
            right_sections = [idx for idx in scaled_source_indices if idx > section]
            if not left_sections or not right_sections:
                resolved_positions.append(float(section))
                continue
            left = max(left_sections)
            right = min(right_sections)
            left_pos = float(source_positions[left])
            right_pos = float(source_positions[right])
            if right == left:
                resolved_positions.append(left_pos)
                continue
            ratio = (float(section) - float(left)) / (float(right) - float(left))
            resolved_positions.append(left_pos + ((right_pos - left_pos) * ratio))
        return (
            section_indices,
            resolved_positions,
            source_positions,
            scaled_source_indices,
        )

    resolved_positions = [float(index) / float(density) for index in section_indices]
    return section_indices, resolved_positions, {}, scaled_source_indices


def build_brain_3d_model(
    sagittal_pages: Sequence[Any],
    config: Brain3DConfig | dict[str, Any] | None,
) -> Brain3DModel:
    cfg = (
        config
        if isinstance(config, Brain3DConfig)
        else Brain3DConfig.from_dict(dict(config or {}))
    )
    pages = _iter_sagittal_pages(list(sagittal_pages or []))
    interpolation_density = max(1, int(cfg.interpolation_density or 1))
    if not pages:
        return Brain3DModel(
            version=2,
            source_orientation=str(cfg.source_orientation or "sagittal"),
            section_indices=[],
            section_positions=[],
            image_shape=None,
            config=cfg,
            regions={},
            metadata={"source_page_count": 0},
        )

    source_section_indices = [int(page["page_index"]) for page in pages]
    source_page_signatures = {
        int(page["page_index"]): _page_polygon_signature(list(page.get("shapes") or []))
        for page in pages
    }
    (
        section_indices,
        section_positions,
        source_positions_by_index,
        scaled_source_section_indices,
    ) = _build_interpolated_section_axis(
        source_section_indices,
        interpolation_density,
        cfg.section_positions,
    )

    image_width = 0.0
    image_height = 0.0
    raw_tracks: dict[str, dict[int, list[Point2]]] = {}
    labels_by_region: dict[str, str] = {}
    observed_sections_by_region: dict[str, set[int]] = {}

    for page in pages:
        page_index = int(page["page_index"]) * int(interpolation_density)
        for shape in list(page.get("shapes") or []):
            if _shape_shape_type(shape) != "polygon":
                continue
            points = _shape_points(shape)
            if len(points) < 3:
                continue
            for x, y in points:
                image_width = max(image_width, float(x))
                image_height = max(image_height, float(y))
            region_id = _region_id_from_shape(shape)
            labels_by_region[region_id] = _shape_label(shape)
            raw_tracks.setdefault(region_id, {})[page_index] = points
            observed_sections_by_region.setdefault(region_id, set()).add(page_index)

    region_tracks: dict[str, RegionTrack] = {}
    section_position_by_index = {
        int(section): float(position)
        for section, position in zip(section_indices, section_positions)
    }
    for region_id, section_map in raw_tracks.items():
        observed_sections = sorted(observed_sections_by_region.get(region_id) or [])
        if not observed_sections:
            continue
        min_section = observed_sections[0]
        max_section = observed_sections[-1]
        point_count = max(3, int(cfg.point_count or 0))
        aligned_observed_sections: dict[int, list[Point2]] = {}
        previous_observed_points: list[Point2] | None = None
        for observed_section in observed_sections:
            observed_points = _resample_polygon_points(
                section_map[observed_section], point_count
            )
            if previous_observed_points is not None:
                observed_points = _align_points_to_reference(
                    previous_observed_points, observed_points
                )
            aligned_observed_sections[int(observed_section)] = observed_points
            previous_observed_points = observed_points

        reconstructed: dict[int, list[Point2]] = {}

        for section in section_indices:
            if section < min_section or section > max_section:
                continue
            if section in section_map:
                reconstructed[section] = list(
                    aligned_observed_sections.get(int(section))
                    or _resample_polygon_points(
                        section_map[section],
                        point_count,
                    )
                )
                continue
            left_sections = [value for value in observed_sections if value < section]
            right_sections = [value for value in observed_sections if value > section]
            if not left_sections or not right_sections:
                continue
            left = left_sections[-1]
            right = right_sections[0]
            if right == left:
                reconstructed[section] = _resample_polygon_points(
                    section_map[left],
                    point_count,
                )
                continue
            ratio = (float(section) - float(left)) / (float(right) - float(left))
            reconstructed[section] = _interpolate_point_lists(
                list(aligned_observed_sections.get(int(left)) or []),
                list(aligned_observed_sections.get(int(right)) or []),
                ratio,
                point_count=point_count,
            )

        reconstructed = _smooth_longitudinal_sections(
            reconstructed,
            float(cfg.smoothing_longitudinal or 0.0),
        )

        region_tracks[region_id] = RegionTrack(
            region_id=region_id,
            label=labels_by_region.get(region_id) or _region_label_from_id(region_id),
            observed_sections=observed_sections,
            reconstructed_sections=reconstructed,
            contour_nodes_3d={
                int(section): [
                    (
                        float(
                            section_position_by_index.get(int(section), float(section))
                        ),
                        float(x),
                        float(y),
                    )
                    for x, y in list(points or [])
                ]
                for section, points in dict(reconstructed or {}).items()
            },
            presence_interval=(min_section, max_section),
        )

    image_shape = None
    if image_width > 0 and image_height > 0:
        image_shape = (int(round(image_width)) + 1, int(round(image_height)) + 1)

    return Brain3DModel(
        version=2,
        source_orientation=str(cfg.source_orientation or "sagittal"),
        section_indices=section_indices,
        section_positions=section_positions,
        image_shape=image_shape,
        config=cfg,
        regions=region_tracks,
        metadata={
            "source_page_count": len(pages),
            "source_page_indices": [int(value) for value in source_section_indices],
            "source_index_scale": int(interpolation_density),
            "source_section_axis": [
                {
                    "page_index": int(page_index),
                    "section_index": int(section_index),
                    "position": float(
                        section_position_by_index.get(
                            int(section_index), float(section_index)
                        )
                    ),
                }
                for page_index, section_index in zip(
                    source_section_indices, scaled_source_section_indices
                )
            ],
            "source_page_signatures": {
                str(int(index)): str(signature or "")
                for index, signature in source_page_signatures.items()
            },
            "source_section_positions": {
                str(int(index)): float(position)
                for index, position in source_positions_by_index.items()
            },
            "region_count": len(region_tracks),
            "source_orientation": str(cfg.source_orientation or "sagittal"),
            "section_positions": [float(value) for value in section_positions],
            "coronal_spacing": (
                None if cfg.coronal_spacing is None else float(cfg.coronal_spacing)
            ),
            "interpolation_density": int(interpolation_density),
            "workflow_mode": "additive",
            "region_presence_intervals": {
                str(region_id): (
                    None
                    if track.presence_interval is None
                    else [
                        int(track.presence_interval[0]),
                        int(track.presence_interval[1]),
                    ]
                )
                for region_id, track in dict(region_tracks or {}).items()
            },
        },
    )


def _coronal_positions(
    model: Brain3DModel,
    *,
    spacing: float | None,
    plane_count: int | None,
) -> list[float]:
    width = (
        int(model.image_shape[0])
        if model.image_shape is not None and len(model.image_shape) >= 1
        else 0
    )
    if width <= 0:
        max_y = 0.0
        for track in model.regions.values():
            for points in track.reconstructed_sections.values():
                for x, _ in points:
                    max_y = max(max_y, float(x))
        width = int(round(max_y)) + 1
    width = max(1, width)

    if plane_count is not None and int(plane_count) > 0:
        count = max(1, int(plane_count))
        if count == 1:
            return [0.0]
        last = float(width - 1)
        step = last / float(count - 1)
        return [step * idx for idx in range(count)]

    spacing_value = (
        float(spacing)
        if spacing is not None
        else (
            float(model.config.coronal_spacing)
            if model.config.coronal_spacing is not None
            else 1.0
        )
    )
    spacing_value = max(1e-6, spacing_value)
    positions: list[float] = []
    cursor = 0.0
    last = float(width - 1)
    while cursor <= (last + 1e-6):
        positions.append(float(cursor))
        cursor += spacing_value
    if positions and abs(positions[-1] - last) > 1e-6:
        positions.append(last)
    if not positions:
        positions = [0.0]
    return positions


def _slice_region_at_coronal(
    model: Brain3DModel,
    track: RegionTrack,
    coronal_position: float,
) -> list[Point2]:
    lower: list[Point2] = []
    upper: list[Point2] = []

    section_lookup = {
        int(section): float(position)
        for section, position in zip(model.section_indices, model.section_positions)
    }
    for section in model.section_indices:
        points = list(track.reconstructed_sections.get(int(section)) or [])
        if len(points) < 3:
            continue
        intersections: list[float] = []
        for index in range(len(points)):
            x1, y1 = points[index - 1]
            x2, y2 = points[index]
            left = min(x1, x2)
            right = max(x1, x2)
            if coronal_position < left or coronal_position > right:
                continue
            if abs(x2 - x1) < 1e-8:
                intersections.append(float((y1 + y2) * 0.5))
                continue
            ratio = (float(coronal_position) - float(x1)) / (float(x2) - float(x1))
            ratio = max(0.0, min(1.0, ratio))
            y = float(y1) + ((float(y2) - float(y1)) * ratio)
            intersections.append(float(y))
        if len(intersections) < 2:
            continue
        intersections.sort()
        section_pos = float(section_lookup.get(int(section), float(section)))
        lower.append((section_pos, float(intersections[0])))
        upper.append((section_pos, float(intersections[-1])))

    if len(lower) < 2 or len(upper) < 2:
        return []
    polygon = list(lower) + list(reversed(upper))
    if len(polygon) < 3:
        return []
    return polygon


def _ensure_region_track_contour_nodes(
    model: Brain3DModel,
    track: RegionTrack,
) -> None:
    if track.contour_nodes_3d:
        return
    section_lookup = {
        int(section): float(position)
        for section, position in zip(model.section_indices, model.section_positions)
    }
    contour_nodes: dict[int, list[Point3]] = {}
    for section, points in dict(track.reconstructed_sections or {}).items():
        section_value = int(section)
        z_value = float(section_lookup.get(section_value, float(section_value)))
        contour_nodes[section_value] = [
            (z_value, float(x), float(y)) for x, y in list(points or [])
        ]
    track.contour_nodes_3d = contour_nodes


def reslice_brain_model(
    model: Brain3DModel,
    orientation: str = "coronal",
    spacing: float | None = None,
    plane_count: int | None = None,
) -> list[PlanePolygonSet]:
    if str(orientation or "").strip().lower() != "coronal":
        raise ValueError("Only coronal reslicing is currently supported.")

    positions = _coronal_positions(
        model,
        spacing=spacing,
        plane_count=plane_count
        if plane_count is not None
        else model.config.coronal_plane_count,
    )
    planes: list[PlanePolygonSet] = []
    for plane_index, position in enumerate(positions):
        region_entries: list[RegionPlanePolygon] = []
        override_by_region = dict(model.coronal_overrides.get(int(plane_index)) or {})
        state_by_region = dict(model.plane_presence.get(int(plane_index)) or {})
        for region_id, track in dict(model.regions or {}).items():
            explicit_state = str(state_by_region.get(region_id) or "").strip().lower()
            if explicit_state == "hidden":
                region_entries.append(
                    RegionPlanePolygon(
                        region_id=region_id,
                        label=track.label,
                        state="hidden",
                        points=[],
                        source="state",
                    )
                )
                continue
            if explicit_state == "zero_area":
                region_entries.append(
                    RegionPlanePolygon(
                        region_id=region_id,
                        label=track.label,
                        state="zero_area",
                        points=[],
                        source="state",
                    )
                )
                continue
            override_points = _as_point2_list(override_by_region.get(region_id))
            if override_points:
                region_entries.append(
                    RegionPlanePolygon(
                        region_id=region_id,
                        label=track.label,
                        state="present",
                        points=override_points,
                        source="override",
                    )
                )
                continue
            points = _slice_region_at_coronal(model, track, float(position))
            if points and float(model.config.smoothing_inplane or 0.0) > 1e-6:
                points = _smooth_closed_polygon(
                    points,
                    float(model.config.smoothing_inplane or 0.0),
                )
            if points:
                region_entries.append(
                    RegionPlanePolygon(
                        region_id=region_id,
                        label=track.label,
                        state="present",
                        points=points,
                        source="model",
                    )
                )
            else:
                fallback_state = (
                    "created"
                    if explicit_state == "created"
                    else ("zero_area" if explicit_state == "zero_area" else "hidden")
                )
                region_entries.append(
                    RegionPlanePolygon(
                        region_id=region_id,
                        label=track.label,
                        state=fallback_state,
                        points=[],
                        source="model",
                    )
                )
        planes.append(
            PlanePolygonSet(
                orientation="coronal",
                plane_index=int(plane_index),
                plane_position=float(position),
                regions=region_entries,
            )
        )
    model.generated_coronal_planes = [float(position) for position in positions]
    model.metadata["generated_coronal_plane_count"] = len(planes)
    return planes


def apply_coronal_polygon_edit(
    model: Brain3DModel,
    plane_index: int,
    region_id: str,
    edited_shape: Any,
    *,
    guide_points: Sequence[Point2] | None = None,
    snapping_strength: float | None = None,
    snapping_max_distance: float = 8.0,
) -> Brain3DModel:
    normalized_region = str(region_id or "")
    if not normalized_region:
        raise ValueError("region_id is required.")
    plane_key = int(plane_index)
    if normalized_region not in model.regions:
        model.regions[normalized_region] = RegionTrack(
            region_id=normalized_region,
            label=_region_label_from_id(normalized_region),
            observed_sections=[],
            reconstructed_sections={},
            presence_interval=None,
        )
    if hasattr(edited_shape, "points"):
        points = _as_point2_list(getattr(edited_shape, "points", []) or [])
    else:
        points = _as_point2_list(edited_shape)
    raw_points = list(points)
    effective_strength = (
        float(snapping_strength)
        if snapping_strength is not None
        else float(model.config.snapping_strength or 0.0)
    )
    if guide_points and effective_strength > 1e-6:
        points = _snap_points_to_guides(
            points,
            list(guide_points or []),
            strength=effective_strength,
            max_distance=float(snapping_max_distance),
        )
        raw_overrides = model.metadata.setdefault("coronal_overrides_raw", {})
        if isinstance(raw_overrides, dict):
            plane_raw = raw_overrides.setdefault(str(int(plane_key)), {})
            if isinstance(plane_raw, dict):
                plane_raw[str(normalized_region)] = [
                    [float(x), float(y)] for x, y in raw_points
                ]
    model.coronal_overrides.setdefault(plane_key, {})[normalized_region] = list(points)
    neighborhood = model.metadata.setdefault("local_regeneration_requests", [])
    if isinstance(neighborhood, list):
        neighborhood.append(
            {
                "plane_index": int(plane_key),
                "region_id": normalized_region,
                "radius": 1,
            }
        )
    if points:
        model.plane_presence.setdefault(plane_key, {})[normalized_region] = "present"
    model.metadata.setdefault("edit_count", 0)
    model.metadata["edit_count"] = int(model.metadata["edit_count"]) + 1
    return model


def set_region_presence_on_plane(
    model: Brain3DModel,
    plane_index: int,
    region_id: str,
    state: str,
) -> Brain3DModel:
    normalized_state = str(state or "present").strip().lower() or "present"
    if normalized_state not in PRESENCE_STATES:
        raise ValueError("state must be one of: present, hidden, created, zero_area.")
    plane_key = int(plane_index)
    region_key = str(region_id or "")
    if not region_key:
        raise ValueError("region_id is required.")
    if region_key not in model.regions:
        model.regions[region_key] = RegionTrack(
            region_id=region_key,
            label=_region_label_from_id(region_key),
            observed_sections=[],
            reconstructed_sections={},
            presence_interval=None,
        )
    model.plane_presence.setdefault(plane_key, {})[region_key] = normalized_state
    if normalized_state == "hidden":
        model.coronal_overrides.setdefault(plane_key, {}).pop(region_key, None)
    return model


def export_brain_model_mesh(
    model: Brain3DModel,
    smoothing: float | None = None,
    *,
    region_ids: set[str] | None = None,
) -> MeshPayload:
    _ = smoothing
    for track in list(model.regions.values()):
        _ensure_region_track_contour_nodes(model, track)
    section_lookup = {
        int(section): float(position)
        for section, position in zip(model.section_indices, model.section_positions)
    }
    regions_payload: dict[str, dict[str, Any]] = {}
    for region_id, track in dict(model.regions or {}).items():
        if region_ids is not None and str(region_id) not in region_ids:
            continue
        section_ids = sorted(int(key) for key in track.reconstructed_sections.keys())
        if len(section_ids) < 2:
            continue
        first_points = list(track.reconstructed_sections.get(section_ids[0]) or [])
        point_count = len(first_points)
        if point_count < 3:
            continue
        vertices: list[Point3] = []
        index_map: dict[tuple[int, int], int] = {}
        for section in section_ids:
            points = list(track.reconstructed_sections.get(section) or [])
            if len(points) != point_count:
                points = _resample_polygon_points(points, point_count)
            for point_index, (x, y) in enumerate(points):
                vertex = (
                    float(section_lookup.get(section, float(section))),
                    float(x),
                    float(y),
                )
                index_map[(section, point_index)] = len(vertices)
                vertices.append(vertex)
        faces: list[tuple[int, int, int]] = []
        for left_section, right_section in zip(section_ids[:-1], section_ids[1:]):
            for point_index in range(point_count):
                next_index = (point_index + 1) % point_count
                a = index_map[(left_section, point_index)]
                b = index_map[(left_section, next_index)]
                c = index_map[(right_section, point_index)]
                d = index_map[(right_section, next_index)]
                faces.append((a, c, b))
                faces.append((b, c, d))
        regions_payload[region_id] = {
            "label": str(track.label or _region_label_from_id(region_id)),
            "vertices": [[x, y, z] for x, y, z in vertices],
            "faces": [[a, b, c] for a, b, c in faces],
            "source": "brain3d_contour_volume",
        }
    payload = MeshPayload(type="tri_mesh", regions=regions_payload)
    model.mesh_cache_metadata = {
        "kind": "tri_mesh",
        "region_count": len(regions_payload),
        "generated_at_epoch_s": float(time.time()),
        "smoothing": None if smoothing is None else float(smoothing),
        "filtered_region_ids": sorted(str(value) for value in (region_ids or set())),
    }
    return payload


def export_brain_model_mesh_ply(
    model: Brain3DModel,
    output_path: str | Path,
    *,
    smoothing: float | None = None,
    region_ids: set[str] | None = None,
) -> Path:
    mesh = export_brain_model_mesh(model, smoothing=smoothing, region_ids=region_ids)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices: list[Point3] = []
    faces: list[tuple[int, int, int]] = []
    vertex_offset = 0
    for region in mesh.regions.values():
        region_vertices = [
            (float(x), float(y), float(z))
            for x, y, z in list(region.get("vertices") or [])
        ]
        region_faces = [
            (int(a), int(b), int(c)) for a, b, c in list(region.get("faces") or [])
        ]
        vertices.extend(region_vertices)
        for a, b, c in region_faces:
            faces.append(
                (
                    int(a) + int(vertex_offset),
                    int(b) + int(vertex_offset),
                    int(c) + int(vertex_offset),
                )
            )
        vertex_offset += len(region_vertices)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model.mesh_cache_metadata = {
        **dict(model.mesh_cache_metadata or {}),
        "format": "ply",
        "path": str(path),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "generated_at_epoch_s": float(time.time()),
    }
    return path


def export_brain_model_mesh_obj(
    model: Brain3DModel,
    output_path: str | Path,
    *,
    smoothing: float | None = None,
    region_ids: set[str] | None = None,
) -> tuple[Path, dict[str, str]]:
    mesh = export_brain_model_mesh(model, smoothing=smoothing, region_ids=region_ids)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Annolid Brain3D mesh export (OBJ)"]
    vertex_offset = 1
    object_region_map: dict[str, str] = {}
    object_index = 1

    for region_id, region in mesh.regions.items():
        region_vertices = [
            (float(x), float(y), float(z))
            for x, y, z in list(region.get("vertices") or [])
        ]
        region_faces = [
            (int(a), int(b), int(c)) for a, b, c in list(region.get("faces") or [])
        ]
        if len(region_vertices) < 3 or not region_faces:
            continue
        object_name = f"brain3d_region_{object_index:04d}"
        object_index += 1
        object_region_map[object_name] = str(region_id)
        lines.append(f"o {object_name}")
        lines.append(f"g {object_name}")
        for x, y, z in region_vertices:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for a, b, c in region_faces:
            lines.append(
                f"f {int(a) + vertex_offset} {int(b) + vertex_offset} {int(c) + vertex_offset}"
            )
        vertex_offset += len(region_vertices)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model.mesh_cache_metadata = {
        **dict(model.mesh_cache_metadata or {}),
        "format": "obj",
        "path": str(path),
        "object_region_count": int(len(object_region_map)),
        "generated_at_epoch_s": float(time.time()),
    }
    return path, object_region_map


def brain_model_from_other_data(
    other_data: dict[str, Any] | None,
) -> Brain3DModel | None:
    payload = dict(other_data or {}).get("brain_3d_model")
    if not isinstance(payload, dict):
        return None
    try:
        return Brain3DModel.from_dict(payload)
    except Exception:
        return None


def store_brain_model_in_other_data(
    other_data: dict[str, Any] | None,
    model: Brain3DModel,
) -> dict[str, Any]:
    merged = dict(other_data or {})
    merged["brain_3d_model"] = model.to_dict()
    return merged


def save_brain_model_sidecar(model: Brain3DModel, sidecar_path: str | Path) -> Path:
    path = Path(sidecar_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    return path


def load_brain_model_sidecar(sidecar_path: str | Path) -> Brain3DModel:
    path = Path(sidecar_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Brain 3D sidecar content must be a JSON object.")
    return Brain3DModel.from_dict(payload)


def materialize_coronal_plane_shapes(
    plane: PlanePolygonSet,
    *,
    include_hidden: bool = False,
) -> list[Any]:
    from annolid.gui.shape import Shape
    from annolid.gui.polygon_tools import set_polygon_edit_state

    shapes: list[Any] = []
    for region in list(plane.regions or []):
        if region.state != "present" and not include_hidden:
            continue
        if len(region.points) < 3:
            continue
        shape = Shape(
            label=str(region.label or _region_label_from_id(region.region_id)),
            shape_type="polygon",
            visible=(region.state == "present"),
        )
        for x, y in region.points:
            shape.addPoint(QtCore.QPointF(float(x), float(y)))
        shape.close()
        set_polygon_edit_state(
            shape,
            "inferred",
            source_orientation="coronal",
            source_plane_index=int(plane.plane_index),
            source_plane_position=float(plane.plane_position),
            region_id=str(region.region_id),
            source_kind=str(region.source or "model"),
            brain3d_state=str(region.state or "present"),
        )
        shapes.append(shape)
    return shapes
