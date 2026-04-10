from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qtpy import QtCore

from annolid.gui.polygon_tools import interpolate_closed_polygon_points
from annolid.gui.polygon_tools import polygon_identity_key


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
PRESENCE_STATES = {"present", "hidden", "created"}


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


@dataclass(slots=True)
class Brain3DConfig:
    source_orientation: str = "sagittal"
    point_count: int = 64
    section_positions: list[float] | None = None
    coronal_spacing: float | None = 1.0
    coronal_plane_count: int | None = None
    smoothing_longitudinal: float = 0.0
    smoothing_inplane: float = 0.0
    snapping_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_orientation": str(self.source_orientation or "sagittal"),
            "point_count": int(self.point_count or 64),
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
            "snapping_strength": float(self.snapping_strength or 0.0),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None):
        data = dict(payload or {})
        section_positions = data.get("section_positions")
        return cls(
            source_orientation=str(data.get("source_orientation") or "sagittal"),
            point_count=max(3, int(data.get("point_count") or 64)),
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
            snapping_strength=float(data.get("snapping_strength") or 0.0),
        )


@dataclass(slots=True)
class RegionTrack:
    region_id: str
    label: str
    observed_sections: list[int] = field(default_factory=list)
    reconstructed_sections: dict[int, list[Point2]] = field(default_factory=dict)
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
        return cls(
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
            metadata=dict(data.get("metadata") or {}),
        )


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
    if not pages:
        return Brain3DModel(
            version=1,
            source_orientation=str(cfg.source_orientation or "sagittal"),
            section_indices=[],
            section_positions=[],
            image_shape=None,
            config=cfg,
            regions={},
            metadata={"source_page_count": 0},
        )

    source_section_indices = [int(page["page_index"]) for page in pages]
    if source_section_indices:
        min_section = int(min(source_section_indices))
        max_section = int(max(source_section_indices))
        section_indices = list(range(min_section, max_section + 1))
    else:
        section_indices = []

    if cfg.section_positions is not None and len(cfg.section_positions) == len(
        source_section_indices
    ):
        source_positions = {
            int(section): float(position)
            for section, position in zip(source_section_indices, cfg.section_positions)
        }
        section_positions: list[float] = []
        for section in section_indices:
            if section in source_positions:
                section_positions.append(float(source_positions[section]))
                continue
            left_sections = [idx for idx in source_section_indices if idx < section]
            right_sections = [idx for idx in source_section_indices if idx > section]
            if not left_sections or not right_sections:
                section_positions.append(float(section))
                continue
            left = max(left_sections)
            right = min(right_sections)
            left_pos = float(source_positions[left])
            right_pos = float(source_positions[right])
            if right == left:
                section_positions.append(left_pos)
                continue
            ratio = (float(section) - float(left)) / (float(right) - float(left))
            section_positions.append(left_pos + ((right_pos - left_pos) * ratio))
    else:
        section_positions = [float(index) for index in section_indices]

    image_width = 0.0
    image_height = 0.0
    raw_tracks: dict[str, dict[int, list[Point2]]] = {}
    labels_by_region: dict[str, str] = {}
    observed_sections_by_region: dict[str, set[int]] = {}

    for page in pages:
        page_index = int(page["page_index"])
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
    for region_id, section_map in raw_tracks.items():
        observed_sections = sorted(observed_sections_by_region.get(region_id) or [])
        if not observed_sections:
            continue
        min_section = observed_sections[0]
        max_section = observed_sections[-1]
        point_count = max(3, int(cfg.point_count or 0))
        reconstructed: dict[int, list[Point2]] = {}

        for section in section_indices:
            if section < min_section or section > max_section:
                continue
            if section in section_map:
                reconstructed[section] = _resample_polygon_points(
                    section_map[section],
                    point_count,
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
                _resample_polygon_points(section_map[left], point_count),
                _resample_polygon_points(section_map[right], point_count),
                ratio,
                point_count=point_count,
            )

        region_tracks[region_id] = RegionTrack(
            region_id=region_id,
            label=labels_by_region.get(region_id) or _region_label_from_id(region_id),
            observed_sections=observed_sections,
            reconstructed_sections=reconstructed,
            presence_interval=(min_section, max_section),
        )

    image_shape = None
    if image_width > 0 and image_height > 0:
        image_shape = (int(round(image_width)) + 1, int(round(image_height)) + 1)

    return Brain3DModel(
        version=1,
        source_orientation=str(cfg.source_orientation or "sagittal"),
        section_indices=section_indices,
        section_positions=section_positions,
        image_shape=image_shape,
        config=cfg,
        regions=region_tracks,
        metadata={
            "source_page_count": len(pages),
            "region_count": len(region_tracks),
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
                fallback_state = "created" if explicit_state == "created" else "hidden"
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
        raise ValueError("state must be one of: present, hidden, created.")
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
) -> MeshPayload:
    _ = smoothing
    section_lookup = {
        int(section): float(position)
        for section, position in zip(model.section_indices, model.section_positions)
    }
    regions_payload: dict[str, dict[str, Any]] = {}
    for region_id, track in dict(model.regions or {}).items():
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
    return MeshPayload(type="tri_mesh", regions=regions_payload)


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
