from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from qtpy import QtCore, QtGui

from annolid.gui.shape import Shape
from annolid.gui.label_file import LabelFile
from annolid.postprocessing.zone_schema import (
    ZONE_SCHEMA_VERSION,
    build_zone_shape,
    normalize_zone_shape,
)
from annolid.utils.shapes import shape_to_dict


_ZONE_KIND_COLORS: dict[str, tuple[str, str]] = {
    "chamber": ("#388E3C", "#A5D6A7"),
    "doorway": ("#F57C00", "#FFE0B2"),
    "barrier_edge": ("#1976D2", "#BBDEFB"),
    "interaction_zone": ("#C2185B", "#F8BBD0"),
    "custom": ("#546E7A", "#CFD8DC"),
}


def _color_from_hex(value: str, alpha: int) -> QtGui.QColor:
    color = QtGui.QColor(str(value))
    if alpha >= 0:
        color.setAlpha(max(0, min(255, int(alpha))))
    return color


def zone_kind_palette(zone_kind: str) -> tuple[QtGui.QColor, QtGui.QColor]:
    kind = str(zone_kind or "custom").strip().lower()
    stroke_hex, fill_hex = _ZONE_KIND_COLORS.get(kind, _ZONE_KIND_COLORS["custom"])
    return _color_from_hex(stroke_hex, 255), _color_from_hex(fill_hex, 90)


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [part.strip() for part in tags.split(",")]
        return [tag for tag in raw if tag]
    if isinstance(tags, Sequence):
        out: list[str] = []
        for item in tags:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out
    text = str(tags or "").strip()
    return [text] if text else []


def _shape_flags(shape: Shape | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(shape, Mapping):
        flags = dict(shape.get("flags") or {})
    else:
        flags = dict(getattr(shape, "flags", None) or {})
    return flags


def _shape_points(shape: Shape | Mapping[str, Any]) -> list[list[float]]:
    if isinstance(shape, Mapping):
        points = shape.get("points") or []
        out: list[list[float]] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                out.append([float(point[0]), float(point[1])])
        return out
    points: list[list[float]] = []
    for point in getattr(shape, "points", []) or []:
        x = point.x() if hasattr(point, "x") else point[0]
        y = point.y() if hasattr(point, "y") else point[1]
        points.append([float(x), float(y)])
    return points


def normalize_zone_flags(
    shape: Shape | Mapping[str, Any],
    *,
    label: str | None = None,
    zone_kind: str | None = None,
    phase: str | None = None,
    occupant_role: str | None = None,
    access_state: str | None = None,
    tags: Any = None,
    extra_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    flags = _shape_flags(shape)
    if extra_flags:
        flags.update(dict(extra_flags))
    resolved_label = str(
        label if label is not None else getattr(shape, "label", flags.get("label", ""))
    ).strip()
    resolved_zone_kind = str(
        zone_kind
        if zone_kind is not None
        else flags.get("zone_kind") or flags.get("zone_type") or "custom"
    ).strip()
    resolved_phase = str(
        phase if phase is not None else flags.get("phase") or "custom"
    ).strip()
    resolved_role = str(
        occupant_role
        if occupant_role is not None
        else flags.get("occupant_role") or "unknown"
    ).strip()
    resolved_access = str(
        access_state
        if access_state is not None
        else flags.get("access_state") or "unknown"
    ).strip()

    normalized_tags = _normalize_tags(tags if tags is not None else flags.get("tags"))
    flags.update(
        {
            "semantic_type": "zone",
            "shape_category": "zone",
            "zone_kind": resolved_zone_kind or "custom",
            "phase": resolved_phase or "custom",
            "occupant_role": resolved_role or "unknown",
            "access_state": resolved_access or "unknown",
            "schema_version": ZONE_SCHEMA_VERSION,
            "compatibility_mode": "explicit",
        }
    )
    if normalized_tags:
        flags["tags"] = normalized_tags
    if resolved_label:
        flags["zone_label"] = resolved_label
    return flags


def shape_to_zone_payload(
    shape: Shape | Mapping[str, Any],
    *,
    label: str | None = None,
    zone_kind: str | None = None,
    phase: str | None = None,
    occupant_role: str | None = None,
    access_state: str | None = None,
    tags: Any = None,
    description: str | None = None,
    extra_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = shape_to_dict(shape) if not isinstance(shape, Mapping) else dict(shape)
    payload["points"] = _shape_points(payload)
    payload["shape_type"] = str(payload.get("shape_type") or "polygon")
    payload["label"] = str(
        label if label is not None else payload.get("label") or ""
    ).strip()
    if description is not None:
        payload["description"] = str(description)
    flags = normalize_zone_flags(
        payload,
        label=payload["label"],
        zone_kind=zone_kind,
        phase=phase,
        occupant_role=occupant_role,
        access_state=access_state,
        tags=tags,
        extra_flags=extra_flags,
    )
    payload["flags"] = flags
    return payload


def zone_payload_to_shape(payload: Mapping[str, Any]) -> Shape:
    shape = Shape(
        label=str(payload.get("label") or "").strip(),
        shape_type=str(payload.get("shape_type") or "polygon"),
        flags=dict(payload.get("flags") or {}),
        description=str(payload.get("description") or "").strip(),
        group_id=payload.get("group_id"),
        visible=bool(payload.get("visible", True)),
    )
    points = payload.get("points") or []
    shape.points = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            shape.points.append(QtCore.QPointF(float(point[0]), float(point[1])))
    if not shape.points:
        shape.points = []
    zone_spec = normalize_zone_shape(payload)
    stroke, fill = zone_kind_palette(zone_spec.zone_kind)
    shape.line_color = stroke
    shape.fill_color = fill
    shape.select_line_color = QtGui.QColor(255, 255, 255, 255)
    shape.select_fill_color = QtGui.QColor(
        stroke.red(), stroke.green(), stroke.blue(), 160
    )
    shape.fill = True
    shape.other_data = dict(payload.get("other_data") or {})
    return shape


def is_zone_shape(shape: Shape | Mapping[str, Any]) -> bool:
    try:
        payload = (
            shape_to_dict(shape) if not isinstance(shape, Mapping) else dict(shape)
        )
        return bool(normalize_zone_shape(payload).is_zone)
    except Exception:
        return False


def iter_zone_shapes(
    shapes: Sequence[Shape | Mapping[str, Any]],
) -> list[Shape | Mapping[str, Any]]:
    return [shape for shape in shapes if is_zone_shape(shape)]


def zone_file_for_source(source_path: str | Path | None) -> Path | None:
    if source_path is None:
        return None
    source = Path(source_path)
    if not str(source).strip():
        return None
    if source.name in {"", ".", ".."}:
        return None
    stem = source.stem or "zones"
    return source.with_name(f"{stem}_zones.json")


def default_zone_label(zone_kind: str, existing_labels: Sequence[str]) -> str:
    prefix = str(zone_kind or "zone").strip().lower() or "zone"
    taken = set(str(label or "").strip().lower() for label in existing_labels)
    index = 1
    while True:
        candidate = f"{prefix}_{index}"
        if candidate.lower() not in taken:
            return candidate
        index += 1


def build_zone_popup_defaults(
    *,
    label: str,
    zone_kind: str,
    phase: str,
    occupant_role: str,
    access_state: str,
    tags: Any = None,
    description: str = "",
) -> dict[str, Any]:
    flags = normalize_zone_flags(
        {},
        label=label,
        zone_kind=zone_kind,
        phase=phase,
        occupant_role=occupant_role,
        access_state=access_state,
        tags=tags,
    )
    return {
        "text": str(label or "").strip(),
        "flags": {
            key: flags[key]
            for key in (
                "semantic_type",
                "shape_category",
                "zone_kind",
                "phase",
                "occupant_role",
                "access_state",
                "schema_version",
                "compatibility_mode",
                "tags",
                "zone_label",
            )
            if key in flags
        },
        "group_id": None,
        "description": str(description or "").strip(),
    }


def _grid_label(row: int, col: int, rows: int, cols: int) -> str:
    position_names = {
        (0, 0): "north_west",
        (0, 1): "north",
        (0, 2): "north_east",
        (1, 0): "west",
        (1, 1): "center",
        (1, 2): "east",
        (2, 0): "south_west",
        (2, 1): "south",
        (2, 2): "south_east",
    }
    if (row, col) in position_names and rows == 3 and cols == 3:
        return f"{position_names[(row, col)]}_chamber"
    return f"chamber_r{row + 1}c{col + 1}"


def generate_chamber_grid_layout(
    image_width: int,
    image_height: int,
    *,
    rows: int = 3,
    cols: int = 3,
    margin_ratio: float = 0.045,
    gap_ratio: float = 0.018,
    phase: str = "custom",
    zone_kind: str = "chamber",
    layout_tag: str = "3x3_chamber",
) -> list[Shape]:
    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    margin_x = max(0.0, min(width / 3.0, width * float(margin_ratio)))
    margin_y = max(0.0, min(height / 3.0, height * float(margin_ratio)))
    gap_x = max(0.0, width * float(gap_ratio))
    gap_y = max(0.0, height * float(gap_ratio))

    usable_width = max(1.0, width - 2.0 * margin_x - max(0, cols - 1) * gap_x)
    usable_height = max(1.0, height - 2.0 * margin_y - max(0, rows - 1) * gap_y)
    cell_w = usable_width / cols
    cell_h = usable_height / rows

    shapes: list[Shape] = []
    for row in range(rows):
        for col in range(cols):
            x1 = margin_x + col * (cell_w + gap_x)
            y1 = margin_y + row * (cell_h + gap_y)
            x2 = min(width - margin_x, x1 + cell_w)
            y2 = min(height - margin_y, y1 + cell_h)
            label = _grid_label(row, col, rows, cols)
            payload = build_zone_shape(
                label,
                [[x1, y1], [x2, y2]],
                shape_type="rectangle",
                zone_kind=zone_kind,
                phase=phase,
                occupant_role="unknown",
                access_state="open",
                description=f"preset:{layout_tag}",
                extra_flags={
                    "layout_tag": layout_tag,
                    "layout_rows": rows,
                    "layout_cols": cols,
                    "layout_position": {"row": row, "col": col},
                    "tags": [layout_tag, f"row_{row + 1}", f"col_{col + 1}"],
                },
            )
            shapes.append(zone_payload_to_shape(payload))
    return shapes


def _shape_bbox(shape: Shape) -> tuple[float, float, float, float] | None:
    points = _shape_points(shape)
    if len(points) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _make_social_zone_shape(
    *,
    label: str,
    edge_bbox: tuple[float, float, float, float],
    boundary: str,
    phase: str,
    layout_tag: str,
    tags: list[str],
    zone_kind: str = "interaction_zone",
    occupant_role: str = "rover",
    access_state: str = "open",
) -> Shape:
    x1, y1, x2, y2 = edge_bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    depth = max(1.0, min(width, height) * 0.18)
    span = max(1.0, min(width, height) * 0.30)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if boundary == "left":
        sx1 = x1
        sx2 = min(x2, x1 + depth)
        sy1 = max(y1, center_y - span / 2.0)
        sy2 = min(y2, center_y + span / 2.0)
    elif boundary == "right":
        sx1 = max(x1, x2 - depth)
        sx2 = x2
        sy1 = max(y1, center_y - span / 2.0)
        sy2 = min(y2, center_y + span / 2.0)
    elif boundary == "top":
        sx1 = max(x1, center_x - span / 2.0)
        sx2 = min(x2, center_x + span / 2.0)
        sy1 = y1
        sy2 = min(y2, y1 + depth)
    elif boundary == "bottom":
        sx1 = max(x1, center_x - span / 2.0)
        sx2 = min(x2, center_x + span / 2.0)
        sy1 = max(y1, y2 - depth)
        sy2 = y2
    else:
        raise ValueError(f"Unsupported boundary placement: {boundary!r}")

    payload = build_zone_shape(
        label,
        [[sx1, sy1], [sx2, sy2]],
        shape_type="rectangle",
        zone_kind=zone_kind,
        phase=phase,
        occupant_role=occupant_role,
        access_state=access_state,
        description=f"preset:{layout_tag}",
        extra_flags={
            "layout_tag": layout_tag,
            "tags": tags,
        },
    )
    return zone_payload_to_shape(payload)


def generate_social_door_layout(
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float = 0.045,
    gap_ratio: float = 0.018,
    phase: str = "social",
    layout_tag: str = "3x3_social_doors",
) -> list[Shape]:
    chamber_layout = generate_chamber_grid_layout(
        image_width,
        image_height,
        rows=3,
        cols=3,
        margin_ratio=margin_ratio,
        gap_ratio=gap_ratio,
        phase=phase,
        zone_kind="chamber",
        layout_tag="3x3_chamber",
    )
    chamber_map: dict[tuple[int, int], Shape] = {}
    for shape in chamber_layout:
        layout_pos = dict(shape.flags or {}).get("layout_position") or {}
        row = layout_pos.get("row")
        col = layout_pos.get("col")
        if row is None or col is None:
            continue
        chamber_map[(int(row), int(col))] = shape

    corner_to_edges = [
        ((0, 0), [((0, 1), "left"), ((1, 0), "top")], "north_west"),
        ((0, 2), [((0, 1), "right"), ((1, 2), "top")], "north_east"),
        ((2, 0), [((2, 1), "left"), ((1, 0), "bottom")], "south_west"),
        ((2, 2), [((2, 1), "right"), ((1, 2), "bottom")], "south_east"),
    ]
    social_shapes: list[Shape] = []
    for _, edge_specs, corner_name in corner_to_edges:
        for (edge_row, edge_col), boundary in edge_specs:
            edge_shape = chamber_map.get((edge_row, edge_col))
            if edge_shape is None:
                continue
            edge_bbox = _shape_bbox(edge_shape)
            if edge_bbox is None:
                continue
            edge_name = _grid_label(edge_row, edge_col, 3, 3).replace("_chamber", "")
            label = f"{corner_name}_{edge_name}_social_zone"
            tags = [
                layout_tag,
                "social_zone",
                "door_social",
                f"corner_{corner_name}",
                f"adjacent_{edge_name}",
            ]
            social_shapes.append(
                _make_social_zone_shape(
                    label=label,
                    edge_bbox=edge_bbox,
                    boundary=boundary,
                    phase=phase,
                    layout_tag=layout_tag,
                    tags=tags,
                )
            )

    return social_shapes


ARENA_LAYOUT_PRESETS: tuple[dict[str, str], ...] = (
    {
        "key": "3x3_chamber",
        "title": "3x3 Chamber Layout",
        "description": (
            "Generate nine editable chamber zones in a 3x3 grid covering the loaded frame."
        ),
    },
    {
        "key": "3x3_social_doors",
        "title": "3x3 Social Door Layout",
        "description": (
            "Generate eight editable rover-side social zones around the mesh doors."
        ),
    },
)


def available_arena_layout_presets() -> list[dict[str, str]]:
    return [dict(preset) for preset in ARENA_LAYOUT_PRESETS]


def generate_arena_layout_preset(
    preset_key: str,
    image_width: int,
    image_height: int,
) -> list[Shape]:
    key = str(preset_key or "").strip().lower()
    if key == "3x3_chamber":
        return generate_chamber_grid_layout(
            image_width,
            image_height,
            rows=3,
            cols=3,
            layout_tag="3x3_chamber",
        )
    if key == "3x3_social_doors":
        return generate_social_door_layout(
            image_width,
            image_height,
            layout_tag="3x3_social_doors",
        )
    raise ValueError(f"Unsupported arena layout preset: {preset_key!r}")


def write_zone_json(
    filename: str | Path,
    *,
    shapes: Sequence[Shape | Mapping[str, Any]],
    image_path: str,
    image_width: int,
    image_height: int,
    image_data: bytes | None = None,
    caption: str | None = None,
    other_data: Mapping[str, Any] | None = None,
) -> None:
    lf = LabelFile()
    zone_shapes = [shape_to_zone_payload(shape) for shape in shapes]
    lf.save(
        str(filename),
        zone_shapes,
        image_path,
        image_height,
        image_width,
        imageData=image_data,
        otherData=dict(other_data or {}),
        flags={},
        caption=caption,
    )
