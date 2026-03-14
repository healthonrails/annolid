from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np


_COMMAND_RE = re.compile(
    r"[MmLlHhVvCcQqTtSsAaZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
)
_MATRIX_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")


@dataclass(frozen=True)
class ImportedPath:
    id: str
    label: str | None
    kind: str
    points: list[tuple[float, float]]
    stroke: str | None = None
    fill: str | None = None
    text: str | None = None
    layer_name: str | None = None
    transform: list[float] | None = None
    source_tag: str | None = None


@dataclass(frozen=True)
class ImportedVectorDocument:
    source_path: str
    width: float | None
    height: float | None
    view_box: list[float] | None
    shapes: list[ImportedPath]
    source_kind: str = "svg"


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else default


def _parse_style(style: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in (style or "").split(";"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _resolve_color(element: ET.Element, name: str) -> str | None:
    direct = element.get(name)
    if direct is not None:
        return direct
    return _parse_style(element.get("style")).get(name)


def _is_hidden(element: ET.Element) -> bool:
    style = _parse_style(element.get("style"))
    return (
        str(element.get("display", "")).strip().lower() == "none"
        or str(element.get("visibility", "")).strip().lower() == "hidden"
        or style.get("display", "").lower() == "none"
        or style.get("visibility", "").lower() == "hidden"
    )


def parse_svg_transform(value: str | None) -> np.ndarray:
    if not value:
        return np.eye(3, dtype=float)
    matrix = np.eye(3, dtype=float)
    for name, raw_args in _MATRIX_RE.findall(value):
        args = [
            float(token)
            for token in re.findall(
                r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw_args or ""
            )
        ]
        name = name.lower()
        current = np.eye(3, dtype=float)
        if name == "translate":
            current = np.array(
                [
                    [1.0, 0.0, args[0] if args else 0.0],
                    [0.0, 1.0, args[1] if len(args) > 1 else 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            current = np.array(
                [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float
            )
        elif name == "rotate":
            angle = math.radians(args[0] if args else 0.0)
            c = math.cos(angle)
            s = math.sin(angle)
            rotation = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float
            )
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                current = (
                    np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
                    @ rotation
                    @ np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
                )
            else:
                current = rotation
        elif name == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            current = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=float)
        matrix = current @ matrix
    return matrix


def _sample_quadratic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    steps: int = 16,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for idx in range(1, steps + 1):
        t = idx / steps
        omt = 1.0 - t
        samples.append(
            (
                omt * omt * p0[0] + 2.0 * omt * t * p1[0] + t * t * p2[0],
                omt * omt * p0[1] + 2.0 * omt * t * p1[1] + t * t * p2[1],
            )
        )
    return samples


def _sample_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    *,
    steps: int = 24,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for idx in range(1, steps + 1):
        t = idx / steps
        omt = 1.0 - t
        samples.append(
            (
                omt**3 * p0[0]
                + 3.0 * omt * omt * t * p1[0]
                + 3.0 * omt * t * t * p2[0]
                + t**3 * p3[0],
                omt**3 * p0[1]
                + 3.0 * omt * omt * t * p1[1]
                + 3.0 * omt * t * t * p2[1]
                + t**3 * p3[1],
            )
        )
    return samples


def _token_iter(path_data: str) -> Iterator[str]:
    yield from _COMMAND_RE.findall(path_data or "")


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / ((dx * dx) + (dy * dy))
    t = max(0.0, min(1.0, t))
    proj_x = x1 + (t * dx)
    proj_y = y1 + (t * dy)
    return math.hypot(px - proj_x, py - proj_y)


def _simplify_polyline(
    points: list[tuple[float, float]],
    *,
    epsilon: float = 0.75,
) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return list(points)

    closed = len(points) > 2 and points[0] == points[-1]
    work_points = list(points[:-1] if closed else points)
    if len(work_points) <= 2:
        return list(points)

    def _rdp(segment: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(segment) <= 2:
            return list(segment)
        start = segment[0]
        end = segment[-1]
        max_distance = -1.0
        split_index = -1
        for index in range(1, len(segment) - 1):
            distance = _point_line_distance(segment[index], start, end)
            if distance > max_distance:
                max_distance = distance
                split_index = index
        if max_distance > epsilon and split_index > 0:
            left = _rdp(segment[: split_index + 1])
            right = _rdp(segment[split_index:])
            return left[:-1] + right
        return [start, end]

    simplified = _rdp(work_points)
    if closed:
        simplified.append(simplified[0])
    return simplified


def flatten_svg_path(path_data: str) -> list[tuple[float, float]]:
    tokens = list(_token_iter(path_data))
    points: list[tuple[float, float]] = []
    idx = 0
    command = None
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    while idx < len(tokens):
        token = tokens[idx]
        if re.fullmatch(r"[A-Za-z]", token):
            command = token
            idx += 1
        if command is None:
            raise ValueError("Invalid SVG path data")
        absolute = command.isupper()
        op = command.upper()

        def next_float() -> float:
            nonlocal idx
            value = float(tokens[idx])
            idx += 1
            return value

        if op == "M":
            x = next_float()
            y = next_float()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            start = current
            points.append(current)
            command = "L" if absolute else "l"
        elif op == "L":
            x = next_float()
            y = next_float()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.append(current)
        elif op == "H":
            x = next_float()
            current = (x, current[1]) if absolute else (current[0] + x, current[1])
            points.append(current)
        elif op == "V":
            y = next_float()
            current = (current[0], y) if absolute else (current[0], current[1] + y)
            points.append(current)
        elif op == "Q":
            x1, y1, x, y = next_float(), next_float(), next_float(), next_float()
            control = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            end = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.extend(_sample_quadratic(current, control, end))
            current = end
        elif op == "T":
            x, y = next_float(), next_float()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.append(current)
        elif op == "C":
            x1, y1, x2, y2, x, y = (
                next_float(),
                next_float(),
                next_float(),
                next_float(),
                next_float(),
                next_float(),
            )
            control1 = (x1, y1) if absolute else (current[0] + x1, current[1] + y1)
            control2 = (x2, y2) if absolute else (current[0] + x2, current[1] + y2)
            end = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.extend(_sample_cubic(current, control1, control2, end))
            current = end
        elif op == "S":
            x2, y2, x, y = next_float(), next_float(), next_float(), next_float()
            control2 = (x2, y2) if absolute else (current[0] + x2, current[1] + y2)
            end = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.extend(_sample_cubic(current, current, control2, end))
            current = end
        elif op == "A":
            _ = [next_float() for _ in range(5)]
            x, y = next_float(), next_float()
            current = (x, y) if absolute else (current[0] + x, current[1] + y)
            points.append(current)
        elif op == "Z":
            current = start
            if points and points[-1] != start:
                points.append(start)
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
    return _simplify_polyline(points)


def _apply_transform(points: list[tuple[float, float]], matrix: np.ndarray):
    if not points:
        return []
    arr = np.asarray(points, dtype=float)
    hom = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=float)], axis=1)
    out = hom @ matrix.T
    return [(float(x), float(y)) for x, y in out[:, :2]]


def _points_from_element(tag: str, element: ET.Element) -> list[tuple[float, float]]:
    if tag in {"polygon", "polyline"}:
        raw = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", element.get("points", "")
        )
        coords = [float(token) for token in raw]
        return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
    if tag == "line":
        return [
            (_parse_float(element.get("x1")), _parse_float(element.get("y1"))),
            (_parse_float(element.get("x2")), _parse_float(element.get("y2"))),
        ]
    if tag == "rect":
        x = _parse_float(element.get("x"))
        y = _parse_float(element.get("y"))
        width = _parse_float(element.get("width"))
        height = _parse_float(element.get("height"))
        return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    if tag == "circle":
        cx = _parse_float(element.get("cx"))
        cy = _parse_float(element.get("cy"))
        radius = _parse_float(element.get("r"))
        return [
            (
                cx + math.cos((2.0 * math.pi * idx) / 32.0) * radius,
                cy + math.sin((2.0 * math.pi * idx) / 32.0) * radius,
            )
            for idx in range(32)
        ]
    if tag == "ellipse":
        cx = _parse_float(element.get("cx"))
        cy = _parse_float(element.get("cy"))
        rx = _parse_float(element.get("rx"))
        ry = _parse_float(element.get("ry"))
        return [
            (
                cx + math.cos((2.0 * math.pi * idx) / 32.0) * rx,
                cy + math.sin((2.0 * math.pi * idx) / 32.0) * ry,
            )
            for idx in range(32)
        ]
    if tag == "path":
        return flatten_svg_path(element.get("d", ""))
    return []


def _iter_paths(
    element: ET.Element,
    *,
    parent_transform: np.ndarray,
    parent_layer: str | None,
    items: list[ImportedPath],
) -> None:
    if _is_hidden(element):
        return
    tag = _strip_ns(element.tag)
    if tag in {
        "defs",
        "clippath",
        "mask",
        "pattern",
        "symbol",
        "marker",
        "metadata",
        "namedview",
        "foreignObject",
    }:
        return
    transform = parse_svg_transform(element.get("transform")) @ parent_transform
    layer_name = parent_layer
    if tag == "g":
        layer_name = (
            element.get("{http://www.inkscape.org/namespaces/inkscape}label")
            or element.get("id")
            or parent_layer
        )
    if tag in {"path", "polygon", "polyline", "line", "rect", "circle", "ellipse"}:
        points = _apply_transform(_points_from_element(tag, element), transform)
        if len(points) >= 2:
            kind = "polyline"
            if tag in {"polygon", "rect", "circle", "ellipse"}:
                kind = "polygon"
            elif tag == "path" and len(points) >= 3 and points[0] == points[-1]:
                kind = "polygon"
            items.append(
                ImportedPath(
                    id=element.get("id", f"shape_{len(items)}"),
                    label=element.get("id") or layer_name,
                    kind=kind,
                    points=points,
                    stroke=_resolve_color(element, "stroke"),
                    fill=_resolve_color(element, "fill"),
                    layer_name=layer_name,
                    transform=[float(v) for v in transform.reshape(-1)],
                    source_tag=tag,
                )
            )
    elif tag == "text":
        x = _parse_float(element.get("x"))
        y = _parse_float(element.get("y"))
        points = _apply_transform([(x, y)], transform)
        if points:
            items.append(
                ImportedPath(
                    id=element.get("id", f"shape_{len(items)}"),
                    label=element.get("id") or layer_name,
                    kind="text",
                    points=points,
                    text="".join(element.itertext()).strip() or None,
                    stroke=_resolve_color(element, "stroke"),
                    fill=_resolve_color(element, "fill"),
                    layer_name=layer_name,
                    transform=[float(v) for v in transform.reshape(-1)],
                    source_tag=tag,
                )
            )
    for child in list(element):
        _iter_paths(
            child,
            parent_transform=transform,
            parent_layer=layer_name,
            items=items,
        )


def import_svg_document(path: str | Path) -> ImportedVectorDocument:
    resolved = Path(path)
    root = ET.parse(resolved).getroot()
    return _import_svg_root(root, source_path=str(resolved))


def _import_svg_root(root: ET.Element, *, source_path: str) -> ImportedVectorDocument:
    view_box_raw = root.get("viewBox")
    view_box = None
    if view_box_raw:
        values = [
            float(token)
            for token in re.findall(
                r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", view_box_raw
            )
        ]
        if len(values) == 4:
            view_box = values
    items: list[ImportedPath] = []
    _iter_paths(
        root,
        parent_transform=np.eye(3, dtype=float),
        parent_layer=None,
        items=items,
    )
    return ImportedVectorDocument(
        source_path=str(source_path),
        source_kind="svg",
        width=_parse_float(root.get("width"), default=0.0) or None,
        height=_parse_float(root.get("height"), default=0.0) or None,
        view_box=view_box,
        shapes=items,
    )


def import_svg_string(
    text: str, *, source_path: str = "<memory>"
) -> ImportedVectorDocument:
    root = ET.fromstring(text)
    return _import_svg_root(root, source_path=source_path)


def import_svg_paths(path: str | Path) -> list[ImportedPath]:
    return import_svg_document(path).shapes
