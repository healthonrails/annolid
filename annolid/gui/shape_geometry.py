"""Pure geometry utilities for canvas shapes, OBBs, 3-point circles, and navigation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple

Point2D = Tuple[float, float]


def point_xy(point: Any) -> Point2D:
    """Extract ``(x, y)`` floats from a Qt point or 2-element sequence."""
    if hasattr(point, "x") and hasattr(point, "y") and callable(point.x):
        return float(point.x()), float(point.y())
    return float(point[0]), float(point[1])


def oriented_rect_from_three_points(
    p0: Any,
    p1: Any,
    p2: Any,
) -> Optional[list[Point2D]]:
    """Build 4 ordered corners of an oriented rectangle from 3 clicks.

    ``p0 -> p1`` defines the primary edge and orientation; ``p2`` projects onto
    the normal to define the perpendicular span.
    """
    x0, y0 = point_xy(p0)
    x1, y1 = point_xy(p1)
    x2, y2 = point_xy(p2)
    ux = x1 - x0
    uy = y1 - y0
    edge_len = math.hypot(ux, uy)
    if edge_len <= 1e-6:
        return None
    nx = -uy / edge_len
    ny = ux / edge_len
    span = (x2 - x0) * nx + (y2 - y0) * ny
    dx = span * nx
    dy = span * ny
    return [
        (x0, y0),
        (x1, y1),
        (x1 + dx, y1 + dy),
        (x0 + dx, y0 + dy),
    ]


def move_oriented_rect_vertex(
    points: Sequence[Any],
    index: int,
    new_pos: Any,
) -> list[Point2D]:
    """Move corner ``index`` of a 4-point oriented rectangle while preserving 90° corners."""
    pts = [point_xy(p) for p in points]
    if len(pts) != 4:
        if 0 <= index < len(pts):
            pts[index] = point_xy(new_pos)
        return pts

    idx = int(index) % 4
    opp_idx = (idx + 2) % 4
    prev_idx = (idx - 1) % 4
    next_idx = (idx + 1) % 4

    ox, oy = pts[opp_idx]
    px, py = pts[prev_idx]
    qx, qy = pts[next_idx]
    tx, ty = point_xy(new_pos)

    ux, uy = px - ox, py - oy
    vx, vy = qx - ox, qy - oy
    u_len = math.hypot(ux, uy)
    v_len = math.hypot(vx, vy)

    if u_len > 1e-6 and v_len > 1e-6:
        eu = (ux / u_len, uy / u_len)
        # Re-orthogonalize v against u to prevent cumulative floating-point drift.
        ev_raw = (-eu[1], eu[0])
        if ev_raw[0] * vx + ev_raw[1] * vy < 0.0:
            ev = (-ev_raw[0], -ev_raw[1])
        else:
            ev = ev_raw
    elif u_len > 1e-6:
        eu = (ux / u_len, uy / u_len)
        ev = (-eu[1], eu[0])
    elif v_len > 1e-6:
        ev = (vx / v_len, vy / v_len)
        eu = (ev[1], -ev[0])
    else:
        eu = (1.0, 0.0)
        ev = (0.0, 1.0)

    dx = tx - ox
    dy = ty - oy
    proj_u = dx * eu[0] + dy * eu[1]
    proj_v = dx * ev[0] + dy * ev[1]

    out = list(pts)
    out[prev_idx] = (ox + proj_u * eu[0], oy + proj_u * eu[1])
    out[next_idx] = (ox + proj_v * ev[0], oy + proj_v * ev[1])
    out[idx] = (
        ox + proj_u * eu[0] + proj_v * ev[0],
        oy + proj_u * eu[1] + proj_v * ev[1],
    )
    return out


def rotate_oriented_rect_by_vertex(
    points: Sequence[Any],
    index: int,
    new_pos: Any,
) -> list[Point2D]:
    """Rotate a 4-point oriented rectangle around its center by dragging vertex ``index``."""
    pts = [point_xy(p) for p in points]
    if len(pts) != 4:
        return pts

    idx = int(index) % 4
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    vx, vy = pts[idx][0] - cx, pts[idx][1] - cy
    tx, ty = point_xy(new_pos)
    nx, ny = tx - cx, ty - cy
    if math.hypot(vx, vy) <= 1e-6 or math.hypot(nx, ny) <= 1e-6:
        return pts

    delta = math.atan2(ny, nx) - math.atan2(vy, vx)
    cos_a = math.cos(delta)
    sin_a = math.sin(delta)
    return [
        (
            cx + (x - cx) * cos_a - (y - cy) * sin_a,
            cy + (x - cx) * sin_a + (y - cy) * cos_a,
        )
        for x, y in pts
    ]


def circle_from_three_points(
    p1: Any,
    p2: Any,
    p3: Any,
) -> Optional[Tuple[Point2D, Point2D]]:
    """Compute ``(center, circumference_point)`` from 3 non-collinear points.

    Returns ``None`` if the points are collinear or coincident.
    """
    x1, y1 = point_xy(p1)
    x2, y2 = point_xy(p2)
    x3, y3 = point_xy(p3)
    denom = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(denom) <= 1e-6:
        return None
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / denom
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / denom
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    if math.hypot(x3 - cx, y3 - cy) <= 1e-6:
        return None
    return (cx, cy), (x3, y3)


def clamp_point_to_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
) -> Point2D:
    """Clamp ``(x, y)`` to ``[0, max(0, width - 1)] x [0, max(0, height - 1)]``."""
    max_x = max(0.0, float(width) - 1.0)
    max_y = max(0.0, float(height) - 1.0)
    return (
        max(0.0, min(max_x, float(x))),
        max(0.0, min(max_y, float(y))),
    )


def intersect_segment_with_bounds(
    p1: Any,
    p2: Any,
    width: float,
    height: float,
) -> Point2D:
    """Project segment ``p1 -> p2`` onto the image rectangle boundary safely.

    Always returns a finite point within ``[0, max_x] x [0, max_y]`` without
    raising on coincident endpoints, degenerate 1x1 images, or floating-point edge
    misses.
    """
    max_x = max(0.0, float(width) - 1.0)
    max_y = max(0.0, float(height) - 1.0)
    x1, y1 = clamp_point_to_bounds(*point_xy(p1), width=width, height=height)
    x2, y2 = point_xy(p2)
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return (x1, y1)

    t_candidates: list[float] = []
    if dx < 0.0:
        t_candidates.append((0.0 - x1) / dx)
    elif dx > 0.0:
        t_candidates.append((max_x - x1) / dx)
    if dy < 0.0:
        t_candidates.append((0.0 - y1) / dy)
    elif dy > 0.0:
        t_candidates.append((max_y - y1) / dy)

    valid_t = [t for t in t_candidates if 0.0 <= t <= 1.0]
    if not valid_t:
        return clamp_point_to_bounds(x2, y2, width=width, height=height)
    t_hit = min(valid_t)
    return clamp_point_to_bounds(
        x1 + t_hit * dx,
        y1 + t_hit * dy,
        width=width,
        height=height,
    )


def shape_bounds(shape: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(min_x, min_y, max_x, max_y)`` for a ``Shape`` or ``None``."""
    if shape is None:
        return None
    shape_type = str(getattr(shape, "shape_type", "polygon") or "polygon").lower()
    points = getattr(shape, "points", None) or []
    mask = getattr(shape, "mask", None)

    if shape_type == "mask" and mask is not None and points:
        x0, y0 = point_xy(points[0])
        h, w = mask.shape[:2]
        if len(points) >= 2:
            x1, y1 = point_xy(points[1])
            return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        return (x0, y0, x0 + float(w), y0 + float(h))

    if not points:
        return None

    coords = [point_xy(p) for p in points]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in coords):
        return None

    if shape_type == "circle" and len(coords) >= 2:
        (cx, cy), (ex, ey) = coords[0], coords[1]
        radius = math.hypot(ex - cx, ey - cy)
        return (cx - radius, cy - radius, cx + radius, cy + radius)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def shapes_bounding_box(
    shapes: Iterable[Any],
) -> Optional[Tuple[float, float, float, float]]:
    """Return combined ``(min_x, min_y, max_x, max_y)`` across ``shapes``."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    for shape in shapes:
        bounds = shape_bounds(shape)
        if bounds is None:
            continue
        bx0, by0, bx1, by1 = bounds
        min_x = min(min_x, bx0)
        min_y = min(min_y, by0)
        max_x = max(max_x, bx1)
        max_y = max(max_y, by1)
        found = True
    if not found:
        return None
    return (min_x, min_y, max_x, max_y)


def find_neighbor_shape(
    shapes: Sequence[Any],
    current_shape: Any,
    direction: str,
    *,
    is_visible: Optional[Any] = None,
) -> Any:
    """Find the best visible neighboring shape in ``direction`` (left/right/up/down).

    Uses bounding-box centers and favors candidates closely aligned with the
    requested cardinal direction (within a 60° half-angle cone before falling back
    to the wider half-plane).
    """
    dir_vectors = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
    }
    vec = dir_vectors.get(str(direction or "").lower())
    if vec is None:
        return None
    dx_dir, dy_dir = vec

    visible_shapes: list[tuple[Any, float, float]] = []
    for candidate in shapes:
        if candidate is None:
            continue
        if is_visible is not None:
            if not is_visible(candidate):
                continue
        elif not bool(getattr(candidate, "visible", True)):
            continue
        bounds = shape_bounds(candidate)
        if bounds is None:
            continue
        cx = 0.5 * (bounds[0] + bounds[2])
        cy = 0.5 * (bounds[1] + bounds[3])
        visible_shapes.append((candidate, cx, cy))

    if not visible_shapes:
        return None

    if current_shape is None:
        return visible_shapes[0][0]

    cur_bounds = shape_bounds(current_shape)
    if cur_bounds is None:
        return visible_shapes[0][0]
    cur_cx = 0.5 * (cur_bounds[0] + cur_bounds[2])
    cur_cy = 0.5 * (cur_bounds[1] + cur_bounds[3])

    cone_candidates: list[tuple[float, float, Any]] = []
    halfplane_candidates: list[tuple[float, float, Any]] = []
    min_progress = 1e-3

    for candidate, cx, cy in visible_shapes:
        if candidate is current_shape:
            continue
        vx = cx - cur_cx
        vy = cy - cur_cy
        along = vx * dx_dir + vy * dy_dir
        if along <= min_progress:
            continue
        perp = abs(vx * (-dy_dir) + vy * dx_dir)
        dist = math.hypot(vx, vy)
        score = along + 2.5 * perp
        # 60° cone: perp <= sqrt(3) * along
        if perp <= 1.7320508 * along:
            cone_candidates.append((score, dist, candidate))
        else:
            halfplane_candidates.append((score, dist, candidate))

    pool = cone_candidates if cone_candidates else halfplane_candidates
    if not pool:
        return None
    pool.sort(key=lambda item: (item[0], item[1]))
    return pool[0][2]
