"""
Modified from here:
https://github.com/ZQPei/deep_sort_pytorch/blob/master/utils/draw.py

"""

import logging
import math
import numpy as np
import cv2
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple
from annolid.utils.config import get_config

logger = logging.getLogger(__name__)

# High-contrast BGR palette for OpenCV skeleton rendering across multiple instances.
SKELETON_INSTANCE_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (255, 229, 0),  # cyan
    (0, 179, 255),  # amber
    (129, 64, 255),  # magenta
    (174, 240, 105),  # emerald
    (255, 136, 179),  # violet
    (64, 110, 255),  # coral
    (255, 196, 64),  # sky
    (88, 238, 255),  # yellow
)


def get_keypoint_connection_rules(keypoint_cfg_file=None):
    """Keypoint connection rules defined in the config file.

    Args:
        keypoint_cfg_file (str, optional): a yaml config file. Defaults to None.

    Returns:
        [(tuples),]: [(body_part_1,body_part2,(225,255,0))]
    """
    if keypoint_cfg_file is None:
        keypoint_cfg_file = Path(__file__).parent.parent / "configs" / "keypoints.yaml"

    keypoints_connection_rules = []
    head_rules = {}
    body_rules = {}
    name_rules = []
    event_rules = []
    zone_rules = []

    if keypoint_cfg_file.exists():
        try:
            loaded_rules = get_config(str(keypoint_cfg_file))
            if isinstance(loaded_rules, dict):
                head_rules = dict(loaded_rules.get("HEAD", {}))
                body_rules = dict(loaded_rules.get("BODY", {}))
                name_rules = list(loaded_rules.get("NAME", []))
                event_rules = list(loaded_rules.get("EVENTS", []))
                zone_rules = list(loaded_rules.get("ZONES", []))
        except Exception as exc:
            logger.debug(
                "Failed to load keypoint config from %s: %s",
                keypoint_cfg_file,
                exc,
            )
    # Fallback to empty collections when the file is missing or invalid.
    head_rules = head_rules or {}
    body_rules = body_rules or {}
    name_rules = name_rules or []
    event_rules = event_rules or []
    zone_rules = zone_rules or []

    # color is a placeholder for future customization
    for k, v in head_rules.items():
        v, r, g, b = v.split(",")
        color = (int(r), int(g), int(b))
        keypoints_connection_rules.append((k, v, color))
    for k, v in body_rules.items():
        v, r, g, b = v.split(",")
        color = (int(r), int(g), int(b))
        keypoints_connection_rules.append((k, v, color))
    return (keypoints_connection_rules, name_rules, event_rules, zone_rules)


def _keypoint_color():
    keypoint_connection_rules, _, _, _ = get_keypoint_connection_rules()
    keypoint_colors = {}
    for kcr in keypoint_connection_rules:
        keypoint_colors[kcr[0]] = kcr[-1]
    return keypoint_colors


KEYPOINT_COLORS = _keypoint_color()


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def get_label_color(label_id):
    if label_id in KEYPOINT_COLORS:
        return label_id, KEYPOINT_COLORS[label_id]
    else:
        try:
            _id = int(label_id) if label_id is not None else 0
            color = compute_color_for_labels(_id)
            label = "{}{:d}".format("", _id)
        except (TypeError, ValueError):
            if ":" in label_id:
                _label_id = label_id.split(":")[0]
            else:
                _label_id = label_id
            color = compute_color_for_labels(hash(_label_id) % 100)
            label = f"{label_id}"
        return label, color


def draw_binary_masks(img, masks, identities=None):
    img_alpha = np.zeros(img.shape, img.dtype)
    for i, _mask in enumerate(masks):
        label, color = get_label_color(identities[i])
        img_alpha[:, :] = color
        img_alpha = cv2.bitwise_and(img_alpha, img_alpha, mask=_mask)
        img = cv2.addWeighted(img_alpha, 0.4, img, 1, 0, img)
    return img


def draw_boxes(
    img, bbox, identities=None, offset=(0, 0), draw_track=False, points=None
):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        _id = identities[i]

        label, color = get_label_color(_id)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )

        if draw_track:
            if isinstance(_id, str):
                _id = hash(_id) % 100
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            points[_id].append(center)
            thickness = 2
            for j in range(len(points[_id])):
                if points[_id][j - 1] is None or points[_id] is None:
                    continue
                cv2.line(img, points[_id][j - 1], points[_id][j], color, thickness)

    return img


def draw_flow(img, flow, step=16, quiver=(0, 100, 0)):
    """
    Draw optical-flow quiver vectors onto a copy of `img`.

    Modified from:
    https://gist.github.com/RodolfoFerro/11d39fad57e21b5e85fe4d4a906cf098
    """
    step_i = max(1, int(step))
    h, w = img.shape[:2]
    vis = img.copy()
    if h <= 0 or w <= 0:
        return vis

    start = step_i / 2.0
    ys = np.arange(start, h, step_i, dtype=np.float64).astype(np.int32)
    xs = np.arange(start, w, step_i, dtype=np.float64).astype(np.int32)
    if ys.size == 0 or xs.size == 0:
        return vis

    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
    y = y_grid.ravel()
    x = x_grid.ravel()
    fx = flow[y, x, 0]
    fy = flow[y, x, 1]

    # Filter out zero and non-finite flow vectors so cv2.polylines never overflows.
    keep = np.isfinite(fx) & np.isfinite(fy) & ((fx != 0) | (fy != 0))
    if not np.any(keep):
        return vis

    x = x[keep]
    y = y[keep]
    fx = fx[keep]
    fy = fy[keep]

    x2 = np.clip(x + fx + 0.5, -1_000_000.0, 1_000_000.0).astype(np.int32)
    y2 = np.clip(y + fy + 0.5, -1_000_000.0, 1_000_000.0).astype(np.int32)
    lines = np.empty((x.size, 2, 2), dtype=np.int32)
    lines[:, 0, 0] = x
    lines[:, 0, 1] = y
    lines[:, 1, 0] = x2
    lines[:, 1, 1] = y2

    cv2.polylines(vis, lines, False, quiver, 3)

    # Stamp the exact 5-pixel footprint of cv2.circle(vis, (x1, y1), 1, color, -1)
    # in vectorized NumPy instead of thousands of per-point Python OpenCV calls.
    dot_color = (0, 255, 0) if vis.ndim == 3 else 255
    vis[y, x] = dot_color
    valid_l = x > 0
    if np.any(valid_l):
        vis[y[valid_l], x[valid_l] - 1] = dot_color
    valid_r = (x + 1) < w
    if np.any(valid_r):
        vis[y[valid_r], x[valid_r] + 1] = dot_color
    valid_u = y > 0
    if np.any(valid_u):
        vis[y[valid_u] - 1, x[valid_u]] = dot_color
    valid_d = (y + 1) < h
    if np.any(valid_d):
        vis[y[valid_d] + 1, x[valid_d]] = dot_color

    return vis


def _draw_dashed_line_cv2(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_px: float = 8.0,
    gap_px: float = 5.0,
) -> None:
    """Draw an antialiased dashed line segment on an OpenCV image."""
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1e-3:
        return
    ux = (x2 - x1) / dist
    uy = (y2 - y1) / dist
    stride = max(2.0, float(dash_px) + float(gap_px))
    pos = 0.0
    while pos < dist:
        end = min(dist, pos + float(dash_px))
        sx = int(round(x1 + ux * pos))
        sy = int(round(y1 + uy * pos))
        ex = int(round(x1 + ux * end))
        ey = int(round(y1 + uy * end))
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
        pos += stride


def draw_keypoint_connections(
    frame, keypoints, keypoint_connection_rules=None, max_dist=250
):
    """Draw antialiased skeleton lines between defined keypoints."""

    if keypoint_connection_rules is None:
        return frame

    for kp0, kp1, color in keypoint_connection_rules:
        if kp0 in keypoints and kp1 in keypoints:
            raw0 = keypoints[kp0]
            raw1 = keypoints[kp1]
            if len(raw0) < 2 or len(raw1) < 2:
                continue
            x0, y0 = float(raw0[0]), float(raw0[1])
            x1, y1 = float(raw1[0]), float(raw1[1])
            if not (
                math.isfinite(x0)
                and math.isfinite(y0)
                and math.isfinite(x1)
                and math.isfinite(y1)
            ):
                continue
            dist = math.hypot(x1 - x0, y1 - y0)
            if max_dist is None or dist < max_dist:
                p0 = (int(round(x0)), int(round(y0)))
                p1 = (int(round(x1)), int(round(y1)))
                cv2.line(frame, p0, p1, color, 3, lineType=cv2.LINE_AA)
            else:
                logger.debug(
                    "Skipping keypoint connection %s-%s: distance %.1f >= max_dist %s",
                    kp0,
                    kp1,
                    dist,
                    max_dist,
                )
    return frame


def draw_pose_skeleton(
    frame: np.ndarray,
    keypoints: Mapping[str, object],
    *,
    pose_schema: Optional[object] = None,
    edges: Optional[Sequence[Tuple[str, str]]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    point_radius: int = 4,
    draw_points: bool = True,
    draw_labels: bool = False,
    max_dist: Optional[float] = None,
) -> np.ndarray:
    """Draw a single- or multi-instance pose skeleton onto an OpenCV BGR image.

    Args:
        frame: HxWx3 uint8 OpenCV image (modified in-place and returned).
        keypoints: Either a single-instance mapping ``{kp_name: (x, y[, vis])}``
            or a multi-instance mapping ``{instance_name: {kp_name: (x, y[, vis])}}``.
        pose_schema: Optional ``PoseSchema`` providing ``edges`` (and base edge stripping).
        edges: Optional explicit ``[(kp_a, kp_b), ...]`` edge list (overrides ``pose_schema``).
        color: Optional fixed BGR tuple for single-instance rendering.
        thickness: Line thickness in pixels.
        point_radius: Keypoint circle radius in pixels.
        draw_points: Whether to draw keypoint circles on top of bones.
        draw_labels: Whether to draw keypoint names beside circles.
        max_dist: Optional maximum bone length in pixels; longer edges are skipped.
    """
    if not keypoints:
        return frame

    edge_pairs: list[Tuple[str, str]] = []
    if edges is not None:
        edge_pairs = [(str(a).strip(), str(b).strip()) for a, b in edges if a and b]
    elif pose_schema is not None:
        if hasattr(pose_schema, "base_edges"):
            edge_pairs = list(pose_schema.base_edges())
        else:
            edge_pairs = [
                (str(a).strip(), str(b).strip())
                for a, b in (getattr(pose_schema, "edges", None) or ())
                if a and b
            ]

    # Detect multi-instance vs single-instance mapping.
    first_val = next(iter(keypoints.values()))
    if isinstance(first_val, Mapping):
        instances: list[Tuple[str, Mapping[str, object]]] = [
            (str(inst_name), inst_kps)
            for inst_name, inst_kps in keypoints.items()
            if isinstance(inst_kps, Mapping)
        ]
    else:
        instances = [("instance_0", keypoints)]

    def _parse_kp(raw: object) -> Optional[Tuple[int, int, int]]:
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) < 2:
            return None
        x = float(raw[0])
        y = float(raw[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        vis = 2
        if len(raw) >= 3:
            try:
                vf = float(raw[2])
                if not math.isfinite(vf) or vf <= 0.0:
                    return None
                vis = 1 if int(round(vf)) == 1 else 2
            except (TypeError, ValueError):
                vis = 2
        return (int(round(x)), int(round(y)), vis)

    for inst_idx, (_inst_name, kp_map) in enumerate(instances):
        inst_color = (
            color
            if (color is not None and len(instances) == 1)
            else SKELETON_INSTANCE_COLORS_BGR[
                inst_idx % len(SKELETON_INSTANCE_COLORS_BGR)
            ]
        )
        parsed: dict[str, Tuple[int, int, int]] = {}
        for k, v in kp_map.items():
            pt = _parse_kp(v)
            if pt is not None:
                parsed[str(k).strip()] = pt

        # Draw bones with subtle dark shadow for readability on bright/dark backgrounds.
        for a, b in edge_pairs:
            pa = parsed.get(a)
            pb = parsed.get(b)
            if pa is None or pb is None:
                continue
            if (
                max_dist is not None
                and math.hypot(pb[0] - pa[0], pb[1] - pa[1]) >= max_dist
            ):
                continue
            occluded = (pa[2] == 1) or (pb[2] == 1)
            p0 = (pa[0], pa[1])
            p1 = (pb[0], pb[1])
            if occluded:
                _draw_dashed_line_cv2(
                    frame, p0, p1, (20, 20, 20), thickness=thickness + 2
                )
                _draw_dashed_line_cv2(frame, p0, p1, inst_color, thickness=thickness)
            else:
                cv2.line(
                    frame, p0, p1, (20, 20, 20), thickness + 2, lineType=cv2.LINE_AA
                )
                cv2.line(frame, p0, p1, inst_color, thickness, lineType=cv2.LINE_AA)

        if draw_points:
            for _kp_name, (px, py, vis) in parsed.items():
                if vis == 1:
                    cv2.circle(
                        frame,
                        (px, py),
                        point_radius + 1,
                        (20, 20, 20),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        (px, py),
                        point_radius,
                        inst_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    cv2.circle(
                        frame,
                        (px, py),
                        point_radius + 1,
                        (20, 20, 20),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        frame,
                        (px, py),
                        point_radius,
                        inst_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )

        if draw_labels and parsed:
            mean_x = sum(pt[0] for pt in parsed.values()) / float(len(parsed))
            for kp_name, (px, py, _vis) in parsed.items():
                (tw, _th), _ = cv2.getTextSize(
                    kp_name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
                )
                if px < mean_x - 5.0 and (px - point_radius - 4 - tw) >= 2:
                    tx = px - point_radius - 4 - tw
                else:
                    tx = px + point_radius + 4
                origin = (int(tx), int(py - 2))
                cv2.putText(
                    frame,
                    kp_name,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (18, 18, 18),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    kp_name,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    inst_color,
                    1,
                    cv2.LINE_AA,
                )
    return frame
