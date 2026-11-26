from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import math
import shutil
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urlencode

import cv2
import numpy as np
import yaml

from .flybody_render_worker import (
    FLY_PART_ID_TO_NAME,
    render_flybody_scenarios,
)
from .models import (
    FLY_KEYPOINT_NAMES,
    MOUSE_KEYPOINT_NAMES,
    RAT_KEYPOINT_NAMES,
    SimDatasetArtifact,
    SimDatasetConfig,
    VideoEnvironmentProfile,
)
from .photometry import (
    apply_environment_photometry,
    composite_foreground_onto_background,
    load_reference_background,
)

LogCallback = Optional[Callable[[str], None]]

REPO_ROOT = Path(__file__).resolve().parents[2]
THREEJS_ROOT = REPO_ROOT / "annolid" / "gui" / "assets" / "threejs"
TWO_MICE_SCENE_REL = "annolid/gui/assets/threejs/two_mice.html"
RAT_NCAP_SCENE_REL = "annolid/gui/assets/threejs/rat_ncap.html"

MOUSE_COARSE_ID_TO_NAME: dict[int, str] = {
    1: "head",
    2: "torso",
    3: "limbs",
    4: "tail",
}

MOUSE_FLIP_IDX: list[int] = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 10, 11, 12]
FLY_FLIP_IDX: list[int] = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9]
RAT_FLIP_IDX: list[int] = [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 15, 14]


def _emit(cb: LogCallback, msg: str) -> None:
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass


def _decode_segmentation_rle(
    seg_payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Three.js WebGL 3D ID-pass 4-tuple RLE ([count, track, fine, coarse]) maps."""
    width = int(seg_payload["width"])
    height = int(seg_payload["height"])
    raw_rle = seg_payload.get("rle") or []
    if not raw_rle:
        empty = np.zeros((height, width), dtype=np.uint8)
        return empty, empty.copy(), empty.copy()

    rle = np.asarray(raw_rle, dtype=np.int32).reshape(-1, 4)
    counts = rle[:, 0]
    colors = rle[:, 1:4].astype(np.uint8)
    flat_rgb = np.repeat(colors, counts, axis=0)
    if flat_rgb.shape[0] != width * height:
        raise ValueError(
            f"Decoded Three.js RLE pixel count {flat_rgb.shape[0]} != {width * height}"
        )
    id_img = flat_rgb.reshape(height, width, 3)
    track_map = id_img[:, :, 0].copy()
    fine_map = id_img[:, :, 1].copy()
    coarse_map = id_img[:, :, 2].copy()

    # Seal 1-2px guard-hair micro-gaps inside each mouse silhouette without merging neighbors.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_track = track_map.copy()
    for tid in [int(t) for t in np.unique(track_map) if int(t) > 0]:
        raw_m = track_map == tid
        other_m = (track_map > 0) & (track_map != tid)
        closed_m = (
            cv2.morphologyEx(
                raw_m.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        ) & (~other_m)
        added = closed_m & (~raw_m) & (closed_track == 0)
        if np.any(added) and np.any(raw_m):
            inv = (~raw_m).astype(np.uint8)
            _dist, nearest_idx = cv2.distanceTransformWithLabels(
                inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
            )
            src_ys, src_xs = np.nonzero(inv == 0)
            lbl_vals = np.clip(nearest_idx[added] - 1, 0, max(0, src_ys.size - 1))
            closed_track[added] = tid
            fine_map[added] = fine_map[src_ys[lbl_vals], src_xs[lbl_vals]]
            coarse_map[added] = coarse_map[src_ys[lbl_vals], src_xs[lbl_vals]]

    return closed_track, fine_map, coarse_map


def _mask_to_polygons(
    binary_mask: np.ndarray,
    *,
    min_area: float = 14.0,
    max_polygons: int = 3,
    epsilon_frac: float = 0.004,
) -> list[list[list[float]]]:
    """Convert a binary uint8/bool mask into simplified LabelMe/YOLO polygon point lists."""
    mask_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(mask_u8)) < int(min_area):
        return []
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    scored = sorted(
        ((cv2.contourArea(c), c) for c in contours),
        key=lambda item: item[0],
        reverse=True,
    )
    polys: list[list[list[float]]] = []
    for area, contour in scored:
        if area < min_area or len(polys) >= max_polygons:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.8, epsilon_frac * peri), True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        polys.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
    return polys


def _bbox_from_polygons(
    polys: Sequence[Sequence[Sequence[float]]], width: int, height: int
) -> tuple[float, float, float, float] | None:
    xs = [float(pt[0]) for poly in polys for pt in poly]
    ys = [float(pt[1]) for poly in polys for pt in poly]
    if not xs or not ys:
        return None
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(float(width), max(xs))
    y1 = min(float(height), max(ys))
    if x1 <= x0 + 1.0 or y1 <= y0 + 1.0:
        return None
    return x0, y0, x1, y1


def _build_labelme_json(
    *,
    image_filename: str,
    width: int,
    height: int,
    instances: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build an Annolid/LabelMe-compatible annotation payload from structured instances."""
    shapes: list[dict[str, Any]] = []
    for inst in instances:
        group_id = int(inst["group_id"])
        inst_label = str(inst["label"])
        for poly in inst.get("polygons") or []:
            shapes.append(
                {
                    "label": inst_label,
                    "points": poly,
                    "group_id": group_id,
                    "shape_type": "polygon",
                    "flags": {},
                }
            )
        for part_name, part_polys in (inst.get("parts") or {}).items():
            for poly in part_polys:
                shapes.append(
                    {
                        "label": f"{inst_label}_{part_name}",
                        "points": poly,
                        "group_id": group_id,
                        "shape_type": "polygon",
                        "flags": {
                            "body_part": True,
                            "part_name": str(part_name),
                            "parent_instance": inst_label,
                        },
                    }
                )
        for kp in inst.get("keypoints") or []:
            vis = int(kp.get("v", 2))
            if vis <= 0:
                continue
            kp_name = str(kp["name"])
            shapes.append(
                {
                    "label": f"{inst_label}_{kp_name}",
                    "display_label": kp_name,
                    "instance_label": inst_label,
                    "kp_visibility": vis,
                    "points": [[float(kp["x"]), float(kp["y"])]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {
                        "kp_visible": vis >= 2,
                        "kp_occluded": vis == 1,
                    },
                }
            )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def _export_yolo_dataset(
    records: Sequence[dict[str, Any]],
    *,
    yolo_dir: Path,
    task: str,
    export_level: str,
    species: str,
    val_fraction: float,
    seed: int,
) -> tuple[Path, list[str], int, int]:
    """Export portable Ultralytics YOLO seg/pose/detect dataset with relative data.yaml."""
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    task_norm = str(task or "yolo_seg").strip().lower()
    level_norm = str(export_level or "instances").strip().lower()
    species_cls = (
        "fly" if species == "fly" else ("rat" if species == "rat" else "mouse")
    )
    keypoint_names = (
        list(FLY_KEYPOINT_NAMES)
        if species == "fly"
        else (
            list(RAT_KEYPOINT_NAMES) if species == "rat" else list(MOUSE_KEYPOINT_NAMES)
        )
    )
    flip_idx = (
        list(FLY_FLIP_IDX)
        if species == "fly"
        else (list(RAT_FLIP_IDX) if species == "rat" else list(MOUSE_FLIP_IDX))
    )

    # Collect deterministic class list.
    class_set: list[str] = []

    def _register_class(name: str) -> int:
        if name not in class_set:
            class_set.append(name)
        return class_set.index(name)

    if task_norm == "yolo_pose":
        _register_class(species_cls)
    elif level_norm == "parts":
        part_order = (
            list(FLY_PART_ID_TO_NAME.values())
            if species == "fly"
            else list(MOUSE_COARSE_ID_TO_NAME.values())
        )
        for p_name in part_order:
            _register_class(p_name)
    else:
        _register_class(species_cls)

    # Deterministic group-aware train/val split so oversampled copies of the
    # same source frame can never leak across both train and val splits.
    group_order: list[str] = []
    group_to_indices: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        gkey = str(rec.get("source_group") or rec.get("image_path") or f"idx_{idx}")
        if gkey not in group_to_indices:
            group_order.append(gkey)
            group_to_indices[gkey] = []
        group_to_indices[gkey].append(idx)

    n_groups = len(group_order)
    rng = np.random.default_rng(int(seed))
    perm = np.arange(n_groups)
    rng.shuffle(perm)
    n_val_groups = (
        max(1, int(round(n_groups * float(np.clip(val_fraction, 0.05, 0.5)))))
        if n_groups >= 2
        else 0
    )
    val_groups = {group_order[i] for i in perm[:n_val_groups].tolist()}
    val_set: set[int] = set()
    # Keep only a single canonical copy of each validation group (never oversample val).
    skipped_val_duplicates: set[int] = set()
    for gkey in val_groups:
        g_indices = group_to_indices[gkey]
        val_set.add(g_indices[0])
        skipped_val_duplicates.update(g_indices[1:])

    train_count = 0
    val_count = 0

    for idx, rec in enumerate(records):
        if idx in skipped_val_duplicates:
            continue
        split = "val" if idx in val_set else "train"
        if split == "val":
            val_count += 1
        else:
            train_count += 1

        src_img = Path(rec["image_path"])
        w = int(rec["width"])
        h = int(rec["height"])
        stem = str(rec["stem"])
        dst_img = yolo_dir / "images" / split / f"{stem}{src_img.suffix}"
        shutil.copy2(src_img, dst_img)

        label_lines: list[str] = []
        for inst in rec.get("instances") or []:
            if task_norm == "yolo_pose":
                cls_name = species_cls
                cls_id = _register_class(cls_name)
                bbox = _bbox_from_polygons(inst.get("polygons") or [], w, h)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                cx = ((x0 + x1) * 0.5) / float(w)
                cy = ((y0 + y1) * 0.5) / float(h)
                bw = (x1 - x0) / float(w)
                bh = (y1 - y0) / float(h)
                kp_by_name = {str(k["name"]): k for k in (inst.get("keypoints") or [])}
                kp_tokens: list[str] = []
                for kp_name in keypoint_names:
                    kp = kp_by_name.get(kp_name)
                    if kp is None or int(kp.get("v", 0)) <= 0:
                        kp_tokens.extend(["0.000000", "0.000000", "0"])
                    else:
                        kx = float(np.clip(float(kp["x"]) / float(w), 0.0, 1.0))
                        ky = float(np.clip(float(kp["y"]) / float(h), 0.0, 1.0))
                        kv = int(np.clip(int(kp.get("v", 2)), 0, 2))
                        kp_tokens.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
                label_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                    + " ".join(kp_tokens)
                )
            elif task_norm == "yolo_detect":
                if level_norm == "parts":
                    for part_name, part_polys in (inst.get("parts") or {}).items():
                        bbox = _bbox_from_polygons(part_polys, w, h)
                        if bbox is None:
                            continue
                        cls_id = _register_class(str(part_name))
                        x0, y0, x1, y1 = bbox
                        label_lines.append(
                            f"{cls_id} {((x0 + x1) * 0.5) / w:.6f} {((y0 + y1) * 0.5) / h:.6f} "
                            f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                        )
                else:
                    bbox = _bbox_from_polygons(inst.get("polygons") or [], w, h)
                    if bbox is not None:
                        cls_name = str(inst.get("class_name") or species_cls)
                        cls_id = _register_class(cls_name)
                        x0, y0, x1, y1 = bbox
                        label_lines.append(
                            f"{cls_id} {((x0 + x1) * 0.5) / w:.6f} {((y0 + y1) * 0.5) / h:.6f} "
                            f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                        )
            else:
                # Default: yolo_seg (polygon segmentation)
                if level_norm == "parts":
                    for part_name, part_polys in (inst.get("parts") or {}).items():
                        cls_id = _register_class(str(part_name))
                        for poly in part_polys:
                            if len(poly) < 3:
                                continue
                            coords = " ".join(
                                f"{float(np.clip(pt[0] / w, 0.0, 1.0)):.6f} "
                                f"{float(np.clip(pt[1] / h, 0.0, 1.0)):.6f}"
                                for pt in poly
                            )
                            label_lines.append(f"{cls_id} {coords}")
                else:
                    cls_name = str(inst.get("class_name") or species_cls)
                    cls_id = _register_class(cls_name)
                    for poly in inst.get("polygons") or []:
                        if len(poly) < 3:
                            continue
                        coords = " ".join(
                            f"{float(np.clip(pt[0] / w, 0.0, 1.0)):.6f} "
                            f"{float(np.clip(pt[1] / h, 0.0, 1.0)):.6f}"
                            for pt in poly
                        )
                        label_lines.append(f"{cls_id} {coords}")

        lbl_path = yolo_dir / "labels" / split / f"{stem}.txt"
        lbl_path.write_text(
            ("\n".join(label_lines) + "\n") if label_lines else "",
            encoding="utf-8",
        )

    data_yaml_payload: dict[str, Any] = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_set)},
    }
    if task_norm == "yolo_pose":
        data_yaml_payload["kpt_shape"] = [len(keypoint_names), 3]
        data_yaml_payload["flip_idx"] = flip_idx

    data_yaml_path = yolo_dir / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(data_yaml_payload, sort_keys=False), encoding="utf-8"
    )
    return data_yaml_path, class_set, train_count, val_count


def _write_preview_assets(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    fps: int = 8,
) -> tuple[list[str], str]:
    """Write up to 4 annotated preview PNGs and a short preview MP4 video."""
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return [], ""

    palette = [
        (72, 180, 255),
        (90, 225, 130),
        (255, 160, 80),
        (225, 110, 255),
        (255, 220, 90),
    ]
    sample_indices = set(
        np.linspace(0, len(records) - 1, num=min(4, len(records)), dtype=int).tolist()
    )
    preview_imgs: list[str] = []
    video_path = preview_dir / "synthetic_preview.mp4"
    first_w = int(records[0]["width"])
    first_h = int(records[0]["height"])
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (first_w, first_h),
    )
    try:
        for idx, rec in enumerate(records):
            bgr = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            overlay = bgr.copy()
            for inst_idx, inst in enumerate(rec.get("instances") or []):
                color = palette[inst_idx % len(palette)]
                for poly in inst.get("polygons") or []:
                    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(bgr, [pts], True, color, 2, cv2.LINE_AA)
                for kp in inst.get("keypoints") or []:
                    if int(kp.get("v", 0)) >= 1:
                        cv2.circle(
                            bgr,
                            (int(round(kp["x"])), int(round(kp["y"]))),
                            3,
                            (255, 255, 255),
                            -1,
                            cv2.LINE_AA,
                        )
            blended = cv2.addWeighted(overlay, 0.28, bgr, 0.72, 0.0)
            cv2.putText(
                blended,
                f"Sim {idx + 1}/{len(records)} | {rec.get('scenario', '')}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if blended.shape[1] != first_w or blended.shape[0] != first_h:
                blended = cv2.resize(blended, (first_w, first_h))
            writer.write(blended)
            if idx in sample_indices:
                p_out = preview_dir / f"preview_{len(preview_imgs) + 1:02d}.png"
                cv2.imwrite(str(p_out), blended)
                preview_imgs.append(str(p_out))
    finally:
        writer.release()

    return preview_imgs, (str(video_path) if video_path.is_file() else "")


class _QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


class _ThreadingTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_static_server(
    root_dir: Path,
) -> tuple[_ThreadingTCPServer, str, threading.Thread]:
    handler = functools.partial(_QuietHTTPRequestHandler, directory=str(root_dir))
    server = _ThreadingTCPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}", thread


def _render_procedural_mouse_frames(
    config: SimDatasetConfig,
    output_dir: Path,
    profile: Optional[VideoEnvironmentProfile],
    log_callback: LogCallback,
) -> list[dict[str, Any]]:
    """Explicit opt-in fallback mouse renderer when Playwright Chromium is unavailable."""
    _emit(
        log_callback,
        "[WARN] Using explicit procedural_mouse_fallback renderer (Playwright disabled/unavailable).",
    )
    img_dir = output_dir / "labelme"
    img_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(config.width), int(config.height)
    rng = np.random.default_rng(int(config.seed))
    ref_bg = load_reference_background(profile)
    records: list[dict[str, Any]] = []

    n_subjects = 2 if (profile is None or profile.estimated_subject_count >= 2) else 1
    coat_colors = {
        "c57_pair": [(35, 35, 38), (44, 42, 46)],
        "balbc_pair": [(225, 222, 220), (212, 208, 206)],
        "cba_pair": [(70, 95, 130), (62, 86, 118)],
        "nude_pair": [(165, 175, 215), (155, 166, 208)],
    }.get(config.coat_pair, [(36, 36, 40), (222, 220, 218)])

    for s_idx, scenario in enumerate(config.scenarios):
        for f_idx in range(int(config.frames_per_scenario)):
            t = float(f_idx) / float(max(1, config.frames_per_scenario))
            if ref_bg is not None and config.composite_on_real_background:
                canvas = cv2.resize(ref_bg, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                bg_val = 48 if config.lighting == "ir_night" else 175
                canvas = np.full((h, w, 3), bg_val, dtype=np.uint8)
                if config.arena == "circular_open_field":
                    cv2.circle(
                        canvas,
                        (w // 2, h // 2),
                        int(min(w, h) * 0.44),
                        (bg_val - 28, bg_val - 28, bg_val - 28),
                        4,
                    )
                else:
                    cv2.rectangle(
                        canvas,
                        (int(w * 0.07), int(h * 0.07)),
                        (int(w * 0.93), int(h * 0.93)),
                        (bg_val - 28, bg_val - 28, bg_val - 28),
                        3,
                    )

            instances: list[dict[str, Any]] = []
            for m_idx in range(n_subjects):
                phase = t * 2.0 * math.pi + m_idx * 2.2 + s_idx * 0.8
                if scenario == "social_huddle":
                    cx = w * (0.48 + 0.07 * (m_idx * 2 - 1) * math.cos(phase * 0.4))
                    cy = h * (0.50 + 0.06 * math.sin(phase * 0.5))
                else:
                    cx = w * (0.35 + 0.30 * m_idx + 0.14 * math.sin(phase))
                    cy = h * (0.42 + 0.18 * math.cos(phase * 0.7 + m_idx))
                angle = float(phase + rng.uniform(-0.25, 0.25))
                body_len = min(w, h) * 0.11
                body_rad = body_len * 0.42
                ca, sa = math.cos(angle), math.sin(angle)

                color = coat_colors[m_idx % len(coat_colors)]
                inst_mask = np.zeros((h, w), dtype=np.uint8)
                part_masks: dict[str, np.ndarray] = {
                    name: np.zeros((h, w), dtype=np.uint8)
                    for name in MOUSE_COARSE_ID_TO_NAME.values()
                }

                # Torso ellipse
                torso_c = (int(round(cx)), int(round(cy)))
                cv2.ellipse(
                    canvas,
                    torso_c,
                    (int(body_len * 0.75), int(body_rad)),
                    math.degrees(angle),
                    0,
                    360,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    part_masks["torso"],
                    torso_c,
                    (int(body_len * 0.75), int(body_rad)),
                    math.degrees(angle),
                    0,
                    360,
                    255,
                    -1,
                )

                # Head ellipse
                head_x = cx + ca * body_len * 0.85
                head_y = cy + sa * body_len * 0.85
                head_c = (int(round(head_x)), int(round(head_y)))
                cv2.ellipse(
                    canvas,
                    head_c,
                    (int(body_len * 0.38), int(body_rad * 0.75)),
                    math.degrees(angle),
                    0,
                    360,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    part_masks["head"],
                    head_c,
                    (int(body_len * 0.38), int(body_rad * 0.75)),
                    math.degrees(angle),
                    0,
                    360,
                    255,
                    -1,
                )

                # Tail curve
                tb_x = cx - ca * body_len * 0.75
                tb_y = cy - sa * body_len * 0.75
                tm_x = (
                    tb_x - ca * body_len * 0.65 - sa * body_len * 0.18 * math.sin(phase)
                )
                tm_y = (
                    tb_y - sa * body_len * 0.65 + ca * body_len * 0.18 * math.sin(phase)
                )
                tt_x = tm_x - ca * body_len * 0.65
                tt_y = tm_y - sa * body_len * 0.65
                tail_pts = np.asarray(
                    [[tb_x, tb_y], [tm_x, tm_y], [tt_x, tt_y]], dtype=np.int32
                )
                tail_color = (165, 145, 175)
                cv2.polylines(canvas, [tail_pts], False, tail_color, 4, cv2.LINE_AA)
                cv2.polylines(part_masks["tail"], [tail_pts], False, 255, 5)

                # Fore/hind limb paws
                for side in (-1.0, 1.0):
                    for along in (0.35, -0.35):
                        lx = cx + ca * body_len * along - sa * body_rad * 1.05 * side
                        ly = cy + sa * body_len * along + ca * body_rad * 1.05 * side
                        cv2.circle(
                            canvas,
                            (int(round(lx)), int(round(ly))),
                            max(3, int(body_rad * 0.30)),
                            tail_color,
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            part_masks["limbs"],
                            (int(round(lx)), int(round(ly))),
                            max(3, int(body_rad * 0.30)),
                            255,
                            -1,
                        )

                for pm in part_masks.values():
                    inst_mask |= pm

                nose_x = head_x + ca * body_len * 0.38
                nose_y = head_y + sa * body_len * 0.38
                neck_x = cx + ca * body_len * 0.50
                neck_y = cy + sa * body_len * 0.50
                kps = [
                    {"name": "nose", "x": nose_x, "y": nose_y, "v": 2},
                    {
                        "name": "left_ear",
                        "x": head_x - sa * body_rad * 0.55,
                        "y": head_y + ca * body_rad * 0.55,
                        "v": 2,
                    },
                    {
                        "name": "right_ear",
                        "x": head_x + sa * body_rad * 0.55,
                        "y": head_y - ca * body_rad * 0.55,
                        "v": 2,
                    },
                    {"name": "neck", "x": neck_x, "y": neck_y, "v": 2},
                    {
                        "name": "left_shoulder",
                        "x": cx + ca * body_len * 0.3 - sa * body_rad * 0.7,
                        "y": cy + sa * body_len * 0.3 + ca * body_rad * 0.7,
                        "v": 2,
                    },
                    {
                        "name": "right_shoulder",
                        "x": cx + ca * body_len * 0.3 + sa * body_rad * 0.7,
                        "y": cy + sa * body_len * 0.3 - ca * body_rad * 0.7,
                        "v": 2,
                    },
                    {"name": "mid_back", "x": cx, "y": cy, "v": 2},
                    {
                        "name": "left_hip",
                        "x": cx - ca * body_len * 0.35 - sa * body_rad * 0.7,
                        "y": cy - sa * body_len * 0.35 + ca * body_rad * 0.7,
                        "v": 2,
                    },
                    {
                        "name": "right_hip",
                        "x": cx - ca * body_len * 0.35 + sa * body_rad * 0.7,
                        "y": cy - sa * body_len * 0.35 - ca * body_rad * 0.7,
                        "v": 2,
                    },
                    {"name": "tail_base", "x": tb_x, "y": tb_y, "v": 2},
                    {"name": "tail_mid", "x": tm_x, "y": tm_y, "v": 2},
                    {"name": "tail_tip", "x": tt_x, "y": tt_y, "v": 2},
                    {"name": "center", "x": cx, "y": cy, "v": 2},
                ]
                for kp in kps:
                    kp["x"] = round(float(np.clip(kp["x"], 1.0, w - 2.0)), 2)
                    kp["y"] = round(float(np.clip(kp["y"], 1.0, h - 2.0)), 2)

                instances.append(
                    {
                        "group_id": m_idx + 1,
                        "label": f"mouse_{m_idx + 1}",
                        "class_name": "mouse",
                        "polygons": _mask_to_polygons(inst_mask),
                        "parts": {
                            pname: _mask_to_polygons(pm)
                            for pname, pm in part_masks.items()
                            if np.any(pm)
                        },
                        "keypoints": kps,
                    }
                )

            if profile is not None and config.match_example_photometry:
                canvas = apply_environment_photometry(
                    canvas,
                    profile,
                    strength=config.photometry_strength,
                    domain_randomization=config.domain_randomization,
                    rng=rng,
                )

            stem = f"mouse_{scenario}_f{f_idx:04d}"
            img_path = img_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lm = _build_labelme_json(
                image_filename=img_path.name,
                width=w,
                height=h,
                instances=instances,
            )
            (img_dir / f"{stem}.json").write_text(
                json.dumps(lm, indent=2) + "\n", encoding="utf-8"
            )
            records.append(
                {
                    "stem": stem,
                    "scenario": scenario,
                    "image_path": str(img_path),
                    "width": w,
                    "height": h,
                    "instances": instances,
                }
            )
    return records


def generate_mouse_3d_dataset(
    config: SimDatasetConfig,
    output_dir: str | Path,
    *,
    profile: Optional[VideoEnvironmentProfile] = None,
    log_callback: LogCallback = None,
) -> SimDatasetArtifact:
    """Generate environment-matched Mouse 3D synthetic dataset via bundled Three.js scene."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labelme_dir = out_dir / "labelme"
    labelme_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(config.width), int(config.height)
    scenarios = list(config.scenarios or ["arena_explore", "social_huddle"])
    notes: list[str] = []
    renderer = "threejs_two_mice_3d"
    records: list[dict[str, Any]] = []

    playwright_ok = importlib.util.find_spec("playwright") is not None
    scene_file = REPO_ROOT / TWO_MICE_SCENE_REL

    if not playwright_ok or not scene_file.is_file():
        if not config.allow_fallback:
            raise RuntimeError(
                "Playwright or bundled Two-Mice 3D scene (`annolid/gui/assets/threejs/two_mice.html`) "
                "is unavailable. Install Playwright Chromium (`playwright install chromium`) or pass "
                "`allow_fallback=True` to opt into the procedural fallback renderer."
            )
        renderer = "procedural_mouse_fallback"
        notes.append(
            "Fallback renderer used because Playwright Chromium was unavailable (`allow_fallback=True`)."
        )
        records = _render_procedural_mouse_frames(
            config, out_dir, profile, log_callback
        )
    else:
        try:
            from playwright.sync_api import sync_playwright

            server, base_url, _thread = _start_static_server(REPO_ROOT)
            ref_bg = load_reference_background(profile)
            rng = np.random.default_rng(int(config.seed))
            n_subjects = (
                1
                if (profile is not None and profile.estimated_subject_count == 1)
                else 2
            )
            _emit(
                log_callback,
                f"Launching Mouse 3D Three.js renderer ({len(scenarios)} scenario(s), "
                f"{config.frames_per_scenario} frames each, arena={config.arena}, "
                f"camera={config.camera_view}, lighting={config.lighting}, coat={config.coat_pair})...",
            )
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            "--use-gl=angle",
                            "--use-angle=swiftshader",
                            "--enable-unsafe-swiftshader",
                            "--enable-webgl",
                            "--ignore-gpu-blocklist",
                        ],
                    )
                    page = browser.new_page(viewport={"width": w, "height": h})
                    try:
                        for s_idx, scenario in enumerate(scenarios):
                            query_dict: dict[str, Any] = {
                                "dataset": 1,
                                "frame": 0,
                                "fps": int(config.fps),
                                "seed": int(config.seed) + s_idx * 97,
                                "camera_view": config.camera_view,
                                "scenario": scenario,
                                "arena": config.arena,
                                "subjects": n_subjects,
                            }
                            if config.lighting and config.lighting != "default":
                                query_dict["lighting"] = config.lighting
                            if config.coat_pair and config.coat_pair != "default":
                                query_dict["coat_pair"] = config.coat_pair
                            if config.occluder and config.occluder != "none":
                                query_dict["occluder"] = config.occluder
                            if config.headstage and config.headstage != "none":
                                query_dict["headstage"] = config.headstage
                                query_dict["headstage_style"] = config.headstage_style

                            url = f"{base_url}/{TWO_MICE_SCENE_REL}?{urlencode(query_dict)}"
                            _emit(log_callback, f"Rendering scenario '{scenario}'...")
                            page.goto(url, wait_until="load", timeout=120_000)
                            page.wait_for_selector(
                                "#c[data-dataset-ready='true']", timeout=120_000
                            )

                            for f_idx in range(int(config.frames_per_scenario)):
                                payload = page.evaluate(
                                    "idx => window.annolidPoseDataset.setFrame(idx)",
                                    f_idx,
                                )
                                seg_payload = page.evaluate(
                                    "() => window.annolidPoseDataset.captureSegmentationMasks()"
                                )
                                png_bytes = page.screenshot(type="png")
                                img_arr = np.frombuffer(png_bytes, dtype=np.uint8)
                                bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                                if bgr is None:
                                    continue

                                track_map, _fine_map, coarse_map = (
                                    _decode_segmentation_rle(seg_payload)
                                )
                                if (
                                    ref_bg is not None
                                    and config.composite_on_real_background
                                ):
                                    bgr = composite_foreground_onto_background(
                                        bgr, track_map > 0, ref_bg
                                    )
                                if (
                                    profile is not None
                                    and config.match_example_photometry
                                ):
                                    bgr = apply_environment_photometry(
                                        bgr,
                                        profile,
                                        strength=config.photometry_strength,
                                        domain_randomization=config.domain_randomization,
                                        rng=rng,
                                    )

                                kp_names = list(
                                    (payload.get("category") or {}).get("keypoints")
                                    or MOUSE_KEYPOINT_NAMES
                                )
                                ann_by_track: dict[int, dict[str, Any]] = {}
                                for ann in payload.get("annotations") or []:
                                    tid = int(ann.get("track_id") or ann.get("id") or 1)
                                    ann_by_track[tid] = ann

                                instances: list[dict[str, Any]] = []
                                for tid in sorted(
                                    int(x) for x in np.unique(track_map) if int(x) > 0
                                ):
                                    inst_mask = track_map == tid
                                    polys = _mask_to_polygons(inst_mask)
                                    if not polys:
                                        continue
                                    parts: dict[str, list[list[list[float]]]] = {}
                                    for cid, pname in MOUSE_COARSE_ID_TO_NAME.items():
                                        pm = inst_mask & (coarse_map == cid)
                                        part_polys = _mask_to_polygons(pm, min_area=8.0)
                                        if part_polys:
                                            parts[pname] = part_polys

                                    kps_list: list[dict[str, Any]] = []
                                    ann = ann_by_track.get(tid) or {}
                                    raw_kps = ann.get("keypoints") or []
                                    for k_idx, k_name in enumerate(kp_names):
                                        base = k_idx * 3
                                        if base + 2 < len(raw_kps):
                                            kps_list.append(
                                                {
                                                    "name": str(k_name),
                                                    "x": round(float(raw_kps[base]), 2),
                                                    "y": round(
                                                        float(raw_kps[base + 1]), 2
                                                    ),
                                                    "v": int(raw_kps[base + 2]),
                                                }
                                            )
                                    instances.append(
                                        {
                                            "group_id": tid,
                                            "label": f"mouse_{tid}",
                                            "class_name": "mouse",
                                            "polygons": polys,
                                            "parts": parts,
                                            "keypoints": kps_list,
                                        }
                                    )

                                stem = f"mouse_{scenario}_f{f_idx:04d}"
                                img_path = labelme_dir / f"{stem}.jpg"
                                cv2.imwrite(
                                    str(img_path),
                                    bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                                )
                                lm = _build_labelme_json(
                                    image_filename=img_path.name,
                                    width=bgr.shape[1],
                                    height=bgr.shape[0],
                                    instances=instances,
                                )
                                (labelme_dir / f"{stem}.json").write_text(
                                    json.dumps(lm, indent=2) + "\n",
                                    encoding="utf-8",
                                )
                                records.append(
                                    {
                                        "stem": stem,
                                        "scenario": scenario,
                                        "image_path": str(img_path),
                                        "width": int(bgr.shape[1]),
                                        "height": int(bgr.shape[0]),
                                        "instances": instances,
                                    }
                                )
                    finally:
                        page.close()
                        browser.close()
            finally:
                server.shutdown()
                server.server_close()
        except Exception as exc:
            if not config.allow_fallback:
                raise RuntimeError(
                    f"Mouse 3D Three.js Playwright capture failed: {exc}. "
                    "Ensure Chromium is installed (`playwright install chromium`) or pass "
                    "`allow_fallback=True` for procedural fallback."
                ) from exc
            renderer = "procedural_mouse_fallback"
            notes.append(
                f"Playwright failed ({exc}); fell back to procedural renderer."
            )
            records = _render_procedural_mouse_frames(
                config, out_dir, profile, log_callback
            )

    return _finalize_generated_dataset(
        records=records,
        config=config,
        out_dir=out_dir,
        labelme_dir=labelme_dir,
        renderer=renderer,
        notes=notes,
        log_callback=log_callback,
    )


def _find_flybody_python() -> Optional[str]:
    """Return a Python executable capable of actually instantiating FlyBody/MuJoCo envs."""
    try:
        import flybody.fly_envs  # noqa: F401
        import dm_control  # noqa: F401
        import mujoco  # noqa: F401

        return sys.executable
    except Exception:
        pass
    try:
        from annolid.gui.flybody_support import flybody_runtime_python_candidates

        candidates = list(flybody_runtime_python_candidates())
    except Exception:
        candidates = []
    for extra in (
        REPO_ROOT / ".venv311" / "bin" / "python",
        REPO_ROOT / ".venv311" / "Scripts" / "python.exe",
    ):
        if extra not in candidates:
            candidates.append(extra)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            probe = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    "import flybody.fly_envs, dm_control, mujoco; print('ok')",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if probe.returncode == 0 and "ok" in (probe.stdout or ""):
                return str(candidate)
        except Exception:
            continue
    return None


def _place_transformed_fly_on_canvas(
    canvas: np.ndarray,
    fly_bgr: np.ndarray,
    fly_coarse: np.ndarray,
    keypoints: Sequence[dict[str, Any]],
    *,
    scale: float,
    angle_deg: float,
    dx: float,
    dy: float,
    occupied_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Affine-transform a single rendered fly crop onto a multi-fly arena canvas."""
    h, w = canvas.shape[:2]
    fg = fly_coarse > 0
    if not np.any(fg):
        return canvas, None
    ys, xs = np.nonzero(fg)
    cx = float(xs.mean())
    cy = float(ys.mean())

    mat = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), float(scale))
    mat[0, 2] += float(dx)
    mat[1, 2] += float(dy)

    warped_bgr = cv2.warpAffine(
        fly_bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    warped_coarse = cv2.warpAffine(
        fly_coarse, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    warped_fg = warped_coarse > 0
    if int(np.count_nonzero(warped_fg)) < 16:
        return canvas, None
    if int(np.count_nonzero(warped_fg & occupied_mask)) > int(
        0.35 * np.count_nonzero(warped_fg)
    ):
        return canvas, None

    out = composite_foreground_onto_background(
        warped_bgr, warped_fg, canvas, feather_radius=3, match_fg_lighting=False
    )
    occupied_mask |= warped_fg

    polys = _mask_to_polygons(warped_fg, min_area=10.0)
    parts: dict[str, list[list[list[float]]]] = {}
    for pid, pname in FLY_PART_ID_TO_NAME.items():
        pm = warped_coarse == pid
        pp = _mask_to_polygons(pm, min_area=4.0)
        if pp:
            parts[pname] = pp

    xformed_kps: list[dict[str, Any]] = []
    for kp in keypoints:
        v = int(kp.get("v", 0))
        if v <= 0:
            xformed_kps.append({"name": kp["name"], "x": 0.0, "y": 0.0, "v": 0})
            continue
        nx = mat[0, 0] * float(kp["x"]) + mat[0, 1] * float(kp["y"]) + mat[0, 2]
        ny = mat[1, 0] * float(kp["x"]) + mat[1, 1] * float(kp["y"]) + mat[1, 2]
        in_b = 0.0 <= nx < float(w) and 0.0 <= ny < float(h)
        xformed_kps.append(
            {
                "name": str(kp["name"]),
                "x": round(float(nx), 2),
                "y": round(float(ny), 2),
                "v": v if in_b else 0,
            }
        )

    return out, {
        "polygons": polys,
        "parts": parts,
        "keypoints": xformed_kps,
    }


def _render_procedural_fly_frames(
    config: SimDatasetConfig,
    output_dir: Path,
    profile: Optional[VideoEnvironmentProfile],
    log_callback: LogCallback,
) -> list[dict[str, Any]]:
    """Explicit opt-in fallback fly renderer when FlyBody/MuJoCo is unavailable."""
    _emit(
        log_callback,
        "[WARN] Using explicit procedural_fly_fallback renderer (FlyBody/MuJoCo unavailable).",
    )
    labelme_dir = output_dir / "labelme"
    labelme_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(config.width), int(config.height)
    rng = np.random.default_rng(int(config.seed))
    ref_bg = load_reference_background(profile)
    n_flies = max(1, min(6, int(config.fly_subject_count)))
    records: list[dict[str, Any]] = []

    for b_idx, behavior in enumerate(config.scenarios):
        for f_idx in range(int(config.frames_per_scenario)):
            t = float(f_idx) / float(max(1, config.frames_per_scenario))
            if ref_bg is not None and config.composite_on_real_background:
                canvas = cv2.resize(ref_bg, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                bg_val = 215 if config.lighting != "ir_night" else 52
                canvas = np.full((h, w, 3), bg_val, dtype=np.uint8)
                if config.arena == "circular_open_field":
                    cv2.circle(
                        canvas,
                        (w // 2, h // 2),
                        int(min(w, h) * 0.44),
                        (bg_val - 35, bg_val - 35, bg_val - 35),
                        3,
                    )

            instances: list[dict[str, Any]] = []
            for fly_i in range(n_flies):
                angle = (t * 2.0 * math.pi + fly_i * 1.7 + b_idx * 0.9) % (
                    2.0 * math.pi
                )
                orbit_r = min(w, h) * (0.12 + 0.08 * (fly_i % 3))
                cx = w * 0.5 + orbit_r * math.cos(angle + fly_i)
                cy = h * 0.5 + orbit_r * math.sin(angle * 0.8 + fly_i)
                body_len = max(14.0, min(w, h) * 0.045)
                ca, sa = math.cos(angle), math.sin(angle)

                part_masks = {
                    name: np.zeros((h, w), dtype=np.uint8)
                    for name in FLY_PART_ID_TO_NAME.values()
                }

                # Thorax
                tc = (int(round(cx)), int(round(cy)))
                cv2.ellipse(
                    canvas,
                    tc,
                    (int(body_len * 0.42), int(body_len * 0.28)),
                    math.degrees(angle),
                    0,
                    360,
                    (40, 52, 72),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    part_masks["thorax"],
                    tc,
                    (int(body_len * 0.42), int(body_len * 0.28)),
                    math.degrees(angle),
                    0,
                    360,
                    255,
                    -1,
                )

                # Head
                hx, hy = cx + ca * body_len * 0.55, cy + sa * body_len * 0.55
                hc = (int(round(hx)), int(round(hy)))
                cv2.circle(
                    canvas,
                    hc,
                    max(3, int(body_len * 0.24)),
                    (32, 34, 96),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    part_masks["head"], hc, max(3, int(body_len * 0.24)), 255, -1
                )

                # Abdomen
                ax, ay = cx - ca * body_len * 0.65, cy - sa * body_len * 0.65
                ac = (int(round(ax)), int(round(ay)))
                cv2.ellipse(
                    canvas,
                    ac,
                    (int(body_len * 0.52), int(body_len * 0.30)),
                    math.degrees(angle),
                    0,
                    360,
                    (46, 62, 84),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.ellipse(
                    part_masks["abdomen"],
                    ac,
                    (int(body_len * 0.52), int(body_len * 0.30)),
                    math.degrees(angle),
                    0,
                    360,
                    255,
                    -1,
                )

                # Translucent wings
                wing_spread = (
                    0.65 if "flight" in behavior or "takeoff" in behavior else 0.28
                )
                wing_centers: list[tuple[float, float]] = []
                for side in (-1.0, 1.0):
                    w_ang = angle + math.pi + side * wing_spread
                    wx = cx + math.cos(w_ang) * body_len * 0.55
                    wy = cy + math.sin(w_ang) * body_len * 0.55
                    wing_centers.append((wx, wy))
                    cv2.ellipse(
                        canvas,
                        (int(round(wx)), int(round(wy))),
                        (int(body_len * 0.55), int(body_len * 0.18)),
                        math.degrees(w_ang),
                        0,
                        360,
                        (160, 165, 172),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.ellipse(
                        part_masks["wings"],
                        (int(round(wx)), int(round(wy))),
                        (int(body_len * 0.55), int(body_len * 0.18)),
                        math.degrees(w_ang),
                        0,
                        360,
                        255,
                        -1,
                    )

                # Six legs
                leg_tips: list[tuple[float, float]] = []
                for along in (0.28, 0.0, -0.28):
                    for side in (-1.0, 1.0):
                        lx = cx + ca * body_len * along - sa * body_len * 0.58 * side
                        ly = cy + sa * body_len * along + ca * body_len * 0.58 * side
                        leg_tips.append((lx, ly))
                        cv2.line(
                            canvas,
                            tc,
                            (int(round(lx)), int(round(ly))),
                            (35, 38, 44),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            part_masks["legs"],
                            tc,
                            (int(round(lx)), int(round(ly))),
                            255,
                            2,
                        )

                inst_mask = np.zeros((h, w), dtype=np.uint8)
                for pm in part_masks.values():
                    inst_mask |= pm

                kps = [
                    {"name": "head", "x": hx, "y": hy, "v": 2},
                    {"name": "thorax", "x": cx, "y": cy, "v": 2},
                    {"name": "abdomen", "x": ax, "y": ay, "v": 2},
                    {
                        "name": "wing_left",
                        "x": wing_centers[0][0],
                        "y": wing_centers[0][1],
                        "v": 2,
                    },
                    {
                        "name": "wing_right",
                        "x": wing_centers[1][0],
                        "y": wing_centers[1][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_front_left",
                        "x": leg_tips[0][0],
                        "y": leg_tips[0][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_front_right",
                        "x": leg_tips[1][0],
                        "y": leg_tips[1][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_mid_left",
                        "x": leg_tips[2][0],
                        "y": leg_tips[2][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_mid_right",
                        "x": leg_tips[3][0],
                        "y": leg_tips[3][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_hind_left",
                        "x": leg_tips[4][0],
                        "y": leg_tips[4][1],
                        "v": 2,
                    },
                    {
                        "name": "leg_hind_right",
                        "x": leg_tips[5][0],
                        "y": leg_tips[5][1],
                        "v": 2,
                    },
                ]
                for kp in kps:
                    kp["x"] = round(float(np.clip(kp["x"], 1.0, w - 2.0)), 2)
                    kp["y"] = round(float(np.clip(kp["y"], 1.0, h - 2.0)), 2)

                instances.append(
                    {
                        "group_id": fly_i + 1,
                        "label": f"fly_{fly_i + 1}",
                        "class_name": "fly",
                        "polygons": _mask_to_polygons(inst_mask, min_area=10.0),
                        "parts": {
                            pname: _mask_to_polygons(pm, min_area=4.0)
                            for pname, pm in part_masks.items()
                            if np.any(pm)
                        },
                        "keypoints": kps,
                    }
                )

            if profile is not None and config.match_example_photometry:
                canvas = apply_environment_photometry(
                    canvas,
                    profile,
                    strength=config.photometry_strength,
                    domain_randomization=config.domain_randomization,
                    rng=rng,
                )

            stem = f"fly_{behavior}_f{f_idx:04d}"
            img_path = labelme_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lm = _build_labelme_json(
                image_filename=img_path.name,
                width=w,
                height=h,
                instances=instances,
            )
            (labelme_dir / f"{stem}.json").write_text(
                json.dumps(lm, indent=2) + "\n", encoding="utf-8"
            )
            records.append(
                {
                    "stem": stem,
                    "scenario": behavior,
                    "image_path": str(img_path),
                    "width": w,
                    "height": h,
                    "instances": instances,
                }
            )
    return records


def generate_fly_sim_dataset(
    config: SimDatasetConfig,
    output_dir: str | Path,
    *,
    profile: Optional[VideoEnvironmentProfile] = None,
    log_callback: LogCallback = None,
) -> SimDatasetArtifact:
    """Generate environment-matched FlyBody 3D MuJoCo synthetic dataset."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labelme_dir = out_dir / "labelme"
    labelme_dir.mkdir(parents=True, exist_ok=True)

    behaviors = list(config.scenarios or ["walk_imitation", "forage_walk"])
    notes: list[str] = []
    renderer = "flybody_mujoco"
    records: list[dict[str, Any]] = []

    flybody_py = _find_flybody_python()
    if flybody_py is None:
        if not config.allow_fallback:
            raise RuntimeError(
                "FlyBody MuJoCo runtime (`flybody`, `dm_control`, `mujoco`) was not found in the "
                "active Python environment or `.venv311`. Install FlyBody via Annolid's "
                "`Install FlyBody...` menu / docs (`docs/simulation_flybody.md`), or pass "
                "`allow_fallback=True` to opt into the procedural fly fallback."
            )
        renderer = "procedural_fly_fallback"
        notes.append(
            "Fallback renderer used because FlyBody/MuJoCo runtime was unavailable (`allow_fallback=True`)."
        )
        records = _render_procedural_fly_frames(config, out_dir, profile, log_callback)
    else:
        raw_work_dir = out_dir / "_flybody_raw"
        _emit(
            log_callback,
            f"Running FlyBody MuJoCo renderer via {flybody_py} "
            f"({len(behaviors)} behavior(s), {config.frames_per_scenario} frames each, camera={config.camera_view})...",
        )
        try:
            if Path(flybody_py).resolve() == Path(sys.executable).resolve():
                raw_manifest = render_flybody_scenarios(
                    raw_work_dir,
                    behaviors=behaviors,
                    frames_per_behavior=int(config.frames_per_scenario),
                    width=int(config.width),
                    height=int(config.height),
                    camera_view=str(config.camera_view),
                    seed=int(config.seed),
                    brain_dynamics=str(config.fly_brain_dynamics or "spiking"),
                )
            else:
                import os

                worker_script = (
                    Path(__file__).resolve().with_name("flybody_render_worker.py")
                )
                cmd = [
                    flybody_py,
                    str(worker_script),
                    "--output",
                    str(raw_work_dir),
                    "--behaviors",
                    *behaviors,
                    "--frames-per-behavior",
                    str(int(config.frames_per_scenario)),
                    "--width",
                    str(int(config.width)),
                    "--height",
                    str(int(config.height)),
                    "--camera",
                    str(config.camera_view),
                    "--seed",
                    str(int(config.seed)),
                    "--brain-dynamics",
                    str(config.fly_brain_dynamics or "spiking"),
                ]
                worker_env = dict(os.environ)
                existing_pp = worker_env.get("PYTHONPATH", "")
                worker_env["PYTHONPATH"] = (
                    f"{REPO_ROOT}{os.pathsep}{existing_pp}"
                    if existing_pp
                    else str(REPO_ROOT)
                )
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=worker_env,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"FlyBody render worker exited with code {proc.returncode}: "
                        f"{(proc.stderr or proc.stdout or '').strip()[-600:]}"
                    )
                raw_manifest = json.loads(
                    (raw_work_dir / "flybody_raw_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )

            ref_bg = load_reference_background(profile)
            rng = np.random.default_rng(int(config.seed))
            w, h = int(config.width), int(config.height)
            n_flies = max(1, min(6, int(config.fly_subject_count)))
            raw_frames = list(raw_manifest.get("frames") or [])

            for idx, rframe in enumerate(raw_frames):
                raw_bgr = cv2.imread(str(rframe["image_path"]), cv2.IMREAD_COLOR)
                packed = cv2.imread(str(rframe["mask_path"]), cv2.IMREAD_UNCHANGED)
                if raw_bgr is None or packed is None:
                    continue
                coarse_map = packed[:, :, 0]
                fg_mask = packed[:, :, 1] > 0

                instances: list[dict[str, Any]] = []
                if n_flies > 1 or (
                    ref_bg is not None and config.composite_on_real_background
                ):
                    if ref_bg is not None and config.composite_on_real_background:
                        canvas = cv2.resize(
                            ref_bg, (w, h), interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        # Clean arena backdrop from background pixels of the rendered frame.
                        bg_vals = raw_bgr[~fg_mask]
                        fill_bgr = (
                            np.median(bg_vals, axis=0).astype(np.uint8)
                            if bg_vals.size > 0
                            else np.array([210, 210, 210], dtype=np.uint8)
                        )
                        canvas = np.broadcast_to(fill_bgr, (h, w, 3)).copy()

                    occupied = np.zeros((h, w), dtype=bool)
                    for copy_idx in range(n_flies):
                        src_idx = (idx + copy_idx * 7) % len(raw_frames)
                        src_rec = raw_frames[src_idx]
                        s_bgr = cv2.imread(str(src_rec["image_path"]), cv2.IMREAD_COLOR)
                        s_pack = cv2.imread(
                            str(src_rec["mask_path"]), cv2.IMREAD_UNCHANGED
                        )
                        if s_bgr is None or s_pack is None:
                            continue
                        s_coarse = s_pack[:, :, 0]
                        if (
                            copy_idx == 0
                            and n_flies == 1
                            and config.domain_randomization < 0.05
                        ):
                            scale, angle_deg, dx, dy = 1.0, 0.0, 0.0, 0.0
                        else:
                            scale = float(rng.uniform(0.85, 1.15))
                            angle_deg = float(rng.uniform(-180.0, 180.0))
                            dx = float(rng.uniform(-w * 0.28, w * 0.28))
                            dy = float(rng.uniform(-h * 0.28, h * 0.28))
                        canvas, inst_data = _place_transformed_fly_on_canvas(
                            canvas,
                            s_bgr,
                            s_coarse,
                            src_rec.get("keypoints") or [],
                            scale=scale,
                            angle_deg=angle_deg,
                            dx=dx,
                            dy=dy,
                            occupied_mask=occupied,
                        )
                        if inst_data is not None:
                            instances.append(
                                {
                                    "group_id": len(instances) + 1,
                                    "label": f"fly_{len(instances) + 1}",
                                    "class_name": "fly",
                                    **inst_data,
                                }
                            )
                    out_bgr = canvas
                else:
                    out_bgr = raw_bgr
                    polys = _mask_to_polygons(fg_mask, min_area=10.0)
                    parts: dict[str, list[list[list[float]]]] = {}
                    for pid, pname in FLY_PART_ID_TO_NAME.items():
                        pm = coarse_map == pid
                        pp = _mask_to_polygons(pm, min_area=4.0)
                        if pp:
                            parts[pname] = pp
                    if polys:
                        instances.append(
                            {
                                "group_id": 1,
                                "label": "fly_1",
                                "class_name": "fly",
                                "polygons": polys,
                                "parts": parts,
                                "keypoints": list(rframe.get("keypoints") or []),
                            }
                        )

                if profile is not None and config.match_example_photometry:
                    out_bgr = apply_environment_photometry(
                        out_bgr,
                        profile,
                        strength=config.photometry_strength,
                        domain_randomization=config.domain_randomization,
                        rng=rng,
                    )

                stem = f"fly_{rframe['stem']}"
                img_path = labelme_dir / f"{stem}.jpg"
                cv2.imwrite(str(img_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                lm = _build_labelme_json(
                    image_filename=img_path.name,
                    width=out_bgr.shape[1],
                    height=out_bgr.shape[0],
                    instances=instances,
                )
                (labelme_dir / f"{stem}.json").write_text(
                    json.dumps(lm, indent=2) + "\n", encoding="utf-8"
                )
                records.append(
                    {
                        "stem": stem,
                        "scenario": str(rframe.get("behavior") or "fly"),
                        "image_path": str(img_path),
                        "width": int(out_bgr.shape[1]),
                        "height": int(out_bgr.shape[0]),
                        "instances": instances,
                    }
                )
        except Exception as exc:
            if not config.allow_fallback:
                raise RuntimeError(
                    f"FlyBody MuJoCo dataset generation failed: {exc}"
                ) from exc
            renderer = "procedural_fly_fallback"
            notes.append(
                f"FlyBody MuJoCo failed ({exc}); fell back to procedural renderer."
            )
            records = _render_procedural_fly_frames(
                config, out_dir, profile, log_callback
            )

    return _finalize_generated_dataset(
        records=records,
        config=config,
        out_dir=out_dir,
        labelme_dir=labelme_dir,
        renderer=renderer,
        notes=notes,
        log_callback=log_callback,
    )


def _render_procedural_rat_frames(
    config: SimDatasetConfig,
    out_dir: Path,
    profile: Optional[VideoEnvironmentProfile],
    log_callback: LogCallback,
) -> list[dict[str, Any]]:
    """Deterministic OpenCV procedural fallback driven by RatNCAPController 3D kinematics."""
    from annolid.simulation.rat_ncap import (
        RAT_V2_KEYPOINT_NAMES,
        RatNCAPConfig,
        RatNCAPController,
    )

    w, h = int(config.width), int(config.height)
    rng = np.random.default_rng(int(config.seed))
    ref_bg = load_reference_background(profile)
    scenarios = list(config.scenarios or ["trot", "walk"])
    img_dir = out_dir / "labelme"
    img_dir.mkdir(parents=True, exist_ok=True)

    coat = str(config.rat_coat or "black_hooded").strip().lower()
    coat_palette = {
        "black_hooded": ((36, 36, 40), (232, 236, 240)),
        "cinnamon": ((46, 78, 124), (218, 228, 238)),
        "gray": ((118, 110, 104), (232, 234, 236)),
    }
    hood_bgr, body_bgr = coat_palette.get(coat, coat_palette["black_hooded"])
    skin_bgr = (168, 178, 224)

    records: list[dict[str, Any]] = []
    for s_idx, scenario in enumerate(scenarios):
        ctrl = RatNCAPController(
            RatNCAPConfig(
                interface=config.rat_interface,
                behavior=scenario,
                coat=coat,
                ablation=config.rat_ablation,
                ablation_scale=float(config.rat_ablation_scale),
                terrain_bump_cm=float(config.rat_terrain_bump_cm),
                seed=int(config.seed) + s_idx * 31,
            )
        )
        for f_idx in range(int(config.frames_per_scenario)):
            for _ in range(3):
                step_data = ctrl.step()

            if ref_bg is not None and config.composite_on_real_background:
                canvas = cv2.resize(ref_bg, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                bg_val = int(profile.mean_luminance) if profile is not None else 210
                canvas = np.full((h, w, 3), bg_val, dtype=np.uint8)

            root_x, root_y, root_z = step_data["root_pos"]
            kps_3d = step_data["keypoints_3d"]
            scale_px = min(w, h) * 1.45
            cx_img = w * 0.5 + float(rng.uniform(-0.10, 0.10)) * w
            cy_img = h * 0.55 + float(rng.uniform(-0.08, 0.08)) * h

            def _proj(pt3: Sequence[float]) -> tuple[float, float]:
                dx = float(pt3[0]) - float(root_x)
                dy = float(pt3[1]) - float(root_y)
                dz = float(pt3[2]) - float(root_z)
                if config.camera_view in {"track_top", "top"}:
                    px = cx_img + dx * scale_px
                    py = cy_img - dy * scale_px
                else:
                    px = cx_img + dx * scale_px + 0.25 * dy * scale_px
                    py = cy_img - dz * scale_px + 0.30 * dy * scale_px
                return float(np.clip(px, 2.0, w - 3.0)), float(
                    np.clip(py, 2.0, h - 3.0)
                )

            kp_2d: dict[str, tuple[float, float]] = {
                name: _proj(kps_3d[name])
                for name in RAT_V2_KEYPOINT_NAMES
                if name in kps_3d
            }

            full_mask = np.zeros((h, w), dtype=np.uint8)
            part_masks = {
                k: np.zeros((h, w), dtype=np.uint8)
                for k in ("head", "torso", "limbs", "tail")
            }

            # Draw tail -> limbs -> torso -> head
            tail_pts = np.asarray(
                [kp_2d["tail_base"], kp_2d["tail_mid"], kp_2d["tail_tip"]],
                dtype=np.int32,
            )
            cv2.polylines(
                canvas,
                [tail_pts],
                False,
                skin_bgr,
                max(4, int(min(w, h) * 0.016)),
                cv2.LINE_AA,
            )
            cv2.polylines(
                part_masks["tail"],
                [tail_pts],
                False,
                255,
                max(4, int(min(w, h) * 0.016)),
            )

            for anchor, paw in (
                ("thorax", "left_forepaw"),
                ("thorax", "right_forepaw"),
                ("pelvis", "left_hindpaw"),
                ("pelvis", "right_hindpaw"),
            ):
                p0 = (int(kp_2d[anchor][0]), int(kp_2d[anchor][1]))
                p1 = (int(kp_2d[paw][0]), int(kp_2d[paw][1]))
                rad = max(5, int(min(w, h) * 0.018))
                cv2.line(canvas, p0, p1, body_bgr, rad, cv2.LINE_AA)
                cv2.circle(canvas, p1, max(4, rad // 2), skin_bgr, -1, cv2.LINE_AA)
                cv2.line(part_masks["limbs"], p0, p1, 255, rad)

            pelvis_pt = (int(kp_2d["pelvis"][0]), int(kp_2d["pelvis"][1]))
            thorax_pt = (int(kp_2d["thorax"][0]), int(kp_2d["thorax"][1]))
            torso_r = max(14, int(min(w, h) * 0.068))
            cv2.line(canvas, pelvis_pt, thorax_pt, body_bgr, torso_r * 2, cv2.LINE_AA)
            cv2.circle(
                canvas, thorax_pt, int(torso_r * 0.95), hood_bgr, -1, cv2.LINE_AA
            )
            cv2.line(part_masks["torso"], pelvis_pt, thorax_pt, 255, torso_r * 2)

            head_pt = (int(kp_2d["head"][0]), int(kp_2d["head"][1]))
            nose_pt = (int(kp_2d["nose"][0]), int(kp_2d["nose"][1]))
            head_r = max(10, int(min(w, h) * 0.044))
            cv2.line(canvas, head_pt, nose_pt, hood_bgr, head_r, cv2.LINE_AA)
            cv2.circle(canvas, head_pt, head_r, hood_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, nose_pt, max(3, head_r // 3), skin_bgr, -1, cv2.LINE_AA)
            cv2.line(part_masks["head"], head_pt, nose_pt, 255, head_r)
            cv2.circle(part_masks["head"], head_pt, head_r, 255, -1)

            for pm in part_masks.values():
                full_mask |= pm

            if profile is not None and config.match_example_photometry:
                canvas = apply_environment_photometry(
                    canvas,
                    profile,
                    strength=config.photometry_strength,
                    domain_randomization=config.domain_randomization,
                    rng=rng,
                )

            polys = _mask_to_polygons(full_mask, min_area=14.0)
            parts: dict[str, list[list[list[float]]]] = {}
            for pname, pm in part_masks.items():
                pp = _mask_to_polygons(pm, min_area=8.0)
                if pp:
                    parts[pname] = pp
            kps_list = [
                {
                    "name": name,
                    "x": round(kp_2d[name][0], 2),
                    "y": round(kp_2d[name][1], 2),
                    "v": 2,
                }
                for name in RAT_V2_KEYPOINT_NAMES
                if name in kp_2d
            ]
            instances = (
                [
                    {
                        "group_id": 1,
                        "label": "rat_1",
                        "class_name": "rat",
                        "polygons": polys,
                        "parts": parts,
                        "keypoints": kps_list,
                    }
                ]
                if polys
                else []
            )

            stem = f"rat_{scenario}_f{f_idx:04d}"
            img_path = img_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lm = _build_labelme_json(
                image_filename=img_path.name,
                width=w,
                height=h,
                instances=instances,
            )
            (img_dir / f"{stem}.json").write_text(
                json.dumps(lm, indent=2) + "\n", encoding="utf-8"
            )
            records.append(
                {
                    "stem": stem,
                    "scenario": scenario,
                    "image_path": str(img_path),
                    "width": w,
                    "height": h,
                    "instances": instances,
                }
            )
    return records


def generate_rat_3d_dataset(
    config: SimDatasetConfig,
    output_dir: str | Path,
    *,
    profile: Optional[VideoEnvironmentProfile] = None,
    log_callback: LogCallback = None,
) -> SimDatasetArtifact:
    """Generate environment-matched Rat v2 / Rat NCAP 3D synthetic dataset via bundled Three.js scene."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labelme_dir = out_dir / "labelme"
    labelme_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(config.width), int(config.height)
    scenarios = list(config.scenarios or ["trot", "walk", "gait_transition"])
    notes: list[str] = []
    renderer = "threejs_rat_ncap_3d"
    records: list[dict[str, Any]] = []

    playwright_ok = importlib.util.find_spec("playwright") is not None
    scene_file = REPO_ROOT / RAT_NCAP_SCENE_REL

    if not playwright_ok or not scene_file.is_file():
        if not config.allow_fallback:
            raise RuntimeError(
                "Playwright or bundled Rat NCAP 3D scene (`annolid/gui/assets/threejs/rat_ncap.html`) "
                "is unavailable. Install Playwright Chromium (`playwright install chromium`) or pass "
                "`allow_fallback=True` to opt into the procedural Rat NCAP fallback renderer."
            )
        renderer = "procedural_rat_fallback"
        notes.append(
            "Fallback Rat NCAP renderer used because Playwright Chromium was unavailable (`allow_fallback=True`)."
        )
        records = _render_procedural_rat_frames(config, out_dir, profile, log_callback)
    else:
        try:
            from playwright.sync_api import sync_playwright

            server, base_url, _thread = _start_static_server(REPO_ROOT)
            ref_bg = load_reference_background(profile)
            rng = np.random.default_rng(int(config.seed))
            coat = str(config.rat_coat or "black_hooded")
            _emit(
                log_callback,
                f"Launching Rat NCAP 3D Three.js renderer ({len(scenarios)} scenario(s), "
                f"{config.frames_per_scenario} frames each, arena={config.arena}, "
                f"camera={config.camera_view}, coat={coat}, interface={config.rat_interface})...",
            )
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            "--use-gl=angle",
                            "--use-angle=swiftshader",
                            "--enable-unsafe-swiftshader",
                            "--enable-webgl",
                            "--ignore-gpu-blocklist",
                        ],
                    )
                    page = browser.new_page(viewport={"width": w, "height": h})
                    try:
                        for s_idx, scenario in enumerate(scenarios):
                            query_dict: dict[str, Any] = {
                                "dataset": 1,
                                "frame": 0,
                                "fps": int(config.fps),
                                "seed": int(config.seed) + s_idx * 97,
                                "camera_view": config.camera_view,
                                "scenario": scenario,
                                "arena": config.arena,
                                "coat": coat,
                                "interface": config.rat_interface,
                            }
                            if config.lighting and config.lighting != "default":
                                query_dict["lighting"] = config.lighting
                            if config.rat_ablation and config.rat_ablation != "none":
                                query_dict["ablation"] = config.rat_ablation
                                query_dict["ablation_scale"] = config.rat_ablation_scale
                            if config.rat_terrain_bump_cm > 0:
                                query_dict["bump_cm"] = config.rat_terrain_bump_cm

                            url = f"{base_url}/{RAT_NCAP_SCENE_REL}?{urlencode(query_dict)}"
                            _emit(
                                log_callback,
                                f"Rendering Rat NCAP scenario '{scenario}'...",
                            )
                            page.goto(url, wait_until="load", timeout=120_000)
                            page.wait_for_selector(
                                "#c[data-dataset-ready='true']", timeout=120_000
                            )

                            for f_idx in range(int(config.frames_per_scenario)):
                                payload = page.evaluate(
                                    "idx => window.annolidPoseDataset.setFrame(idx)",
                                    f_idx,
                                )
                                seg_payload = page.evaluate(
                                    "() => window.annolidPoseDataset.captureSegmentationMasks()"
                                )
                                png_bytes = page.screenshot(type="png")
                                img_arr = np.frombuffer(png_bytes, dtype=np.uint8)
                                bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                                if bgr is None:
                                    continue

                                track_map, _fine_map, coarse_map = (
                                    _decode_segmentation_rle(seg_payload)
                                )
                                if (
                                    ref_bg is not None
                                    and config.composite_on_real_background
                                ):
                                    bgr = composite_foreground_onto_background(
                                        bgr, track_map > 0, ref_bg
                                    )
                                if (
                                    profile is not None
                                    and config.match_example_photometry
                                ):
                                    bgr = apply_environment_photometry(
                                        bgr,
                                        profile,
                                        strength=config.photometry_strength,
                                        domain_randomization=config.domain_randomization,
                                        rng=rng,
                                    )

                                subj_by_track: dict[int, dict[str, Any]] = {}
                                for subj in payload.get("subjects") or []:
                                    tid = int(subj.get("track_id") or 1)
                                    subj_by_track[tid] = subj

                                instances: list[dict[str, Any]] = []
                                for tid in sorted(
                                    int(x) for x in np.unique(track_map) if int(x) > 0
                                ):
                                    inst_mask = track_map == tid
                                    polys = _mask_to_polygons(inst_mask)
                                    if not polys:
                                        continue
                                    parts: dict[str, list[list[list[float]]]] = {}
                                    for cid, pname in MOUSE_COARSE_ID_TO_NAME.items():
                                        pm = inst_mask & (coarse_map == cid)
                                        part_polys = _mask_to_polygons(pm, min_area=8.0)
                                        if part_polys:
                                            parts[pname] = part_polys

                                    subj = subj_by_track.get(tid) or {}
                                    kp_map = subj.get("keypoints") or {}
                                    kps_list: list[dict[str, Any]] = []
                                    for k_name in RAT_KEYPOINT_NAMES:
                                        vals = kp_map.get(k_name)
                                        if (
                                            isinstance(vals, (list, tuple))
                                            and len(vals) >= 3
                                        ):
                                            kps_list.append(
                                                {
                                                    "name": str(k_name),
                                                    "x": round(float(vals[0]), 2),
                                                    "y": round(float(vals[1]), 2),
                                                    "v": int(vals[2]),
                                                }
                                            )
                                    instances.append(
                                        {
                                            "group_id": tid,
                                            "label": f"rat_{tid}",
                                            "class_name": "rat",
                                            "polygons": polys,
                                            "parts": parts,
                                            "keypoints": kps_list,
                                        }
                                    )

                                stem = f"rat_{scenario}_f{f_idx:04d}"
                                img_path = labelme_dir / f"{stem}.jpg"
                                cv2.imwrite(
                                    str(img_path),
                                    bgr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                                )
                                lm = _build_labelme_json(
                                    image_filename=img_path.name,
                                    width=bgr.shape[1],
                                    height=bgr.shape[0],
                                    instances=instances,
                                )
                                (labelme_dir / f"{stem}.json").write_text(
                                    json.dumps(lm, indent=2) + "\n",
                                    encoding="utf-8",
                                )
                                records.append(
                                    {
                                        "stem": stem,
                                        "scenario": scenario,
                                        "image_path": str(img_path),
                                        "width": int(bgr.shape[1]),
                                        "height": int(bgr.shape[0]),
                                        "instances": instances,
                                    }
                                )
                    finally:
                        page.close()
                        browser.close()
            finally:
                server.shutdown()
                server.server_close()
        except Exception as exc:
            if not config.allow_fallback:
                raise RuntimeError(
                    f"Rat NCAP 3D Three.js Playwright capture failed: {exc}. "
                    "Ensure Chromium is installed (`playwright install chromium`) or pass "
                    "`allow_fallback=True` for procedural fallback."
                ) from exc
            renderer = "procedural_rat_fallback"
            notes.append(
                f"Playwright failed ({exc}); fell back to procedural Rat NCAP renderer."
            )
            records = _render_procedural_rat_frames(
                config, out_dir, profile, log_callback
            )

    return _finalize_generated_dataset(
        records=records,
        config=config,
        out_dir=out_dir,
        labelme_dir=labelme_dir,
        renderer=renderer,
        notes=notes,
        log_callback=log_callback,
    )


def _load_real_labelme_records(
    real_dir: Path, *, species: str, oversample: int = 1
) -> list[dict[str, Any]]:
    """Load optional user-provided real LabelMe JSON+image frames for sim+real mixing."""
    loaded: list[dict[str, Any]] = []
    if not real_dir.is_dir():
        return loaded
    default_cls = (
        "fly" if species == "fly" else ("rat" if species == "rat" else "mouse")
    )
    for jf in sorted(real_dir.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_name = str(payload.get("imagePath") or "")
        img_path = (jf.parent / Path(img_name).name) if img_name else None
        if img_path is None or not img_path.is_file():
            for ext in (".jpg", ".png", ".jpeg"):
                cand = jf.with_suffix(ext)
                if cand.is_file():
                    img_path = cand
                    break
        if img_path is None or not img_path.is_file():
            continue
        w = int(payload.get("imageWidth") or 0)
        h = int(payload.get("imageHeight") or 0)
        if w <= 0 or h <= 0:
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]

        instances: list[dict[str, Any]] = []
        for idx, sh in enumerate(payload.get("shapes") or []):
            stype = str(sh.get("shape_type") or "polygon").lower()
            pts = sh.get("points") or []
            if stype == "polygon" and len(pts) >= 3:
                poly = [[float(p[0]), float(p[1])] for p in pts]
            elif stype == "rectangle" and len(pts) >= 2:
                x0, y0 = float(pts[0][0]), float(pts[0][1])
                x1, y1 = float(pts[1][0]), float(pts[1][1])
                poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                continue
            raw_lbl = str(sh.get("label") or default_cls).strip()
            cls_name = (
                default_cls
                if raw_lbl.startswith(("mouse_", "rat_", "fly_"))
                else raw_lbl
            )
            instances.append(
                {
                    "group_id": int(sh.get("group_id") or (idx + 1)),
                    "label": raw_lbl,
                    "class_name": cls_name,
                    "polygons": [poly],
                    "parts": {},
                    "keypoints": [],
                }
            )
        if not instances:
            continue
        for copy_i in range(max(1, int(oversample))):
            loaded.append(
                {
                    "stem": f"real_{jf.stem}_c{copy_i}",
                    "source_group": f"real::{jf.resolve()}",
                    "scenario": "real_labeled",
                    "image_path": str(img_path),
                    "width": w,
                    "height": h,
                    "instances": instances,
                }
            )
    return loaded


def _finalize_generated_dataset(
    *,
    records: list[dict[str, Any]],
    config: SimDatasetConfig,
    out_dir: Path,
    labelme_dir: Path,
    renderer: str,
    notes: list[str],
    log_callback: LogCallback,
) -> SimDatasetArtifact:
    if config.real_labeled_dir:
        real_path = Path(config.real_labeled_dir).expanduser().resolve()
        real_recs = _load_real_labelme_records(
            real_path,
            species=config.species,
            oversample=int(config.real_oversample_factor),
        )
        if real_recs:
            records = list(records) + real_recs
            notes.append(
                f"Mixed {len(real_recs)} real labeled frame entries (oversample={config.real_oversample_factor}) from {real_path}."
            )

    if not records:
        raise RuntimeError("No synthetic frames were generated.")

    yolo_dir = out_dir / "YOLO_dataset"
    data_yaml_path, class_names, train_n, val_n = _export_yolo_dataset(
        records,
        yolo_dir=yolo_dir,
        task=config.task,
        export_level=config.export_level,
        species=config.species,
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )
    preview_imgs, preview_vid = _write_preview_assets(
        records, out_dir, fps=int(config.fps)
    )

    manifest = {
        "schema_version": 1,
        "renderer": renderer,
        "species": config.species,
        "task": config.task,
        "export_level": config.export_level,
        "config": config.to_dict(),
        "total_frames": len(records),
        "train_frames": train_n,
        "val_frames": val_n,
        "class_names": class_names,
        "data_yaml": str(data_yaml_path),
        "labelme_dir": str(labelme_dir),
        "preview_images": preview_imgs,
        "preview_video": preview_vid,
        "notes": notes,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _emit(
        log_callback,
        f"Dataset ready ({renderer}): {len(records)} frames ({train_n} train / {val_n} val), "
        f"classes={class_names} -> {data_yaml_path}",
    )
    return SimDatasetArtifact(
        dataset_dir=str(out_dir),
        data_yaml_path=str(data_yaml_path),
        labelme_dir=str(labelme_dir),
        manifest_path=str(manifest_path),
        renderer=renderer,
        species=config.species,
        task=config.task,
        export_level=config.export_level,
        total_frames=len(records),
        train_frames=train_n,
        val_frames=val_n,
        class_names=class_names,
        preview_images=preview_imgs,
        preview_video=preview_vid,
        notes=notes,
    )


def generate_sim_dataset(
    config: SimDatasetConfig,
    output_dir: str | Path,
    *,
    profile: Optional[VideoEnvironmentProfile] = None,
    log_callback: LogCallback = None,
) -> SimDatasetArtifact:
    """Unified entry point generating a Mouse 3D, Rat NCAP 3D, or FlyBody 3D synthetic dataset."""
    species = str(config.species or "mouse").strip().lower()
    if species == "fly":
        return generate_fly_sim_dataset(
            config, output_dir, profile=profile, log_callback=log_callback
        )
    if species == "rat":
        return generate_rat_3d_dataset(
            config, output_dir, profile=profile, log_callback=log_callback
        )
    if species == "mouse":
        return generate_mouse_3d_dataset(
            config, output_dir, profile=profile, log_callback=log_callback
        )
    raise ValueError(
        f"Unsupported simulation species '{species}'. Use 'mouse', 'rat', or 'fly'."
    )
