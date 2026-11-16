#!/usr/bin/env python3
"""Generate two-mouse 3D simulation videos + ground-truth pose/mask datasets and benchmark keypoint tracking."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import gc
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

import cv2
import numpy as np
from PIL import Image
import torch

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.features.dino_models import DEFAULT_DINO_FEATURE_MODEL_ID
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import (
    DinoKeypointTracker,
    DinoKeypointVideoProcessor,
)
from annolid.utils.body_part_shapes import (
    build_body_part_shape_dict,
    infer_part_from_keypoint_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = "annolid/gui/assets/threejs/two_mice.html"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "two_mice_keypoint_tracking_eval"

COARSE_ID_TO_NAME: Dict[int, str] = {
    1: "head",
    2: "torso",
    3: "limbs",
    4: "tail",
}
COARSE_NAME_TO_ID: Dict[str, int] = {v: k for k, v in COARSE_ID_TO_NAME.items()}

FINE_ID_TO_NAME: Dict[int, str] = {
    1: "head",
    2: "snout",
    3: "left_ear",
    4: "right_ear",
    5: "neck",
    6: "torso",
    7: "left_forelimb",
    8: "right_forelimb",
    9: "left_hindlimb",
    10: "right_hindlimb",
    11: "tail",
}

KEYPOINT_PART_GROUPS: Dict[str, Tuple[str, ...]] = {
    "head": ("nose", "left_ear", "right_ear", "neck"),
    "spine": ("spine_1", "spine_2", "tail_base"),
    "paws": ("left_forepaw", "right_forepaw", "left_hindpaw", "right_hindpaw"),
    "tail": ("tail_mid", "tail_tip"),
}

SKELETON_EDGES: Tuple[Tuple[str, str], ...] = (
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("nose", "neck"),
    ("neck", "spine_1"),
    ("spine_1", "spine_2"),
    ("spine_2", "tail_base"),
    ("tail_base", "tail_mid"),
    ("tail_mid", "tail_tip"),
    ("neck", "left_forepaw"),
    ("neck", "right_forepaw"),
    ("spine_2", "left_hindpaw"),
    ("spine_2", "right_hindpaw"),
)

LR_SYMMETRY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("left_ear", "right_ear"),
    ("left_forepaw", "right_forepaw"),
    ("left_hindpaw", "right_hindpaw"),
)


@dataclass(frozen=True)
class SimulationScenarioSpec:
    """Deterministic simulation scenario configuration."""

    name: str
    description: str
    camera_view: str
    scenario: str
    seed: int
    fps: int
    frame_count: int


DEFAULT_SCENARIOS: Tuple[SimulationScenarioSpec, ...] = (
    SimulationScenarioSpec(
        name="arena_explore",
        description="Overhead full-arena exploration with C57BL/6 (black) and BALB/c (white) mice turning and sniffing with full tails visible.",
        camera_view="top",
        scenario="arena_explore",
        seed=20260710,
        fps=15,
        frame_count=24,
    ),
    SimulationScenarioSpec(
        name="social_huddle",
        description="Overhead close side-by-side approach, flank contact/huddling, cross-animal occlusion, and pass-through.",
        camera_view="top",
        scenario="social_huddle",
        seed=20260711,
        fps=15,
        frame_count=28,
    ),
    SimulationScenarioSpec(
        name="curled_grooming",
        description="Close-up curled C-shaped flank-grooming posture on mouse_1 (occluding forepaws) while mouse_2 explores nearby.",
        camera_view="top-close",
        scenario="curled_groom",
        seed=20260712,
        fps=15,
        frame_count=22,
    ),
    SimulationScenarioSpec(
        name="rearing_oblique",
        description="Oblique 3D studio angle with upright rearing exposing head, torso, fore/hindlimbs, and full tail.",
        camera_view="oblique",
        scenario="rearing_social",
        seed=20260713,
        fps=15,
        frame_count=22,
    ),
)


class _QuietRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _serve_repo() -> Iterator[str]:
    handler = partial(_QuietRequestHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _launch_chromium(playwright: Any) -> Any:
    try:
        return playwright.chromium.launch(headless=True)
    except Exception:
        return playwright.chromium.launch(headless=True, channel="chromium")


def decode_segmentation_rle(
    seg_payload: Mapping[str, Any],
    *,
    smooth_fur_gaps: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode WebGL RLE segmentation payload into ``(track_map, fine_map, coarse_map)``."""
    width = int(seg_payload["width"])
    height = int(seg_payload["height"])
    rle = np.asarray(seg_payload["rle"], dtype=np.int32).reshape(-1, 4)
    counts = rle[:, 0]
    colors = rle[:, 1:4].astype(np.uint8)
    flat_rgb = np.repeat(colors, counts, axis=0)
    if flat_rgb.shape[0] != width * height:
        raise ValueError(
            f"Decoded RLE pixel count {flat_rgb.shape[0]} != {width * height}"
        )
    id_img = flat_rgb.reshape(height, width, 3)
    track_map = id_img[:, :, 0].copy()
    fine_map = id_img[:, :, 1].copy()
    coarse_map = id_img[:, :, 2].copy()

    if not smooth_fur_gaps:
        return track_map, fine_map, coarse_map

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    unique_tracks = [int(t) for t in np.unique(track_map) if int(t) > 0]
    closed_track = track_map.copy()
    for tid in unique_tracks:
        raw_m = track_map == tid
        other_m = (track_map > 0) & (track_map != tid)
        closed_m = (
            (cv2.morphologyEx(raw_m.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0)
            | raw_m
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


def classify_keypoint_surface_visibility(
    kp_name: str,
    x: float,
    y: float,
    raw_vis: int,
    *,
    track_id: int,
    track_map: np.ndarray,
    coarse_map: np.ndarray,
    radius_px: int = 6,
) -> int:
    """Classify a projected 3D landmark as MISSING (0), OCCLUDED (1), or VISIBLE (2) using the 3D ID depth pass."""
    if raw_vis <= 0:
        return int(KeypointVisibility.MISSING)
    h, w = track_map.shape
    ix, iy = int(round(x)), int(round(y))
    if not (0 <= ix < w and 0 <= iy < h):
        return int(KeypointVisibility.MISSING)

    x0 = max(0, ix - radius_px)
    x1 = min(w, ix + radius_px + 1)
    y0 = max(0, iy - radius_px)
    y1 = min(h, iy + radius_px + 1)
    patch_track = track_map[y0:y1, x0:x1]
    patch_coarse = coarse_map[y0:y1, x0:x1]

    on_subject = patch_track == int(track_id)
    if not np.any(on_subject):
        return int(KeypointVisibility.OCCLUDED)

    expected_coarse = infer_part_from_keypoint_name(kp_name, granularity="coarse")
    allowed_ids = {COARSE_NAME_TO_ID.get(expected_coarse, 2)}
    if kp_name == "tail_base":
        allowed_ids.update({COARSE_NAME_TO_ID["torso"], COARSE_NAME_TO_ID["tail"]})
    elif kp_name == "neck":
        allowed_ids.update({COARSE_NAME_TO_ID["torso"], COARSE_NAME_TO_ID["head"]})

    subject_parts = patch_coarse[on_subject]
    if any(int(pid) in allowed_ids for pid in np.unique(subject_parts)):
        return int(KeypointVisibility.VISIBLE)
    return int(KeypointVisibility.OCCLUDED)


def mask_to_polygons(
    mask: np.ndarray,
    *,
    epsilon: float = 0.75,
    min_area_px: int = 6,
    single_parent_contour: bool = False,
) -> List[List[List[float]]]:
    """Convert a binary mask into clean LabelMe exterior polygons."""
    m_u8 = (np.asarray(mask, dtype=bool)).astype(np.uint8) * 255
    if int(np.count_nonzero(m_u8)) < min_area_px:
        return []

    if single_parent_contour:
        n_cc, _ = cv2.connectedComponents(m_u8, connectivity=8)
        if n_cc > 2:
            bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, bridge_k)

    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if single_parent_contour:
        contours = [max(contours, key=lambda c: float(cv2.contourArea(c)))]

    polys: List[List[List[float]]] = []
    for cnt in contours:
        if float(cv2.contourArea(cnt)) < float(min_area_px) and len(cnt) < 4:
            continue
        approx = cv2.approxPolyDP(cnt, float(epsilon), True) if epsilon > 0 else cnt
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            pts = cnt.reshape(-1, 2)
        if len(pts) >= 3:
            polys.append([[round(float(px), 2), round(float(py), 2)] for px, py in pts])
    return polys


def polygons_to_mask(
    polygons: Sequence[Sequence[Sequence[float]]],
    height: int,
    width: int,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        arr = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32)
        if arr.ndim == 2 and arr.shape[0] >= 3:
            cv2.fillPoly(canvas, [arr], 1)
    return canvas > 0


def build_labelme_payload(
    *,
    image_filename: str,
    width: int,
    height: int,
    annotations: Sequence[Mapping[str, Any]],
    keypoint_names: Sequence[str],
    track_map: np.ndarray,
    coarse_map: np.ndarray,
    include_masks: bool = True,
    include_gt_body_parts: bool = False,
) -> Dict[str, Any]:
    """Construct a LabelMe JSON dictionary from simulation 3D ground truth."""
    shapes: List[Dict[str, Any]] = []
    for ann in sorted(annotations, key=lambda item: int(item["track_id"])):
        tid = int(ann["track_id"])
        inst_label = f"mouse_{tid}"
        inst_mask = track_map == tid
        if include_masks:
            parent_polys = mask_to_polygons(
                inst_mask, epsilon=0.65, min_area_px=12, single_parent_contour=True
            )
            if parent_polys:
                shapes.append(
                    {
                        "label": inst_label,
                        "points": parent_polys[0],
                        "group_id": tid,
                        "shape_type": "polygon",
                        "flags": {},
                        "visible": True,
                        "description": str(
                            ann.get("subject_name") or "simulated_mouse"
                        ),
                    }
                )

        raw_kps = ann.get("keypoints") or []
        for kp_idx, kp_name in enumerate(keypoint_names):
            base = kp_idx * 3
            if base + 2 >= len(raw_kps):
                continue
            kx = float(raw_kps[base])
            ky = float(raw_kps[base + 1])
            raw_v = int(raw_kps[base + 2])
            vis = classify_keypoint_surface_visibility(
                kp_name,
                kx,
                ky,
                raw_v,
                track_id=tid,
                track_map=track_map,
                coarse_map=coarse_map,
            )
            if vis <= int(KeypointVisibility.MISSING):
                continue
            shapes.append(
                {
                    "label": kp_name,
                    "points": [[round(kx, 2), round(ky, 2)]],
                    "group_id": tid,
                    "shape_type": "point",
                    "flags": {
                        "kp_visible": bool(vis == int(KeypointVisibility.VISIBLE)),
                    },
                    "kp_visibility": int(vis),
                    "instance_label": inst_label,
                    "display_label": kp_name,
                    "visible": True,
                }
            )

        if include_masks and include_gt_body_parts:
            parent_area = max(1, int(np.count_nonzero(inst_mask)))
            for cid, part_name in COARSE_ID_TO_NAME.items():
                pmask = inst_mask & (coarse_map == cid)
                p_area = int(np.count_nonzero(pmask))
                if p_area < 6:
                    continue
                for poly in mask_to_polygons(pmask, epsilon=0.65, min_area_px=6):
                    shapes.append(
                        build_body_part_shape_dict(
                            points=poly,
                            parent_instance=inst_label,
                            part_label=part_name,
                            group_id=tid,
                            confidence=1.0,
                            area_ratio=float(p_area) / float(parent_area),
                            mode="ground_truth_3d",
                            description="ThreeJSGroundTruth",
                        )
                    )

    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def _capture_scenario_in_page(
    page: Any,
    base_url: str,
    spec: SimulationScenarioSpec,
    scenario_dir: Path,
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Capture a single simulation scenario into ``scenario_dir`` (.mp4 + GT PNGs + LabelMe JSONs)."""
    gt_dir = scenario_dir / "ground_truth"
    images_dir = gt_dir / "images"
    masks_dir = gt_dir / "masks"
    labelme_dir = gt_dir / "labelme"
    seed_dir = scenario_dir / "seed_template"
    for d in (images_dir, masks_dir, labelme_dir, seed_dir):
        d.mkdir(parents=True, exist_ok=True)

    query = urlencode(
        {
            "dataset": 1,
            "frame": 0,
            "fps": spec.fps,
            "seed": spec.seed,
            "camera_view": spec.camera_view,
            "scenario": spec.scenario,
        }
    )
    page.goto(f"{base_url}/{SCENE_PATH}?{query}", wait_until="load", timeout=120_000)
    page.wait_for_selector("#c[data-dataset-ready='true']", timeout=120_000)

    video_path = scenario_dir / f"{spec.name}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(spec.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")

    frame_records: List[Dict[str, Any]] = []
    keypoint_names: List[str] = []
    category_meta: Dict[str, Any] = {}
    try:
        for frame_idx in range(spec.frame_count):
            payload = page.evaluate(
                "idx => window.annolidPoseDataset.setFrame(idx)", frame_idx
            )
            seg_payload = page.evaluate(
                "() => window.annolidPoseDataset.captureSegmentationMasks()"
            )
            png_bytes = page.screenshot(type="png")
            bgr = cv2.imdecode(
                np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is None or bgr.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Unexpected frame dimensions for {spec.name}#{frame_idx}"
                )
            writer.write(bgr)

            img_name = f"frame_{frame_idx:06d}.jpg"
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            track_map, fine_map, coarse_map = decode_segmentation_rle(seg_payload)
            id_bgr = np.stack([coarse_map, fine_map, track_map], axis=-1)
            mask_name = f"frame_{frame_idx:06d}_ids.png"
            cv2.imwrite(str(masks_dir / mask_name), id_bgr)

            if not keypoint_names:
                category_meta = dict(payload["category"])
                keypoint_names = list(category_meta["keypoints"])

            lm_payload = build_labelme_payload(
                image_filename=f"../images/{img_name}",
                width=width,
                height=height,
                annotations=payload["annotations"],
                keypoint_names=keypoint_names,
                track_map=track_map,
                coarse_map=coarse_map,
                include_masks=True,
                include_gt_body_parts=True,
            )
            (labelme_dir / f"frame_{frame_idx:06d}.json").write_text(
                json.dumps(lm_payload, indent=2) + "\n", encoding="utf-8"
            )

            if frame_idx == 0:
                seed_png_name = f"{spec.name}_000000000.png"
                (seed_dir / seed_png_name).write_bytes(png_bytes)
                seed_lm = build_labelme_payload(
                    image_filename=seed_png_name,
                    width=width,
                    height=height,
                    annotations=payload["annotations"],
                    keypoint_names=keypoint_names,
                    track_map=track_map,
                    coarse_map=coarse_map,
                    include_masks=True,
                    include_gt_body_parts=False,
                )
                (seed_dir / f"{spec.name}_000000000.json").write_text(
                    json.dumps(seed_lm, indent=2) + "\n", encoding="utf-8"
                )
                seed_maskless = build_labelme_payload(
                    image_filename=seed_png_name,
                    width=width,
                    height=height,
                    annotations=payload["annotations"],
                    keypoint_names=keypoint_names,
                    track_map=track_map,
                    coarse_map=coarse_map,
                    include_masks=False,
                    include_gt_body_parts=False,
                )
                (seed_dir / f"{spec.name}_000000000_maskless.json").write_text(
                    json.dumps(seed_maskless, indent=2) + "\n", encoding="utf-8"
                )

            updated_anns = []
            for ann in payload["annotations"]:
                tid = int(ann["track_id"])
                raw_kps = list(ann["keypoints"])
                new_kps: List[float | int] = []
                vis_count = 0
                ys, xs = np.nonzero(track_map == tid)
                if xs.size > 0:
                    x0, y0 = float(xs.min()), float(ys.min())
                    bw, bh = (
                        float(xs.max() - xs.min() + 1),
                        float(ys.max() - ys.min() + 1),
                    )
                    bbox = [x0, y0, bw, bh]
                else:
                    bbox = list(ann.get("bbox") or [0.0, 0.0, 0.0, 0.0])
                polys = mask_to_polygons(
                    track_map == tid,
                    epsilon=0.75,
                    min_area_px=12,
                    single_parent_contour=True,
                )
                seg_coco = [
                    [coord for pt in poly for coord in pt] for poly in polys if poly
                ]
                for kp_i, kp_name in enumerate(keypoint_names):
                    kx = float(raw_kps[kp_i * 3])
                    ky = float(raw_kps[kp_i * 3 + 1])
                    rv = int(raw_kps[kp_i * 3 + 2])
                    sv = classify_keypoint_surface_visibility(
                        kp_name,
                        kx,
                        ky,
                        rv,
                        track_id=tid,
                        track_map=track_map,
                        coarse_map=coarse_map,
                    )
                    if sv > 0:
                        vis_count += 1
                    new_kps.extend([round(kx, 2), round(ky, 2), int(sv)])
                ann_copy = dict(ann)
                ann_copy["keypoints"] = new_kps
                ann_copy["num_keypoints"] = vis_count
                ann_copy["bbox"] = [round(float(v), 2) for v in bbox]
                ann_copy["area"] = int(np.count_nonzero(track_map == tid))
                ann_copy["pixel_area"] = int(np.count_nonzero(track_map == tid))
                if seg_coco:
                    ann_copy["segmentation"] = seg_coco
                updated_anns.append(ann_copy)

            frame_records.append(
                {
                    "frame_id": frame_idx,
                    "image_file": f"images/{img_name}",
                    "id_mask_file": f"masks/{mask_name}",
                    "labelme_file": f"labelme/frame_{frame_idx:06d}.json",
                    "telemetry": payload["image"],
                    "annotations": updated_anns,
                }
            )
    finally:
        writer.release()

    sequence_gt = {
        "schema_version": 1,
        "scenario": spec.name,
        "scene_scenario": spec.scenario,
        "description": spec.description,
        "camera_view": spec.camera_view,
        "seed": spec.seed,
        "fps": spec.fps,
        "width": width,
        "height": height,
        "frame_count": spec.frame_count,
        "video_file": f"{spec.name}.mp4",
        "keypoint_names": keypoint_names,
        "category": category_meta,
        "coarse_part_id_to_name": {str(k): v for k, v in COARSE_ID_TO_NAME.items()},
        "fine_part_id_to_name": {str(k): v for k, v in FINE_ID_TO_NAME.items()},
        "frames": frame_records,
    }
    (gt_dir / "sequence_gt.json").write_text(
        json.dumps(sequence_gt, indent=2) + "\n", encoding="utf-8"
    )
    return sequence_gt


def _export_unified_coco_dataset(
    output_dir: Path,
    sequences: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Export a combined Annolid COCO pose + tracking dataset across all scenarios."""
    coco_dir = output_dir / "coco_dataset"
    if coco_dir.exists():
        shutil.rmtree(coco_dir)
    ann_dir = coco_dir / "annotations"
    images_root = coco_dir / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    videos: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    train_image_ids: set[int] = set()
    val_image_ids: set[int] = set()
    categories: List[Dict[str, Any]] = []
    keypoint_names: List[str] = []

    next_image_id = 1
    next_ann_id = 1
    for vid_idx, (scenario_name, seq) in enumerate(sequences.items(), start=1):
        if not keypoint_names:
            keypoint_names = list(seq["keypoint_names"])
            cat = dict(seq.get("category") or {})
            cat.setdefault("id", 1)
            cat.setdefault("name", "mouse")
            cat.setdefault("supercategory", "animal")
            cat["keypoints"] = keypoint_names
            categories = [cat]

        frame_count = int(seq["frame_count"])
        split_idx = max(1, int(round(frame_count * 0.7)))
        videos.append(
            {
                "id": vid_idx,
                "name": scenario_name,
                "file_name": f"../{scenario_name}/{seq['video_file']}",
                "fps": int(seq["fps"]),
                "width": int(seq["width"]),
                "height": int(seq["height"]),
                "frame_count": frame_count,
                "camera_view": seq["camera_view"],
                "seed": int(seq["seed"]),
            }
        )
        scen_img_dir = images_root / scenario_name
        scen_img_dir.mkdir(parents=True, exist_ok=True)
        src_gt_dir = output_dir / scenario_name / "ground_truth"

        for fr in seq["frames"]:
            fidx = int(fr["frame_id"])
            src_img = src_gt_dir / str(fr["image_file"])
            dst_rel = f"images/{scenario_name}/frame_{fidx:06d}.jpg"
            shutil.copy2(src_img, coco_dir / dst_rel)

            img_id = next_image_id
            next_image_id += 1
            if fidx < split_idx:
                train_image_ids.add(img_id)
            else:
                val_image_ids.add(img_id)

            img_entry = dict(fr.get("telemetry") or {})
            img_entry.update(
                {
                    "id": img_id,
                    "video_id": vid_idx,
                    "frame_id": fidx,
                    "scenario": scenario_name,
                    "file_name": dst_rel,
                    "width": int(seq["width"]),
                    "height": int(seq["height"]),
                }
            )
            images.append(img_entry)

            for ann in fr["annotations"]:
                ann_entry = dict(ann)
                ann_entry["id"] = next_ann_id
                next_ann_id += 1
                ann_entry["image_id"] = img_id
                ann_entry["video_id"] = vid_idx
                ann_entry["frame_id"] = fidx
                ann_entry["category_id"] = 1
                ann_entry["iscrowd"] = 0
                annotations.append(ann_entry)

    base_info = {
        "description": "Annolid Two-Mice 3D Simulation Multi-Scenario Pose & Tracking Benchmark",
        "version": "1.0",
    }
    full_coco = {
        "info": base_info,
        "licenses": [],
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    def _subset_coco(target_ids: set[int]) -> Dict[str, Any]:
        return {
            "info": base_info,
            "licenses": [],
            "videos": videos,
            "images": [im for im in images if int(im["id"]) in target_ids],
            "annotations": [
                an for an in annotations if int(an["image_id"]) in target_ids
            ],
            "categories": categories,
        }

    (ann_dir / "sequence.json").write_text(
        json.dumps(full_coco, indent=2) + "\n", encoding="utf-8"
    )
    (ann_dir / "train.json").write_text(
        json.dumps(_subset_coco(train_image_ids), indent=2) + "\n", encoding="utf-8"
    )
    (ann_dir / "val.json").write_text(
        json.dumps(_subset_coco(val_image_ids), indent=2) + "\n", encoding="utf-8"
    )
    spec_lines = [
        "format: coco",
        "path: .",
        "image_root: .",
        "train: annotations/train.json",
        "val: annotations/val.json",
        f"kpt_shape: [{len(keypoint_names)}, 3]",
        "keypoint_names:",
        *[f"  - {name}" for name in keypoint_names],
        "",
    ]
    (coco_dir / "coco_spec.yaml").write_text("\n".join(spec_lines), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scenarios": list(sequences.keys()),
        "video_count": len(videos),
        "total_frames": len(images),
        "train_frames": len(train_image_ids),
        "val_frames": len(val_image_ids),
        "total_annotations": len(annotations),
        "keypoint_names": keypoint_names,
    }
    (coco_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_simulation_dataset(
    output_dir: Path,
    *,
    scenarios: Sequence[SimulationScenarioSpec] = DEFAULT_SCENARIOS,
    width: int = 640,
    height: int = 480,
) -> Dict[str, Dict[str, Any]]:
    """Render all simulation videos and pixel-exact 3D ground truth via Playwright Chromium."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required. Activate .venv before running."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    browser_errors: List[str] = []
    generated: Dict[str, Dict[str, Any]] = {}

    with _serve_repo() as base_url, sync_playwright() as playwright:
        browser = _launch_chromium(playwright)
        page = browser.new_page(
            viewport={"width": width, "height": height}, device_scale_factor=1
        )
        page.on(
            "console",
            lambda msg: browser_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda err: browser_errors.append(str(err)))

        for spec in scenarios:
            scenario_dir = output_dir / spec.name
            if scenario_dir.exists():
                shutil.rmtree(scenario_dir)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            generated[spec.name] = _capture_scenario_in_page(
                page, base_url, spec, scenario_dir, width=width, height=height
            )

        browser.close()

    if browser_errors:
        details = "\n".join(f"- {m}" for m in browser_errors)
        raise RuntimeError(f"Browser errors during simulation capture:\n{details}")

    _export_unified_coco_dataset(output_dir, generated)
    return generated


def load_existing_sequences(
    output_dir: Path,
    scenarios: Sequence[SimulationScenarioSpec] = DEFAULT_SCENARIOS,
) -> Dict[str, Dict[str, Any]]:
    loaded: Dict[str, Dict[str, Any]] = {}
    for spec in scenarios:
        seq_path = output_dir / spec.name / "ground_truth" / "sequence_gt.json"
        if not seq_path.exists():
            raise FileNotFoundError(f"Missing sequence_gt.json at {seq_path}")
        loaded[spec.name] = json.loads(seq_path.read_text(encoding="utf-8"))
    return loaded


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b)) / float(union)


def _extract_predictions_from_labelme_json(
    json_path: Path,
    width: int,
    height: int,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float, bool]]], Dict[str, np.ndarray]]:
    """Return ``(kp_by_inst, mask_by_inst)`` from a persisted LabelMe frame JSON."""
    adapter = AnnotationAdapter(image_height=height, image_width=width)
    registry = adapter.read_annotation(json_path)
    kp_by_inst: Dict[str, Dict[str, Tuple[float, float, bool]]] = {}
    mask_by_inst: Dict[str, np.ndarray] = {}
    for inst in registry:
        if inst.mask_bitmap is not None:
            mask_by_inst[inst.label] = inst.mask_bitmap.astype(bool)
        elif inst.polygon:
            mask_by_inst[inst.label] = adapter.mask_bitmap_from_polygon(inst.polygon)
        inst_kps: Dict[str, Tuple[float, float, bool]] = {}
        for kp in inst.keypoints.values():
            inst_kps[kp.label] = (float(kp.x), float(kp.y), bool(kp.visible))
        kp_by_inst[inst.label] = inst_kps
    return kp_by_inst, mask_by_inst


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _run_video_processor_mode(
    *,
    scenario_dir: Path,
    scenario_name: str,
    mode_name: str,
    model_name: str,
    config: CutieDinoTrackerConfig,
    device: Optional[str] = None,
    maskless_seed: bool = False,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, Tuple[float, float, bool]]]],
    Dict[int, Dict[str, np.ndarray]],
    Path,
    float,
]:
    run_root = scenario_dir / "runs" / mode_name
    if run_root.exists():
        shutil.rmtree(run_root)
    result_dir = run_root / scenario_name
    result_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = scenario_dir / "seed_template"
    seed_png = seed_dir / f"{scenario_name}_000000000.png"
    seed_json = (
        seed_dir / f"{scenario_name}_000000000_maskless.json"
        if maskless_seed
        else seed_dir / f"{scenario_name}_000000000.json"
    )
    shutil.copy2(seed_png, result_dir / f"{scenario_name}_000000000.png")
    shutil.copy2(seed_json, result_dir / f"{scenario_name}_000000000.json")

    if device:
        config.cutie_device = device
    video_path = scenario_dir / f"{scenario_name}.mp4"
    t0 = time.perf_counter()
    proc: Optional[DinoKeypointVideoProcessor] = DinoKeypointVideoProcessor(
        str(video_path),
        result_folder=result_dir,
        model_name=model_name,
        device=device,
        runtime_config=config,
    )
    vid_w, vid_h = proc.video_width, proc.video_height
    try:
        proc.process_video(start_frame=0)
        elapsed = time.perf_counter() - t0
    finally:
        proc.mask_manager.reset_state()
        del proc
        _release_accelerator_memory()

    preds_by_frame: Dict[int, Dict[str, Dict[str, Tuple[float, float, bool]]]] = {}
    masks_by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for jf in sorted(result_dir.glob(f"{scenario_name}_*.json")):
        parts = jf.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        fidx = int(parts[1])
        kps, msks = _extract_predictions_from_labelme_json(jf, vid_w, vid_h)
        preds_by_frame[fidx] = kps
        masks_by_frame[fidx] = msks

    return preds_by_frame, masks_by_frame, result_dir, elapsed


def _run_oracle_mask_mode(
    *,
    scenario_dir: Path,
    scenario_name: str,
    seq_gt: Mapping[str, Any],
    mode_name: str,
    model_name: str,
    config: CutieDinoTrackerConfig,
    device: Optional[str] = None,
) -> Tuple[
    Dict[int, Dict[str, Dict[str, Tuple[float, float, bool]]]],
    Dict[int, Dict[str, np.ndarray]],
    Path,
    float,
]:
    run_root = scenario_dir / "runs" / mode_name
    if run_root.exists():
        shutil.rmtree(run_root)
    result_dir = run_root / scenario_name
    result_dir.mkdir(parents=True, exist_ok=True)

    width = int(seq_gt["width"])
    height = int(seq_gt["height"])
    seed_json = scenario_dir / "seed_template" / f"{scenario_name}_000000000.json"
    adapter = AnnotationAdapter(
        image_height=height,
        image_width=width,
        persist_json=True,
    )
    registry = adapter.read_annotation(seed_json)

    gt_dir = scenario_dir / "ground_truth"
    video_path = scenario_dir / f"{scenario_name}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    t0 = time.perf_counter()
    tracker = DinoKeypointTracker(
        model_name=model_name,
        device=device,
        runtime_config=config,
    )
    preds_by_frame: Dict[int, Dict[str, Dict[str, Tuple[float, float, bool]]]] = {}
    masks_by_frame: Dict[int, Dict[str, np.ndarray]] = {}

    try:
        for fr in seq_gt["frames"]:
            fidx = int(fr["frame_id"])
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Failed reading frame {fidx} from {video_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            id_bgr = cv2.imread(str(gt_dir / str(fr["id_mask_file"])), cv2.IMREAD_COLOR)
            track_map = id_bgr[:, :, 2]
            mask_lookup = {
                "mouse_1": (track_map == 1),
                "mouse_2": (track_map == 2),
            }
            if fidx == 0:
                shutil.copy2(
                    scenario_dir / "seed_template" / f"{scenario_name}_000000000.png",
                    result_dir / f"{scenario_name}_000000000.png",
                )
                shutil.copy2(
                    seed_json,
                    result_dir / f"{scenario_name}_000000000.json",
                )
                tracker.start(pil_img, registry, mask_lookup)
                kps0, msks0 = _extract_predictions_from_labelme_json(
                    seed_json, width, height
                )
                preds_by_frame[0] = kps0
                masks_by_frame[0] = mask_lookup
                continue

            results = tracker.update(pil_img, mask_lookup)
            for inst_lbl, m in mask_lookup.items():
                inst_state = registry.ensure_instance(inst_lbl)
                polys = mask_to_polygons(
                    m, epsilon=0.75, min_area_px=12, single_parent_contour=True
                )
                poly_tuples = (
                    [(float(px), float(py)) for px, py in polys[0]] if polys else []
                )
                inst_state.set_mask(bitmap=m, polygon=poly_tuples)
            registry.apply_tracker_results(results, frame_number=fidx)
            adapter.write_annotation(
                frame_number=fidx, registry=registry, output_dir=result_dir
            )

            inst_preds: Dict[str, Dict[str, Tuple[float, float, bool]]] = {}
            for inst in registry:
                inst_preds[inst.label] = {
                    kp.label: (float(kp.x), float(kp.y), bool(kp.visible))
                    for kp in inst.keypoints.values()
                }
            preds_by_frame[fidx] = inst_preds
            masks_by_frame[fidx] = mask_lookup
    finally:
        cap.release()
        del tracker
        _release_accelerator_memory()

    elapsed = time.perf_counter() - t0
    return preds_by_frame, masks_by_frame, result_dir, elapsed


def _summarize_error_list(errs: Sequence[float]) -> Dict[str, float]:
    if not errs:
        return {
            "count": 0,
            "mean_px": 0.0,
            "median_px": 0.0,
            "rmse_px": 0.0,
            "p90_px": 0.0,
            "pck_4px": 0.0,
            "pck_8px": 0.0,
            "pck_12px": 0.0,
        }
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_px": round(float(arr.mean()), 3),
        "median_px": round(float(np.median(arr)), 3),
        "rmse_px": round(float(np.sqrt(np.mean(arr * arr))), 3),
        "p90_px": round(float(np.percentile(arr, 90)), 3),
        "pck_4px": round(float(np.mean(arr <= 4.0) * 100.0), 2),
        "pck_8px": round(float(np.mean(arr <= 8.0) * 100.0), 2),
        "pck_12px": round(float(np.mean(arr <= 12.0) * 100.0), 2),
    }


def evaluate_predictions_against_gt(
    *,
    scenario_dir: Path,
    seq_gt: Mapping[str, Any],
    preds_by_frame: Mapping[int, Mapping[str, Mapping[str, Tuple[float, float, bool]]]],
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    result_dir: Path,
) -> Dict[str, Any]:
    """Compute quantitative pose, identity, occlusion, and mask metrics on frames 1..N-1."""
    gt_dir = scenario_dir / "ground_truth"
    keypoint_names: List[str] = list(seq_gt["keypoint_names"])
    width = int(seq_gt["width"])
    height = int(seq_gt["height"])

    # Identify keypoints occluded on frame 0 seed
    seed_vis_by_inst_kp: Dict[Tuple[str, str], int] = {}
    frame0_anns = seq_gt["frames"][0]["annotations"]
    for ann in frame0_anns:
        inst_lbl = f"mouse_{int(ann['track_id'])}"
        kps = ann["keypoints"]
        for idx, kp_name in enumerate(keypoint_names):
            seed_vis_by_inst_kp[(inst_lbl, kp_name)] = int(kps[idx * 3 + 2])

    vis_errors_all: List[float] = []
    vis_errors_if_detected: List[float] = []
    group_errors: Dict[str, List[float]] = {g: [] for g in KEYPOINT_PART_GROUPS}
    per_kp_errors: Dict[str, List[float]] = {k: [] for k in keypoint_names}
    occluded_seed_reacq_errors: List[float] = []

    gt_vis_total = 0
    gt_vis_detected = 0
    gt_occ_total = 0
    gt_occ_pred_visible = 0
    gt_occ_cross_bleed = 0

    pred_visible_total = 0
    cross_bleed_count = 0

    lr_pair_checks = 0
    lr_swap_count = 0
    ap_checks = 0
    ap_flip_count = 0

    mask_ious: List[float] = []

    kp_to_group: Dict[str, str] = {}
    for g_name, members in KEYPOINT_PART_GROUPS.items():
        for m in members:
            kp_to_group[m] = g_name

    for fr in seq_gt["frames"]:
        fidx = int(fr["frame_id"])
        if fidx == 0:
            continue  # Exclude seed frame 0

        id_bgr = cv2.imread(str(gt_dir / str(fr["id_mask_file"])), cv2.IMREAD_COLOR)
        track_map = id_bgr[:, :, 2]

        frame_preds = preds_by_frame.get(fidx, {})
        frame_masks = masks_by_frame.get(fidx, {})

        for ann in fr["annotations"]:
            tid = int(ann["track_id"])
            other_tid = 2 if tid == 1 else 1
            inst_lbl = f"mouse_{tid}"
            inst_preds = frame_preds.get(inst_lbl, {})

            gt_mask = track_map == tid
            if inst_lbl in frame_masks and np.any(gt_mask):
                mask_ious.append(compute_iou(frame_masks[inst_lbl], gt_mask))

            raw_kps = ann["keypoints"]
            gt_xy_vis: Dict[str, Tuple[float, float, int]] = {}
            for idx, kp_name in enumerate(keypoint_names):
                gx = float(raw_kps[idx * 3])
                gy = float(raw_kps[idx * 3 + 1])
                gv = int(raw_kps[idx * 3 + 2])
                gt_xy_vis[kp_name] = (gx, gy, gv)
                pred_tuple = inst_preds.get(kp_name)
                px = float(pred_tuple[0]) if pred_tuple is not None else float("nan")
                py = float(pred_tuple[1]) if pred_tuple is not None else float("nan")
                pvis = bool(pred_tuple[2]) if pred_tuple is not None else False

                in_other = False
                if math.isfinite(px) and math.isfinite(py):
                    ix, iy = int(round(px)), int(round(py))
                    if 0 <= ix < width and 0 <= iy < height:
                        in_other = bool(track_map[iy, ix] == other_tid)

                if pvis:
                    pred_visible_total += 1
                    if in_other:
                        cross_bleed_count += 1

                if gv == int(KeypointVisibility.VISIBLE):
                    gt_vis_total += 1
                    if (
                        pred_tuple is not None
                        and math.isfinite(px)
                        and math.isfinite(py)
                    ):
                        err = math.hypot(px - gx, py - gy)
                        # Penalize missed detections cleanly in recall while also tracking raw coordinate error
                        vis_errors_all.append(err)
                        group_errors[kp_to_group[kp_name]].append(err)
                        per_kp_errors[kp_name].append(err)
                        if pvis:
                            gt_vis_detected += 1
                            vis_errors_if_detected.append(err)
                        if seed_vis_by_inst_kp.get((inst_lbl, kp_name)) == int(
                            KeypointVisibility.OCCLUDED
                        ):
                            occluded_seed_reacq_errors.append(err)
                elif gv == int(KeypointVisibility.OCCLUDED):
                    gt_occ_total += 1
                    if pvis:
                        gt_occ_pred_visible += 1
                        if in_other:
                            gt_occ_cross_bleed += 1

            # Check L/R swaps when both GT landmarks are visible and separated by >= 5px
            for left_kp, right_kp in LR_SYMMETRY_PAIRS:
                lx, ly, lv = gt_xy_vis[left_kp]
                rx, ry, rv = gt_xy_vis[right_kp]
                lp = inst_preds.get(left_kp)
                rp = inst_preds.get(right_kp)
                if (
                    lv == int(KeypointVisibility.VISIBLE)
                    and rv == int(KeypointVisibility.VISIBLE)
                    and lp is not None
                    and rp is not None
                    and math.hypot(lx - rx, ly - ry) >= 5.0
                ):
                    lr_pair_checks += 1
                    direct = math.hypot(lp[0] - lx, lp[1] - ly) + math.hypot(
                        rp[0] - rx, rp[1] - ry
                    )
                    swapped = math.hypot(lp[0] - rx, lp[1] - ry) + math.hypot(
                        rp[0] - lx, rp[1] - ly
                    )
                    if swapped + 2.0 < direct:
                        lr_swap_count += 1

            # Check head-tail (nose vs tail_base) flip
            nx, ny, nv = gt_xy_vis["nose"]
            tx, ty, tv = gt_xy_vis["tail_base"]
            np_pred = inst_preds.get("nose")
            tp_pred = inst_preds.get("tail_base")
            if (
                nv == int(KeypointVisibility.VISIBLE)
                and tv == int(KeypointVisibility.VISIBLE)
                and np_pred is not None
                and tp_pred is not None
                and math.hypot(nx - tx, ny - ty) >= 12.0
            ):
                ap_checks += 1
                direct = math.hypot(np_pred[0] - nx, np_pred[1] - ny) + math.hypot(
                    tp_pred[0] - tx, tp_pred[1] - ty
                )
                flipped = math.hypot(np_pred[0] - tx, np_pred[1] - ty) + math.hypot(
                    tp_pred[0] - nx, tp_pred[1] - ny
                )
                if flipped + 4.0 < direct:
                    ap_flip_count += 1

    qc_report = evaluate_track_qc(str(result_dir))
    qc_summary = qc_report.get("summary", {}) if isinstance(qc_report, Mapping) else {}

    return {
        "overall_visible": _summarize_error_list(vis_errors_all),
        "detected_visible": _summarize_error_list(vis_errors_if_detected),
        "visible_recall_pct": round(100.0 * gt_vis_detected / max(1, gt_vis_total), 2),
        "part_groups": {
            g: _summarize_error_list(errs) for g, errs in group_errors.items()
        },
        "per_keypoint_mean_px": {
            k: round(float(np.mean(v)), 3) if v else 0.0
            for k, v in per_kp_errors.items()
        },
        "occluded_seed_reacquisition": _summarize_error_list(
            occluded_seed_reacq_errors
        ),
        "occlusion": {
            "gt_occluded_total": gt_occ_total,
            "pred_visible_when_occluded": gt_occ_pred_visible,
            "pred_visible_when_occluded_pct": round(
                100.0 * gt_occ_pred_visible / max(1, gt_occ_total), 2
            ),
            "occluded_cross_instance_bleed": gt_occ_cross_bleed,
        },
        "identity": {
            "pred_visible_total": pred_visible_total,
            "cross_instance_bleed_count": cross_bleed_count,
            "cross_instance_bleed_pct": round(
                100.0 * cross_bleed_count / max(1, pred_visible_total), 2
            ),
            "lr_pair_checks": lr_pair_checks,
            "lr_swap_count": lr_swap_count,
            "lr_swap_pct": round(100.0 * lr_swap_count / max(1, lr_pair_checks), 2),
            "ap_checks": ap_checks,
            "ap_flip_count": ap_flip_count,
            "ap_flip_pct": round(100.0 * ap_flip_count / max(1, ap_checks), 2),
        },
        "mask_mean_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else None,
        "track_qc": {
            "flagged_frame_count": int(qc_summary.get("flagged_frame_count", 0) or 0),
            "total_flags": int(qc_summary.get("total_flags", 0) or 0),
        },
    }


def render_diagnostic_comparison_video(
    *,
    scenario_dir: Path,
    seq_gt: Mapping[str, Any],
    preds_by_frame: Mapping[int, Mapping[str, Mapping[str, Tuple[float, float, bool]]]],
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    out_video_path: Path,
    mode_label: str,
) -> None:
    """Render side-by-side Ground Truth vs Tracked Keypoints + Error Vectors video."""
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(seq_gt["width"])
    height = int(seq_gt["height"])
    fps = float(seq_gt["fps"])
    keypoint_names: List[str] = list(seq_gt["keypoint_names"])

    src_video = scenario_dir / f"{seq_gt['scenario']}.mp4"
    cap = cv2.VideoCapture(str(src_video))
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    inst_colors = {
        "mouse_1": (80, 230, 255),
        "mouse_2": (130, 255, 120),
    }

    try:
        for fr in seq_gt["frames"]:
            fidx = int(fr["frame_id"])
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            left = bgr.copy()
            right = bgr.copy()

            frame_preds = preds_by_frame.get(fidx, {})
            frame_masks = masks_by_frame.get(fidx, {})

            # Draw predicted mask contours on right panel
            for inst_lbl, m in sorted(frame_masks.items()):
                cnts, _ = cv2.findContours(
                    m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                col = inst_colors.get(inst_lbl, (200, 200, 200))
                cv2.drawContours(right, cnts, -1, col, 1, lineType=cv2.LINE_AA)

            frame_errs: List[float] = []
            for ann in fr["annotations"]:
                tid = int(ann["track_id"])
                inst_lbl = f"mouse_{tid}"
                col = inst_colors.get(inst_lbl, (220, 220, 220))
                raw_kps = ann["keypoints"]
                gt_pts: Dict[str, Tuple[float, float, int]] = {}
                for idx, kp_name in enumerate(keypoint_names):
                    gt_pts[kp_name] = (
                        float(raw_kps[idx * 3]),
                        float(raw_kps[idx * 3 + 1]),
                        int(raw_kps[idx * 3 + 2]),
                    )

                for a, b in SKELETON_EDGES:
                    ax, ay, av = gt_pts[a]
                    bx, by, bv = gt_pts[b]
                    if av > 0 and bv > 0:
                        cv2.line(
                            left,
                            (int(round(ax)), int(round(ay))),
                            (int(round(bx)), int(round(by))),
                            col,
                            1,
                            cv2.LINE_AA,
                        )
                for kp_name, (gx, gy, gv) in gt_pts.items():
                    if gv == int(KeypointVisibility.VISIBLE):
                        cv2.circle(
                            left,
                            (int(round(gx)), int(round(gy))),
                            3,
                            col,
                            -1,
                            cv2.LINE_AA,
                        )
                    elif gv == int(KeypointVisibility.OCCLUDED):
                        cv2.circle(
                            left,
                            (int(round(gx)), int(round(gy))),
                            3,
                            (60, 140, 255),
                            1,
                            cv2.LINE_AA,
                        )

                inst_preds = frame_preds.get(inst_lbl, {})
                for a, b in SKELETON_EDGES:
                    pa = inst_preds.get(a)
                    pb = inst_preds.get(b)
                    if pa is not None and pb is not None and pa[2] and pb[2]:
                        cv2.line(
                            right,
                            (int(round(pa[0])), int(round(pa[1]))),
                            (int(round(pb[0])), int(round(pb[1]))),
                            col,
                            1,
                            cv2.LINE_AA,
                        )

                for kp_name, (gx, gy, gv) in gt_pts.items():
                    pr = inst_preds.get(kp_name)
                    if pr is None:
                        continue
                    px, py, pvis = pr
                    if not (math.isfinite(px) and math.isfinite(py)):
                        continue
                    if gv == int(KeypointVisibility.VISIBLE):
                        err = math.hypot(px - gx, py - gy)
                        frame_errs.append(err)
                        if err > 4.0:
                            cv2.line(
                                right,
                                (int(round(gx)), int(round(gy))),
                                (int(round(px)), int(round(py))),
                                (60, 70, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    if pvis:
                        cv2.circle(
                            right,
                            (int(round(px)), int(round(py))),
                            3,
                            col,
                            -1,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.circle(
                            right,
                            (int(round(px)), int(round(py))),
                            3,
                            (140, 140, 140),
                            1,
                            cv2.LINE_AA,
                        )

            mean_e = float(np.mean(frame_errs)) if frame_errs else 0.0
            cv2.rectangle(left, (0, 0), (width, 26), (22, 22, 22), -1)
            cv2.putText(
                left,
                f"3D Ground Truth | {seq_gt['scenario']} frame {fidx}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(right, (0, 0), (width, 26), (22, 22, 22), -1)
            tag = "SEED" if fidx == 0 else f"err={mean_e:.2f}px"
            cv2.putText(
                right,
                f"Tracked ({mode_label}) | {tag}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            writer.write(np.hstack([left, right]))
    finally:
        cap.release()
        writer.release()


def run_benchmark(
    output_dir: Path,
    *,
    sequences: Mapping[str, Mapping[str, Any]],
    model_name: str = DEFAULT_DINO_FEATURE_MODEL_ID,
    device: Optional[str] = None,
    modes: Sequence[str] = (
        "cutie_dinov3_default",
        "cutie_dinov3_rodent",
        "oracle_mask_rodent",
        "maskless_rodent",
    ),
    video_tag: str = "eval",
) -> Dict[str, Any]:
    """Run all requested keypoint tracking modes across all generated scenarios."""
    mode_results: Dict[str, Any] = {}

    for mode in modes:
        per_scenario: Dict[str, Any] = {}
        agg_vis_errs: List[float] = []
        agg_group_errs: Dict[str, List[float]] = {g: [] for g in KEYPOINT_PART_GROUPS}
        agg_occ_seed_errs: List[float] = []
        agg_gt_vis = 0
        agg_gt_det = 0
        agg_pred_vis = 0
        agg_bleed = 0
        agg_lr_checks = 0
        agg_lr_swaps = 0
        agg_ap_checks = 0
        agg_ap_flips = 0
        agg_gt_occ = 0
        agg_occ_pred_vis = 0
        agg_mask_ious: List[float] = []
        oom_crashes: List[str] = []
        total_elapsed = 0.0

        for scen_name, seq_gt in sequences.items():
            scenario_dir = output_dir / scen_name
            try:
                if mode == "cutie_dinov3_default":
                    cfg = CutieDinoTrackerConfig(
                        use_cutie_tracking=True,
                        persist_labelme_json=True,
                    )
                    preds, masks, res_dir, elapsed = _run_video_processor_mode(
                        scenario_dir=scenario_dir,
                        scenario_name=scen_name,
                        mode_name=f"{video_tag}_{mode}",
                        model_name=model_name,
                        config=cfg,
                        device=device,
                        maskless_seed=False,
                    )
                elif mode == "cutie_dinov3_rodent":
                    cfg = CutieDinoTrackerConfig.from_preset(
                        "rodent_30fps_occlusions",
                        use_cutie_tracking=True,
                        persist_labelme_json=True,
                        tracking_enable=True,
                        tracking_matcher_algorithm="hungarian",
                    )
                    preds, masks, res_dir, elapsed = _run_video_processor_mode(
                        scenario_dir=scenario_dir,
                        scenario_name=scen_name,
                        mode_name=f"{video_tag}_{mode}",
                        model_name=model_name,
                        config=cfg,
                        device=device,
                        maskless_seed=False,
                    )
                elif mode == "oracle_mask_rodent":
                    cfg = CutieDinoTrackerConfig.from_preset(
                        "rodent_30fps_occlusions",
                        use_cutie_tracking=False,
                        persist_labelme_json=True,
                        tracking_enable=True,
                        tracking_matcher_algorithm="hungarian",
                    )
                    preds, masks, res_dir, elapsed = _run_oracle_mask_mode(
                        scenario_dir=scenario_dir,
                        scenario_name=scen_name,
                        seq_gt=seq_gt,
                        mode_name=f"{video_tag}_{mode}",
                        model_name=model_name,
                        config=cfg,
                        device=device,
                    )
                elif mode == "maskless_rodent":
                    cfg = CutieDinoTrackerConfig.from_preset(
                        "rodent_30fps_occlusions",
                        use_cutie_tracking=False,
                        mask_enforce_position=False,
                        mask_enforce_reject_outside=False,
                        support_probe_mask_only=False,
                        persist_labelme_json=True,
                        tracking_enable=True,
                        tracking_matcher_algorithm="hungarian",
                    )
                    preds, masks, res_dir, elapsed = _run_video_processor_mode(
                        scenario_dir=scenario_dir,
                        scenario_name=scen_name,
                        mode_name=f"{video_tag}_{mode}",
                        model_name=model_name,
                        config=cfg,
                        device=device,
                        maskless_seed=True,
                    )
                else:
                    raise ValueError(f"Unknown benchmark mode: {mode}")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _release_accelerator_memory()
                    oom_crashes.append(f"{scen_name}: {exc}")
                    per_scenario[scen_name] = {
                        "status": "oom_crash",
                        "error": str(exc),
                    }
                    print(f"[OOM CRASH] mode={mode} scenario={scen_name}: {exc}")
                    continue
                raise

            metrics = evaluate_predictions_against_gt(
                scenario_dir=scenario_dir,
                seq_gt=seq_gt,
                preds_by_frame=preds,
                masks_by_frame=masks,
                result_dir=res_dir,
            )
            metrics["elapsed_sec"] = round(elapsed, 2)
            per_scenario[scen_name] = metrics
            total_elapsed += elapsed

            if mode in ("cutie_dinov3_rodent", "cutie_dinov3_default"):
                diag_video = (
                    scenario_dir / "diagnostics" / f"{scen_name}_{video_tag}_{mode}.mp4"
                )
                render_diagnostic_comparison_video(
                    scenario_dir=scenario_dir,
                    seq_gt=seq_gt,
                    preds_by_frame=preds,
                    masks_by_frame=masks,
                    out_video_path=diag_video,
                    mode_label=f"{video_tag}:{mode}",
                )

            # Collect raw per-frame errors for exact aggregate statistics
            kp_names = list(seq_gt["keypoint_names"])
            kp_to_group = {
                m: g for g, members in KEYPOINT_PART_GROUPS.items() for m in members
            }
            seed_vis = {}
            for ann0 in seq_gt["frames"][0]["annotations"]:
                il0 = f"mouse_{int(ann0['track_id'])}"
                for i0, kn0 in enumerate(kp_names):
                    seed_vis[(il0, kn0)] = int(ann0["keypoints"][i0 * 3 + 2])

            for fr in seq_gt["frames"]:
                fidx = int(fr["frame_id"])
                if fidx == 0:
                    continue
                for ann in fr["annotations"]:
                    il = f"mouse_{int(ann['track_id'])}"
                    ip = preds.get(fidx, {}).get(il, {})
                    rk = ann["keypoints"]
                    for idx, kn in enumerate(kp_names):
                        gx, gy, gv = (
                            float(rk[idx * 3]),
                            float(rk[idx * 3 + 1]),
                            int(rk[idx * 3 + 2]),
                        )
                        pt = ip.get(kn)
                        if gv == int(KeypointVisibility.VISIBLE):
                            agg_gt_vis += 1
                            if (
                                pt is not None
                                and math.isfinite(pt[0])
                                and math.isfinite(pt[1])
                            ):
                                e = math.hypot(pt[0] - gx, pt[1] - gy)
                                agg_vis_errs.append(e)
                                agg_group_errs[kp_to_group[kn]].append(e)
                                if pt[2]:
                                    agg_gt_det += 1
                                if seed_vis.get((il, kn)) == int(
                                    KeypointVisibility.OCCLUDED
                                ):
                                    agg_occ_seed_errs.append(e)

            agg_pred_vis += int(metrics["identity"]["pred_visible_total"])
            agg_bleed += int(metrics["identity"]["cross_instance_bleed_count"])
            agg_lr_checks += int(metrics["identity"]["lr_pair_checks"])
            agg_lr_swaps += int(metrics["identity"]["lr_swap_count"])
            agg_ap_checks += int(metrics["identity"]["ap_checks"])
            agg_ap_flips += int(metrics["identity"]["ap_flip_count"])
            agg_gt_occ += int(metrics["occlusion"]["gt_occluded_total"])
            agg_occ_pred_vis += int(metrics["occlusion"]["pred_visible_when_occluded"])
            if metrics["mask_mean_iou"] is not None:
                agg_mask_ious.append(float(metrics["mask_mean_iou"]))

        mode_results[mode] = {
            "aggregate": {
                "overall_visible": _summarize_error_list(agg_vis_errs),
                "visible_recall_pct": round(100.0 * agg_gt_det / max(1, agg_gt_vis), 2),
                "part_groups": {
                    g: _summarize_error_list(errs) for g, errs in agg_group_errs.items()
                },
                "occluded_seed_reacquisition": _summarize_error_list(agg_occ_seed_errs),
                "cross_instance_bleed_pct": round(
                    100.0 * agg_bleed / max(1, agg_pred_vis), 2
                ),
                "lr_swap_pct": round(100.0 * agg_lr_swaps / max(1, agg_lr_checks), 2),
                "ap_flip_pct": round(100.0 * agg_ap_flips / max(1, agg_ap_checks), 2),
                "pred_visible_when_occluded_pct": round(
                    100.0 * agg_occ_pred_vis / max(1, agg_gt_occ), 2
                ),
                "mask_mean_iou": (
                    round(float(np.mean(agg_mask_ious)), 4) if agg_mask_ious else None
                ),
                "oom_crash_count": len(oom_crashes),
                "oom_crashes": oom_crashes,
                "total_elapsed_sec": round(total_elapsed, 2),
            },
            "scenarios": per_scenario,
        }

    return {
        "schema_version": 1,
        "model_name": model_name,
        "video_tag": video_tag,
        "modes": mode_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated two-mice simulation videos, ground truth, and tracking runs.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing generated videos and sequence_gt.json files.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate simulation videos and ground-truth dataset without running tracking.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_DINO_FEATURE_MODEL_ID,
        help="Vision backbone model ID for DinoKeypointTracker.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional compute device override ('cpu', 'mps', 'cuda').",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "cutie_dinov3_default",
            "cutie_dinov3_rodent",
            "oracle_mask_rodent",
            "maskless_rodent",
        ],
        help="Tracking modes to evaluate.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="eval",
        help="Run tag prefix for saved run folders and diagnostic videos.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional explicit JSON path for the evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if args.skip_generate:
        sequences = load_existing_sequences(output_dir)
    else:
        sequences = generate_simulation_dataset(output_dir)
    if args.generate_only:
        print(
            f"Generated {len(sequences)} simulation videos and datasets in {output_dir}"
        )
        return

    report = run_benchmark(
        output_dir,
        sequences=sequences,
        model_name=args.model,
        device=args.device,
        modes=args.modes,
        video_tag=args.tag,
    )
    report_path = args.report_out or (output_dir / f"report_{args.tag}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Two-Mice Keypoint Tracking Benchmark ({args.tag}) ===")
    for mode_name, mdata in report["modes"].items():
        agg = mdata["aggregate"]
        ov = agg["overall_visible"]
        pg = agg["part_groups"]
        print(
            f"[{mode_name}] "
            f"mean={ov['mean_px']:.2f}px med={ov['median_px']:.2f}px rmse={ov['rmse_px']:.2f}px | "
            f"PCK@4={ov['pck_4px']:.1f}% PCK@8={ov['pck_8px']:.1f}% PCK@12={ov['pck_12px']:.1f}% | "
            f"head={pg['head']['mean_px']:.2f}px spine={pg['spine']['mean_px']:.2f}px "
            f"paws={pg['paws']['mean_px']:.2f}px tail={pg['tail']['mean_px']:.2f}px | "
            f"bleed={agg['cross_instance_bleed_pct']:.2f}% swap={agg['lr_swap_pct']:.2f}% "
            f"flip={agg['ap_flip_pct']:.2f}% recall={agg['visible_recall_pct']:.1f}% "
            f"maskIoU={agg['mask_mean_iou']}"
        )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
