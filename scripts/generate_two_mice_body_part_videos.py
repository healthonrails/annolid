#!/usr/bin/env python3
"""Generate two-mouse 3D simulation videos + pixel-exact ground truth and evaluate body-part segmentation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

import cv2
import numpy as np

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.segmentation.body_parts import (
    BodyPartSegmentationConfig,
    InstanceBodyPartSegmenter,
)
from annolid.services.body_part_segmentation import run_body_part_segmentation
from annolid.utils.body_part_shapes import (
    build_body_part_shape_dict,
    infer_part_from_keypoint_name,
    is_body_part_shape_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = "annolid/gui/assets/threejs/two_mice.html"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "two_mice_body_part_eval"
DEFAULT_FIXTURE_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "two_mice_body_part_tiny"

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

# Mapping from 3D fine mesh IDs to Annolid's keypoint_granularity="fine" canonical labels.
FINE_ID_TO_KEYPOINT_FINE_LABEL: Dict[int, str] = {
    1: "head",
    2: "head",
    3: "head",
    4: "head",
    5: "torso",
    6: "torso",
    7: "left_forelimb",
    8: "right_forelimb",
    9: "left_hindlimb",
    10: "right_hindlimb",
    11: "tail",
}

COARSE_ID_TO_NAME: Dict[int, str] = {
    1: "head",
    2: "torso",
    3: "limbs",
    4: "tail",
}
COARSE_NAME_TO_ID: Dict[str, int] = {v: k for k, v in COARSE_ID_TO_NAME.items()}

PART_VIS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "head": (70, 90, 245),
    "snout": (120, 80, 255),
    "left_ear": (90, 140, 255),
    "right_ear": (50, 165, 255),
    "neck": (160, 210, 90),
    "torso": (95, 205, 85),
    "limbs": (240, 185, 55),
    "left_forelimb": (250, 205, 65),
    "right_forelimb": (235, 160, 45),
    "left_hindlimb": (215, 130, 85),
    "right_hindlimb": (195, 110, 120),
    "tail": (215, 85, 210),
}


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
    coat_pair: str = ""
    lighting: str = ""
    occluder: str = ""
    arena: str = ""
    headstage: str = ""
    width_override: Optional[int] = None
    height_override: Optional[int] = None


DEFAULT_SCENARIOS: Tuple[SimulationScenarioSpec, ...] = (
    SimulationScenarioSpec(
        name="arena_explore",
        description="Overhead arena exploration with C57BL/6 (black) and BALB/c (white) mice turning and sniffing.",
        camera_view="top-close",
        scenario="arena_explore",
        seed=20260710,
        fps=8,
        frame_count=12,
    ),
    SimulationScenarioSpec(
        name="social_huddle",
        description="Close side-by-side approach, flank contact/huddling, and pass-through.",
        camera_view="top-close",
        scenario="social_huddle",
        seed=20260711,
        fps=8,
        frame_count=14,
    ),
    SimulationScenarioSpec(
        name="curled_grooming",
        description="Curled C-shaped flank-grooming posture on mouse_1 while mouse_2 explores nearby.",
        camera_view="top-close",
        scenario="curled_groom",
        seed=20260712,
        fps=8,
        frame_count=10,
    ),
    SimulationScenarioSpec(
        name="rearing_oblique",
        description="Oblique 3D studio angle with upright rearing exposing head, torso, fore/hindlimbs, and tail.",
        camera_view="oblique-close",
        scenario="rearing_social",
        seed=20260713,
        fps=8,
        frame_count=10,
    ),
)

DOMAIN_RANDOMIZATION_SCENARIOS: Tuple[SimulationScenarioSpec, ...] = (
    SimulationScenarioSpec(
        name="same_strain_c57_huddle",
        description="Two identical black C57BL/6 mice huddling with zero coat-albedo contrast.",
        camera_view="top-close",
        scenario="social_huddle",
        seed=20260714,
        fps=8,
        frame_count=10,
        coat_pair="c57_pair",
    ),
    SimulationScenarioSpec(
        name="ir_night_explore",
        description="Dim red/IR night-vision behavioral arena illumination.",
        camera_view="top-close",
        scenario="arena_explore",
        seed=20260715,
        fps=8,
        frame_count=10,
        lighting="ir_night",
    ),
    SimulationScenarioSpec(
        name="ventral_glass_gait",
        description="Bottom-up glass-floor gait imaging exposing all four paws, snout, and tail base.",
        camera_view="bottom-up",
        scenario="ventral_gait",
        seed=20260716,
        fps=8,
        frame_count=10,
    ),
    SimulationScenarioSpec(
        name="wire_bar_occluded",
        description="Overhead stainless-steel cage-lid bars casting partial 3D depth occlusion across subjects.",
        camera_view="top-close",
        scenario="arena_explore",
        seed=20260717,
        fps=8,
        frame_count=10,
        occluder="wire_bars",
    ),
    SimulationScenarioSpec(
        name="calms_resident_intruder",
        description="CalMS21-style monochrome overhead shoebox home-cage with corncob bedding, shredded nestlet, water port, cranial neural headstage + tether, and directed resident-intruder social pursuit.",
        camera_view="home-cage-top",
        scenario="calms_resident_intruder",
        seed=20260718,
        fps=10,
        frame_count=14,
        lighting="calms_gray",
        arena="home_cage",
        headstage="resident",
        width_override=768,
        height_override=432,
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

    # Seal 1-2px guard-hair micro-gaps inside each mouse silhouette without crossing into neighbors.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    unique_tracks = [int(t) for t in np.unique(track_map) if int(t) > 0]
    closed_track = track_map.copy()
    for tid in unique_tracks:
        raw_m = track_map == tid
        other_m = (track_map > 0) & (track_map != tid)
        closed_m = (
            cv2.morphologyEx(raw_m.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        ) & (~other_m)
        # Keep connected components of at least 8 px so isolated 1px hair specks are dropped.
        n_cc, cc_lbl, stats, _ = cv2.connectedComponentsWithStats(
            closed_m.astype(np.uint8), connectivity=8
        )
        clean_m = np.zeros_like(closed_m)
        for cc_i in range(1, n_cc):
            if int(stats[cc_i, cv2.CC_STAT_AREA]) >= 8:
                clean_m |= cc_lbl == cc_i

        # Remove pruned specks that belonged to tid
        dropped = raw_m & (~clean_m)
        closed_track[dropped] = 0
        fine_map[dropped] = 0
        coarse_map[dropped] = 0

        added = clean_m & (~raw_m) & (closed_track == 0)
        if np.any(added) and np.any(raw_m & clean_m):
            # Nearest valid foreground label fill via distanceTransformWithLabels
            inv = (~(raw_m & clean_m)).astype(np.uint8)
            _dist, nearest_idx = cv2.distanceTransformWithLabels(
                inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
            )
            # Map each distanceTransform pixel label (1-based over zero pixels of inv) back to coords
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
        # tail_base sits right on the sacral torso/tail junction.
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
        # Bridge any 1-2px tail gap if present so an instance has one coherent outer polygon.
        n_cc, _ = cv2.connectedComponents(m_u8, connectivity=8)
        if n_cc > 2:
            bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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


def build_ground_truth_labelme_payload(
    *,
    image_filename: str,
    width: int,
    height: int,
    annotations: Sequence[Mapping[str, Any]],
    keypoint_names: Sequence[str],
    track_map: np.ndarray,
    coarse_map: np.ndarray,
    include_gt_body_parts: bool = True,
) -> Dict[str, Any]:
    """Construct a complete LabelMe JSON dictionary from simulation 3D ground truth."""
    shapes: List[Dict[str, Any]] = []
    for ann in sorted(annotations, key=lambda item: int(item["track_id"])):
        tid = int(ann["track_id"])
        inst_label = f"mouse_{tid}"
        inst_mask = track_map == tid
        # Bridge tiny 1-2px tail gaps, while retaining genuine multi-fragment splits (e.g. wire-bar occlusion).
        m_bridged = cv2.morphologyEx(
            inst_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        parent_polys = mask_to_polygons(
            m_bridged > 0, epsilon=0.65, min_area_px=16, single_parent_contour=False
        )
        if not parent_polys:
            continue

        for p_poly in parent_polys:
            shapes.append(
                {
                    "label": inst_label,
                    "points": p_poly,
                    "group_id": tid,
                    "shape_type": "polygon",
                    "flags": {},
                    "visible": True,
                    "description": str(ann.get("subject_name") or "simulated_mouse"),
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

        if include_gt_body_parts:
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


def _render_overlay_panel(
    bgr: np.ndarray,
    part_masks_by_inst: Mapping[str, Mapping[str, np.ndarray]],
    parent_masks_by_inst: Mapping[str, np.ndarray],
    title: str,
) -> np.ndarray:
    canvas = bgr.copy()
    overlay = bgr.copy()
    for inst_lbl, pmap in part_masks_by_inst.items():
        _ = inst_lbl
        for part_lbl, pmask in pmap.items():
            if not np.any(pmask):
                continue
            color = PART_VIS_COLORS_BGR.get(part_lbl, (180, 180, 180))
            overlay[pmask] = color

    canvas = cv2.addWeighted(overlay, 0.46, canvas, 0.54, 0.0)
    for idx, (inst_lbl, inst_mask) in enumerate(sorted(parent_masks_by_inst.items())):
        cnts, _ = cv2.findContours(
            inst_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        border_col = (255, 240, 90) if idx == 0 else (110, 255, 190)
        cv2.drawContours(canvas, cnts, -1, border_col, 1, lineType=cv2.LINE_AA)
        ys, xs = np.nonzero(inst_mask)
        if xs.size > 0:
            tx = int(np.clip(xs.min(), 4, canvas.shape[1] - 90))
            ty = int(np.clip(ys.min() - 6, 16, canvas.shape[0] - 8))
            cv2.putText(
                canvas,
                inst_lbl,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                border_col,
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (24, 24, 24), -1)
    cv2.putText(
        canvas,
        title,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (242, 242, 242),
        1,
        cv2.LINE_AA,
    )
    return canvas


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b)) / float(union)


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
    for d in (images_dir, masks_dir, labelme_dir):
        d.mkdir(parents=True, exist_ok=True)

    query_params: Dict[str, Any] = {
        "dataset": 1,
        "frame": 0,
        "fps": spec.fps,
        "seed": spec.seed,
        "camera_view": spec.camera_view,
        "scenario": spec.scenario,
    }
    if spec.coat_pair:
        query_params["coat_pair"] = spec.coat_pair
    if spec.lighting:
        query_params["lighting"] = spec.lighting
    if spec.occluder:
        query_params["occluder"] = spec.occluder
    if spec.arena:
        query_params["arena"] = spec.arena
    if spec.headstage:
        query_params["headstage"] = spec.headstage
    query = urlencode(query_params)
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
    try:
        for frame_idx in range(spec.frame_count):
            payload = page.evaluate(
                "idx => window.annolidPoseDataset.setFrame(idx)", frame_idx
            )
            seg_payload = page.evaluate(
                "() => window.annolidPoseDataset.captureSegmentationMasks()"
            )
            jpg_bytes = page.screenshot(type="jpeg", quality=92)
            bgr = cv2.imdecode(
                np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if bgr is None or bgr.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Unexpected frame dimensions for {spec.name}#{frame_idx}"
                )
            writer.write(bgr)

            img_name = f"frame_{frame_idx:06d}.jpg"
            img_path = images_dir / img_name
            img_path.write_bytes(jpg_bytes)

            track_map, fine_map, coarse_map = decode_segmentation_rle(seg_payload)
            # Lossless ID PNG: B = coarse_id, G = fine_id, R = track_id
            id_bgr = np.stack([coarse_map, fine_map, track_map], axis=-1)
            mask_name = f"frame_{frame_idx:06d}_ids.png"
            cv2.imwrite(str(masks_dir / mask_name), id_bgr)

            if not keypoint_names:
                keypoint_names = list(payload["category"]["keypoints"])

            lm_payload = build_ground_truth_labelme_payload(
                image_filename=f"../images/{img_name}",
                width=width,
                height=height,
                annotations=payload["annotations"],
                keypoint_names=keypoint_names,
                track_map=track_map,
                coarse_map=coarse_map,
                include_gt_body_parts=True,
            )
            (labelme_dir / f"frame_{frame_idx:06d}.json").write_text(
                json.dumps(lm_payload, indent=2) + "\n", encoding="utf-8"
            )

            # Update COCO annotation visibility with 3D surface occlusion labels
            updated_anns = []
            for ann in payload["annotations"]:
                tid = int(ann["track_id"])
                raw_kps = list(ann["keypoints"])
                new_kps: List[float | int] = []
                vis_count = 0
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
                    new_kps.extend([kx, ky, sv])
                ann_copy = dict(ann)
                ann_copy["keypoints"] = new_kps
                ann_copy["num_keypoints"] = vis_count
                ann_copy["pixel_area"] = int(np.count_nonzero(track_map == tid))
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
        "description": spec.description,
        "camera_view": spec.camera_view,
        "seed": spec.seed,
        "fps": spec.fps,
        "width": width,
        "height": height,
        "frame_count": spec.frame_count,
        "video_file": f"{spec.name}.mp4",
        "keypoint_names": keypoint_names,
        "coarse_part_id_to_name": {str(k): v for k, v in COARSE_ID_TO_NAME.items()},
        "fine_part_id_to_name": {str(k): v for k, v in FINE_ID_TO_NAME.items()},
        "frames": frame_records,
    }
    (gt_dir / "sequence_gt.json").write_text(
        json.dumps(sequence_gt, indent=2) + "\n", encoding="utf-8"
    )
    return sequence_gt


def generate_simulation_videos_and_ground_truth(
    output_dir: Path,
    *,
    scenarios: Sequence[SimulationScenarioSpec] = DEFAULT_SCENARIOS,
    width: int = 640,
    height: int = 480,
    fixture_output_dir: Optional[Path] = None,
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
            scen_w = int(spec.width_override or width)
            scen_h = int(spec.height_override or height)
            page.set_viewport_size({"width": scen_w, "height": scen_h})
            scenario_dir = output_dir / spec.name
            if scenario_dir.exists():
                shutil.rmtree(scenario_dir)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            generated[spec.name] = _capture_scenario_in_page(
                page, base_url, spec, scenario_dir, width=scen_w, height=scen_h
            )

        if fixture_output_dir is not None:
            _materialize_compact_test_fixture(
                browser, base_url, fixture_output_dir, width=320, height=240
            )

        browser.close()

    if browser_errors:
        details = "\n".join(f"- {m}" for m in browser_errors)
        raise RuntimeError(f"Browser errors during simulation capture:\n{details}")

    return generated


def _materialize_compact_test_fixture(
    browser: Any,
    base_url: str,
    fixture_dir: Path,
    *,
    width: int = 320,
    height: int = 240,
) -> None:
    """Export a compact, deterministic 8-frame multi-scenario fixture for fast offline pytest runs."""
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    images_dir = fixture_dir / "images"
    masks_dir = fixture_dir / "masks"
    labelme_dir = fixture_dir / "labelme"
    for d in (images_dir, masks_dir, labelme_dir):
        d.mkdir(parents=True, exist_ok=True)

    page = browser.new_page(
        viewport={"width": width, "height": height}, device_scale_factor=1
    )
    fixture_specs = [
        # 4 frames of social_huddle (separated frame 0 -> peak flank contact frames 2, 3 -> post-contact frame 6)
        ("social_huddle", "top-close", "social_huddle", 20260711, 8, [0, 2, 3, 6]),
        # 2 frames of curled flank-grooming
        ("curled_grooming", "top-close", "curled_groom", 20260712, 8, [0, 3]),
        # 2 frames of oblique 3D rearing with visible articulated limbs
        ("rearing_oblique", "oblique-close", "rearing_social", 20260713, 8, [2, 4]),
    ]
    records: List[Dict[str, Any]] = []
    keypoint_names: List[str] = []
    out_idx = 0
    try:
        for name, cam_view, scen, seed, fps, target_frames in fixture_specs:
            query = urlencode(
                {
                    "dataset": 1,
                    "frame": 0,
                    "fps": fps,
                    "seed": seed,
                    "camera_view": cam_view,
                    "scenario": scen,
                }
            )
            page.goto(
                f"{base_url}/{SCENE_PATH}?{query}", wait_until="load", timeout=120_000
            )
            page.wait_for_selector("#c[data-dataset-ready='true']", timeout=120_000)
            for src_f in target_frames:
                payload = page.evaluate(
                    "idx => window.annolidPoseDataset.setFrame(idx)", src_f
                )
                seg_payload = page.evaluate(
                    "() => window.annolidPoseDataset.captureSegmentationMasks()"
                )
                jpg_bytes = page.screenshot(type="jpeg", quality=88)
                stem = f"{name}_f{src_f:02d}"
                img_rel = f"images/{stem}.jpg"
                mask_rel = f"masks/{stem}_ids.png"
                lm_rel = f"labelme/{stem}.json"

                (fixture_dir / img_rel).write_bytes(jpg_bytes)
                track_map, fine_map, coarse_map = decode_segmentation_rle(seg_payload)
                id_bgr = np.stack([coarse_map, fine_map, track_map], axis=-1)
                cv2.imwrite(str(fixture_dir / mask_rel), id_bgr)

                if not keypoint_names:
                    keypoint_names = list(payload["category"]["keypoints"])

                lm_payload = build_ground_truth_labelme_payload(
                    image_filename=f"../{img_rel}",
                    width=width,
                    height=height,
                    annotations=payload["annotations"],
                    keypoint_names=keypoint_names,
                    track_map=track_map,
                    coarse_map=coarse_map,
                    include_gt_body_parts=True,
                )
                (fixture_dir / lm_rel).write_text(
                    json.dumps(lm_payload, indent=2) + "\n", encoding="utf-8"
                )
                records.append(
                    {
                        "index": out_idx,
                        "scenario": name,
                        "source_frame": src_f,
                        "camera_view": cam_view,
                        "image_file": img_rel,
                        "id_mask_file": mask_rel,
                        "labelme_file": lm_rel,
                    }
                )
                out_idx += 1
    finally:
        page.close()

    manifest = {
        "schema_version": 1,
        "generator": "scripts/generate_two_mice_body_part_videos.py",
        "width": width,
        "height": height,
        "keypoint_names": keypoint_names,
        "coarse_part_id_to_name": {str(k): v for k, v in COARSE_ID_TO_NAME.items()},
        "fine_part_id_to_name": {str(k): v for k, v in FINE_ID_TO_NAME.items()},
        "frames": records,
    }
    (fixture_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# Two-Mouse 3D Body-Part Segmentation Fixture

Deterministic 320x240 synthetic fixture generated from Annolid's bundled Three.js
Two-Mice simulation (`annolid/gui/assets/threejs/two_mice.html`) with exact
WebGL 3D mesh ground-truth instance and body-part ID masks (`masks/*_ids.png`)
and surface-occlusion-classified 13-keypoint LabelMe annotations (`labelme/*.json`).

Scenarios included:
- `social_huddle` (4 sequential frames from separated approach into close contact)
- `curled_grooming` (2 frames with C-curved flank-grooming posture)
- `rearing_oblique` (2 oblique 3D frames with upright rearing and visible fore/hindlimbs)
"""
    (fixture_dir / "README.md").write_text(readme, encoding="utf-8")


def _load_scenario_frame_gt(
    scenario_dir: Path, frame_record: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    gt_dir = scenario_dir / "ground_truth"
    bgr = cv2.imread(str(gt_dir / frame_record["image_file"]), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    id_bgr = cv2.imread(
        str(gt_dir / frame_record["id_mask_file"]), cv2.IMREAD_UNCHANGED
    )
    coarse_map = id_bgr[:, :, 0]
    fine_map = id_bgr[:, :, 1]
    track_map = id_bgr[:, :, 2]
    lm_payload = json.loads(
        (gt_dir / frame_record["labelme_file"]).read_text(encoding="utf-8")
    )
    return rgb, track_map, fine_map, coarse_map, lm_payload


def _extract_visible_keypoints_from_labelme(
    lm_payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    kps: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for sh in lm_payload.get("shapes", []):
        if sh.get("shape_type") != "point":
            continue
        vis = int(sh.get("kp_visibility", 2))
        flags = sh.get("flags") or {}
        if flags.get("kp_visible") is False or vis < int(KeypointVisibility.VISIBLE):
            continue
        inst = str(sh.get("instance_label") or f"mouse_{sh.get('group_id')}")
        lbl = str(sh.get("display_label") or sh.get("label"))
        pt = sh["points"][0]
        kps.setdefault(inst, {})[lbl] = (float(pt[0]), float(pt[1]))
    return kps


def evaluate_simulation_suite(
    output_dir: Path,
    *,
    scenarios: Sequence[SimulationScenarioSpec] = DEFAULT_SCENARIOS,
) -> Dict[str, Any]:
    """Run all implemented body-part segmentation features against generated Two-Mice 3D ground truth."""
    scenario_summaries: Dict[str, Any] = {}
    overall_part_ious: Dict[str, List[float]] = {
        name: [] for name in COARSE_ID_TO_NAME.values()
    }
    total_containment_violations = 0
    total_overlap_pixels = 0
    polarity_checks = 0
    polarity_correct = 0

    active_scenarios = [
        s
        for s in scenarios
        if (output_dir / s.name / "ground_truth" / "sequence_gt.json").exists()
    ]

    for spec in active_scenarios:
        scenario_dir = output_dir / spec.name
        seq_meta = json.loads(
            (scenario_dir / "ground_truth" / "sequence_gt.json").read_text(
                encoding="utf-8"
            )
        )
        width = int(seq_meta["width"])
        height = int(seq_meta["height"])
        frames = seq_meta["frames"]

        # 1. Keypoint-guided coarse segmentation + side-by-side GT vs Pred video
        kp_seg = InstanceBodyPartSegmenter(
            BodyPartSegmentationConfig(
                mode="keypoint_guided",
                feature_backend="fast",
                keypoint_granularity="coarse",
            )
        )
        comp_video_path = scenario_dir / f"{spec.name}_gt_vs_pred.mp4"
        comp_writer = cv2.VideoWriter(
            str(comp_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(spec.fps),
            (width * 2, height),
        )
        scen_part_ious: Dict[str, List[float]] = {
            name: [] for name in COARSE_ID_TO_NAME.values()
        }

        try:
            for f_rec in frames:
                rgb, track_map, _fine_map, coarse_map, lm_payload = (
                    _load_scenario_frame_gt(scenario_dir, f_rec)
                )
                parent_masks = {
                    f"mouse_{tid}": (track_map == tid)
                    for tid in (1, 2)
                    if np.any(track_map == tid)
                }
                kps_by_inst = _extract_visible_keypoints_from_labelme(lm_payload)
                preds = kp_seg.segment_frame(
                    rgb, parent_masks, keypoints_by_instance=kps_by_inst
                )

                gt_parts_vis: Dict[str, Dict[str, np.ndarray]] = {}
                pred_parts_vis: Dict[str, Dict[str, np.ndarray]] = {}

                for inst_lbl, pmask in parent_masks.items():
                    tid = int(inst_lbl.split("_")[1])
                    inst_preds_dict = preds.get(inst_lbl, {})
                    pred_map_inst = {
                        k: p.to_full_mask(height, width)
                        for k, p in inst_preds_dict.items()
                    }
                    pred_parts_vis[inst_lbl] = pred_map_inst
                    gt_map_inst = {
                        pname: (pmask & (coarse_map == cid))
                        for cid, pname in COARSE_ID_TO_NAME.items()
                        if int(np.count_nonzero(pmask & (coarse_map == cid))) >= 8
                    }
                    gt_parts_vis[inst_lbl] = gt_map_inst

                    # Strict invariant checks: containment + pairwise mutual exclusion
                    accum = np.zeros_like(pmask)
                    for p_m in pred_map_inst.values():
                        total_containment_violations += int(
                            np.count_nonzero(p_m & (~pmask))
                        )
                        total_overlap_pixels += int(np.count_nonzero(accum & p_m))
                        accum |= p_m

                    # Per-part IoU on visible ground-truth regions
                    for pname, gt_m in gt_map_inst.items():
                        pred_m = pred_map_inst.get(
                            pname, np.zeros_like(gt_m, dtype=bool)
                        )
                        iou = compute_iou(pred_m, gt_m)
                        scen_part_ious[pname].append(iou)
                        overall_part_ious[pname].append(iou)

                    # Head-to-tail polarity check against 3D ground truth centroids
                    if (
                        "head" in gt_map_inst
                        and "tail" in gt_map_inst
                        and "head" in pred_map_inst
                        and "tail" in pred_map_inst
                        and np.any(pred_map_inst["head"])
                        and np.any(pred_map_inst["tail"])
                    ):
                        gy_h, gx_h = np.nonzero(gt_map_inst["head"])
                        gy_t, gx_t = np.nonzero(gt_map_inst["tail"])
                        py_h, px_h = np.nonzero(pred_map_inst["head"])
                        py_t, px_t = np.nonzero(pred_map_inst["tail"])
                        gt_vec = (
                            float(gx_h.mean() - gx_t.mean()),
                            float(gy_h.mean() - gy_t.mean()),
                        )
                        pr_vec = (
                            float(px_h.mean() - px_t.mean()),
                            float(py_h.mean() - py_t.mean()),
                        )
                        if math.hypot(*gt_vec) > 6.0 and math.hypot(*pr_vec) > 6.0:
                            polarity_checks += 1
                            if gt_vec[0] * pr_vec[0] + gt_vec[1] * pr_vec[1] > 0.0:
                                polarity_correct += 1
                    _ = tid

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                left_panel = _render_overlay_panel(
                    bgr, gt_parts_vis, parent_masks, f"{spec.name} | 3D Ground Truth"
                )
                right_panel = _render_overlay_panel(
                    bgr,
                    pred_parts_vis,
                    parent_masks,
                    f"{spec.name} | Annolid Predicted",
                )
                sbs = np.concatenate([left_panel, right_panel], axis=1)
                comp_writer.write(sbs)
                if int(f_rec["frame_id"]) == len(frames) // 2:
                    cv2.imwrite(
                        str(scenario_dir / f"{spec.name}_midframe_preview.png"), sbs
                    )
        finally:
            comp_writer.release()

        # 2. Seeded frame-0 propagation (without keypoints on frames 1..N-1) + cross-instance seed sharing
        rgb0, track0, _, coarse0, _ = _load_scenario_frame_gt(scenario_dir, frames[0])
        parent0 = {
            f"mouse_{tid}": (track0 == tid) for tid in (1, 2) if np.any(track0 == tid)
        }
        # Seed ONLY axial regions (head, torso, tail) of mouse_1 on frame 0 to test
        # temporal propagation + share_seed_across_instances -> mouse_2 (black C57BL/6 -> white BALB/c!)
        axial_coarse_ids = {
            cid: pname
            for cid, pname in COARSE_ID_TO_NAME.items()
            if pname in {"head", "torso", "tail"}
        }
        seed_only_m1 = {
            "mouse_1": {
                pname: (
                    (parent0["mouse_1"] & (coarse0 == cid))
                    if pname != "torso"
                    else (parent0["mouse_1"] & np.isin(coarse0, [2, 3]))
                )
                for cid, pname in axial_coarse_ids.items()
                if int(np.count_nonzero(parent0["mouse_1"] & (coarse0 == cid))) >= 8
            }
        }
        seeded_seg = InstanceBodyPartSegmenter(
            BodyPartSegmentationConfig(
                mode="seeded",
                feature_backend="fast",
                share_seed_across_instances=True,
                partition_full_instance=True,
            )
        )
        seeded_seg.fit_seed_frame(rgb0, parent0, seed_only_m1)
        seeded_ious: List[float] = []
        seeded_m1_ious: List[float] = []
        shared_m2_ious: List[float] = []
        for f_rec in frames[1:]:
            rgb_f, track_f, _, coarse_f, _ = _load_scenario_frame_gt(
                scenario_dir, f_rec
            )
            p_masks = {
                f"mouse_{tid}": (track_f == tid)
                for tid in (1, 2)
                if np.any(track_f == tid)
            }
            s_preds = seeded_seg.segment_frame(rgb_f, p_masks)
            for inst_lbl, pmask in p_masks.items():
                pmap = {
                    k: p.to_full_mask(height, width)
                    for k, p in s_preds.get(inst_lbl, {}).items()
                }
                for cid, pname in axial_coarse_ids.items():
                    gt_m = (
                        (pmask & (coarse_f == cid))
                        if pname != "torso"
                        else (pmask & np.isin(coarse_f, [2, 3]))
                    )
                    if (
                        int(np.count_nonzero(gt_m)) < 12
                        or pname not in seed_only_m1["mouse_1"]
                    ):
                        continue
                    iou_val = compute_iou(pmap.get(pname, np.zeros_like(gt_m)), gt_m)
                    seeded_ious.append(iou_val)
                    if inst_lbl == "mouse_1":
                        seeded_m1_ious.append(iou_val)
                    elif inst_lbl == "mouse_2":
                        shared_m2_ious.append(iou_val)

        # 3. Unsupervised 3-part mode (zero seeds, zero keypoints)
        unsup_seg = InstanceBodyPartSegmenter(
            BodyPartSegmentationConfig(
                mode="unsupervised",
                feature_backend="fast",
                num_parts=3,
                part_names=("head", "torso", "tail"),
            )
        )
        unsup_nonempty_frames = 0
        max_tortuosity_observed = 1.0
        for f_rec in frames:
            rgb_f, track_f, _, _, _ = _load_scenario_frame_gt(scenario_dir, f_rec)
            p_masks = {
                f"mouse_{tid}": (track_f == tid)
                for tid in (1, 2)
                if np.any(track_f == tid)
            }
            u_preds = unsup_seg.segment_frame(rgb_f, p_masks)
            if all(len(u_preds.get(k, {})) >= 2 for k in p_masks):
                unsup_nonempty_frames += 1
            for st in unsup_seg._temporal_states.values():
                max_tortuosity_observed = max(
                    max_tortuosity_observed, float(st.pose.tortuosity)
                )

        scenario_summaries[spec.name] = {
            "frames": len(frames),
            "video": (scenario_dir / f"{spec.name}.mp4")
            .relative_to(output_dir)
            .as_posix(),
            "comparison_video": comp_video_path.relative_to(output_dir).as_posix(),
            "keypoint_guided_coarse_part_miou": {
                k: round(float(np.mean(v)), 4) for k, v in scen_part_ious.items() if v
            },
            "keypoint_guided_overall_miou": round(
                float(np.mean([x for vals in scen_part_ious.values() for x in vals])),
                4,
            ),
            "seeded_propagation_miou": round(float(np.mean(seeded_ious)), 4)
            if seeded_ious
            else 0.0,
            "seeded_same_instance_mouse1_miou": round(float(np.mean(seeded_m1_ious)), 4)
            if seeded_m1_ious
            else 0.0,
            "cross_strain_shared_seed_mouse2_miou": round(
                float(np.mean(shared_m2_ious)), 4
            )
            if shared_m2_ious
            else 0.0,
            "unsupervised_valid_frame_ratio": round(
                float(unsup_nonempty_frames) / float(len(frames)), 4
            ),
            "max_geodesic_tortuosity": round(max_tortuosity_observed, 4),
        }

    # 4. Fine keypoint granularity evaluation on oblique 3D rearing video (left/right fore/hind limbs)
    rearing_dir = output_dir / "rearing_oblique"
    rearing_meta = json.loads(
        (rearing_dir / "ground_truth" / "sequence_gt.json").read_text(encoding="utf-8")
    )
    fine_seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="fast",
            keypoint_granularity="fine",
        )
    )
    fine_limb_ious: List[float] = []
    fine_all_ious: List[float] = []
    for f_rec in rearing_meta["frames"]:
        rgb, track_map, fine_map, _, lm_payload = _load_scenario_frame_gt(
            rearing_dir, f_rec
        )
        parent_masks = {
            f"mouse_{tid}": (track_map == tid)
            for tid in (1, 2)
            if np.any(track_map == tid)
        }
        kps_by_inst = _extract_visible_keypoints_from_labelme(lm_payload)
        f_preds = fine_seg.segment_frame(
            rgb, parent_masks, keypoints_by_instance=kps_by_inst
        )
        for inst_lbl, pmask in parent_masks.items():
            r_h, r_w = pmask.shape
            pmap = {
                k: p.to_full_mask(r_h, r_w)
                for k, p in f_preds.get(inst_lbl, {}).items()
            }
            gt_fine_grouped: Dict[str, np.ndarray] = {}
            for fid, mapped_lbl in FINE_ID_TO_KEYPOINT_FINE_LABEL.items():
                gt_fine_grouped.setdefault(mapped_lbl, np.zeros_like(pmask, dtype=bool))
                gt_fine_grouped[mapped_lbl] |= pmask & (fine_map == fid)
            for lbl, gt_m in gt_fine_grouped.items():
                if int(np.count_nonzero(gt_m)) < 8 or lbl not in pmap:
                    continue
                iou_v = compute_iou(pmap[lbl], gt_m)
                fine_all_ious.append(iou_v)
                if "limb" in lbl:
                    fine_limb_ious.append(iou_v)

    # 5. End-to-end video processor + Merged-Parent Huddling Split & Partner Recovery + CSV + Track QC
    huddle_dir = output_dir / "social_huddle"
    huddle_meta = json.loads(
        (huddle_dir / "ground_truth" / "sequence_gt.json").read_text(encoding="utf-8")
    )
    huddle_video = huddle_dir / "social_huddle.mp4"
    corrupted_run_dir = huddle_dir / "merged_tracker_failure_run"
    if corrupted_run_dir.exists():
        shutil.rmtree(corrupted_run_dir)
    corrupted_run_dir.mkdir(parents=True, exist_ok=True)

    # Find the 3 closest-contact frames in social_huddle and merge mouse_1 + mouse_2 into a single bloated mouse_1 polygon!
    frame_clearances = [
        (
            int(fr["frame_id"]),
            float(
                fr["telemetry"]["minimum_subject_clearance"]
                if fr["telemetry"]["minimum_subject_clearance"] is not None
                else 99.0
            ),
        )
        for fr in huddle_meta["frames"]
        if int(fr["frame_id"]) >= 2
    ]
    frame_clearances.sort(key=lambda item: item[1])
    merged_frame_ids = sorted([fid for fid, _ in frame_clearances[:3]])

    for f_rec in huddle_meta["frames"]:
        fid = int(f_rec["frame_id"])
        rgb, track_map, _, coarse_map, lm_payload = _load_scenario_frame_gt(
            huddle_dir, f_rec
        )
        frame_shapes: List[Dict[str, Any]] = []
        if fid in merged_frame_ids:
            # Simulate severe tracker merge: bridge mouse_1 + mouse_2 into ONE bloated mouse_1 mask and drop mouse_2's parent polygon!
            union_m = (track_map == 1) | (track_map == 2)
            # Connect across narrow clearance gap via convex/morphological bridge
            ys, xs = np.nonzero(union_m)
            pts_all = np.column_stack([xs, ys]).astype(np.int32)
            hull = cv2.convexHull(pts_all)
            bloated_m = union_m.copy().astype(np.uint8) * 255
            # Draw a thick bridge between centroids so the two masks form a single merged connected blob
            y1, x1 = np.nonzero(track_map == 1)
            y2, x2 = np.nonzero(track_map == 2)
            if x1.size > 0 and x2.size > 0:
                c1 = (int(round(x1.mean())), int(round(y1.mean())))
                c2 = (int(round(x2.mean())), int(round(y2.mean())))
                cv2.line(bloated_m, c1, c2, 255, thickness=22)
            _ = hull
            merged_poly = mask_to_polygons(
                bloated_m > 0, epsilon=0.75, min_area_px=20, single_parent_contour=True
            )[0]
            frame_shapes.append(
                {
                    "label": "mouse_1",
                    "points": merged_poly,
                    "group_id": 1,
                    "shape_type": "polygon",
                    "flags": {},
                    "visible": True,
                }
            )
            # Keep only keypoints on 2 of the 3 merged frames; on the 3rd merged frame drop ALL keypoints too so BOTH Trigger 1 (keypoints) and Trigger 2 (area-surge + track history) are exercised!
            if fid != merged_frame_ids[-1]:
                for sh in lm_payload["shapes"]:
                    if sh.get("shape_type") == "point":
                        frame_shapes.append(sh)
        else:
            for sh in lm_payload["shapes"]:
                if not is_body_part_shape_payload(sh):
                    if fid == 0:
                        frame_shapes.append(sh)
                    elif sh.get("shape_type") in {"polygon", "point"}:
                        frame_shapes.append(sh)
            if fid == 0:
                # Add manual frame-0 seed body parts + PNG marker so frame 0 acts as an immutable seed frame!
                for sh in lm_payload["shapes"]:
                    if is_body_part_shape_payload(sh):
                        frame_shapes.append(sh)
                cv2.imwrite(
                    str(corrupted_run_dir / f"social_huddle_{fid:09d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )

        run_json = {
            "version": "5.4.1",
            "flags": {},
            "shapes": frame_shapes,
            "imagePath": f"social_huddle_{fid:09d}.png",
            "imageData": None,
            "imageHeight": int(huddle_meta["height"]),
            "imageWidth": int(huddle_meta["width"]),
        }
        (corrupted_run_dir / f"social_huddle_{fid:09d}.json").write_text(
            json.dumps(run_json, indent=2) + "\n", encoding="utf-8"
        )

    proc_summary = run_body_part_segmentation(
        huddle_video,
        annotation_dir=corrupted_run_dir,
        config=BodyPartSegmentationConfig(
            mode="auto",
            feature_backend="fast",
            split_merged_parent_masks=True,
            recover_missing_merged_instances=True,
            persist_json=True,
        ),
        export_csv=True,
        export_ethogram=True,
        fps=float(huddle_meta.get("fps", 8.0)),
    )
    csv_out = Path(
        proc_summary.get("tracking_csv")
        or (corrupted_run_dir.parent / f"{corrupted_run_dir.name}_tracking.csv")
    )
    ethogram_summary = proc_summary.get("ethogram") or {}

    recovered_m1_ious: List[float] = []
    recovered_m2_ious: List[float] = []
    for fid in merged_frame_ids:
        repaired_json = json.loads(
            (corrupted_run_dir / f"social_huddle_{fid:09d}.json").read_text(
                encoding="utf-8"
            )
        )
        f_rec = huddle_meta["frames"][fid]
        _, gt_track, _, _, _ = _load_scenario_frame_gt(huddle_dir, f_rec)
        h_h, h_w = gt_track.shape
        parent_polys_out = {
            sh["label"]: sh["points"]
            for sh in repaired_json["shapes"]
            if sh.get("shape_type") == "polygon" and not is_body_part_shape_payload(sh)
        }
        if "mouse_1" in parent_polys_out:
            m1_pred = polygons_to_mask([parent_polys_out["mouse_1"]], h_h, h_w)
            recovered_m1_ious.append(compute_iou(m1_pred, gt_track == 1))
        if "mouse_2" in parent_polys_out:
            m2_pred = polygons_to_mask([parent_polys_out["mouse_2"]], h_h, h_w)
            recovered_m2_ious.append(compute_iou(m2_pred, gt_track == 2))

    qc_report = evaluate_track_qc(corrupted_run_dir, expected_instances=2)
    missing_parent_issues = [
        iss for iss in qc_report.issues if iss.code == "body_part_missing_parent"
    ]
    containment_qc_issues = [
        iss
        for iss in qc_report.issues
        if iss.code in {"body_part_containment_violation", "body_part_overlap"}
    ]

    from annolid.postprocessing.body_part_ethogram import analyze_body_part_ethogram

    posture_disentanglement: Dict[str, Any] = {}
    rearing_lm_dir = output_dir / "rearing_oblique" / "ground_truth" / "labelme"
    curled_lm_dir = output_dir / "curled_grooming" / "ground_truth" / "labelme"
    if rearing_lm_dir.is_dir() and curled_lm_dir.is_dir():
        rear_eth = analyze_body_part_ethogram(
            rearing_lm_dir, fps=8.0, camera_view="auto"
        )
        curl_eth = analyze_body_part_ethogram(
            curled_lm_dir, fps=8.0, camera_view="auto"
        )
        r_m1 = rear_eth.kinematics_df[
            rear_eth.kinematics_df["instance_name"] == "mouse_1"
        ]
        c_m1 = curl_eth.kinematics_df[
            curl_eth.kinematics_df["instance_name"] == "mouse_1"
        ]
        posture_disentanglement = {
            "rearing_oblique_mouse1_max_rearing_index": (
                round(float(r_m1["rearing_index"].max()), 4) if not r_m1.empty else 0.0
            ),
            "rearing_oblique_mouse1_mean_curl_index": (
                round(float(r_m1["curl_groom_index"].mean()), 4)
                if not r_m1.empty
                else 0.0
            ),
            "curled_grooming_mouse1_mean_curl_index": (
                round(float(c_m1["curl_groom_index"].mean()), 4)
                if not c_m1.empty
                else 0.0
            ),
            "curled_grooming_mouse1_mean_rearing_index": (
                round(float(c_m1["rearing_index"].mean()), 4) if not c_m1.empty else 0.0
            ),
        }

    summary = {
        "schema_version": 1,
        "total_scenarios": len(active_scenarios),
        "total_frames_evaluated": sum(s.frame_count for s in active_scenarios),
        "invariants": {
            "containment_violation_pixels": total_containment_violations,
            "pairwise_part_overlap_pixels": total_overlap_pixels,
            "head_tail_polarity_checks": polarity_checks,
            "head_tail_polarity_accuracy": round(
                float(polarity_correct) / float(max(1, polarity_checks)), 4
            ),
        },
        "overall_keypoint_guided_coarse_miou": {
            k: round(float(np.mean(v)), 4) for k, v in overall_part_ious.items() if v
        },
        "rearing_fine_granularity": {
            "all_parts_miou": round(float(np.mean(fine_all_ious)), 4)
            if fine_all_ious
            else 0.0,
            "laterality_limbs_miou": round(float(np.mean(fine_limb_ious)), 4)
            if fine_limb_ious
            else 0.0,
        },
        "posture_disentanglement": posture_disentanglement,
        "merged_huddle_recovery": {
            "corrupted_merged_frames": merged_frame_ids,
            "merged_parent_splits": int(proc_summary.get("merged_parent_splits", 0)),
            "recovered_parent_instances": int(
                proc_summary.get("recovered_parent_instances", 0)
            ),
            "recovered_mouse_1_parent_miou": round(float(np.mean(recovered_m1_ious)), 4)
            if recovered_m1_ious
            else 0.0,
            "recovered_mouse_2_parent_miou": round(float(np.mean(recovered_m2_ious)), 4)
            if recovered_m2_ious
            else 0.0,
            "csv_exported": csv_out.exists(),
            "ethogram_exported": bool(ethogram_summary),
            "ethogram_total_bouts": int(ethogram_summary.get("total_bouts", 0)),
            "ethogram_bout_counts": dict(
                ethogram_summary.get("bout_counts_by_type") or {}
            ),
            "uncertain_active_learning_frames": len(
                proc_summary.get("uncertain_frames") or []
            ),
            "qc_missing_parent_warnings": len(missing_parent_issues),
            "qc_body_part_geometry_violations": len(containment_qc_issues),
        },
        "scenarios": scenario_summaries,
    }

    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown_report(output_dir, summary)
    return summary


def _write_markdown_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    inv = summary["invariants"]
    coarse = summary["overall_keypoint_guided_coarse_miou"]
    merge = summary["merged_huddle_recovery"]
    lines = [
        "# Two-Mice 3D Simulation Body-Part Segmentation Evaluation",
        "",
        f"- **Total scenarios**: `{summary['total_scenarios']}`",
        f"- **Total rendered frames**: `{summary['total_frames_evaluated']}`",
        f"- **Containment violation pixels**: `{inv['containment_violation_pixels']}`",
        f"- **Pairwise part overlap pixels**: `{inv['pairwise_part_overlap_pixels']}`",
        f"- **Head-tail polarity accuracy**: `{inv['head_tail_polarity_accuracy']:.2%}` (`{inv['head_tail_polarity_checks']}` checks)",
        "",
        "## Coarse Body-Part mIoU vs 3D Mesh Ground Truth",
        "",
        "| Body Part | mIoU |",
        "| --- | --- |",
    ]
    for k, v in coarse.items():
        lines.append(f"| `{k}` | `{v:.4f}` |")
    lines.extend(
        [
            "",
            "## Scenario Breakdown",
            "",
            "| Scenario | Frames | Keypoint Coarse mIoU | Seeded Propagation mIoU | Cross-Strain Shared Seed mIoU | Max Tortuosity |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, s in summary["scenarios"].items():
        lines.append(
            f"| `{name}` | `{s['frames']}` | `{s['keypoint_guided_overall_miou']:.4f}` | "
            f"`{s['seeded_propagation_miou']:.4f}` | `{s['cross_strain_shared_seed_mouse2_miou']:.4f}` | "
            f"`{s['max_geodesic_tortuosity']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Merged-Parent Huddle Recovery (`social_huddle.mp4`)",
            "",
            f"- **Corrupted merged frames**: `{merge['corrupted_merged_frames']}`",
            f"- **Merged parent splits triggered**: `{merge['merged_parent_splits']}`",
            f"- **Recovered missing partner instances**: `{merge['recovered_parent_instances']}`",
            f"- **Recovered `mouse_1` parent mIoU**: `{merge['recovered_mouse_1_parent_miou']:.4f}`",
            f"- **Recovered `mouse_2` parent mIoU**: `{merge['recovered_mouse_2_parent_miou']:.4f}`",
            f"- **Track QC `body_part_missing_parent` warnings**: `{merge['qc_missing_parent_warnings']}`",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def export_simulation_training_datasets(output_dir: Path) -> Dict[str, str]:
    """Export hierarchy-separated whole-animal and body-part COCO/YOLO datasets from simulation LabelMe GT."""
    from annolid.annotation import labelme2coco
    from annolid.annotation.labelme2yolo import Labelme2YOLO

    staging_dir = output_dir / "training_datasets" / "labelme_gt_merged"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_count = 0
    for scen_dir in sorted(output_dir.iterdir()):
        lm_dir = scen_dir / "ground_truth" / "labelme"
        img_dir = scen_dir / "ground_truth" / "images"
        if not lm_dir.is_dir() or not img_dir.is_dir():
            continue
        for jf in sorted(lm_dir.glob("*.json")):
            payload = json.loads(jf.read_text(encoding="utf-8"))
            src_img = img_dir / Path(str(payload["imagePath"])).name
            if not src_img.exists():
                continue
            stem = f"{scen_dir.name}_{jf.stem}"
            dst_img = staging_dir / f"{stem}{src_img.suffix}"
            shutil.copy2(src_img, dst_img)
            payload["imagePath"] = dst_img.name
            (staging_dir / f"{stem}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            staged_count += 1

    if staged_count == 0:
        return {}

    ds_root = output_dir / "training_datasets"
    coco_inst_dir = ds_root / "coco_instances"
    coco_parts_dir = ds_root / "coco_parts"
    list(
        labelme2coco.convert(
            str(staging_dir),
            str(coco_inst_dir),
            labels_file=None,
            train_valid_split=0.8,
            export_level="instances",
        )
    )
    list(
        labelme2coco.convert(
            str(staging_dir),
            str(coco_parts_dir),
            labels_file=None,
            train_valid_split=0.8,
            export_level="parts",
        )
    )

    yolo_inst_stage = ds_root / "yolo_instances_src"
    yolo_parts_stage = ds_root / "yolo_parts_src"
    for target_stage in (yolo_inst_stage, yolo_parts_stage):
        if target_stage.exists():
            shutil.rmtree(target_stage)
        shutil.copytree(staging_dir, target_stage)

    Labelme2YOLO(str(yolo_inst_stage), export_level="instances").convert(
        val_size=0.2, test_size=0.0
    )
    Labelme2YOLO(str(yolo_parts_stage), export_level="parts").convert(
        val_size=0.2, test_size=0.0
    )
    return {
        "coco_instances": str(coco_inst_dir),
        "coco_parts": str(coco_parts_dir),
        "yolo_instances": str(yolo_inst_stage / "YOLO_dataset"),
        "yolo_parts": str(yolo_parts_stage / "YOLO_dataset"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--include-domain-variants",
        action="store_true",
        help="Also render/evaluate same-strain C57 pair, IR night lighting, ventral glass-floor gait, and wire-bar occlusion scenarios.",
    )
    parser.add_argument(
        "--render-missing-only",
        action="store_true",
        help="Only invoke Playwright capture for scenarios whose sequence_gt.json does not yet exist in --output.",
    )
    parser.add_argument(
        "--export-training-datasets",
        action="store_true",
        help="Export separated COCO and YOLO-Seg whole-animal + body-part training datasets from generated 3D LabelMe GT.",
    )
    parser.add_argument(
        "--write-tiny-fixture",
        action="store_true",
        help="Also materialize compact 320x240 fixture in tests/fixtures/two_mice_body_part_tiny.",
    )
    parser.add_argument("--fixture-output", type=Path, default=DEFAULT_FIXTURE_OUTPUT)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip Playwright capture and evaluate existing generated ground truth in --output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output.resolve()
    scenarios: Tuple[SimulationScenarioSpec, ...] = (
        DEFAULT_SCENARIOS + DOMAIN_RANDOMIZATION_SCENARIOS
        if args.include_domain_variants
        else DEFAULT_SCENARIOS
    )
    if not args.evaluate_only:
        to_render: Sequence[SimulationScenarioSpec] = (
            tuple(
                s
                for s in scenarios
                if not (out / s.name / "ground_truth" / "sequence_gt.json").exists()
            )
            if args.render_missing_only
            else scenarios
        )
        fixture_out = args.fixture_output.resolve() if args.write_tiny_fixture else None
        if to_render or fixture_out is not None:
            generate_simulation_videos_and_ground_truth(
                out,
                scenarios=to_render,
                width=args.width,
                height=args.height,
                fixture_output_dir=fixture_out,
            )
    summary = evaluate_simulation_suite(out, scenarios=scenarios)
    if args.export_training_datasets:
        summary["exported_training_datasets"] = export_simulation_training_datasets(out)
        (out / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
