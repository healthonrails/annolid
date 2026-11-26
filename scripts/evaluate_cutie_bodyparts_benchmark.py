#!/usr/bin/env python3
"""Evaluate baseline Cutie (cutie-base-mega.pth) vs SAM3-upgraded Cutie (cutie-sam3-bodyparts.pth)
across 4 held-out multi-domain instance & body-part VOS benchmarks:
  1. Annolid 9-Scenario 3D Rodent Video Benchmark (Instance, Coarse Parts, Fine Parts)
  2. Open-Source PartImageNet Official Test Split (Biological Body Parts + Multi-Animal Crossing)
  3. Open-Source Pascal-Part Official Val Split (Body Parts + Multi-Instance Crossing)
  4. Open-Source ATR Human Parsing Held-Out Split (Human Body Parts + Multi-Person Crossing)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from annolid.segmentation.cutie_vos.engine import unwrap_cutie_state_dict
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.interactive_utils import image_to_torch
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.utils.devices import get_device
from annolid.utils.safe_torch_load import safe_torch_load

REPO_ROOT = Path(__file__).resolve().parents[1]
CUTIE_CFG_DIR = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "config"
DEFAULT_BASE_WEIGHTS = (
    REPO_ROOT
    / "annolid"
    / "segmentation"
    / "cutie_vos"
    / "weights"
    / "cutie-base-mega.pth"
)
DEFAULT_ORIG_BACKUP_WEIGHTS = DEFAULT_BASE_WEIGHTS.parent / "cutie-base-mega.orig.pth"
DEFAULT_SAM3_WEIGHTS = (
    REPO_ROOT
    / "annolid"
    / "segmentation"
    / "cutie_vos"
    / "weights"
    / "cutie-sam3-bodyparts.pth"
)
DEFAULT_TWO_MICE_EVAL_DIR = REPO_ROOT / "output" / "two_mice_body_part_eval"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "output" / "cutie_sam3_bodyparts_eval"

COARSE_ID_TO_NAME: Dict[int, str] = {
    1: "head",
    2: "torso",
    3: "limbs",
    4: "tail",
}
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

PALETTE_RGB: np.ndarray = np.array(
    [
        [0, 0, 0],
        [240, 78, 62],
        [66, 196, 98],
        [62, 148, 245],
        [245, 192, 52],
        [198, 88, 232],
        [52, 212, 208],
        [250, 134, 62],
        [152, 218, 68],
        [238, 86, 158],
        [110, 118, 245],
        [235, 218, 95],
        [88, 178, 142],
        [210, 110, 90],
        [135, 185, 235],
        [195, 145, 225],
        [245, 165, 125],
        [98, 225, 165],
        [225, 95, 95],
        [95, 155, 215],
        [185, 205, 85],
        [165, 95, 195],
        [95, 205, 205],
    ],
    dtype=np.uint8,
)


def compute_j_and_f(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Compute DAVIS-standard region similarity J (IoU) and contour accuracy F."""
    pred_b = np.asarray(pred, dtype=bool)
    gt_b = np.asarray(gt, dtype=bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    j_score = 1.0 if union == 0 else float(inter) / float(union)

    kernel3 = np.ones((3, 3), dtype=np.uint8)
    pred_bnd = (
        cv2.morphologyEx(pred_b.astype(np.uint8), cv2.MORPH_GRADIENT, kernel3) > 0
    )
    gt_bnd = cv2.morphologyEx(gt_b.astype(np.uint8), cv2.MORPH_GRADIENT, kernel3) > 0
    if not pred_bnd.any() and not gt_bnd.any():
        return j_score, 1.0
    if not pred_bnd.any() or not gt_bnd.any():
        return j_score, 0.0

    tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_dil = cv2.dilate(pred_bnd.astype(np.uint8), tol) > 0
    gt_dil = cv2.dilate(gt_bnd.astype(np.uint8), tol) > 0
    prec = float(np.logical_and(pred_bnd, gt_dil).sum()) / float(pred_bnd.sum())
    rec = float(np.logical_and(gt_bnd, pred_dil).sum()) / float(gt_bnd.sum())
    f_score = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return j_score, f_score


def build_two_mice_gt_mask(id_bgr: np.ndarray, granularity: str) -> np.ndarray:
    """Decode lossless 3-channel Two-Mice PNG into a single 2D indexed mask."""
    coarse_map = id_bgr[:, :, 0].astype(np.int32)
    fine_map = id_bgr[:, :, 1].astype(np.int32)
    track_map = id_bgr[:, :, 2].astype(np.int32)
    out = np.zeros_like(track_map, dtype=np.int32)
    if granularity == "instance":
        for tid in (1, 2):
            out[track_map == tid] = tid
    elif granularity == "coarse":
        for tid in (1, 2):
            for pid in COARSE_ID_TO_NAME:
                out[(track_map == tid) & (coarse_map == pid)] = (tid - 1) * 4 + pid
    elif granularity == "fine":
        for tid in (1, 2):
            for pid in FINE_ID_TO_NAME:
                out[(track_map == tid) & (fine_map == pid)] = (tid - 1) * 11 + pid
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return out


def synthesize_articulated_part_sequence(
    rgb: np.ndarray,
    part_mask: np.ndarray,
    *,
    num_frames: int = 8,
    seed: int = 0,
    target_hw: Optional[Tuple[int, int]] = None,
    add_occluder: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Synthesize a deterministic non-rigid articulated + camera-motion VOS sequence."""
    rng = np.random.default_rng(seed)
    if target_hw is not None:
        th, tw = target_hw
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        part_mask = cv2.resize(part_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    max_rot_deg = float(rng.uniform(-13.0, 13.0))
    max_tx = float(rng.uniform(-0.10, 0.10)) * w
    max_ty = float(rng.uniform(-0.10, 0.10)) * h
    max_zoom = float(rng.uniform(0.88, 1.14))
    amp_x = float(rng.uniform(4.5, 9.5))
    amp_y = float(rng.uniform(4.5, 9.5))
    phase_x = float(rng.uniform(0.0, 2.0 * math.pi))
    phase_y = float(rng.uniform(0.0, 2.0 * math.pi))
    illum_delta = float(rng.uniform(-22.0, 22.0))
    occ_vertical = bool(rng.integers(0, 2) == 0)

    frames_rgb: List[np.ndarray] = [rgb.copy()]
    frames_mask: List[np.ndarray] = [part_mask.astype(np.int32).copy()]

    # Determine foreground centroid for targeted transient occlusion
    fg_ys, fg_xs = np.nonzero(part_mask > 0)
    cy_fg = float(fg_ys.mean()) if fg_ys.size > 0 else h * 0.5
    cx_fg = float(fg_xs.mean()) if fg_xs.size > 0 else w * 0.5

    for t in range(1, num_frames):
        alpha = float(t) / float(max(1, num_frames - 1))
        smooth = math.sin(0.5 * math.pi * alpha)
        wave = math.sin(math.pi * alpha)

        rot = max_rot_deg * smooth
        scale = 1.0 + (max_zoom - 1.0) * wave
        tx = max_tx * smooth
        ty = max_ty * smooth

        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), rot, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        inv_mat = cv2.invertAffineTransform(mat)

        src_x = inv_mat[0, 0] * xx + inv_mat[0, 1] * yy + inv_mat[0, 2]
        src_y = inv_mat[1, 0] * xx + inv_mat[1, 1] * yy + inv_mat[1, 2]

        # Non-rigid articulated harmonic deformation
        dx = amp_x * wave * np.sin(2.0 * math.pi * yy / float(h) + phase_x + alpha)
        dy = amp_y * wave * np.cos(2.0 * math.pi * xx / float(w) + phase_y + alpha)
        map_x = (src_x + dx).astype(np.float32)
        map_y = (src_y + dy).astype(np.float32)

        warped_rgb = cv2.remap(
            rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped_mask = cv2.remap(
            part_mask.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.int32)

        # Smooth illumination drift
        lit = np.clip(
            warped_rgb.astype(np.float32) + illum_delta * wave, 0.0, 255.0
        ).astype(np.uint8)

        # Transient partial bar occlusion on middle frame(s)
        if (
            add_occluder
            and num_frames >= 6
            and t in (num_frames // 2 - 1, num_frames // 2)
        ):
            bar_half = max(5, min(h, w) // 36)
            shift = int(round((t - num_frames // 2) * bar_half * 1.5))
            if occ_vertical:
                bx0 = max(0, int(round(cx_fg + tx)) - bar_half + shift)
                bx1 = min(w, bx0 + 2 * bar_half)
                if bx1 > bx0:
                    lit[:, bx0:bx1] = np.array([118, 122, 128], dtype=np.uint8)
                    warped_mask[:, bx0:bx1] = 0
            else:
                by0 = max(0, int(round(cy_fg + ty)) - bar_half + shift)
                by1 = min(h, by0 + 2 * bar_half)
                if by1 > by0:
                    lit[by0:by1, :] = np.array([118, 122, 128], dtype=np.uint8)
                    warped_mask[by0:by1, :] = 0

        frames_rgb.append(lit)
        frames_mask.append(warped_mask)

    return frames_rgb, frames_mask


def _extract_subject_sprite(
    rgb: np.ndarray, mask: np.ndarray, target_max_dim: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    fg = mask > 0
    ys, xs = np.nonzero(fg)
    if ys.size < 60:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_fg = fg[y0:y1, x0:x1].astype(np.uint8)
    ch, cw = crop_rgb.shape[:2]
    scale = float(target_max_dim) / float(max(ch, cw))
    nh = max(24, int(round(ch * scale)))
    nw = max(24, int(round(cw * scale)))
    s_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    s_fg = cv2.resize(crop_fg, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    return s_rgb, s_fg


def synthesize_multi_instance_crossing_sequence(
    rgb_a: np.ndarray,
    whole_a: np.ndarray,
    rgb_b: np.ndarray,
    whole_b: np.ndarray,
    *,
    num_frames: int = 8,
    seed: int = 0,
    canvas_hw: Tuple[int, int] = (320, 416),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Synthesize a two-subject close-contact crossing & partial-occlusion VOS clip."""
    rng = np.random.default_rng(seed)
    ch, cw = canvas_hw
    bg = cv2.GaussianBlur(
        cv2.resize(rgb_a, (cw, ch), interpolation=cv2.INTER_LINEAR), (31, 31), 9.0
    )
    bg = np.clip(bg.astype(np.float32) * 0.78 + 28.0, 0, 255).astype(np.uint8)

    target_dim = int(min(ch, cw) * 0.52)
    sp_a = _extract_subject_sprite(
        rgb_a, whole_a == 1 if (whole_a == 1).any() else whole_a > 0, target_dim
    )
    sp_b = _extract_subject_sprite(
        rgb_b, whole_b == 1 if (whole_b == 1).any() else whole_b > 0, target_dim
    )
    if sp_a is None or sp_b is None:
        return synthesize_articulated_part_sequence(
            rgb_a, (whole_a > 0).astype(np.int32), num_frames=num_frames, seed=seed
        )

    s_rgb_a, s_fg_a = sp_a
    s_rgb_b, s_fg_b = sp_b

    frames_rgb: List[np.ndarray] = []
    frames_mask: List[np.ndarray] = []

    def _paste(
        canvas: np.ndarray,
        idx_mask: np.ndarray,
        spr_rgb: np.ndarray,
        spr_fg: np.ndarray,
        cx: float,
        cy: float,
        obj_id: int,
        rot_deg: float,
    ) -> None:
        sh, sw = spr_rgb.shape[:2]
        m = cv2.getRotationMatrix2D((sw * 0.5, sh * 0.5), rot_deg, 1.0)
        r_rgb = cv2.warpAffine(
            spr_rgb,
            m,
            (sw, sh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        r_fg = (
            cv2.warpAffine(
                spr_fg.astype(np.uint8),
                m,
                (sw, sh),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        x0 = int(round(cx - sw * 0.5))
        y0 = int(round(cy - sh * 0.5))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(cw, x0 + sw), min(ch, y0 + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sx0, sy0 = cx0 - x0, cy0 - y0
        sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
        sub_fg = r_fg[sy0:sy1, sx0:sx1]
        if not sub_fg.any():
            return
        roi = canvas[cy0:cy1, cx0:cx1]
        roi[sub_fg] = r_rgb[sy0:sy1, sx0:sx1][sub_fg]
        idx_mask[cy0:cy1, cx0:cx1][sub_fg] = obj_id

    y_off = float(rng.uniform(-14.0, 14.0))
    for t in range(num_frames):
        alpha = float(t) / float(max(1, num_frames - 1))
        # A moves from x=0.27*cw to x=0.73*cw; B moves from x=0.73*cw to x=0.27*cw
        # In the middle (alpha ~ 0.5), they closely pass & partially occlude (~35% overlap)
        cx_a = cw * (0.26 + 0.48 * alpha)
        cy_a = ch * 0.50 + 18.0 * math.sin(math.pi * alpha) + y_off * 0.3
        cx_b = cw * (0.74 - 0.48 * alpha)
        cy_b = ch * 0.52 - 16.0 * math.sin(math.pi * alpha) - y_off * 0.3

        canvas = bg.copy()
        idx_mask = np.zeros((ch, cw), dtype=np.int32)
        # Paste B (id=2) behind A (id=1) so B undergoes realistic partial occlusion mid-clip
        _paste(
            canvas,
            idx_mask,
            s_rgb_b,
            s_fg_b,
            cx_b,
            cy_b,
            2,
            -8.0 * math.sin(math.pi * alpha),
        )
        _paste(
            canvas,
            idx_mask,
            s_rgb_a,
            s_fg_a,
            cx_a,
            cy_a,
            1,
            8.0 * math.sin(math.pi * alpha),
        )

        frames_rgb.append(canvas)
        frames_mask.append(idx_mask)

    return frames_rgb, frames_mask


def load_cutie_for_eval(
    weights_path: Path,
    device: str,
    *,
    pretrained_backbone: Optional[bool] = None,
) -> Tuple[CUTIE, Any, float]:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(CUTIE_CFG_DIR)):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg.weights = str(weights_path)
        cfg.mem_every = 3
        cfg.max_mem_frames = 5
        cfg.max_internal_size = -1
        cfg.use_long_term = False
        cfg.amp = False
        cfg.chunk_size = -1
        cfg.top_k = 30
        cfg.stagger_updates = 3

    t0 = time.perf_counter()
    model = CUTIE(cfg, pretrained_backbone=pretrained_backbone).to(device).eval()
    raw_sd = safe_torch_load(weights_path, map_location=device)
    model.load_weights(unwrap_cutie_state_dict(raw_sd))
    load_ms = (time.perf_counter() - t0) * 1000.0
    return model, cfg, load_ms


def propagate_sequence_with_cutie(
    model: CUTIE,
    cfg: Any,
    frames_rgb: Sequence[np.ndarray],
    seed_mask: np.ndarray,
    device: str,
) -> List[np.ndarray]:
    """Run InferenceCore seeded at frame 0 and return predicted index masks for all frames."""
    core = InferenceCore(model, cfg=cfg)
    seeded_oids = [int(v) for v in np.unique(seed_mask) if int(v) > 0]
    if not seeded_oids:
        return [np.zeros_like(seed_mask, dtype=np.int32) for _ in frames_rgb]

    preds: List[np.ndarray] = []
    with torch.inference_mode():
        for fi, rgb in enumerate(frames_rgb):
            ft = image_to_torch(rgb, device=device)
            if fi == 0:
                mt = torch.from_numpy(np.ascontiguousarray(seed_mask, dtype=np.int32))
                out = core.step(
                    ft,
                    mt,
                    objects=seeded_oids,
                    idx_mask=True,
                    force_permanent=True,
                    return_index_mask=True,
                    output_size=rgb.shape[:2],
                )
            else:
                out = core.step(
                    ft,
                    return_index_mask=True,
                    output_size=rgb.shape[:2],
                )
            preds.append(core.output_prob_to_mask(out).cpu().numpy().astype(np.int32))
    return preds


def score_sequence_predictions(
    preds: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    seeded_oids: Sequence[int],
) -> Tuple[float, float, float]:
    j_vals: List[float] = []
    f_vals: List[float] = []
    for fi in range(1, len(gts)):
        gt_m = gts[fi]
        pr_m = preds[fi]
        for oid in seeded_oids:
            gt_b = gt_m == oid
            pr_b = pr_m == oid
            if not gt_b.any() and not pr_b.any():
                continue
            jv, fv = compute_j_and_f(pr_b, gt_b)
            j_vals.append(jv)
            f_vals.append(fv)
    if not j_vals:
        return 1.0, 1.0, 1.0
    mj = float(np.mean(j_vals))
    mf = float(np.mean(f_vals))
    return mj, mf, 0.5 * (mj + mf)


def colorize_mask_overlay(
    rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.58
) -> np.ndarray:
    color = PALETTE_RGB[np.mod(mask, len(PALETTE_RGB))]
    fg = mask > 0
    out = rgb.copy()
    if fg.any():
        blended = (
            rgb[fg].astype(np.float32) * (1.0 - alpha)
            + color[fg].astype(np.float32) * alpha
        ).astype(np.uint8)
        out[fg] = blended
        # Crisp 1px contour per object
        for oid in [int(v) for v in np.unique(mask) if int(v) > 0]:
            cnt = cv2.morphologyEx(
                (mask == oid).astype(np.uint8),
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), np.uint8),
            )
            out[cnt > 0] = (255, 255, 255)
    return out


def save_comparison_panel(
    out_path: Path,
    title: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred_base: np.ndarray,
    pred_up: np.ndarray,
    jf_base: float,
    jf_up: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    panels = [
        ("RGB Frame", rgb),
        ("Ground Truth", colorize_mask_overlay(rgb, gt)),
        (f"Base Cutie (J&F {jf_base:.3f})", colorize_mask_overlay(rgb, pred_base)),
        (f"SAM3-Cutie (J&F {jf_up:.3f})", colorize_mask_overlay(rgb, pred_up)),
    ]
    header_h = 34
    canvas = np.full((h + header_h, w * 4 + 12, 3), 24, dtype=np.uint8)
    for idx, (lbl, img_rgb) in enumerate(panels):
        x0 = idx * (w + 4)
        canvas[header_h : header_h + h, x0 : x0 + w] = cv2.cvtColor(
            img_rgb, cv2.COLOR_RGB2BGR
        )
        cv2.putText(
            canvas,
            lbl,
            (x0 + 8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), canvas)


def evaluate_all_benchmarks(
    base_weights: Path,
    upgraded_weights: Path,
    *,
    two_mice_eval_dir: Path = DEFAULT_TWO_MICE_EVAL_DIR,
    datasets_dir: Path = DEFAULT_BENCHMARK_ROOT / "datasets",
    out_dir: Path = DEFAULT_BENCHMARK_ROOT,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_device = device or str(get_device())
    print(f"Loading models on {resolved_device}...")
    # Measure legacy upstream init+load (pretrained_backbone=True) vs upgraded compact init+load
    base_model, base_cfg, base_load_ms = load_cutie_for_eval(
        base_weights, resolved_device, pretrained_backbone=True
    )
    up_model, up_cfg, up_load_ms = load_cutie_for_eval(
        upgraded_weights, resolved_device, pretrained_backbone=False
    )

    base_params = sum(p.numel() for p in base_model.parameters())
    up_params = sum(p.numel() for p in up_model.parameters())
    base_size_mb = base_weights.stat().st_size / (1024.0 * 1024.0)
    up_size_mb = upgraded_weights.stat().st_size / (1024.0 * 1024.0)

    # Warm up PyTorch allocator and convolution kernels equally on both models
    warm_rgb = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(2)]
    warm_seed = np.zeros((256, 256), dtype=np.int32)
    warm_seed[64:128, 64:128] = 1
    _ = propagate_sequence_with_cutie(
        base_model, base_cfg, warm_rgb, warm_seed, resolved_device
    )
    _ = propagate_sequence_with_cutie(
        up_model, up_cfg, warm_rgb, warm_seed, resolved_device
    )

    results: Dict[str, Any] = {
        "device": resolved_device,
        "base_weights": str(base_weights.name),
        "upgraded_weights": str(upgraded_weights.name),
        "efficiency": {
            "base_parameters": base_params,
            "upgraded_parameters": up_params,
            "parameter_reduction_pct": round(
                100.0 * (1.0 - up_params / max(base_params, 1)), 2
            ),
            "base_file_size_mb": round(base_size_mb, 2),
            "upgraded_file_size_mb": round(up_size_mb, 2),
            "size_compression_ratio": round(base_size_mb / max(up_size_mb, 1e-6), 2),
            "base_load_ms": round(base_load_ms, 1),
            "upgraded_load_ms": round(up_load_ms, 1),
            "load_speedup_x": round(base_load_ms / max(up_load_ms, 1e-6), 2),
        },
        "suites": {},
    }
    print(
        f"Efficiency summary: params {base_params:,} -> {up_params:,} "
        f"(-{results['efficiency']['parameter_reduction_pct']:.1f}%), "
        f"disk {base_size_mb:.2f} MB -> {up_size_mb:.2f} MB "
        f"({results['efficiency']['size_compression_ratio']:.2f}x), "
        f"load {base_load_ms:.0f} ms -> {up_load_ms:.0f} ms "
        f"({results['efficiency']['load_speedup_x']:.2f}x)"
    )

    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    total_frames_all = 0
    total_base_sec = 0.0
    total_up_sec = 0.0

    # Suite 1: Annolid 9-Scenario 3D Behavioral Video Benchmark
    scenarios = sorted(
        p.name
        for p in two_mice_eval_dir.iterdir()
        if (p / "ground_truth" / "sequence_gt.json").exists()
    )
    for gran in ("instance", "coarse", "fine"):
        suite_key = f"two_mice_3d_{gran}"
        b_j, b_f, b_jf = [], [], []
        u_j, u_f, u_jf = [], [], []
        suite_frames = 0
        suite_base_sec = 0.0
        suite_up_sec = 0.0
        per_scenario = {}
        for scen in scenarios:
            gt_dir = two_mice_eval_dir / scen / "ground_truth"
            img_paths = sorted((gt_dir / "images").glob("*.jpg"))
            msk_paths = sorted((gt_dir / "masks").glob("*_ids.png"))
            frames_rgb = [
                cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB) for ip in img_paths
            ]
            gts = [
                build_two_mice_gt_mask(cv2.imread(str(mp), cv2.IMREAD_UNCHANGED), gran)
                for mp in msk_paths
            ]
            oids = [int(v) for v in np.unique(gts[0]) if int(v) > 0]
            tb0 = time.perf_counter()
            p_base = propagate_sequence_with_cutie(
                base_model, base_cfg, frames_rgb, gts[0], resolved_device
            )
            tb1 = time.perf_counter()
            tu0 = time.perf_counter()
            p_up = propagate_sequence_with_cutie(
                up_model, up_cfg, frames_rgb, gts[0], resolved_device
            )
            tu1 = time.perf_counter()
            b_sec = tb1 - tb0
            u_sec = tu1 - tu0
            suite_frames += len(frames_rgb)
            suite_base_sec += b_sec
            suite_up_sec += u_sec

            bj, bf, bjf = score_sequence_predictions(p_base, gts, oids)
            uj, uf, ujf = score_sequence_predictions(p_up, gts, oids)
            b_j.append(bj)
            b_f.append(bf)
            b_jf.append(bjf)
            u_j.append(uj)
            u_f.append(uf)
            u_jf.append(ujf)
            per_scenario[scen] = {
                "base": {"J": round(bj, 4), "F": round(bf, 4), "J&F": round(bjf, 4)},
                "upgraded": {
                    "J": round(uj, 4),
                    "F": round(uf, 4),
                    "J&F": round(ujf, 4),
                },
                "delta_JF": round(ujf - bjf, 4),
                "speedup_x": round(b_sec / max(u_sec, 1e-6), 2),
            }
            if scen in ("social_huddle", "arena_explore", "rearing_oblique"):
                mid = len(frames_rgb) // 2
                save_comparison_panel(
                    previews_dir / f"{suite_key}_{scen}.png",
                    f"{suite_key}:{scen}",
                    frames_rgb[mid],
                    gts[mid],
                    p_base[mid],
                    p_up[mid],
                    bjf,
                    ujf,
                )

        total_frames_all += suite_frames
        total_base_sec += suite_base_sec
        total_up_sec += suite_up_sec
        base_fps = suite_frames / max(suite_base_sec, 1e-6)
        up_fps = suite_frames / max(suite_up_sec, 1e-6)

        results["suites"][suite_key] = {
            "num_sequences": len(scenarios),
            "num_frames": suite_frames,
            "base": {
                "J": round(float(np.mean(b_j)), 4),
                "F": round(float(np.mean(b_f)), 4),
                "J&F": round(float(np.mean(b_jf)), 4),
                "fps": round(base_fps, 2),
                "runtime_sec": round(suite_base_sec, 2),
            },
            "upgraded": {
                "J": round(float(np.mean(u_j)), 4),
                "F": round(float(np.mean(u_f)), 4),
                "J&F": round(float(np.mean(u_jf)), 4),
                "fps": round(up_fps, 2),
                "runtime_sec": round(suite_up_sec, 2),
            },
            "delta_J": round(float(np.mean(u_j) - np.mean(b_j)), 4),
            "delta_F": round(float(np.mean(u_f) - np.mean(b_f)), 4),
            "delta_JF": round(float(np.mean(u_jf) - np.mean(b_jf)), 4),
            "speedup_x": round(suite_base_sec / max(suite_up_sec, 1e-6), 2),
            "scenarios": per_scenario,
        }
        s = results["suites"][suite_key]
        print(
            f"[{suite_key:26s}] Base J&F={s['base']['J&F']:.4f} ({base_fps:5.2f} FPS) -> "
            f"Upgraded J&F={s['upgraded']['J&F']:.4f} ({up_fps:5.2f} FPS) | "
            f"Δ={s['delta_JF']:+.4f}, speedup={s['speedup_x']:.2f}x"
        )

    # Suites 2-4: Open-Source Held-Out Benchmarks (PartImageNet, Pascal-Part, Human Parsing)
    os_configs = [
        ("partimagenet", "test", 700000),
        ("pascal_part", "test", 800000),
        ("human_parsing", "test", 900000),
    ]
    for ds_name, split_name, salt in os_configs:
        manifest_path = datasets_dir / ds_name / split_name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_root = manifest_path.parent

        samples = []
        for rec in manifest:
            rgb = cv2.cvtColor(
                cv2.imread(str(split_root / rec["image"])), cv2.COLOR_BGR2RGB
            )
            pmask = cv2.imread(
                str(split_root / rec["part_mask"]), cv2.IMREAD_UNCHANGED
            ).astype(np.int32)
            wmask = cv2.imread(
                str(split_root / rec["whole_mask"]), cv2.IMREAD_UNCHANGED
            ).astype(np.int32)
            samples.append((rec["id"], rgb, pmask, wmask))

        # (a) Body-Part Articulated VOS
        bp_key = f"{ds_name}_body_parts"
        bj_l, bf_l, bjf_l, uj_l, uf_l, ujf_l = [], [], [], [], [], []
        suite_frames = 0
        suite_base_sec = 0.0
        suite_up_sec = 0.0
        for idx, (sid, rgb, pmask, _wmask) in enumerate(samples):
            f_rgb, f_gt = synthesize_articulated_part_sequence(
                rgb, pmask, num_frames=8, seed=salt + idx
            )
            oids = [int(v) for v in np.unique(f_gt[0]) if int(v) > 0]
            tb0 = time.perf_counter()
            p_base = propagate_sequence_with_cutie(
                base_model, base_cfg, f_rgb, f_gt[0], resolved_device
            )
            tb1 = time.perf_counter()
            tu0 = time.perf_counter()
            p_up = propagate_sequence_with_cutie(
                up_model, up_cfg, f_rgb, f_gt[0], resolved_device
            )
            tu1 = time.perf_counter()
            suite_frames += len(f_rgb)
            suite_base_sec += tb1 - tb0
            suite_up_sec += tu1 - tu0

            bj, bf, bjf = score_sequence_predictions(p_base, f_gt, oids)
            uj, uf, ujf = score_sequence_predictions(p_up, f_gt, oids)
            bj_l.append(bj)
            bf_l.append(bf)
            bjf_l.append(bjf)
            uj_l.append(uj)
            uf_l.append(uf)
            ujf_l.append(ujf)
            if idx == 0:
                save_comparison_panel(
                    previews_dir / f"{bp_key}_{sid}.png",
                    f"{bp_key}:{sid}",
                    f_rgb[5],
                    f_gt[5],
                    p_base[5],
                    p_up[5],
                    bjf,
                    ujf,
                )

        total_frames_all += suite_frames
        total_base_sec += suite_base_sec
        total_up_sec += suite_up_sec
        base_fps = suite_frames / max(suite_base_sec, 1e-6)
        up_fps = suite_frames / max(suite_up_sec, 1e-6)

        results["suites"][bp_key] = {
            "num_sequences": len(samples),
            "num_frames": suite_frames,
            "base": {
                "J": round(float(np.mean(bj_l)), 4),
                "F": round(float(np.mean(bf_l)), 4),
                "J&F": round(float(np.mean(bjf_l)), 4),
                "fps": round(base_fps, 2),
                "runtime_sec": round(suite_base_sec, 2),
            },
            "upgraded": {
                "J": round(float(np.mean(uj_l)), 4),
                "F": round(float(np.mean(uf_l)), 4),
                "J&F": round(float(np.mean(ujf_l)), 4),
                "fps": round(up_fps, 2),
                "runtime_sec": round(suite_up_sec, 2),
            },
            "delta_J": round(float(np.mean(uj_l) - np.mean(bj_l)), 4),
            "delta_F": round(float(np.mean(uf_l) - np.mean(bf_l)), 4),
            "delta_JF": round(float(np.mean(ujf_l) - np.mean(bjf_l)), 4),
            "speedup_x": round(suite_base_sec / max(suite_up_sec, 1e-6), 2),
        }
        s = results["suites"][bp_key]
        print(
            f"[{bp_key:26s}] Base J&F={s['base']['J&F']:.4f} ({base_fps:5.2f} FPS) -> "
            f"Upgraded J&F={s['upgraded']['J&F']:.4f} ({up_fps:5.2f} FPS) | "
            f"Δ={s['delta_JF']:+.4f}, speedup={s['speedup_x']:.2f}x"
        )

        # (b) Multi-Instance Close-Contact Crossing VOS
        inst_key = f"{ds_name}_multi_instance"
        bj_l, bf_l, bjf_l, uj_l, uf_l, ujf_l = [], [], [], [], [], []
        suite_frames = 0
        suite_base_sec = 0.0
        suite_up_sec = 0.0
        num_pairs = len(samples) // 2
        for pidx in range(num_pairs):
            sid_a, rgb_a, _, w_a = samples[2 * pidx]
            sid_b, rgb_b, _, w_b = samples[2 * pidx + 1]
            f_rgb, f_gt = synthesize_multi_instance_crossing_sequence(
                rgb_a, w_a, rgb_b, w_b, num_frames=8, seed=salt + 5000 + pidx
            )
            oids = [int(v) for v in np.unique(f_gt[0]) if int(v) > 0]
            tb0 = time.perf_counter()
            p_base = propagate_sequence_with_cutie(
                base_model, base_cfg, f_rgb, f_gt[0], resolved_device
            )
            tb1 = time.perf_counter()
            tu0 = time.perf_counter()
            p_up = propagate_sequence_with_cutie(
                up_model, up_cfg, f_rgb, f_gt[0], resolved_device
            )
            tu1 = time.perf_counter()
            suite_frames += len(f_rgb)
            suite_base_sec += tb1 - tb0
            suite_up_sec += tu1 - tu0

            bj, bf, bjf = score_sequence_predictions(p_base, f_gt, oids)
            uj, uf, ujf = score_sequence_predictions(p_up, f_gt, oids)
            bj_l.append(bj)
            bf_l.append(bf)
            bjf_l.append(bjf)
            uj_l.append(uj)
            uf_l.append(uf)
            ujf_l.append(ujf)
            if pidx == 0:
                save_comparison_panel(
                    previews_dir / f"{inst_key}_{sid_a}_{sid_b}.png",
                    f"{inst_key}:{sid_a}+{sid_b}",
                    f_rgb[4],
                    f_gt[4],
                    p_base[4],
                    p_up[4],
                    bjf,
                    ujf,
                )

        total_frames_all += suite_frames
        total_base_sec += suite_base_sec
        total_up_sec += suite_up_sec
        base_fps = suite_frames / max(suite_base_sec, 1e-6)
        up_fps = suite_frames / max(suite_up_sec, 1e-6)

        results["suites"][inst_key] = {
            "num_sequences": num_pairs,
            "num_frames": suite_frames,
            "base": {
                "J": round(float(np.mean(bj_l)), 4),
                "F": round(float(np.mean(bf_l)), 4),
                "J&F": round(float(np.mean(bjf_l)), 4),
                "fps": round(base_fps, 2),
                "runtime_sec": round(suite_base_sec, 2),
            },
            "upgraded": {
                "J": round(float(np.mean(uj_l)), 4),
                "F": round(float(np.mean(uf_l)), 4),
                "J&F": round(float(np.mean(ujf_l)), 4),
                "fps": round(up_fps, 2),
                "runtime_sec": round(suite_up_sec, 2),
            },
            "delta_J": round(float(np.mean(uj_l) - np.mean(bj_l)), 4),
            "delta_F": round(float(np.mean(uf_l) - np.mean(bf_l)), 4),
            "delta_JF": round(float(np.mean(ujf_l) - np.mean(bjf_l)), 4),
            "speedup_x": round(suite_base_sec / max(suite_up_sec, 1e-6), 2),
        }
        s = results["suites"][inst_key]
        print(
            f"[{inst_key:26s}] Base J&F={s['base']['J&F']:.4f} ({base_fps:5.2f} FPS) -> "
            f"Upgraded J&F={s['upgraded']['J&F']:.4f} ({up_fps:5.2f} FPS) | "
            f"Δ={s['delta_JF']:+.4f}, speedup={s['speedup_x']:.2f}x"
        )

    results["efficiency"]["total_frames_evaluated"] = total_frames_all
    results["efficiency"]["total_base_runtime_sec"] = round(total_base_sec, 2)
    results["efficiency"]["total_upgraded_runtime_sec"] = round(total_up_sec, 2)
    results["efficiency"]["overall_base_fps"] = round(
        total_frames_all / max(total_base_sec, 1e-6), 2
    )
    results["efficiency"]["overall_upgraded_fps"] = round(
        total_frames_all / max(total_up_sec, 1e-6), 2
    )
    results["efficiency"]["overall_speedup_x"] = round(
        total_base_sec / max(total_up_sec, 1e-6), 2
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown_report(out_dir / "README.md", results)
    return results


def _write_markdown_report(report_path: Path, results: Mapping[str, Any]) -> None:
    eff = results.get("efficiency", {})
    lines = [
        f"# Upstream Cutie (`{results['base_weights']}`) vs Directly Upgraded & Compressed Cutie (`{results['upgraded_weights']}`)",
        "",
        "## Model Efficiency Summary",
        "",
        "| Metric | Upstream Dense Cutie | Upgraded Compact Cutie | Improvement |",
        "| --- | ---: | ---: | ---: |",
        f"| Parameters | `{eff.get('base_parameters', 0):,}` | `{eff.get('upgraded_parameters', 0):,}` | `-{eff.get('parameter_reduction_pct', 0):.2f}%` |",
        f"| Checkpoint Size | `{eff.get('base_file_size_mb', 0):.2f} MB` | `{eff.get('upgraded_file_size_mb', 0):.2f} MB` | `{eff.get('size_compression_ratio', 0):.2f}x smaller` |",
        f"| Cold Model Load | `{eff.get('base_load_ms', 0):.1f} ms` | `{eff.get('upgraded_load_ms', 0):.1f} ms` | `{eff.get('load_speedup_x', 0):.2f}x faster` |",
        f"| Total Propagation Time ({eff.get('total_frames_evaluated', 0)} frames) | `{eff.get('total_base_runtime_sec', 0):.2f} s` (`{eff.get('overall_base_fps', 0):.2f} FPS`) | `{eff.get('total_upgraded_runtime_sec', 0):.2f} s` (`{eff.get('overall_upgraded_fps', 0):.2f} FPS`) | `{eff.get('overall_speedup_x', 0):.2f}x faster` |",
        "",
        "## Held-Out Multi-Domain Accuracy & Throughput",
        "",
        "| Held-Out Benchmark Suite | Seq | Base J | Base F | Base J&F | Upgraded J | Upgraded F | Upgraded J&F | Δ J&F | Base FPS | Upgraded FPS | Speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in results["suites"].items():
        b = s["base"]
        u = s["upgraded"]
        lines.append(
            f"| `{name}` | `{s['num_sequences']}` | `{b['J']:.4f}` | `{b['F']:.4f}` | "
            f"`{b['J&F']:.4f}` | `{u['J']:.4f}` | `{u['F']:.4f}` | `{u['J&F']:.4f}` | "
            f"`{s['delta_JF']:+.4f}` | `{b['fps']:.2f}` | `{u['fps']:.2f}` | `{s['speedup_x']:.2f}x` |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default_base = (
        DEFAULT_ORIG_BACKUP_WEIGHTS
        if DEFAULT_ORIG_BACKUP_WEIGHTS.is_file()
        else DEFAULT_BASE_WEIGHTS
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-weights", type=Path, default=default_base)
    parser.add_argument("--upgraded-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    evaluate_all_benchmarks(
        args.base_weights, args.upgraded_weights, device=args.device
    )
    print(f"\nCompleted full multi-domain benchmark in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
