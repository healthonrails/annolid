"""Hierarchical, ROI-cropped body-part segmentation inside tracked animal instances.

This module segments anatomical sub-regions (e.g., ``head``, ``torso``, ``tail``,
``limbs``, ``wings``) strictly inside tracked whole-instance masks. Operating on
tight per-instance crops preserves fine anatomical detail on small animals while
avoiding full-frame background compute and preventing cross-animal bleed during
close social contact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from annolid.annotation.polygon_constraints import make_instance_masks_exclusive
from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)
from annolid.utils.atomic_io import atomic_write_json
from annolid.utils.body_part_shapes import (
    build_body_part_shape_dict,
    extract_body_part_label,
    extract_body_part_parent_instance,
    infer_part_from_keypoint_name,
    is_body_part_shape_payload,
    keypoint_axial_polarity,
)
from annolid.utils.mask_geometry import tight_mask_bbox
from annolid.utils.zone_shapes import is_zone_shape_payload

VALID_MODES = ("auto", "seeded", "keypoint_guided", "unsupervised")
VALID_FEATURE_BACKENDS = ("fast", "dino", "hybrid")
VALID_GRANULARITIES = ("coarse", "fine", "per_keypoint")
_UNASSIGNED_PART = "__unassigned__"


@dataclass(slots=True)
class BodyPartSegmentationConfig:
    """Configuration for intra-instance body-part segmentation."""

    mode: str = "auto"
    feature_backend: str = "fast"
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    crop_size: int = 224
    crop_padding_ratio: float = 0.12
    num_parts: int = 3
    part_names: Tuple[str, ...] = ()
    keypoint_granularity: str = "coarse"
    keypoint_part_map: Dict[str, str] = field(default_factory=dict)
    partition_full_instance: Optional[bool] = None
    share_seed_across_instances: bool = True
    appearance_weight: float = 1.0
    canonical_prior_weight: float = 0.55
    temporal_prior_weight: float = 0.25
    keypoint_boost_weight: float = 0.45
    geodesic_edge_weight: float = 3.5
    max_part_area_growth: float = 2.5
    min_part_area_px: int = 6
    ema_alpha: float = 0.12
    ema_min_confidence: float = 0.65
    boundary_refine: bool = True
    guided_radius: int = 4
    guided_eps: float = 1e-3
    crf_refine: bool = False
    crf_backend: str = "opencv"
    crf_band_px: int = 8
    crf_iterations: int = 6
    polygon_epsilon: float = 1.0
    qualify_label_with_instance: bool = False
    persist_json: bool = False
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode or "auto").strip().lower()
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported body-part segmentation mode '{self.mode}'. "
                f"Expected one of {VALID_MODES}."
            )
        self.feature_backend = str(self.feature_backend or "fast").strip().lower()
        if self.feature_backend not in VALID_FEATURE_BACKENDS:
            raise ValueError(
                f"Unsupported feature_backend '{self.feature_backend}'. "
                f"Expected one of {VALID_FEATURE_BACKENDS}."
            )
        self.keypoint_granularity = (
            str(self.keypoint_granularity or "coarse").strip().lower()
        )
        if self.keypoint_granularity not in VALID_GRANULARITIES:
            raise ValueError(
                f"Unsupported keypoint_granularity '{self.keypoint_granularity}'. "
                f"Expected one of {VALID_GRANULARITIES}."
            )
        self.crop_size = max(32, int(self.crop_size))
        self.crop_padding_ratio = max(0.0, float(self.crop_padding_ratio))
        self.num_parts = max(2, int(self.num_parts))
        self.part_names = tuple(
            str(name).strip() for name in self.part_names if str(name).strip()
        )
        self.keypoint_part_map = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (self.keypoint_part_map or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.min_part_area_px = max(1, int(self.min_part_area_px))
        self.max_part_area_growth = max(1.05, float(self.max_part_area_growth))
        self.polygon_epsilon = max(0.0, float(self.polygon_epsilon))


@dataclass(frozen=True, slots=True)
class BodyFramePose:
    """Oriented 2D canonical coordinate frame of an animal instance mask."""

    centroid_x: float
    centroid_y: float
    axial_vec: Tuple[float, float]
    lateral_vec: Tuple[float, float]
    axial_scale: float
    lateral_scale: float
    aspect_ratio: float
    u_map: np.ndarray
    v_map: np.ndarray
    thickness_map: np.ndarray


@dataclass(slots=True)
class BodyPartPrototype:
    """Seed-anchored + EMA appearance and canonical body-frame statistics for one part."""

    part_label: str
    seed_prototypes: np.ndarray  # (P, C), L2-normalized
    working_prototype: np.ndarray  # (C,), L2-normalized
    mean_u: float
    std_u: float
    mean_v: float
    std_v: float
    mean_thickness: float
    std_thickness: float
    seed_area_ratio: float
    is_unassigned: bool = False


@dataclass(slots=True)
class PartPrediction:
    """Predicted intra-instance body-part mask and polygon in full-frame coordinates."""

    part_label: str
    parent_instance: str
    crop_mask: np.ndarray  # bool (crop_h, crop_w)
    crop_bbox: Tuple[int, int, int, int]  # (cx0, cy0, cx1, cy1) in full frame
    polygon: List[Tuple[float, float]]
    confidence: float
    area_px: int
    area_ratio: float
    group_id: Optional[int | str] = None

    def to_full_mask(self, height: int, width: int) -> np.ndarray:
        """Materialize a full-frame boolean mask ``(height, width)`` on demand."""
        full = np.zeros((int(height), int(width)), dtype=bool)
        cx0, cy0, cx1, cy1 = self.crop_bbox
        sub_h = max(0, min(int(height), cy1) - max(0, cy0))
        sub_w = max(0, min(int(width), cx1) - max(0, cx0))
        if sub_h > 0 and sub_w > 0:
            full[cy0 : cy0 + sub_h, cx0 : cx0 + sub_w] = self.crop_mask[:sub_h, :sub_w]
        return full


@dataclass(slots=True)
class _InstanceCrop:
    crop_rgb: np.ndarray
    crop_mask: np.ndarray
    work_rgb: np.ndarray
    work_mask: np.ndarray
    crop_bbox: Tuple[int, int, int, int]  # (cx0, cy0, cx1, cy1)
    scale_x: float
    scale_y: float


@dataclass(slots=True)
class _TemporalInstanceState:
    pose: BodyFramePose
    part_labels: Tuple[str, ...]
    work_probs: np.ndarray  # (K, work_h, work_w)
    u_grid: np.ndarray
    v_grid: np.ndarray


def _l2_normalize_channels(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=0, keepdims=True)
    return arr / np.maximum(norms, eps)


def _l2_normalize_vec(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def _prepare_instance_crop(
    frame_rgb: np.ndarray,
    instance_mask: np.ndarray,
    *,
    crop_size: int,
    padding_ratio: float,
) -> Optional[_InstanceCrop]:
    mask_bool = np.asarray(instance_mask, dtype=bool)
    bbox = tight_mask_bbox(mask_bool)
    if bbox is None:
        return None
    H, W = mask_bool.shape
    x0, y0, x1, y1 = bbox
    box_span = max(x1 - x0, y1 - y0)
    pad = max(6, int(round(box_span * float(padding_ratio))))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(W, x1 + pad)
    cy1 = min(H, y1 + pad)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    crop_rgb = np.asarray(frame_rgb[cy0:cy1, cx0:cx1], dtype=np.uint8)
    crop_mask = mask_bool[cy0:cy1, cx0:cx1].copy()
    if not np.any(crop_mask):
        return None

    ch, cw = crop_mask.shape
    max_side = max(ch, cw)
    if max_side > crop_size:
        ratio = float(crop_size) / float(max_side)
        wh = max(8, int(round(ch * ratio)))
        ww = max(8, int(round(cw * ratio)))
        work_rgb = cv2.resize(crop_rgb, (ww, wh), interpolation=cv2.INTER_AREA)
        work_mask = (
            cv2.resize(
                crop_mask.astype(np.uint8),
                (ww, wh),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        if not np.any(work_mask):
            work_mask = (
                cv2.resize(
                    crop_mask.astype(np.float32),
                    (ww, wh),
                    interpolation=cv2.INTER_LINEAR,
                )
                > 0.05
            )
        scale_x = float(ww) / float(cw)
        scale_y = float(wh) / float(ch)
    else:
        work_rgb = crop_rgb
        work_mask = crop_mask
        scale_x = 1.0
        scale_y = 1.0

    return _InstanceCrop(
        crop_rgb=crop_rgb,
        crop_mask=crop_mask,
        work_rgb=work_rgb,
        work_mask=work_mask,
        crop_bbox=(cx0, cy0, cx1, cy1),
        scale_x=scale_x,
        scale_y=scale_y,
    )


def estimate_body_frame_pose(
    work_mask: np.ndarray,
    *,
    keypoints_work: Optional[Mapping[str, Tuple[float, float]]] = None,
    keypoint_part_map: Optional[Mapping[str, str]] = None,
    prev_pose: Optional[BodyFramePose] = None,
    features: Optional[np.ndarray] = None,
    prototypes: Optional[Sequence[BodyPartPrototype]] = None,
) -> BodyFramePose:
    """Compute an oriented, polarity-disambiguated canonical 2D pose for ``work_mask``."""
    mask_bool = np.asarray(work_mask, dtype=bool)
    wh, ww = mask_bool.shape
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        zeros = np.zeros((wh, ww), dtype=np.float32)
        return BodyFramePose(
            centroid_x=0.5 * ww,
            centroid_y=0.5 * wh,
            axial_vec=(1.0, 0.0),
            lateral_vec=(0.0, 1.0),
            axial_scale=max(4.0, 0.5 * ww),
            lateral_scale=max(4.0, 0.5 * wh),
            aspect_ratio=1.0,
            u_map=zeros,
            v_map=zeros,
            thickness_map=zeros,
        )

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    mx = float(xs_f.mean())
    my = float(ys_f.mean())
    dx = xs_f - mx
    dy = ys_f - my

    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))
    trace = cxx + cyy
    disc = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam1 = max(0.25, 0.5 * (trace + disc))
    lam2 = max(0.25, 0.5 * (trace - disc))
    aspect_ratio = math.sqrt(lam1 / max(lam2, 1e-6))

    if disc < 1e-4 and prev_pose is not None:
        ax_x, ax_y = prev_pose.axial_vec
    elif abs(cxy) > 1e-6 or cxx >= cyy:
        vx = lam1 - cyy
        vy = cxy
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            ax_x, ax_y = 1.0, 0.0
        else:
            ax_x, ax_y = vx / norm, vy / norm
    else:
        vx = cxy
        vy = lam1 - cxx
        norm = math.hypot(vx, vy)
        if norm <= 1e-6:
            ax_x, ax_y = 0.0, 1.0
        else:
            ax_x, ax_y = vx / norm, vy / norm

    axial_scale = max(3.0, 2.0 * math.sqrt(lam1))
    lateral_scale = max(3.0, 2.0 * math.sqrt(lam2))

    edt = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    max_edt = float(edt.max())
    thickness_map = (
        (edt / max_edt).astype(np.float32)
        if max_edt > 1e-6
        else np.zeros((wh, ww), dtype=np.float32)
    )

    grid_y, grid_x = np.meshgrid(
        np.arange(wh, dtype=np.float32),
        np.arange(ww, dtype=np.float32),
        indexing="ij",
    )
    rel_x = grid_x - mx
    rel_y = grid_y - my
    raw_u = (rel_x * ax_x + rel_y * ax_y) / axial_scale
    raw_v = (rel_x * (-ax_y) + rel_y * ax_x) / lateral_scale

    # Disambiguate +1 vs -1 anterior-posterior polarity.
    kp_polarity_score = 0.0
    if keypoints_work:
        ant_pts: List[Tuple[float, float, float]] = []
        post_pts: List[Tuple[float, float, float]] = []
        kp_map = keypoint_part_map or {}
        for kp_name, (kx, ky) in keypoints_work.items():
            pol = keypoint_axial_polarity(kp_name)
            mapped = kp_map.get(kp_name.strip().lower(), "").lower()
            if pol == 0.0 and mapped:
                if "head" in mapped or "anterior" in mapped:
                    pol = 1.0
                elif "tail" in mapped or "posterior" in mapped:
                    pol = -1.0
            if pol > 0.0:
                ant_pts.append((float(kx), float(ky), pol))
            elif pol < 0.0:
                post_pts.append((float(kx), float(ky), -pol))
        if ant_pts and post_pts:
            ax_ant = sum(px * w for px, _py, w in ant_pts) / sum(
                w for _, _, w in ant_pts
            )
            ay_ant = sum(py * w for _px, py, w in ant_pts) / sum(
                w for _, _, w in ant_pts
            )
            ax_post = sum(px * w for px, _py, w in post_pts) / sum(
                w for _, _, w in post_pts
            )
            ay_post = sum(py * w for _px, py, w in post_pts) / sum(
                w for _, _, w in post_pts
            )
            kp_polarity_score = (ax_ant - ax_post) * ax_x + (ay_ant - ay_post) * ax_y
        elif ant_pts:
            ax_ant = sum(px * w for px, _py, w in ant_pts) / sum(
                w for _, _, w in ant_pts
            )
            ay_ant = sum(py * w for _px, py, w in ant_pts) / sum(
                w for _, _, w in ant_pts
            )
            kp_polarity_score = (ax_ant - mx) * ax_x + (ay_ant - my) * ax_y
        elif post_pts:
            ax_post = sum(px * w for px, _py, w in post_pts) / sum(
                w for _, _, w in post_pts
            )
            ay_post = sum(py * w for _px, py, w in post_pts) / sum(
                w for _, _, w in post_pts
            )
            kp_polarity_score = (mx - ax_post) * ax_x + (my - ay_post) * ax_y

    sign = 1.0
    if abs(kp_polarity_score) > 1e-3:
        sign = 1.0 if kp_polarity_score >= 0.0 else -1.0
    else:
        # Bidirectional hypothesis check using seeded prototypes + thickness + temporal prior.
        score_pos = 0.0
        score_neg = 0.0
        if prev_pose is not None:
            cos_prev = ax_x * prev_pose.axial_vec[0] + ax_y * prev_pose.axial_vec[1]
            score_pos += 0.30 * cos_prev
            score_neg -= 0.30 * cos_prev

        if features is not None and prototypes:
            fg_feats = features[:, mask_bool]  # (C, N)
            fg_u = raw_u[mask_bool]
            fg_v = raw_v[mask_bool]
            for proto in prototypes:
                if proto.is_unassigned or (
                    abs(proto.mean_u) < 0.15 and abs(proto.mean_v) < 0.15
                ):
                    continue
                sim = np.max(proto.seed_prototypes @ fg_feats, axis=0)  # (N,)
                weights = np.clip(sim, 0.0, 1.0) ** 2
                w_sum = float(weights.sum())
                if w_sum <= 1e-6:
                    continue
                est_u = float(np.sum(weights * fg_u) / w_sum)
                est_v = float(np.sum(weights * fg_v) / w_sum)
                score_pos -= (
                    (est_u - proto.mean_u) ** 2 / max(proto.std_u**2, 0.08)
                    + 0.5 * (est_v - proto.mean_v) ** 2 / max(proto.std_v**2, 0.08)
                ) * proto.seed_area_ratio
                score_neg -= (
                    (-est_u - proto.mean_u) ** 2 / max(proto.std_u**2, 0.08)
                    + 0.5 * (-est_v - proto.mean_v) ** 2 / max(proto.std_v**2, 0.08)
                ) * proto.seed_area_ratio
        elif prev_pose is None:
            # Mild morphological asymmetry tie-breaker: anterior head/thorax thicker than tail taper.
            fg_u = raw_u[mask_bool]
            fg_t = thickness_map[mask_bool]
            asym = float(np.mean(np.sign(fg_u) * fg_t))
            score_pos += 0.15 * asym
            score_neg -= 0.15 * asym

        sign = 1.0 if score_pos >= score_neg else -1.0

    ax_x *= sign
    ax_y *= sign
    lat_x = -ax_y
    lat_y = ax_x
    u_map = (sign * raw_u).astype(np.float32)
    v_map = (sign * raw_v).astype(np.float32)

    return BodyFramePose(
        centroid_x=mx,
        centroid_y=my,
        axial_vec=(float(ax_x), float(ax_y)),
        lateral_vec=(float(lat_x), float(lat_y)),
        axial_scale=float(axial_scale),
        lateral_scale=float(lateral_scale),
        aspect_ratio=float(aspect_ratio),
        u_map=u_map,
        v_map=v_map,
        thickness_map=thickness_map,
    )


def _extract_fast_crop_features(
    work_rgb: np.ndarray,
    work_mask: np.ndarray,
) -> np.ndarray:
    """Extract fast illumination-normalized multi-scale Lab + gradient features ``(C, H, W)``."""
    lab = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask_bool = np.asarray(work_mask, dtype=bool)
    if np.any(mask_bool):
        l_vals = lab[:, :, 0][mask_bool]
        l_mean = float(l_vals.mean())
        l_std = max(12.0, float(l_vals.std()))
    else:
        l_mean = 128.0
        l_std = 32.0

    l_norm = (lab[:, :, 0] - l_mean) / l_std
    a_norm = (lab[:, :, 1] - 128.0) / 24.0
    b_norm = (lab[:, :, 2] - 128.0) / 24.0
    base = np.stack([l_norm, a_norm, b_norm], axis=-1)  # (H, W, 3)

    smooth_fine = cv2.GaussianBlur(base, (0, 0), sigmaX=1.4, sigmaY=1.4)
    smooth_coarse = cv2.GaussianBlur(base, (0, 0), sigmaX=3.5, sigmaY=3.5)

    gx = cv2.Sobel(l_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    if np.any(mask_bool):
        g_scale = max(0.25, float(np.percentile(grad_mag[mask_bool], 90)))
    else:
        g_scale = 1.0
    grad_norm = np.clip(grad_mag / g_scale, 0.0, 2.0)
    local_detail = np.linalg.norm(smooth_fine - smooth_coarse, axis=-1)

    stacked = np.concatenate(
        [
            base,
            smooth_fine,
            0.75 * smooth_coarse,
            0.45 * grad_norm[..., None],
            0.45 * local_detail[..., None],
        ],
        axis=-1,
    )  # (H, W, 11)
    chw = np.transpose(stacked, (2, 0, 1)).astype(np.float32, copy=False)
    if np.any(mask_bool):
        fg_mean = chw[:, mask_bool].mean(axis=1, keepdims=True)[:, :, None]
        chw = chw - 0.25 * fg_mean
    return _l2_normalize_channels(chw)


def _compute_crop_edge_strength(
    work_rgb: np.ndarray, work_mask: np.ndarray
) -> np.ndarray:
    """Compute normalized visual boundary strength ``[0, 1]`` inside ``work_mask``."""
    lab = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    smoothed = cv2.GaussianBlur(lab, (0, 0), sigmaX=1.0, sigmaY=1.0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(np.sum(gx * gx + gy * gy, axis=-1))
    mask_bool = np.asarray(work_mask, dtype=bool)
    if np.any(mask_bool):
        p95 = float(np.percentile(mag[mask_bool], 95))
        scale = max(1.0, p95)
    else:
        scale = max(1.0, float(mag.max()))
    return np.clip(mag / scale, 0.0, 1.5).astype(np.float32)


def compute_masked_geodesic_distances(
    seed_masks: Sequence[np.ndarray],
    domain_mask: np.ndarray,
    *,
    edge_strength: Optional[np.ndarray] = None,
    edge_weight: float = 3.5,
) -> np.ndarray:
    """Compute edge-weighted geodesic distances ``(K, H, W)`` strictly inside ``domain_mask``.

    Paths cannot cross background pixels ``~domain_mask``, preventing curled/coiled
    appendages (e.g. a tail tip near the nose) from bleeding across air gaps.
    """
    domain = np.asarray(domain_mask, dtype=bool)
    H, W = domain.shape
    num_seeds = len(seed_masks)
    if num_seeds == 0:
        return np.zeros((0, H, W), dtype=np.float32)
    if not np.any(domain):
        return np.full((num_seeds, H, W), np.inf, dtype=np.float32)

    cost = np.ones((H, W), dtype=np.float32)
    if edge_strength is not None and float(edge_weight) > 0.0:
        cost = cost + float(edge_weight) * np.asarray(edge_strength, dtype=np.float32)

    # Try fast C-level multi-source Dijkstra over foreground pixels when scipy is installed.
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        node_ids = np.full((H, W), -1, dtype=np.int32)
        fg_ys, fg_xs = np.nonzero(domain)
        num_fg = int(fg_xs.size)
        node_ids[fg_ys, fg_xs] = np.arange(num_fg, dtype=np.int32)

        src_list: List[np.ndarray] = []
        dst_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        for dy, dx, step_len in (
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
        ):
            y0_s = slice(0, H - dy)
            y1_s = slice(dy, H)
            if dx >= 0:
                x0_s = slice(0, W - dx)
                x1_s = slice(dx, W)
            else:
                x0_s = slice(-dx, W)
                x1_s = slice(0, W + dx)
            u = node_ids[y0_s, x0_s]
            v = node_ids[y1_s, x1_s]
            valid = (u >= 0) & (v >= 0)
            if not np.any(valid):
                continue
            u_valid = u[valid]
            v_valid = v[valid]
            c_uv = (
                0.5 * step_len * (cost[y0_s, x0_s][valid] + cost[y1_s, x1_s][valid])
            ).astype(np.float32)
            src_list.append(u_valid)
            dst_list.append(v_valid)
            w_list.append(c_uv)

        if src_list:
            rows = np.concatenate(src_list)
            cols = np.concatenate(dst_list)
            weights = np.concatenate(w_list)
            graph = csr_matrix((weights, (rows, cols)), shape=(num_fg, num_fg))
        else:
            graph = csr_matrix((num_fg, num_fg), dtype=np.float32)

        out = np.full((num_seeds, H, W), np.inf, dtype=np.float32)
        for idx, smask in enumerate(seed_masks):
            s_bool = np.asarray(smask, dtype=bool) & domain
            seed_nodes = node_ids[s_bool]
            if seed_nodes.size == 0:
                continue
            dists_fg = dijkstra(
                graph,
                directed=False,
                indices=seed_nodes,
                min_only=True,
            )
            out[idx, fg_ys, fg_xs] = dists_fg.astype(np.float32)
        return out
    except ImportError:
        pass

    # Pure-NumPy fallback: vectorized 8-neighbor Bellman-Ford on compact grid (<= 112 px).
    max_grid = 112
    if max(H, W) > max_grid:
        scale = float(max_grid) / float(max(H, W))
        gh = max(8, int(round(H * scale)))
        gw = max(8, int(round(W * scale)))
        g_domain = (
            cv2.resize(
                domain.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
        if not np.any(g_domain):
            g_domain = (
                cv2.resize(
                    domain.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR
                )
                > 0.05
            )
        g_cost = cv2.resize(cost, (gw, gh), interpolation=cv2.INTER_AREA) / scale
        g_seeds = []
        for smask in seed_masks:
            s_bool = np.asarray(smask, dtype=bool) & domain
            gs = (
                cv2.resize(
                    s_bool.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST
                )
                > 0
            ) & g_domain
            if not np.any(gs) and np.any(s_bool):
                ys_s, xs_s = np.nonzero(s_bool)
                cy_s = min(gh - 1, max(0, int(round(float(ys_s.mean()) * gh / H))))
                cx_s = min(gw - 1, max(0, int(round(float(xs_s.mean()) * gw / W))))
                gs[cy_s, cx_s] = True
                g_domain[cy_s, cx_s] = True
            g_seeds.append(gs)
    else:
        gh, gw = H, W
        g_domain = domain
        g_cost = cost
        g_seeds = [np.asarray(sm, dtype=bool) & domain for sm in seed_masks]

    dist = np.full((num_seeds, gh, gw), np.inf, dtype=np.float32)
    for idx, gs in enumerate(g_seeds):
        dist[idx, gs] = 0.0

    ch = 0.5 * (g_cost[:, :-1] + g_cost[:, 1:])
    cv_step = 0.5 * (g_cost[:-1, :] + g_cost[1:, :])
    cd1 = 0.5 * math.sqrt(2.0) * (g_cost[:-1, :-1] + g_cost[1:, 1:])
    cd2 = 0.5 * math.sqrt(2.0) * (g_cost[:-1, 1:] + g_cost[1:, :-1])
    inv_dom = ~g_domain

    max_iters = 2 * (gh + gw) + 8
    for _ in range(max_iters):
        prev = dist.copy()
        dist[:, :, 1:] = np.minimum(dist[:, :, 1:], dist[:, :, :-1] + ch)
        dist[:, :, :-1] = np.minimum(dist[:, :, :-1], dist[:, :, 1:] + ch)
        dist[:, 1:, :] = np.minimum(dist[:, 1:, :], dist[:, :-1, :] + cv_step)
        dist[:, :-1, :] = np.minimum(dist[:, :-1, :], dist[:, 1:, :] + cv_step)
        dist[:, 1:, 1:] = np.minimum(dist[:, 1:, 1:], dist[:, :-1, :-1] + cd1)
        dist[:, :-1, :-1] = np.minimum(dist[:, :-1, :-1], dist[:, 1:, 1:] + cd1)
        dist[:, 1:, :-1] = np.minimum(dist[:, 1:, :-1], dist[:, :-1, 1:] + cd2)
        dist[:, :-1, 1:] = np.minimum(dist[:, :-1, 1:], dist[:, 1:, :-1] + cd2)
        dist[:, inv_dom] = np.inf
        cur_finite = np.isfinite(dist)
        prev_finite = np.isfinite(prev)
        if np.array_equal(cur_finite, prev_finite) and (
            not np.any(cur_finite)
            or float(np.max(np.abs(dist[cur_finite] - prev[cur_finite]))) < 1e-3
        ):
            break

    if (gh, gw) == (H, W):
        return dist

    out_full = np.full((num_seeds, H, W), np.inf, dtype=np.float32)
    for idx in range(num_seeds):
        finite_slice = np.isfinite(dist[idx])
        if not np.any(finite_slice):
            continue
        cap_val = float(np.max(dist[idx][finite_slice])) + 4.0 * max(H, W)
        filled = np.where(finite_slice, dist[idx], cap_val).astype(np.float32)
        up = cv2.resize(filled, (W, H), interpolation=cv2.INTER_LINEAR)
        reach = (
            cv2.resize(
                finite_slice.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        ) & domain
        out_full[idx, reach] = up[reach]
    return out_full


def guided_filter_probabilities(
    guide_rgb: np.ndarray,
    probs: np.ndarray,
    domain_mask: np.ndarray,
    *,
    radius: int = 4,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving O(1) guided image filter on multi-part probabilities ``(K, H, W)``."""
    K, H, W = probs.shape
    if K == 0 or radius <= 0 or H < 4 or W < 4:
        return probs.copy()
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE)
    corr_i = cv2.boxFilter(
        guide * guide, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE
    )
    var_i = np.maximum(0.0, corr_i - mean_i * mean_i)

    refined = np.zeros_like(probs, dtype=np.float32)
    for k in range(K):
        p = probs[k].astype(np.float32, copy=False)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE)
        corr_ip = cv2.boxFilter(
            guide * p, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE
        )
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + float(eps))
        b = mean_p - a * mean_i
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REPLICATE)
        refined[k] = np.clip(mean_a * guide + mean_b, 0.0, 1.0)

    mask_bool = np.asarray(domain_mask, dtype=bool)
    refined[:, ~mask_bool] = 0.0
    denom = np.sum(refined, axis=0, keepdims=True)
    valid = (denom > 1e-6) & mask_bool[None, :, :]
    out = np.where(valid, refined / np.maximum(denom, 1e-6), probs)
    out[:, ~mask_bool] = 0.0
    return out.astype(np.float32, copy=False)


def _select_diverse_prototypes(
    fg_features: np.ndarray,
    max_prototypes: int = 4,
) -> np.ndarray:
    """Deterministically select up to ``max_prototypes`` diverse L2-normalized prototypes."""
    # fg_features: (C, N), L2-normalized columns.
    C, N = fg_features.shape
    if N == 0:
        return np.zeros((1, C), dtype=np.float32)
    mean_proto = _l2_normalize_vec(fg_features.mean(axis=1))
    if N <= 4 or max_prototypes <= 1:
        return mean_proto[None, :]

    chosen: List[np.ndarray] = [mean_proto]
    min_dist = 1.0 - (mean_proto @ fg_features)  # cosine distance (N,)
    for _ in range(max_prototypes - 1):
        idx = int(np.argmax(min_dist))
        if float(min_dist[idx]) < 0.08:
            break
        cand = _l2_normalize_vec(fg_features[:, idx])
        # Refine prototype as mean of its local neighborhood for noise robustness.
        sims = cand @ fg_features
        top_mask = sims >= max(0.75, float(np.percentile(sims, 80)))
        if np.any(top_mask):
            cand = _l2_normalize_vec(fg_features[:, top_mask].mean(axis=1))
        chosen.append(cand)
        min_dist = np.minimum(min_dist, 1.0 - (cand @ fg_features))
    return np.stack(chosen, axis=0).astype(np.float32, copy=False)


class InstanceBodyPartSegmenter:
    """Robust, ROI-cropped intra-instance body-part segmenter."""

    def __init__(
        self,
        config: Optional[BodyPartSegmentationConfig] = None,
        *,
        extractor: Optional[Any] = None,
    ) -> None:
        self.config = config or BodyPartSegmentationConfig()
        self._extractor = extractor
        self._crf_refiner: Optional[Any] = None
        self._instance_prototypes: Dict[str, List[BodyPartPrototype]] = {}
        self._shared_prototypes: List[BodyPartPrototype] = []
        self._unsupervised_prototypes: Dict[str, List[BodyPartPrototype]] = {}
        self._temporal_states: Dict[str, _TemporalInstanceState] = {}

    @property
    def extractor(self) -> Optional[Any]:
        if self._extractor is None and self.config.feature_backend in {
            "dino",
            "hybrid",
        }:
            from annolid.features import Dinov3Config, Dinov3FeatureExtractor

            self._extractor = Dinov3FeatureExtractor(
                Dinov3Config(
                    model_name=self.config.model_name,
                    short_side=max(128, int(self.config.crop_size)),
                    device=self.config.device,
                    use_amp=True,
                    return_layer="all",
                )
            )
        return self._extractor

    @property
    def crf_refiner(self) -> Optional[Any]:
        if not self.config.crf_refine:
            return None
        if self._crf_refiner is None:
            from annolid.segmentation.crf_refinement import (
                CrfMaskRefiner,
                CrfRefinementConfig,
            )

            self._crf_refiner = CrfMaskRefiner(
                CrfRefinementConfig(
                    enabled=True,
                    backend=self.config.crf_backend,
                    band_px=self.config.crf_band_px,
                    iterations=self.config.crf_iterations,
                )
            )
        return self._crf_refiner

    def reset_temporal_state(self) -> None:
        """Clear frame-to-frame temporal priors while retaining fitted seed prototypes."""
        self._temporal_states.clear()

    def _extract_crop_features(self, crop: _InstanceCrop) -> np.ndarray:
        wh, ww = crop.work_mask.shape
        backend = self.config.feature_backend
        if backend == "fast":
            return _extract_fast_crop_features(crop.work_rgb, crop.work_mask)

        from PIL import Image
        import torch

        extractor = self.extractor
        if extractor is None:
            return _extract_fast_crop_features(crop.work_rgb, crop.work_mask)

        pil_img = Image.fromarray(crop.work_rgb)
        raw_feats = extractor.extract(pil_img, return_layer="all", normalize=True)
        if isinstance(raw_feats, torch.Tensor):
            feat_np = raw_feats.detach().cpu().float().numpy()
        else:
            feat_np = np.asarray(raw_feats, dtype=np.float32)
        if feat_np.ndim == 4:
            feat_np = feat_np[-2:].mean(axis=0) if feat_np.shape[0] >= 2 else feat_np[0]
        if feat_np.ndim != 3:
            raise ValueError(
                f"Expected 3D or 4D feature tensor ([L,] C, gh, gw), received {feat_np.shape}."
            )
        C, gh, gw = feat_np.shape
        grid_mask = (
            cv2.resize(
                crop.work_mask.astype(np.uint8),
                (gw, gh),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        if np.any(grid_mask):
            fg_mean = feat_np[:, grid_mask].mean(axis=1, keepdims=True)[:, :, None]
        else:
            fg_mean = feat_np.mean(axis=(1, 2), keepdims=True)
        debiased = _l2_normalize_channels(feat_np - 0.35 * fg_mean)
        if (gh, gw) != (wh, ww):
            hwc = np.transpose(debiased, (1, 2, 0))
            # OpenCV resize handles arbitrary channel counts cleanly via chunking > 512 if needed.
            if C <= 512:
                up_hwc = cv2.resize(hwc, (ww, wh), interpolation=cv2.INTER_LINEAR)
                if up_hwc.ndim == 2:
                    up_hwc = up_hwc[:, :, None]
            else:
                chunks = [
                    cv2.resize(
                        hwc[:, :, i : i + 256],
                        (ww, wh),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for i in range(0, C, 256)
                ]
                up_hwc = np.concatenate(
                    [c[:, :, None] if c.ndim == 2 else c for c in chunks], axis=-1
                )
            dino_feats = _l2_normalize_channels(np.transpose(up_hwc, (2, 0, 1)))
        else:
            dino_feats = debiased

        if backend == "dino":
            return dino_feats

        fast_feats = _extract_fast_crop_features(crop.work_rgb, crop.work_mask)
        fused = np.concatenate([0.82 * dino_feats, 0.58 * fast_feats], axis=0)
        return _l2_normalize_channels(fused)

    @staticmethod
    def _scale_keypoints_to_work(
        keypoints: Optional[Mapping[str, Tuple[float, float]]],
        crop: _InstanceCrop,
    ) -> Dict[str, Tuple[float, float]]:
        if not keypoints:
            return {}
        cx0, cy0, _cx1, _cy1 = crop.crop_bbox
        wh, ww = crop.work_mask.shape
        scaled: Dict[str, Tuple[float, float]] = {}
        for name, pt in keypoints.items():
            if pt is None or len(pt) < 2:
                continue
            kx, ky = float(pt[0]), float(pt[1])
            if not (math.isfinite(kx) and math.isfinite(ky)):
                continue
            wx = (kx - cx0) * crop.scale_x
            wy = (ky - cy0) * crop.scale_y
            if -4.0 <= wx <= ww + 4.0 and -4.0 <= wy <= wh + 4.0:
                scaled[str(name)] = (
                    float(np.clip(wx, 0.0, max(0.0, ww - 1.0))),
                    float(np.clip(wy, 0.0, max(0.0, wh - 1.0))),
                )
        return scaled

    def _build_prototype_from_mask(
        self,
        *,
        part_label: str,
        part_work_mask: np.ndarray,
        instance_work_mask: np.ndarray,
        features: np.ndarray,
        pose: BodyFramePose,
        is_unassigned: bool = False,
    ) -> Optional[BodyPartPrototype]:
        pmask = np.asarray(part_work_mask, dtype=bool) & np.asarray(
            instance_work_mask, dtype=bool
        )
        inst_area = max(1, int(np.count_nonzero(instance_work_mask)))
        part_area = int(np.count_nonzero(pmask))
        if part_area == 0:
            return None

        fg_feats = features[:, pmask]  # (C, N)
        seed_protos = _select_diverse_prototypes(
            fg_feats, max_prototypes=5 if is_unassigned else 4
        )
        working = _l2_normalize_vec(fg_feats.mean(axis=1))
        u_vals = pose.u_map[pmask]
        v_vals = pose.v_map[pmask]
        t_vals = pose.thickness_map[pmask]
        return BodyPartPrototype(
            part_label=part_label,
            seed_prototypes=seed_protos,
            working_prototype=working,
            mean_u=float(u_vals.mean()),
            std_u=max(0.16, float(u_vals.std()) + 0.08),
            mean_v=float(v_vals.mean()),
            std_v=max(0.20, float(v_vals.std()) + 0.10),
            mean_thickness=float(t_vals.mean()),
            std_thickness=max(0.15, float(t_vals.std()) + 0.08),
            seed_area_ratio=float(part_area) / float(inst_area),
            is_unassigned=is_unassigned,
        )

    def fit_seed_frame(
        self,
        frame_rgb: np.ndarray,
        instance_masks: Mapping[str, np.ndarray],
        seed_parts_by_instance: Mapping[str, Mapping[str, np.ndarray]],
        *,
        keypoints_by_instance: Optional[
            Mapping[str, Mapping[str, Tuple[float, float]]]
        ] = None,
    ) -> int:
        """Register or merge manual body-part seed masks inside parent ``instance_masks``.

        Returns the number of part prototypes built on this frame.
        """
        built_count = 0
        shared_accum: Dict[str, List[BodyPartPrototype]] = defaultdict(list)

        for inst_label, inst_mask in instance_masks.items():
            part_map = seed_parts_by_instance.get(inst_label)
            if not part_map:
                continue
            crop = _prepare_instance_crop(
                frame_rgb,
                inst_mask,
                crop_size=self.config.crop_size,
                padding_ratio=self.config.crop_padding_ratio,
            )
            if crop is None:
                continue
            cx0, cy0, cx1, cy1 = crop.crop_bbox
            wh, ww = crop.work_mask.shape
            kps_work = self._scale_keypoints_to_work(
                (keypoints_by_instance or {}).get(inst_label), crop
            )
            feats = self._extract_crop_features(crop)
            pose = estimate_body_frame_pose(
                crop.work_mask,
                keypoints_work=kps_work,
                keypoint_part_map=self.config.keypoint_part_map,
                prev_pose=(
                    self._temporal_states[inst_label].pose
                    if inst_label in self._temporal_states
                    else None
                ),
            )

            # Convert part masks to work-crop resolution and enforce mutual exclusion.
            ordered_names = list(part_map.keys())
            crop_part_masks: List[np.ndarray] = []
            for pname in ordered_names:
                full_pmask = np.asarray(part_map[pname], dtype=bool)
                sub_p = full_pmask[cy0:cy1, cx0:cx1] & crop.crop_mask
                if (wh, ww) != sub_p.shape:
                    wp = (
                        cv2.resize(
                            sub_p.astype(np.uint8),
                            (ww, wh),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                    ) & crop.work_mask
                    if not np.any(wp) and np.any(sub_p):
                        ys_p, xs_p = np.nonzero(sub_p)
                        py_w = min(
                            wh - 1,
                            max(0, int(round(float(ys_p.mean()) * crop.scale_y))),
                        )
                        px_w = min(
                            ww - 1,
                            max(0, int(round(float(xs_p.mean()) * crop.scale_x))),
                        )
                        wp[py_w, px_w] = True
                else:
                    wp = sub_p
                crop_part_masks.append(wp)

            exclusive_parts = make_instance_masks_exclusive(crop_part_masks)
            inst_protos: List[BodyPartPrototype] = []
            union_seeded = np.zeros_like(crop.work_mask, dtype=bool)
            for pname, wp in zip(ordered_names, exclusive_parts):
                proto = self._build_prototype_from_mask(
                    part_label=pname,
                    part_work_mask=wp,
                    instance_work_mask=crop.work_mask,
                    features=feats,
                    pose=pose,
                    is_unassigned=False,
                )
                if proto is not None:
                    inst_protos.append(proto)
                    shared_accum[pname].append(proto)
                    union_seeded |= wp
                    built_count += 1

            if not inst_protos:
                continue

            seeded_coverage = float(np.count_nonzero(union_seeded)) / float(
                max(1, int(np.count_nonzero(crop.work_mask)))
            )
            full_partition = (
                bool(self.config.partition_full_instance)
                if self.config.partition_full_instance is not None
                else (seeded_coverage >= 0.75)
            )
            if not full_partition:
                remainder = crop.work_mask & ~union_seeded
                if int(np.count_nonzero(remainder)) >= 4:
                    rem_proto = self._build_prototype_from_mask(
                        part_label=_UNASSIGNED_PART,
                        part_work_mask=remainder,
                        instance_work_mask=crop.work_mask,
                        features=feats,
                        pose=pose,
                        is_unassigned=True,
                    )
                    if rem_proto is not None:
                        inst_protos.append(rem_proto)
                        shared_accum[_UNASSIGNED_PART].append(rem_proto)

            self._instance_prototypes[inst_label] = self._merge_prototype_lists(
                self._instance_prototypes.get(inst_label, []), inst_protos
            )

        if shared_accum:
            merged_shared: List[BodyPartPrototype] = []
            for pname, plist in shared_accum.items():
                all_seed_vecs = np.concatenate(
                    [p.seed_prototypes for p in plist], axis=0
                )
                if all_seed_vecs.shape[0] > 6:
                    all_seed_vecs = _select_diverse_prototypes(all_seed_vecs.T, 6)
                mean_work = _l2_normalize_vec(
                    np.mean([p.working_prototype for p in plist], axis=0)
                )
                merged_shared.append(
                    BodyPartPrototype(
                        part_label=pname,
                        seed_prototypes=all_seed_vecs,
                        working_prototype=mean_work,
                        mean_u=float(np.mean([p.mean_u for p in plist])),
                        std_u=float(np.mean([p.std_u for p in plist])),
                        mean_v=float(np.mean([p.mean_v for p in plist])),
                        std_v=float(np.mean([p.std_v for p in plist])),
                        mean_thickness=float(
                            np.mean([p.mean_thickness for p in plist])
                        ),
                        std_thickness=float(np.mean([p.std_thickness for p in plist])),
                        seed_area_ratio=float(
                            np.mean([p.seed_area_ratio for p in plist])
                        ),
                        is_unassigned=any(p.is_unassigned for p in plist),
                    )
                )
            self._shared_prototypes = self._merge_prototype_lists(
                self._shared_prototypes, merged_shared
            )
        return built_count

    @staticmethod
    def _merge_prototype_lists(
        existing: Sequence[BodyPartPrototype],
        incoming: Sequence[BodyPartPrototype],
    ) -> List[BodyPartPrototype]:
        if not existing:
            return list(incoming)
        by_name: Dict[str, BodyPartPrototype] = {p.part_label: p for p in existing}
        for inc in incoming:
            prev = by_name.get(inc.part_label)
            if prev is None:
                by_name[inc.part_label] = inc
                continue
            combined_seeds = np.concatenate(
                [prev.seed_prototypes, inc.seed_prototypes], axis=0
            )
            if combined_seeds.shape[0] > 6:
                combined_seeds = _select_diverse_prototypes(combined_seeds.T, 6)
            by_name[inc.part_label] = BodyPartPrototype(
                part_label=inc.part_label,
                seed_prototypes=combined_seeds,
                working_prototype=inc.working_prototype,
                mean_u=0.5 * (prev.mean_u + inc.mean_u),
                std_u=0.5 * (prev.std_u + inc.std_u),
                mean_v=0.5 * (prev.mean_v + inc.mean_v),
                std_v=0.5 * (prev.std_v + inc.std_v),
                mean_thickness=0.5 * (prev.mean_thickness + inc.mean_thickness),
                std_thickness=0.5 * (prev.std_thickness + inc.std_thickness),
                seed_area_ratio=0.5 * (prev.seed_area_ratio + inc.seed_area_ratio),
                is_unassigned=prev.is_unassigned or inc.is_unassigned,
            )
        return list(by_name.values())

    def _resolve_mode_for_instance(
        self,
        inst_label: str,
        keypoints: Optional[Mapping[str, Tuple[float, float]]],
    ) -> str:
        has_seed = bool(self._instance_prototypes.get(inst_label)) or (
            self.config.share_seed_across_instances and bool(self._shared_prototypes)
        )
        has_kps = bool(keypoints)
        if self.config.mode == "seeded":
            return "seeded" if has_seed else "unsupervised"
        if self.config.mode == "keypoint_guided":
            return (
                "keypoint_guided"
                if has_kps
                else ("seeded" if has_seed else "unsupervised")
            )
        if self.config.mode == "unsupervised":
            return "unsupervised"
        # "auto"
        if has_seed:
            return "seeded"
        if has_kps:
            return "keypoint_guided"
        return "unsupervised"

    def _project_temporal_prior(
        self,
        inst_label: str,
        part_labels: Sequence[str],
        pose: BodyFramePose,
        work_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        prev_state = self._temporal_states.get(inst_label)
        if prev_state is None or self.config.temporal_prior_weight <= 0.0:
            return None
        if tuple(part_labels) != prev_state.part_labels:
            return None

        # Warp previous probabilities into current canonical (u, v) coordinates via affine map!
        # In previous crop: x_prev = mx_p + ax_p * (u * L_ax_p) + lat_p * (v * L_lat_p).
        p_pose = prev_state.pose
        u_cur = pose.u_map
        v_cur = pose.v_map
        map_x = (
            p_pose.centroid_x
            + p_pose.axial_vec[0] * (u_cur * p_pose.axial_scale)
            + p_pose.lateral_vec[0] * (v_cur * p_pose.lateral_scale)
        ).astype(np.float32)
        map_y = (
            p_pose.centroid_y
            + p_pose.axial_vec[1] * (u_cur * p_pose.axial_scale)
            + p_pose.lateral_vec[1] * (v_cur * p_pose.lateral_scale)
        ).astype(np.float32)

        K = len(part_labels)
        wh, ww = work_mask.shape
        warped = np.zeros((K, wh, ww), dtype=np.float32)
        for k in range(K):
            warped[k] = cv2.remap(
                prev_state.work_probs[k],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        warped[:, ~work_mask] = 0.0
        return warped

    def _score_seeded_parts(
        self,
        inst_label: str,
        crop: _InstanceCrop,
        feats: np.ndarray,
        pose: BodyFramePose,
        kps_work: Mapping[str, Tuple[float, float]],
    ) -> Tuple[List[str], np.ndarray, Optional[List[BodyPartPrototype]]]:
        protos = self._instance_prototypes.get(inst_label)
        if not protos and self.config.share_seed_across_instances:
            protos = [
                BodyPartPrototype(
                    part_label=p.part_label,
                    seed_prototypes=p.seed_prototypes.copy(),
                    working_prototype=p.working_prototype.copy(),
                    mean_u=p.mean_u,
                    std_u=p.std_u,
                    mean_v=p.mean_v,
                    std_v=p.std_v,
                    mean_thickness=p.mean_thickness,
                    std_thickness=p.std_thickness,
                    seed_area_ratio=p.seed_area_ratio,
                    is_unassigned=p.is_unassigned,
                )
                for p in self._shared_prototypes
            ]
            self._instance_prototypes[inst_label] = protos
        if not protos:
            return [], np.zeros((0, *crop.work_mask.shape), dtype=np.float32), None

        wh, ww = crop.work_mask.shape
        K = len(protos)
        part_labels = [p.part_label for p in protos]
        flat_feats = feats.reshape(feats.shape[0], -1)  # (C, H*W)

        logits = np.zeros((K, wh, ww), dtype=np.float32)
        for idx, proto in enumerate(protos):
            seed_sim = np.max(proto.seed_prototypes @ flat_feats, axis=0).reshape(
                wh, ww
            )
            work_sim = (proto.working_prototype @ flat_feats).reshape(wh, ww)
            app_score = 0.72 * seed_sim + 0.28 * work_sim

            du = (pose.u_map - proto.mean_u) / max(proto.std_u, 0.14)
            dv = (pose.v_map - proto.mean_v) / max(proto.std_v, 0.18)
            dt = (pose.thickness_map - proto.mean_thickness) / max(
                proto.std_thickness, 0.15
            )
            canon_score = -0.5 * (du * du + 0.75 * dv * dv + 0.35 * dt * dt)
            # Mildly relax canonical spatial penalty for catch-all unassigned remainder.
            canon_mult = 0.35 if proto.is_unassigned else 1.0

            logits[idx] = (
                self.config.appearance_weight * 3.2 * app_score
                + self.config.canonical_prior_weight * canon_mult * canon_score
            )

        # Optional keypoint geodesic boost if keypoints exist and match seeded part names!
        if kps_work and self.config.keypoint_boost_weight > 0.0:
            kp_seeds_by_part: Dict[str, np.ndarray] = {}
            for kp_name, (kx, ky) in kps_work.items():
                mapped = self.config.keypoint_part_map.get(
                    kp_name.strip().lower()
                ) or infer_part_from_keypoint_name(
                    kp_name, granularity=self.config.keypoint_granularity
                )
                matched_label = next(
                    (
                        pl
                        for pl in part_labels
                        if pl != _UNASSIGNED_PART
                        and (
                            pl.lower() == mapped.lower()
                            or mapped.lower() in pl.lower()
                            or pl.lower() in mapped.lower()
                        )
                    ),
                    None,
                )
                if matched_label is None:
                    continue
                sm = kp_seeds_by_part.setdefault(
                    matched_label, np.zeros((wh, ww), dtype=bool)
                )
                cv2.circle(
                    sm.view(np.uint8), (int(round(kx)), int(round(ky))), 2, 1, -1
                )
                sm &= crop.work_mask
            if kp_seeds_by_part:
                matched_names = list(kp_seeds_by_part.keys())
                edge_map = _compute_crop_edge_strength(crop.work_rgb, crop.work_mask)
                g_dists = compute_masked_geodesic_distances(
                    [kp_seeds_by_part[n] for n in matched_names],
                    crop.work_mask,
                    edge_strength=edge_map,
                    edge_weight=self.config.geodesic_edge_weight,
                )
                scale_len = max(6.0, 0.65 * pose.axial_scale)
                for m_idx, m_name in enumerate(matched_names):
                    p_idx = part_labels.index(m_name)
                    d_norm = np.where(
                        np.isfinite(g_dists[m_idx]),
                        g_dists[m_idx] / scale_len,
                        3.0,
                    )
                    logits[p_idx] += self.config.keypoint_boost_weight * np.exp(
                        -0.8 * d_norm
                    )

        temp_prior = self._project_temporal_prior(
            inst_label, part_labels, pose, crop.work_mask
        )
        if temp_prior is not None:
            logits += self.config.temporal_prior_weight * 1.8 * temp_prior

        # Softmax across candidate parts strictly inside work_mask.
        logits_shifted = logits - np.max(logits, axis=0, keepdims=True)
        exp_l = np.exp(np.clip(logits_shifted, -20.0, 0.0))
        exp_l[:, ~crop.work_mask] = 0.0
        denom = np.sum(exp_l, axis=0, keepdims=True)
        probs = np.where(denom > 1e-8, exp_l / np.maximum(denom, 1e-8), 0.0).astype(
            np.float32
        )
        return part_labels, probs, protos

    def _score_keypoint_guided_parts(
        self,
        inst_label: str,
        crop: _InstanceCrop,
        feats: np.ndarray,
        pose: BodyFramePose,
        kps_work: Mapping[str, Tuple[float, float]],
        skeleton_edges: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        wh, ww = crop.work_mask.shape
        kp_to_part: Dict[str, str] = {}
        grouped_pts: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for kp_name, (kx, ky) in kps_work.items():
            mapped = self.config.keypoint_part_map.get(
                kp_name.strip().lower()
            ) or infer_part_from_keypoint_name(
                kp_name, granularity=self.config.keypoint_granularity
            )
            kp_to_part[kp_name] = mapped
            grouped_pts[mapped].append((kx, ky))

        part_labels = list(grouped_pts.keys())
        if not part_labels:
            return [], np.zeros((0, wh, ww), dtype=np.float32)

        seed_masks: List[np.ndarray] = []
        radius = max(1, int(round(0.04 * max(wh, ww))))
        for pname in part_labels:
            canvas = np.zeros((wh, ww), dtype=np.uint8)
            for kx, ky in grouped_pts[pname]:
                cv2.circle(canvas, (int(round(kx)), int(round(ky))), radius, 1, -1)
            # Rasterize intra-part skeleton bones when available!
            if skeleton_edges:
                for src, dst in skeleton_edges:
                    if (
                        src in kps_work
                        and dst in kps_work
                        and kp_to_part.get(src) == pname
                        and kp_to_part.get(dst) == pname
                    ):
                        p0 = (
                            int(round(kps_work[src][0])),
                            int(round(kps_work[src][1])),
                        )
                        p1 = (
                            int(round(kps_work[dst][0])),
                            int(round(kps_work[dst][1])),
                        )
                        cv2.line(canvas, p0, p1, 1, thickness=max(1, radius))
            sm = (canvas > 0) & crop.work_mask
            if not np.any(sm):
                # Snap slightly off-mask keypoint to nearest foreground pixel in work_mask!
                ys_fg, xs_fg = np.nonzero(crop.work_mask)
                for kx, ky in grouped_pts[pname]:
                    d2 = (xs_fg - kx) ** 2 + (ys_fg - ky) ** 2
                    nearest = int(np.argmin(d2))
                    sm[ys_fg[nearest], xs_fg[nearest]] = True
            seed_masks.append(sm)

        edge_map = _compute_crop_edge_strength(crop.work_rgb, crop.work_mask)
        geo_dists = compute_masked_geodesic_distances(
            seed_masks,
            crop.work_mask,
            edge_strength=edge_map,
            edge_weight=self.config.geodesic_edge_weight,
        )

        # Combine masked geodesic proximity + local seed appearance affinity!
        K = len(part_labels)
        flat_feats = feats.reshape(feats.shape[0], -1)
        scale_len = max(5.0, 0.45 * (pose.axial_scale + pose.lateral_scale))
        logits = np.zeros((K, wh, ww), dtype=np.float32)

        ys_fg, xs_fg = np.nonzero(crop.work_mask)
        for idx, sm in enumerate(seed_masks):
            d_slice = geo_dists[idx]
            # Fallback Euclidean distance only for disconnected foreground mask islands!
            unreachable = crop.work_mask & ~np.isfinite(d_slice)
            if np.any(unreachable):
                pts = grouped_pts[part_labels[idx]]
                u_ys, u_xs = np.nonzero(unreachable)
                min_euc = np.min(
                    np.stack(
                        [np.hypot(u_xs - px, u_ys - py) for px, py in pts],
                        axis=0,
                    ),
                    axis=0,
                )
                d_slice = d_slice.copy()
                d_slice[unreachable] = min_euc * 1.35

            geo_score = -d_slice / scale_len
            if np.any(sm):
                loc_proto = _l2_normalize_vec(feats[:, sm].mean(axis=1))
                app_sim = (loc_proto @ flat_feats).reshape(wh, ww)
            else:
                app_sim = np.zeros((wh, ww), dtype=np.float32)
            logits[idx] = (
                2.4 * geo_score + 0.85 * self.config.appearance_weight * app_sim
            )

        temp_prior = self._project_temporal_prior(
            inst_label, part_labels, pose, crop.work_mask
        )
        if temp_prior is not None:
            logits += self.config.temporal_prior_weight * 1.4 * temp_prior

        logits[:, ~crop.work_mask] = -1e6
        logits_shifted = logits - np.max(logits, axis=0, keepdims=True)
        exp_l = np.exp(np.clip(logits_shifted, -20.0, 0.0))
        exp_l[:, ~crop.work_mask] = 0.0
        denom = np.sum(exp_l, axis=0, keepdims=True)
        probs = np.where(denom > 1e-8, exp_l / np.maximum(denom, 1e-8), 0.0).astype(
            np.float32
        )
        return part_labels, probs

    def _default_unsupervised_part_names(self, k: int) -> List[str]:
        if len(self.config.part_names) == k:
            return list(self.config.part_names)
        if k == 2:
            return ["anterior", "posterior"]
        if k == 3:
            return ["head", "torso", "tail"]
        if k == 4:
            return ["head", "anterior_torso", "posterior_torso", "tail"]
        return [f"part_{idx + 1}" for idx in range(k)]

    def _score_unsupervised_parts(
        self,
        inst_label: str,
        crop: _InstanceCrop,
        feats: np.ndarray,
        pose: BodyFramePose,
    ) -> Tuple[List[str], np.ndarray]:
        wh, ww = crop.work_mask.shape
        K = self.config.num_parts
        part_labels = self._default_unsupervised_part_names(K)
        existing_protos = self._unsupervised_prototypes.get(inst_label)

        if not existing_protos:
            # Deterministic initialization along canonical axial coordinate u (descending: head -> tail).
            fg_u = pose.u_map[crop.work_mask]
            quantiles = np.linspace(100.0, 0.0, K + 1)
            bounds = np.percentile(fg_u, quantiles)
            init_masks: List[np.ndarray] = []
            for idx in range(K):
                hi = bounds[idx]
                lo = bounds[idx + 1]
                if idx == K - 1:
                    pm = crop.work_mask & (pose.u_map <= hi) & (pose.u_map >= lo - 1e-5)
                else:
                    pm = crop.work_mask & (pose.u_map <= hi + 1e-5) & (pose.u_map > lo)
                init_masks.append(pm)

            # 3 deterministic EM-like appearance + canonical refinement iterations.
            flat_feats = feats.reshape(feats.shape[0], -1)
            cur_masks = init_masks
            for _ in range(3):
                iter_logits = np.full((K, wh, ww), -1e6, dtype=np.float32)
                for idx in range(K):
                    pm = cur_masks[idx]
                    if not np.any(pm):
                        continue
                    proto = _l2_normalize_vec(feats[:, pm].mean(axis=1))
                    sim = (proto @ flat_feats).reshape(wh, ww)
                    mu_u = float(pose.u_map[pm].mean())
                    sd_u = max(0.18, float(pose.u_map[pm].std()) + 0.08)
                    mu_t = float(pose.thickness_map[pm].mean())
                    sd_t = max(0.18, float(pose.thickness_map[pm].std()) + 0.08)
                    canon = -0.5 * (
                        ((pose.u_map - mu_u) / sd_u) ** 2
                        + 0.35 * ((pose.thickness_map - mu_t) / sd_t) ** 2
                    )
                    iter_logits[idx] = 1.8 * sim + 1.1 * canon
                best_k = np.argmax(iter_logits, axis=0)
                cur_masks = [crop.work_mask & (best_k == idx) for idx in range(K)]

            built: List[BodyPartPrototype] = []
            for pname, pm in zip(part_labels, cur_masks):
                p_obj = self._build_prototype_from_mask(
                    part_label=pname,
                    part_work_mask=pm,
                    instance_work_mask=crop.work_mask,
                    features=feats,
                    pose=pose,
                )
                if p_obj is not None:
                    built.append(p_obj)
            if built:
                self._unsupervised_prototypes[inst_label] = built
                existing_protos = built

        if not existing_protos:
            return [], np.zeros((0, wh, ww), dtype=np.float32)

        part_labels = [p.part_label for p in existing_protos]
        flat_feats = feats.reshape(feats.shape[0], -1)
        K_eff = len(existing_protos)
        logits = np.zeros((K_eff, wh, ww), dtype=np.float32)
        for idx, proto in enumerate(existing_protos):
            sim = 0.65 * np.max(proto.seed_prototypes @ flat_feats, axis=0).reshape(
                wh, ww
            ) + 0.35 * (proto.working_prototype @ flat_feats).reshape(wh, ww)
            du = (pose.u_map - proto.mean_u) / max(proto.std_u, 0.16)
            dt = (pose.thickness_map - proto.mean_thickness) / max(
                proto.std_thickness, 0.16
            )
            logits[idx] = 2.0 * sim - 0.55 * (du * du + 0.35 * dt * dt)

        temp_prior = self._project_temporal_prior(
            inst_label, part_labels, pose, crop.work_mask
        )
        if temp_prior is not None:
            logits += self.config.temporal_prior_weight * 1.5 * temp_prior

        logits[:, ~crop.work_mask] = -1e6
        logits_shifted = logits - np.max(logits, axis=0, keepdims=True)
        exp_l = np.exp(np.clip(logits_shifted, -20.0, 0.0))
        exp_l[:, ~crop.work_mask] = 0.0
        denom = np.sum(exp_l, axis=0, keepdims=True)
        probs = np.where(denom > 1e-8, exp_l / np.maximum(denom, 1e-8), 0.0).astype(
            np.float32
        )
        return part_labels, probs

    def _apply_area_guardrail_and_disjoint_assignment(
        self,
        part_labels: Sequence[str],
        crop_probs: np.ndarray,
        crop_rgb: np.ndarray,
        crop_mask: np.ndarray,
        prototypes: Optional[Sequence[BodyPartPrototype]],
        *,
        full_partition: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Convert probability maps on ``(crop_h, crop_w)`` into mutually exclusive masks inside ``crop_mask``."""
        K, ch, cw = crop_probs.shape
        if K == 0 or not np.any(crop_mask):
            return {}, {}

        probs = crop_probs.copy()
        if self.config.boundary_refine:
            probs = guided_filter_probabilities(
                crop_rgb,
                probs,
                crop_mask,
                radius=self.config.guided_radius,
                eps=self.config.guided_eps,
            )

        inst_area = int(np.count_nonzero(crop_mask))
        winners = np.argmax(probs, axis=0)
        part_masks: Dict[str, np.ndarray] = {}
        confidences: Dict[str, float] = {}

        proto_by_name = (
            {p.part_label: p for p in prototypes} if prototypes is not None else {}
        )
        unassigned_idx = (
            part_labels.index(_UNASSIGNED_PART)
            if _UNASSIGNED_PART in part_labels
            else -1
        )

        # Optional per-part CRF boundary snap on tight crop, re-arbitrated via score competition.
        refiner = self.crf_refiner
        if refiner is not None:
            for k, pname in enumerate(part_labels):
                if pname == _UNASSIGNED_PART:
                    continue
                raw_m = crop_mask & (winners == k)
                if not np.any(raw_m):
                    continue
                refined_m = (
                    refiner.refine(crop_rgb, raw_m, score_map=probs[k]) & crop_mask
                )
                probs[k] = np.where(
                    refined_m, np.maximum(probs[k], 0.55), probs[k] * 0.75
                )
            winners = np.argmax(probs, axis=0)

        # Relative area-growth guardrail when unassigned channel exists (partial-part mode).
        if unassigned_idx >= 0 and proto_by_name:
            for k, pname in enumerate(part_labels):
                if k == unassigned_idx:
                    continue
                proto = proto_by_name.get(pname)
                if proto is None:
                    continue
                max_allowed = max(
                    self.config.min_part_area_px * 2,
                    int(
                        round(
                            proto.seed_area_ratio
                            * self.config.max_part_area_growth
                            * inst_area
                        )
                    ),
                )
                claimed = crop_mask & (winners == k)
                cur_area = int(np.count_nonzero(claimed))
                if cur_area > max_allowed:
                    margin = probs[k] - probs[unassigned_idx]
                    vals = margin[claimed]
                    keep_thresh = float(
                        np.percentile(
                            vals,
                            100.0 * (1.0 - float(max_allowed) / float(cur_area)),
                        )
                    )
                    demote = claimed & (margin < keep_thresh)
                    winners[demote] = unassigned_idx

        # Remove tiny speckles and absorb holes if full_partition=True.
        min_area = min(self.config.min_part_area_px, max(1, inst_area // 6))
        assigned_union = np.zeros_like(crop_mask, dtype=bool)
        real_indices = [
            idx for idx, name in enumerate(part_labels) if name != _UNASSIGNED_PART
        ]
        for k in real_indices:
            pname = part_labels[k]
            raw_m = crop_mask & (winners == k)
            cleaned = self._clean_part_components(raw_m, probs[k], min_area_px=min_area)
            cleaned &= crop_mask & ~assigned_union
            if np.any(cleaned):
                part_masks[pname] = cleaned
                assigned_union |= cleaned

        if full_partition and real_indices:
            leftover = crop_mask & ~assigned_union
            if np.any(leftover):
                real_probs = probs[real_indices]
                best_sub = np.argmax(real_probs, axis=0)
                for sub_i, k in enumerate(real_indices):
                    pname = part_labels[k]
                    extra = leftover & (best_sub == sub_i)
                    if np.any(extra):
                        part_masks[pname] = (
                            (part_masks[pname] | extra)
                            if pname in part_masks
                            else extra
                        )

        for k in real_indices:
            pname = part_labels[k]
            pm = part_masks.get(pname)
            if pm is not None and np.any(pm):
                confidences[pname] = float(np.clip(np.mean(probs[k][pm]), 0.0, 1.0))

        return part_masks, confidences

    @staticmethod
    def _clean_part_components(
        mask: np.ndarray,
        score_map: np.ndarray,
        *,
        min_area_px: int,
    ) -> np.ndarray:
        if not np.any(mask):
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return mask
        keep = np.zeros_like(mask, dtype=bool)
        best_comp = 1
        best_mass = -1.0
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            comp_mask = labels == comp_id
            mass = float(score_map[comp_mask].sum())
            if mass > best_mass:
                best_mass = mass
                best_comp = comp_id
            if area >= min_area_px:
                keep |= comp_mask
        if not np.any(keep) and int(stats[best_comp, cv2.CC_STAT_AREA]) >= max(
            1, min_area_px // 2
        ):
            keep |= labels == best_comp
        return keep

    def _polygonize_crop_parts(
        self,
        part_masks: Mapping[str, np.ndarray],
        confidences: Mapping[str, float],
        crop_mask: np.ndarray,
        crop_bbox: Tuple[int, int, int, int],
        *,
        parent_polygon: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Extract simplified, non-overlapping polygons strictly inside the parent geometry."""
        from shapely import make_valid
        from shapely.geometry import MultiPolygon, Polygon

        cx0, cy0, _cx1, _cy1 = crop_bbox
        # Build parent Shapely geometry in full-frame coordinates for exact containment clipping.
        parent_geom = None
        if parent_polygon is not None and len(parent_polygon) >= 3:
            try:
                raw_parent = Polygon([(float(x), float(y)) for x, y in parent_polygon])
                if not raw_parent.is_valid:
                    raw_parent = make_valid(raw_parent)
                parent_geom = _largest_valid_polygon(raw_parent)
            except Exception:
                parent_geom = None
        if parent_geom is None:
            parent_contours, _ = cv2.findContours(
                crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            parent_polys = []
            for cnt in parent_contours:
                pts = cnt.reshape(-1, 2).astype(np.float64)
                if len(pts) >= 3:
                    poly = Polygon([(float(x + cx0), float(y + cy0)) for x, y in pts])
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    valid_p = _largest_valid_polygon(poly)
                    if valid_p is not None:
                        parent_polys.append(valid_p)
            if len(parent_polys) == 1:
                parent_geom = parent_polys[0]
            elif len(parent_polys) > 1:
                parent_geom = MultiPolygon(parent_polys)

        ordered = sorted(
            part_masks.keys(),
            key=lambda name: (-float(confidences.get(name, 0.0)), name),
        )
        claimed_geom = None
        polygons_out: Dict[str, List[Tuple[float, float]]] = {}
        clearance = 1e-3

        for pname in ordered:
            pm = part_masks[pname]
            contours, _ = cv2.findContours(
                pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            if self.config.polygon_epsilon > 0.0:
                approx = cv2.approxPolyDP(
                    largest, float(self.config.polygon_epsilon), True
                )
            else:
                approx = largest
            pts = approx.reshape(-1, 2).astype(np.float64)
            if len(pts) < 3:
                pts = largest.reshape(-1, 2).astype(np.float64)
            if len(pts) < 3:
                continue

            shifted = [(float(x + cx0), float(y + cy0)) for x, y in pts]
            try:
                geom = Polygon(shifted)
                if not geom.is_valid:
                    geom = make_valid(geom)
                if parent_geom is not None:
                    geom = make_valid(geom.intersection(parent_geom))
                if claimed_geom is not None:
                    geom = make_valid(
                        geom.difference(
                            claimed_geom.buffer(
                                clearance, quad_segs=1, join_style="mitre"
                            )
                        )
                    )
                poly = _largest_valid_polygon(geom)
            except Exception:
                poly = None

            if poly is None or poly.area < 0.5:
                continue
            coords = [(float(x), float(y)) for x, y in poly.exterior.coords[:-1]]
            if len(coords) < 3:
                continue
            polygons_out[pname] = coords
            claimed_geom = (
                poly if claimed_geom is None else make_valid(claimed_geom.union(poly))
            )
        return polygons_out

    def segment_instance(
        self,
        frame_rgb: np.ndarray,
        instance_label: str,
        instance_mask: np.ndarray,
        *,
        keypoints: Optional[Mapping[str, Tuple[float, float]]] = None,
        skeleton_edges: Optional[Sequence[Tuple[str, str]]] = None,
        parent_polygon: Optional[Sequence[Tuple[float, float]]] = None,
        group_id: Optional[int | str] = None,
    ) -> Dict[str, PartPrediction]:
        """Segment body parts strictly inside a single tracked ``instance_mask``."""
        crop = _prepare_instance_crop(
            frame_rgb,
            instance_mask,
            crop_size=self.config.crop_size,
            padding_ratio=self.config.crop_padding_ratio,
        )
        if crop is None:
            return {}

        inst_area_px = int(np.count_nonzero(crop.crop_mask))
        if inst_area_px < self.config.min_part_area_px:
            return {}

        kps_work = self._scale_keypoints_to_work(keypoints, crop)
        feats = self._extract_crop_features(crop)
        active_mode = self._resolve_mode_for_instance(instance_label, kps_work)

        candidate_protos = self._instance_prototypes.get(instance_label) or (
            self._shared_prototypes if self.config.share_seed_across_instances else None
        )
        prev_pose = (
            self._temporal_states[instance_label].pose
            if instance_label in self._temporal_states
            else None
        )
        pose = estimate_body_frame_pose(
            crop.work_mask,
            keypoints_work=kps_work,
            keypoint_part_map=self.config.keypoint_part_map,
            prev_pose=prev_pose,
            features=feats if active_mode == "seeded" else None,
            prototypes=candidate_protos if active_mode == "seeded" else None,
        )

        used_protos: Optional[List[BodyPartPrototype]] = None
        if active_mode == "seeded":
            part_labels, work_probs, used_protos = self._score_seeded_parts(
                instance_label, crop, feats, pose, kps_work
            )
        elif active_mode == "keypoint_guided":
            part_labels, work_probs = self._score_keypoint_guided_parts(
                instance_label,
                crop,
                feats,
                pose,
                kps_work,
                skeleton_edges=skeleton_edges,
            )
        else:
            part_labels, work_probs = self._score_unsupervised_parts(
                instance_label, crop, feats, pose
            )
            used_protos = self._unsupervised_prototypes.get(instance_label)

        if not part_labels or work_probs.shape[0] == 0:
            return {}

        # Update temporal state and EMA working prototypes (when high confidence).
        self._temporal_states[instance_label] = _TemporalInstanceState(
            pose=pose,
            part_labels=tuple(part_labels),
            work_probs=work_probs.copy(),
            u_grid=pose.u_map,
            v_grid=pose.v_map,
        )
        if used_protos and self.config.ema_alpha > 0.0:
            work_winners = np.argmax(work_probs, axis=0)
            for idx, proto in enumerate(used_protos):
                pm_w = crop.work_mask & (work_winners == idx)
                if int(np.count_nonzero(pm_w)) < 3:
                    continue
                conf_w = float(work_probs[idx][pm_w].mean())
                if conf_w >= self.config.ema_min_confidence:
                    obs_vec = _l2_normalize_vec(feats[:, pm_w].mean(axis=1))
                    proto.working_prototype = _l2_normalize_vec(
                        (1.0 - self.config.ema_alpha) * proto.working_prototype
                        + self.config.ema_alpha * obs_vec
                    )

        # Upsample work_probs -> native crop resolution (crop_h, crop_w).
        ch, cw = crop.crop_mask.shape
        wh, ww = crop.work_mask.shape
        if (wh, ww) != (ch, cw):
            crop_probs = np.zeros((len(part_labels), ch, cw), dtype=np.float32)
            for idx in range(len(part_labels)):
                crop_probs[idx] = cv2.resize(
                    work_probs[idx], (cw, ch), interpolation=cv2.INTER_LINEAR
                )
            crop_probs[:, ~crop.crop_mask] = 0.0
        else:
            crop_probs = work_probs

        full_partition = (
            bool(self.config.partition_full_instance)
            if self.config.partition_full_instance is not None
            else (_UNASSIGNED_PART not in part_labels)
        )
        part_masks, confidences = self._apply_area_guardrail_and_disjoint_assignment(
            part_labels,
            crop_probs,
            crop.crop_rgb,
            crop.crop_mask,
            used_protos,
            full_partition=full_partition,
        )
        polygons = self._polygonize_crop_parts(
            part_masks,
            confidences,
            crop.crop_mask,
            crop.crop_bbox,
            parent_polygon=parent_polygon,
        )

        predictions: Dict[str, PartPrediction] = {}
        for pname, pmask in part_masks.items():
            poly = polygons.get(pname)
            if not poly:
                continue
            area_px = int(np.count_nonzero(pmask))
            predictions[pname] = PartPrediction(
                part_label=pname,
                parent_instance=instance_label,
                crop_mask=pmask,
                crop_bbox=crop.crop_bbox,
                polygon=poly,
                confidence=float(confidences.get(pname, 1.0)),
                area_px=area_px,
                area_ratio=float(area_px) / float(max(1, inst_area_px)),
                group_id=group_id,
            )
        return predictions

    def segment_frame(
        self,
        frame_rgb: np.ndarray,
        instance_masks: Mapping[str, np.ndarray],
        *,
        keypoints_by_instance: Optional[
            Mapping[str, Mapping[str, Tuple[float, float]]]
        ] = None,
        skeleton_edges: Optional[Sequence[Tuple[str, str]]] = None,
        parent_polygons: Optional[Mapping[str, Sequence[Tuple[float, float]]]] = None,
        group_ids: Optional[Mapping[str, Optional[int | str]]] = None,
    ) -> Dict[str, Dict[str, PartPrediction]]:
        """Segment body parts inside every tracked instance on ``frame_rgb``."""
        if not instance_masks:
            return {}
        ordered_labels = list(instance_masks.keys())
        exclusive_parents = make_instance_masks_exclusive(
            [instance_masks[lbl] for lbl in ordered_labels]
        )
        results: Dict[str, Dict[str, PartPrediction]] = {}
        for lbl, excl_mask in zip(ordered_labels, exclusive_parents):
            if not np.any(excl_mask):
                continue
            results[lbl] = self.segment_instance(
                frame_rgb,
                lbl,
                excl_mask,
                keypoints=(keypoints_by_instance or {}).get(lbl),
                skeleton_edges=skeleton_edges,
                parent_polygon=(parent_polygons or {}).get(lbl),
                group_id=(group_ids or {}).get(lbl),
            )
        return results


def _largest_valid_polygon(geom: Any) -> Any:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0.0 else None
    if isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if isinstance(p, Polygon) and p.area > 0.0]
        return max(polys, key=lambda p: p.area) if polys else None
    if isinstance(geom, GeometryCollection):
        polys = [
            cand
            for item in geom.geoms
            if (cand := _largest_valid_polygon(item)) is not None
        ]
        return max(polys, key=lambda p: p.area) if polys else None
    return None


def _rasterize_polygon(
    points: Sequence[Sequence[float]],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if len(points) < 3:
        return mask.astype(bool)
    arr = np.rint(np.asarray(points, dtype=np.float64)[:, :2]).astype(np.int32)
    cv2.fillPoly(mask, [arr], 1)
    return mask.astype(bool)


def parse_frame_instances_and_seed_parts(
    payload: Mapping[str, Any],
    *,
    default_height: int,
    default_width: int,
    infer_nested_seeds: bool = True,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, List[Tuple[float, float]]],
    Dict[str, Optional[int | str]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Tuple[float, float]]],
]:
    """Extract parent instances, explicit/nested body-part seeds, and keypoints from a LabelMe payload."""
    img_h = int(payload.get("imageHeight") or default_height)
    img_w = int(payload.get("imageWidth") or default_width)
    shapes = payload.get("shapes") or []
    if not isinstance(shapes, list):
        shapes = []

    raw_polygons: List[
        Tuple[
            int,
            Mapping[str, Any],
            str,
            Optional[int | str],
            List[Tuple[float, float]],
            np.ndarray,
        ]
    ] = []
    explicit_parts: List[
        Tuple[Mapping[str, Any], str, str, Optional[int | str], np.ndarray]
    ] = []
    point_shapes: List[Mapping[str, Any]] = []

    for idx, shape in enumerate(shapes):
        if not isinstance(shape, Mapping) or is_zone_shape_payload(shape):
            continue
        stype = str(shape.get("shape_type") or "polygon").strip().lower()
        if stype == "point":
            point_shapes.append(shape)
            continue
        if stype != "polygon":
            continue
        pts_raw = shape.get("points") or []
        pts = [
            (float(pt[0]), float(pt[1]))
            for pt in pts_raw
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(pts) < 3:
            continue
        mask = _rasterize_polygon(pts, img_h, img_w)
        if not np.any(mask):
            continue
        gid = shape.get("group_id")
        if is_body_part_shape_payload(shape):
            parent_hint = extract_body_part_parent_instance(shape)
            part_lbl = extract_body_part_label(shape) or "part"
            explicit_parts.append((shape, parent_hint, part_lbl, gid, mask))
        else:
            flags = (
                shape.get("flags") if isinstance(shape.get("flags"), Mapping) else {}
            )
            inst_lbl = str(
                shape.get("instance_label")
                or (flags.get("instance_label") if flags else "")
                or shape.get("label")
                or ""
            ).strip()
            if inst_lbl:
                raw_polygons.append((idx, shape, inst_lbl, gid, pts, mask))

    # Optional automatic nested seed detection when a user manually draws child polygons
    # (e.g. "head", "torso", "tail") inside a parent animal polygon ("mouse_1") without custom flags.
    nested_child_indices: set[int] = set()
    inferred_parts: List[Tuple[str, str, np.ndarray]] = []
    if infer_nested_seeds and len(raw_polygons) >= 2:
        areas = [int(np.count_nonzero(item[5])) for item in raw_polygons]
        for c_i, (_s_idx, c_shape, c_lbl, c_gid, _c_pts, c_mask) in enumerate(
            raw_polygons
        ):
            c_area = areas[c_i]
            if c_area <= 0:
                continue
            best_parent_i: Optional[int] = None
            best_containment = 0.0
            for p_i, (_ps_idx, _p_shape, p_lbl, p_gid, _p_pts, p_mask) in enumerate(
                raw_polygons
            ):
                if p_i == c_i or p_i in nested_child_indices:
                    continue
                p_area = areas[p_i]
                if c_area >= 0.90 * p_area:
                    continue
                # Never treat two same-class animals ("mouse" inside "mouse") as parent/part!
                if c_lbl.lower() == p_lbl.lower():
                    continue
                if (
                    c_gid is not None
                    and p_gid is not None
                    and str(c_gid).strip() != str(p_gid).strip()
                ):
                    continue
                inter = int(np.count_nonzero(c_mask & p_mask))
                containment = float(inter) / float(c_area)
                if containment >= 0.85 and containment > best_containment:
                    best_containment = containment
                    best_parent_i = p_i
            if best_parent_i is not None:
                nested_child_indices.add(c_i)

        # Assign stable disambiguated parent names first, then attach inferred nested parts.
        parent_entries = [
            item for i, item in enumerate(raw_polygons) if i not in nested_child_indices
        ]
    else:
        parent_entries = list(raw_polygons)

    label_counts: Dict[str, int] = defaultdict(int)
    for _idx, _sh, lbl, _gid, _pts, _m in parent_entries:
        label_counts[lbl] += 1

    parent_masks: Dict[str, np.ndarray] = {}
    parent_polys: Dict[str, List[Tuple[float, float]]] = {}
    parent_gids: Dict[str, Optional[int | str]] = {}
    gid_to_parent: Dict[str, str] = {}
    entry_index_to_parent_name: Dict[int, str] = {}
    seen_ord: Dict[str, int] = defaultdict(int)

    for orig_i, (_idx, _sh, lbl, gid, pts, mask) in enumerate(raw_polygons):
        if orig_i in nested_child_indices:
            continue
        if label_counts[lbl] > 1:
            suffix = str(gid).strip() if gid not in (None, "") else str(seen_ord[lbl])
            seen_ord[lbl] += 1
            inst_name = f"{lbl}_{suffix}"
            while inst_name in parent_masks:
                seen_ord[lbl] += 1
                inst_name = f"{lbl}_{seen_ord[lbl]}"
        else:
            inst_name = lbl
        parent_masks[inst_name] = mask
        parent_polys[inst_name] = pts
        parent_gids[inst_name] = gid
        entry_index_to_parent_name[orig_i] = inst_name
        if gid not in (None, ""):
            gid_to_parent[str(gid).strip()] = inst_name

    # Resolve nested inferred parts to their containing disambiguated parent.
    for c_i in sorted(nested_child_indices):
        _idx, _c_sh, c_lbl, c_gid, _c_pts, c_mask = raw_polygons[c_i]
        part_name = extract_body_part_label(_c_sh) or c_lbl
        chosen_parent = None
        if c_gid not in (None, "") and str(c_gid).strip() in gid_to_parent:
            chosen_parent = gid_to_parent[str(c_gid).strip()]
        if chosen_parent is None:
            best_inter = 0
            for pname, pmask in parent_masks.items():
                inter = int(np.count_nonzero(c_mask & pmask))
                if inter > best_inter:
                    best_inter = inter
                    chosen_parent = pname
        if chosen_parent is not None:
            if part_name.lower().startswith(f"{chosen_parent.lower()}:"):
                part_name = part_name.split(":", 1)[1].strip() or part_name
            inferred_parts.append((chosen_parent, part_name, c_mask))

    seed_parts: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for _sh, parent_hint, part_lbl, gid, pmask in explicit_parts:
        target_parent = None
        if parent_hint and parent_hint in parent_masks:
            target_parent = parent_hint
        elif gid not in (None, "") and str(gid).strip() in gid_to_parent:
            target_parent = gid_to_parent[str(gid).strip()]
        else:
            best_inter = 0
            for pname, inst_m in parent_masks.items():
                inter = int(np.count_nonzero(pmask & inst_m))
                if inter > best_inter:
                    best_inter = inter
                    target_parent = pname
        if target_parent is not None:
            seed_parts[target_parent][part_lbl] = pmask & parent_masks[target_parent]

    for target_parent, part_lbl, pmask in inferred_parts:
        seed_parts[target_parent].setdefault(
            part_lbl, pmask & parent_masks[target_parent]
        )

    # Associate visible keypoints with their parent instance.
    keypoints_by_inst: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    for shape in point_shapes:
        if shape.get("visible") is False:
            continue
        pts_raw = shape.get("points") or []
        if (
            not pts_raw
            or not isinstance(pts_raw[0], (list, tuple))
            or len(pts_raw[0]) < 2
        ):
            continue
        px, py = float(pts_raw[0][0]), float(pts_raw[0][1])
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        flags = shape.get("flags") if isinstance(shape.get("flags"), Mapping) else {}
        inst_hint = str(
            shape.get("instance_label")
            or (flags.get("instance_label") if flags else "")
            or ""
        ).strip()
        kp_lbl = str(
            shape.get("display_label")
            or (flags.get("display_label") if flags else "")
            or shape.get("label")
            or ""
        ).strip()
        if not kp_lbl:
            continue
        gid = shape.get("group_id")
        owner = None
        if inst_hint and inst_hint in parent_masks:
            owner = inst_hint
        elif gid not in (None, "") and str(gid).strip() in gid_to_parent:
            owner = gid_to_parent[str(gid).strip()]
        elif len(parent_masks) == 1:
            owner = next(iter(parent_masks))
        else:
            ix = int(round(px))
            iy = int(round(py))
            containing = [
                pname
                for pname, pmask in parent_masks.items()
                if 0 <= iy < img_h and 0 <= ix < img_w and bool(pmask[iy, ix])
            ]
            if len(containing) == 1:
                owner = containing[0]
        if owner is not None:
            if kp_lbl.lower().startswith(f"{owner.lower()}:"):
                kp_lbl = kp_lbl.split(":", 1)[1].strip() or kp_lbl
            keypoints_by_inst[owner][kp_lbl] = (px, py)

    return (
        parent_masks,
        parent_polys,
        parent_gids,
        dict(seed_parts),
        dict(keypoints_by_inst),
    )


def _runtime_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


class InstanceBodyPartVideoProcessor:
    """Video processor that segments body parts inside tracked instances across a video."""

    _RUNTIME_KWARG_EXCLUDE = {
        "results_folder",
        "start_frame",
        "end_frame",
        "step",
        "mem_every",
        "point_tracking",
        "has_occlusion",
        "save_video_with_color_mask",
        "is_cutie",
        "is_new_segment",
        "pred_worker",
        "stop_event",
    }

    def __init__(self, video_path: str, *args: Any, **kwargs: Any) -> None:
        _ = args
        from annolid.data.videos import CV2Video
        from annolid.features.dino_models import (
            DEFAULT_DINOV3_MODEL_ID,
            resolve_dino_model_from_runtime,
        )

        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        self.num_frames = int(self.video_loader.total_frames())
        self.video_height = int(self.video_loader.get_height())
        self.video_width = int(self.video_loader.get_width())
        self.results_folder = Path(
            kwargs.get("results_folder") or Path(video_path).with_suffix("")
        )
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.pred_worker: Optional[Any] = kwargs.get("pred_worker")
        self._stop_event: Optional[Any] = kwargs.get("stop_event")
        self.skeleton_edges: Tuple[Tuple[str, str], ...] = tuple(
            kwargs.get("skeleton_edges") or ()
        )

        runtime = {
            k: v for k, v in kwargs.items() if k not in self._RUNTIME_KWARG_EXCLUDE
        }
        explicit_config = runtime.get("body_part_config")
        if isinstance(explicit_config, BodyPartSegmentationConfig):
            self.config = explicit_config
        else:
            model_name = resolve_dino_model_from_runtime(
                runtime, fallback=DEFAULT_DINOV3_MODEL_ID
            )
            part_names_raw = runtime.get(
                "body_part_names", runtime.get("part_names", ())
            )
            if isinstance(part_names_raw, str):
                part_names = tuple(
                    p.strip() for p in part_names_raw.split(",") if p.strip()
                )
            else:
                part_names = tuple(
                    str(p).strip() for p in (part_names_raw or ()) if str(p).strip()
                )

            part_full_raw = runtime.get(
                "body_part_partition_full_instance",
                runtime.get("partition_full_instance", None),
            )
            part_full = None if part_full_raw is None else _runtime_bool(part_full_raw)
            self.config = BodyPartSegmentationConfig(
                mode=str(runtime.get("body_part_mode", runtime.get("mode", "auto"))),
                feature_backend=str(
                    runtime.get(
                        "body_part_feature_backend",
                        runtime.get("feature_backend", "fast"),
                    )
                ),
                model_name=model_name,
                crop_size=int(
                    runtime.get("body_part_crop_size", runtime.get("crop_size", 224))
                ),
                crop_padding_ratio=float(
                    runtime.get(
                        "body_part_crop_padding_ratio",
                        runtime.get("crop_padding_ratio", 0.12),
                    )
                ),
                num_parts=int(
                    runtime.get("body_part_num_parts", runtime.get("num_parts", 3))
                ),
                part_names=part_names,
                keypoint_granularity=str(
                    runtime.get(
                        "body_part_keypoint_granularity",
                        runtime.get("keypoint_granularity", "coarse"),
                    )
                ),
                keypoint_part_map=dict(runtime.get("keypoint_part_map") or {}),
                partition_full_instance=part_full,
                share_seed_across_instances=_runtime_bool(
                    runtime.get("share_seed_across_instances", True), default=True
                ),
                appearance_weight=float(runtime.get("appearance_weight", 1.0)),
                canonical_prior_weight=float(
                    runtime.get("canonical_prior_weight", 0.55)
                ),
                temporal_prior_weight=float(runtime.get("temporal_prior_weight", 0.25)),
                geodesic_edge_weight=float(runtime.get("geodesic_edge_weight", 3.5)),
                max_part_area_growth=float(runtime.get("max_part_area_growth", 2.5)),
                min_part_area_px=int(runtime.get("min_part_area_px", 6)),
                boundary_refine=_runtime_bool(
                    runtime.get("boundary_refine", True), default=True
                ),
                crf_refine=_runtime_bool(
                    runtime.get(
                        "body_part_crf_refine", runtime.get("crf_refine", False)
                    )
                ),
                crf_backend=str(runtime.get("crf_backend", "opencv")),
                polygon_epsilon=float(
                    runtime.get(
                        "epsilon_for_polygon", runtime.get("polygon_epsilon", 1.0)
                    )
                ),
                qualify_label_with_instance=_runtime_bool(
                    runtime.get("qualify_label_with_instance", False)
                ),
                persist_json=_runtime_bool(runtime.get("persist_json", False)),
                device=(
                    "cpu"
                    if _runtime_bool(runtime.get("use_cpu_only", False))
                    else runtime.get("device")
                ),
            )
        self.segmenter = InstanceBodyPartSegmenter(
            config=self.config,
            extractor=kwargs.get("extractor"),
        )
        self.last_summary: Dict[str, Any] = {}

    def set_pred_worker(self, pred_worker: Any) -> None:
        self.pred_worker = pred_worker

    def get_total_frames(self) -> int:
        return int(self.num_frames)

    def cleanup(self) -> None:
        try:
            self.video_loader.release()
        except Exception:
            pass

    def _resolve_stop_event(self) -> Optional[Any]:
        if self._stop_event is not None:
            return self._stop_event
        worker = self.pred_worker
        if worker is not None:
            return getattr(worker, "_stop_event", None)
        return None

    def _should_stop(self) -> bool:
        ev = self._resolve_stop_event()
        if ev is not None and hasattr(ev, "is_set") and ev.is_set():
            return True
        worker = self.pred_worker
        if worker is not None and hasattr(worker, "is_stop_requested"):
            try:
                return bool(worker.is_stop_requested())
            except Exception:
                return False
        return False

    def _report_progress(self, percent: int, message: str = "") -> None:
        worker = self.pred_worker
        if worker is None:
            return
        sig = getattr(worker, "progress_reported", None)
        if sig is not None and hasattr(sig, "emit"):
            try:
                sig.emit(int(max(0, min(100, percent))))
            except Exception:
                pass
        if message:
            msg_sig = getattr(worker, "message_reported", None)
            if msg_sig is not None and hasattr(msg_sig, "emit"):
                try:
                    msg_sig.emit(str(message))
                except Exception:
                    pass

    def _discover_annotated_frames(self) -> Dict[int, Path]:
        frames: Dict[int, Path] = {}
        stem = self.results_folder.name
        for name in sorted(os.listdir(self.results_folder)):
            if not name.lower().endswith(".json"):
                continue
            m = re.search(r"(\d+)(?=\.json$)", name)
            if m is None:
                continue
            fnum = int(m.group(1))
            if 0 <= fnum < max(1, self.num_frames):
                frames[fnum] = self.results_folder / name

        store = AnnotationStore.for_frame_path(
            self.results_folder / f"{stem}_000000000.json"
        )
        if store.store_path.exists():
            for fnum in store.iter_frames():
                idx = int(fnum)
                if 0 <= idx < max(1, self.num_frames):
                    frames.setdefault(
                        idx, self.results_folder / f"{stem}_{idx:09d}.json"
                    )
        return dict(sorted(frames.items()))

    def _write_updated_frame_payload(
        self,
        frame_number: int,
        frame_path: Path,
        payload: Dict[str, Any],
    ) -> None:
        import json as _json

        store = AnnotationStore.for_frame_path(frame_path)
        record = dict(payload)
        record["frame"] = int(frame_number)
        if not record.get("imagePath"):
            record["imagePath"] = f"{self.results_folder.name}_{frame_number:09d}.png"

        if frame_path.exists():
            is_stub = False
            try:
                raw_disk = _json.loads(frame_path.read_text(encoding="utf-8"))
                is_stub = isinstance(raw_disk, dict) and "annotation_store" in raw_disk
            except Exception:
                is_stub = False

            if is_stub:
                store.put_frame(int(frame_number), record)
                store.write_stub(frame_path, int(frame_number), record)
            else:
                atomic_write_json(frame_path, payload, indent=2, ensure_ascii=False)
                if store.store_path.exists():
                    store.put_frame(int(frame_number), record)
            return

        store.put_frame(int(frame_number), record)
        if self.config.persist_json:
            atomic_write_json(frame_path, payload, indent=2, ensure_ascii=False)

    def process_video(self, *args: Any, **kwargs: Any) -> str:
        """Alias compatible with keypoint/second-pass video processors."""
        return self.process_video_frames(*args, **kwargs)

    def process_video_frames(self, *args: Any, **kwargs: Any) -> str:
        _ = args
        if kwargs.get("stop_event") is not None:
            self._stop_event = kwargs["stop_event"]
        if kwargs.get("pred_worker") is not None and self.pred_worker is None:
            self.pred_worker = kwargs["pred_worker"]
        if self._should_stop():
            return "Body-part segmentation stopped by user."

        frame_paths = self._discover_annotated_frames()
        if not frame_paths:
            return (
                f"No tracked instance annotations found in '{self.results_folder}'. "
                "Track instances first (e.g. with Cutie, SAM2/SAM3, or YOLO), then segment body parts."
            )

        start_frame = max(0, int(kwargs.get("start_frame", min(frame_paths))))
        end_arg = kwargs.get("end_frame")
        end_frame = (
            min(max(0, self.num_frames - 1), int(end_arg))
            if end_arg is not None
            else max(frame_paths)
        )
        step = max(1, int(kwargs.get("step", 1) or 1))

        stop_event = self._resolve_stop_event()
        from annolid.utils.cancellation import torch_cancellable_forward

        backbone = (
            getattr(self.segmenter._extractor, "model", None)
            if self.segmenter._extractor is not None
            else None
        )

        # Pass 1: Fit seed frames that contain manual/explicit body-part polygons.
        seed_frames_fitted = 0
        if self.config.mode in {"auto", "seeded"}:
            for fnum, fpath in frame_paths.items():
                if self._should_stop():
                    return "Body-part segmentation stopped by user."
                has_png = fpath.with_suffix(".png").exists()
                try:
                    payload = load_labelme_json(fpath)
                except (FileNotFoundError, AnnotationStoreError, ValueError):
                    continue
                shapes_raw = payload.get("shapes") or []
                has_explicit_part = any(
                    isinstance(s, Mapping) and is_body_part_shape_payload(s)
                    for s in shapes_raw
                )
                if not has_png and not has_explicit_part:
                    continue
                (
                    p_masks,
                    _p_polys,
                    _p_gids,
                    s_parts,
                    s_kps,
                ) = parse_frame_instances_and_seed_parts(
                    payload,
                    default_height=self.video_height,
                    default_width=self.video_width,
                    infer_nested_seeds=has_png,
                )
                if not p_masks or not s_parts:
                    continue
                frame_rgb = self.video_loader.load_frame(int(fnum))
                if backbone is None and self.config.feature_backend in {
                    "dino",
                    "hybrid",
                }:
                    backbone = getattr(self.segmenter.extractor, "model", None)
                with torch_cancellable_forward(backbone, stop_event):
                    built = self.segmenter.fit_seed_frame(
                        frame_rgb,
                        p_masks,
                        s_parts,
                        keypoints_by_instance=s_kps,
                    )
                if built > 0:
                    seed_frames_fitted += 1

        # Pass 2: Segment body parts inside tracked instances across target frame range.
        target_frames = [
            fnum
            for fnum in frame_paths
            if start_frame <= fnum <= end_frame and ((fnum - start_frame) % step == 0)
        ]
        if not target_frames:
            return "No tracked frames matched the requested frame range."

        self.segmenter.reset_temporal_state()
        frames_processed = 0
        instances_segmented = 0
        parts_emitted = 0
        part_counts: Dict[str, int] = defaultdict(int)
        modes_used: set[str] = set()

        total_targets = max(1, len(target_frames))
        for idx, fnum in enumerate(target_frames, start=1):
            if self._should_stop():
                return "Body-part segmentation stopped by user."
            fpath = frame_paths[fnum]
            try:
                payload = load_labelme_json(fpath)
            except (FileNotFoundError, AnnotationStoreError, ValueError):
                continue

            has_png = fpath.with_suffix(".png").exists()
            (
                p_masks,
                p_polys,
                p_gids,
                s_parts,
                kps_by_inst,
            ) = parse_frame_instances_and_seed_parts(
                payload,
                default_height=self.video_height,
                default_width=self.video_width,
                infer_nested_seeds=has_png,
            )
            if not p_masks:
                continue

            frame_rgb = self.video_loader.load_frame(int(fnum))
            if backbone is None and self.config.feature_backend in {"dino", "hybrid"}:
                backbone = getattr(self.segmenter.extractor, "model", None)

            for inst_lbl in p_masks:
                modes_used.add(
                    self.segmenter._resolve_mode_for_instance(
                        inst_lbl, kps_by_inst.get(inst_lbl)
                    )
                )

            with torch_cancellable_forward(backbone, stop_event):
                preds_by_inst = self.segmenter.segment_frame(
                    frame_rgb,
                    p_masks,
                    keypoints_by_instance=kps_by_inst,
                    skeleton_edges=self.skeleton_edges,
                    parent_polygons=p_polys,
                    group_ids=p_gids,
                )
            if self._should_stop():
                return "Body-part segmentation stopped by user."

            # Merge predicted body-part shapes while preserving non-body-part shapes & manual seeds!
            existing_shapes = payload.get("shapes") or []
            kept_shapes: List[Dict[str, Any]] = []
            for shape in existing_shapes:
                if not isinstance(shape, Mapping):
                    continue
                if is_body_part_shape_payload(shape):
                    # On manual seed frames, preserve manual seed parts if user drew them;
                    # on predicted frames, replace body parts cleanly.
                    if (
                        has_png
                        and str(shape.get("description") or "") != "InstanceBodyPartSeg"
                    ):
                        kept_shapes.append(dict(shape))
                    continue
                # If a shape on a manual frame was auto-inferred as a nested seed part, keep it
                # promoted with proper body-part metadata so downstream tools treat it cleanly!
                if has_png and s_parts:
                    raw_lbl = str(shape.get("label") or "").strip()
                    matched_parent = next(
                        (
                            p_name
                            for p_name, part_dict in s_parts.items()
                            if raw_lbl in part_dict and raw_lbl != p_name
                        ),
                        None,
                    )
                    if (
                        matched_parent is not None
                        and str(shape.get("shape_type") or "").lower() == "polygon"
                    ):
                        promoted = dict(shape)
                        promoted.update(
                            build_body_part_shape_dict(
                                points=list(shape.get("points") or []),
                                parent_instance=matched_parent,
                                part_label=raw_lbl,
                                group_id=(
                                    shape.get("group_id")
                                    if shape.get("group_id") is not None
                                    else p_gids.get(matched_parent)
                                ),
                                confidence=1.0,
                                mode="manual_seed",
                                description=str(
                                    shape.get("description") or "manual_seed"
                                ),
                            )
                        )
                        kept_shapes.append(promoted)
                        continue
                kept_shapes.append(dict(shape))

            # Build set of (parent, part) already covered by preserved manual seed shapes on this frame.
            manual_covered = {
                (
                    extract_body_part_parent_instance(s),
                    extract_body_part_label(s),
                )
                for s in kept_shapes
                if is_body_part_shape_payload(s)
            }

            new_part_shapes: List[Dict[str, Any]] = []
            for inst_lbl, part_preds in preds_by_inst.items():
                if part_preds:
                    instances_segmented += 1
                for part_lbl, pred in part_preds.items():
                    if (inst_lbl, part_lbl) in manual_covered:
                        continue
                    label_override = (
                        f"{inst_lbl}:{part_lbl}"
                        if self.config.qualify_label_with_instance
                        else part_lbl
                    )
                    mode_used = self.segmenter._resolve_mode_for_instance(
                        inst_lbl, kps_by_inst.get(inst_lbl)
                    )
                    new_part_shapes.append(
                        build_body_part_shape_dict(
                            points=[[float(x), float(y)] for x, y in pred.polygon],
                            parent_instance=inst_lbl,
                            part_label=part_lbl,
                            group_id=pred.group_id,
                            label_override=label_override,
                            confidence=pred.confidence,
                            area_ratio=pred.area_ratio,
                            mode=mode_used,
                        )
                    )
                    parts_emitted += 1
                    part_counts[part_lbl] += 1

            updated_payload = dict(payload)
            updated_payload["shapes"] = kept_shapes + new_part_shapes
            updated_payload.setdefault("imageHeight", self.video_height)
            updated_payload.setdefault("imageWidth", self.video_width)
            self._write_updated_frame_payload(fnum, fpath, updated_payload)
            frames_processed += 1
            self._report_progress(int(round(100.0 * idx / total_targets)))

        self.last_summary = {
            "frames_processed": frames_processed,
            "seed_frames_fitted": seed_frames_fitted,
            "instances_segmented": instances_segmented,
            "parts_emitted": parts_emitted,
            "part_counts": dict(sorted(part_counts.items())),
            "modes_used": sorted(modes_used),
        }
        return (
            f"Segmented {parts_emitted} body-part polygon(s) across "
            f"{frames_processed} frame(s) ({instances_segmented} instance observation(s))."
        )
