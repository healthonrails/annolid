"""Geometric body-part segmentation -> pose keypoint estimation.

Converts per-animal body-part segmentation masks (from either an instance
segmentation model like ``YOLO11n-seg`` or a dense temporal semantic body-part
tracker like ``Compressed Cutie-BodyPartSeg``) into standard Annolid
``PoseInstance`` / ``PoseFrame`` / ``PoseDataset`` objects.

Supports the CalMS21/MARS 7-keypoint mouse skeleton:
``(nose, left_ear, right_ear, neck, left_hip, right_hip, tail_base)``
with explicit chirality control (``calms21`` vs. ``anatomical``) and zero
learned keypoint coordinate bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from annolid.annotation.pose_interop import (
    PoseFrame,
    PoseInstance,
    PosePoint,
)

CALMS7_KEYPOINTS: Tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_hip",
    "right_hip",
    "tail_base",
)

INSECT_3TAGMA_KEYPOINTS: Tuple[str, ...] = (
    "head_tip",
    "neck_joint",
    "thorax_center",
    "thorax_abdomen_joint",
    "abdomen_tip",
)


@dataclass(frozen=True)
class BodyPartPoseConfig:
    """Geometric hyperparameters for deriving 7 mouse keypoints from body-part masks."""

    chirality: str = (
        "calms21"  # "calms21" (cross(neck->nose, R->L) > 0) or "anatomical" (< 0)
    )
    hip_along_spine_frac: float = 0.62
    hip_flank_inset_ratio: float = 0.94
    default_hip_half_width_ratio: float = 0.30
    default_ear_half_width_ratio: float = 0.24
    neck_head_weight: float = 0.28
    nose_extreme_percentile: float = 95.0
    tail_base_extreme_percentile: float = 96.0
    min_part_area_px: int = 4


def _mask_points_and_centroid(
    mask: Optional[np.ndarray],
    *,
    threshold: float = 0.35,
    min_area: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Return ``(pts_xy [N,2], weighted_centroid [2], area_px)`` for a binary/prob mask."""
    if mask is None:
        return None, None, 0.0
    arr = np.asarray(mask, dtype=np.float32)
    fg = arr > threshold
    n = int(np.count_nonzero(fg))
    if n < min_area:
        return None, None, 0.0
    ys, xs = np.nonzero(fg)
    weights = arr[ys, xs]
    wsum = float(weights.sum())
    if wsum <= 1e-6:
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        cx = float((xs * weights).sum() / wsum)
        cy = float((ys * weights).sum() / wsum)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=-1)
    return pts, np.array([cx, cy], dtype=np.float64), float(n)


def _directional_extreme(
    pts_xy: np.ndarray,
    direction_u: np.ndarray,
    *,
    percentile: float = 95.0,
) -> np.ndarray:
    """Return robust mean of the top ``(100 - percentile)%`` pixels along ``direction_u``."""
    proj = pts_xy @ direction_u
    thresh = float(np.percentile(proj, percentile))
    top = pts_xy[proj >= thresh]
    if top.shape[0] == 0:
        return pts_xy[int(np.argmax(proj))]
    return top.mean(axis=0)


def _lateral_flank_bounds(
    pts_xy: np.ndarray,
    center: np.ndarray,
    axis_fwd: np.ndarray,
    axis_left: np.ndarray,
    *,
    slice_half_thickness: float,
    inset_ratio: float,
) -> Optional[Tuple[float, float]]:
    """Measure ``(left_dist, right_dist)`` lateral contour extent along ``axis_left`` at ``center``."""
    rel = pts_xy - center[None, :]
    along = rel @ axis_fwd
    band = rel[np.abs(along) <= slice_half_thickness]
    if band.shape[0] < 6:
        band = rel[np.abs(along) <= slice_half_thickness * 2.0]
    if band.shape[0] < 6:
        return None
    lat = band @ axis_left
    pos = float(np.percentile(lat, 97.0)) * inset_ratio
    neg = float(np.percentile(lat, 3.0)) * inset_ratio
    if pos <= 0.0 or neg >= 0.0:
        return None
    return pos, neg


def _split_ear_components(
    ear_masks: Sequence[np.ndarray],
    *,
    threshold: float = 0.35,
    min_area: int = 4,
) -> List[Tuple[np.ndarray, float]]:
    """Return list of ``(centroid_xy, area)`` for individual ear masks or connected components."""
    centroids: List[Tuple[np.ndarray, float]] = []
    for em in ear_masks:
        arr = np.asarray(em, dtype=np.float32)
        bin_u8 = (arr > threshold).astype(np.uint8)
        n_cc, labels, stats, cc_centroids = cv2.connectedComponentsWithStats(
            bin_u8, connectivity=8
        )
        for ci in range(1, n_cc):
            area = float(stats[ci, cv2.CC_STAT_AREA])
            if area >= min_area:
                centroids.append(
                    (
                        np.array(
                            [float(cc_centroids[ci, 0]), float(cc_centroids[ci, 1])],
                            dtype=np.float64,
                        ),
                        area,
                    )
                )
    centroids.sort(key=lambda item: item[1], reverse=True)
    return centroids


def extract_pose_from_part_masks(
    part_masks: Mapping[str, Optional[np.ndarray]],
    *,
    instance_mask: Optional[np.ndarray] = None,
    ear_masks: Optional[Sequence[np.ndarray]] = None,
    prev_heading_u: Optional[np.ndarray] = None,
    config: Optional[BodyPartPoseConfig] = None,
) -> Tuple[Dict[str, PosePoint], Optional[np.ndarray]]:
    """Derive 7 CalMS21 keypoints deterministically from an animal's body-part masks.

    Parameters
    ----------
    part_masks:
        Mapping with optional binary/probability 2D masks for keys
        ``"head"``, ``"snout"``, ``"ear"``, ``"torso"``, ``"tail"``.
    instance_mask:
        Optional whole-body silhouette mask (used as fallback when ``torso`` or
        ``head`` is partially missing).
    ear_masks:
        Optional list of individual ear masks (e.g. multiple ``ear`` instance
        detections assigned to this mouse). If omitted, ``part_masks.get("ear")``
        is split by connected components.
    prev_heading_u:
        Optional previous-frame unit heading vector ``(tail_base -> nose)`` used
        only as a tie-breaker when ``head``, ``snout``, and ``tail`` are ALL absent.
    config:
        :class:`BodyPartPoseConfig`.

    Returns
    -------
    ``(keypoints_dict, heading_unit_vector)``
    """
    cfg = config or BodyPartPoseConfig()
    min_a = int(cfg.min_part_area_px)

    head_pts, head_c, head_a = _mask_points_and_centroid(
        part_masks.get("head"), min_area=min_a
    )
    snout_pts, snout_c, snout_a = _mask_points_and_centroid(
        part_masks.get("snout"), min_area=min_a
    )
    torso_pts, torso_c, torso_a = _mask_points_and_centroid(
        part_masks.get("torso"), min_area=min_a
    )
    tail_pts, tail_c, tail_a = _mask_points_and_centroid(
        part_masks.get("tail"), min_area=min_a
    )
    inst_pts, inst_c, inst_a = _mask_points_and_centroid(instance_mask, min_area=min_a)

    # Build combined body cloud for fallback/geometry
    body_clouds = [c for c in (head_pts, snout_pts, torso_pts) if c is not None]
    if body_clouds:
        body_pts = np.concatenate(body_clouds, axis=0)
    elif inst_pts is not None:
        body_pts = inst_pts
    else:
        return {}, prev_heading_u

    body_c = body_pts.mean(axis=0)

    # Collect ear centroids
    raw_ear_list: List[np.ndarray] = []
    if ear_masks:
        raw_ear_list.extend(ear_masks)
    elif part_masks.get("ear") is not None:
        raw_ear_list.append(part_masks["ear"])  # type: ignore[arg-type]
    ear_items = _split_ear_components(raw_ear_list, min_area=max(3, min_a - 1))

    # ------------------------------------------------------------------
    # 1. Estimate unambiguous forward heading unit vector (tail -> nose)
    # ------------------------------------------------------------------
    votes: List[Tuple[np.ndarray, float]] = []
    anchor_back = torso_c if torso_c is not None else body_c
    if snout_c is not None and head_c is not None:
        v = snout_c - head_c
        if np.linalg.norm(v) > 2.0:
            votes.append((v / np.linalg.norm(v), 2.5))
    if snout_c is not None and anchor_back is not None:
        v = snout_c - anchor_back
        if np.linalg.norm(v) > 4.0:
            votes.append((v / np.linalg.norm(v), 3.0))
    if head_c is not None and torso_c is not None:
        v = head_c - torso_c
        if np.linalg.norm(v) > 3.0:
            votes.append((v / np.linalg.norm(v), 3.5))
    if ear_items and torso_c is not None:
        ear_mean = np.mean([c for c, _ in ear_items[:2]], axis=0)
        v = ear_mean - torso_c
        if np.linalg.norm(v) > 3.0:
            votes.append((v / np.linalg.norm(v), 2.0))
    if tail_pts is not None and anchor_back is not None:
        # Proximal tail sits at the back of torso; vector tail_prox -> body_c points forward
        d_tb = np.linalg.norm(tail_pts - anchor_back[None, :], axis=1)
        prox_tail = tail_pts[
            np.argsort(d_tb)[: max(4, min(20, len(tail_pts) // 4))]
        ].mean(axis=0)
        v = anchor_back - prox_tail
        if np.linalg.norm(v) > 3.0:
            votes.append((v / np.linalg.norm(v), 2.2))

    heading_u: Optional[np.ndarray] = None
    if votes:
        vec_sum = np.zeros(2, dtype=np.float64)
        for u, w in votes:
            vec_sum += u * w
        nrm = float(np.linalg.norm(vec_sum))
        if nrm > 1e-6:
            heading_u = vec_sum / nrm

    if heading_u is None:
        # PCA major axis of body silhouette, disambiguated by previous heading
        centered = body_pts - body_c[None, :]
        if centered.shape[0] >= 4:
            cov = (centered.T @ centered) / float(centered.shape[0])
            eigvals, eigvecs = np.linalg.eigh(cov)
            major = eigvecs[:, int(np.argmax(eigvals))]
            major = major / max(1e-6, float(np.linalg.norm(major)))
            if (
                prev_heading_u is not None
                and float(np.dot(major, prev_heading_u)) < 0.0
            ):
                major = -major
            heading_u = major
        elif prev_heading_u is not None:
            heading_u = prev_heading_u / max(
                1e-6, float(np.linalg.norm(prev_heading_u))
            )
        else:
            heading_u = np.array([1.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Extract nose
    # ------------------------------------------------------------------
    if snout_pts is not None:
        nose_xy = _directional_extreme(
            snout_pts, heading_u, percentile=min(98.0, cfg.nose_extreme_percentile)
        )
        nose_conf = 0.95
    elif head_pts is not None:
        nose_xy = _directional_extreme(
            head_pts, heading_u, percentile=cfg.nose_extreme_percentile
        )
        nose_conf = 0.80
    else:
        nose_xy = _directional_extreme(
            body_pts, heading_u, percentile=cfg.nose_extreme_percentile
        )
        nose_conf = 0.55

    # ------------------------------------------------------------------
    # 3. Extract tail_base
    # ------------------------------------------------------------------
    torso_ref_pts = torso_pts if torso_pts is not None else body_pts
    body_scale = max(24.0, float(np.sqrt(float(max(1, body_pts.shape[0])))))
    if tail_pts is not None and torso_ref_pts is not None:
        # Closest point on posterior half of torso to proximal tail
        t_proj = (torso_ref_pts - body_c[None, :]) @ heading_u
        posterior_torso = torso_ref_pts[t_proj <= np.percentile(t_proj, 40.0)]
        if posterior_torso.shape[0] < 4:
            posterior_torso = torso_ref_pts
        # Subsample for speed when masks are large
        p_sub = posterior_torso[:: max(1, len(posterior_torso) // 160)]
        t_sub = tail_pts[:: max(1, len(tail_pts) // 160)]
        dists = np.linalg.norm(p_sub[:, None, :] - t_sub[None, :, :], axis=-1)
        min_idx = int(np.argmin(dists))
        pi, ti = divmod(min_idx, t_sub.shape[0])
        min_dist = float(dists[pi, ti])
        if min_dist < 0.45 * body_scale:
            tail_base_xy = 0.65 * p_sub[pi] + 0.35 * t_sub[ti]
            tail_conf = 0.92
        else:
            tail_base_xy = _directional_extreme(
                torso_ref_pts, -heading_u, percentile=cfg.tail_base_extreme_percentile
            )
            tail_conf = 0.72
    else:
        tail_base_xy = _directional_extreme(
            torso_ref_pts, -heading_u, percentile=cfg.tail_base_extreme_percentile
        )
        tail_conf = 0.70

    # Refine spine axis from tail_base -> nose if well-separated
    spine_vec = nose_xy - tail_base_xy
    body_len = float(np.linalg.norm(spine_vec))
    if body_len > 15.0:
        spine_u = spine_vec / body_len
        if float(np.dot(spine_u, heading_u)) > 0.3:
            heading_u = spine_u

    # ------------------------------------------------------------------
    # 4. Extract neck (scapular interface between head and torso)
    # ------------------------------------------------------------------
    if head_pts is not None and torso_pts is not None:
        # Anterior edge of torso + posterior edge of head blended toward scapular junction
        head_back = _directional_extreme(head_pts, -heading_u, percentile=82.0)
        torso_front = _directional_extreme(torso_pts, heading_u, percentile=84.0)
        # Shift slightly posteriorly into upper torso to match CalMS21 scapular neck
        neck_xy = (
            cfg.neck_head_weight * head_back
            + (1.0 - cfg.neck_head_weight) * torso_front
        )
        neck_conf = 0.90
    elif head_c is not None and torso_c is not None:
        neck_xy = 0.45 * head_c + 0.55 * torso_c
        neck_conf = 0.80
    else:
        neck_xy = tail_base_xy + 0.72 * (nose_xy - tail_base_xy)
        neck_conf = 0.55

    # Local head axis for ear chirality (neck -> nose)
    head_fwd = nose_xy - neck_xy
    head_len = float(np.linalg.norm(head_fwd))
    if head_len > 5.0:
        head_fwd_u = head_fwd / head_len
    else:
        head_fwd_u = heading_u
    if cfg.chirality.lower() == "anatomical":
        head_left_u = np.array([head_fwd_u[1], -head_fwd_u[0]], dtype=np.float64)
    else:
        head_left_u = np.array([-head_fwd_u[1], head_fwd_u[0]], dtype=np.float64)

    # ------------------------------------------------------------------
    # 5. Extract left_ear & right_ear (with exact chirality assignment)
    # ------------------------------------------------------------------
    ear_center_ref = 0.55 * neck_xy + 0.45 * nose_xy
    if head_c is not None:
        ear_center_ref = 0.5 * head_c + 0.5 * ear_center_ref
    ear_half_w = max(8.0, cfg.default_ear_half_width_ratio * max(body_len, 40.0) * 0.55)

    # Filter ear candidates to reasonable proximity of head/neck
    valid_ears: List[Tuple[np.ndarray, float]] = []
    max_ear_dist = max(45.0, 0.45 * body_len)
    for ec, ea in ear_items:
        if float(np.linalg.norm(ec - ear_center_ref)) <= max_ear_dist:
            valid_ears.append((ec, ea))

    left_ear_xy: Optional[np.ndarray] = None
    right_ear_xy: Optional[np.ndarray] = None
    le_conf, re_conf = 0.65, 0.65

    if len(valid_ears) >= 2:
        c0, c1 = valid_ears[0][0], valid_ears[1][0]
        # Order by projection onto head_left_u so cross(head_fwd, right->left) matches chirality!
        if float(np.dot(c0 - c1, head_left_u)) >= 0.0:
            left_ear_xy, right_ear_xy = c0, c1
        else:
            left_ear_xy, right_ear_xy = c1, c0
        le_conf, re_conf = 0.92, 0.92
    elif len(valid_ears) == 1:
        ec = valid_ears[0][0]
        side_s = float(np.dot(ec - ear_center_ref, head_left_u))
        if side_s >= 0.0:
            left_ear_xy = ec
            right_ear_xy = ear_center_ref - max(ear_half_w, abs(side_s)) * head_left_u
            le_conf, re_conf = 0.88, 0.65
        else:
            right_ear_xy = ec
            left_ear_xy = ear_center_ref + max(ear_half_w, abs(side_s)) * head_left_u
            le_conf, re_conf = 0.65, 0.88
    else:
        # Geometric fallback from head lateral silhouette contour
        flank = (
            _lateral_flank_bounds(
                head_pts,
                ear_center_ref,
                head_fwd_u,
                head_left_u,
                slice_half_thickness=max(5.0, 0.14 * max(head_len, 25.0)),
                inset_ratio=0.95,
            )
            if head_pts is not None
            else None
        )
        if flank is not None:
            l_dist, r_dist = flank
            left_ear_xy = ear_center_ref + l_dist * head_left_u
            right_ear_xy = ear_center_ref + r_dist * head_left_u
            le_conf, re_conf = 0.72, 0.72
        else:
            left_ear_xy = ear_center_ref + ear_half_w * head_left_u
            right_ear_xy = ear_center_ref - ear_half_w * head_left_u
            le_conf, re_conf = 0.50, 0.50

    # ------------------------------------------------------------------
    # 6. Extract left_hip & right_hip from torso outer pelvic contour
    # ------------------------------------------------------------------
    nt_vec = tail_base_xy - neck_xy
    nt_len = float(np.linalg.norm(nt_vec))
    if nt_len > 10.0:
        torso_fwd_u = -nt_vec / nt_len
    else:
        torso_fwd_u = heading_u
    if cfg.chirality.lower() == "anatomical":
        torso_left_u = np.array([torso_fwd_u[1], -torso_fwd_u[0]], dtype=np.float64)
    else:
        torso_left_u = np.array([-torso_fwd_u[1], torso_fwd_u[0]], dtype=np.float64)

    hip_center = neck_xy + cfg.hip_along_spine_frac * nt_vec
    hip_flank = _lateral_flank_bounds(
        torso_ref_pts,
        hip_center,
        torso_fwd_u,
        torso_left_u,
        slice_half_thickness=max(6.0, 0.14 * max(nt_len, 30.0)),
        inset_ratio=cfg.hip_flank_inset_ratio,
    )
    default_hw = max(10.0, cfg.default_hip_half_width_ratio * max(nt_len, 40.0))
    if hip_flank is not None:
        l_d, r_d = hip_flank
        l_d = float(np.clip(l_d, 0.16 * max(nt_len, 30.0), 0.55 * max(nt_len, 30.0)))
        r_d = -float(np.clip(-r_d, 0.16 * max(nt_len, 30.0), 0.55 * max(nt_len, 30.0)))
        left_hip_xy = hip_center + l_d * torso_left_u
        right_hip_xy = hip_center + r_d * torso_left_u
        hip_conf = 0.90 if torso_pts is not None else 0.72
    else:
        left_hip_xy = hip_center + default_hw * torso_left_u
        right_hip_xy = hip_center - default_hw * torso_left_u
        hip_conf = 0.55

    keypoints: Dict[str, PosePoint] = {
        "nose": PosePoint(
            x=float(nose_xy[0]), y=float(nose_xy[1]), visibility=2, score=nose_conf
        ),
        "left_ear": PosePoint(
            x=float(left_ear_xy[0]),
            y=float(left_ear_xy[1]),
            visibility=2,
            score=le_conf,
        ),
        "right_ear": PosePoint(
            x=float(right_ear_xy[0]),
            y=float(right_ear_xy[1]),
            visibility=2,
            score=re_conf,
        ),
        "neck": PosePoint(
            x=float(neck_xy[0]), y=float(neck_xy[1]), visibility=2, score=neck_conf
        ),
        "left_hip": PosePoint(
            x=float(left_hip_xy[0]),
            y=float(left_hip_xy[1]),
            visibility=2,
            score=hip_conf,
        ),
        "right_hip": PosePoint(
            x=float(right_hip_xy[0]),
            y=float(right_hip_xy[1]),
            visibility=2,
            score=hip_conf,
        ),
        "tail_base": PosePoint(
            x=float(tail_base_xy[0]),
            y=float(tail_base_xy[1]),
            visibility=2,
            score=tail_conf,
        ),
    }
    return keypoints, heading_u


def _closest_mask_junction(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    *,
    weight_a: float = 0.5,
    band_px: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """Return centroid midpoint and min distance of the boundary interface between two masks.

    Averages all pixel pairs within ``min_dist + band_px`` so touching masks place the
    joint at the center of the shared anatomical interface rather than the top scanline pixel.
    """
    sub_a = pts_a[:: max(1, len(pts_a) // 180)]
    sub_b = pts_b[:: max(1, len(pts_b) // 180)]
    dists = np.linalg.norm(sub_a[:, None, :] - sub_b[None, :, :], axis=-1)
    min_d = float(dists.min())
    ia, ib = np.nonzero(dists <= min_d + max(0.5, float(band_px)))
    if ia.size == 0:
        idx = int(np.argmin(dists))
        r, c = divmod(idx, sub_b.shape[0])
        return weight_a * sub_a[r] + (1.0 - weight_a) * sub_b[c], min_d
    mids = weight_a * sub_a[ia] + (1.0 - weight_a) * sub_b[ib]
    return mids.mean(axis=0), min_d


def _core_tagma_points(
    mask: Optional[np.ndarray],
    *,
    open_radius_px: int = 4,
    min_area: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Strip thin legs/antennae via morphological opening when core area remains intact."""
    if mask is None:
        return None, None, 0.0
    arr = np.asarray(mask, dtype=np.float32)
    fg = (arr > 0.35).astype(np.uint8)
    raw_area = int(np.count_nonzero(fg))
    if raw_area < min_area:
        return None, None, 0.0
    if open_radius_px >= 2:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * open_radius_px + 1, 2 * open_radius_px + 1)
        )
        opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        if int(np.count_nonzero(opened)) >= max(min_area, int(0.25 * raw_area)):
            return _mask_points_and_centroid(
                opened.astype(np.float32), threshold=0.5, min_area=min_area
            )
    return _mask_points_and_centroid(arr, threshold=0.35, min_area=min_area)


def extract_insect_pose_from_part_masks(
    part_masks: Mapping[str, Optional[np.ndarray]],
    *,
    prev_heading_u: Optional[np.ndarray] = None,
    min_part_area_px: int = 4,
) -> Tuple[Dict[str, PosePoint], Optional[np.ndarray]]:
    """Derive 5 canonical insect keypoints from chirality-free tagma masks.

    Supports ``{"head", "thorax", "abdomen"}`` masks (e.g., side-view or top-down
    *Drosophila* body-part segmentation) and returns deterministic landmarks:
    ``head_tip``, ``neck_joint``, ``thorax_center``, ``thorax_abdomen_joint``,
    and ``abdomen_tip`` along with the unit anterior body heading vector.
    """
    raw_present = [
        p
        for p in (
            _mask_points_and_centroid(
                part_masks.get("head"), min_area=min_part_area_px
            )[0],
            _mask_points_and_centroid(
                part_masks.get("thorax"), min_area=min_part_area_px
            )[0],
            _mask_points_and_centroid(
                part_masks.get("abdomen"), min_area=min_part_area_px
            )[0],
        )
        if p is not None
    ]
    if not raw_present:
        return {}, prev_heading_u
    raw_scale = max(16.0, float(np.sqrt(float(sum(p.shape[0] for p in raw_present)))))
    open_r = max(2, min(7, int(round(0.055 * raw_scale))))

    head_pts, head_c, _ = _core_tagma_points(
        part_masks.get("head"), open_radius_px=open_r, min_area=min_part_area_px
    )
    thorax_pts, thorax_c, _ = _core_tagma_points(
        part_masks.get("thorax"), open_radius_px=open_r, min_area=min_part_area_px
    )
    abdomen_pts, abdomen_c, _ = _core_tagma_points(
        part_masks.get("abdomen"), open_radius_px=open_r, min_area=min_part_area_px
    )

    present = [p for p in (head_pts, thorax_pts, abdomen_pts) if p is not None]
    if not present:
        return {}, prev_heading_u
    body_pts = np.concatenate(present, axis=0)
    body_c = body_pts.mean(axis=0)
    body_scale = max(16.0, float(np.sqrt(float(max(1, body_pts.shape[0])))))

    votes: List[Tuple[np.ndarray, float]] = []
    if head_c is not None and abdomen_c is not None:
        v = head_c - abdomen_c
        nrm = float(np.linalg.norm(v))
        if nrm > 2.0:
            votes.append((v / nrm, 3.5))
    if head_c is not None and thorax_c is not None:
        v = head_c - thorax_c
        nrm = float(np.linalg.norm(v))
        if nrm > 2.0:
            votes.append((v / nrm, 3.0))
    if thorax_c is not None and abdomen_c is not None:
        v = thorax_c - abdomen_c
        nrm = float(np.linalg.norm(v))
        if nrm > 2.0:
            votes.append((v / nrm, 3.0))

    heading_u: Optional[np.ndarray] = None
    if votes:
        vec_sum = sum((u * w for u, w in votes), np.zeros(2, dtype=np.float64))
        nrm = float(np.linalg.norm(vec_sum))
        if nrm > 1e-6:
            heading_u = vec_sum / nrm
    if heading_u is None:
        centered = body_pts - body_c[None, :]
        if centered.shape[0] >= 4:
            cov = (centered.T @ centered) / float(centered.shape[0])
            eigvals, eigvecs = np.linalg.eigh(cov)
            major = eigvecs[:, int(np.argmax(eigvals))]
            major = major / max(1e-6, float(np.linalg.norm(major)))
            if (
                prev_heading_u is not None
                and float(np.dot(major, prev_heading_u)) < 0.0
            ):
                major = -major
            heading_u = major
        elif prev_heading_u is not None:
            heading_u = prev_heading_u / max(
                1e-6, float(np.linalg.norm(prev_heading_u))
            )
        else:
            heading_u = np.array([1.0, 0.0], dtype=np.float64)

    # 1. head_tip: anterior rim of the solid head capsule via distance-transform peak
    # (immune to outstretched forelegs/antennae/proboscis occasionally merged into head masks)
    if head_pts is not None:
        head_axis = heading_u
        if head_c is not None and thorax_c is not None:
            hv = head_c - thorax_c
            if float(np.linalg.norm(hv)) > 2.0:
                head_axis = hv / float(np.linalg.norm(hv))
        h_raw = part_masks.get("head")
        if h_raw is not None:
            h_bin = (np.asarray(h_raw, dtype=np.float32) > 0.35).astype(np.uint8)
            dt = cv2.distanceTransform(h_bin, cv2.DIST_L2, 5)
            max_r = float(dt.max())
            if max_r >= 3.0:
                cy_pk, cx_pk = np.unravel_index(int(np.argmax(dt)), dt.shape)
                core_c = np.array([float(cx_pk), float(cy_pk)], dtype=np.float64)
                if thorax_c is not None:
                    hv2 = core_c - thorax_c
                    if float(np.linalg.norm(hv2)) > 2.0:
                        head_axis = hv2 / float(np.linalg.norm(hv2))
                head_tip_xy = core_c + 0.85 * max_r * head_axis
            else:
                head_tip_xy = _directional_extreme(head_pts, head_axis, percentile=94.0)
        else:
            head_tip_xy = _directional_extreme(head_pts, head_axis, percentile=94.0)
        head_tip_conf = 0.94
    else:
        head_tip_xy = _directional_extreme(body_pts, heading_u, percentile=94.0)
        head_tip_conf = 0.55

    # 2. neck_joint: closest interface between head and thorax
    if head_pts is not None and thorax_pts is not None:
        j_xy, j_dist = _closest_mask_junction(head_pts, thorax_pts, weight_a=0.5)
        if j_dist <= 0.45 * body_scale:
            neck_xy = j_xy
            neck_conf = 0.94
        else:
            neck_xy = 0.5 * _directional_extreme(
                head_pts, -heading_u, percentile=84.0
            ) + 0.5 * _directional_extreme(thorax_pts, heading_u, percentile=84.0)
            neck_conf = 0.78
    elif head_c is not None and thorax_c is not None:
        neck_xy = 0.5 * head_c + 0.5 * thorax_c
        neck_conf = 0.75
    else:
        neck_xy = body_c + 0.25 * body_scale * heading_u
        neck_conf = 0.50

    # 3. thorax_center: centroid of thorax tagma
    if thorax_c is not None:
        thorax_xy = thorax_c
        thorax_conf = 0.96
    else:
        thorax_xy = body_c
        thorax_conf = 0.55

    # 4. thorax_abdomen_joint: closest interface between thorax and abdomen
    if thorax_pts is not None and abdomen_pts is not None:
        ta_xy, ta_dist = _closest_mask_junction(thorax_pts, abdomen_pts, weight_a=0.5)
        if ta_dist <= 0.45 * body_scale:
            ta_joint_xy = ta_xy
            ta_conf = 0.94
        else:
            ta_joint_xy = 0.5 * _directional_extreme(
                thorax_pts, -heading_u, percentile=84.0
            ) + 0.5 * _directional_extreme(abdomen_pts, heading_u, percentile=84.0)
            ta_conf = 0.78
    elif thorax_c is not None and abdomen_c is not None:
        ta_joint_xy = 0.5 * thorax_c + 0.5 * abdomen_c
        ta_conf = 0.75
    else:
        ta_joint_xy = body_c - 0.25 * body_scale * heading_u
        ta_conf = 0.50

    # 5. abdomen_tip: posterior extreme of abdomen along abdomen axis
    if abdomen_pts is not None:
        abd_axis = -heading_u
        if thorax_c is not None and abdomen_c is not None:
            av = abdomen_c - thorax_c
            if float(np.linalg.norm(av)) > 2.0:
                abd_axis = av / float(np.linalg.norm(av))
        abd_tip_xy = _directional_extreme(abdomen_pts, abd_axis, percentile=94.0)
        abd_tip_conf = 0.94
    else:
        abd_tip_xy = _directional_extreme(body_pts, -heading_u, percentile=94.0)
        abd_tip_conf = 0.55

    keypoints: Dict[str, PosePoint] = {
        "head_tip": PosePoint(
            x=float(head_tip_xy[0]),
            y=float(head_tip_xy[1]),
            visibility=2,
            score=head_tip_conf,
        ),
        "neck_joint": PosePoint(
            x=float(neck_xy[0]),
            y=float(neck_xy[1]),
            visibility=2,
            score=neck_conf,
        ),
        "thorax_center": PosePoint(
            x=float(thorax_xy[0]),
            y=float(thorax_xy[1]),
            visibility=2,
            score=thorax_conf,
        ),
        "thorax_abdomen_joint": PosePoint(
            x=float(ta_joint_xy[0]),
            y=float(ta_joint_xy[1]),
            visibility=2,
            score=ta_conf,
        ),
        "abdomen_tip": PosePoint(
            x=float(abd_tip_xy[0]),
            y=float(abd_tip_xy[1]),
            visibility=2,
            score=abd_tip_conf,
        ),
    }
    return keypoints, heading_u


def _subsample_mask_points(
    binary_mask: np.ndarray,
    *,
    max_points: int = 200,
) -> np.ndarray:
    """Return subsampled ``(M, 2)`` xy pixel coordinates of ``binary_mask``."""
    ys, xs = np.nonzero(binary_mask)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    step = max(1, xs.size // max_points)
    return np.stack([xs[::step], ys[::step]], axis=-1).astype(np.float64)


def _min_mask_boundary_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Return minimum Euclidean pixel distance between two subsampled mask point clouds."""
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return float("inf")
    dists = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    return float(dists.min())


def group_part_detections_to_pose_frame(
    detections: Sequence[Mapping[str, Any]],
    *,
    frame_index: int,
    image_hw: Tuple[int, int],
    parent_classes: Sequence[str] = ("resident", "intruder"),
    prev_headings: Optional[Mapping[str, np.ndarray]] = None,
    config: Optional[BodyPartPoseConfig] = None,
    dedup_parents: bool = True,
) -> Tuple[PoseFrame, Dict[str, np.ndarray]]:
    """Group multi-class body-part mask detections into per-animal ``PoseInstance`` objects.

    Each item in ``detections`` is a dict containing:
    - ``class_name`` (str): e.g. ``"resident"``, ``"intruder"``, ``"head"``, ``"snout"``, ``"ear"``, ``"torso"``, ``"tail"``
    - ``score`` (float): detection confidence
    - ``mask`` (np.ndarray): 2D float/bool mask in ``image_hw`` coordinates
    """
    h, w = image_hw
    parent_set = set(parent_classes)
    parents_raw = [d for d in detections if str(d.get("class_name", "")) in parent_set]
    parts_raw = [
        d for d in detections if str(d.get("class_name", "")) not in parent_set
    ]

    parents_raw.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    selected_parents: List[Dict[str, Any]] = []
    parent_idx_by_name: Dict[str, int] = {}
    parent_pts_sub: List[np.ndarray] = []
    parent_areas: List[int] = []

    for p in parents_raw:
        pname = str(p["class_name"])
        pmask = np.asarray(p["mask"], dtype=np.float32)
        pbin = pmask > 0.35
        parea = int(np.count_nonzero(pbin))
        if parea < 8:
            continue
        ppts = _subsample_mask_points(pbin, max_points=220)
        if dedup_parents and pname in parent_idx_by_name:
            # Merge nearby smaller same-class fragments (e.g., head separated from torso
            # by a bright cranial headstage/miniscope or tether cable) into the parent.
            pi_existing = parent_idx_by_name[pname]
            if parea <= 0.65 * parent_areas[pi_existing]:
                scale_exist = max(
                    24.0, float(np.sqrt(float(parent_areas[pi_existing])))
                )
                d_frag = _min_mask_boundary_distance(parent_pts_sub[pi_existing], ppts)
                if d_frag <= 0.55 * scale_exist:
                    merged_m = np.maximum(selected_parents[pi_existing]["mask"], pmask)
                    selected_parents[pi_existing]["mask"] = merged_m
                    merged_bin = merged_m > 0.35
                    parent_areas[pi_existing] = int(np.count_nonzero(merged_bin))
                    parent_pts_sub[pi_existing] = _subsample_mask_points(
                        merged_bin, max_points=220
                    )
            continue
        if not dedup_parents:
            # Multi-animal N >= 3 same-class mode: keep distinct full-size animals separate,
            # while suppressing high-IoU duplicates and merging nearby smaller headstage fragments.
            merged_or_dup = False
            best_frag_idx: Optional[int] = None
            best_frag_dist = float("inf")
            for ej, exist in enumerate(selected_parents):
                if str(exist["class_name"]) != pname:
                    continue
                exist_bin = np.asarray(exist["mask"], dtype=np.float32) > 0.35
                inter = int(np.count_nonzero(pbin & exist_bin))
                union = int(np.count_nonzero(pbin | exist_bin))
                iou_e = float(inter) / float(max(1, union))
                if iou_e >= 0.45:
                    merged_or_dup = True
                    break
                if parea <= 0.65 * parent_areas[ej]:
                    scale_e = max(24.0, float(np.sqrt(float(parent_areas[ej]))))
                    d_frag = _min_mask_boundary_distance(parent_pts_sub[ej], ppts)
                    if d_frag <= 0.55 * scale_e and d_frag < best_frag_dist:
                        best_frag_dist = d_frag
                        best_frag_idx = ej
            if merged_or_dup:
                continue
            if best_frag_idx is not None:
                merged_m = np.maximum(selected_parents[best_frag_idx]["mask"], pmask)
                selected_parents[best_frag_idx]["mask"] = merged_m
                merged_bin = merged_m > 0.35
                parent_areas[best_frag_idx] = int(np.count_nonzero(merged_bin))
                parent_pts_sub[best_frag_idx] = _subsample_mask_points(
                    merged_bin, max_points=220
                )
                continue
        parent_idx_by_name.setdefault(pname, len(selected_parents))
        selected_parents.append(
            {
                "class_name": pname,
                "score": float(p.get("score", 1.0)),
                "mask": pmask.copy(),
            }
        )
        parent_pts_sub.append(ppts)
        parent_areas.append(parea)

    # Precompute binary parent masks & centroids for part assignment
    parent_bins: List[np.ndarray] = []
    parent_centers: List[np.ndarray] = []
    for p in selected_parents:
        bm = np.asarray(p["mask"], dtype=np.float32) > 0.35
        parent_bins.append(bm)
        ys, xs = np.nonzero(bm)
        if xs.size > 0:
            parent_centers.append(
                np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
            )
        else:
            parent_centers.append(np.array([0.5 * w, 0.5 * h], dtype=np.float64))

    assigned: List[Dict[str, List[Mapping[str, Any]]]] = [
        {"head": [], "snout": [], "ear": [], "torso": [], "tail": []}
        for _ in selected_parents
    ]

    for part in parts_raw:
        cname = str(part.get("class_name", ""))
        if cname not in ("head", "snout", "ear", "torso", "tail"):
            continue
        pm = np.asarray(part["mask"], dtype=np.float32) > 0.35
        p_area = int(np.count_nonzero(pm))
        if p_area < 3 or not selected_parents:
            continue
        p_pts_sub = _subsample_mask_points(pm, max_points=120)

        best_pi: Optional[int] = None
        best_score = -1.0
        for pi, pbin in enumerate(parent_bins):
            scale_p = max(24.0, float(np.sqrt(float(max(1, parent_areas[pi])))))
            inter = int(np.count_nonzero(pm & pbin))
            containment = float(inter) / float(max(1, p_area))
            if cname == "torso":
                if containment >= 0.15:
                    assoc = 1.0 + containment + 0.05 * float(part.get("score", 0.5))
                else:
                    assoc = -1.0
            elif cname == "tail":
                # Tails extend outside the body mask; check scale-relative boundary distance
                if inter > 0:
                    assoc = 1.0 + containment
                else:
                    dmin = _min_mask_boundary_distance(parent_pts_sub[pi], p_pts_sub)
                    tail_r = 0.45 * scale_p
                    assoc = max(0.0, 1.0 - dmin / tail_r) if dmin <= tail_r else -1.0
            else:
                # head, snout, ear: prefer containment, else scale-relative boundary proximity
                # (handles headstage/tether gaps separating anterior head/snout from dark torso)
                if containment >= 0.10:
                    assoc = 1.0 + containment + 0.05 * float(part.get("score", 0.5))
                else:
                    dmin = _min_mask_boundary_distance(parent_pts_sub[pi], p_pts_sub)
                    part_r = 0.60 * scale_p
                    if dmin <= part_r:
                        assoc = (1.0 - dmin / part_r) + 0.05 * float(
                            part.get("score", 0.5)
                        )
                    else:
                        assoc = -1.0
            if assoc > best_score:
                best_score = assoc
                best_pi = pi

        if best_pi is not None and best_score > 0.0:
            assigned[best_pi][cname].append(part)

    instances: List[PoseInstance] = []
    new_headings: Dict[str, np.ndarray] = dict(prev_headings or {})

    # Count occurrences per class name so multi-animal scenes with duplicate parent
    # classes (e.g. N=4 mice all labeled "mouse") receive unique instance names & IDs.
    class_counts: Dict[str, int] = {}
    for parent in selected_parents:
        pn = str(parent["class_name"])
        class_counts[pn] = class_counts.get(pn, 0) + 1
    class_seen: Dict[str, int] = {}

    for pi, parent in enumerate(selected_parents):
        pname = str(parent["class_name"])
        class_seen[pname] = class_seen.get(pname, 0) + 1
        inst_name = f"{pname}_{class_seen[pname]}" if class_counts[pname] > 1 else pname
        p_score = float(parent.get("score", 1.0))
        parent_mask = np.asarray(parent["mask"], dtype=np.float32)

        # Merge part masks per type; unite adjacent same-side head pieces when a headstage splits the head
        part_masks: Dict[str, Optional[np.ndarray]] = {}
        torso_cands = sorted(
            assigned[pi]["torso"],
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )
        torso_ref_c = parent_centers[pi]
        if torso_cands:
            torso_m = np.asarray(torso_cands[0]["mask"], dtype=np.float32) * (
                parent_mask > 0.25
            )
            part_masks["torso"] = torso_m
            _, tc_tmp, _ = _mask_points_and_centroid(torso_m)
            if tc_tmp is not None:
                torso_ref_c = tc_tmp
        else:
            part_masks["torso"] = None

        head_cands = sorted(
            assigned[pi]["head"],
            key=lambda d: float(d.get("score", 0.0)),
            reverse=True,
        )
        if head_cands:
            head_union = np.asarray(head_cands[0]["mask"], dtype=np.float32).copy()
            _, hc0, _ = _mask_points_and_centroid(head_union)
            h_pts0 = _subsample_mask_points(head_union > 0.35, max_points=120)
            for cand in head_cands[1:4]:
                if float(cand.get("score", 0.0)) < 0.20:
                    continue
                cm = np.asarray(cand["mask"], dtype=np.float32)
                _, hck, _ = _mask_points_and_centroid(cm)
                if hc0 is None or hck is None:
                    continue
                same_side = float(np.dot(hc0 - torso_ref_c, hck - torso_ref_c)) > 0.0
                if not same_side:
                    continue
                d_h = _min_mask_boundary_distance(
                    h_pts0, _subsample_mask_points(cm > 0.35, max_points=120)
                )
                scale_pi = max(24.0, float(np.sqrt(float(max(1, parent_areas[pi])))))
                if (
                    d_h <= 0.45 * scale_pi
                    or float(np.linalg.norm(hck - hc0)) <= 0.75 * scale_pi
                ):
                    head_union = np.maximum(head_union, cm)
                    h_pts0 = _subsample_mask_points(head_union > 0.35, max_points=120)
            part_masks["head"] = head_union
        else:
            part_masks["head"] = None

        for ptype in ("snout", "tail"):
            cands = sorted(
                assigned[pi][ptype],
                key=lambda d: float(d.get("score", 0.0)),
                reverse=True,
            )
            part_masks[ptype] = (
                np.asarray(cands[0]["mask"], dtype=np.float32) if cands else None
            )

        ear_cands = sorted(
            assigned[pi]["ear"], key=lambda d: float(d.get("score", 0.0)), reverse=True
        )
        ear_mask_list = [np.asarray(e["mask"], dtype=np.float32) for e in ear_cands[:3]]

        kps, heading_u = extract_pose_from_part_masks(
            part_masks,
            instance_mask=parent_mask,
            ear_masks=ear_mask_list,
            prev_heading_u=(prev_headings or {}).get(inst_name)
            if inst_name in (prev_headings or {})
            else (prev_headings or {}).get(pname),
            config=config,
        )
        if heading_u is not None:
            new_headings[inst_name] = heading_u
        if not kps:
            continue

        if class_counts[pname] > 1:
            tid = pi
        else:
            tid = list(parent_classes).index(pname) if pname in parent_classes else pi
        instances.append(
            PoseInstance(
                instance_name=inst_name,
                keypoints=kps,
                track_id=tid,
                skeleton_name="mouse_calms21_7kp",
                score=p_score,
                is_predicted=True,
            )
        )

    frame = PoseFrame(
        frame_index=int(frame_index),
        instances=instances,
        image_width=w,
        image_height=h,
    )
    return frame, new_headings
