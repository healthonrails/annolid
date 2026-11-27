"""Smart video frame extraction (uniform, k-means visual diversity) and active-learning pose outlier sampling.

Bridges the major frame-selection workflows from DeepLabCut (k-means clustering,
uniform extraction, outlier frame extraction) and SLEAP (visual diversity
sampling and human-in-the-loop refinement queues) cleanly into Annolid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import median
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import numpy as np

from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.core.media.video import CV2Video, read_capture_frame_bgr
from annolid.utils.atomic_io import atomic_write_json

if TYPE_CHECKING:
    from annolid.annotation.pose_interop.model import PoseDataset, PoseFrame

__all__ = [
    "FrameExtractionResult",
    "OutlierFrameScore",
    "OutlierExtractionResult",
    "select_uniform_frame_indices",
    "select_kmeans_frame_indices",
    "select_farthest_point_indices",
    "select_hybrid_frame_indices",
    "select_flow_frame_indices",
    "select_keyframe_indices",
    "discover_labeled_frame_indices",
    "extract_frames_smart",
    "score_pose_outlier_frames",
    "select_diverse_outlier_frames",
    "extract_pose_outlier_frames",
]


@dataclass(frozen=True)
class FrameExtractionResult:
    """Summary of smart video frame extraction."""

    video_path: str
    output_dir: str
    algo: str
    total_video_frames: int
    selected_indices: Tuple[int, ...]
    saved_paths: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "algo": self.algo,
            "total_video_frames": int(self.total_video_frames),
            "num_extracted": len(self.saved_paths),
            "selected_indices": list(self.selected_indices),
            "saved_paths": list(self.saved_paths),
        }


@dataclass(frozen=True)
class OutlierFrameScore:
    """Ranked anomaly / uncertainty score for a single pose frame."""

    frame_index: int
    video_index: int
    total_score: float
    low_conf_count: int
    min_confidence: Optional[float]
    jump_count: int
    max_jump_px: float
    bone_outlier_count: int
    missing_keypoint_count: int
    instance_count: int
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "video_index": int(self.video_index),
            "total_score": round(float(self.total_score), 6),
            "low_conf_count": int(self.low_conf_count),
            "min_confidence": (
                round(float(self.min_confidence), 6)
                if self.min_confidence is not None
                else None
            ),
            "jump_count": int(self.jump_count),
            "max_jump_px": round(float(self.max_jump_px), 4),
            "bone_outlier_count": int(self.bone_outlier_count),
            "missing_keypoint_count": int(self.missing_keypoint_count),
            "instance_count": int(self.instance_count),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OutlierExtractionResult:
    """Summary of active-learning outlier frame extraction."""

    source_path: str
    video_path: str
    output_dir: str
    frames_evaluated: int
    outliers_found: int
    frames_extracted: int
    selected_frames: Tuple[OutlierFrameScore, ...]
    saved_images: Tuple[str, ...]
    manifest_path: str
    pose_schema_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "frames_evaluated": int(self.frames_evaluated),
            "outliers_found": int(self.outliers_found),
            "frames_extracted": int(self.frames_extracted),
            "selected_frames": [item.to_dict() for item in self.selected_frames],
            "saved_images": list(self.saved_images),
            "manifest_path": self.manifest_path,
            "pose_schema_path": self.pose_schema_path,
        }


def select_uniform_frame_indices(
    total_frames: int,
    num_frames: int,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> List[int]:
    """Select ``num_frames`` evenly spaced frame indices in ``[start_frame, end_frame]``."""
    n_total = max(0, int(total_frames))
    if n_total == 0:
        return []
    lo = max(0, min(int(start_frame), n_total - 1))
    hi = n_total - 1 if end_frame is None else max(lo, min(int(end_frame), n_total - 1))
    span = hi - lo + 1
    if num_frames == -1 or num_frames >= span:
        return list(range(lo, hi + 1))
    k = int(num_frames)
    if k <= 0:
        return []
    if k == 1:
        return [(lo + hi) // 2]
    indices = [int(round(lo + i * (hi - lo) / float(k - 1))) for i in range(k)]
    # Deduplicate while preserving order.
    seen: set[int] = set()
    out: List[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _frame_visual_descriptor(
    frame_bgr: np.ndarray, *, resize_width: int = 48
) -> np.ndarray:
    """Compute a compact spatial appearance + edge descriptor for k-means diversity clustering."""
    h, w = frame_bgr.shape[:2]
    target_w = max(16, int(resize_width))
    target_h = max(16, int(round(h * (target_w / float(max(w, 1))))))
    small = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    feat = np.concatenate([lab.reshape(-1), (grad_mag * 1.5).reshape(-1)], axis=0)
    norm = float(np.linalg.norm(feat))
    if norm > 1e-8:
        feat = feat / norm
    return feat


def _frame_quality_metrics(frame_bgr: np.ndarray) -> Tuple[float, float]:
    """Return ``(sharpness_laplacian_var, mean_luma_0_255)`` on a downsampled thumbnail."""
    h, w = frame_bgr.shape[:2]
    scale_w = min(w, 160)
    if scale_w < w and w > 0:
        scale_h = max(16, int(round(h * (scale_w / float(w)))))
        thumb = cv2.resize(frame_bgr, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
    else:
        thumb = frame_bgr
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(lap.var())
    mean_luma = float(np.mean(gray))
    return sharpness, mean_luma


def _compute_foreground_visual_features(
    thumbnails_bgr: Sequence[np.ndarray],
    *,
    foreground_weight: float = 1.5,
    appearance_weight: float = 0.85,
    edge_weight: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute foreground-aware spatial LAB + edge descriptors and motion scores.

    Estimates a static median background across up to 40 evenly-spaced thumbnails
    so fixed-camera arena setups focus diversity on the moving animal(s) rather
    than static background pixels.
    """
    n = len(thumbnails_bgr)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    labs: List[np.ndarray] = []
    grays: List[np.ndarray] = []
    for thumb in thumbnails_bgr:
        labs.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0)
        grays.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)

    # Estimate static background from up to 40 evenly-spaced samples.
    bg_indices = select_uniform_frame_indices(n, min(n, 40))
    bg_stack = np.stack([labs[i] for i in bg_indices], axis=0)
    bg_lab = np.median(bg_stack, axis=0).astype(np.float32, copy=False)

    motion_scores = np.zeros((n,), dtype=np.float32)
    descriptors: List[np.ndarray] = []
    w_fg = max(0.0, float(foreground_weight))
    w_app = max(0.0, float(appearance_weight))
    w_edge = max(0.0, float(edge_weight))

    for i in range(n):
        lab = labs[i]
        gray = grays[i]
        if i > 0:
            motion_scores[i] = float(np.mean(np.abs(gray - grays[i - 1])))

        fg_diff = np.linalg.norm(lab - bg_lab, axis=2).reshape(-1)
        fg_norm = float(np.linalg.norm(fg_diff))
        if fg_norm > 1e-6:
            fg_diff = (fg_diff / fg_norm) * w_fg
        else:
            fg_diff = np.zeros_like(fg_diff)

        app_vec = lab.reshape(-1)
        app_norm = float(np.linalg.norm(app_vec))
        if app_norm > 1e-6:
            app_vec = (app_vec / app_norm) * w_app

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_vec = np.sqrt(gx * gx + gy * gy).reshape(-1)
        grad_norm = float(np.linalg.norm(grad_vec))
        if grad_norm > 1e-6:
            grad_vec = (grad_vec / grad_norm) * w_edge

        feat = np.concatenate([fg_diff, app_vec, grad_vec], axis=0).astype(
            np.float32, copy=False
        )
        total_norm = float(np.linalg.norm(feat))
        if total_norm > 1e-8:
            feat = feat / total_norm
        descriptors.append(feat)

    if n > 1:
        motion_scores[0] = motion_scores[1]
    return np.stack(descriptors, axis=0), motion_scores


def _normalize_feature_matrix(features: np.ndarray) -> np.ndarray:
    mat = np.asarray(features, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1) if mat.size > 0 else mat.reshape(0, 0)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    return (mat / safe_norms).astype(np.float32, copy=False)


def _kmeans_cluster_medoids(
    features: np.ndarray,
    k: int,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    min_frame_gap: int = 0,
    max_iter: int = 25,
    seed: int = 42,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Deterministic k-means++ returning ``(medoid_rows, labels, centroids)``."""
    n_samples = int(features.shape[0])
    if n_samples == 0 or k <= 0:
        return [], np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
    k_eff = min(int(k), n_samples)
    if k_eff >= n_samples:
        return (
            list(range(n_samples)),
            np.arange(n_samples, dtype=np.int32),
            features.astype(np.float32, copy=True),
        )

    rng = np.random.default_rng(int(seed))
    centroids = np.empty((k_eff, features.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = features[first_idx]
    min_sq_dist = np.sum((features - centroids[0]) ** 2, axis=1)

    for c_idx in range(1, k_eff):
        total = float(np.sum(min_sq_dist))
        if not math.isfinite(total) or total <= 1e-12:
            next_idx = int(rng.integers(0, n_samples))
        else:
            probs = np.clip(min_sq_dist, 0.0, None)
            psum = float(np.sum(probs))
            if not math.isfinite(psum) or psum <= 1e-12:
                next_idx = int(rng.integers(0, n_samples))
            else:
                next_idx = int(rng.choice(n_samples, p=probs / psum))
        centroids[c_idx] = features[next_idx]
        d_new = np.sum((features - centroids[c_idx]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, d_new)

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max(1, int(max_iter))):
        new_labels = np.empty(n_samples, dtype=np.int32)
        chunk = 256
        c_sq = np.sum(centroids * centroids, axis=1, keepdims=True).T
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            f_chunk = features[start:end]
            f_sq = np.sum(f_chunk * f_chunk, axis=1, keepdims=True)
            dists = f_sq + c_sq - 2.0 * (f_chunk @ centroids.T)
            new_labels[start:end] = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k_eff):
            members = np.where(labels == c_idx)[0]
            if members.size > 0:
                centroids[c_idx] = np.mean(features[members], axis=0)
            else:
                assigned_centroids = centroids[labels]
                err = np.sum((features - assigned_centroids) ** 2, axis=1)
                farthest = int(np.argmax(err))
                centroids[c_idx] = features[farthest]
                labels[farthest] = c_idx

    gap = max(0, int(min_frame_gap))
    fidx_arr = (
        np.asarray(frame_indices, dtype=np.int64)
        if frame_indices is not None
        else np.arange(n_samples, dtype=np.int64)
    )

    chosen: List[int] = []
    chosen_set: set[int] = set()
    chosen_frames: List[int] = []

    # Process larger clusters first so major modes claim their best medoids.
    cluster_order = sorted(
        range(k_eff),
        key=lambda c: -int(np.sum(labels == c)),
    )
    for c_idx in cluster_order:
        members = np.where(labels == c_idx)[0]
        if members.size == 0:
            continue
        dists = np.sum((features[members] - centroids[c_idx]) ** 2, axis=1)
        ranked = members[np.argsort(dists)]
        picked: Optional[int] = None
        for row in ranked.tolist():
            if row in chosen_set:
                continue
            if gap > 0 and any(
                abs(int(fidx_arr[row]) - prev_f) < gap for prev_f in chosen_frames
            ):
                continue
            picked = int(row)
            break
        if picked is None:
            for row in ranked.tolist():
                if row not in chosen_set:
                    picked = int(row)
                    break
        if picked is not None:
            chosen.append(picked)
            chosen_set.add(picked)
            chosen_frames.append(int(fidx_arr[picked]))

    if len(chosen) < k_eff:
        for idx in range(n_samples):
            if idx in chosen_set:
                continue
            if gap > 0 and any(
                abs(int(fidx_arr[idx]) - prev_f) < gap for prev_f in chosen_frames
            ):
                continue
            chosen.append(idx)
            chosen_set.add(idx)
            chosen_frames.append(int(fidx_arr[idx]))
            if len(chosen) >= k_eff:
                break

    if len(chosen) < k_eff:
        for idx in range(n_samples):
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
                if len(chosen) >= k_eff:
                    break

    return sorted(chosen), labels, centroids


def _kmeans_medoid_indices(
    features: np.ndarray,
    k: int,
    *,
    max_iter: int = 25,
    seed: int = 42,
) -> List[int]:
    """Deterministic pure-NumPy k-means++ clustering returning closest-to-centroid medoid rows."""
    chosen, _labels, _centroids = _kmeans_cluster_medoids(
        features, k, max_iter=max_iter, seed=seed
    )
    return chosen


def select_farthest_point_indices(
    features: np.ndarray,
    k: int,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    seed_features: Optional[np.ndarray] = None,
    already_selected_rows: Sequence[int] = (),
    min_frame_gap: int = 0,
    relevance_scores: Optional[np.ndarray] = None,
    relevance_weight: float = 0.0,
) -> List[int]:
    """Select ``k`` rows via deterministic cosine Farthest-Point (k-center greedy) coreset.

    When ``seed_features`` (e.g., embeddings of already-labeled frames) or
    ``already_selected_rows`` are provided, each next pick maximizes minimum
    cosine distance to the already-covered set while respecting ``min_frame_gap``.
    """
    feat = _normalize_feature_matrix(features)
    n_samples = int(feat.shape[0])
    target = max(0, int(k))
    if n_samples == 0 or target == 0:
        return []

    gap = max(0, int(min_frame_gap))
    fidx_arr = (
        np.asarray(frame_indices, dtype=np.int64)
        if frame_indices is not None
        else np.arange(n_samples, dtype=np.int64)
    )
    excluded_rows = {int(r) for r in already_selected_rows if 0 <= int(r) < n_samples}
    covered_frames: List[int] = [int(fidx_arr[r]) for r in excluded_rows]

    rel_norm: Optional[np.ndarray] = None
    rel_w = float(np.clip(relevance_weight, 0.0, 0.85))
    if relevance_scores is not None and rel_w > 0.0:
        raw_rel = np.asarray(relevance_scores, dtype=np.float32).reshape(-1)
        if raw_rel.shape[0] == n_samples:
            rmin = float(np.min(raw_rel))
            rmax = float(np.max(raw_rel))
            if rmax - rmin > 1e-6:
                rel_norm = (raw_rel - rmin) / (rmax - rmin)
            else:
                rel_norm = np.zeros((n_samples,), dtype=np.float32)

    # Initialize minimum cosine distance to already-covered embeddings.
    min_dist = np.full((n_samples,), np.inf, dtype=np.float32)
    seed_blocks: List[np.ndarray] = []
    if seed_features is not None:
        sf = _normalize_feature_matrix(seed_features)
        if sf.shape[0] > 0 and sf.shape[1] == feat.shape[1]:
            seed_blocks.append(sf)
    if excluded_rows:
        seed_blocks.append(feat[sorted(excluded_rows)])

    if seed_blocks:
        all_seeds = np.concatenate(seed_blocks, axis=0)
        chunk = 512
        for start in range(0, all_seeds.shape[0], chunk):
            s_chunk = all_seeds[start : start + chunk]
            dists = 1.0 - (feat @ s_chunk.T)
            min_dist = np.minimum(min_dist, np.min(dists, axis=1))
    else:
        # Pick a clean initial anchor closest to the global centroid (or highest relevance).
        if rel_norm is not None:
            first_row = int(np.argmax(rel_norm))
        else:
            centroid = np.mean(feat, axis=0)
            c_norm = float(np.linalg.norm(centroid))
            if c_norm > 1e-8:
                centroid = centroid / c_norm
                first_row = int(np.argmax(feat @ centroid))
            else:
                first_row = 0
        min_dist = (1.0 - (feat @ feat[first_row])).astype(np.float32, copy=False)
        chosen: List[int] = [first_row]
        excluded_rows.add(first_row)
        covered_frames.append(int(fidx_arr[first_row]))
        if len(chosen) >= target:
            return chosen
        return chosen + _continue_farthest_greedy(
            feat=feat,
            fidx_arr=fidx_arr,
            min_dist=min_dist,
            excluded_rows=excluded_rows,
            covered_frames=covered_frames,
            remaining=target - len(chosen),
            gap=gap,
            rel_norm=rel_norm,
            rel_w=rel_w,
        )

    return _continue_farthest_greedy(
        feat=feat,
        fidx_arr=fidx_arr,
        min_dist=min_dist,
        excluded_rows=excluded_rows,
        covered_frames=covered_frames,
        remaining=target,
        gap=gap,
        rel_norm=rel_norm,
        rel_w=rel_w,
    )


def _continue_farthest_greedy(
    *,
    feat: np.ndarray,
    fidx_arr: np.ndarray,
    min_dist: np.ndarray,
    excluded_rows: set[int],
    covered_frames: List[int],
    remaining: int,
    gap: int,
    rel_norm: Optional[np.ndarray],
    rel_w: float,
) -> List[int]:
    n_samples = int(feat.shape[0])
    chosen: List[int] = []
    # Track temporal-gap mask dynamically.
    gap_blocked = np.zeros((n_samples,), dtype=bool)
    if gap > 0 and covered_frames:
        for cf in covered_frames:
            gap_blocked |= np.abs(fidx_arr - int(cf)) < gap

    taken = np.zeros((n_samples,), dtype=bool)
    for r in excluded_rows:
        taken[r] = True

    for _ in range(max(0, int(remaining))):
        eligible = (~taken) & (~gap_blocked)
        if not np.any(eligible):
            eligible = ~taken
        if not np.any(eligible):
            break

        cand_dists = np.where(eligible, min_dist, -np.inf)
        if rel_norm is not None and rel_w > 0.0:
            elig_vals = min_dist[eligible]
            d_max = float(np.max(elig_vals)) if elig_vals.size > 0 else 1.0
            d_scale = d_max if d_max > 1e-6 else 1.0
            combined = (1.0 - rel_w) * (min_dist / d_scale) + rel_w * rel_norm
            scores = np.where(eligible, combined, -np.inf)
            best_row = int(np.argmax(scores))
        else:
            best_row = int(np.argmax(cand_dists))

        chosen.append(best_row)
        taken[best_row] = True
        if gap > 0:
            gap_blocked |= np.abs(fidx_arr - int(fidx_arr[best_row])) < gap
        new_dists = 1.0 - (feat @ feat[best_row])
        min_dist = np.minimum(min_dist, new_dists)

    return chosen


def select_hybrid_frame_indices(
    features: np.ndarray,
    k: int,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    seed_features: Optional[np.ndarray] = None,
    min_frame_gap: int = 0,
    representative_ratio: float = 0.65,
    include_start_boundary: bool = False,
    relevance_scores: Optional[np.ndarray] = None,
    relevance_weight: float = 0.0,
    max_iter: int = 25,
    seed: int = 42,
) -> Tuple[List[int], Dict[int, str], np.ndarray]:
    """Combine K-Means++ medoids (representative coverage) + Farthest-Point coreset (rare states).

    Returns ``(chosen_rows_sorted, role_by_row, cluster_labels)``.
    """
    feat = _normalize_feature_matrix(features)
    n_samples = int(feat.shape[0])
    target = max(0, int(k))
    if n_samples == 0 or target == 0:
        return [], {}, np.zeros((0,), dtype=np.int32)

    if target >= n_samples:
        roles = {i: "representative" for i in range(n_samples)}
        return list(range(n_samples)), roles, np.arange(n_samples, dtype=np.int32)

    role_by_row: Dict[int, str] = {}
    chosen: List[int] = []
    chosen_set: set[int] = set()

    if include_start_boundary and n_samples > 0:
        chosen.append(0)
        chosen_set.add(0)
        role_by_row[0] = "boundary"

    remaining_total = max(0, target - len(chosen))
    ratio = float(np.clip(representative_ratio, 0.0, 1.0))
    k_rep = int(round(remaining_total * ratio))
    if remaining_total >= 2 and ratio > 0.0:
        k_rep = max(1, min(k_rep, remaining_total))
    k_div = max(0, remaining_total - k_rep)

    cluster_labels = np.zeros((n_samples,), dtype=np.int32)
    if k_rep > 0:
        medoid_rows, cluster_labels, _centroids = _kmeans_cluster_medoids(
            feat,
            k_rep,
            frame_indices=frame_indices,
            min_frame_gap=min_frame_gap,
            max_iter=max_iter,
            seed=seed,
        )
        for row in medoid_rows:
            if row not in chosen_set:
                chosen.append(row)
                chosen_set.add(row)
                role_by_row[row] = "representative"

    still_needed = max(0, target - len(chosen))
    if still_needed > 0 or k_div > 0:
        extra_rows = select_farthest_point_indices(
            feat,
            max(still_needed, k_div),
            frame_indices=frame_indices,
            seed_features=seed_features,
            already_selected_rows=chosen,
            min_frame_gap=min_frame_gap,
            relevance_scores=relevance_scores,
            relevance_weight=relevance_weight,
        )
        for row in extra_rows:
            if len(chosen) >= target:
                break
            if row not in chosen_set:
                chosen.append(row)
                chosen_set.add(row)
                role_by_row[row] = "diverse_extreme"

    return sorted(chosen), role_by_row, cluster_labels


def select_flow_frame_indices(
    video_path: str | Path,
    num_frames: int,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_candidates: int = 1200,
    min_frame_gap: int = 5,
    resize_width: int = 160,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """Select top motion-energy frames via bounded-memory Farneback optical flow.

    Returns ``(selected_frame_indices, motion_score_by_frame)`` without holding
    full-resolution video frames in RAM.
    """
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            return [], {}
        lo = max(0, min(int(start_frame), total_frames - 1))
        hi = (
            total_frames - 1
            if end_frame is None
            else max(lo, min(int(end_frame), total_frames - 1))
        )
        span = hi - lo + 1
        if num_frames == -1 or num_frames >= span:
            return list(range(lo, hi + 1)), {}
        k = max(0, int(num_frames))
        if k == 0:
            return [], {}

        cand_count = min(span, max(int(max_candidates), k * 4))
        candidate_indices = select_uniform_frame_indices(
            total_frames, cand_count, start_frame=lo, end_frame=hi
        )

        scored: List[Tuple[float, int]] = []
        motion_by_frame: Dict[int, float] = {}
        prev_gray: Optional[np.ndarray] = None
        next_expected: Optional[int] = 0
        grab_skip_disabled = False
        total_cand = max(1, len(candidate_indices))
        target_w = max(48, int(resize_width))

        for i, fidx in enumerate(candidate_indices):
            if is_stopped is not None and is_stopped():
                break
            ret, frame, next_expected, grab_skip_disabled = read_capture_frame_bgr(
                cap,
                fidx,
                expected_next=next_expected,
                max_forward_grab_skip=CV2Video._MAX_FORWARD_GRAB_SKIP,
                grab_skip_disabled=grab_skip_disabled,
            )
            if not ret or frame is None:
                continue
            h, w = frame.shape[:2]
            target_h = max(32, int(round(h * (target_w / float(max(w, 1))))))
            small = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 13, 2, 5, 1.1, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                score = float(np.mean(mag))
            else:
                score = 0.0
            prev_gray = gray
            scored.append((score, int(fidx)))
            motion_by_frame[int(fidx)] = score
            if progress_callback is not None and (i % 20 == 0 or i == total_cand - 1):
                pct = 80.0 * (i + 1) / float(total_cand)
                progress_callback(
                    pct,
                    f"Computing optical flow motion ({i + 1}/{total_cand} frames)...",
                )

        if not scored:
            return [], {}
        scored.sort(key=lambda item: (-item[0], item[1]))
        gap = max(0, int(min_frame_gap))
        selected: List[int] = []
        for _score, fidx in scored:
            if gap > 0 and any(abs(fidx - prev) < gap for prev in selected):
                continue
            selected.append(fidx)
            if len(selected) >= k:
                break
        if len(selected) < k:
            selected_set = set(selected)
            for _score, fidx in scored:
                if fidx not in selected_set:
                    selected.append(fidx)
                    selected_set.add(fidx)
                    if len(selected) >= k:
                        break
        return sorted(selected), motion_by_frame
    finally:
        cap.release()


def select_keyframe_indices(
    video_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Return video I-frame (keyframe) indices within ``[start_frame, end_frame]``."""
    from annolid.core.media.video import get_keyframe_timestamps

    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if total_frames <= 0:
        return []
    lo = max(0, min(int(start_frame), total_frames - 1))
    hi = (
        total_frames - 1
        if end_frame is None
        else max(lo, min(int(end_frame), total_frames - 1))
    )

    indices: List[int] = []
    try:
        times = get_keyframe_timestamps(str(vpath))
    except Exception:
        times = []

    if times and fps > 0.0:
        seen: set[int] = set()
        for ts in times:
            fidx = int(round(float(ts) * fps))
            if lo <= fidx <= hi and fidx not in seen:
                seen.add(fidx)
                indices.append(fidx)
        indices.sort()
    else:
        step = max(int(round(fps)) if fps > 0.0 else 30, 1)
        indices = list(range(lo, hi + 1, step))
        if indices and indices[-1] != hi:
            indices.append(hi)

    if (
        max_frames is not None
        and int(max_frames) > 0
        and len(indices) > int(max_frames)
    ):
        sub_pos = select_uniform_frame_indices(len(indices), int(max_frames))
        indices = [indices[p] for p in sub_pos]
    return indices


_IGNORED_JSON_NAMES = frozenset(
    {
        "pose_schema.json",
        "project_schema.json",
        "frame_selection_queue.json",
        "frame_selection_manifest.json",
        "outlier_frames_manifest.json",
    }
)


def discover_labeled_frame_indices(
    annotation_dir: Optional[str | Path],
    *,
    require_shapes: bool = True,
) -> set[int]:
    """Discover already-labeled frame indices across both per-frame JSONs and AnnotationStore."""
    if annotation_dir is None:
        return set()
    folder = Path(annotation_dir).expanduser()
    if not folder.is_dir():
        return set()

    from annolid.infrastructure import AnnotationStore, load_labelme_json

    labeled: set[int] = set()

    # 1. Per-frame disk LabelMe JSONs.
    for json_path in folder.glob("*.json"):
        name_lower = json_path.name.lower()
        if name_lower in _IGNORED_JSON_NAMES or name_lower.endswith(
            "_tracking_stats.json"
        ):
            continue
        fidx = AnnotationStore.frame_number_from_path(json_path)
        if fidx is None:
            continue
        if not require_shapes:
            labeled.add(int(fidx))
            continue
        try:
            payload = load_labelme_json(json_path)
            if isinstance(payload, dict) and payload.get("shapes"):
                labeled.add(int(fidx))
        except Exception:
            continue

    # 2. NDJSON AnnotationStore (`*_annotations.ndjson`).
    try:
        stub = folder / f"{folder.name}_000000000.json"
        store = AnnotationStore.for_frame_path(stub)
        if store.store_path.is_file():
            for fidx in store.iter_frames():
                if not require_shapes:
                    labeled.add(int(fidx))
                    continue
                record = store.get_frame(int(fidx))
                if isinstance(record, dict) and record.get("shapes"):
                    labeled.add(int(fidx))
    except Exception:
        pass

    return labeled


def select_kmeans_frame_indices(
    video_path: str | Path,
    num_frames: int,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_candidates: int = 1200,
    resize_width: int = 48,
    max_iter: int = 25,
    seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[int]:
    """Select visually diverse frames from a video via k-means++ clustering."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            return []
        lo = max(0, min(int(start_frame), total_frames - 1))
        hi = (
            total_frames - 1
            if end_frame is None
            else max(lo, min(int(end_frame), total_frames - 1))
        )
        span = hi - lo + 1
        if num_frames == -1 or num_frames >= span:
            return list(range(lo, hi + 1))
        k = max(0, int(num_frames))
        if k == 0:
            return []
        if k == 1:
            return [(lo + hi) // 2]

        cand_count = min(span, max(int(max_candidates), k * 4))
        candidate_indices = select_uniform_frame_indices(
            total_frames, cand_count, start_frame=lo, end_frame=hi
        )

        descriptors: List[np.ndarray] = []
        valid_indices: List[int] = []
        next_expected: Optional[int] = 0
        grab_skip_disabled = False
        total_cand = max(1, len(candidate_indices))
        for i, fidx in enumerate(candidate_indices):
            ret, frame, next_expected, grab_skip_disabled = read_capture_frame_bgr(
                cap,
                fidx,
                expected_next=next_expected,
                max_forward_grab_skip=CV2Video._MAX_FORWARD_GRAB_SKIP,
                grab_skip_disabled=grab_skip_disabled,
            )
            if not ret or frame is None:
                continue
            descriptors.append(
                _frame_visual_descriptor(frame, resize_width=resize_width)
            )
            valid_indices.append(int(fidx))
            if progress_callback is not None and (i % 20 == 0 or i == total_cand - 1):
                pct = 75.0 * (i + 1) / float(total_cand)
                progress_callback(
                    pct,
                    f"Extracting visual features ({i + 1}/{total_cand} candidate frames)...",
                )

        if not descriptors:
            return []
        if len(valid_indices) <= k:
            return sorted(valid_indices)

        if progress_callback is not None:
            progress_callback(
                82.0,
                f"Clustering {len(valid_indices)} frames into {k} diverse poses...",
            )

        feat_matrix = np.stack(descriptors, axis=0)
        medoid_rows = _kmeans_medoid_indices(
            feat_matrix, k, max_iter=max_iter, seed=seed
        )
        return sorted(valid_indices[r] for r in medoid_rows)
    finally:
        cap.release()


def _resolve_frame_bounds(
    cap: cv2.VideoCapture,
    *,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
) -> Tuple[int, int, int]:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        return 0, 0, 0
    lo = 0
    hi = total_frames - 1
    if fps > 0.0:
        if start_seconds is not None:
            lo = max(0, min(int(round(float(start_seconds) * fps)), total_frames - 1))
        if end_seconds is not None:
            hi = max(lo, min(int(round(float(end_seconds) * fps)), total_frames - 1))
    return total_frames, lo, hi


def _write_selected_frames(
    video_path: Path,
    output_dir: Path,
    indices: Sequence[int],
    *,
    image_format: str = "png",
    filename_prefix: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    progress_start: float = 0.0,
) -> Tuple[List[int], List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (filename_prefix or video_path.stem).replace(" ", "_")
    ext = str(image_format or "png").lstrip(".").lower() or "png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    saved_indices: List[int] = []
    saved_paths: List[str] = []
    try:
        next_expected: Optional[int] = 0
        grab_skip_disabled = False
        sorted_indices = sorted({int(i) for i in indices if int(i) >= 0})
        total = max(1, len(sorted_indices))
        for i, idx in enumerate(sorted_indices):
            ret, frame, next_expected, grab_skip_disabled = read_capture_frame_bgr(
                cap,
                idx,
                expected_next=next_expected,
                max_forward_grab_skip=CV2Video._MAX_FORWARD_GRAB_SKIP,
                grab_skip_disabled=grab_skip_disabled,
            )
            if not ret or frame is None:
                continue
            out_path = output_dir / f"{prefix}_{idx:09d}.{ext}"
            if cv2.imwrite(str(out_path), frame):
                saved_indices.append(idx)
                saved_paths.append(str(out_path))
            if progress_callback is not None:
                pct = progress_start + (100.0 - progress_start) * (i + 1) / float(total)
                progress_callback(pct, f"Saved frame {idx} ({i + 1}/{total})")
    finally:
        cap.release()
    return saved_indices, saved_paths


def extract_frames_smart(
    video_path: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    algo: str = "kmeans",
    num_frames: int = 50,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    image_format: str = "png",
    max_candidates: int = 1200,
    seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> FrameExtractionResult:
    """Extract training frames using ``hybrid``, ``kmeans``, ``farthest``, ``uniform``, ``random``, ``flow``, or ``keyframes``."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    algo_norm = str(algo or "kmeans").strip().lower()
    supported = {
        "hybrid",
        "kmeans",
        "farthest",
        "uniform",
        "random",
        "flow",
        "keyframes",
    }
    if algo_norm not in supported:
        raise ValueError(
            f"Unsupported frame extraction algorithm {algo!r}. "
            f"Choose one of: {', '.join(sorted(supported))}."
        )

    out_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else vpath.parent / f"{vpath.stem.replace(' ', '_')}_{algo_norm}"
    )

    if algo_norm in {"hybrid", "farthest"}:
        from annolid.services.frame_selection import (
            FrameSelectionConfig,
            select_frames_for_labeling,
        )

        sel = select_frames_for_labeling(
            vpath,
            config=FrameSelectionConfig(
                task="general",
                algo=algo_norm,
                num_frames=int(num_frames),
                feature_backend="visual_fg",
                extract_images=True,
                output_dir=str(out_root),
                image_format=str(image_format or "png"),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                max_candidates=int(max_candidates),
                exclude_labeled=False,
                seed_from_labeled=False,
                filter_blur=False,
                include_boundaries=False,
                seed=int(seed),
            ),
            progress_callback=progress_callback,
        )
        return FrameExtractionResult(
            video_path=sel.video_path,
            output_dir=str(sel.output_dir or out_root),
            algo=algo_norm,
            total_video_frames=sel.total_video_frames,
            selected_indices=sel.selected_indices,
            saved_paths=sel.saved_paths,
        )

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")
    try:
        total_frames, lo, hi = _resolve_frame_bounds(
            cap, start_seconds=start_seconds, end_seconds=end_seconds
        )
    finally:
        cap.release()

    if total_frames <= 0:
        return FrameExtractionResult(
            video_path=str(vpath),
            output_dir=str(out_root),
            algo=algo_norm,
            total_video_frames=0,
            selected_indices=(),
            saved_paths=(),
        )

    if algo_norm == "uniform":
        indices = select_uniform_frame_indices(
            total_frames, num_frames, start_frame=lo, end_frame=hi
        )
        saved_idx, saved_paths = _write_selected_frames(
            vpath,
            out_root,
            indices,
            image_format=image_format,
            progress_callback=progress_callback,
            progress_start=10.0,
        )
        return FrameExtractionResult(
            video_path=str(vpath),
            output_dir=str(out_root),
            algo=algo_norm,
            total_video_frames=total_frames,
            selected_indices=tuple(saved_idx),
            saved_paths=tuple(saved_paths),
        )

    if algo_norm == "kmeans":
        indices = select_kmeans_frame_indices(
            vpath,
            num_frames,
            start_frame=lo,
            end_frame=hi,
            max_candidates=max_candidates,
            seed=seed,
            progress_callback=progress_callback,
        )
        saved_idx, saved_paths = _write_selected_frames(
            vpath,
            out_root,
            indices,
            image_format=image_format,
            progress_callback=progress_callback,
            progress_start=85.0,
        )
        return FrameExtractionResult(
            video_path=str(vpath),
            output_dir=str(out_root),
            algo=algo_norm,
            total_video_frames=total_frames,
            selected_indices=tuple(saved_idx),
            saved_paths=tuple(saved_paths),
        )

    if algo_norm == "random":
        rng = np.random.default_rng(int(seed))
        span_indices = list(range(lo, hi + 1))
        if num_frames == -1 or num_frames >= len(span_indices):
            indices = span_indices
        else:
            k = max(0, int(num_frames))
            indices = (
                sorted(int(v) for v in rng.choice(span_indices, size=k, replace=False))
                if k > 0
                else []
            )
        saved_idx, saved_paths = _write_selected_frames(
            vpath,
            out_root,
            indices,
            image_format=image_format,
            progress_callback=progress_callback,
            progress_start=10.0,
        )
        return FrameExtractionResult(
            video_path=str(vpath),
            output_dir=str(out_root),
            algo=algo_norm,
            total_video_frames=total_frames,
            selected_indices=tuple(saved_idx),
            saved_paths=tuple(saved_paths),
        )

    if algo_norm == "flow":
        indices, _motion = select_flow_frame_indices(
            vpath,
            num_frames,
            start_frame=lo,
            end_frame=hi,
            max_candidates=max_candidates,
            progress_callback=progress_callback,
        )
        saved_idx, saved_paths = _write_selected_frames(
            vpath,
            out_root,
            indices,
            image_format=image_format,
            progress_callback=progress_callback,
            progress_start=85.0,
        )
        return FrameExtractionResult(
            video_path=str(vpath),
            output_dir=str(out_root),
            algo=algo_norm,
            total_video_frames=total_frames,
            selected_indices=tuple(saved_idx),
            saved_paths=tuple(saved_paths),
        )

    # keyframes
    max_kf = None if int(num_frames) == -1 else max(0, int(num_frames))
    indices = select_keyframe_indices(
        vpath, start_frame=lo, end_frame=hi, max_frames=max_kf
    )
    saved_idx, saved_paths = _write_selected_frames(
        vpath,
        out_root,
        indices,
        image_format=image_format,
        progress_callback=progress_callback,
        progress_start=20.0,
    )
    return FrameExtractionResult(
        video_path=str(vpath),
        output_dir=str(out_root),
        algo=algo_norm,
        total_video_frames=total_frames,
        selected_indices=tuple(saved_idx),
        saved_paths=tuple(saved_paths),
    )


def _resolve_pose_schema_for_dataset(
    dataset: PoseDataset,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    source_path: Optional[Path] = None,
) -> PoseSchema:
    if isinstance(pose_schema, PoseSchema):
        return pose_schema
    if pose_schema is not None:
        return PoseSchema.load(pose_schema)
    if source_path is not None:
        found = find_pose_schema_in_dir(source_path, search_parent=True)
        if found is not None:
            return PoseSchema.load(found)
    return dataset.to_pose_schema(preset="auto")


def score_pose_outlier_frames(
    dataset: PoseDataset,
    *,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    min_confidence: float = 0.6,
    jump_threshold_px: Optional[float] = None,
    jump_mad_multiplier: float = 5.0,
    max_edge_stretch_ratio: float = 1.8,
    min_edge_shrink_ratio: float = 0.35,
    expected_instances: Optional[int] = None,
) -> List[OutlierFrameScore]:
    """Score every frame in a ``PoseDataset`` for active-learning outlier refinement.

    Combines four complementary signals used by DeepLabCut and SLEAP:
    1. Low-confidence keypoints (``score < min_confidence``),
    2. Per-keypoint sudden displacement/jump spikes (fixed pixel or adaptive MAD threshold),
    3. Anatomical skeleton bone stretch or collapse vs. per-bone median length,
    4. Missing keypoints / incomplete skeletons or unexpected instance counts.
    """
    if not dataset.frames:
        return []

    schema = _resolve_pose_schema_for_dataset(dataset, pose_schema)
    expected_kps = list(schema.keypoints or dataset.resolved_keypoints())
    edges = list(schema.base_edges())
    sorted_frames = sorted(
        dataset.frames, key=lambda f: (int(f.video_index), int(f.frame_index))
    )

    # Pass 1: Collect per-(video, instance, keypoint) trajectories and per-edge lengths.
    kp_series: Dict[Tuple[int, str, str], List[Tuple[int, float, float]]] = defaultdict(
        list
    )
    edge_lengths: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for frame in sorted_frames:
        vid = int(frame.video_index)
        fidx = int(frame.frame_index)
        for inst_i, inst in enumerate(frame.instances):
            inst_key = str(inst.instance_name or f"instance_{inst_i}")
            for kp_name, pt in inst.keypoints.items():
                if pt.is_valid and int(pt.visibility) > 0:
                    kp_series[(vid, inst_key, kp_name)].append(
                        (fidx, float(pt.x), float(pt.y))
                    )
            for a, b in edges:
                pa = inst.keypoints.get(a)
                pb = inst.keypoints.get(b)
                if (
                    pa is not None
                    and pb is not None
                    and pa.is_valid
                    and pb.is_valid
                    and int(pa.visibility) > 0
                    and int(pb.visibility) > 0
                ):
                    edge_lengths[(a, b)].append(
                        math.hypot(float(pb.x) - float(pa.x), float(pb.y) - float(pa.y))
                    )

    median_edge_len: Dict[Tuple[str, str], float] = {
        edge: float(median(vals))
        for edge, vals in edge_lengths.items()
        if len(vals) >= 3 and float(median(vals)) > 1e-3
    }

    # Compute per-frame keypoint jumps.
    frame_jumps: Dict[Tuple[int, int], List[Tuple[str, float]]] = defaultdict(list)
    for (vid, inst_key, kp_name), pts in kp_series.items():
        if len(pts) < 2:
            continue
        speeds: List[Tuple[int, float]] = []
        for idx in range(1, len(pts)):
            f0, x0, y0 = pts[idx - 1]
            f1, x1, y1 = pts[idx]
            df = max(1, int(f1) - int(f0))
            step = math.hypot(x1 - x0, y1 - y0) / float(df)
            speeds.append((int(f1), float(step)))
        if not speeds:
            continue
        if jump_threshold_px is not None:
            thr = float(jump_threshold_px)
        else:
            vals = [s for _, s in speeds]
            med = float(median(vals))
            mad = float(median([abs(v - med) for v in vals]))
            # Robust floor of 15px so stationary segments do not flag sub-pixel jitter.
            thr = max(15.0, med + float(jump_mad_multiplier) * max(1.4826 * mad, 2.0))
        for f1, step in speeds:
            if step > thr:
                frame_jumps[(vid, f1)].append((f"{inst_key}:{kp_name}", step))

    # Infer expected instance count if not provided.
    exp_inst = expected_instances
    if exp_inst is None:
        counts = [len(f.instances) for f in sorted_frames if f.instances]
        if counts:
            exp_inst = int(round(median(counts)))

    scored: List[OutlierFrameScore] = []
    for frame in sorted_frames:
        vid = int(frame.video_index)
        fidx = int(frame.frame_index)
        low_conf_count = 0
        min_conf_seen: Optional[float] = None
        bone_outliers = 0
        missing_kps = 0
        reasons: List[str] = []

        for inst in frame.instances:
            present = 0
            for kp_name, pt in inst.keypoints.items():
                if not pt.is_valid or int(pt.visibility) <= 0:
                    continue
                present += 1
                if pt.score is not None and math.isfinite(float(pt.score)):
                    sc = float(pt.score)
                    if min_conf_seen is None or sc < min_conf_seen:
                        min_conf_seen = sc
                    if sc < float(min_confidence):
                        low_conf_count += 1
            if expected_kps:
                missing_kps += max(0, len(expected_kps) - present)

            for a, b in edges:
                med_len = median_edge_len.get((a, b))
                if med_len is None:
                    continue
                pa = inst.keypoints.get(a)
                pb = inst.keypoints.get(b)
                if (
                    pa is None
                    or pb is None
                    or not pa.is_valid
                    or not pb.is_valid
                    or int(pa.visibility) <= 0
                    or int(pb.visibility) <= 0
                ):
                    continue
                dist = math.hypot(float(pb.x) - float(pa.x), float(pb.y) - float(pa.y))
                ratio = dist / med_len
                if ratio >= float(max_edge_stretch_ratio) or ratio <= float(
                    min_edge_shrink_ratio
                ):
                    bone_outliers += 1

        jumps = frame_jumps.get((vid, fidx), [])
        jump_count = len(jumps)
        max_jump = max((d for _, d in jumps), default=0.0)

        inst_mismatch = 0
        if exp_inst is not None and len(frame.instances) != int(exp_inst):
            inst_mismatch = abs(len(frame.instances) - int(exp_inst))

        # Weighted active-learning priority score.
        score = (
            3.0 * float(bone_outliers)
            + 2.5 * float(jump_count)
            + 1.5 * float(low_conf_count)
            + 2.0 * float(inst_mismatch)
            + 0.75 * float(missing_kps)
        )
        if min_conf_seen is not None and min_conf_seen < float(min_confidence):
            score += float(min_confidence) - min_conf_seen
        if max_jump > 0.0:
            score += min(5.0, max_jump / 50.0)

        if bone_outliers > 0:
            reasons.append(f"bone_outliers={bone_outliers}")
        if jump_count > 0:
            reasons.append(f"jumps={jump_count} (max={max_jump:.1f}px)")
        if low_conf_count > 0:
            reasons.append(
                f"low_conf={low_conf_count} (min={min_conf_seen:.2f})"
                if min_conf_seen is not None
                else f"low_conf={low_conf_count}"
            )
        if missing_kps > 0:
            reasons.append(f"missing_kps={missing_kps}")
        if inst_mismatch > 0:
            reasons.append(
                f"instance_count={len(frame.instances)} (expected={exp_inst})"
            )

        scored.append(
            OutlierFrameScore(
                frame_index=fidx,
                video_index=vid,
                total_score=float(score),
                low_conf_count=low_conf_count,
                min_confidence=min_conf_seen,
                jump_count=jump_count,
                max_jump_px=float(max_jump),
                bone_outlier_count=bone_outliers,
                missing_keypoint_count=missing_kps,
                instance_count=len(frame.instances),
                reasons=tuple(reasons),
            )
        )

    scored.sort(
        key=lambda item: (-item.total_score, item.video_index, item.frame_index)
    )
    return scored


def select_diverse_outlier_frames(
    scored_frames: Sequence[OutlierFrameScore],
    *,
    num_frames: int = 20,
    min_frame_gap: int = 10,
    min_score: float = 0.01,
) -> List[OutlierFrameScore]:
    """Select up to ``num_frames`` highest-scoring outlier frames spaced by ``min_frame_gap``."""
    k = max(0, int(num_frames))
    if k == 0:
        return []
    gap = max(0, int(min_frame_gap))
    selected: List[OutlierFrameScore] = []
    for item in scored_frames:
        if item.total_score < float(min_score):
            continue
        if any(
            prev.video_index == item.video_index
            and abs(prev.frame_index - item.frame_index) < gap
            for prev in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= k:
            break
    selected.sort(key=lambda item: (item.video_index, item.frame_index))
    return selected


def extract_pose_outlier_frames(
    source: str | Path | PoseDataset,
    video_path: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    source_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    num_frames: int = 20,
    min_frame_gap: int = 10,
    min_confidence: float = 0.6,
    jump_threshold_px: Optional[float] = None,
    jump_mad_multiplier: float = 5.0,
    max_edge_stretch_ratio: float = 1.8,
    min_edge_shrink_ratio: float = 0.35,
    expected_instances: Optional[int] = None,
    include_predictions: bool = True,
    overwrite: bool = True,
    extract_images: bool = True,
) -> OutlierExtractionResult:
    """Select (and optionally extract) highest-risk pose outlier frames for active learning.

    When ``extract_images=False``, scores and selects the diverse outlier frames
    without writing PNG or LabelMe JSON copies to disk (for direct in-video
    labeling via the Label Queue).
    """
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")
    out_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None and str(output_dir).strip()
        else vpath.with_suffix("")
    )
    if extract_images:
        out_root.mkdir(parents=True, exist_ok=True)

    from annolid.annotation.pose_interop.converters import read_pose_dataset
    from annolid.annotation.pose_interop.labelme import write_pose_to_labelme_dir
    from annolid.annotation.pose_interop.model import PoseDataset, PoseFrame

    if isinstance(source, PoseDataset):
        dataset = source
        src_path_obj: Optional[Path] = None
        src_str = "<PoseDataset>"
    else:
        src_path_obj = Path(source).expanduser().resolve()
        src_str = str(src_path_obj)
        dataset = read_pose_dataset(
            src_path_obj,
            source_format=source_format,
            video_path=vpath,
            pose_schema=pose_schema,
        )

    schema = _resolve_pose_schema_for_dataset(dataset, pose_schema, src_path_obj)
    scored = score_pose_outlier_frames(
        dataset,
        pose_schema=schema,
        min_confidence=min_confidence,
        jump_threshold_px=jump_threshold_px,
        jump_mad_multiplier=jump_mad_multiplier,
        max_edge_stretch_ratio=max_edge_stretch_ratio,
        min_edge_shrink_ratio=min_edge_shrink_ratio,
        expected_instances=expected_instances,
    )
    outliers_found = sum(1 for s in scored if s.total_score >= 0.01)
    selected = select_diverse_outlier_frames(
        scored, num_frames=num_frames, min_frame_gap=min_frame_gap
    )

    # Fallback: if no anomalies triggered (e.g., clean video), pick lowest-confidence / highest-score frames.
    if not selected and scored:
        selected = select_diverse_outlier_frames(
            scored, num_frames=num_frames, min_frame_gap=min_frame_gap, min_score=0.0
        )

    if not extract_images:
        return OutlierExtractionResult(
            source_path=src_str,
            video_path=str(vpath),
            output_dir=str(out_root),
            frames_evaluated=len(scored),
            outliers_found=outliers_found,
            frames_extracted=0,
            selected_frames=tuple(selected),
            saved_images=(),
            manifest_path="",
            pose_schema_path=None,
        )

    selected_keys = {(item.video_index, item.frame_index) for item in selected}
    selected_indices = [item.frame_index for item in selected]

    saved_idx, saved_images = _write_selected_frames(
        vpath,
        out_root,
        selected_indices,
        image_format="png",
        filename_prefix=vpath.stem,
    )
    saved_img_by_idx = {
        idx: Path(path).name for idx, path in zip(saved_idx, saved_images)
    }

    frame_to_image: Dict[Tuple[int, int], str] = {}
    frame_to_wh: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cap = cv2.VideoCapture(str(vpath))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    subset_frames: List[PoseFrame] = []
    for frame in dataset.frames:
        key = (int(frame.video_index), int(frame.frame_index))
        if key not in selected_keys:
            continue
        img_name = saved_img_by_idx.get(int(frame.frame_index))
        if img_name:
            frame_to_image[key] = img_name
            frame_to_wh[key] = (
                int(frame.image_width or vid_w),
                int(frame.image_height or vid_h),
            )
        subset_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                instances=list(frame.instances) if include_predictions else [],
                image_path=img_name or frame.image_path,
                image_width=int(frame.image_width or vid_w) or None,
                image_height=int(frame.image_height or vid_h) or None,
                video_index=frame.video_index,
                video_name=vpath.stem,
            )
        )

    schema_out: Optional[str] = None
    if subset_frames:
        subset_ds = PoseDataset(
            frames=subset_frames,
            keypoints=list(schema.keypoints or dataset.resolved_keypoints()),
            edges=list(schema.base_edges()),
            symmetry_pairs=list(schema.base_symmetry_pairs()),
            instances=list(dataset.resolved_instances()),
            scorer=dataset.scorer,
            multi_animal=dataset.is_multi_animal(),
            source_format=dataset.source_format,
            video_paths={0: str(vpath)},
        )
        write_summary = write_pose_to_labelme_dir(
            subset_ds,
            out_root,
            overwrite=overwrite,
            write_pose_schema=True,
            filename_prefix=vpath.stem.replace(" ", "_"),
            frame_to_image=frame_to_image or None,
            frame_to_wh=frame_to_wh or None,
        )
        schema_out = write_summary.get("pose_schema")

    manifest_path = out_root / "outlier_frames_manifest.json"
    result = OutlierExtractionResult(
        source_path=src_str,
        video_path=str(vpath),
        output_dir=str(out_root),
        frames_evaluated=len(scored),
        outliers_found=outliers_found,
        frames_extracted=len(saved_images),
        selected_frames=tuple(selected),
        saved_images=tuple(saved_images),
        manifest_path=str(manifest_path),
        pose_schema_path=schema_out,
    )
    atomic_write_json(manifest_path, result.to_dict(), indent=2)
    return result
