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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from annolid.annotation.pose_interop.converters import read_pose_dataset
from annolid.annotation.pose_interop.labelme import write_pose_to_labelme_dir
from annolid.annotation.pose_interop.model import PoseDataset, PoseFrame
from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.core.media.video import CV2Video, read_capture_frame_bgr
from annolid.utils.atomic_io import atomic_write_json

__all__ = [
    "FrameExtractionResult",
    "OutlierFrameScore",
    "OutlierExtractionResult",
    "select_uniform_frame_indices",
    "select_kmeans_frame_indices",
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


def _kmeans_medoid_indices(
    features: np.ndarray,
    k: int,
    *,
    max_iter: int = 25,
    seed: int = 42,
) -> List[int]:
    """Deterministic pure-NumPy k-means++ clustering returning closest-to-centroid medoid rows."""
    n_samples = int(features.shape[0])
    if n_samples == 0 or k <= 0:
        return []
    if k >= n_samples:
        return list(range(n_samples))

    rng = np.random.default_rng(int(seed))
    # k-means++ initialization (deterministic via rng).
    centroids = np.empty((k, features.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = features[first_idx]
    min_sq_dist = np.sum((features - centroids[0]) ** 2, axis=1)

    for c_idx in range(1, k):
        total = float(np.sum(min_sq_dist))
        if not math.isfinite(total) or total <= 1e-12:
            next_idx = int(rng.integers(0, n_samples))
        else:
            probs = min_sq_dist / total
            next_idx = int(rng.choice(n_samples, p=probs))
        centroids[c_idx] = features[next_idx]
        d_new = np.sum((features - centroids[c_idx]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, d_new)

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max(1, int(max_iter))):
        # Compute squared Euclidean distances in chunks to cap peak memory.
        new_labels = np.empty(n_samples, dtype=np.int32)
        chunk = 256
        c_sq = np.sum(centroids * centroids, axis=1, keepdims=True).T  # (1, k)
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            f_chunk = features[start:end]
            f_sq = np.sum(f_chunk * f_chunk, axis=1, keepdims=True)
            dists = f_sq + c_sq - 2.0 * (f_chunk @ centroids.T)
            new_labels[start:end] = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            members = np.where(labels == c_idx)[0]
            if members.size > 0:
                centroids[c_idx] = np.mean(features[members], axis=0)
            else:
                # Re-seed empty cluster to farthest point from its current centroid.
                assigned_centroids = centroids[labels]
                err = np.sum((features - assigned_centroids) ** 2, axis=1)
                farthest = int(np.argmax(err))
                centroids[c_idx] = features[farthest]
                labels[farthest] = c_idx

    chosen: List[int] = []
    chosen_set: set[int] = set()
    for c_idx in range(k):
        members = np.where(labels == c_idx)[0]
        if members.size == 0:
            continue
        dists = np.sum((features[members] - centroids[c_idx]) ** 2, axis=1)
        best_local = int(members[int(np.argmin(dists))])
        if best_local not in chosen_set:
            chosen.append(best_local)
            chosen_set.add(best_local)

    # Top up with farthest remaining candidates if any cluster collapsed.
    if len(chosen) < k:
        for idx in range(n_samples):
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
                if len(chosen) >= k:
                    break
    return sorted(chosen)


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
    """Extract training frames using ``uniform``, ``kmeans``, ``random``, ``flow``, or ``keyframes``."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    algo_norm = str(algo or "kmeans").strip().lower()
    supported = {"uniform", "kmeans", "random", "flow", "keyframes"}
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

    # Delegate legacy "flow" and "keyframes" to annolid.data.videos.extract_frames.
    from annolid.data.videos import extract_frames as _legacy_extract_frames

    out_root.mkdir(parents=True, exist_ok=True)
    sub_clip = start_seconds is not None and end_seconds is not None
    # Legacy extract_frames appends _{algo} to out_dir/video_name; invoke directly when
    # out_dir was not explicitly customized, or collect its saved files cleanly.
    gen = _legacy_extract_frames(
        video_file=str(vpath),
        num_frames=num_frames,
        out_dir=str(out_root.parent),
        algo=algo_norm,
        sub_clip=sub_clip,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        prediction=False,
    )
    for pct, msg in gen:
        if progress_callback is not None:
            progress_callback(float(pct), str(msg))
    legacy_dir = out_root.parent / f"{vpath.stem.replace(' ', '_')}_{algo_norm}"
    actual_dir = legacy_dir if legacy_dir.is_dir() else out_root
    saved_files = sorted(str(p) for p in actual_dir.glob("*.png"))
    return FrameExtractionResult(
        video_path=str(vpath),
        output_dir=str(actual_dir),
        algo=algo_norm,
        total_video_frames=total_frames,
        selected_indices=(),
        saved_paths=tuple(saved_files),
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
    output_dir: str | Path,
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
) -> OutlierExtractionResult:
    """Extract highest-risk pose outlier frames + editable LabelMe JSONs for active learning."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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
