"""Offline pose trajectory filtering, gap interpolation, outlier cleaning, and forward identity swap.

Implements the postprocessing workflow familiar to DeepLabCut (``filterpredictions``
median/Savitzky-Golay filtering + ``p-cutoff`` masking) and SLEAP (bounded track
interpolation, smoothing, and swap-identities-from-frame-forward proofreading),
while strictly preserving LabelMe polygons, zones, flags, and visibility contracts.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    from scipy.signal import medfilt as _scipy_medfilt
    from scipy.signal import savgol_filter as _scipy_savgol_filter
except Exception:  # pragma: no cover - pure NumPy fallbacks provided below
    _scipy_medfilt = None  # type: ignore[assignment]
    _scipy_savgol_filter = None  # type: ignore[assignment]

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.converters import (
    normalize_pose_format,
    read_pose_dataset,
)
from annolid.annotation.pose_interop.deeplabcut import (
    write_dlc_analysis_csv,
    write_dlc_analysis_h5,
)
from annolid.annotation.pose_interop.labelme import (
    build_labelme_point_shape,
    write_pose_to_coco_json,
    write_pose_to_labelme_dir,
)
from annolid.annotation.pose_interop.model import (
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
    detect_pose_source_format,
)
from annolid.annotation.pose_interop.sleap import (
    write_sleap_analysis_h5,
    write_sleap_slp,
)
from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.utils.atomic_io import atomic_write_json
from annolid.utils.zone_shapes import is_zone_shape_payload

__all__ = [
    "PoseFilterConfig",
    "PoseFilterReport",
    "PoseSwapResult",
    "filter_pose_dataset",
    "filter_pose_source",
    "swap_pose_instances_from_frame",
]


@dataclass(frozen=True)
class PoseFilterConfig:
    """Configuration for offline pose cleaning, gap interpolation, and temporal smoothing."""

    min_confidence: Optional[float] = None
    max_jump_px: Optional[float] = None
    jump_mad_multiplier: Optional[float] = None
    enforce_bone_bounds: bool = False
    max_edge_stretch_ratio: float = 1.8
    min_edge_shrink_ratio: float = 0.35
    interpolate: str = "linear"  # "none" | "linear" | "cubic" | "nearest"
    max_gap_frames: int = 5
    method: str = (
        "savgol"  # "none" | "median" | "savgol" | "one_euro" | "moving_average"
    )
    window_length: int = 5
    polyorder: int = 2
    fps: float = 30.0
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.05
    one_euro_d_cutoff: float = 1.0

    def validated(self) -> "PoseFilterConfig":
        interp = str(self.interpolate or "none").strip().lower()
        if interp not in {"none", "linear", "cubic", "nearest"}:
            raise ValueError(
                f"Unsupported interpolation mode {self.interpolate!r}. "
                "Choose one of: none, linear, cubic, nearest."
            )
        method = str(self.method or "none").strip().lower()
        if method not in {"none", "median", "savgol", "one_euro", "moving_average"}:
            raise ValueError(
                f"Unsupported filter method {self.method!r}. "
                "Choose one of: none, median, savgol, one_euro, moving_average."
            )
        win = max(3, int(self.window_length))
        if win % 2 == 0:
            win += 1
        poly = max(1, min(int(self.polyorder), win - 1))
        return PoseFilterConfig(
            min_confidence=(
                float(self.min_confidence) if self.min_confidence is not None else None
            ),
            max_jump_px=(
                float(self.max_jump_px) if self.max_jump_px is not None else None
            ),
            jump_mad_multiplier=(
                float(self.jump_mad_multiplier)
                if self.jump_mad_multiplier is not None
                else None
            ),
            enforce_bone_bounds=bool(self.enforce_bone_bounds),
            max_edge_stretch_ratio=max(1.05, float(self.max_edge_stretch_ratio)),
            min_edge_shrink_ratio=max(
                0.05, min(0.95, float(self.min_edge_shrink_ratio))
            ),
            interpolate=interp,
            max_gap_frames=max(0, int(self.max_gap_frames)),
            method=method,
            window_length=win,
            polyorder=poly,
            fps=max(1e-3, float(self.fps)),
            one_euro_min_cutoff=max(1e-4, float(self.one_euro_min_cutoff)),
            one_euro_beta=max(0.0, float(self.one_euro_beta)),
            one_euro_d_cutoff=max(1e-4, float(self.one_euro_d_cutoff)),
        )


@dataclass(frozen=True)
class PoseFilterReport:
    """Audit summary returned after pose filtering and interpolation."""

    source_path: str
    output_path: Optional[str]
    method: str
    interpolate: str
    frames_processed: int
    instances_processed: int
    keypoints_total: int
    masked_low_confidence: int
    masked_jump_outliers: int
    masked_bone_outliers: int
    gaps_interpolated: int
    points_smoothed: int
    mean_displacement_px: float
    max_displacement_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "method": self.method,
            "interpolate": self.interpolate,
            "frames_processed": int(self.frames_processed),
            "instances_processed": int(self.instances_processed),
            "keypoints_total": int(self.keypoints_total),
            "masked_low_confidence": int(self.masked_low_confidence),
            "masked_jump_outliers": int(self.masked_jump_outliers),
            "masked_bone_outliers": int(self.masked_bone_outliers),
            "gaps_interpolated": int(self.gaps_interpolated),
            "points_smoothed": int(self.points_smoothed),
            "mean_displacement_px": round(float(self.mean_displacement_px), 4),
            "max_displacement_px": round(float(self.max_displacement_px), 4),
        }


@dataclass(frozen=True)
class PoseSwapResult:
    """Summary of forward pose-instance identity swapping across a LabelMe/AnnotationStore folder."""

    source_dir: str
    start_frame: int
    end_frame: Optional[int]
    instance_a: str
    instance_b: str
    dry_run: bool
    frames_scanned: int
    frames_modified: int
    shapes_swapped: int
    points_swapped: int
    polygons_swapped: int
    modified_frame_indices: Tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_dir": self.source_dir,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame) if self.end_frame is not None else None,
            "instance_a": self.instance_a,
            "instance_b": self.instance_b,
            "dry_run": bool(self.dry_run),
            "frames_scanned": int(self.frames_scanned),
            "frames_modified": int(self.frames_modified),
            "shapes_swapped": int(self.shapes_swapped),
            "points_swapped": int(self.points_swapped),
            "polygons_swapped": int(self.polygons_swapped),
            "modified_frame_indices": list(self.modified_frame_indices),
        }


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    n = values.size
    if n < 3:
        return values.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return values.copy()
    if _scipy_medfilt is not None:
        # Reflect-pad edges to avoid scipy's zero-padding boundary artifacts.
        half = w // 2
        padded = np.pad(values, (half, half), mode="edge")
        return np.asarray(_scipy_medfilt(padded, kernel_size=w), dtype=np.float64)[
            half : half + n
        ]
    half = w // 2
    padded = np.pad(values, (half, half), mode="edge")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(np.median(padded[i : i + w]))
    return out


def _savgol_filter_1d(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = values.size
    if n < 3:
        return values.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return values.copy()
    p = min(polyorder, w - 1)
    if _scipy_savgol_filter is not None:
        return np.asarray(
            _scipy_savgol_filter(values, window_length=w, polyorder=p, mode="interp"),
            dtype=np.float64,
        )
    # Pure NumPy local polynomial least-squares fallback.
    half = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, min(i - half, n - w))
        hi = lo + w
        xs = np.arange(lo, hi, dtype=np.float64) - float(i)
        coeffs = np.polyfit(xs, values[lo:hi], deg=p)
        out[i] = float(coeffs[-1])
    return out


def _moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    n = values.size
    if n < 2:
        return values.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 2:
        return values.copy()
    half = w // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(padded, kernel, mode="valid")[:n]


def _one_euro_forward_1d(
    values: np.ndarray,
    *,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    n = values.size
    if n < 2:
        return values.copy()
    dt = 1.0 / max(1e-3, float(fps))

    def _alpha(cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / dt)

    out = np.empty(n, dtype=np.float64)
    x_prev = float(values[0])
    dx_prev = 0.0
    out[0] = x_prev
    a_d = _alpha(d_cutoff)
    for i in range(1, n):
        x = float(values[i])
        dx = (x - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(dx_hat)
        a_x = _alpha(cutoff)
        x_hat = a_x * x + (1.0 - a_x) * x_prev
        out[i] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


def _one_euro_zero_phase_1d(
    values: np.ndarray,
    *,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    fwd = _one_euro_forward_1d(
        values, fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
    )
    bwd = _one_euro_forward_1d(
        values[::-1], fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
    )[::-1]
    return 0.5 * (fwd + bwd)


def _catmull_rom_eval(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Evaluate a centripetal-like uniform Catmull-Rom cubic spline on t in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _interpolate_bounded_gaps(
    frame_indices: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    valid: np.ndarray,
    *,
    mode: str,
    max_gap_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate NaN gaps whose frame span is <= ``max_gap_frames``."""
    out_x = xs.copy()
    out_y = ys.copy()
    out_valid = valid.copy()
    interpolated = np.zeros_like(valid, dtype=bool)

    if mode == "none" or max_gap_frames <= 0:
        return out_x, out_y, out_valid, interpolated

    valid_positions = np.where(valid)[0]
    if valid_positions.size < 2:
        return out_x, out_y, out_valid, interpolated

    for idx_pos in range(valid_positions.size - 1):
        i_left = int(valid_positions[idx_pos])
        i_right = int(valid_positions[idx_pos + 1])
        if i_right <= i_left + 1:
            continue
        f_left = int(frame_indices[i_left])
        f_right = int(frame_indices[i_right])
        gap_frames = (f_right - f_left) - 1
        if gap_frames <= 0 or gap_frames > max_gap_frames:
            continue

        # Prepare neighbor points for cubic Catmull-Rom when requested.
        i_prev = int(valid_positions[idx_pos - 1]) if idx_pos > 0 else i_left
        i_next = (
            int(valid_positions[idx_pos + 2])
            if idx_pos + 2 < valid_positions.size
            else i_right
        )

        for k in range(i_left + 1, i_right):
            t = float(frame_indices[k] - f_left) / float(f_right - f_left)
            if mode == "nearest":
                src = i_left if t < 0.5 else i_right
                out_x[k] = xs[src]
                out_y[k] = ys[src]
            elif mode == "cubic" and valid_positions.size >= 4:
                out_x[k] = _catmull_rom_eval(
                    float(xs[i_prev]),
                    float(xs[i_left]),
                    float(xs[i_right]),
                    float(xs[i_next]),
                    t,
                )
                out_y[k] = _catmull_rom_eval(
                    float(ys[i_prev]),
                    float(ys[i_left]),
                    float(ys[i_right]),
                    float(ys[i_next]),
                    t,
                )
            else:
                out_x[k] = float(xs[i_left]) + t * (
                    float(xs[i_right]) - float(xs[i_left])
                )
                out_y[k] = float(ys[i_left]) + t * (
                    float(ys[i_right]) - float(ys[i_left])
                )
            out_valid[k] = True
            interpolated[k] = True

    return out_x, out_y, out_valid, interpolated


def _smooth_contiguous_runs(
    xs: np.ndarray,
    ys: np.ndarray,
    valid: np.ndarray,
    cfg: PoseFilterConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply configured zero-phase smoother independently to each contiguous valid run."""
    if cfg.method == "none":
        return xs.copy(), ys.copy(), 0

    out_x = xs.copy()
    out_y = ys.copy()
    smoothed_count = 0
    n = valid.size
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        run_len = j - i
        if run_len >= 3:
            seg_x = xs[i:j]
            seg_y = ys[i:j]
            if cfg.method == "median":
                out_x[i:j] = _median_filter_1d(seg_x, cfg.window_length)
                out_y[i:j] = _median_filter_1d(seg_y, cfg.window_length)
            elif cfg.method == "savgol":
                out_x[i:j] = _savgol_filter_1d(seg_x, cfg.window_length, cfg.polyorder)
                out_y[i:j] = _savgol_filter_1d(seg_y, cfg.window_length, cfg.polyorder)
            elif cfg.method == "moving_average":
                out_x[i:j] = _moving_average_1d(seg_x, cfg.window_length)
                out_y[i:j] = _moving_average_1d(seg_y, cfg.window_length)
            elif cfg.method == "one_euro":
                out_x[i:j] = _one_euro_zero_phase_1d(
                    seg_x,
                    fps=cfg.fps,
                    min_cutoff=cfg.one_euro_min_cutoff,
                    beta=cfg.one_euro_beta,
                    d_cutoff=cfg.one_euro_d_cutoff,
                )
                out_y[i:j] = _one_euro_zero_phase_1d(
                    seg_y,
                    fps=cfg.fps,
                    min_cutoff=cfg.one_euro_min_cutoff,
                    beta=cfg.one_euro_beta,
                    d_cutoff=cfg.one_euro_d_cutoff,
                )
            smoothed_count += run_len
        i = j
    return out_x, out_y, smoothed_count


def filter_pose_dataset(
    dataset: PoseDataset,
    config: Optional[PoseFilterConfig] = None,
    *,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    source_label: str = "<PoseDataset>",
) -> Tuple[PoseDataset, PoseFilterReport]:
    """Apply confidence masking, jump/bone outlier cleaning, gap interpolation, and smoothing."""
    cfg = (config or PoseFilterConfig()).validated()

    if isinstance(pose_schema, PoseSchema):
        schema = pose_schema
    elif pose_schema is not None:
        schema = PoseSchema.load(pose_schema)
    else:
        schema = dataset.to_pose_schema(preset="auto")

    edges = list(schema.base_edges())
    all_kps = list(schema.keypoints or dataset.resolved_keypoints())
    if not dataset.frames:
        empty_report = PoseFilterReport(
            source_path=source_label,
            output_path=None,
            method=cfg.method,
            interpolate=cfg.interpolate,
            frames_processed=0,
            instances_processed=0,
            keypoints_total=0,
            masked_low_confidence=0,
            masked_jump_outliers=0,
            masked_bone_outliers=0,
            gaps_interpolated=0,
            points_smoothed=0,
            mean_displacement_px=0.0,
            max_displacement_px=0.0,
        )
        return copy.deepcopy(dataset), empty_report

    # Group frames by video_index and sort chronologically.
    by_video: Dict[int, List[PoseFrame]] = defaultdict(list)
    for frame in dataset.frames:
        by_video[int(frame.video_index)].append(frame)

    masked_low_conf = 0
    masked_jumps = 0
    masked_bones = 0
    gaps_interpolated = 0
    points_smoothed = 0
    keypoints_total = 0
    displacements: List[float] = []
    seen_instances: set[Tuple[int, str]] = set()

    new_frames: List[PoseFrame] = []

    for vid, vframes in sorted(by_video.items()):
        vframes_sorted = sorted(vframes, key=lambda f: int(f.frame_index))
        frame_indices = np.array(
            [int(f.frame_index) for f in vframes_sorted], dtype=np.int64
        )
        n_frames = len(vframes_sorted)

        # Build dense per-instance, per-keypoint arrays over this video's frame timeline.
        inst_names: List[str] = []
        inst_meta: Dict[str, Tuple[Optional[int], bool]] = {}
        inst_frame_map: Dict[str, Dict[int, PoseInstance]] = defaultdict(dict)
        for t_idx, frame in enumerate(vframes_sorted):
            for i_pos, inst in enumerate(frame.instances):
                name = (
                    str(inst.instance_name or f"instance_{i_pos}").strip()
                    or f"instance_{i_pos}"
                )
                if name not in inst_meta:
                    inst_names.append(name)
                    inst_meta[name] = (inst.track_id, inst.is_predicted)
                inst_frame_map[name][t_idx] = inst

        # Precompute per-edge median bone lengths across this video if bone bounds are enabled.
        median_edge_len: Dict[Tuple[str, str], float] = {}
        if cfg.enforce_bone_bounds and edges:
            edge_samples: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for frame in vframes_sorted:
                for inst in frame.instances:
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
                            edge_samples[(a, b)].append(
                                math.hypot(pb.x - pa.x, pb.y - pa.y)
                            )
            for edge, vals in edge_samples.items():
                if len(vals) >= 3:
                    m = float(median(vals))
                    if m > 1e-3:
                        median_edge_len[edge] = m

        # Process each instance independently.
        filtered_inst_frames: Dict[int, List[PoseInstance]] = defaultdict(list)
        for inst_name in inst_names:
            seen_instances.add((vid, inst_name))
            track_id, is_pred_default = inst_meta[inst_name]
            kps_for_inst = list(all_kps)
            for inst_obj in inst_frame_map[inst_name].values():
                for kp_name in inst_obj.keypoints:
                    if kp_name not in kps_for_inst:
                        kps_for_inst.append(kp_name)

            xs_by_kp: Dict[str, np.ndarray] = {}
            ys_by_kp: Dict[str, np.ndarray] = {}
            orig_xs_by_kp: Dict[str, np.ndarray] = {}
            orig_ys_by_kp: Dict[str, np.ndarray] = {}
            valid_by_kp: Dict[str, np.ndarray] = {}
            scores_by_kp: Dict[str, List[Optional[float]]] = {}
            vis_by_kp: Dict[str, np.ndarray] = {}
            interp_by_kp: Dict[str, np.ndarray] = {}

            for kp in kps_for_inst:
                xs = np.full(n_frames, np.nan, dtype=np.float64)
                ys = np.full(n_frames, np.nan, dtype=np.float64)
                valid = np.zeros(n_frames, dtype=bool)
                vis_arr = np.zeros(n_frames, dtype=np.int32)
                scores: List[Optional[float]] = [None] * n_frames

                for t_idx, inst_obj in inst_frame_map[inst_name].items():
                    pt = inst_obj.keypoints.get(kp)
                    if pt is None or not pt.is_valid or int(pt.visibility) <= 0:
                        continue
                    keypoints_total += 1
                    xs[t_idx] = float(pt.x)
                    ys[t_idx] = float(pt.y)
                    vis_arr[t_idx] = int(pt.visibility)
                    scores[t_idx] = pt.score

                    # Stage 1: Confidence (p-cutoff) masking.
                    if (
                        cfg.min_confidence is not None
                        and pt.score is not None
                        and math.isfinite(float(pt.score))
                        and float(pt.score) < float(cfg.min_confidence)
                    ):
                        masked_low_conf += 1
                        continue
                    valid[t_idx] = True

                orig_xs_by_kp[kp] = xs.copy()
                orig_ys_by_kp[kp] = ys.copy()

                # Stage 2: Jump / velocity outlier masking.
                if cfg.max_jump_px is not None or cfg.jump_mad_multiplier is not None:
                    v_pos = np.where(valid)[0]
                    if v_pos.size >= 3:
                        step_speeds: List[float] = []
                        for k_i in range(1, v_pos.size):
                            p0, p1 = int(v_pos[k_i - 1]), int(v_pos[k_i])
                            df = max(1, int(frame_indices[p1] - frame_indices[p0]))
                            step_speeds.append(
                                math.hypot(xs[p1] - xs[p0], ys[p1] - ys[p0]) / float(df)
                            )
                        thr = (
                            float(cfg.max_jump_px)
                            if cfg.max_jump_px is not None
                            else None
                        )
                        if cfg.jump_mad_multiplier is not None and step_speeds:
                            med_s = float(median(step_speeds))
                            mad_s = float(median([abs(s - med_s) for s in step_speeds]))
                            mad_thr = max(
                                15.0,
                                med_s
                                + float(cfg.jump_mad_multiplier)
                                * max(1.4826 * mad_s, 2.0),
                            )
                            thr = min(thr, mad_thr) if thr is not None else mad_thr
                        if thr is not None:
                            # Mark a point as an outlier if it jumps away from its local
                            # neighbors (isolated spike: both incoming & outgoing jumps,
                            # or departure from previous valid anchor).
                            to_mask: List[int] = []
                            for k_i in range(v_pos.size):
                                cur = int(v_pos[k_i])
                                prev_i = int(v_pos[k_i - 1]) if k_i > 0 else None
                                next_i = (
                                    int(v_pos[k_i + 1])
                                    if k_i + 1 < v_pos.size
                                    else None
                                )
                                if prev_i is not None and next_i is not None:
                                    # Compare against linear prediction between prev and next.
                                    span = max(
                                        1,
                                        int(
                                            frame_indices[next_i]
                                            - frame_indices[prev_i]
                                        ),
                                    )
                                    alpha = float(
                                        frame_indices[cur] - frame_indices[prev_i]
                                    ) / float(span)
                                    exp_x = xs[prev_i] + alpha * (
                                        xs[next_i] - xs[prev_i]
                                    )
                                    exp_y = ys[prev_i] + alpha * (
                                        ys[next_i] - ys[prev_i]
                                    )
                                    dev = math.hypot(xs[cur] - exp_x, ys[cur] - exp_y)
                                    if dev > thr:
                                        to_mask.append(cur)
                                elif prev_i is not None:
                                    df = max(
                                        1,
                                        int(frame_indices[cur] - frame_indices[prev_i]),
                                    )
                                    if (
                                        math.hypot(
                                            xs[cur] - xs[prev_i], ys[cur] - ys[prev_i]
                                        )
                                        / float(df)
                                        > thr
                                    ):
                                        to_mask.append(cur)
                            for bad_idx in to_mask:
                                valid[bad_idx] = False
                                masked_jumps += 1

                xs_by_kp[kp] = xs
                ys_by_kp[kp] = ys
                valid_by_kp[kp] = valid
                scores_by_kp[kp] = scores
                vis_by_kp[kp] = vis_arr

            # Stage 3: Anatomical bone-length outlier masking.
            if cfg.enforce_bone_bounds and median_edge_len:
                for t_idx in range(n_frames):
                    bad_votes: Dict[str, int] = defaultdict(int)
                    for a, b in edges:
                        med_len = median_edge_len.get((a, b))
                        if med_len is None:
                            continue
                        if (
                            a not in valid_by_kp
                            or b not in valid_by_kp
                            or not valid_by_kp[a][t_idx]
                            or not valid_by_kp[b][t_idx]
                        ):
                            continue
                        dist = math.hypot(
                            float(xs_by_kp[b][t_idx] - xs_by_kp[a][t_idx]),
                            float(ys_by_kp[b][t_idx] - ys_by_kp[a][t_idx]),
                        )
                        ratio = dist / med_len
                        if (
                            ratio >= cfg.max_edge_stretch_ratio
                            or ratio <= cfg.min_edge_shrink_ratio
                        ):
                            bad_votes[a] += 1
                            bad_votes[b] += 1
                    if bad_votes:
                        # Prefer masking the node implicated in more broken bones (or lower score).
                        max_vote = max(bad_votes.values())
                        candidates = [k for k, v in bad_votes.items() if v == max_vote]
                        if len(candidates) > 1:
                            candidates.sort(
                                key=lambda kp_n: (
                                    scores_by_kp[kp_n][t_idx]
                                    if scores_by_kp[kp_n][t_idx] is not None
                                    else 1.0
                                )
                            )
                            to_drop = [candidates[0]]
                        else:
                            to_drop = candidates
                        for kp_drop in to_drop:
                            if valid_by_kp[kp_drop][t_idx]:
                                valid_by_kp[kp_drop][t_idx] = False
                                masked_bones += 1

            # Stage 4 & 5: Bounded gap interpolation + zero-phase smoothing.
            for kp in kps_for_inst:
                ix, iy, iv, interp_mask = _interpolate_bounded_gaps(
                    frame_indices,
                    xs_by_kp[kp],
                    ys_by_kp[kp],
                    valid_by_kp[kp],
                    mode=cfg.interpolate,
                    max_gap_frames=cfg.max_gap_frames,
                )
                gaps_interpolated += int(np.sum(interp_mask))
                sx, sy, sm_n = _smooth_contiguous_runs(ix, iy, iv, cfg)
                points_smoothed += sm_n

                for t_idx in range(n_frames):
                    if (
                        iv[t_idx]
                        and not interp_mask[t_idx]
                        and math.isfinite(orig_xs_by_kp[kp][t_idx])
                        and math.isfinite(orig_ys_by_kp[kp][t_idx])
                    ):
                        disp = math.hypot(
                            float(sx[t_idx] - orig_xs_by_kp[kp][t_idx]),
                            float(sy[t_idx] - orig_ys_by_kp[kp][t_idx]),
                        )
                        displacements.append(disp)

                xs_by_kp[kp] = sx
                ys_by_kp[kp] = sy
                valid_by_kp[kp] = iv
                interp_by_kp[kp] = interp_mask

            # Reassemble filtered PoseInstances per frame.
            for t_idx in range(n_frames):
                orig_inst = inst_frame_map[inst_name].get(t_idx)
                kps_out: Dict[str, PosePoint] = {}
                for kp in kps_for_inst:
                    if not valid_by_kp[kp][t_idx]:
                        continue
                    is_interp = bool(interp_by_kp[kp][t_idx])
                    vis_val = (
                        int(vis_by_kp[kp][t_idx])
                        if vis_by_kp[kp][t_idx] > 0
                        else int(KeypointVisibility.VISIBLE)
                    )
                    score_val = None if is_interp else scores_by_kp[kp][t_idx]
                    kps_out[kp] = PosePoint(
                        x=float(xs_by_kp[kp][t_idx]),
                        y=float(ys_by_kp[kp][t_idx]),
                        score=score_val,
                        visibility=vis_val,
                    )
                if kps_out:
                    filtered_inst_frames[t_idx].append(
                        PoseInstance(
                            instance_name=inst_name,
                            keypoints=kps_out,
                            track_id=(
                                orig_inst.track_id
                                if orig_inst is not None
                                else track_id
                            ),
                            score=orig_inst.score if orig_inst is not None else None,
                            is_predicted=(
                                orig_inst.is_predicted
                                if orig_inst is not None
                                else is_pred_default
                            ),
                            skeleton_name=(
                                orig_inst.skeleton_name
                                if orig_inst is not None
                                else None
                            ),
                        )
                    )

        for t_idx, orig_frame in enumerate(vframes_sorted):
            new_frames.append(
                PoseFrame(
                    frame_index=orig_frame.frame_index,
                    instances=filtered_inst_frames.get(t_idx, []),
                    image_path=orig_frame.image_path,
                    source_image_path=orig_frame.source_image_path,
                    image_width=orig_frame.image_width,
                    image_height=orig_frame.image_height,
                    video_index=orig_frame.video_index,
                    video_name=orig_frame.video_name,
                )
            )

    filtered_ds = PoseDataset(
        frames=new_frames,
        keypoints=list(dataset.keypoints or all_kps),
        edges=list(dataset.edges or edges),
        symmetry_pairs=list(dataset.symmetry_pairs),
        instances=list(dataset.instances),
        scorer=dataset.scorer,
        multi_animal=dataset.is_multi_animal(),
        source_format=dataset.source_format,
        video_paths=dict(dataset.video_paths),
        metadata=dict(dataset.metadata),
        warnings=list(dataset.warnings),
    )

    mean_disp = float(np.mean(displacements)) if displacements else 0.0
    max_disp = float(np.max(displacements)) if displacements else 0.0
    report = PoseFilterReport(
        source_path=source_label,
        output_path=None,
        method=cfg.method,
        interpolate=cfg.interpolate,
        frames_processed=len(new_frames),
        instances_processed=len(seen_instances),
        keypoints_total=keypoints_total,
        masked_low_confidence=masked_low_conf,
        masked_jump_outliers=masked_jumps,
        masked_bone_outliers=masked_bones,
        gaps_interpolated=gaps_interpolated,
        points_smoothed=points_smoothed,
        mean_displacement_px=mean_disp,
        max_displacement_px=max_disp,
    )
    return filtered_ds, report


def _write_filtered_labelme_preserving_non_points(
    source_dir: Path,
    output_dir: Path,
    filtered_ds: PoseDataset,
    *,
    pose_schema: Optional[PoseSchema] = None,
) -> None:
    """Write filtered point shapes while strictly preserving all non-point shapes (polygons/zones)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    is_multi = filtered_ds.is_multi_animal()
    inst_order = filtered_ds.resolved_instances()
    inst_to_gid = {name: idx for idx, name in enumerate(inst_order)}

    frame_map: Dict[int, PoseFrame] = {
        int(f.frame_index): f for f in filtered_ds.frames
    }
    json_files: Dict[int, Path] = {}
    for name in sorted(os.listdir(source_dir)):
        if not name.lower().endswith(".json"):
            continue
        if name.lower() in ("pose_schema.json", "project.annolid.json"):
            continue
        m = re.search(r"(\d+)(?=\.json$)", name)
        if m is not None:
            json_files[int(m.group(1))] = source_dir / name

    # Copy pose_schema file if writing into a new output directory.
    if output_dir.resolve() != source_dir.resolve():
        for fname in ("pose_schema.json", "pose_schema.yaml", "pose_schema.yml"):
            src_schema = source_dir / fname
            if src_schema.is_file():
                shutil.copy2(src_schema, output_dir / fname)

    if not json_files:
        # Fallback when source_dir only had AnnotationStore NDJSON.
        write_pose_to_labelme_dir(filtered_ds, output_dir, overwrite=True)
        return

    for fidx, src_json in sorted(json_files.items()):
        try:
            payload = load_labelme_json(src_json)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        updated_payload = copy.deepcopy(payload)
        existing_shapes = list(updated_payload.get("shapes") or [])
        kept_non_points = [
            s
            for s in existing_shapes
            if not (
                isinstance(s, Mapping)
                and str(s.get("shape_type") or "").strip().lower() == "point"
                and not is_zone_shape_payload(s)
            )
        ]

        # Detect whether original points used prefixed labels.
        orig_points = [
            s
            for s in existing_shapes
            if isinstance(s, Mapping)
            and str(s.get("shape_type") or "").strip().lower() == "point"
        ]
        use_prefix = False
        sep = getattr(pose_schema, "instance_separator", "_") if pose_schema else "_"
        if is_multi and orig_points:
            for pt_s in orig_points:
                lbl = str(pt_s.get("label") or "")
                inst_lbl = str(pt_s.get("instance_label") or "")
                if inst_lbl and lbl.startswith(f"{inst_lbl}{sep}"):
                    use_prefix = True
                    break

        new_point_shapes: List[Dict[str, Any]] = []
        filtered_frame = frame_map.get(fidx)
        if filtered_frame is not None:
            for inst in filtered_frame.instances:
                gid = (
                    inst.track_id
                    if inst.track_id is not None
                    else (inst_to_gid.get(inst.instance_name) if is_multi else None)
                )
                for kp_name, pt in inst.keypoints.items():
                    if not pt.is_valid or int(pt.visibility) <= 0:
                        continue
                    shape_dict = build_labelme_point_shape(
                        keypoint_label=kp_name,
                        x=pt.x,
                        y=pt.y,
                        instance_label=inst.instance_name if is_multi else None,
                        group_id=gid,
                        visibility=pt.visibility,
                        score=pt.score,
                        is_predicted=inst.is_predicted,
                        prefix_with_instance=use_prefix,
                        instance_separator=sep,
                    )
                    new_point_shapes.append(shape_dict)

        updated_payload["shapes"] = kept_non_points + new_point_shapes
        dst_json = output_dir / src_json.name
        atomic_write_json(dst_json, updated_payload, indent=2)


def filter_pose_source(
    source: str | Path,
    output_path: Optional[str | Path] = None,
    config: Optional[PoseFilterConfig] = None,
    *,
    source_format: str = "auto",
    target_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    video_path: Optional[str | Path] = None,
    scorer: str = "annolid",
    in_place: bool = False,
) -> PoseFilterReport:
    """Load, filter/interpolate, and write pose predictions to LabelMe, DLC, SLEAP, or COCO."""
    src_path = Path(source).expanduser().resolve()
    detected_src = (
        detect_pose_source_format(src_path)
        if normalize_pose_format(source_format) == "auto"
        else normalize_pose_format(source_format)
    )
    schema_obj: Optional[PoseSchema] = None
    if isinstance(pose_schema, PoseSchema):
        schema_obj = pose_schema
    elif pose_schema is not None:
        schema_obj = PoseSchema.load(pose_schema)
    else:
        found = find_pose_schema_in_dir(src_path, search_parent=True)
        if found is not None:
            schema_obj = PoseSchema.load(found)

    dataset = read_pose_dataset(
        src_path,
        source_format=detected_src,
        video_path=video_path,
        pose_schema=schema_obj,
    )
    filtered_ds, report = filter_pose_dataset(
        dataset,
        config=config,
        pose_schema=schema_obj,
        source_label=str(src_path),
    )

    tgt_fmt = normalize_pose_format(target_format)
    if tgt_fmt == "auto":
        if detected_src in ("labelme_dir", "labelme_json"):
            tgt_fmt = "labelme_dir"
        elif detected_src == "deeplabcut_h5":
            tgt_fmt = "deeplabcut_h5"
        elif detected_src == "deeplabcut_csv":
            tgt_fmt = "deeplabcut_csv"
        elif detected_src == "sleap_analysis_h5":
            tgt_fmt = "sleap_analysis_h5"
        elif detected_src == "sleap_slp":
            tgt_fmt = "sleap_slp"
        elif detected_src == "coco_json":
            tgt_fmt = "coco_json"
        else:
            tgt_fmt = "labelme_dir"

    if in_place and tgt_fmt == "labelme_dir" and src_path.is_dir():
        out = src_path
    elif output_path is not None:
        out = Path(output_path).expanduser().resolve()
    else:
        suffix_map = {
            "labelme_dir": "_filtered",
            "deeplabcut_csv": "_filtered.csv",
            "deeplabcut_h5": "_filtered.h5",
            "sleap_analysis_h5": "_filtered.analysis.h5",
            "sleap_slp": "_filtered.slp",
            "coco_json": "_filtered_coco.json",
        }
        out = src_path.parent / f"{src_path.stem}{suffix_map.get(tgt_fmt, '_filtered')}"

    if tgt_fmt == "labelme_dir":
        if src_path.is_dir() and detected_src == "labelme_dir":
            _write_filtered_labelme_preserving_non_points(
                src_path, out, filtered_ds, pose_schema=schema_obj
            )
        else:
            write_pose_to_labelme_dir(filtered_ds, out, overwrite=True)
    elif tgt_fmt == "deeplabcut_csv":
        write_dlc_analysis_csv(filtered_ds, out, scorer=scorer)
    elif tgt_fmt == "deeplabcut_h5":
        write_dlc_analysis_h5(filtered_ds, out, scorer=scorer)
    elif tgt_fmt == "sleap_analysis_h5":
        write_sleap_analysis_h5(filtered_ds, out, video_path=video_path)
    elif tgt_fmt in ("sleap_slp", "sleap"):
        write_sleap_slp(filtered_ds, out, video_path=video_path)
    elif tgt_fmt == "coco_json":
        write_pose_to_coco_json(filtered_ds, out)
    else:
        raise ValueError(f"Unsupported filtered pose target_format={target_format!r}.")

    return PoseFilterReport(
        source_path=report.source_path,
        output_path=str(out),
        method=report.method,
        interpolate=report.interpolate,
        frames_processed=report.frames_processed,
        instances_processed=report.instances_processed,
        keypoints_total=report.keypoints_total,
        masked_low_confidence=report.masked_low_confidence,
        masked_jump_outliers=report.masked_jump_outliers,
        masked_bone_outliers=report.masked_bone_outliers,
        gaps_interpolated=report.gaps_interpolated,
        points_smoothed=report.points_smoothed,
        mean_displacement_px=report.mean_displacement_px,
        max_displacement_px=report.max_displacement_px,
    )


def _match_shape_instance_identity(
    shape: Mapping[str, Any],
    candidates: Tuple[str, str],
    *,
    sep: str = "_",
) -> Optional[str]:
    """Identify whether a shape (polygon, bbox, or pose keypoint) belongs to ``candidates[0]`` or ``candidates[1]``."""
    a_name, b_name = candidates
    target_set = {a_name, b_name}
    flags = shape.get("flags") if isinstance(shape.get("flags"), Mapping) else {}

    for key in ("instance_label", "parent_instance", "instance_name"):
        for container in (shape, flags):
            val = str(container.get(key) or "").strip()
            if val in target_set:
                return val

    raw_label = str(shape.get("label") or "").strip()
    if raw_label in target_set:
        return raw_label

    if ":" in raw_label:
        left = raw_label.split(":", 1)[0].strip()
        if left in target_set:
            return left

    # Check longest prefix match with separator (e.g. "mouse_1_nose" vs "mouse_2_nose").
    for cand in sorted(target_set, key=len, reverse=True):
        for delim in (sep, "_", "-", ":", "|", " "):
            if (
                raw_label.startswith(f"{cand}{delim}")
                and len(raw_label) > len(cand) + 1
            ):
                return cand

    # Check group_id match when candidate is a group_id string/int.
    gid = shape.get("group_id")
    if gid is not None and str(gid).strip() in target_set:
        return str(gid).strip()

    return None


def _swap_single_shape_identity(
    shape: Dict[str, Any],
    old_inst: str,
    new_inst: str,
    *,
    gid_swap_map: Mapping[Any, Any],
    sep: str = "_",
) -> bool:
    """Mutate a copied shape dict so its identity switches from ``old_inst`` to ``new_inst``."""
    changed = False
    flags = shape.get("flags")
    if isinstance(flags, dict):
        for key in ("instance_label", "parent_instance", "instance_name"):
            if str(flags.get(key) or "").strip() == old_inst:
                flags[key] = new_inst
                changed = True

    for key in ("instance_label", "parent_instance", "instance_name"):
        if str(shape.get(key) or "").strip() == old_inst:
            shape[key] = new_inst
            changed = True

    raw_label = str(shape.get("label") or "").strip()
    if raw_label == old_inst:
        shape["label"] = new_inst
        changed = True
    elif ":" in raw_label and raw_label.split(":", 1)[0].strip() == old_inst:
        suffix = raw_label.split(":", 1)[1].strip()
        shape["label"] = f"{new_inst}:{suffix}" if suffix else new_inst
        changed = True
    else:
        for delim in (sep, "_", "-", ":", "|", " "):
            prefix = f"{old_inst}{delim}"
            if raw_label.startswith(prefix) and len(raw_label) > len(prefix):
                shape["label"] = f"{new_inst}{delim}{raw_label[len(prefix) :]}"
                changed = True
                break

    gid = shape.get("group_id")
    if gid in gid_swap_map:
        shape["group_id"] = gid_swap_map[gid]
        changed = True
    elif gid is not None and str(gid).strip() == old_inst:
        shape["group_id"] = (
            int(new_inst)
            if isinstance(gid, int)
            and not isinstance(gid, bool)
            and new_inst.lstrip("-").isdigit()
            else new_inst
        )
        changed = True

    return changed


def swap_pose_instances_from_frame(
    source_dir: str | Path,
    *,
    start_frame: int,
    instance_a: str,
    instance_b: str,
    end_frame: Optional[int] = None,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    dry_run: bool = False,
) -> PoseSwapResult:
    """Swap all shapes (pose keypoints AND parent polygons/bboxes) between two instances from ``start_frame`` forward.

    Implements SLEAP's signature proofreading action ("Swap track identities from
    current frame forward") across both per-frame LabelMe JSON files and
    ``AnnotationStore`` NDJSON sidecars.
    """
    folder = Path(source_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {folder}")

    inst_a = str(instance_a or "").strip()
    inst_b = str(instance_b or "").strip()
    if not inst_a or not inst_b or inst_a == inst_b:
        raise ValueError(
            "instance_a and instance_b must be two distinct non-empty names."
        )

    sep = "_"
    if isinstance(pose_schema, PoseSchema):
        sep = pose_schema.instance_separator or "_"
    elif pose_schema is not None:
        sep = PoseSchema.load(pose_schema).instance_separator or "_"
    else:
        found = find_pose_schema_in_dir(folder, search_parent=True)
        if found is not None:
            sep = PoseSchema.load(found).instance_separator or "_"

    frame_paths: Dict[int, Path] = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".json"):
            continue
        if name.lower() in ("pose_schema.json", "project.annolid.json"):
            continue
        m = re.search(r"(\d+)(?=\.json$)", name)
        if m is not None:
            frame_paths[int(m.group(1))] = folder / name

    stem = folder.name
    store = AnnotationStore.for_frame_path(folder / f"{stem}_000000000.json")
    store_exists = store.store_path.exists()
    store_only_frames: set[int] = set()
    if store_exists:
        for fnum in store.iter_frames():
            f_i = int(fnum)
            if f_i not in frame_paths:
                frame_paths[f_i] = folder / f"{stem}_{f_i:09d}.json"
                store_only_frames.add(f_i)

    frames_scanned = 0
    frames_modified = 0
    shapes_swapped = 0
    points_swapped = 0
    polygons_swapped = 0
    modified_indices: List[int] = []

    for fnum in sorted(frame_paths):
        if fnum < int(start_frame):
            continue
        if end_frame is not None and fnum > int(end_frame):
            continue
        fpath = frame_paths[fnum]
        try:
            payload = load_labelme_json(fpath)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        frames_scanned += 1
        shapes = payload.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            continue

        # Discover paired group_ids on this frame so group_ids swap cleanly alongside labels.
        gid_by_inst: Dict[str, Any] = {}
        matched_owners: List[Optional[str]] = []
        for s in shapes:
            if not isinstance(s, Mapping) or is_zone_shape_payload(s):
                matched_owners.append(None)
                continue
            owner = _match_shape_instance_identity(s, (inst_a, inst_b), sep=sep)
            matched_owners.append(owner)
            if owner is not None and owner not in gid_by_inst:
                gid = s.get("group_id")
                if gid not in (None, ""):
                    gid_by_inst[owner] = gid

        gid_swap_map: Dict[Any, Any] = {}
        if inst_a in gid_by_inst and inst_b in gid_by_inst:
            ga, gb = gid_by_inst[inst_a], gid_by_inst[inst_b]
            if ga != gb:
                gid_swap_map[ga] = gb
                gid_swap_map[gb] = ga

        # Also associate plain keypoints that only carry a swapped group_id.
        if gid_swap_map:
            gid_to_owner = {gid_by_inst[inst_a]: inst_a, gid_by_inst[inst_b]: inst_b}
            for idx, (s, owner) in enumerate(zip(shapes, matched_owners)):
                if (
                    owner is None
                    and isinstance(s, Mapping)
                    and not is_zone_shape_payload(s)
                ):
                    gid = s.get("group_id")
                    if gid in gid_to_owner:
                        matched_owners[idx] = gid_to_owner[gid]

        updated_shapes = copy.deepcopy(shapes)
        frame_changed = False
        for s_copy, owner in zip(updated_shapes, matched_owners):
            if owner is None or not isinstance(s_copy, dict):
                continue
            target_inst = inst_b if owner == inst_a else inst_a
            if _swap_single_shape_identity(
                s_copy,
                owner,
                target_inst,
                gid_swap_map=gid_swap_map,
                sep=sep,
            ):
                frame_changed = True
                shapes_swapped += 1
                stype = str(s_copy.get("shape_type") or "").strip().lower()
                if stype == "point":
                    points_swapped += 1
                else:
                    polygons_swapped += 1

        if frame_changed:
            frames_modified += 1
            modified_indices.append(fnum)
            if not dry_run:
                updated_payload = copy.deepcopy(payload)
                updated_payload["shapes"] = updated_shapes
                if fnum in store_only_frames and store_exists:
                    store.save_frame(fnum, updated_payload)
                else:
                    atomic_write_json(fpath, updated_payload, indent=2)
                    if store_exists and store.has_frame(fnum):
                        store.save_frame(fnum, updated_payload)

    return PoseSwapResult(
        source_dir=str(folder),
        start_frame=int(start_frame),
        end_frame=int(end_frame) if end_frame is not None else None,
        instance_a=inst_a,
        instance_b=inst_b,
        dry_run=bool(dry_run),
        frames_scanned=frames_scanned,
        frames_modified=frames_modified,
        shapes_swapped=shapes_swapped,
        points_swapped=points_swapped,
        polygons_swapped=polygons_swapped,
        modified_frame_indices=tuple(modified_indices),
    )
