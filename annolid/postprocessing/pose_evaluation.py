"""Model-agnostic pose estimation evaluation suite (DeepLabCut + SLEAP + COCO OKS metrics).

Compares ground-truth and predicted pose datasets from any Annolid-supported
format (LabelMe directory / AnnotationStore, SLEAP ``.slp``/``.analysis.h5``,
DeepLabCut CSV/H5/training project, COCO keypoints JSON, or Annolid tracking CSV)
and computes:

- DeepLabCut metrics: pixel RMSE and MAE (all predictions and ``score >= pcutoff``),
- SLEAP localization metrics: median (p50), p75, p90, p95, max Euclidean error,
- Fixed-pixel PCK (e.g., 2/4/8/16 px) and instance-scale-normalized PCK (e.g., 5%/10%/20%),
- Multi-animal Hungarian instance matching and COCO/SLEAP OKS mAP (0.50:0.05:0.95), AP50, AP75, AR,
- Keypoint visibility/detection precision, recall, and F1,
- Multi-animal left/right symmetry swap rate,
- Per-keypoint breakdown and ranked worst-error frames for active-learning refinement,
- JSON, CSV, and self-contained HTML5 + inline SVG report generation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import html
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover - pure-Python fallback below
    _scipy_linear_sum_assignment = None  # type: ignore[assignment]

from annolid.annotation.pose_interop.converters import read_pose_dataset
from annolid.annotation.pose_interop.model import PoseDataset, PoseFrame, PoseInstance
from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "PerKeypointEvalMetrics",
    "FrameEvalSummary",
    "PoseEvaluationReport",
    "evaluate_pose_predictions",
    "format_pose_eval_table",
    "write_pose_eval_report_bundle",
]

_OKS_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def _linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve rectangular minimum-cost assignment via SciPy or a deterministic greedy fallback."""
    if cost.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost)
    # Deterministic greedy fallback when scipy is unavailable.
    n_rows, n_cols = cost.shape
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    flat = [
        (float(cost[r, c]), r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if math.isfinite(float(cost[r, c]))
    ]
    flat.sort(key=lambda item: (item[0], item[1], item[2]))
    for _val, r, c in flat:
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        pairs.append((r, c))
    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return (
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    )


def _average_precision(
    scores: Sequence[float],
    is_tp: Sequence[bool],
    *,
    num_gt: int,
) -> Tuple[float, float]:
    """Return ``(AP, max_recall)`` using 101-point COCO-style precision envelope."""
    if num_gt <= 0 or not scores:
        return 0.0, 0.0
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    tp_cum = 0.0
    fp_cum = 0.0
    recalls: List[float] = []
    precisions: List[float] = []
    for idx in order:
        if is_tp[idx]:
            tp_cum += 1.0
        else:
            fp_cum += 1.0
        recalls.append(tp_cum / float(num_gt))
        precisions.append(tp_cum / max(1e-12, tp_cum + fp_cum))

    max_recall = recalls[-1] if recalls else 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap), float(max_recall)


def _instance_scale_and_area(inst: PoseInstance) -> Tuple[float, float]:
    """Return ``(scale_px, area_px2)`` for OKS and normalized PCK."""
    inst_bbox = getattr(inst, "bbox", None)
    if inst_bbox is not None and len(inst_bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in inst_bbox]
        bw = max(1.0, abs(x2 - x1))
        bh = max(1.0, abs(y2 - y1))
        area = max(1.0, bw * bh)
        return float(math.hypot(bw, bh)), float(area)

    xs: List[float] = []
    ys: List[float] = []
    for pt in inst.keypoints.values():
        if pt.is_valid and int(pt.visibility) > 0:
            xs.append(float(pt.x))
            ys.append(float(pt.y))
    if len(xs) >= 2:
        bw = max(1.0, max(xs) - min(xs))
        bh = max(1.0, max(ys) - min(ys))
        diag = max(10.0, math.hypot(bw, bh))
        area = max(100.0, bw * bh, (0.5 * diag) ** 2)
        return float(diag), float(area)
    return 64.0, 4096.0


def _compute_instance_oks(
    gt_inst: PoseInstance,
    pred_inst: PoseInstance,
    *,
    keypoints: Sequence[str],
    sigmas: Mapping[str, float],
    default_sigma: float = 0.05,
) -> Tuple[float, float, int]:
    """Return ``(oks, mean_dist_px, shared_visible_count)`` for one GT/Pred pair."""
    _scale, area = _instance_scale_and_area(gt_inst)
    oks_terms: List[float] = []
    dists: List[float] = []
    gt_visible = 0

    for kp in keypoints:
        gt_pt = gt_inst.keypoints.get(kp)
        if gt_pt is None or not gt_pt.is_valid or int(gt_pt.visibility) <= 0:
            continue
        gt_visible += 1
        pred_pt = pred_inst.keypoints.get(kp)
        if pred_pt is None or not pred_pt.is_valid or int(pred_pt.visibility) <= 0:
            oks_terms.append(0.0)
            continue
        dx = float(pred_pt.x) - float(gt_pt.x)
        dy = float(pred_pt.y) - float(gt_pt.y)
        d2 = dx * dx + dy * dy
        dists.append(math.sqrt(d2))
        sigma = max(1e-4, float(sigmas.get(kp, default_sigma)))
        denom = 2.0 * (sigma * sigma) * area * 2.0
        oks_terms.append(math.exp(-d2 / max(denom, 1e-9)))

    if gt_visible == 0:
        return 0.0, float("inf"), 0
    oks = float(sum(oks_terms) / float(gt_visible))
    mean_dist = float(sum(dists) / float(len(dists))) if dists else float("inf")
    return oks, mean_dist, len(dists)


def _percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "rmse": None,
            "mae": None,
            "std": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "mae": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _round_opt(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), ndigits)


@dataclass(frozen=True)
class PerKeypointEvalMetrics:
    """Localization and detection metrics for one anatomical keypoint."""

    keypoint: str
    gt_count: int
    matched_count: int
    pcutoff_count: int
    recall: float
    rmse_px: Optional[float]
    rmse_pcutoff_px: Optional[float]
    mae_px: Optional[float]
    p50_px: Optional[float]
    p75_px: Optional[float]
    p90_px: Optional[float]
    p95_px: Optional[float]
    pck_px: Dict[str, Optional[float]] = field(default_factory=dict)
    pck_norm: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keypoint": self.keypoint,
            "gt_count": int(self.gt_count),
            "matched_count": int(self.matched_count),
            "pcutoff_count": int(self.pcutoff_count),
            "recall": round(float(self.recall), 6),
            "rmse_px": _round_opt(self.rmse_px),
            "rmse_pcutoff_px": _round_opt(self.rmse_pcutoff_px),
            "mae_px": _round_opt(self.mae_px),
            "p50_px": _round_opt(self.p50_px),
            "p75_px": _round_opt(self.p75_px),
            "p90_px": _round_opt(self.p90_px),
            "p95_px": _round_opt(self.p95_px),
            "pck_px": {k: _round_opt(v, 6) for k, v in self.pck_px.items()},
            "pck_norm": {k: _round_opt(v, 6) for k, v in self.pck_norm.items()},
        }


@dataclass(frozen=True)
class FrameEvalSummary:
    """Per-frame error summary used to rank worst frames for proofreading."""

    frame_index: int
    video_index: int
    image_path: Optional[str]
    gt_instances: int
    pred_instances: int
    matched_instances: int
    matched_keypoints: int
    missing_keypoints: int
    mean_error_px: Optional[float]
    max_error_px: Optional[float]
    mean_oks: Optional[float]
    swaps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "video_index": int(self.video_index),
            "image_path": self.image_path,
            "gt_instances": int(self.gt_instances),
            "pred_instances": int(self.pred_instances),
            "matched_instances": int(self.matched_instances),
            "matched_keypoints": int(self.matched_keypoints),
            "missing_keypoints": int(self.missing_keypoints),
            "mean_error_px": _round_opt(self.mean_error_px),
            "max_error_px": _round_opt(self.max_error_px),
            "mean_oks": _round_opt(self.mean_oks, 6),
            "swaps": int(self.swaps),
        }


@dataclass(frozen=True)
class PoseEvaluationReport:
    """Complete DLC + SLEAP + COCO OKS pose evaluation report."""

    gt_source: str
    pred_source: str
    keypoints: Tuple[str, ...]
    pcutoff: float
    frames_gt: int
    frames_evaluated: int
    gt_instances_total: int
    pred_instances_total: int
    matched_instances_total: int
    gt_keypoints_total: int
    matched_keypoints_total: int
    pcutoff_keypoints_total: int
    rmse_px: Optional[float]
    rmse_pcutoff_px: Optional[float]
    mae_px: Optional[float]
    mae_pcutoff_px: Optional[float]
    std_error_px: Optional[float]
    p50_px: Optional[float]
    p75_px: Optional[float]
    p90_px: Optional[float]
    p95_px: Optional[float]
    max_error_px: Optional[float]
    mean_oks: Optional[float]
    oks_map: float
    oks_ap50: float
    oks_ap75: float
    oks_ar: float
    keypoint_precision: float
    keypoint_recall: float
    keypoint_f1: float
    instance_precision: float
    instance_recall: float
    instance_f1: float
    swap_pairs_total: int
    swap_pairs_swapped: int
    swap_rate: Optional[float]
    pck_px: Dict[str, Optional[float]]
    pck_norm: Dict[str, Optional[float]]
    per_keypoint: Tuple[PerKeypointEvalMetrics, ...]
    worst_frames: Tuple[FrameEvalSummary, ...]
    all_errors_px: Tuple[float, ...] = field(default_factory=tuple, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gt_source": self.gt_source,
            "pred_source": self.pred_source,
            "keypoints": list(self.keypoints),
            "pcutoff": float(self.pcutoff),
            "frames_gt": int(self.frames_gt),
            "frames_evaluated": int(self.frames_evaluated),
            "gt_instances_total": int(self.gt_instances_total),
            "pred_instances_total": int(self.pred_instances_total),
            "matched_instances_total": int(self.matched_instances_total),
            "gt_keypoints_total": int(self.gt_keypoints_total),
            "matched_keypoints_total": int(self.matched_keypoints_total),
            "pcutoff_keypoints_total": int(self.pcutoff_keypoints_total),
            "rmse_px": _round_opt(self.rmse_px),
            "rmse_pcutoff_px": _round_opt(self.rmse_pcutoff_px),
            "mae_px": _round_opt(self.mae_px),
            "mae_pcutoff_px": _round_opt(self.mae_pcutoff_px),
            "std_error_px": _round_opt(self.std_error_px),
            "median_error_px": _round_opt(self.p50_px),
            "p50_px": _round_opt(self.p50_px),
            "p75_px": _round_opt(self.p75_px),
            "p90_px": _round_opt(self.p90_px),
            "p95_px": _round_opt(self.p95_px),
            "max_error_px": _round_opt(self.max_error_px),
            "mean_oks": _round_opt(self.mean_oks, 6),
            "oks_map": round(float(self.oks_map), 6),
            "oks_ap50": round(float(self.oks_ap50), 6),
            "oks_ap75": round(float(self.oks_ap75), 6),
            "oks_ar": round(float(self.oks_ar), 6),
            "keypoint_precision": round(float(self.keypoint_precision), 6),
            "keypoint_recall": round(float(self.keypoint_recall), 6),
            "keypoint_f1": round(float(self.keypoint_f1), 6),
            "instance_precision": round(float(self.instance_precision), 6),
            "instance_recall": round(float(self.instance_recall), 6),
            "instance_f1": round(float(self.instance_f1), 6),
            "swap_pairs_total": int(self.swap_pairs_total),
            "swap_pairs_swapped": int(self.swap_pairs_swapped),
            "swap_rate": _round_opt(self.swap_rate, 6),
            "pck_px": {k: _round_opt(v, 6) for k, v in self.pck_px.items()},
            "pck_norm": {k: _round_opt(v, 6) for k, v in self.pck_norm.items()},
            "per_keypoint": [item.to_dict() for item in self.per_keypoint],
            "worst_frames": [item.to_dict() for item in self.worst_frames],
        }


def _align_frames(
    gt_frames: Sequence[PoseFrame],
    pred_frames: Sequence[PoseFrame],
) -> List[Tuple[PoseFrame, Optional[PoseFrame]]]:
    """Align GT and Predicted frames by ``(video_index, frame_index)``, falling back to image stem."""
    pred_by_vf: Dict[Tuple[int, int], PoseFrame] = {}
    pred_by_stem: Dict[str, PoseFrame] = {}
    for pf in pred_frames:
        pred_by_vf[(int(pf.video_index), int(pf.frame_index))] = pf
        if pf.image_path:
            pred_by_stem[Path(pf.image_path).stem.lower()] = pf

    direct_matches = sum(
        1
        for gf in gt_frames
        if (int(gf.video_index), int(gf.frame_index)) in pred_by_vf
    )
    stem_matches = sum(
        1
        for gf in gt_frames
        if gf.image_path and Path(gf.image_path).stem.lower() in pred_by_stem
    )
    use_stem_first = direct_matches == 0 and stem_matches > 0

    aligned: List[Tuple[PoseFrame, Optional[PoseFrame]]] = []
    for gf in sorted(gt_frames, key=lambda f: (int(f.video_index), int(f.frame_index))):
        pf: Optional[PoseFrame] = None
        if use_stem_first and gf.image_path:
            pf = pred_by_stem.get(Path(gf.image_path).stem.lower())
        if pf is None:
            pf = pred_by_vf.get((int(gf.video_index), int(gf.frame_index)))
        if pf is None and gf.image_path:
            pf = pred_by_stem.get(Path(gf.image_path).stem.lower())
        aligned.append((gf, pf))
    return aligned


def evaluate_pose_predictions(
    gt_source: str | Path | PoseDataset,
    pred_source: str | Path | PoseDataset,
    *,
    gt_format: str = "auto",
    pred_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    pcutoff: float = 0.6,
    pck_thresholds_px: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
    pck_norm_thresholds: Sequence[float] = (0.05, 0.10, 0.20),
    oks_sigma: float | Mapping[str, float] = 0.05,
    top_k_worst_frames: int = 50,
) -> PoseEvaluationReport:
    """Evaluate predicted poses against ground truth with DLC, SLEAP, and COCO OKS metrics."""
    if isinstance(gt_source, PoseDataset):
        gt_ds = gt_source
        gt_label = "<PoseDataset:gt>"
        gt_path_obj: Optional[Path] = None
    else:
        gt_path_obj = Path(gt_source).expanduser().resolve()
        gt_label = str(gt_path_obj)
        gt_ds = read_pose_dataset(
            gt_path_obj, source_format=gt_format, pose_schema=pose_schema
        )

    if isinstance(pred_source, PoseDataset):
        pred_ds = pred_source
        pred_label = "<PoseDataset:pred>"
    else:
        pred_path_obj = Path(pred_source).expanduser().resolve()
        pred_label = str(pred_path_obj)
        pred_ds = read_pose_dataset(
            pred_path_obj, source_format=pred_format, pose_schema=pose_schema
        )

    schema: Optional[PoseSchema] = None
    if isinstance(pose_schema, PoseSchema):
        schema = pose_schema
    elif pose_schema is not None:
        schema = PoseSchema.load(pose_schema)
    elif gt_path_obj is not None:
        found = find_pose_schema_in_dir(gt_path_obj, search_parent=True)
        if found is not None:
            schema = PoseSchema.load(found)
    if schema is None:
        schema = gt_ds.to_pose_schema(preset="auto")

    keypoints: List[str] = list(schema.keypoints or gt_ds.resolved_keypoints())
    if not keypoints:
        keypoints = list(pred_ds.resolved_keypoints())
    kp_set = set(keypoints)

    sym_pairs = list(schema.base_symmetry_pairs()) if schema else []
    if not sym_pairs:
        sym_pairs = PoseSchema.infer_symmetry_pairs(keypoints)
    sym_pairs = [(a, b) for a, b in sym_pairs if a in kp_set and b in kp_set]

    if isinstance(oks_sigma, Mapping):
        sigmas: Dict[str, float] = {str(k): float(v) for k, v in oks_sigma.items()}
        default_sigma = 0.05
    else:
        sigmas = {}
        default_sigma = max(1e-4, float(oks_sigma))

    px_thresholds = sorted({float(t) for t in pck_thresholds_px if float(t) > 0.0})
    norm_thresholds = sorted({float(t) for t in pck_norm_thresholds if float(t) > 0.0})

    aligned = _align_frames(gt_ds.frames, pred_ds.frames)

    all_errors: List[float] = []
    all_errors_pcutoff: List[float] = []
    all_norm_errors: List[float] = []
    all_matched_oks: List[float] = []

    kp_gt_counts: Dict[str, int] = {kp: 0 for kp in keypoints}
    kp_errors: Dict[str, List[float]] = {kp: [] for kp in keypoints}
    kp_errors_pcutoff: Dict[str, List[float]] = {kp: [] for kp in keypoints}
    kp_norm_errors: Dict[str, List[float]] = {kp: [] for kp in keypoints}

    kp_tp = 0
    kp_fn = 0
    kp_fp = 0

    gt_instances_total = 0
    pred_instances_total = 0
    matched_instances_total = 0
    swap_pairs_total = 0
    swap_pairs_swapped = 0

    # Per-threshold OKS detection records for COCO-style AP/AR calculation.
    oks_det_scores: Dict[float, List[float]] = {thr: [] for thr in _OKS_THRESHOLDS}
    oks_det_tp: Dict[float, List[bool]] = {thr: [] for thr in _OKS_THRESHOLDS}

    frame_summaries: List[FrameEvalSummary] = []
    frames_evaluated = 0

    for gf, pf in aligned:
        gt_insts = list(gf.instances)
        pred_insts = list(pf.instances) if pf is not None else []
        if not gt_insts and not pred_insts:
            continue
        if pf is not None:
            frames_evaluated += 1

        n_gt = len(gt_insts)
        n_pred = len(pred_insts)
        gt_instances_total += n_gt
        pred_instances_total += n_pred

        # Build pairwise OKS & distance matrices for multi-animal assignment.
        oks_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
        dist_mat = np.full((n_gt, n_pred), 1e9, dtype=np.float64)
        cost_mat = np.full((n_gt, n_pred), 1e6, dtype=np.float64)

        for gi, g_inst in enumerate(gt_insts):
            scale_px, _area = _instance_scale_and_area(g_inst)
            for pi, p_inst in enumerate(pred_insts):
                oks, mean_d, shared = _compute_instance_oks(
                    g_inst,
                    p_inst,
                    keypoints=keypoints,
                    sigmas=sigmas,
                    default_sigma=default_sigma,
                )
                oks_mat[gi, pi] = oks
                dist_mat[gi, pi] = mean_d
                if shared > 0 and math.isfinite(mean_d):
                    # Combine OKS (primary) and normalized distance (tie-breaker for 0-OKS far matches).
                    cost_mat[gi, pi] = (1.0 - oks) + 0.05 * min(
                        10.0, mean_d / max(scale_px, 1.0)
                    )

        # COCO-style greedy OKS matching sorted by prediction score per threshold.
        pred_scores_sorted = sorted(
            range(n_pred),
            key=lambda idx: (
                float(pred_insts[idx].score)
                if pred_insts[idx].score is not None
                and math.isfinite(float(pred_insts[idx].score))
                else 1.0
            ),
            reverse=True,
        )
        for thr in _OKS_THRESHOLDS:
            matched_gt_for_thr: set[int] = set()
            for pi in pred_scores_sorted:
                raw_sc = pred_insts[pi].score
                sc = (
                    float(raw_sc)
                    if raw_sc is not None and math.isfinite(float(raw_sc))
                    else 1.0
                )
                best_gi = -1
                best_oks = thr - 1e-12
                for gi in range(n_gt):
                    if gi in matched_gt_for_thr:
                        continue
                    if oks_mat[gi, pi] >= best_oks:
                        best_oks = oks_mat[gi, pi]
                        best_gi = gi
                oks_det_scores[thr].append(sc)
                if best_gi >= 0:
                    matched_gt_for_thr.add(best_gi)
                    oks_det_tp[thr].append(True)
                else:
                    oks_det_tp[thr].append(False)

        # Optimal Hungarian instance assignment for localization metrics & L/R swaps.
        row_ind, col_ind = _linear_sum_assignment(cost_mat)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        frame_errors: List[float] = []
        frame_oks_vals: List[float] = []
        frame_missing_kps = 0
        frame_swaps = 0

        for gi, pi in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_mat[gi, pi] >= 1e5:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            matched_instances_total += 1
            g_inst = gt_insts[gi]
            p_inst = pred_insts[pi]
            scale_px, _area = _instance_scale_and_area(g_inst)
            oks_val = float(oks_mat[gi, pi])
            all_matched_oks.append(oks_val)
            frame_oks_vals.append(oks_val)

            for kp in keypoints:
                g_pt = g_inst.keypoints.get(kp)
                p_pt = p_inst.keypoints.get(kp)
                g_vis = g_pt is not None and g_pt.is_valid and int(g_pt.visibility) > 0
                p_vis = p_pt is not None and p_pt.is_valid and int(p_pt.visibility) > 0
                if g_vis:
                    kp_gt_counts[kp] += 1
                    if p_vis and p_pt is not None and g_pt is not None:
                        err = math.hypot(
                            float(p_pt.x) - float(g_pt.x),
                            float(p_pt.y) - float(g_pt.y),
                        )
                        norm_err = err / max(scale_px, 1.0)
                        all_errors.append(err)
                        all_norm_errors.append(norm_err)
                        kp_errors[kp].append(err)
                        kp_norm_errors[kp].append(norm_err)
                        frame_errors.append(err)
                        kp_tp += 1

                        passes_pcut = (
                            p_pt.score is None
                            or not math.isfinite(float(p_pt.score))
                            or float(p_pt.score) >= float(pcutoff)
                        )
                        if passes_pcut:
                            all_errors_pcutoff.append(err)
                            kp_errors_pcutoff[kp].append(err)
                    else:
                        kp_fn += 1
                        frame_missing_kps += 1
                elif p_vis:
                    kp_fp += 1

            # Check bilateral symmetry swaps for this matched instance.
            for left_kp, right_kp in sym_pairs:
                gl = g_inst.keypoints.get(left_kp)
                gr = g_inst.keypoints.get(right_kp)
                pl = p_inst.keypoints.get(left_kp)
                pr = p_inst.keypoints.get(right_kp)
                if not (
                    gl is not None
                    and gr is not None
                    and pl is not None
                    and pr is not None
                    and gl.is_valid
                    and gr.is_valid
                    and pl.is_valid
                    and pr.is_valid
                    and int(gl.visibility) > 0
                    and int(gr.visibility) > 0
                    and int(pl.visibility) > 0
                    and int(pr.visibility) > 0
                ):
                    continue
                swap_pairs_total += 1
                d_direct = math.hypot(pl.x - gl.x, pl.y - gl.y) + math.hypot(
                    pr.x - gr.x, pr.y - gr.y
                )
                d_cross = math.hypot(pl.x - gr.x, pl.y - gr.y) + math.hypot(
                    pr.x - gl.x, pr.y - gl.y
                )
                if d_cross + 1e-3 < d_direct:
                    swap_pairs_swapped += 1
                    frame_swaps += 1

        # Unmatched GT instances count toward GT totals & FN keypoints.
        for gi, g_inst in enumerate(gt_insts):
            if gi in matched_gt:
                continue
            for kp in keypoints:
                g_pt = g_inst.keypoints.get(kp)
                if g_pt is not None and g_pt.is_valid and int(g_pt.visibility) > 0:
                    kp_gt_counts[kp] += 1
                    kp_fn += 1
                    frame_missing_kps += 1

        # Unmatched Pred instances count toward FP keypoints.
        for pi, p_inst in enumerate(pred_insts):
            if pi in matched_pred:
                continue
            for kp in keypoints:
                p_pt = p_inst.keypoints.get(kp)
                if p_pt is not None and p_pt.is_valid and int(p_pt.visibility) > 0:
                    kp_fp += 1

        frame_summaries.append(
            FrameEvalSummary(
                frame_index=int(gf.frame_index),
                video_index=int(gf.video_index),
                image_path=gf.image_path,
                gt_instances=n_gt,
                pred_instances=n_pred,
                matched_instances=len(matched_gt),
                matched_keypoints=len(frame_errors),
                missing_keypoints=frame_missing_kps,
                mean_error_px=(
                    float(sum(frame_errors) / len(frame_errors))
                    if frame_errors
                    else None
                ),
                max_error_px=float(max(frame_errors)) if frame_errors else None,
                mean_oks=(
                    float(sum(frame_oks_vals) / len(frame_oks_vals))
                    if frame_oks_vals
                    else None
                ),
                swaps=frame_swaps,
            )
        )

    overall_stats = _percentiles(all_errors)
    pcutoff_stats = _percentiles(all_errors_pcutoff)

    pck_px_overall: Dict[str, Optional[float]] = {
        f"{thr:g}": (
            float(sum(1 for e in all_errors if e <= thr) / len(all_errors))
            if all_errors
            else None
        )
        for thr in px_thresholds
    }
    pck_norm_overall: Dict[str, Optional[float]] = {
        f"{thr:g}": (
            float(sum(1 for e in all_norm_errors if e <= thr) / len(all_norm_errors))
            if all_norm_errors
            else None
        )
        for thr in norm_thresholds
    }

    aps: List[float] = []
    ars: List[float] = []
    ap50 = 0.0
    ap75 = 0.0
    for thr in _OKS_THRESHOLDS:
        ap, ar = _average_precision(
            oks_det_scores[thr], oks_det_tp[thr], num_gt=gt_instances_total
        )
        aps.append(ap)
        ars.append(ar)
        if math.isclose(thr, 0.50, abs_tol=1e-6):
            ap50 = ap
        if math.isclose(thr, 0.75, abs_tol=1e-6):
            ap75 = ap
    oks_map = float(sum(aps) / len(aps)) if aps else 0.0
    oks_ar = float(sum(ars) / len(ars)) if ars else 0.0

    kp_prec = float(kp_tp / (kp_tp + kp_fp)) if (kp_tp + kp_fp) > 0 else 0.0
    kp_rec = float(kp_tp / (kp_tp + kp_fn)) if (kp_tp + kp_fn) > 0 else 0.0
    kp_f1 = (
        float(2.0 * kp_prec * kp_rec / (kp_prec + kp_rec))
        if (kp_prec + kp_rec) > 0.0
        else 0.0
    )

    inst_prec = (
        float(matched_instances_total / pred_instances_total)
        if pred_instances_total > 0
        else 0.0
    )
    inst_rec = (
        float(matched_instances_total / gt_instances_total)
        if gt_instances_total > 0
        else 0.0
    )
    inst_f1 = (
        float(2.0 * inst_prec * inst_rec / (inst_prec + inst_rec))
        if (inst_prec + inst_rec) > 0.0
        else 0.0
    )

    per_kp_rows: List[PerKeypointEvalMetrics] = []
    for kp in keypoints:
        gt_n = kp_gt_counts.get(kp, 0)
        errs = kp_errors.get(kp, [])
        errs_pc = kp_errors_pcutoff.get(kp, [])
        n_errs = kp_norm_errors.get(kp, [])
        st = _percentiles(errs)
        st_pc = _percentiles(errs_pc)
        per_kp_rows.append(
            PerKeypointEvalMetrics(
                keypoint=kp,
                gt_count=gt_n,
                matched_count=len(errs),
                pcutoff_count=len(errs_pc),
                recall=float(len(errs) / gt_n) if gt_n > 0 else 0.0,
                rmse_px=st["rmse"],
                rmse_pcutoff_px=st_pc["rmse"],
                mae_px=st["mae"],
                p50_px=st["p50"],
                p75_px=st["p75"],
                p90_px=st["p90"],
                p95_px=st["p95"],
                pck_px={
                    f"{thr:g}": (
                        float(sum(1 for e in errs if e <= thr) / len(errs))
                        if errs
                        else None
                    )
                    for thr in px_thresholds
                },
                pck_norm={
                    f"{thr:g}": (
                        float(sum(1 for e in n_errs if e <= thr) / len(n_errs))
                        if n_errs
                        else None
                    )
                    for thr in norm_thresholds
                },
            )
        )

    # Sort frames worst-first (high error, missing keypoints, or swaps).
    def _frame_severity(item: FrameEvalSummary) -> Tuple[float, int, int]:
        err_term = item.mean_error_px if item.mean_error_px is not None else 1e6
        return (
            err_term + 15.0 * float(item.missing_keypoints) + 25.0 * float(item.swaps),
            item.missing_keypoints,
            -item.frame_index,
        )

    frame_summaries.sort(key=_frame_severity, reverse=True)
    worst = tuple(frame_summaries[: max(0, int(top_k_worst_frames))])

    return PoseEvaluationReport(
        gt_source=gt_label,
        pred_source=pred_label,
        keypoints=tuple(keypoints),
        pcutoff=float(pcutoff),
        frames_gt=len(gt_ds.frames),
        frames_evaluated=frames_evaluated,
        gt_instances_total=gt_instances_total,
        pred_instances_total=pred_instances_total,
        matched_instances_total=matched_instances_total,
        gt_keypoints_total=kp_tp + kp_fn,
        matched_keypoints_total=len(all_errors),
        pcutoff_keypoints_total=len(all_errors_pcutoff),
        rmse_px=overall_stats["rmse"],
        rmse_pcutoff_px=pcutoff_stats["rmse"],
        mae_px=overall_stats["mae"],
        mae_pcutoff_px=pcutoff_stats["mae"],
        std_error_px=overall_stats["std"],
        p50_px=overall_stats["p50"],
        p75_px=overall_stats["p75"],
        p90_px=overall_stats["p90"],
        p95_px=overall_stats["p95"],
        max_error_px=overall_stats["max"],
        mean_oks=(
            float(sum(all_matched_oks) / len(all_matched_oks))
            if all_matched_oks
            else None
        ),
        oks_map=oks_map,
        oks_ap50=ap50,
        oks_ap75=ap75,
        oks_ar=oks_ar,
        keypoint_precision=kp_prec,
        keypoint_recall=kp_rec,
        keypoint_f1=kp_f1,
        instance_precision=inst_prec,
        instance_recall=inst_rec,
        instance_f1=inst_f1,
        swap_pairs_total=swap_pairs_total,
        swap_pairs_swapped=swap_pairs_swapped,
        swap_rate=(
            float(swap_pairs_swapped / swap_pairs_total)
            if swap_pairs_total > 0
            else None
        ),
        pck_px=pck_px_overall,
        pck_norm=pck_norm_overall,
        per_keypoint=tuple(per_kp_rows),
        worst_frames=worst,
        all_errors_px=tuple(float(v) for v in all_errors),
    )


def format_pose_eval_table(report: PoseEvaluationReport) -> str:
    """Format a compact CLI summary table for ``PoseEvaluationReport``."""

    def _fmt(val: Optional[float], suffix: str = "") -> str:
        if val is None or not math.isfinite(float(val)):
            return "n/a"
        return f"{float(val):.2f}{suffix}"

    lines = [
        "=== Pose Evaluation Summary ===",
        f"Frames evaluated : {report.frames_evaluated}/{report.frames_gt}",
        f"Instances matched: {report.matched_instances_total}/{report.gt_instances_total} (pred={report.pred_instances_total})",
        f"Keypoints matched: {report.matched_keypoints_total}/{report.gt_keypoints_total} (pcutoff={report.pcutoff:.2f} -> {report.pcutoff_keypoints_total})",
        f"RMSE (all / pcut): {_fmt(report.rmse_px, ' px')} / {_fmt(report.rmse_pcutoff_px, ' px')}",
        f"MAE  (all / pcut): {_fmt(report.mae_px, ' px')} / {_fmt(report.mae_pcutoff_px, ' px')}",
        f"Percentiles      : p50={_fmt(report.p50_px, 'px')}  p75={_fmt(report.p75_px, 'px')}  p90={_fmt(report.p90_px, 'px')}  p95={_fmt(report.p95_px, 'px')}",
        f"OKS mAP / AP50 / AP75 / AR: {report.oks_map:.4f} / {report.oks_ap50:.4f} / {report.oks_ap75:.4f} / {report.oks_ar:.4f}",
        f"KP Prec / Rec / F1: {report.keypoint_precision:.4f} / {report.keypoint_recall:.4f} / {report.keypoint_f1:.4f}",
    ]
    if report.swap_rate is not None:
        lines.append(
            f"L/R Swap Rate    : {report.swap_rate * 100.0:.2f}% ({report.swap_pairs_swapped}/{report.swap_pairs_total})"
        )
    if report.pck_px:
        pck_str = "  ".join(
            f"@{k}px={v * 100.0:.1f}%"
            for k, v in report.pck_px.items()
            if v is not None
        )
        lines.append(f"PCK (px)         : {pck_str}")
    if report.pck_norm:
        pck_n_str = "  ".join(
            f"@{k}={v * 100.0:.1f}%"
            for k, v in report.pck_norm.items()
            if v is not None
        )
        lines.append(f"PCK (scale-norm) : {pck_n_str}")

    if report.per_keypoint:
        lines.append("")
        lines.append(
            f"{'Keypoint':<20} {'N':>5} {'Recall':>7} {'RMSE':>8} {'p50':>8} {'p90':>8} {'p95':>8}"
        )
        lines.append("-" * 68)
        for row in report.per_keypoint:
            lines.append(
                f"{row.keypoint:<20} {row.matched_count:>5} {row.recall * 100.0:>6.1f}% "
                f"{_fmt(row.rmse_px):>8} {_fmt(row.p50_px):>8} {_fmt(row.p90_px):>8} {_fmt(row.p95_px):>8}"
            )
    return "\n".join(lines)


def _render_per_keypoint_svg(report: PoseEvaluationReport) -> str:
    rows = [r for r in report.per_keypoint if r.rmse_px is not None]
    if not rows:
        return "<p>No matched keypoints to chart.</p>"
    bar_h = 24
    pad_l, pad_r, pad_t, pad_b = 150, 50, 26, 24
    width = 680
    height = pad_t + pad_b + bar_h * len(rows)
    plot_w = width - pad_l - pad_r
    max_val = max(
        max((r.p90_px or 0.0, r.rmse_px or 0.0) for r in rows),
        default=(1.0,),
    )
    max_scale = max(
        1.0, float(max(max_val) if isinstance(max_val, tuple) else max_val) * 1.1
    )

    elems = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font-family:system-ui,sans-serif;">',
        f'<text x="{pad_l}" y="16" font-size="12" fill="#334155" font-weight="600">Per-Keypoint Localization Error (px): RMSE (blue) &amp; p90 (amber)</text>',
    ]
    for i, r in enumerate(rows):
        y = pad_t + i * bar_h
        rmse = float(r.rmse_px or 0.0)
        p90 = float(r.p90_px or 0.0)
        w_rmse = max(1.0, (rmse / max_scale) * plot_w)
        w_p90 = max(1.0, (p90 / max_scale) * plot_w)
        elems.append(
            f'<text x="{pad_l - 8}" y="{y + 15}" text-anchor="end" font-size="11" fill="#0f172a">{html.escape(r.keypoint)}</text>'
        )
        elems.append(
            f'<rect x="{pad_l}" y="{y + 3}" width="{w_p90:.1f}" height="16" rx="3" fill="#fbbf24" opacity="0.55"/>'
        )
        elems.append(
            f'<rect x="{pad_l}" y="{y + 5}" width="{w_rmse:.1f}" height="12" rx="3" fill="#0284c7"/>'
        )
        elems.append(
            f'<text x="{pad_l + max(w_rmse, w_p90) + 6:.1f}" y="{y + 15}" font-size="10" fill="#475569">{rmse:.1f} / {p90:.1f}px</text>'
        )
    elems.append("</svg>")
    return "\n".join(elems)


def _render_error_cdf_svg(report: PoseEvaluationReport) -> str:
    if not report.all_errors_px:
        return ""
    errs = np.sort(np.asarray(report.all_errors_px, dtype=np.float64))
    max_x = max(1.0, float(np.percentile(errs, 98)) * 1.1)
    width, height = 680, 220
    pad_l, pad_r, pad_t, pad_b = 54, 24, 24, 34
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    xs = np.linspace(0.0, max_x, 80)
    ys = [float(np.mean(errs <= x)) for x in xs]
    pts = " ".join(
        f"{pad_l + (x / max_x) * pw:.1f},{pad_t + (1.0 - y) * ph:.1f}"
        for x, y in zip(xs, ys)
    )
    return "\n".join(
        [
            f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font-family:system-ui,sans-serif;">',
            f'<text x="{pad_l}" y="16" font-size="12" fill="#334155" font-weight="600">Cumulative Localization Error Distribution (PCK Curve)</text>',
            f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#f8fafc" stroke="#cbd5e1"/>',
            f'<polyline fill="none" stroke="#0ea5e9" stroke-width="2.5" points="{pts}"/>',
            f'<text x="{pad_l}" y="{height - 10}" font-size="10" fill="#475569">0 px</text>',
            f'<text x="{pad_l + pw:.1f}" y="{height - 10}" text-anchor="end" font-size="10" fill="#475569">{max_x:.1f} px</text>',
            f'<text x="{pad_l - 6}" y="{pad_t + 10}" text-anchor="end" font-size="10" fill="#475569">100%</text>',
            f'<text x="{pad_l - 6}" y="{pad_t + ph}" text-anchor="end" font-size="10" fill="#475569">0%</text>',
            "</svg>",
        ]
    )


def write_pose_eval_report_bundle(
    report: PoseEvaluationReport,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write JSON, per-keypoint CSV, worst-frames CSV, and self-contained HTML report."""
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    json_path = out_root / "pose_eval_report.json"
    atomic_write_json(json_path, report.to_dict(), indent=2)

    kp_csv_path = out_root / "pose_eval_per_keypoint.csv"
    buf_kp = io.StringIO()
    kp_writer = csv.writer(buf_kp, lineterminator="\n")
    px_keys = list(report.pck_px.keys())
    norm_keys = list(report.pck_norm.keys())
    kp_writer.writerow(
        [
            "keypoint",
            "gt_count",
            "matched_count",
            "pcutoff_count",
            "recall",
            "rmse_px",
            "rmse_pcutoff_px",
            "mae_px",
            "p50_px",
            "p75_px",
            "p90_px",
            "p95_px",
        ]
        + [f"pck_{k}px" for k in px_keys]
        + [f"pck_norm_{k}" for k in norm_keys]
    )
    for r in report.per_keypoint:
        kp_writer.writerow(
            [
                r.keypoint,
                r.gt_count,
                r.matched_count,
                r.pcutoff_count,
                round(r.recall, 6),
                _round_opt(r.rmse_px),
                _round_opt(r.rmse_pcutoff_px),
                _round_opt(r.mae_px),
                _round_opt(r.p50_px),
                _round_opt(r.p75_px),
                _round_opt(r.p90_px),
                _round_opt(r.p95_px),
            ]
            + [_round_opt(r.pck_px.get(k), 6) for k in px_keys]
            + [_round_opt(r.pck_norm.get(k), 6) for k in norm_keys]
        )
    atomic_write_text(kp_csv_path, buf_kp.getvalue(), encoding="utf-8")

    wf_csv_path = out_root / "pose_eval_worst_frames.csv"
    buf_wf = io.StringIO()
    wf_writer = csv.writer(buf_wf, lineterminator="\n")
    wf_writer.writerow(
        [
            "frame_index",
            "video_index",
            "image_path",
            "gt_instances",
            "pred_instances",
            "matched_instances",
            "matched_keypoints",
            "missing_keypoints",
            "mean_error_px",
            "max_error_px",
            "mean_oks",
            "swaps",
        ]
    )
    for wf in report.worst_frames:
        wf_writer.writerow(
            [
                wf.frame_index,
                wf.video_index,
                wf.image_path or "",
                wf.gt_instances,
                wf.pred_instances,
                wf.matched_instances,
                wf.matched_keypoints,
                wf.missing_keypoints,
                _round_opt(wf.mean_error_px),
                _round_opt(wf.max_error_px),
                _round_opt(wf.mean_oks, 6),
                wf.swaps,
            ]
        )
    atomic_write_text(wf_csv_path, buf_wf.getvalue(), encoding="utf-8")

    def _f(v: Optional[float], suffix: str = "") -> str:
        return "n/a" if v is None else f"{v:.2f}{suffix}"

    kp_rows_html = "\n".join(
        f"<tr><td>{html.escape(r.keypoint)}</td><td>{r.matched_count}/{r.gt_count}</td>"
        f"<td>{r.recall * 100:.1f}%</td><td>{_f(r.rmse_px)}</td><td>{_f(r.rmse_pcutoff_px)}</td>"
        f"<td>{_f(r.p50_px)}</td><td>{_f(r.p90_px)}</td><td>{_f(r.p95_px)}</td></tr>"
        for r in report.per_keypoint
    )
    wf_rows_html = "\n".join(
        f"<tr><td>{wf.frame_index}</td><td>{html.escape(wf.image_path or '')}</td>"
        f"<td>{wf.matched_instances}/{wf.gt_instances}</td><td>{wf.missing_keypoints}</td>"
        f"<td>{_f(wf.mean_error_px)}</td><td>{_f(wf.max_error_px)}</td><td>{wf.swaps}</td></tr>"
        for wf in report.worst_frames[:25]
    )
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Annolid Pose Evaluation Report</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #0f172a; background: #ffffff; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }}
.card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; background: #f8fafc; }}
.card .label {{ font-size: 12px; color: #475569; }}
.card .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
</style>
</head>
<body>
<h1>Annolid Pose Evaluation Report</h1>
<p><strong>GT:</strong> {html.escape(report.gt_source)}<br/><strong>Pred:</strong> {html.escape(report.pred_source)}</p>
<div class="cards">
  <div class="card"><div class="label">RMSE (all / pcut={report.pcutoff:.2f})</div><div class="value">{_f(report.rmse_px, "px")} / {_f(report.rmse_pcutoff_px, "px")}</div></div>
  <div class="card"><div class="label">Median / p90 / p95</div><div class="value">{_f(report.p50_px)} / {_f(report.p90_px)} / {_f(report.p95_px)} px</div></div>
  <div class="card"><div class="label">OKS mAP / AP50</div><div class="value">{report.oks_map:.3f} / {report.oks_ap50:.3f}</div></div>
  <div class="card"><div class="label">Keypoint F1</div><div class="value">{report.keypoint_f1:.3f}</div></div>
  <div class="card"><div class="label">L/R Swap Rate</div><div class="value">{"n/a" if report.swap_rate is None else f"{report.swap_rate * 100:.1f}%"}</div></div>
</div>
{_render_per_keypoint_svg(report)}
{_render_error_cdf_svg(report)}
<h2>Per-Keypoint Metrics</h2>
<table>
<thead><tr><th>Keypoint</th><th>Matched</th><th>Recall</th><th>RMSE (px)</th><th>RMSE@pcut (px)</th><th>p50 (px)</th><th>p90 (px)</th><th>p95 (px)</th></tr></thead>
<tbody>{kp_rows_html}</tbody>
</table>
<h2>Worst Frames (Active-Learning Refinement Candidates)</h2>
<table>
<thead><tr><th>Frame</th><th>Image</th><th>Instances</th><th>Missing KPs</th><th>Mean Err (px)</th><th>Max Err (px)</th><th>L/R Swaps</th></tr></thead>
<tbody>{wf_rows_html}</tbody>
</table>
</body>
</html>
"""
    html_path = out_root / "pose_eval_report.html"
    atomic_write_text(html_path, html_text, encoding="utf-8")

    return {
        "json": str(json_path),
        "per_keypoint_csv": str(kp_csv_path),
        "worst_frames_csv": str(wf_csv_path),
        "html": str(html_path),
    }
