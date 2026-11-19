"""Inter-Rater Reliability (IRR) metrics for Annolid behavior event exports.

Provides deterministic time-unit rasterization and onset-tolerance matching across
two or more raters:
- Pairwise and mean-pairwise Cohen's kappa, observed/expected agreement, PABAK,
  prevalence/bias indices, bin-level F1 (positive agreement), and Jaccard IoU.
- Multi-rater Fleiss' kappa across 2+ raters.
- Session-level dominant-state multiclass Cohen's kappa (including idle background).
- Event-onset / point-event tolerance F1 matching within ``onset_tolerance_seconds``.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import io
from itertools import combinations
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from annolid.behavior.event_utils import DEFAULT_SUBJECT_LABEL, ParsedEventRow
from annolid.behavior.project_schema import ProjectSchema
from annolid.behavior.qc import (
    EthogramQCReport,
    PointBehaviorEvent,
    validate_behavior_csv,
    validate_behavior_events,
)
from annolid.behavior.time_budget import BehaviorInterval
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "IRRComputationError",
    "BehaviorIRRRow",
    "IRRDisagreementBout",
    "IRRReport",
    "compute_binary_cohens_kappa",
    "compute_fleiss_kappa",
    "compute_krippendorff_alpha_binary",
    "compute_onset_matching_f1",
    "compute_irr",
    "compute_irr_from_csvs",
    "format_irr_report_table",
    "write_irr_report_json",
    "write_irr_summary_csv",
    "write_irr_disagreements_csv",
]

_EPSILON = 1e-12
_IDLE_STATE_TOKEN = "__none__"


class IRRComputationError(Exception):
    """Raised when inter-rater reliability cannot be computed from inputs."""


@dataclass(frozen=True)
class IRRDisagreementBout:
    """Contiguous time window where raters disagreed on a behavior."""

    subject: str
    behavior: str
    start_s: float
    end_s: float
    duration_s: float
    raters_active: Tuple[str, ...]
    raters_inactive: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "behavior": self.behavior,
            "start_s": round(float(self.start_s), 6),
            "end_s": round(float(self.end_s), 6),
            "duration_s": round(float(self.duration_s), 6),
            "raters_active": list(self.raters_active),
            "raters_inactive": list(self.raters_inactive),
        }


@dataclass(frozen=True)
class BehaviorIRRRow:
    """Reliability statistics for a single subject/behavior series."""

    subject: str
    behavior: str
    total_bins: int
    cohens_kappa: float
    fleiss_kappa: float
    pabak: float
    observed_agreement: float
    expected_agreement: float
    prevalence_index: float
    bias_index: float
    positive_agreement: float
    iou: float
    onset_f1: float
    rater_durations: Dict[str, float]
    rater_event_counts: Dict[str, int]
    krippendorff_alpha: float = 1.0
    cohens_kappa_ci95_low: float = 1.0
    cohens_kappa_ci95_high: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "behavior": self.behavior,
            "total_bins": int(self.total_bins),
            "cohens_kappa": round(float(self.cohens_kappa), 6),
            "cohens_kappa_ci95_low": round(float(self.cohens_kappa_ci95_low), 6),
            "cohens_kappa_ci95_high": round(float(self.cohens_kappa_ci95_high), 6),
            "fleiss_kappa": round(float(self.fleiss_kappa), 6),
            "krippendorff_alpha": round(float(self.krippendorff_alpha), 6),
            "pabak": round(float(self.pabak), 6),
            "observed_agreement": round(float(self.observed_agreement), 6),
            "expected_agreement": round(float(self.expected_agreement), 6),
            "prevalence_index": round(float(self.prevalence_index), 6),
            "bias_index": round(float(self.bias_index), 6),
            "positive_agreement": round(float(self.positive_agreement), 6),
            "iou": round(float(self.iou), 6),
            "onset_f1": round(float(self.onset_f1), 6),
            "rater_durations": {
                k: round(float(v), 6) for k, v in self.rater_durations.items()
            },
            "rater_event_counts": {
                k: int(v) for k, v in self.rater_event_counts.items()
            },
        }


@dataclass(frozen=True)
class IRRReport:
    """Complete Inter-Rater Reliability report across 2+ raters."""

    raters: Tuple[str, ...]
    bin_seconds: float
    onset_tolerance_seconds: float
    window_start: float
    window_end: float
    total_bins: int
    pool_subjects: bool
    rows: Tuple[BehaviorIRRRow, ...]
    macro_cohens_kappa: float
    macro_fleiss_kappa: float
    macro_pabak: float
    macro_onset_f1: float
    multiclass_cohens_kappa: float
    rater_qc_issue_counts: Dict[str, Dict[str, int]]
    macro_krippendorff_alpha: float = 1.0
    disagreements: Tuple[IRRDisagreementBout, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "behavior_irr_report/v1",
            "raters": list(self.raters),
            "bin_seconds": float(self.bin_seconds),
            "onset_tolerance_seconds": float(self.onset_tolerance_seconds),
            "window_start": round(float(self.window_start), 6),
            "window_end": round(float(self.window_end), 6),
            "total_bins": int(self.total_bins),
            "pool_subjects": bool(self.pool_subjects),
            "macro_cohens_kappa": round(float(self.macro_cohens_kappa), 6),
            "macro_fleiss_kappa": round(float(self.macro_fleiss_kappa), 6),
            "macro_krippendorff_alpha": round(float(self.macro_krippendorff_alpha), 6),
            "macro_pabak": round(float(self.macro_pabak), 6),
            "macro_onset_f1": round(float(self.macro_onset_f1), 6),
            "multiclass_cohens_kappa": round(float(self.multiclass_cohens_kappa), 6),
            "rater_qc_issue_counts": {
                name: dict(counts)
                for name, counts in self.rater_qc_issue_counts.items()
            },
            "rows": [row.to_dict() for row in self.rows],
            "disagreements": [d.to_dict() for d in self.disagreements],
        }


def _safe_kappa(observed: float, expected: float) -> float:
    denom = 1.0 - expected
    if denom <= _EPSILON:
        return 1.0 if observed >= 1.0 - _EPSILON else 0.0
    value = (observed - expected) / denom
    return max(-1.0, min(1.0, value))


def compute_binary_cohens_kappa(
    seq_a: Sequence[int],
    seq_b: Sequence[int],
) -> Dict[str, float]:
    """Compute pairwise Cohen's kappa and binary agreement statistics."""
    n = len(seq_a)
    if n != len(seq_b):
        raise ValueError("Binary rater sequences must have equal length.")
    if n == 0:
        return {
            "cohens_kappa": 1.0,
            "pabak": 1.0,
            "observed_agreement": 1.0,
            "expected_agreement": 1.0,
            "prevalence_index": 0.0,
            "bias_index": 0.0,
            "positive_agreement": 1.0,
            "iou": 1.0,
        }

    a_arr = np.asarray(seq_a).astype(bool, copy=False)
    b_arr = np.asarray(seq_b).astype(bool, copy=False)
    n11 = int(np.count_nonzero(a_arr & b_arr))
    a_ones = int(np.count_nonzero(a_arr))
    b_ones = int(np.count_nonzero(b_arr))
    n10 = a_ones - n11
    n01 = b_ones - n11
    n00 = n - n11 - n10 - n01

    inv_n = 1.0 / float(n)
    p_o = (n11 + n00) * inv_n
    p_a1 = (n11 + n10) * inv_n
    p_b1 = (n11 + n01) * inv_n
    p_e = (p_a1 * p_b1) + ((1.0 - p_a1) * (1.0 - p_b1))

    kappa = _safe_kappa(p_o, p_e)
    pabak = 2.0 * p_o - 1.0
    prevalence_index = abs(n11 - n00) * inv_n
    bias_index = abs(n10 - n01) * inv_n

    union = n11 + n10 + n01
    iou = (float(n11) / float(union)) if union > 0 else 1.0
    denom_f1 = 2 * n11 + n10 + n01
    pos_agreement = (float(2 * n11) / float(denom_f1)) if denom_f1 > 0 else 1.0

    return {
        "cohens_kappa": kappa,
        "pabak": pabak,
        "observed_agreement": p_o,
        "expected_agreement": p_e,
        "prevalence_index": prevalence_index,
        "bias_index": bias_index,
        "positive_agreement": pos_agreement,
        "iou": iou,
    }


def compute_fleiss_kappa(rater_sequences: Sequence[Sequence[int]]) -> float:
    """Compute multi-rater Fleiss' kappa across ``R >= 2`` binary sequences."""
    num_raters = len(rater_sequences)
    if num_raters < 2:
        raise ValueError("Fleiss' kappa requires at least 2 raters.")
    n_bins = len(rater_sequences[0])
    if any(len(seq) != n_bins for seq in rater_sequences):
        raise ValueError("All rater sequences must have equal length.")
    if n_bins == 0:
        return 1.0

    pairs_per_bin = float(num_raters * (num_raters - 1))
    mat = np.asarray(rater_sequences).astype(bool, copy=False)
    ones = mat.sum(axis=0, dtype=np.int64)
    zeros = num_raters - ones
    total_ones = int(ones.sum())
    sum_p_i = (
        float(np.sum(ones * (ones - 1) + zeros * (zeros - 1), dtype=np.float64))
        / pairs_per_bin
    )

    p_bar = sum_p_i / float(n_bins)
    p_1 = float(total_ones) / float(n_bins * num_raters)
    p_0 = 1.0 - p_1
    p_e_bar = (p_1 * p_1) + (p_0 * p_0)
    return _safe_kappa(p_bar, p_e_bar)


def compute_krippendorff_alpha_binary(
    rater_sequences: Sequence[Sequence[int]],
) -> float:
    """Compute nominal Krippendorff's alpha across ``R >= 2`` binary sequences."""
    num_raters = len(rater_sequences)
    if num_raters < 2:
        raise ValueError("Krippendorff's alpha requires at least 2 raters.")
    n_bins = len(rater_sequences[0])
    if any(len(seq) != n_bins for seq in rater_sequences):
        raise ValueError("All rater sequences must have equal length.")
    if n_bins == 0:
        return 1.0

    mat = np.asarray(rater_sequences).astype(bool, copy=False)
    ones_per_bin = mat.sum(axis=0, dtype=np.int64)
    zeros_per_bin = num_raters - ones_per_bin
    total_vals = n_bins * num_raters
    total_ones = int(ones_per_bin.sum())
    total_zeros = total_vals - total_ones
    if total_ones == 0 or total_zeros == 0 or total_vals <= 1:
        return 1.0

    # Observed disagreement D_o across within-unit rater pairs
    d_o = float(np.sum(2.0 * ones_per_bin * zeros_per_bin)) / float(
        n_bins * num_raters * (num_raters - 1)
    )
    # Expected disagreement D_e across all coded values
    d_e = float(2.0 * total_ones * total_zeros) / float(total_vals * (total_vals - 1))
    if d_e <= _EPSILON:
        return 1.0 if d_o <= _EPSILON else 0.0
    return max(-1.0, min(1.0, 1.0 - (d_o / d_e)))


def _bootstrap_mean_cohens_kappa_ci95(
    per_rater_grids: Mapping[str, Sequence[int]],
    rater_pairs: Sequence[Tuple[str, str]],
    *,
    bin_seconds: float = 0.1,
    block_seconds: float = 5.0,
    n_boot: int = 200,
    seed: int = 42,
) -> Tuple[float, float]:
    if not rater_pairs:
        return 1.0, 1.0
    first_seq = per_rater_grids[rater_pairs[0][0]]
    n_bins = len(first_seq)
    if n_bins <= 1:
        val = mean(
            compute_binary_cohens_kappa(per_rater_grids[ra], per_rater_grids[rb])[
                "cohens_kappa"
            ]
            for ra, rb in rater_pairs
        )
        return val, val

    arrays = {
        name: np.asarray(seq, dtype=bool) for name, seq in per_rater_grids.items()
    }
    rng = np.random.default_rng(seed)
    if n_bins >= 8:
        target_block = max(
            1, int(round(float(block_seconds) / max(float(bin_seconds), 1e-6)))
        )
        block_len = min(max(1, n_bins // 4), target_block)
    else:
        block_len = 1
    if block_len <= 1:
        boot_indices = rng.integers(0, n_bins, size=(n_boot, n_bins))
    else:
        n_blocks = int(math.ceil(float(n_bins) / float(block_len)))
        starts = rng.integers(0, n_bins, size=(n_boot, n_blocks))
        offsets = np.arange(block_len, dtype=np.int64)
        boot_indices = ((starts[:, :, None] + offsets[None, None, :]) % n_bins).reshape(
            n_boot, n_blocks * block_len
        )[:, :n_bins]
    pair_boot_kappas = np.zeros(n_boot, dtype=np.float64)
    inv_n = 1.0 / float(n_bins)

    for ra, rb in rater_pairs:
        a_boot = arrays[ra][boot_indices]
        b_boot = arrays[rb][boot_indices]
        n11 = np.sum(a_boot & b_boot, axis=1, dtype=np.float64)
        a1 = np.sum(a_boot, axis=1, dtype=np.float64)
        b1 = np.sum(b_boot, axis=1, dtype=np.float64)
        n10 = a1 - n11
        n01 = b1 - n11
        n00 = float(n_bins) - n11 - n10 - n01
        p_o = (n11 + n00) * inv_n
        p_a1 = a1 * inv_n
        p_b1 = b1 * inv_n
        p_e = (p_a1 * p_b1) + ((1.0 - p_a1) * (1.0 - p_b1))
        denom = 1.0 - p_e
        k_vec = np.where(
            denom <= _EPSILON,
            np.where(p_o >= 1.0 - _EPSILON, 1.0, 0.0),
            np.clip((p_o - p_e) / np.maximum(denom, _EPSILON), -1.0, 1.0),
        )
        pair_boot_kappas += k_vec

    pair_boot_kappas /= float(len(rater_pairs))
    lo = float(np.percentile(pair_boot_kappas, 2.5))
    hi = float(np.percentile(pair_boot_kappas, 97.5))
    return lo, hi


def _extract_disagreement_bouts(
    *,
    subject: str,
    behavior: str,
    rater_names: Sequence[str],
    per_rater_grids: Mapping[str, Sequence[int]],
    window_start: float,
    window_end: float,
    bin_seconds: float,
) -> List[IRRDisagreementBout]:
    if not rater_names:
        return []
    n_bins = len(per_rater_grids[rater_names[0]])
    if n_bins == 0:
        return []
    mat = np.asarray([per_rater_grids[r] for r in rater_names], dtype=bool)
    ones = mat.sum(axis=0)
    num_raters = len(rater_names)
    disagreed_mask = (ones > 0) & (ones < num_raters)
    if not np.any(disagreed_mask):
        return []

    bouts: List[IRRDisagreementBout] = []
    idx = 0
    while idx < n_bins:
        if not disagreed_mask[idx]:
            idx += 1
            continue
        active_sig = tuple(
            rater_names[r_i] for r_i in range(num_raters) if mat[r_i, idx]
        )
        end_idx = idx + 1
        while end_idx < n_bins and disagreed_mask[end_idx]:
            next_sig = tuple(
                rater_names[r_i] for r_i in range(num_raters) if mat[r_i, end_idx]
            )
            if next_sig != active_sig:
                break
            end_idx += 1
        s_time = window_start + float(idx) * bin_seconds
        e_time = min(window_end, window_start + float(end_idx) * bin_seconds)
        inactive_sig = tuple(r for r in rater_names if r not in active_sig)
        bouts.append(
            IRRDisagreementBout(
                subject=subject,
                behavior=behavior,
                start_s=s_time,
                end_s=e_time,
                duration_s=max(0.0, e_time - s_time),
                raters_active=active_sig,
                raters_inactive=inactive_sig,
            )
        )
        idx = end_idx
    return bouts


def _compute_multiclass_cohens_kappa_pair(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float:
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("Multiclass label sequences must have equal length.")
    if n == 0:
        return 1.0

    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = float(matches) / float(n)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    all_keys = set(counts_a) | set(counts_b)
    inv_n2 = 1.0 / float(n * n)
    p_e = sum(float(counts_a[k] * counts_b[k]) for k in all_keys) * inv_n2
    return _safe_kappa(p_o, p_e)


def compute_onset_matching_f1(
    onsets_a: Sequence[float],
    onsets_b: Sequence[float],
    *,
    tolerance_seconds: float,
) -> float:
    """Maximum-cardinality 1D timestamp matching F1 within ``tolerance_seconds``."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")
    sorted_a = sorted(float(t) for t in onsets_a)
    sorted_b = sorted(float(t) for t in onsets_b)
    if not sorted_a and not sorted_b:
        return 1.0
    if not sorted_a or not sorted_b:
        return 0.0

    idx_a = 0
    idx_b = 0
    matched = 0
    tol = float(tolerance_seconds) + 1e-9
    while idx_a < len(sorted_a) and idx_b < len(sorted_b):
        diff = sorted_a[idx_a] - sorted_b[idx_b]
        if abs(diff) <= tol:
            matched += 1
            idx_a += 1
            idx_b += 1
        elif sorted_a[idx_a] < sorted_b[idx_b] - tol:
            idx_a += 1
        else:
            idx_b += 1

    return float(2 * matched) / float(len(sorted_a) + len(sorted_b))


def _disambiguate_rater_names(raw_names: Sequence[str]) -> Tuple[str, ...]:
    seen: Dict[str, int] = {}
    resolved: List[str] = []
    for idx, raw in enumerate(raw_names, start=1):
        base = str(raw or "").strip() or f"rater_{idx}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        resolved.append(base if count == 1 else f"{base}_{count}")
    return tuple(resolved)


def _rasterize_series(
    intervals: Sequence[BehaviorInterval],
    points: Sequence[PointBehaviorEvent],
    *,
    window_start: float,
    window_end: float,
    bin_seconds: float,
    n_bins: int,
) -> List[int]:
    if n_bins <= 0:
        return []
    grid = np.zeros(n_bins, dtype=np.int8)

    for interval in intervals:
        s = max(window_start, interval.start)
        e = min(window_end, interval.end)
        if e < s - 1e-9:
            continue
        if e - s <= 1e-9:
            # Zero-duration interval: treat like a point at s.
            idx = min(
                n_bins - 1, max(0, int(math.floor((s - window_start) / bin_seconds)))
            )
            grid[idx] = 1
            continue

        start_idx = min(
            n_bins - 1, max(0, int(math.floor((s - window_start) / bin_seconds)))
        )
        end_idx = min(
            n_bins - 1,
            max(0, int(math.floor((e - window_start - 1e-9) / bin_seconds))),
        )
        marked_any = False
        if end_idx - start_idx > 1:
            grid[start_idx + 1 : end_idx] = 1
            marked_any = True

        boundary_indices = (
            (start_idx,) if start_idx == end_idx else (start_idx, end_idx)
        )
        for idx in boundary_indices:
            b0 = window_start + idx * bin_seconds
            b1 = min(window_end, b0 + bin_seconds)
            bin_width = max(1e-9, b1 - b0)
            overlap = min(e, b1) - max(s, b0)
            if overlap >= 0.5 * bin_width - 1e-9:
                grid[idx] = 1
                marked_any = True
        if not marked_any:
            mid = 0.5 * (s + e)
            mid_idx = min(
                n_bins - 1,
                max(0, int(math.floor((mid - window_start) / bin_seconds))),
            )
            grid[mid_idx] = 1

    if points:
        ts_arr = np.fromiter(
            (pt.timestamp for pt in points), dtype=np.float64, count=len(points)
        )
        valid = (ts_arr >= window_start - 1e-9) & (ts_arr <= window_end + 1e-9)
        if np.any(valid):
            pt_idx = np.clip(
                np.floor((ts_arr[valid] - window_start) / bin_seconds).astype(np.int64),
                0,
                n_bins - 1,
            )
            grid[pt_idx] = 1

    return grid.tolist()


def compute_irr(
    rater_inputs: Sequence[
        Tuple[
            str,
            EthogramQCReport | Iterable[ParsedEventRow | Mapping[str, object]],
        ]
    ],
    *,
    bin_seconds: float = 0.1,
    onset_tolerance_seconds: float = 0.5,
    schema: Optional[ProjectSchema] = None,
    subject: Optional[str] = None,
    pool_subjects: bool = False,
    session_start: Optional[float] = None,
    session_end: Optional[float] = None,
    session_duration: Optional[float] = None,
    include_empty_schema_behaviors: bool = False,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> IRRReport:
    """Compute Inter-Rater Reliability across 2 or more raters."""
    if len(rater_inputs) < 2:
        raise IRRComputationError(
            "Inter-rater reliability requires at least 2 rater inputs."
        )
    if bin_seconds <= 0 or not math.isfinite(bin_seconds):
        raise IRRComputationError("bin_seconds must be a positive finite number.")
    if onset_tolerance_seconds < 0 or not math.isfinite(onset_tolerance_seconds):
        raise IRRComputationError(
            "onset_tolerance_seconds must be a non-negative finite number."
        )
    if session_duration is not None and (
        session_duration <= 0 or not math.isfinite(session_duration)
    ):
        raise IRRComputationError("session_duration must be a positive finite number.")

    rater_names = _disambiguate_rater_names([name for name, _ in rater_inputs])
    qc_reports: Dict[str, EthogramQCReport] = {}
    for resolved_name, (_raw_name, source) in zip(rater_names, rater_inputs):
        if isinstance(source, EthogramQCReport):
            qc_reports[resolved_name] = source
        else:
            qc_reports[resolved_name] = validate_behavior_events(
                source,
                schema=schema,
                default_subject=default_subject,
            )

    subject_filter = (
        str(subject).strip() if subject is not None and str(subject).strip() else None
    )

    all_timestamps: List[float] = []
    observed_subjects: Set[str] = set()
    observed_behaviors: Set[str] = set()

    filtered_intervals: Dict[str, List[BehaviorInterval]] = {
        name: [] for name in rater_names
    }
    filtered_points: Dict[str, List[PointBehaviorEvent]] = {
        name: [] for name in rater_names
    }

    for rater_name in rater_names:
        report = qc_reports[rater_name]
        for iv in report.intervals:
            if subject_filter is not None and iv.subject != subject_filter:
                continue
            target_subject = "ALL" if pool_subjects else iv.subject
            norm_iv = BehaviorInterval(
                subject=target_subject,
                behavior=iv.behavior,
                start=iv.start,
                end=iv.end,
            )
            filtered_intervals[rater_name].append(norm_iv)
            observed_subjects.add(target_subject)
            observed_behaviors.add(iv.behavior)
            all_timestamps.append(iv.start)
            all_timestamps.append(iv.end)

        for pt in report.point_events:
            if subject_filter is not None and pt.subject != subject_filter:
                continue
            target_subject = "ALL" if pool_subjects else pt.subject
            norm_pt = PointBehaviorEvent(
                subject=target_subject,
                behavior=pt.behavior,
                timestamp=pt.timestamp,
                line_number=pt.line_number,
            )
            filtered_points[rater_name].append(norm_pt)
            observed_subjects.add(target_subject)
            observed_behaviors.add(pt.behavior)
            all_timestamps.append(pt.timestamp)

    if include_empty_schema_behaviors and schema is not None:
        for beh in schema.behaviors:
            if beh.code:
                observed_behaviors.add(beh.code)
        if not observed_subjects:
            if pool_subjects:
                observed_subjects.add("ALL")
            elif subject_filter is not None:
                observed_subjects.add(subject_filter)
            elif schema.subjects:
                observed_subjects.update(
                    subj.name or subj.id for subj in schema.subjects
                )
            else:
                observed_subjects.add(default_subject)

    win_start = (
        float(session_start)
        if session_start is not None
        else (min(all_timestamps) if all_timestamps else 0.0)
    )
    if session_end is not None:
        win_end = float(session_end)
    elif session_duration is not None:
        win_end = win_start + float(session_duration)
    else:
        win_end = max(all_timestamps) if all_timestamps else win_start

    if win_end < win_start - 1e-9:
        raise IRRComputationError(
            f"session_end ({win_end:.3f}s) must not precede session_start ({win_start:.3f}s)."
        )

    span = max(0.0, win_end - win_start)
    if span <= 1e-9:
        n_bins = 1 if all_timestamps else 0
    else:
        n_bins = max(1, int(math.ceil((span - 1e-12) / bin_seconds)))

    rater_qc_issue_counts = {
        name: {
            "errors": qc_reports[name].error_count,
            "warnings": qc_reports[name].warning_count,
        }
        for name in rater_names
    }

    if not observed_behaviors or not observed_subjects or n_bins == 0:
        return IRRReport(
            raters=rater_names,
            bin_seconds=float(bin_seconds),
            onset_tolerance_seconds=float(onset_tolerance_seconds),
            window_start=win_start,
            window_end=win_end,
            total_bins=n_bins,
            pool_subjects=bool(pool_subjects),
            rows=(),
            macro_cohens_kappa=1.0,
            macro_fleiss_kappa=1.0,
            macro_pabak=1.0,
            macro_onset_f1=1.0,
            multiclass_cohens_kappa=1.0,
            rater_qc_issue_counts=rater_qc_issue_counts,
        )

    subjects_ordered = sorted(observed_subjects)
    behaviors_ordered = sorted(observed_behaviors)
    rater_pairs = list(combinations(rater_names, 2))

    # Store rasterized grids per (rater, subject, behavior) to reuse for multiclass kappa.
    grids: Dict[Tuple[str, str, str], List[int]] = {}
    rows_out: List[BehaviorIRRRow] = []
    disagreements_all: List[IRRDisagreementBout] = []

    for subj in subjects_ordered:
        for beh in behaviors_ordered:
            # Skip combinations that never appeared in any rater unless include_empty_schema_behaviors
            has_any = include_empty_schema_behaviors
            per_rater_ivs: Dict[str, List[BehaviorInterval]] = {}
            per_rater_pts: Dict[str, List[PointBehaviorEvent]] = {}
            for rater_name in rater_names:
                ivs = [
                    iv
                    for iv in filtered_intervals[rater_name]
                    if iv.subject == subj and iv.behavior == beh
                ]
                pts = [
                    pt
                    for pt in filtered_points[rater_name]
                    if pt.subject == subj and pt.behavior == beh
                ]
                if ivs or pts:
                    has_any = True
                per_rater_ivs[rater_name] = ivs
                per_rater_pts[rater_name] = pts

            if not has_any:
                continue

            per_rater_grids: Dict[str, List[int]] = {}
            durations: Dict[str, float] = {}
            event_counts: Dict[str, int] = {}
            onsets_by_rater: Dict[str, List[float]] = {}

            for rater_name in rater_names:
                ivs = per_rater_ivs[rater_name]
                pts = per_rater_pts[rater_name]
                grid = _rasterize_series(
                    ivs,
                    pts,
                    window_start=win_start,
                    window_end=win_end,
                    bin_seconds=bin_seconds,
                    n_bins=n_bins,
                )
                per_rater_grids[rater_name] = grid
                grids[(rater_name, subj, beh)] = grid

                clipped_dur = 0.0
                for iv in ivs:
                    clipped_dur += max(
                        0.0, min(win_end, iv.end) - max(win_start, iv.start)
                    )
                durations[rater_name] = clipped_dur
                event_counts[rater_name] = len(ivs) + len(pts)
                onsets = [
                    iv.start
                    for iv in ivs
                    if win_start - 1e-9 <= iv.start <= win_end + 1e-9
                ] + [
                    pt.timestamp
                    for pt in pts
                    if win_start - 1e-9 <= pt.timestamp <= win_end + 1e-9
                ]
                onsets_by_rater[rater_name] = sorted(onsets)

            pair_stats = [
                compute_binary_cohens_kappa(per_rater_grids[ra], per_rater_grids[rb])
                for ra, rb in rater_pairs
            ]
            pair_onset_f1 = [
                compute_onset_matching_f1(
                    onsets_by_rater[ra],
                    onsets_by_rater[rb],
                    tolerance_seconds=onset_tolerance_seconds,
                )
                for ra, rb in rater_pairs
            ]
            fleiss = compute_fleiss_kappa([per_rater_grids[r] for r in rater_names])
            kripp_alpha = compute_krippendorff_alpha_binary(
                [per_rater_grids[r] for r in rater_names]
            )
            ci_lo, ci_hi = _bootstrap_mean_cohens_kappa_ci95(
                per_rater_grids, rater_pairs, bin_seconds=bin_seconds
            )
            disagreements_all.extend(
                _extract_disagreement_bouts(
                    subject=subj,
                    behavior=beh,
                    rater_names=rater_names,
                    per_rater_grids=per_rater_grids,
                    window_start=win_start,
                    window_end=win_end,
                    bin_seconds=bin_seconds,
                )
            )

            rows_out.append(
                BehaviorIRRRow(
                    subject=subj,
                    behavior=beh,
                    total_bins=n_bins,
                    cohens_kappa=mean(s["cohens_kappa"] for s in pair_stats),
                    fleiss_kappa=fleiss,
                    pabak=mean(s["pabak"] for s in pair_stats),
                    observed_agreement=mean(
                        s["observed_agreement"] for s in pair_stats
                    ),
                    expected_agreement=mean(
                        s["expected_agreement"] for s in pair_stats
                    ),
                    prevalence_index=mean(s["prevalence_index"] for s in pair_stats),
                    bias_index=mean(s["bias_index"] for s in pair_stats),
                    positive_agreement=mean(
                        s["positive_agreement"] for s in pair_stats
                    ),
                    iou=mean(s["iou"] for s in pair_stats),
                    onset_f1=mean(pair_onset_f1),
                    rater_durations=durations,
                    rater_event_counts=event_counts,
                    krippendorff_alpha=kripp_alpha,
                    cohens_kappa_ci95_low=ci_lo,
                    cohens_kappa_ci95_high=ci_hi,
                )
            )

    # Dominant-state multiclass sequence across (subject, bin) units per rater.
    # Order overlapping behaviors by ascending overall prevalence (rarer/briefer
    # behaviors take precedence over broad background states; ties break by name).
    multiclass_sequences: Dict[str, List[str]] = {name: [] for name in rater_names}
    for subj in subjects_ordered:
        subj_behaviors = [
            beh
            for beh in behaviors_ordered
            if any((r, subj, beh) in grids for r in rater_names)
        ]
        if not subj_behaviors:
            continue
        beh_prevalence = {
            beh: sum(sum(grids.get((r, subj, beh), ())) for r in rater_names)
            for beh in subj_behaviors
        }
        prioritized_behaviors = sorted(
            subj_behaviors, key=lambda b: (beh_prevalence[b], b)
        )
        for rater_name in rater_names:
            seq = multiclass_sequences[rater_name]
            for bin_idx in range(n_bins):
                chosen = _IDLE_STATE_TOKEN
                for beh in prioritized_behaviors:
                    grid = grids.get((rater_name, subj, beh))
                    if grid is not None and grid[bin_idx]:
                        chosen = beh
                        break
                seq.append(chosen)

    multiclass_pair_kappas = [
        _compute_multiclass_cohens_kappa_pair(
            multiclass_sequences[ra], multiclass_sequences[rb]
        )
        for ra, rb in rater_pairs
    ]
    multiclass_kappa = mean(multiclass_pair_kappas) if multiclass_pair_kappas else 1.0

    if rows_out:
        macro_cohen = mean(r.cohens_kappa for r in rows_out)
        macro_fleiss = mean(r.fleiss_kappa for r in rows_out)
        macro_alpha = mean(r.krippendorff_alpha for r in rows_out)
        macro_pabak = mean(r.pabak for r in rows_out)
        macro_onset = mean(r.onset_f1 for r in rows_out)
    else:
        macro_cohen = 1.0
        macro_fleiss = 1.0
        macro_alpha = 1.0
        macro_pabak = 1.0
        macro_onset = 1.0

    disagreements_sorted = tuple(
        sorted(
            disagreements_all,
            key=lambda d: (-d.duration_s, d.start_s, d.subject, d.behavior),
        )
    )

    return IRRReport(
        raters=rater_names,
        bin_seconds=float(bin_seconds),
        onset_tolerance_seconds=float(onset_tolerance_seconds),
        window_start=win_start,
        window_end=win_end,
        total_bins=n_bins,
        pool_subjects=bool(pool_subjects),
        rows=tuple(rows_out),
        macro_cohens_kappa=macro_cohen,
        macro_fleiss_kappa=macro_fleiss,
        macro_pabak=macro_pabak,
        macro_onset_f1=macro_onset,
        multiclass_cohens_kappa=multiclass_kappa,
        rater_qc_issue_counts=rater_qc_issue_counts,
        macro_krippendorff_alpha=macro_alpha,
        disagreements=disagreements_sorted,
    )


def compute_irr_from_csvs(
    csv_paths: Sequence[Path | str],
    *,
    rater_names: Optional[Sequence[str]] = None,
    bin_seconds: float = 0.1,
    onset_tolerance_seconds: float = 0.5,
    schema: Optional[ProjectSchema] = None,
    subject: Optional[str] = None,
    pool_subjects: bool = False,
    session_start: Optional[float] = None,
    session_end: Optional[float] = None,
    session_duration: Optional[float] = None,
    include_empty_schema_behaviors: bool = False,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> IRRReport:
    """Load 2+ behavior event CSV files and compute Inter-Rater Reliability."""
    if len(csv_paths) < 2:
        raise IRRComputationError(
            "compute_irr_from_csvs requires at least 2 CSV file paths."
        )
    resolved_paths = [Path(p) for p in csv_paths]
    if rater_names is not None:
        if len(rater_names) != len(resolved_paths):
            raise IRRComputationError(
                f"Expected {len(resolved_paths)} rater names, received {len(rater_names)}."
            )
        raw_names = [str(name) for name in rater_names]
    else:
        raw_names = [p.stem for p in resolved_paths]

    try:
        rater_inputs = [
            (
                name,
                validate_behavior_csv(
                    path,
                    schema=schema,
                    default_subject=default_subject,
                ),
            )
            for name, path in zip(raw_names, resolved_paths)
        ]
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise IRRComputationError(str(exc)) from exc

    return compute_irr(
        rater_inputs,
        bin_seconds=bin_seconds,
        onset_tolerance_seconds=onset_tolerance_seconds,
        schema=schema,
        subject=subject,
        pool_subjects=pool_subjects,
        session_start=session_start,
        session_end=session_end,
        session_duration=session_duration,
        include_empty_schema_behaviors=include_empty_schema_behaviors,
        default_subject=default_subject,
    )


def format_irr_report_table(report: IRRReport) -> str:
    """Render an :class:`IRRReport` as an aligned plain-text summary table."""
    lines = [
        f"Inter-Rater Reliability ({len(report.raters)} raters: {', '.join(report.raters)})",
        (
            f"Window: {report.window_start:.3f}s - {report.window_end:.3f}s  "
            f"|  Bin: {report.bin_seconds:.3f}s ({report.total_bins} bins)  "
            f"|  Onset Tol: {report.onset_tolerance_seconds:.3f}s"
        ),
        (
            f"Macro Cohen's Kappa: {report.macro_cohens_kappa:.4f}  "
            f"|  Multiclass Kappa: {report.multiclass_cohens_kappa:.4f}  "
            f"|  Macro Fleiss' Kappa: {report.macro_fleiss_kappa:.4f}  "
            f"|  Macro PABAK: {report.macro_pabak:.4f}"
        ),
    ]

    qc_notes = [
        f"{name} ({counts['errors']} errors, {counts['warnings']} warnings)"
        for name, counts in report.rater_qc_issue_counts.items()
        if counts.get("errors", 0) or counts.get("warnings", 0)
    ]
    if qc_notes:
        lines.append("Rater QC Issues: " + "; ".join(qc_notes))

    if not report.rows:
        lines.append("No behavior events matched the evaluation window.")
        return "\n".join(lines)

    lines.append("")
    headers = [
        "Subject",
        "Behavior",
        "Cohen k",
        "Fleiss k",
        "PABAK",
        "Agree %",
        "Bin F1",
        "IoU",
        "Onset F1",
    ]
    str_rows: List[List[str]] = [
        [
            row.subject,
            row.behavior,
            f"{row.cohens_kappa:.4f}",
            f"{row.fleiss_kappa:.4f}",
            f"{row.pabak:.4f}",
            f"{row.observed_agreement * 100.0:.2f}",
            f"{row.positive_agreement:.4f}",
            f"{row.iou:.4f}",
            f"{row.onset_f1:.4f}",
        ]
        for row in report.rows
    ]

    col_widths = [len(h) for h in headers]
    for r in str_rows:
        for idx, val in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(val))

    def _fmt(values: Sequence[str]) -> str:
        return "  ".join(
            values[idx].ljust(col_widths[idx]) for idx in range(len(headers))
        )

    lines.append(_fmt(headers))
    lines.append(_fmt(["-" * len(h) for h in headers]))
    lines.extend(_fmt(r) for r in str_rows)
    return "\n".join(lines)


def write_irr_report_json(report: IRRReport, path: Path | str) -> None:
    """Atomically write an :class:`IRRReport` as JSON."""
    atomic_write_json(Path(path), report.to_dict(), indent=2, ensure_ascii=False)


def write_irr_summary_csv(report: IRRReport, path: Path | str) -> None:
    """Atomically write per-behavior IRR rows to CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = [
        "Subject",
        "Behavior",
        "TotalBins",
        "CohensKappa",
        "FleissKappa",
        "PABAK",
        "ObservedAgreement",
        "ExpectedAgreement",
        "PrevalenceIndex",
        "BiasIndex",
        "PositiveAgreementF1",
        "IoU",
        "OnsetF1",
    ]
    for rater in report.raters:
        headers.append(f"{rater}_Seconds")
        headers.append(f"{rater}_Events")
    headers.extend(["KrippendorffAlpha", "CohensKappaCI95Low", "CohensKappaCI95High"])
    writer.writerow(headers)

    for row in report.rows:
        values: List[object] = [
            row.subject,
            row.behavior,
            int(row.total_bins),
            f"{row.cohens_kappa:.6f}",
            f"{row.fleiss_kappa:.6f}",
            f"{row.pabak:.6f}",
            f"{row.observed_agreement:.6f}",
            f"{row.expected_agreement:.6f}",
            f"{row.prevalence_index:.6f}",
            f"{row.bias_index:.6f}",
            f"{row.positive_agreement:.6f}",
            f"{row.iou:.6f}",
            f"{row.onset_f1:.6f}",
        ]
        for rater in report.raters:
            values.append(f"{row.rater_durations.get(rater, 0.0):.6f}")
            values.append(int(row.rater_event_counts.get(rater, 0)))
        values.extend(
            [
                f"{row.krippendorff_alpha:.6f}",
                f"{row.cohens_kappa_ci95_low:.6f}",
                f"{row.cohens_kappa_ci95_high:.6f}",
            ]
        )
        writer.writerow(values)

    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")


def write_irr_disagreements_csv(report: IRRReport, path: Path | str) -> None:
    """Atomically write contiguous rater-disagreement bouts to CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Subject",
            "Behavior",
            "StartSeconds",
            "EndSeconds",
            "DurationSeconds",
            "RatersActive",
            "RatersInactive",
        ]
    )
    for d in report.disagreements:
        writer.writerow(
            [
                d.subject,
                d.behavior,
                f"{d.start_s:.6f}",
                f"{d.end_s:.6f}",
                f"{d.duration_s:.6f}",
                "|".join(d.raters_active),
                "|".join(d.raters_inactive),
            ]
        )
    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")
