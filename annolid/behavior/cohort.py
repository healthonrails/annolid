"""Cohort-level behavior experiment aggregation and Markov transition matrices.

Supports blinded-to-unblinded neuroscience cohort analysis across multiple
behavior event CSV exports:

- Recording-level and group-level summary statistics (mean, SD, SEM, median, IQR)
  for bout count, rate/min, total duration, % of session, and bout duration.
- Scientifically rigorous first-bout latency reporting: distinguishes responder-only
  latency (``latency_to_first_s``) from session-censored survival latency
  (``censored_latency_s``) and explicitly includes zero-occurrence subjects in
  group denominators.
- First-order Markov behavioral transition matrices ($N(i \\to j)$,
  $P(j \\mid i)$, target base rate, and observed/expected lift) per experimental
  condition and pooled across the cohort.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - optional scientific tier
    _scipy_stats = None  # type: ignore[assignment]

from annolid.behavior.event_utils import ParsedEventRow, read_parsed_event_rows
from annolid.behavior.time_budget import (
    BehaviorInterval,
    TimeBudgetComputationError,
    extract_behavior_intervals,
)
from annolid.core.behavior.spec import ProjectSchema
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "CohortComputationError",
    "CohortRecordingSpec",
    "CohortSubjectRow",
    "CohortGroupSummaryRow",
    "CohortComparisonRow",
    "TransitionEdge",
    "CohortBehaviorReport",
    "load_cohort_manifest",
    "apply_cohort_metadata_csv",
    "analyze_behavior_cohort",
    "compute_cohort_comparisons",
    "compute_transition_matrix",
    "format_cohort_report_table",
    "write_cohort_report_json",
    "write_cohort_group_summary_csv",
    "write_cohort_subject_rows_csv",
    "write_cohort_comparisons_csv",
    "write_transition_matrix_csv",
]


class CohortComputationError(Exception):
    """Raised when cohort inputs, manifests, or event CSVs are invalid."""


@dataclass(frozen=True)
class CohortRecordingSpec:
    """Input descriptor for a single recording in a behavioral cohort."""

    recording_id: str
    events_csv: Path
    group: str = "ungrouped"
    subject_groups: Dict[str, str] = field(default_factory=dict)
    session_start_s: Optional[float] = None
    session_duration_s: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CohortSubjectRow:
    """Per-(recording, subject, behavior) quantitative endpoint row."""

    recording_id: str
    subject: str
    group: str
    behavior: str
    category: str
    session_duration_s: float
    occurrences: int
    rate_per_min: float
    total_duration_s: float
    percent_of_session: float
    mean_bout_duration_s: float
    median_bout_duration_s: float
    latency_to_first_s: Optional[float]
    censored_latency_s: float
    censored: bool
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "subject": self.subject,
            "group": self.group,
            "behavior": self.behavior,
            "category": self.category,
            "session_duration_s": round(float(self.session_duration_s), 6),
            "occurrences": int(self.occurrences),
            "rate_per_min": round(float(self.rate_per_min), 6),
            "total_duration_s": round(float(self.total_duration_s), 6),
            "percent_of_session": round(float(self.percent_of_session), 6),
            "mean_bout_duration_s": round(float(self.mean_bout_duration_s), 6),
            "median_bout_duration_s": round(float(self.median_bout_duration_s), 6),
            "latency_to_first_s": (
                round(float(self.latency_to_first_s), 6)
                if self.latency_to_first_s is not None
                else None
            ),
            "censored_latency_s": round(float(self.censored_latency_s), 6),
            "censored": bool(self.censored),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CohortGroupSummaryRow:
    """Experimental condition summary statistics for a behavior across subjects."""

    group: str
    behavior: str
    category: str
    n_subjects: int
    n_responders: int
    responder_fraction: float
    mean_total_duration_s: float
    sd_total_duration_s: float
    sem_total_duration_s: float
    median_total_duration_s: float
    iqr_total_duration_s: float
    mean_percent_of_session: float
    sem_percent_of_session: float
    mean_occurrences: float
    mean_rate_per_min: float
    sem_rate_per_min: float
    mean_latency_responders_s: Optional[float]
    median_latency_responders_s: Optional[float]
    mean_latency_censored_s: float
    median_latency_censored_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "behavior": self.behavior,
            "category": self.category,
            "n_subjects": int(self.n_subjects),
            "n_responders": int(self.n_responders),
            "responder_fraction": round(float(self.responder_fraction), 6),
            "mean_total_duration_s": round(float(self.mean_total_duration_s), 6),
            "sd_total_duration_s": round(float(self.sd_total_duration_s), 6),
            "sem_total_duration_s": round(float(self.sem_total_duration_s), 6),
            "median_total_duration_s": round(float(self.median_total_duration_s), 6),
            "iqr_total_duration_s": round(float(self.iqr_total_duration_s), 6),
            "mean_percent_of_session": round(float(self.mean_percent_of_session), 6),
            "sem_percent_of_session": round(float(self.sem_percent_of_session), 6),
            "mean_occurrences": round(float(self.mean_occurrences), 6),
            "mean_rate_per_min": round(float(self.mean_rate_per_min), 6),
            "sem_rate_per_min": round(float(self.sem_rate_per_min), 6),
            "mean_latency_responders_s": (
                round(float(self.mean_latency_responders_s), 6)
                if self.mean_latency_responders_s is not None
                else None
            ),
            "median_latency_responders_s": (
                round(float(self.median_latency_responders_s), 6)
                if self.median_latency_responders_s is not None
                else None
            ),
            "mean_latency_censored_s": round(float(self.mean_latency_censored_s), 6),
            "median_latency_censored_s": round(
                float(self.median_latency_censored_s), 6
            ),
        }


@dataclass(frozen=True)
class TransitionEdge:
    """First-order Markov transition edge ``from_behavior -> to_behavior``."""

    group: str
    from_behavior: str
    to_behavior: str
    count: int
    from_total: int
    probability: float
    target_base_rate: float
    lift: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "from_behavior": self.from_behavior,
            "to_behavior": self.to_behavior,
            "count": int(self.count),
            "from_total": int(self.from_total),
            "probability": round(float(self.probability), 6),
            "target_base_rate": round(float(self.target_base_rate), 6),
            "lift": round(float(self.lift), 6),
        }


@dataclass(frozen=True)
class CohortComparisonRow:
    """Inferential group comparison, effect sizes, and survival latency statistics."""

    behavior: str
    category: str
    groups: Tuple[str, ...]
    analysis_unit: str
    parametric_test: str
    parametric_stat: float
    p_parametric: float
    q_parametric_bh: float
    nonparametric_test: str
    nonparametric_stat: float
    p_nonparametric: float
    q_nonparametric_bh: float
    hedges_g: Optional[float]
    cliffs_delta: Optional[float]
    mean_diff_s: Optional[float]
    mean_diff_ci95_low_s: Optional[float]
    mean_diff_ci95_high_s: Optional[float]
    responder_p_value: float
    km_median_latency_s: Dict[str, Optional[float]]
    logrank_chi2: float
    logrank_p_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior": self.behavior,
            "category": self.category,
            "groups": list(self.groups),
            "analysis_unit": self.analysis_unit,
            "parametric_test": self.parametric_test,
            "parametric_stat": round(float(self.parametric_stat), 6),
            "p_parametric": round(float(self.p_parametric), 6),
            "q_parametric_bh": round(float(self.q_parametric_bh), 6),
            "nonparametric_test": self.nonparametric_test,
            "nonparametric_stat": round(float(self.nonparametric_stat), 6),
            "p_nonparametric": round(float(self.p_nonparametric), 6),
            "q_nonparametric_bh": round(float(self.q_nonparametric_bh), 6),
            "hedges_g": (
                round(float(self.hedges_g), 6) if self.hedges_g is not None else None
            ),
            "cliffs_delta": (
                round(float(self.cliffs_delta), 6)
                if self.cliffs_delta is not None
                else None
            ),
            "mean_diff_s": (
                round(float(self.mean_diff_s), 6)
                if self.mean_diff_s is not None
                else None
            ),
            "mean_diff_ci95_low_s": (
                round(float(self.mean_diff_ci95_low_s), 6)
                if self.mean_diff_ci95_low_s is not None
                else None
            ),
            "mean_diff_ci95_high_s": (
                round(float(self.mean_diff_ci95_high_s), 6)
                if self.mean_diff_ci95_high_s is not None
                else None
            ),
            "responder_p_value": round(float(self.responder_p_value), 6),
            "km_median_latency_s": {
                k: (round(float(v), 6) if v is not None else None)
                for k, v in self.km_median_latency_s.items()
            },
            "logrank_chi2": round(float(self.logrank_chi2), 6),
            "logrank_p_value": round(float(self.logrank_p_value), 6),
        }


@dataclass(frozen=True)
class CohortBehaviorReport:
    """Full cohort aggregation and behavioral transition matrix report."""

    recording_count: int
    subject_unit_count: int
    groups: Tuple[str, ...]
    behaviors: Tuple[str, ...]
    subject_rows: Tuple[CohortSubjectRow, ...]
    group_summaries: Tuple[CohortGroupSummaryRow, ...]
    transitions: Tuple[TransitionEdge, ...]
    warnings: Tuple[str, ...] = ()
    comparisons: Tuple[CohortComparisonRow, ...] = ()
    analysis_unit: str = "subject"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_count": int(self.recording_count),
            "subject_unit_count": int(self.subject_unit_count),
            "analysis_unit": self.analysis_unit,
            "groups": list(self.groups),
            "behaviors": list(self.behaviors),
            "group_summaries": [row.to_dict() for row in self.group_summaries],
            "comparisons": [row.to_dict() for row in self.comparisons],
            "subject_rows": [row.to_dict() for row in self.subject_rows],
            "transitions": [edge.to_dict() for edge in self.transitions],
            "warnings": list(self.warnings),
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _percentile_linear(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, float(q))) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _dispersion(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """Return ``(mean, sd, sem, median, iqr)`` for non-empty ``values``."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ordered = sorted(float(v) for v in values)
    avg = float(mean(ordered))
    med = float(median(ordered))
    if len(ordered) >= 2:
        sd = float(stdev(ordered))
        sem = sd / math.sqrt(len(ordered))
        iqr = _percentile_linear(ordered, 0.75) - _percentile_linear(ordered, 0.25)
    else:
        sd = 0.0
        sem = 0.0
        iqr = 0.0
    return avg, sd, sem, med, max(0.0, iqr)


def _match_header(
    fieldnames: Sequence[str], candidates: Sequence[str]
) -> Optional[str]:
    lower_map = {name.strip().lower(): name for name in fieldnames if name}
    for cand in candidates:
        if cand in fieldnames:
            return cand
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def load_cohort_manifest(manifest_path: Path | str) -> List[CohortRecordingSpec]:
    """Load cohort recording specifications from a CSV or JSON manifest."""
    resolved = Path(manifest_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Cohort manifest not found: {resolved}")

    base_dir = resolved.parent
    suffix = resolved.suffix.lower()

    if suffix == ".json":
        raw_payload = json.loads(resolved.read_text(encoding="utf-8"))
        entries = (
            raw_payload.get("recordings")
            if isinstance(raw_payload, Mapping)
            else raw_payload
        )
        if not isinstance(entries, list) or not entries:
            raise CohortComputationError(
                f"JSON manifest '{resolved}' must contain a non-empty list of recordings."
            )
        specs: List[CohortRecordingSpec] = []
        for idx, item in enumerate(entries, start=1):
            if not isinstance(item, Mapping):
                raise CohortComputationError(
                    f"Manifest entry #{idx} in '{resolved}' must be an object."
                )
            raw_csv = str(
                item.get("events_csv") or item.get("csv") or item.get("path") or ""
            ).strip()
            if not raw_csv:
                raise CohortComputationError(
                    f"Manifest entry #{idx} in '{resolved}' is missing 'events_csv'."
                )
            csv_path = Path(raw_csv).expanduser()
            if not csv_path.is_absolute():
                csv_path = (base_dir / csv_path).resolve()
            rec_id = str(item.get("recording_id") or csv_path.stem).strip()
            grp = str(
                item.get("group")
                or item.get("condition")
                or item.get("treatment")
                or item.get("genotype")
                or "ungrouped"
            ).strip()
            subj_groups_raw = item.get("subject_groups")
            subj_groups = (
                {
                    str(k).strip(): str(v).strip()
                    for k, v in subj_groups_raw.items()
                    if str(k).strip() and str(v).strip()
                }
                if isinstance(subj_groups_raw, Mapping)
                else {}
            )
            meta = {
                str(k): str(v)
                for k, v in item.items()
                if k
                not in {
                    "recording_id",
                    "events_csv",
                    "csv",
                    "path",
                    "group",
                    "condition",
                    "treatment",
                    "genotype",
                    "subject_groups",
                    "session_start_s",
                    "session_duration_s",
                }
                and v is not None
            }
            specs.append(
                CohortRecordingSpec(
                    recording_id=rec_id,
                    events_csv=csv_path,
                    group=grp or "ungrouped",
                    subject_groups=subj_groups,
                    session_start_s=_safe_float(item.get("session_start_s")),
                    session_duration_s=_safe_float(item.get("session_duration_s")),
                    metadata=meta,
                )
            )
        return specs

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CohortComputationError(f"CSV manifest '{resolved}' is empty.")
        fields = list(reader.fieldnames)
        csv_col = _match_header(
            fields, ("events_csv", "csv_path", "csv", "file", "path")
        )
        if csv_col is None:
            raise CohortComputationError(
                f"CSV manifest '{resolved}' must include an 'events_csv' or 'path' column."
            )
        id_col = _match_header(fields, ("recording_id", "video_id", "session_id", "id"))
        group_col = _match_header(
            fields, ("group", "condition", "treatment", "genotype", "cohort")
        )
        subject_col = _match_header(fields, ("subject", "animal_id", "instance"))
        start_col = _match_header(fields, ("session_start_s", "session_start"))
        dur_col = _match_header(
            fields, ("session_duration_s", "session_duration", "duration_s")
        )
        reserved = {
            c
            for c in (csv_col, id_col, group_col, subject_col, start_col, dur_col)
            if c is not None
        }

        by_recording: Dict[Tuple[str, Path], CohortRecordingSpec] = {}
        for row in reader:
            raw_csv = str(row.get(csv_col) or "").strip()
            if not raw_csv:
                continue
            csv_path = Path(raw_csv).expanduser()
            if not csv_path.is_absolute():
                csv_path = (base_dir / csv_path).resolve()
            rec_id = (
                str(row.get(id_col) or "").strip()
                if id_col and str(row.get(id_col) or "").strip()
                else csv_path.stem
            )
            grp = (
                str(row.get(group_col) or "").strip()
                if group_col and str(row.get(group_col) or "").strip()
                else "ungrouped"
            )
            subj = (
                str(row.get(subject_col) or "").strip()
                if subject_col and str(row.get(subject_col) or "").strip()
                else ""
            )
            start_s = _safe_float(row.get(start_col)) if start_col else None
            dur_s = _safe_float(row.get(dur_col)) if dur_col else None
            meta = {
                str(k): str(v).strip()
                for k, v in row.items()
                if k and k not in reserved and str(v or "").strip()
            }

            key = (rec_id, csv_path)
            existing = by_recording.get(key)
            if existing is None:
                subj_map = {subj: grp} if subj else {}
                by_recording[key] = CohortRecordingSpec(
                    recording_id=rec_id,
                    events_csv=csv_path,
                    group=grp,
                    subject_groups=subj_map,
                    session_start_s=start_s,
                    session_duration_s=dur_s,
                    metadata=meta,
                )
            else:
                merged_subj = dict(existing.subject_groups)
                if subj:
                    merged_subj[subj] = grp
                by_recording[key] = CohortRecordingSpec(
                    recording_id=existing.recording_id,
                    events_csv=existing.events_csv,
                    group=existing.group,
                    subject_groups=merged_subj,
                    session_start_s=(
                        existing.session_start_s
                        if existing.session_start_s is not None
                        else start_s
                    ),
                    session_duration_s=(
                        existing.session_duration_s
                        if existing.session_duration_s is not None
                        else dur_s
                    ),
                    metadata={**existing.metadata, **meta},
                )

    if not by_recording:
        raise CohortComputationError(
            f"Cohort manifest '{resolved}' contained no valid recording rows."
        )
    return list(by_recording.values())


def apply_cohort_metadata_csv(
    specs: Sequence[CohortRecordingSpec],
    metadata_csv: Path | str,
) -> List[CohortRecordingSpec]:
    """Unblind/join experimental group metadata onto ``specs`` by recording ID or file stem."""
    meta_path = Path(metadata_csv).expanduser().resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Cohort metadata CSV not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CohortComputationError(f"Metadata CSV '{meta_path}' is empty.")
        fields = list(reader.fieldnames)
        id_col = _match_header(
            fields, ("recording_id", "video_id", "file", "filename", "stem", "id")
        )
        group_col = _match_header(
            fields, ("group", "condition", "treatment", "genotype", "cohort")
        )
        if id_col is None or group_col is None:
            raise CohortComputationError(
                f"Metadata CSV '{meta_path}' must contain recording_id and group/condition columns."
            )
        subject_col = _match_header(fields, ("subject", "animal_id", "instance"))
        dur_col = _match_header(
            fields, ("session_duration_s", "session_duration", "duration_s")
        )
        reserved = {
            c for c in (id_col, group_col, subject_col, dur_col) if c is not None
        }

        rec_group: Dict[str, str] = {}
        rec_subj_groups: Dict[str, Dict[str, str]] = defaultdict(dict)
        rec_dur: Dict[str, float] = {}
        rec_extra: Dict[str, Dict[str, str]] = defaultdict(dict)

        for row in reader:
            raw_key = str(row.get(id_col) or "").strip()
            if not raw_key:
                continue
            key_variants = {raw_key, Path(raw_key).stem, Path(raw_key).name}
            grp = str(row.get(group_col) or "").strip() or "ungrouped"
            subj = (
                str(row.get(subject_col) or "").strip()
                if subject_col and str(row.get(subject_col) or "").strip()
                else ""
            )
            dur = _safe_float(row.get(dur_col)) if dur_col else None
            extra = {
                str(k): str(v).strip()
                for k, v in row.items()
                if k and k not in reserved and str(v or "").strip()
            }
            for k_var in key_variants:
                if subj:
                    rec_subj_groups[k_var][subj] = grp
                else:
                    rec_group[k_var] = grp
                if dur is not None:
                    rec_dur[k_var] = dur
                rec_extra[k_var].update(extra)

    enriched: List[CohortRecordingSpec] = []
    for spec in specs:
        lookup_keys = (
            spec.recording_id,
            spec.events_csv.stem,
            spec.events_csv.name,
        )
        matched_group = spec.group
        matched_subj = dict(spec.subject_groups)
        matched_dur = spec.session_duration_s
        matched_meta = dict(spec.metadata)
        for lk in lookup_keys:
            if lk in rec_group:
                matched_group = rec_group[lk]
            if lk in rec_subj_groups:
                matched_subj.update(rec_subj_groups[lk])
            if matched_dur is None and lk in rec_dur:
                matched_dur = rec_dur[lk]
            if lk in rec_extra:
                matched_meta.update(rec_extra[lk])
        enriched.append(
            CohortRecordingSpec(
                recording_id=spec.recording_id,
                events_csv=spec.events_csv,
                group=matched_group,
                subject_groups=matched_subj,
                session_start_s=spec.session_start_s,
                session_duration_s=matched_dur,
                metadata=matched_meta,
            )
        )
    return enriched


def compute_transition_matrix(
    sequences_by_group: Mapping[str, Sequence[Sequence[BehaviorInterval]]],
    *,
    max_transition_gap_s: Optional[float] = None,
    include_self_transitions: bool = True,
) -> Tuple[TransitionEdge, ...]:
    """Compute first-order Markov transition probabilities per group and pooled ``__all__``."""

    def _edges_for_group(
        group_name: str, sequences: Sequence[Sequence[BehaviorInterval]]
    ) -> List[TransitionEdge]:
        pair_counts: Counter[Tuple[str, str]] = Counter()
        from_totals: Counter[str] = Counter()
        target_totals: Counter[str] = Counter()
        grand_total = 0

        for seq in sequences:
            ordered = sorted(seq, key=lambda iv: (iv.start, iv.end, iv.behavior))
            for prev_iv, next_iv in zip(ordered, ordered[1:]):
                if (
                    not include_self_transitions
                    and prev_iv.behavior == next_iv.behavior
                ):
                    continue
                gap = next_iv.start - prev_iv.end
                if max_transition_gap_s is not None and gap > float(
                    max_transition_gap_s
                ):
                    continue
                pair_counts[(prev_iv.behavior, next_iv.behavior)] += 1
                from_totals[prev_iv.behavior] += 1
                target_totals[next_iv.behavior] += 1
                grand_total += 1

        if grand_total == 0:
            return []

        edges: List[TransitionEdge] = []
        for (src, dst), count in sorted(
            pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        ):
            denom = from_totals[src]
            prob = count / float(denom) if denom > 0 else 0.0
            base_rate = target_totals[dst] / float(grand_total)
            lift = (prob / base_rate) if base_rate > 0 else 0.0
            edges.append(
                TransitionEdge(
                    group=group_name,
                    from_behavior=src,
                    to_behavior=dst,
                    count=int(count),
                    from_total=int(denom),
                    probability=prob,
                    target_base_rate=base_rate,
                    lift=lift,
                )
            )
        return edges

    all_edges: List[TransitionEdge] = []
    pooled_sequences: List[Sequence[BehaviorInterval]] = []
    for grp in sorted(sequences_by_group):
        seqs = list(sequences_by_group[grp])
        pooled_sequences.extend(seqs)
        all_edges.extend(_edges_for_group(grp, seqs))

    if len(sequences_by_group) > 1 and pooled_sequences:
        all_edges.extend(_edges_for_group("__all__", pooled_sequences))

    return tuple(all_edges)


def _extract_recording_events(
    parsed_rows: Sequence[ParsedEventRow],
    *,
    schema: Optional[ProjectSchema],
    strict: bool,
) -> Tuple[List[BehaviorInterval], List[str], Optional[float], Optional[float]]:
    state_behavior_map: Dict[str, bool] = {}
    if schema is not None:
        state_behavior_map = {b.code: bool(b.is_state) for b in schema.behaviors}
        state_behavior_map.update({b.name: bool(b.is_state) for b in schema.behaviors})

    raw_state_rows: List[Dict[str, str]] = []
    point_intervals: List[BehaviorInterval] = []
    timestamps: List[float] = []
    subjects_seen: List[str] = []

    for row in parsed_rows:
        if row.recording_time is not None:
            timestamps.append(float(row.recording_time))
        if row.subject and row.subject not in subjects_seen:
            subjects_seen.append(row.subject)
        if not row.behavior or row.recording_time is None:
            continue

        is_point = row.event_kind == "point" or (
            row.event_kind is None and state_behavior_map.get(row.behavior) is False
        )
        if is_point:
            point_intervals.append(
                BehaviorInterval(
                    subject=row.subject,
                    behavior=row.behavior,
                    start=float(row.recording_time),
                    end=float(row.recording_time),
                )
            )
        else:
            raw_state_rows.append(
                {
                    "recording time": str(row.recording_time),
                    "subject": row.subject,
                    "behavior": row.behavior,
                    "event": row.raw_event,
                }
            )

    warnings: List[str] = []
    state_intervals: List[BehaviorInterval] = []
    if raw_state_rows:
        state_intervals, warnings = extract_behavior_intervals(raw_state_rows)
        if strict and warnings:
            raise TimeBudgetComputationError(warnings[0])

    combined = sorted(
        [*state_intervals, *point_intervals],
        key=lambda iv: (iv.subject, iv.start, iv.end, iv.behavior),
    )
    t_min = min(timestamps) if timestamps else None
    t_max = max(timestamps) if timestamps else None
    return combined, warnings, t_min, t_max


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(
        enumerate(max(0.0, min(1.0, float(p))) for p in p_values),
        key=lambda item: item[1],
    )
    adjusted = [1.0] * m
    running_min = 1.0
    for rank_rev in range(m - 1, -1, -1):
        orig_idx, p_val = indexed[rank_rev]
        rank = rank_rev + 1
        q_val = min(running_min, p_val * float(m) / float(rank))
        running_min = q_val
        adjusted[orig_idx] = max(0.0, min(1.0, q_val))
    return adjusted


def _betainc_cf(a: float, b: float, x: float) -> float:
    """Modified Lentz continued fraction for regularized incomplete beta ``I_x(a, b)``."""
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 251):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Exact regularized incomplete beta ``I_x(a, b)`` without SciPy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    if x > (a + 1.0) / (a + b + 2.0):
        return max(0.0, min(1.0, 1.0 - _betainc_reg(b, a, 1.0 - x)))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_front = a * math.log(x) + b * math.log1p(-x) - lbeta
    if log_front < -740.0:
        return 0.0
    front = math.exp(log_front) / a
    return max(0.0, min(1.0, front * _betainc_cf(a, b, x)))


def _gammaincc_reg(a: float, x: float) -> float:
    """Exact regularized upper incomplete gamma ``Q(a, x)`` without SciPy."""
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    lg_a = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(300):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        log_pref = -x + a * math.log(x) - lg_a
        if log_pref < -740.0:
            return 1.0
        p_lower = total * math.exp(log_pref)
        return max(0.0, min(1.0, 1.0 - p_lower))

    tiny = 1e-30
    b0 = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / (b0 if abs(b0) >= tiny else tiny)
    h = d
    for idx in range(1, 301):
        an = -float(idx) * (float(idx) - a)
        b0 += 2.0
        d = an * d + b0
        if abs(d) < tiny:
            d = tiny
        c = b0 + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    log_pref = -x + a * math.log(x) - lg_a
    if log_pref < -740.0:
        return 0.0
    return max(0.0, min(1.0, math.exp(log_pref) * h))


def _student_t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for Student's t distribution with real ``df > 0``."""
    if df <= 0.0 or not math.isfinite(t_stat):
        return 1.0
    t_abs = abs(float(t_stat))
    if t_abs == 0.0:
        return 1.0
    if _scipy_stats is not None:
        return float(max(0.0, min(1.0, 2.0 * _scipy_stats.t.sf(t_abs, df))))
    x = float(df) / (float(df) + t_abs * t_abs)
    return _betainc_reg(0.5 * float(df), 0.5, x)


def _snedecor_f_sf(f_stat: float, df1: float, df2: float) -> float:
    """Upper-tail survival probability for Snedecor's F(df1, df2) distribution."""
    if df1 <= 0.0 or df2 <= 0.0 or not math.isfinite(f_stat) or f_stat <= 0.0:
        return 1.0
    if _scipy_stats is not None:
        return float(max(0.0, min(1.0, _scipy_stats.f.sf(float(f_stat), df1, df2))))
    x = float(df2) / (float(df2) + float(df1) * float(f_stat))
    return _betainc_reg(0.5 * float(df2), 0.5 * float(df1), x)


def _chi2_sf_fallback(chi2_val: float, df: int) -> float:
    x = max(0.0, float(chi2_val))
    if df <= 0 or x == 0.0:
        return 1.0
    if _scipy_stats is not None:
        return float(max(0.0, min(1.0, _scipy_stats.chi2.sf(x, df))))
    return _gammaincc_reg(0.5 * float(df), 0.5 * x)


def _midranks_and_ties(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return 1-based midranks and tie correction sum ``sum(t^3 - t)``."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.float64), 0.0
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    tie_sum = 0.0
    idx = 0
    while idx < n:
        end = idx + 1
        while end < n and sorted_vals[end] == sorted_vals[idx]:
            end += 1
        avg_rank = 0.5 * float(idx + 1 + end)
        ranks[order[idx:end]] = avg_rank
        t_len = end - idx
        if t_len > 1:
            tie_sum += float(t_len * t_len * t_len - t_len)
        idx = end
    return ranks, tie_sum


def _one_way_anova_exact(groups: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Compute one-way ANOVA ``(F_stat, p_value)`` with or without SciPy."""
    valid_g = [np.asarray(g, dtype=np.float64) for g in groups if len(g) >= 1]
    k = len(valid_g)
    n_tot = sum(len(g) for g in valid_g)
    if k < 2 or n_tot <= k:
        return 0.0, 1.0
    all_vals = np.concatenate(valid_g)
    grand_mean = float(np.mean(all_vals))
    ss_between = sum(
        float(len(g)) * ((float(np.mean(g)) - grand_mean) ** 2) for g in valid_g
    )
    ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in valid_g)
    df1 = float(k - 1)
    df2 = float(n_tot - k)
    if ss_within <= 1e-24:
        return (0.0, 1.0) if ss_between <= 1e-24 else (1e9, 0.0)
    f_stat = (ss_between / df1) / (ss_within / df2)
    if not math.isfinite(f_stat):
        return 0.0, 1.0
    return float(f_stat), _snedecor_f_sf(float(f_stat), df1, df2)


def _mann_whitney_u_exact(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:
    """Compute two-sided Mann-Whitney U ``(U_stat, p_value)`` with exact small-N DP & tie correction."""
    n1, n2 = len(x1), len(x2)
    if n1 < 1 or n2 < 1:
        return 0.0, 1.0
    if _scipy_stats is not None:
        try:
            res_u = _scipy_stats.mannwhitneyu(x1, x2, alternative="two-sided")
            if math.isfinite(float(res_u.statistic)):
                return float(res_u.statistic), float(res_u.pvalue)
        except ValueError:
            return 0.5 * float(n1 * n2), 1.0

    pooled = np.concatenate([x1, x2])
    ranks, tie_sum = _midranks_and_ties(pooled)
    r1 = float(np.sum(ranks[:n1]))
    u1 = r1 - 0.5 * float(n1 * (n1 + 1))
    n_tot = n1 + n2

    # Exact discrete distribution via dynamic programming when tie-free and N <= 40
    if tie_sum == 0.0 and n_tot <= 40:
        n_small = min(n1, n2)
        max_u = n1 * n2
        dp = [[0] * (max_u + 1) for _ in range(n_small + 1)]
        dp[0][0] = 1
        for item_idx in range(n_tot):
            for c in range(min(item_idx + 1, n_small), 0, -1):
                contrib = item_idx - (c - 1)
                if contrib < 0:
                    continue
                row_prev = dp[c - 1]
                row_cur = dp[c]
                for u_prev in range(max_u - contrib + 1):
                    if row_prev[u_prev]:
                        row_cur[u_prev + contrib] += row_prev[u_prev]
        u_int = int(round(u1))
        u_tail = min(u_int, max_u - u_int)
        total_ways = math.comb(n_tot, n_small)
        p_exact = min(
            1.0, 2.0 * float(sum(dp[n_small][: u_tail + 1])) / float(total_ways)
        )
        return float(u1), p_exact

    # Tie-corrected asymptotic normal with continuity correction
    mu_u = 0.5 * float(n1 * n2)
    tie_term = tie_sum / float(n_tot * (n_tot - 1)) if n_tot > 1 else 0.0
    var_u = (float(n1 * n2) / 12.0) * max(0.0, float(n_tot + 1) - tie_term)
    if var_u <= 1e-24:
        return float(u1), 1.0
    z_num = max(0.0, abs(u1 - mu_u) - 0.5)
    z_u = z_num / math.sqrt(var_u)
    return float(u1), float(math.erfc(z_u / math.sqrt(2.0)))


def _kruskal_wallis_exact(groups: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Compute tie-corrected Kruskal-Wallis ``(H_stat, p_value)`` with or without SciPy."""
    valid_g = [np.asarray(g, dtype=np.float64) for g in groups if len(g) >= 1]
    if len(valid_g) < 2:
        return 0.0, 1.0
    pooled = np.concatenate(valid_g)
    if len(pooled) <= len(valid_g) or float(np.ptp(pooled)) <= 0.0:
        return 0.0, 1.0
    if _scipy_stats is not None:
        try:
            res_kw = _scipy_stats.kruskal(*valid_g)
            if math.isfinite(float(res_kw.statistic)):
                return float(res_kw.statistic), float(res_kw.pvalue)
        except ValueError:
            return 0.0, 1.0

    n_tot = len(pooled)
    ranks, tie_sum = _midranks_and_ties(pooled)
    denom_ties = float(n_tot * n_tot * n_tot - n_tot)
    if denom_ties <= 0.0 or tie_sum >= denom_ties - 1e-12:
        return 0.0, 1.0
    offset = 0
    sum_sq_over_n = 0.0
    for g in valid_g:
        ng = len(g)
        rg = float(np.sum(ranks[offset : offset + ng]))
        sum_sq_over_n += (rg * rg) / float(ng)
        offset += ng
    h_raw = (12.0 / float(n_tot * (n_tot + 1))) * sum_sq_over_n - 3.0 * float(n_tot + 1)
    tie_corr = 1.0 - (tie_sum / denom_ties)
    h_stat = max(0.0, h_raw / tie_corr)
    return float(h_stat), _chi2_sf_fallback(float(h_stat), len(valid_g) - 1)


def _fisher_exact_2x2(table: np.ndarray) -> float:
    """Two-sided exact Fisher hypergeometric p-value for a 2x2 contingency table."""
    if _scipy_stats is not None:
        try:
            _, p_val = _scipy_stats.fisher_exact(table, alternative="two-sided")
            return float(max(0.0, min(1.0, p_val)))
        except Exception:
            pass
    a, b = int(table[0, 0]), int(table[0, 1])
    c, d = int(table[1, 0]), int(table[1, 1])
    r1 = a + b
    r2 = c + d
    c1 = a + c
    n_tot = r1 + r2
    if n_tot <= 0:
        return 1.0
    k_min = max(0, c1 - r2)
    k_max = min(r1, c1)

    def _log_hypergeom(k_val: int) -> float:
        return (
            math.lgamma(r1 + 1)
            - math.lgamma(k_val + 1)
            - math.lgamma(r1 - k_val + 1)
            + math.lgamma(r2 + 1)
            - math.lgamma(c1 - k_val + 1)
            - math.lgamma(r2 - (c1 - k_val) + 1)
            - (
                math.lgamma(n_tot + 1)
                - math.lgamma(c1 + 1)
                - math.lgamma(n_tot - c1 + 1)
            )
        )

    lp_obs = _log_hypergeom(a)
    total_p = 0.0
    for k_val in range(k_min, k_max + 1):
        lp_k = _log_hypergeom(k_val)
        if lp_k <= lp_obs + 1e-12:
            total_p += math.exp(lp_k)
    return max(0.0, min(1.0, total_p))


def _chi2_contingency_exact(table: np.ndarray) -> float:
    """Pearson chi-square contingency test p-value for a R x C table."""
    if _scipy_stats is not None:
        try:
            _, p_val, _, _ = _scipy_stats.chi2_contingency(table)
            return float(max(0.0, min(1.0, p_val)))
        except Exception:
            return 1.0
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return 1.0
    row_sums = arr.sum(axis=1)
    col_sums = arr.sum(axis=0)
    n_tot = float(row_sums.sum())
    if n_tot <= 0.0 or np.any(row_sums <= 0.0) or np.any(col_sums <= 0.0):
        return 1.0
    expected = np.outer(row_sums, col_sums) / n_tot
    chi2_stat = float(np.sum(((arr - expected) ** 2) / expected))
    df = int((arr.shape[0] - 1) * (arr.shape[1] - 1))
    return _chi2_sf_fallback(chi2_stat, df)


def _kaplan_meier_median(
    times: Sequence[float], events: Sequence[bool]
) -> Optional[float]:
    if not times:
        return None
    pairs = sorted(zip((float(t) for t in times), (bool(e) for e in events)))
    n_at_risk = len(pairs)
    surv = 1.0
    idx = 0
    m = len(pairs)
    while idx < m and n_at_risk > 0:
        t_cur = pairs[idx][0]
        d_j = 0
        c_j = 0
        while idx < m and abs(pairs[idx][0] - t_cur) <= 1e-12:
            if pairs[idx][1]:
                d_j += 1
            else:
                c_j += 1
            idx += 1
        if d_j > 0:
            surv *= 1.0 - float(d_j) / float(n_at_risk)
            if surv <= 0.5 + 1e-12:
                return float(t_cur)
        n_at_risk -= d_j + c_j
    return None


def _logrank_test(
    grouped_survival: Mapping[str, Sequence[Tuple[float, bool]]],
    ordered_groups: Sequence[str],
) -> Tuple[float, float]:
    k = len(ordered_groups)
    if k < 2:
        return 0.0, 1.0

    obs_records: List[Tuple[float, int, bool]] = []
    n_vec = np.zeros(k, dtype=np.float64)
    for g_idx, g in enumerate(ordered_groups):
        for t, ev in grouped_survival.get(g, ()):
            t_f = float(t)
            if math.isfinite(t_f):
                obs_records.append((t_f, g_idx, bool(ev)))
                n_vec[g_idx] += 1.0

    if not any(ev for _, _, ev in obs_records):
        return 0.0, 1.0

    obs_records.sort(key=lambda item: item[0])
    obs_minus_exp = np.zeros(k - 1, dtype=np.float64)
    cov = np.zeros((k - 1, k - 1), dtype=np.float64)

    idx = 0
    m = len(obs_records)
    while idx < m:
        t_cur = obs_records[idx][0]
        d_vec = np.zeros(k, dtype=np.float64)
        c_vec = np.zeros(k, dtype=np.float64)
        while idx < m and abs(obs_records[idx][0] - t_cur) <= 1e-12:
            _, g_idx, is_ev = obs_records[idx]
            if is_ev:
                d_vec[g_idx] += 1.0
            else:
                c_vec[g_idx] += 1.0
            idx += 1
        n_tot = float(n_vec.sum())
        d_tot = float(d_vec.sum())
        if n_tot > 1.0 and d_tot > 0.0:
            e_vec = n_vec * (d_tot / n_tot)
            obs_minus_exp += (d_vec - e_vec)[: k - 1]
            factor = (d_tot * (n_tot - d_tot)) / (n_tot * n_tot * (n_tot - 1.0))
            if factor > 0.0:
                p_vec = n_vec[: k - 1]
                v_step = -np.outer(p_vec, p_vec) * factor
                for idx_d in range(k - 1):
                    v_step[idx_d, idx_d] = (
                        p_vec[idx_d] * (n_tot - p_vec[idx_d]) * factor
                    )
                cov += v_step
        n_vec -= d_vec + c_vec

    try:
        inv_cov = np.linalg.pinv(cov)
        chi2_stat = float(max(0.0, obs_minus_exp @ inv_cov @ obs_minus_exp))
    except Exception:
        chi2_stat = 0.0
    p_val = _chi2_sf_fallback(chi2_stat, k - 1)
    return chi2_stat, p_val


def compute_cohort_comparisons(
    rows: Sequence[CohortSubjectRow],
    *,
    ordered_groups: Sequence[str],
    ordered_behaviors: Sequence[str],
    category_lookup: Mapping[str, str],
    analysis_unit: str = "subject",
) -> Tuple[CohortComparisonRow, ...]:
    """Compute parametric, nonparametric, effect-size, and survival comparisons across groups."""
    if len(ordered_groups) < 2 or not ordered_behaviors:
        return ()

    by_beh_group: Dict[Tuple[str, str], List[CohortSubjectRow]] = defaultdict(list)
    for r in rows:
        by_beh_group[(r.behavior, r.group)].append(r)

    raw_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for beh in ordered_behaviors:
        active_groups = [
            g for g in ordered_groups if len(by_beh_group.get((beh, g), ())) > 0
        ]
        if len(active_groups) < 2:
            continue

        group_vals = [
            np.asarray(
                [float(r.total_duration_s) for r in by_beh_group[(beh, g)]],
                dtype=np.float64,
            )
            for g in active_groups
        ]
        k = len(active_groups)

        # Parametric test (Welch t for 2 groups, One-way ANOVA for >=3)
        param_name = "welch_t" if k == 2 else "one_way_anova"
        param_stat = 0.0
        p_param = 1.0
        if k == 2:
            x1, x2 = group_vals[0], group_vals[1]
            n1_v, n2_v = len(x1), len(x2)
            if n1_v >= 2 and n2_v >= 2:
                v1 = float(np.var(x1, ddof=1))
                v2 = float(np.var(x2, ddof=1))
                term1 = v1 / float(n1_v)
                term2 = v2 / float(n2_v)
                se = math.sqrt(term1 + term2)
                mean_diff_val = float(np.mean(x1) - np.mean(x2))
                if se > 1e-12:
                    param_stat = float(mean_diff_val / se)
                    df_denom = (term1 * term1) / float(n1_v - 1) + (
                        term2 * term2
                    ) / float(n2_v - 1)
                    df_welch = (
                        ((term1 + term2) ** 2) / df_denom
                        if df_denom > 1e-24
                        else float(n1_v + n2_v - 2)
                    )
                    p_param = _student_t_two_sided_p(param_stat, df_welch)
                elif abs(mean_diff_val) > 1e-12:
                    param_stat = math.copysign(1e9, mean_diff_val)
                    p_param = 0.0
        else:
            param_stat, p_param = _one_way_anova_exact(group_vals)

        # Nonparametric test (Mann-Whitney U for 2 groups, Kruskal-Wallis for >=3)
        nonparam_name = "mann_whitney_u" if k == 2 else "kruskal_wallis"
        if k == 2:
            nonparam_stat, p_nonparam = _mann_whitney_u_exact(
                group_vals[0], group_vals[1]
            )
        else:
            nonparam_stat, p_nonparam = _kruskal_wallis_exact(group_vals)

        # Effect sizes & bootstrap 95% CI for 2-group comparison
        hedges_g: Optional[float] = None
        cliffs_d: Optional[float] = None
        mean_diff: Optional[float] = None
        ci_low: Optional[float] = None
        ci_high: Optional[float] = None
        if k == 2:
            x1, x2 = group_vals[0], group_vals[1]
            n1, n2 = len(x1), len(x2)
            if n1 >= 1 and n2 >= 1:
                mean_diff = float(np.mean(x1) - np.mean(x2))
                gt = float(np.sum(x1[:, None] > x2[None, :]))
                lt = float(np.sum(x1[:, None] < x2[None, :]))
                cliffs_d = (gt - lt) / float(n1 * n2)
                if n1 + n2 >= 3:
                    ss1 = float(np.sum((x1 - np.mean(x1)) ** 2)) if n1 > 1 else 0.0
                    ss2 = float(np.sum((x2 - np.mean(x2)) ** 2)) if n2 > 1 else 0.0
                    s_pool = math.sqrt((ss1 + ss2) / float(n1 + n2 - 2))
                    if s_pool > 1e-12:
                        cohen_d = mean_diff / s_pool
                        j_corr = 1.0 - (3.0 / (4.0 * float(n1 + n2 - 2) - 1.0))
                        hedges_g = cohen_d * j_corr
                    elif abs(mean_diff) <= 1e-12:
                        hedges_g = 0.0
                # Deterministic percentile bootstrap (n_boot=500)
                idx1 = rng.integers(0, n1, size=(500, n1))
                idx2 = rng.integers(0, n2, size=(500, n2))
                boot_diffs = x1[idx1].mean(axis=1) - x2[idx2].mean(axis=1)
                ci_low = float(np.percentile(boot_diffs, 2.5))
                ci_high = float(np.percentile(boot_diffs, 97.5))

        # Responder contingency test
        resp_table = np.asarray(
            [
                [
                    sum(1 for r in by_beh_group[(beh, g)] if not r.censored),
                    sum(1 for r in by_beh_group[(beh, g)] if r.censored),
                ]
                for g in active_groups
            ],
            dtype=np.int64,
        )
        resp_p = (
            _fisher_exact_2x2(resp_table)
            if k == 2
            else _chi2_contingency_exact(resp_table)
        )

        # Kaplan-Meier median survival & Mantel-Cox log-rank test
        km_medians: Dict[str, Optional[float]] = {}
        surv_by_group: Dict[str, List[Tuple[float, bool]]] = {}
        for g in active_groups:
            g_rows = by_beh_group[(beh, g)]
            t_list = [float(r.censored_latency_s) for r in g_rows]
            e_list = [not bool(r.censored) for r in g_rows]
            km_medians[g] = _kaplan_meier_median(t_list, e_list)
            surv_by_group[g] = list(zip(t_list, e_list))
        lr_chi2, lr_p = _logrank_test(surv_by_group, active_groups)

        raw_rows.append(
            {
                "behavior": beh,
                "category": category_lookup.get(beh, ""),
                "groups": tuple(active_groups),
                "analysis_unit": analysis_unit,
                "parametric_test": param_name,
                "parametric_stat": param_stat,
                "p_parametric": max(0.0, min(1.0, p_param)),
                "nonparametric_test": nonparam_name,
                "nonparametric_stat": nonparam_stat,
                "p_nonparametric": max(0.0, min(1.0, p_nonparam)),
                "hedges_g": hedges_g,
                "cliffs_delta": cliffs_d,
                "mean_diff_s": mean_diff,
                "mean_diff_ci95_low_s": ci_low,
                "mean_diff_ci95_high_s": ci_high,
                "responder_p_value": max(0.0, min(1.0, resp_p)),
                "km_median_latency_s": km_medians,
                "logrank_chi2": lr_chi2,
                "logrank_p_value": max(0.0, min(1.0, lr_p)),
            }
        )

    if not raw_rows:
        return ()

    q_param = _benjamini_hochberg([r["p_parametric"] for r in raw_rows])
    q_nonparam = _benjamini_hochberg([r["p_nonparametric"] for r in raw_rows])

    out: List[CohortComparisonRow] = []
    for idx, r in enumerate(raw_rows):
        out.append(
            CohortComparisonRow(
                behavior=r["behavior"],
                category=r["category"],
                groups=r["groups"],
                analysis_unit=r["analysis_unit"],
                parametric_test=r["parametric_test"],
                parametric_stat=r["parametric_stat"],
                p_parametric=r["p_parametric"],
                q_parametric_bh=q_param[idx],
                nonparametric_test=r["nonparametric_test"],
                nonparametric_stat=r["nonparametric_stat"],
                p_nonparametric=r["p_nonparametric"],
                q_nonparametric_bh=q_nonparam[idx],
                hedges_g=r["hedges_g"],
                cliffs_delta=r["cliffs_delta"],
                mean_diff_s=r["mean_diff_s"],
                mean_diff_ci95_low_s=r["mean_diff_ci95_low_s"],
                mean_diff_ci95_high_s=r["mean_diff_ci95_high_s"],
                responder_p_value=r["responder_p_value"],
                km_median_latency_s=r["km_median_latency_s"],
                logrank_chi2=r["logrank_chi2"],
                logrank_p_value=r["logrank_p_value"],
            )
        )
    return tuple(out)


def _aggregate_subject_rows_by_recording(
    subject_rows: Sequence[CohortSubjectRow],
) -> List[CohortSubjectRow]:
    """Collapse nested subjects within the same ``(recording_id, group, behavior)``."""
    buckets: Dict[Tuple[str, str, str], List[CohortSubjectRow]] = defaultdict(list)
    for row in subject_rows:
        buckets[(row.recording_id, row.group, row.behavior)].append(row)

    collapsed: List[CohortSubjectRow] = []
    for (rec_id, grp, beh), items in sorted(buckets.items()):
        if len(items) == 1:
            collapsed.append(items[0])
            continue
        dur_s = float(mean(r.session_duration_s for r in items))
        occ_mean = float(mean(r.occurrences for r in items))
        rate = float(mean(r.rate_per_min for r in items))
        tot_dur = float(mean(r.total_duration_s for r in items))
        pct = float(mean(r.percent_of_session for r in items))
        m_bout = float(mean(r.mean_bout_duration_s for r in items))
        med_bout = float(median(r.median_bout_duration_s for r in items))
        resp_lats = [
            r.latency_to_first_s
            for r in items
            if r.latency_to_first_s is not None and not r.censored
        ]
        if resp_lats:
            lat_first: Optional[float] = float(min(resp_lats))
            censored = False
            cens_lat = lat_first
        else:
            lat_first = None
            censored = True
            cens_lat = dur_s
        occ = (
            max(1, int(math.floor(occ_mean + 0.5)))
            if (not censored or occ_mean > 0.0)
            else 0
        )
        collapsed.append(
            CohortSubjectRow(
                recording_id=rec_id,
                subject="__recording_mean__",
                group=grp,
                behavior=beh,
                category=items[0].category,
                session_duration_s=dur_s,
                occurrences=occ,
                rate_per_min=rate,
                total_duration_s=tot_dur,
                percent_of_session=pct,
                mean_bout_duration_s=m_bout,
                median_bout_duration_s=med_bout,
                latency_to_first_s=lat_first,
                censored_latency_s=cens_lat,
                censored=censored,
                metadata=dict(items[0].metadata),
            )
        )
    return collapsed


def analyze_behavior_cohort(
    recordings: Sequence[CohortRecordingSpec],
    *,
    schema: Optional[ProjectSchema] = None,
    session_duration_override_s: Optional[float] = None,
    max_transition_gap_s: Optional[float] = None,
    include_self_transitions: bool = True,
    include_schema_behaviors: bool = False,
    strict: bool = False,
    analysis_unit: str = "subject",
) -> CohortBehaviorReport:
    """Aggregate behavioral metrics and transition matrices across a cohort of recordings."""
    if not recordings:
        raise CohortComputationError(
            "At least one recording is required for cohort analysis."
        )

    category_lookup: Dict[str, str] = {}
    if schema is not None:
        cats = schema.category_map()
        for beh in schema.behaviors:
            cat_name = (
                cats[beh.category_id].name
                if beh.category_id and beh.category_id in cats
                else (beh.category_id or "")
            )
            category_lookup[beh.code] = cat_name
            category_lookup[beh.name] = cat_name

    per_recording_data: List[
        Tuple[
            CohortRecordingSpec,
            List[str],
            List[BehaviorInterval],
            float,
            float,
        ]
    ] = []
    all_behaviors_set: set[str] = set()
    if include_schema_behaviors and schema is not None:
        for beh in schema.behaviors:
            all_behaviors_set.add(beh.code)

    warnings_out: List[str] = []

    for spec in recordings:
        csv_path = Path(spec.events_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Events CSV for recording '{spec.recording_id}' not found: {csv_path}"
            )
        parsed_rows = read_parsed_event_rows(csv_path)
        try:
            intervals, rec_warnings, t_min, t_max = _extract_recording_events(
                parsed_rows, schema=schema, strict=strict
            )
        except TimeBudgetComputationError as exc:
            raise CohortComputationError(
                f"{spec.recording_id} ({csv_path.name}): {exc}"
            ) from exc

        for w in rec_warnings:
            warnings_out.append(f"{spec.recording_id}: {w}")

        for iv in intervals:
            all_behaviors_set.add(iv.behavior)

        subjects_in_rec = sorted(
            {iv.subject for iv in intervals}
            | {r.subject for r in parsed_rows if r.subject}
            | set(spec.subject_groups.keys())
        )
        if not subjects_in_rec:
            subjects_in_rec = ["Subject 1"]

        start_s = (
            float(spec.session_start_s)
            if spec.session_start_s is not None
            else (min(0.0, float(t_min)) if t_min is not None else 0.0)
        )
        explicit_dur = (
            float(session_duration_override_s)
            if session_duration_override_s is not None
            else spec.session_duration_s
        )
        if explicit_dur is not None and explicit_dur > 0:
            duration_s = float(explicit_dur)
        elif t_max is not None:
            duration_s = max(0.0, float(t_max) - start_s)
        else:
            duration_s = 0.0

        per_recording_data.append(
            (spec, subjects_in_rec, intervals, start_s, duration_s)
        )

    ordered_behaviors = tuple(sorted(all_behaviors_set))
    subject_rows: List[CohortSubjectRow] = []
    sequences_by_group: Dict[str, List[List[BehaviorInterval]]] = defaultdict(list)
    groups_seen: set[str] = set()
    subject_unit_count = 0

    for spec, subjects_in_rec, intervals, start_s, duration_s in per_recording_data:
        by_subj: Dict[str, List[BehaviorInterval]] = defaultdict(list)
        for iv in intervals:
            by_subj[iv.subject].append(iv)

        for subj in subjects_in_rec:
            subject_unit_count += 1
            grp = spec.subject_groups.get(subj, spec.group) or "ungrouped"
            groups_seen.add(grp)
            subj_intervals = sorted(
                by_subj.get(subj, ()),
                key=lambda iv: (iv.start, iv.end, iv.behavior),
            )
            if subj_intervals:
                sequences_by_group[grp].append(subj_intervals)

            by_beh: Dict[str, List[BehaviorInterval]] = defaultdict(list)
            for iv in subj_intervals:
                by_beh[iv.behavior].append(iv)

            for beh in ordered_behaviors:
                beh_ivs = by_beh.get(beh, [])
                occ = len(beh_ivs)
                durations = [iv.duration for iv in beh_ivs]
                tot_dur = sum(durations)
                mean_dur = float(mean(durations)) if durations else 0.0
                med_dur = float(median(durations)) if durations else 0.0
                pct_session = (
                    (tot_dur / duration_s) * 100.0 if duration_s > 0.0 else 0.0
                )
                rate_min = (occ / (duration_s / 60.0)) if duration_s > 0.0 else 0.0
                if occ > 0:
                    first_onset = min(iv.start for iv in beh_ivs)
                    lat_first: Optional[float] = max(0.0, first_onset - start_s)
                    censored = False
                    censored_lat = lat_first
                else:
                    lat_first = None
                    censored = True
                    censored_lat = duration_s

                subject_rows.append(
                    CohortSubjectRow(
                        recording_id=spec.recording_id,
                        subject=subj,
                        group=grp,
                        behavior=beh,
                        category=category_lookup.get(beh, ""),
                        session_duration_s=duration_s,
                        occurrences=occ,
                        rate_per_min=rate_min,
                        total_duration_s=tot_dur,
                        percent_of_session=pct_session,
                        mean_bout_duration_s=mean_dur,
                        median_bout_duration_s=med_dur,
                        latency_to_first_s=lat_first,
                        censored_latency_s=censored_lat,
                        censored=censored,
                        metadata=dict(spec.metadata),
                    )
                )

    unit_norm = str(analysis_unit or "subject").strip().lower()
    if unit_norm not in {"subject", "recording"}:
        raise CohortComputationError(
            f"Unsupported analysis_unit '{analysis_unit}'. Expected 'subject' or 'recording'."
        )

    # Check nested pseudoreplication when unit == 'subject'
    subjects_per_rec_group: Counter[Tuple[str, str]] = Counter()
    seen_rec_subj: set[Tuple[str, str, str]] = set()
    for s_row in subject_rows:
        key3 = (s_row.recording_id, s_row.group, s_row.subject)
        if key3 not in seen_rec_subj:
            seen_rec_subj.add(key3)
            subjects_per_rec_group[(s_row.recording_id, s_row.group)] += 1
    if unit_norm == "subject" and any(c > 1 for c in subjects_per_rec_group.values()):
        warnings_out.append(
            "Nested subjects detected (potential pseudoreplication): multiple subjects share a "
            "recording_id within the same group; consider analysis_unit='recording' (--unit recording) "
            "if cage/dyad is the independent experimental unit."
        )

    summary_input_rows = (
        _aggregate_subject_rows_by_recording(subject_rows)
        if unit_norm == "recording"
        else subject_rows
    )

    # Aggregate Group × Behavior summary rows across ALL units in each group.
    rows_by_group_beh: Dict[Tuple[str, str], List[CohortSubjectRow]] = defaultdict(list)
    for s_row in summary_input_rows:
        rows_by_group_beh[(s_row.group, s_row.behavior)].append(s_row)

    ordered_groups = tuple(sorted(groups_seen))
    group_summaries: List[CohortGroupSummaryRow] = []

    for grp in ordered_groups:
        for beh in ordered_behaviors:
            cell_rows = rows_by_group_beh.get((grp, beh), [])
            if not cell_rows:
                continue
            n_subj = len(cell_rows)
            responder_lats = [
                r.latency_to_first_s
                for r in cell_rows
                if r.latency_to_first_s is not None and not r.censored
            ]
            n_resp = len(responder_lats)
            resp_frac = n_resp / float(n_subj) if n_subj > 0 else 0.0

            dur_mean, dur_sd, dur_sem, dur_med, dur_iqr = _dispersion(
                [r.total_duration_s for r in cell_rows]
            )
            pct_mean, _pct_sd, pct_sem, _pct_med, _pct_iqr = _dispersion(
                [r.percent_of_session for r in cell_rows]
            )
            occ_mean = float(mean(r.occurrences for r in cell_rows))
            rate_mean, _rate_sd, rate_sem, _rate_med, _rate_iqr = _dispersion(
                [r.rate_per_min for r in cell_rows]
            )
            cens_lats = [r.censored_latency_s for r in cell_rows]
            cens_mean = float(mean(cens_lats)) if cens_lats else 0.0
            cens_med = float(median(cens_lats)) if cens_lats else 0.0

            resp_lat_mean = float(mean(responder_lats)) if responder_lats else None
            resp_lat_med = float(median(responder_lats)) if responder_lats else None

            group_summaries.append(
                CohortGroupSummaryRow(
                    group=grp,
                    behavior=beh,
                    category=category_lookup.get(beh, ""),
                    n_subjects=n_subj,
                    n_responders=n_resp,
                    responder_fraction=resp_frac,
                    mean_total_duration_s=dur_mean,
                    sd_total_duration_s=dur_sd,
                    sem_total_duration_s=dur_sem,
                    median_total_duration_s=dur_med,
                    iqr_total_duration_s=dur_iqr,
                    mean_percent_of_session=pct_mean,
                    sem_percent_of_session=pct_sem,
                    mean_occurrences=occ_mean,
                    mean_rate_per_min=rate_mean,
                    sem_rate_per_min=rate_sem,
                    mean_latency_responders_s=resp_lat_mean,
                    median_latency_responders_s=resp_lat_med,
                    mean_latency_censored_s=cens_mean,
                    median_latency_censored_s=cens_med,
                )
            )

    transitions = compute_transition_matrix(
        sequences_by_group,
        max_transition_gap_s=max_transition_gap_s,
        include_self_transitions=include_self_transitions,
    )
    comparisons = compute_cohort_comparisons(
        summary_input_rows,
        ordered_groups=ordered_groups,
        ordered_behaviors=ordered_behaviors,
        category_lookup=category_lookup,
        analysis_unit=unit_norm,
    )

    return CohortBehaviorReport(
        recording_count=len(recordings),
        subject_unit_count=subject_unit_count,
        groups=ordered_groups,
        behaviors=ordered_behaviors,
        subject_rows=tuple(subject_rows),
        group_summaries=tuple(group_summaries),
        transitions=transitions,
        warnings=tuple(warnings_out),
        comparisons=comparisons,
        analysis_unit=unit_norm,
    )


def format_cohort_report_table(report: CohortBehaviorReport) -> str:
    """Format a publication-style cohort summary and transition probability table."""
    lines: List[str] = [
        (
            f"Recordings: {report.recording_count} | "
            f"Subject units: {report.subject_unit_count} | "
            f"Groups: {', '.join(report.groups) if report.groups else '(none)'}"
        ),
    ]
    if report.group_summaries:
        lines.append("")
        lines.append(
            f"{'Group':<14} {'Behavior':<18} {'Resp/N':>7} "
            f"{'MeanDur(s)':>11} {'SEM':>8} {'%Sess':>7} "
            f"{'Rate/m':>8} {'LatResp(s)':>11}"
        )
        lines.append("-" * 90)
        for row in report.group_summaries:
            resp_str = f"{row.n_responders}/{row.n_subjects}"
            lat_str = (
                f"{row.mean_latency_responders_s:.2f}"
                if row.mean_latency_responders_s is not None
                else "n/a"
            )
            lines.append(
                f"{row.group:<14} {row.behavior:<18} {resp_str:>7} "
                f"{row.mean_total_duration_s:>11.2f} {row.sem_total_duration_s:>8.2f} "
                f"{row.mean_percent_of_session:>6.1f}% "
                f"{row.mean_rate_per_min:>8.2f} {lat_str:>11}"
            )
    else:
        lines.append("")
        lines.append("No behavioral events observed across the cohort.")

    if report.comparisons:
        lines.append("")
        lines.append(f"Statistical Group Comparisons (unit={report.analysis_unit}):")
        lines.append(
            f"{'Behavior':<18} {'p(param)':>9} {'q(BH)':>8} {'p(nonp)':>9} "
            f"{'Hedges g':>9} {'Cliffs d':>9} {'LogRank p':>10}"
        )
        lines.append("-" * 80)
        for comp in report.comparisons:
            g_str = f"{comp.hedges_g:+.3f}" if comp.hedges_g is not None else "n/a"
            d_str = (
                f"{comp.cliffs_delta:+.3f}" if comp.cliffs_delta is not None else "n/a"
            )
            lines.append(
                f"{comp.behavior:<18} {comp.p_parametric:>9.4f} {comp.q_parametric_bh:>8.4f} "
                f"{comp.p_nonparametric:>9.4f} {g_str:>9} {d_str:>9} {comp.logrank_p_value:>10.4f}"
            )

    non_pooled_edges = [e for e in report.transitions if e.group != "__all__"]
    if non_pooled_edges:
        lines.append("")
        lines.append("Top Behavioral Transitions:")
        lines.append(
            f"{'Group':<14} {'From':<18} {'To':<18} {'Count':>6} {'P(To|From)':>11} {'Lift':>7}"
        )
        lines.append("-" * 80)
        for edge in non_pooled_edges[:25]:
            lines.append(
                f"{edge.group:<14} {edge.from_behavior:<18} {edge.to_behavior:<18} "
                f"{edge.count:>6d} {edge.probability:>11.3f} {edge.lift:>7.2f}"
            )

    return "\n".join(lines)


def write_cohort_report_json(
    report: CohortBehaviorReport, output_path: Path | str
) -> Path:
    """Atomically write the full ``CohortBehaviorReport`` to JSON."""
    dst = Path(output_path)
    atomic_write_json(dst, report.to_dict(), indent=2, ensure_ascii=False)
    return dst


def write_cohort_group_summary_csv(
    rows: Iterable[CohortGroupSummaryRow], output_path: Path | str
) -> Path:
    """Atomically write group-level behavior summary statistics to CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Group",
            "Behavior",
            "Category",
            "NSubjects",
            "NResponders",
            "ResponderFraction",
            "MeanDurationSeconds",
            "SDDurationSeconds",
            "SEMDurationSeconds",
            "MedianDurationSeconds",
            "IQRDurationSeconds",
            "MeanPercentOfSession",
            "SEMPercentOfSession",
            "MeanOccurrences",
            "MeanRatePerMin",
            "SEMRatePerMin",
            "MeanLatencyRespondersSeconds",
            "MedianLatencyRespondersSeconds",
            "MeanLatencyCensoredSeconds",
            "MedianLatencyCensoredSeconds",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.group,
                r.behavior,
                r.category,
                r.n_subjects,
                r.n_responders,
                f"{r.responder_fraction:.6f}",
                f"{r.mean_total_duration_s:.6f}",
                f"{r.sd_total_duration_s:.6f}",
                f"{r.sem_total_duration_s:.6f}",
                f"{r.median_total_duration_s:.6f}",
                f"{r.iqr_total_duration_s:.6f}",
                f"{r.mean_percent_of_session:.6f}",
                f"{r.sem_percent_of_session:.6f}",
                f"{r.mean_occurrences:.6f}",
                f"{r.mean_rate_per_min:.6f}",
                f"{r.sem_rate_per_min:.6f}",
                (
                    f"{r.mean_latency_responders_s:.6f}"
                    if r.mean_latency_responders_s is not None
                    else ""
                ),
                (
                    f"{r.median_latency_responders_s:.6f}"
                    if r.median_latency_responders_s is not None
                    else ""
                ),
                f"{r.mean_latency_censored_s:.6f}",
                f"{r.median_latency_censored_s:.6f}",
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def write_cohort_subject_rows_csv(
    rows: Iterable[CohortSubjectRow], output_path: Path | str
) -> Path:
    """Atomically write per-(recording, subject, behavior) tidy cohort rows to CSV."""
    dst = Path(output_path)
    row_list = list(rows)
    extra_keys = sorted({k for r in row_list for k in r.metadata})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "RecordingID",
            "Subject",
            "Group",
            "Behavior",
            "Category",
            "SessionDurationSeconds",
            "Occurrences",
            "RatePerMin",
            "TotalDurationSeconds",
            "PercentOfSession",
            "MeanBoutDurationSeconds",
            "MedianBoutDurationSeconds",
            "LatencyToFirstSeconds",
            "CensoredLatencySeconds",
            "Censored",
            *extra_keys,
        ]
    )
    for r in row_list:
        writer.writerow(
            [
                r.recording_id,
                r.subject,
                r.group,
                r.behavior,
                r.category,
                f"{r.session_duration_s:.6f}",
                r.occurrences,
                f"{r.rate_per_min:.6f}",
                f"{r.total_duration_s:.6f}",
                f"{r.percent_of_session:.6f}",
                f"{r.mean_bout_duration_s:.6f}",
                f"{r.median_bout_duration_s:.6f}",
                (
                    f"{r.latency_to_first_s:.6f}"
                    if r.latency_to_first_s is not None
                    else ""
                ),
                f"{r.censored_latency_s:.6f}",
                str(bool(r.censored)).lower(),
                *[r.metadata.get(k, "") for k in extra_keys],
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def write_cohort_comparisons_csv(
    comparisons: Iterable[CohortComparisonRow], output_path: Path | str
) -> Path:
    """Atomically write inferential group comparison rows to CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Behavior",
            "Category",
            "Groups",
            "AnalysisUnit",
            "ParametricTest",
            "ParametricStat",
            "PParametric",
            "QParametricBH",
            "NonparametricTest",
            "NonparametricStat",
            "PNonparametric",
            "QNonparametricBH",
            "HedgesG",
            "CliffsDelta",
            "MeanDiffSeconds",
            "MeanDiffCI95LowSeconds",
            "MeanDiffCI95HighSeconds",
            "ResponderPValue",
            "LogRankChi2",
            "LogRankPValue",
            "KMMedianLatencyJSON",
        ]
    )
    for c in comparisons:
        writer.writerow(
            [
                c.behavior,
                c.category,
                "|".join(c.groups),
                c.analysis_unit,
                c.parametric_test,
                f"{c.parametric_stat:.6f}",
                f"{c.p_parametric:.6f}",
                f"{c.q_parametric_bh:.6f}",
                c.nonparametric_test,
                f"{c.nonparametric_stat:.6f}",
                f"{c.p_nonparametric:.6f}",
                f"{c.q_nonparametric_bh:.6f}",
                f"{c.hedges_g:.6f}" if c.hedges_g is not None else "",
                f"{c.cliffs_delta:.6f}" if c.cliffs_delta is not None else "",
                f"{c.mean_diff_s:.6f}" if c.mean_diff_s is not None else "",
                (
                    f"{c.mean_diff_ci95_low_s:.6f}"
                    if c.mean_diff_ci95_low_s is not None
                    else ""
                ),
                (
                    f"{c.mean_diff_ci95_high_s:.6f}"
                    if c.mean_diff_ci95_high_s is not None
                    else ""
                ),
                f"{c.responder_p_value:.6f}",
                f"{c.logrank_chi2:.6f}",
                f"{c.logrank_p_value:.6f}",
                json.dumps(c.km_median_latency_s, sort_keys=True),
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def write_transition_matrix_csv(
    edges: Iterable[TransitionEdge], output_path: Path | str
) -> Path:
    """Atomically write behavioral Markov transition matrix edges to CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Group",
            "FromBehavior",
            "ToBehavior",
            "Count",
            "FromTotal",
            "Probability",
            "TargetBaseRate",
            "Lift",
        ]
    )
    for e in edges:
        writer.writerow(
            [
                e.group,
                e.from_behavior,
                e.to_behavior,
                e.count,
                e.from_total,
                f"{e.probability:.6f}",
                f"{e.target_base_rate:.6f}",
                f"{e.lift:.6f}",
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst
