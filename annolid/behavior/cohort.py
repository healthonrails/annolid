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
    "TransitionEdge",
    "CohortBehaviorReport",
    "load_cohort_manifest",
    "apply_cohort_metadata_csv",
    "analyze_behavior_cohort",
    "compute_transition_matrix",
    "format_cohort_report_table",
    "write_cohort_report_json",
    "write_cohort_group_summary_csv",
    "write_cohort_subject_rows_csv",
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_count": int(self.recording_count),
            "subject_unit_count": int(self.subject_unit_count),
            "groups": list(self.groups),
            "behaviors": list(self.behaviors),
            "group_summaries": [row.to_dict() for row in self.group_summaries],
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


def analyze_behavior_cohort(
    recordings: Sequence[CohortRecordingSpec],
    *,
    schema: Optional[ProjectSchema] = None,
    session_duration_override_s: Optional[float] = None,
    max_transition_gap_s: Optional[float] = None,
    include_self_transitions: bool = True,
    include_schema_behaviors: bool = False,
    strict: bool = False,
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

    # Aggregate Group × Behavior summary rows across ALL subject units in each group.
    rows_by_group_beh: Dict[Tuple[str, str], List[CohortSubjectRow]] = defaultdict(list)
    for s_row in subject_rows:
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

    return CohortBehaviorReport(
        recording_count=len(recordings),
        subject_unit_count=subject_unit_count,
        groups=ordered_groups,
        behaviors=ordered_behaviors,
        subject_rows=tuple(subject_rows),
        group_summaries=tuple(group_summaries),
        transitions=transitions,
        warnings=tuple(warnings_out),
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
