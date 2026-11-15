"""Deterministic disambiguation for plain keypoints inside overlapping masks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - optional scipy acceleration
    _distance_transform_edt = None  # type: ignore[assignment]

VALID_AMBIGUOUS_POINT_POLICIES: Tuple[str, ...] = (
    "withhold",
    "error",
    "polygon_order",
)


class AmbiguousKeypointError(ValueError):
    """Raised when ``ambiguous_point_policy='error'`` rejects unresolved points."""

    def __init__(
        self,
        message: str,
        *,
        ambiguous_points: Sequence[Mapping[str, object]] = (),
    ) -> None:
        super().__init__(message)
        self.ambiguous_points: Tuple[Dict[str, object], ...] = tuple(
            dict(item) for item in ambiguous_points
        )


def normalize_ambiguous_point_policy(
    policy: Optional[str],
    *,
    strict: bool = False,
) -> str:
    """Normalize an ambiguous-point policy token."""
    norm = str(policy or "withhold").strip().lower()
    if norm in VALID_AMBIGUOUS_POINT_POLICIES:
        return norm
    if strict:
        allowed = ", ".join(VALID_AMBIGUOUS_POINT_POLICIES)
        raise ValueError(
            f"Unsupported ambiguous_point_policy={policy!r}; expected one of: {allowed}"
        )
    return "withhold"


@dataclass(frozen=True, slots=True)
class OverlappingPointCandidate:
    """An unassociated point falling inside two or more polygon masks."""

    index: int
    keypoint_label: str
    x: float
    y: float
    containing_instances: Tuple[str, ...]


def match_instance_prefix(
    label: str,
    known_instances: Iterable[str],
    separators: str = "_-:| ",
) -> Optional[Tuple[str, str]]:
    """Match ``<instance><sep><keypoint>`` against known frame polygon instances."""
    raw = str(label or "").strip()
    if not raw:
        return None
    candidates = sorted(
        {str(inst).strip() for inst in known_instances if str(inst).strip()},
        key=len,
        reverse=True,
    )
    lower_raw = raw.lower()
    for inst in candidates:
        if len(raw) <= len(inst):
            continue
        if not lower_raw.startswith(inst.lower()):
            continue
        sep = raw[len(inst)]
        if sep not in separators:
            continue
        remainder = raw[len(inst) + 1 :].lstrip(separators).strip()
        if remainder:
            return inst, remainder
    return None


def containing_instances_for_point(
    x: float,
    y: float,
    polygon_masks: Sequence[Tuple[str, np.ndarray]],
) -> List[str]:
    """Return unique instance labels whose mask contains ``(round(x), round(y))``."""
    pixel_x = int(round(float(x)))
    pixel_y = int(round(float(y)))
    containing: List[str] = []
    for instance_label, mask in polygon_masks:
        if (
            0 <= pixel_y < int(mask.shape[0])
            and 0 <= pixel_x < int(mask.shape[1])
            and bool(mask[pixel_y, pixel_x])
        ):
            containing.append(instance_label)
    return list(dict.fromkeys(containing))


def _enumerate_injective_assignments(
    avail_lists: Sequence[Sequence[str]],
) -> List[Tuple[str, ...]]:
    results: List[Tuple[str, ...]] = []
    used: set[str] = set()
    current: List[str] = []

    def _backtrack(pos: int) -> None:
        if pos == len(avail_lists):
            results.append(tuple(current))
            return
        for inst in avail_lists[pos]:
            if inst in used:
                continue
            used.add(inst)
            current.append(inst)
            _backtrack(pos + 1)
            current.pop()
            used.remove(inst)

    _backtrack(0)
    return results


def resolve_overlapping_point_instances(
    *,
    polygon_masks: Sequence[Tuple[str, np.ndarray]],
    assigned_keypoints: Mapping[str, Sequence[Tuple[str, float, float]]],
    candidates: Sequence[OverlappingPointCandidate],
    prior_keypoints: Optional[Mapping[str, Sequence[Tuple[str, float, float]]]] = None,
    multi_point_margin_ratio: float = 0.75,
    single_point_margin_ratio: float = 0.5,
    nested_area_margin_ratio: float = 0.5,
) -> Dict[int, str]:
    """Resolve overlapping-polygon point ownership when provably unambiguous.

    Resolution proceeds conservatively:
    1. Iterative landmark-uniqueness elimination (only one containing polygon
       still lacks that landmark).
    2. 1-to-1 minimum-distance bipartite matching against each instance's
       already-assigned pose cluster (or exclusive non-overlapping mask core),
       accepted only when separated from the runner-up assignment by a strict
       relative margin.
    3. Temporal continuity matching against ``prior_keypoints`` from the
       preceding/nearest resolved frame when single-frame masks are concentric
       or unanchored.
    4. Nested concentric polygon scale & interior-depth disambiguation when one
       polygon is strictly nested inside a significantly larger enclosing
       polygon (``area_small <= nested_area_margin_ratio * area_large``).
       Identical/symmetric polygons remain unresolved rather than guessed.
    """
    if not candidates or not polygon_masks:
        return {}

    mask_by_inst: Dict[str, np.ndarray] = {}
    mask_area: Dict[str, float] = {}
    for inst, mask in polygon_masks:
        if inst not in mask_by_inst:
            arr = np.asarray(mask, dtype=bool)
            mask_by_inst[inst] = arr
            mask_area[inst] = float(np.count_nonzero(arr))

    overlap_counts: Optional[np.ndarray] = None
    for mask in mask_by_inst.values():
        if overlap_counts is None:
            overlap_counts = mask.astype(np.int32)
        elif overlap_counts.shape == mask.shape:
            overlap_counts = overlap_counts + mask.astype(np.int32)

    core_xy: Dict[str, Optional[Tuple[float, float]]] = {}
    has_distinct_core: Dict[str, bool] = {}
    for inst, mask in mask_by_inst.items():
        if overlap_counts is not None and overlap_counts.shape == mask.shape:
            exclusive = mask & (overlap_counts == 1)
        else:
            exclusive = np.zeros_like(mask, dtype=bool)
        if np.any(exclusive):
            ys, xs = np.nonzero(exclusive)
            core_xy[inst] = (float(xs.mean()), float(ys.mean()))
            has_distinct_core[inst] = True
        elif np.any(mask):
            ys, xs = np.nonzero(mask)
            core_xy[inst] = (float(xs.mean()), float(ys.mean()))
            has_distinct_core[inst] = False
        else:
            core_xy[inst] = None
            has_distinct_core[inst] = False

    depth_cache: Dict[str, Tuple[Optional[np.ndarray], float]] = {}

    def _normalized_interior_depth(inst: str, x: float, y: float) -> float:
        if inst not in depth_cache:
            mask = mask_by_inst.get(inst)
            if mask is None or not np.any(mask):
                depth_cache[inst] = (None, 0.0)
            elif _distance_transform_edt is not None:
                padded = np.pad(mask, 1, mode="constant", constant_values=False)
                edt = np.asarray(_distance_transform_edt(padded), dtype=np.float32)[
                    1:-1, 1:-1
                ]
                depth_cache[inst] = (edt, float(edt.max()))
            else:
                ys, xs = np.nonzero(mask)
                x0, x1 = float(xs.min()), float(xs.max())
                y0, y1 = float(ys.min()), float(ys.max())
                yy, xx = np.indices(mask.shape, dtype=np.float32)
                d = np.minimum(
                    np.minimum(xx - x0 + 1.0, x1 - xx + 1.0),
                    np.minimum(yy - y0 + 1.0, y1 - yy + 1.0),
                )
                d = np.where(mask, np.maximum(0.0, d), 0.0)
                depth_cache[inst] = (d, float(d.max()))
        edt_map, max_d = depth_cache[inst]
        if edt_map is None or max_d <= 1e-6:
            return 0.0
        px = int(round(float(x)))
        py = int(round(float(y)))
        if 0 <= py < int(edt_map.shape[0]) and 0 <= px < int(edt_map.shape[1]):
            return float(edt_map[py, px]) / max_d
        return 0.0

    def _is_nested_pair(small_inst: str, large_inst: str) -> bool:
        a_small = mask_area.get(small_inst, 0.0)
        a_large = mask_area.get(large_inst, 0.0)
        if a_small <= 0.0 or a_large <= 0.0:
            return False
        if a_small > float(nested_area_margin_ratio) * a_large:
            return False
        m_small = mask_by_inst.get(small_inst)
        m_large = mask_by_inst.get(large_inst)
        if m_small is None or m_large is None or m_small.shape != m_large.shape:
            return False
        shared = float(np.count_nonzero(m_small & m_large))
        return (shared / a_small) >= 0.85

    owned_labels: Dict[str, set[str]] = defaultdict(set)
    instance_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for inst, pts in assigned_keypoints.items():
        for kpt_label, px, py in pts:
            norm = str(kpt_label or "").strip().lower()
            if norm:
                owned_labels[inst].add(norm)
            if math.isfinite(px) and math.isfinite(py):
                instance_points[inst].append((float(px), float(py)))

    prior_by_inst_label: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    prior_inst_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    if prior_keypoints:
        for inst, pts in prior_keypoints.items():
            for kpt_label, px, py in pts:
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue
                coord = (float(px), float(py))
                norm = str(kpt_label or "").strip().lower()
                if norm and norm not in prior_by_inst_label[inst]:
                    prior_by_inst_label[inst][norm] = coord
                prior_inst_points[inst].append(coord)

    def _anchor(inst: str) -> Tuple[Optional[Tuple[float, float]], bool]:
        pts = instance_points.get(inst)
        if pts:
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            return (float(mx), float(my)), True
        return core_xy.get(inst), bool(has_distinct_core.get(inst, False))

    def _temporal_anchor(inst: str, norm_label: str) -> Optional[Tuple[float, float]]:
        by_lbl = prior_by_inst_label.get(inst)
        if by_lbl and norm_label in by_lbl:
            return by_lbl[norm_label]
        cur_pts = instance_points.get(inst)
        if cur_pts:
            mx = sum(p[0] for p in cur_pts) / len(cur_pts)
            my = sum(p[1] for p in cur_pts) / len(cur_pts)
            return (float(mx), float(my))
        prev_pts = prior_inst_points.get(inst)
        if prev_pts:
            mx = sum(p[0] for p in prev_pts) / len(prev_pts)
            my = sum(p[1] for p in prev_pts) / len(prev_pts)
            return (float(mx), float(my))
        return None

    resolved: Dict[int, str] = {}

    def _commit(cand: OverlappingPointCandidate, inst: str) -> None:
        resolved[cand.index] = inst
        norm = cand.keypoint_label.strip().lower()
        if norm:
            owned_labels[inst].add(norm)
        instance_points[inst].append((float(cand.x), float(cand.y)))

    while True:
        # Pass 1: Iterative landmark-uniqueness elimination.
        while True:
            step_progress = False
            by_label: Dict[str, List[OverlappingPointCandidate]] = defaultdict(list)
            for cand in candidates:
                if cand.index in resolved:
                    continue
                by_label[cand.keypoint_label.strip().lower()].append(cand)

            for norm_label, group in by_label.items():
                sole_claims: Counter[str] = Counter()
                avail_map: Dict[int, List[str]] = {}
                for cand in group:
                    avail = [
                        inst
                        for inst in cand.containing_instances
                        if norm_label not in owned_labels[inst]
                    ]
                    avail_map[cand.index] = avail
                    if len(avail) == 1:
                        sole_claims[avail[0]] += 1

                for cand in group:
                    avail = avail_map[cand.index]
                    if len(avail) == 1 and sole_claims[avail[0]] == 1:
                        _commit(cand, avail[0])
                        step_progress = True

            if not step_progress:
                break

        # Pass 2: Combinatorial or anchored 1-to-1 bipartite assignment.
        by_label = defaultdict(list)
        for cand in candidates:
            if cand.index in resolved:
                continue
            by_label[cand.keypoint_label.strip().lower()].append(cand)

        bipartite_progress = False
        for norm_label, group in by_label.items():
            active: List[OverlappingPointCandidate] = []
            avail_lists: List[List[str]] = []
            for cand in group:
                avail = [
                    inst
                    for inst in cand.containing_instances
                    if norm_label not in owned_labels[inst]
                ]
                if avail:
                    active.append(cand)
                    avail_lists.append(avail)

            if not active:
                continue

            if len(active) == 1:
                cand = active[0]
                avail = avail_lists[0]
                if len(avail) < 2:
                    continue
                scored_single: List[Tuple[float, str, bool]] = []
                valid_anchors = True
                for inst in avail:
                    anchor_xy, is_distinct = _anchor(inst)
                    if anchor_xy is None:
                        valid_anchors = False
                        break
                    dist = math.hypot(cand.x - anchor_xy[0], cand.y - anchor_xy[1])
                    scored_single.append((dist, inst, is_distinct))
                if not valid_anchors or len(scored_single) < 2:
                    continue
                scored_single.sort(key=lambda item: item[0])
                best_d, best_inst, best_distinct = scored_single[0]
                second_d, _, _ = scored_single[1]
                if (
                    best_distinct
                    and second_d > 1e-6
                    and best_d <= float(single_point_margin_ratio) * second_d
                ):
                    _commit(cand, best_inst)
                    bipartite_progress = True
                continue

            if len(active) <= 6:
                assignments = _enumerate_injective_assignments(avail_lists)
            else:
                assignments = []

            if len(assignments) == 1:
                for cand, inst in zip(active, assignments[0]):
                    _commit(cand, inst)
                bipartite_progress = True
                continue

            if len(assignments) >= 2:
                scored_multi: List[Tuple[float, Tuple[str, ...], bool]] = []
                for perm in assignments:
                    total_cost = 0.0
                    any_distinct = False
                    possible = True
                    for cand, inst in zip(active, perm):
                        anchor_xy, is_distinct = _anchor(inst)
                        if anchor_xy is None:
                            possible = False
                            break
                        any_distinct = any_distinct or is_distinct
                        total_cost += math.hypot(
                            cand.x - anchor_xy[0], cand.y - anchor_xy[1]
                        )
                    if possible:
                        scored_multi.append((total_cost, perm, any_distinct))

                if len(scored_multi) >= 2:
                    scored_multi.sort(key=lambda item: item[0])
                    best_cost, best_perm, best_distinct = scored_multi[0]
                    second_cost, _, _ = scored_multi[1]
                    if (
                        best_distinct
                        and second_cost > 1e-6
                        and best_cost <= float(multi_point_margin_ratio) * second_cost
                    ):
                        for cand, inst in zip(active, best_perm):
                            _commit(cand, inst)
                        bipartite_progress = True

        temporal_progress = False
        if not bipartite_progress and prior_inst_points:
            by_label = defaultdict(list)
            for cand in candidates:
                if cand.index in resolved:
                    continue
                by_label[cand.keypoint_label.strip().lower()].append(cand)

            for norm_label, group in by_label.items():
                active = []
                avail_lists = []
                for cand in group:
                    avail = [
                        inst
                        for inst in cand.containing_instances
                        if norm_label not in owned_labels[inst]
                    ]
                    if avail:
                        active.append(cand)
                        avail_lists.append(avail)

                if not active:
                    continue

                if len(active) == 1:
                    cand = active[0]
                    avail = avail_lists[0]
                    if len(avail) < 2:
                        continue
                    scored_temporal_single: List[Tuple[float, str]] = []
                    anchors_seen: List[Tuple[float, float]] = []
                    valid_temporal = True
                    for inst in avail:
                        t_anchor = _temporal_anchor(inst, norm_label)
                        if t_anchor is None:
                            valid_temporal = False
                            break
                        anchors_seen.append(t_anchor)
                        scored_temporal_single.append(
                            (
                                math.hypot(cand.x - t_anchor[0], cand.y - t_anchor[1]),
                                inst,
                            )
                        )
                    if not valid_temporal or len(scored_temporal_single) < 2:
                        continue
                    anchors_distinct = any(
                        math.hypot(a[0] - anchors_seen[0][0], a[1] - anchors_seen[0][1])
                        > 1e-6
                        for a in anchors_seen[1:]
                    )
                    if not anchors_distinct:
                        continue
                    scored_temporal_single.sort(key=lambda item: item[0])
                    best_d, best_inst = scored_temporal_single[0]
                    second_d, _ = scored_temporal_single[1]
                    if second_d > 1e-6 and best_d <= (
                        float(single_point_margin_ratio) * second_d
                    ):
                        _commit(cand, best_inst)
                        temporal_progress = True
                    continue

                if len(active) <= 6:
                    assignments = _enumerate_injective_assignments(avail_lists)
                else:
                    assignments = []

                if len(assignments) >= 2:
                    scored_temporal_multi: List[Tuple[float, Tuple[str, ...]]] = []
                    for perm in assignments:
                        total_cost = 0.0
                        possible = True
                        for cand, inst in zip(active, perm):
                            t_anchor = _temporal_anchor(inst, norm_label)
                            if t_anchor is None:
                                possible = False
                                break
                            total_cost += math.hypot(
                                cand.x - t_anchor[0], cand.y - t_anchor[1]
                            )
                        if possible:
                            scored_temporal_multi.append((total_cost, perm))
                    if len(scored_temporal_multi) >= 2:
                        scored_temporal_multi.sort(key=lambda item: item[0])
                        best_cost, best_perm = scored_temporal_multi[0]
                        second_cost, _ = scored_temporal_multi[1]
                        if second_cost > 1e-6 and best_cost <= (
                            float(multi_point_margin_ratio) * second_cost
                        ):
                            for cand, inst in zip(active, best_perm):
                                _commit(cand, inst)
                            temporal_progress = True

        nested_progress = False
        if not bipartite_progress and not temporal_progress:
            by_label = defaultdict(list)
            for cand in candidates:
                if cand.index in resolved:
                    continue
                by_label[cand.keypoint_label.strip().lower()].append(cand)

            for norm_label, group in by_label.items():
                active = []
                avail_lists = []
                for cand in group:
                    avail = [
                        inst
                        for inst in cand.containing_instances
                        if norm_label not in owned_labels[inst]
                    ]
                    if avail:
                        active.append(cand)
                        avail_lists.append(avail)

                if not active:
                    continue

                if len(active) == 1:
                    cand = active[0]
                    avail = avail_lists[0]
                    if len(avail) < 2:
                        continue
                    by_area = sorted(
                        avail,
                        key=lambda inst: (mask_area.get(inst, float("inf")), inst),
                    )
                    smallest, second = by_area[0], by_area[1]
                    if _is_nested_pair(smallest, second):
                        _commit(cand, smallest)
                        nested_progress = True
                    continue

                if len(active) <= 6:
                    assignments = _enumerate_injective_assignments(avail_lists)
                else:
                    assignments = []

                if len(assignments) >= 2:
                    involved = sorted({inst for avail in avail_lists for inst in avail})
                    has_nested_scale = any(
                        _is_nested_pair(u, v) or _is_nested_pair(v, u)
                        for idx_u, u in enumerate(involved)
                        for v in involved[idx_u + 1 :]
                    )
                    if not has_nested_scale:
                        continue
                    scored_nested: List[Tuple[float, Tuple[str, ...]]] = []
                    for perm in assignments:
                        deficit = 0.0
                        for cand, inst in zip(active, perm):
                            deficit += 1.0 - _normalized_interior_depth(
                                inst, cand.x, cand.y
                            )
                        scored_nested.append((deficit, perm))
                    scored_nested.sort(key=lambda item: item[0])
                    best_def, best_perm = scored_nested[0]
                    second_def, _ = scored_nested[1]
                    if second_def > 1e-6 and best_def <= (
                        float(multi_point_margin_ratio) * second_def
                    ):
                        for cand, inst in zip(active, best_perm):
                            _commit(cand, inst)
                        nested_progress = True

        if not bipartite_progress and not temporal_progress and not nested_progress:
            break

    return resolved


def apply_ambiguous_point_policy(
    *,
    policy: str,
    candidates: Sequence[OverlappingPointCandidate],
    resolved: Mapping[int, str],
    assigned_keypoints: Mapping[str, Sequence[Tuple[str, float, float]]],
    source_path: Optional[Path | str] = None,
) -> Dict[int, str]:
    """Apply ``ambiguous_point_policy`` after deterministic passes complete."""
    norm_policy = normalize_ambiguous_point_policy(policy, strict=True)
    merged: Dict[int, str] = dict(resolved)
    unresolved = [c for c in candidates if c.index not in merged]
    if not unresolved or norm_policy == "withhold":
        return merged

    if norm_policy == "polygon_order":
        owned_labels: Dict[str, set[str]] = defaultdict(set)
        for inst, pts in assigned_keypoints.items():
            for kpt_label, _px, _py in pts:
                norm = str(kpt_label or "").strip().lower()
                if norm:
                    owned_labels[inst].add(norm)
        cand_by_index = {c.index: c for c in candidates}
        for idx, inst in merged.items():
            cand_obj = cand_by_index.get(idx)
            if cand_obj is not None:
                norm = cand_obj.keypoint_label.strip().lower()
                if norm:
                    owned_labels[inst].add(norm)

        for cand in unresolved:
            norm = cand.keypoint_label.strip().lower()
            chosen: Optional[str] = None
            for inst in cand.containing_instances:
                if not norm or norm not in owned_labels[inst]:
                    chosen = inst
                    break
            if chosen is None and cand.containing_instances:
                chosen = cand.containing_instances[0]
            if chosen is not None:
                merged[cand.index] = chosen
                if norm:
                    owned_labels[chosen].add(norm)
        return merged

    # norm_policy == "error"
    records: List[Dict[str, object]] = [
        {
            "label": c.keypoint_label,
            "keypoint_label": c.keypoint_label,
            "x": float(c.x),
            "y": float(c.y),
            "containing_instances": list(c.containing_instances),
            "json_path": str(source_path) if source_path is not None else "",
        }
        for c in unresolved
    ]
    preview = ", ".join(
        f"'{r['label']}'@({r['x']:.1f},{r['y']:.1f})->{r['containing_instances']}"
        for r in records[:3]
    )
    if len(records) > 3:
        preview += f", ... (+{len(records) - 3} more)"
    where = f" in {source_path}" if source_path is not None else ""
    raise AmbiguousKeypointError(
        f"Ambiguous overlapping polygon keypoint ownership{where}: {preview}. "
        "Add group_id or instance_label metadata, run 'annolid-run track-qc', "
        "or set ambiguous_point_policy to 'withhold' or 'polygon_order'.",
        ambiguous_points=records,
    )


def build_local_polygon_raster(
    polygons: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    points_xy: Sequence[Tuple[float, float]] = (),
    *,
    image_hw: Optional[Tuple[int, int]] = None,
    max_side: int = 2048,
) -> Tuple[List[Tuple[str, np.ndarray]], float, float, float]:
    """Rasterize ``(instance_key, polygon_xy)`` into aligned boolean masks.

    Returns ``(polygon_masks, offset_x, offset_y, scale)`` where original image
    coordinates map to raster coordinates via ``((x - offset_x) * scale, (y - offset_y) * scale)``.
    """
    valid_polys: List[Tuple[str, List[Tuple[float, float]]]] = []
    xs: List[float] = []
    ys: List[float] = []
    for inst_key, poly in polygons:
        pts = [
            (float(pt[0]), float(pt[1]))
            for pt in poly
            if len(pt) >= 2 and math.isfinite(pt[0]) and math.isfinite(pt[1])
        ]
        if len(pts) >= 3:
            valid_polys.append((str(inst_key), pts))
            xs.extend(p[0] for p in pts)
            ys.extend(p[1] for p in pts)
    for px, py in points_xy:
        if math.isfinite(px) and math.isfinite(py):
            xs.append(float(px))
            ys.append(float(py))

    if not valid_polys:
        return [], 0.0, 0.0, 1.0

    if image_hw is not None and int(image_hw[0]) > 0 and int(image_hw[1]) > 0:
        offset_x = 0.0
        offset_y = 0.0
        span_h = float(image_hw[0])
        span_w = float(image_hw[1])
    else:
        offset_x = math.floor(min(xs)) - 2.0
        offset_y = math.floor(min(ys)) - 2.0
        span_w = max(8.0, math.ceil(max(xs) - offset_x) + 3.0)
        span_h = max(8.0, math.ceil(max(ys) - offset_y) + 3.0)

    longest = max(span_w, span_h)
    scale = (
        float(max_side) / longest if max_side > 0 and longest > float(max_side) else 1.0
    )
    grid_w = max(4, int(math.ceil(span_w * scale)))
    grid_h = max(4, int(math.ceil(span_h * scale)))

    masks: List[Tuple[str, np.ndarray]] = []
    for inst_key, pts in valid_polys:
        scaled_pts = [
            ((px - offset_x) * scale, (py - offset_y) * scale) for px, py in pts
        ]
        if scaled_pts[0] != scaled_pts[-1]:
            scaled_pts.append(scaled_pts[0])
        canvas = Image.new("L", (grid_w, grid_h), 0)
        ImageDraw.Draw(canvas).polygon(scaled_pts, outline=1, fill=1)
        masks.append((inst_key, np.asarray(canvas, dtype=bool)))

    return masks, offset_x, offset_y, scale


def find_neighbor_frame_paths(
    json_path: Path,
    *,
    max_neighbors: int = 2,
) -> List[Tuple[int, Path]]:
    """Return up to ``max_neighbors`` closest same-stem frame JSONs in ``json_path.parent``."""
    stem = json_path.stem
    if "_" not in stem:
        return []
    prefix, frame_token = stem.rsplit("_", 1)
    if not prefix or not frame_token.isdigit():
        return []
    current_frame = int(frame_token)
    pad_width = max(1, len(frame_token))
    parent = json_path.parent
    if not parent.exists() or not parent.is_dir():
        return []

    by_frame: Dict[int, Path] = {}
    prefix_lower = f"{prefix}_".lower()
    try:
        for entry in os.scandir(parent):
            if not entry.is_file() or not entry.name.lower().endswith(".json"):
                continue
            entry_stem = entry.name[:-5]
            if not entry_stem.lower().startswith(prefix_lower):
                continue
            suffix = entry_stem[len(prefix) + 1 :]
            if not suffix.isdigit():
                continue
            other_frame = int(suffix)
            if other_frame != current_frame:
                by_frame[other_frame] = Path(entry.path)
    except OSError:
        pass

    if parent.name.lower() == prefix.lower():
        try:
            from annolid.utils.annotation_store import AnnotationStore

            store = AnnotationStore.for_frame_path(json_path)
            if store.store_path.exists():
                for frame_idx in store.iter_frames():
                    other_frame = int(frame_idx)
                    if other_frame != current_frame and other_frame not in by_frame:
                        by_frame[other_frame] = (
                            parent / f"{prefix}_{other_frame:0{pad_width}d}.json"
                        )
        except Exception:
            pass

    ordered = sorted(
        by_frame.items(),
        key=lambda item: (
            abs(item[0] - current_frame),
            0 if item[0] < current_frame else 1,
            item[0],
        ),
    )
    return ordered[: max(0, int(max_neighbors))]
