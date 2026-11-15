"""Helpers for parsing instance polygons + keypoints for DinoKPSEG tracking."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.tracking.overlap_assignment import (
    OverlappingPointCandidate,
    apply_ambiguous_point_policy,
    containing_instances_for_point,
    find_neighbor_frame_paths,
    match_instance_prefix,
    normalize_ambiguous_point_policy,
    resolve_overlapping_point_instances,
)
from annolid.utils.annotation_store import load_labelme_json
from annolid.utils.body_part_shapes import is_body_part_shape_payload
from annolid.utils.logger import logger
from annolid.utils.zone_shapes import is_zone_shape_payload


@dataclass(frozen=True)
class ManualAnnotation:
    frame_number: int
    registry: InstanceRegistry
    keypoints_by_instance: Dict[str, Dict[str, Tuple[float, float]]]
    display_labels: Dict[str, str]
    ambiguous_keypoints: Tuple[Dict[str, object], ...] = ()


class DinoKPSEGAnnotationParser:
    """Parse LabelMe JSON with polygon instances and point keypoints.

    Instances are keyed by group_id when available; otherwise stable group_ids are
    derived from polygon labels and/or first-seen ordering.
    """

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        adapter: AnnotationAdapter,
        ambiguous_point_policy: Optional[str] = None,
    ) -> None:
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.adapter = adapter
        raw_policy = (
            ambiguous_point_policy
            if ambiguous_point_policy is not None
            else getattr(adapter, "ambiguous_point_policy", "withhold")
        )
        self.ambiguous_point_policy = normalize_ambiguous_point_policy(
            raw_policy, strict=True
        )
        self._polygon_label_to_gid: Dict[str, int] = {}

    @staticmethod
    def _normalize_group_id(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):  # bool is also int
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                return int(raw)
        return None

    @staticmethod
    def _next_group_id(used: set[int]) -> int:
        candidate = max(used) + 1 if used else 0
        while candidate in used:
            candidate += 1
        return candidate

    @classmethod
    def _extract_prior_keypoints_for_gids(
        cls,
        source_registry: InstanceRegistry,
        polygon_gids: List[int],
        unique_label_to_gid: Dict[str, int],
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        src_by_gid: Dict[int, object] = {}
        for src_label, src_inst in source_registry.instances.items():
            norm_gid = cls._normalize_group_id(src_inst.group_id)
            if norm_gid is None:
                norm_gid = cls._normalize_group_id(src_label)
            if norm_gid is not None and int(norm_gid) not in src_by_gid:
                src_by_gid[int(norm_gid)] = src_inst

        gid_to_unique_label: Dict[int, str] = {
            int(u_gid): u_lbl for u_lbl, u_gid in unique_label_to_gid.items()
        }
        extracted: Dict[str, List[Tuple[str, float, float]]] = {}
        for gid in polygon_gids:
            gid_key = str(int(gid))
            src_inst = source_registry.instances.get(gid_key) or src_by_gid.get(
                int(gid)
            )
            if src_inst is None and int(gid) in gid_to_unique_label:
                u_lbl = gid_to_unique_label[int(gid)]
                src_inst = source_registry.instances.get(u_lbl)
            if src_inst is None or not getattr(src_inst, "keypoints", None):
                continue
            vis_pts = [
                (kp.label, float(kp.x), float(kp.y))
                for kp in src_inst.keypoints.values()
                if kp.visible and math.isfinite(kp.x) and math.isfinite(kp.y)
            ]
            if not vis_pts:
                vis_pts = [
                    (kp.label, float(kp.x), float(kp.y))
                    for kp in src_inst.keypoints.values()
                    if math.isfinite(kp.x) and math.isfinite(kp.y)
                ]
            if vis_pts:
                extracted[gid_key] = vis_pts
        return extracted

    def read_manual_annotation(
        self,
        frame_number: int,
        json_path: Path,
        *,
        prior_registry: Optional[InstanceRegistry] = None,
        ambiguous_point_policy: Optional[str] = None,
        _allow_neighbor_lookup: bool = True,
    ) -> ManualAnnotation:
        effective_policy = (
            "withhold"
            if not _allow_neighbor_lookup
            else normalize_ambiguous_point_policy(
                ambiguous_point_policy or self.ambiguous_point_policy,
                strict=True,
            )
        )
        payload = load_labelme_json(json_path)
        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list):
            shapes = []

        polygons: List[Tuple[int, str, List[Tuple[float, float]]]] = []
        used_gids: set[int] = set()
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if is_zone_shape_payload(shape) or is_body_part_shape_payload(shape):
                continue
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            if shape_type != "polygon":
                continue
            raw_label = str(shape.get("label", "")).strip()
            gid = self._normalize_group_id(shape.get("group_id"))
            if gid is None:
                if raw_label and raw_label in self._polygon_label_to_gid:
                    gid = int(self._polygon_label_to_gid[raw_label])
                else:
                    gid = self._next_group_id(used_gids)
            if int(gid) in used_gids:
                remapped = self._next_group_id(used_gids)
                logger.warning(
                    "Duplicate polygon group_id=%s in %s; remapped to %s.",
                    gid,
                    json_path,
                    remapped,
                )
                gid = remapped
            used_gids.add(int(gid))
            points = shape.get("points") or []
            polygon = [
                (float(pt[0]), float(pt[1]))
                for pt in points
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
            polygons.append((int(gid), raw_label, polygon))

        if not polygons:
            raise RuntimeError(
                f"No polygon instances found in {json_path}; Cutie+DinoKPSEG requires at least one instance polygon."
            )

        display_labels: Dict[str, str] = {}
        polygon_masks: Dict[int, np.ndarray] = {}
        label_counts: Counter[str] = Counter()
        for gid, raw_label, polygon in polygons:
            display = raw_label or f"instance_{gid}"
            display_labels[str(gid)] = display
            if raw_label:
                label_counts[raw_label.lower()] += 1
                self._polygon_label_to_gid[raw_label] = int(gid)
            if polygon:
                polygon_masks[gid] = self.adapter.mask_bitmap_from_polygon(polygon)

        unique_label_to_gid: Dict[str, int] = {}
        for gid, raw_label, _ in polygons:
            if raw_label and label_counts[raw_label.lower()] == 1:
                unique_label_to_gid[raw_label] = int(gid)

        str_polygon_masks: List[Tuple[str, np.ndarray]] = [
            (str(gid), mask) for gid, mask in polygon_masks.items()
        ]
        assigned_for_overlap: Dict[str, List[Tuple[str, float, float]]] = defaultdict(
            list
        )
        overlap_candidates: List[OverlappingPointCandidate] = []
        pending_points: List[
            Tuple[int, str, float, float, Optional[int], List[str]]
        ] = []

        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            if shape_type != "point":
                continue
            label = str(shape.get("label", "")).strip()
            if not label:
                continue
            gid = self._normalize_group_id(shape.get("group_id"))
            point = (shape.get("points") or [[None, None]])[0]
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            if point[0] is None or point[1] is None:
                continue
            x, y = float(point[0]), float(point[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            flags = shape.get("flags") if isinstance(shape.get("flags"), dict) else {}
            flag_inst = str(
                shape.get("instance_label") or flags.get("instance_label") or ""
            ).strip()
            flag_disp = str(
                shape.get("display_label") or flags.get("display_label") or ""
            ).strip()
            kpt_label = flag_disp or label

            resolved_gid: Optional[int] = None
            if flag_inst:
                for u_lbl, u_gid in unique_label_to_gid.items():
                    if u_lbl.lower() == flag_inst.lower():
                        resolved_gid = int(u_gid)
                        break
                if resolved_gid is None and flag_inst.isdigit():
                    cand_gid = int(flag_inst)
                    if cand_gid in polygon_masks:
                        resolved_gid = cand_gid
            elif unique_label_to_gid:
                prefixed = match_instance_prefix(label, unique_label_to_gid.keys())
                if prefixed is not None:
                    matched_inst, remainder = prefixed
                    resolved_gid = int(unique_label_to_gid[matched_inst])
                    kpt_label = flag_disp or remainder

            containing: List[str] = []
            if resolved_gid is None:
                if gid is not None:
                    gid_int = int(gid)
                    mask = polygon_masks.get(gid_int)
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    if (
                        mask is not None
                        and 0 <= xi < int(mask.shape[1])
                        and 0 <= yi < int(mask.shape[0])
                        and bool(mask[yi, xi])
                    ):
                        resolved_gid = gid_int

                if resolved_gid is None:
                    containing = containing_instances_for_point(x, y, str_polygon_masks)
                    if len(containing) == 1:
                        resolved_gid = int(containing[0])
                    elif not containing:
                        resolved_gid = self._assign_group_for_point(
                            x, y, polygon_masks, polygons
                        )

            rec_idx = len(pending_points)
            if resolved_gid is not None:
                assigned_for_overlap[str(int(resolved_gid))].append((kpt_label, x, y))
            elif len(containing) >= 2:
                overlap_candidates.append(
                    OverlappingPointCandidate(
                        index=rec_idx,
                        keypoint_label=kpt_label,
                        x=x,
                        y=y,
                        containing_instances=tuple(containing),
                    )
                )

            pending_points.append((rec_idx, kpt_label, x, y, resolved_gid, containing))

        polygon_gids = [int(gid) for gid, _, _ in polygons]
        prior_kps: Optional[Dict[str, List[Tuple[str, float, float]]]] = None
        if prior_registry is not None and overlap_candidates:
            prior_kps = self._extract_prior_keypoints_for_gids(
                prior_registry, polygon_gids, unique_label_to_gid
            )

        resolved_overlaps = resolve_overlapping_point_instances(
            polygon_masks=str_polygon_masks,
            assigned_keypoints=assigned_for_overlap,
            candidates=overlap_candidates,
            prior_keypoints=prior_kps,
        )

        if (
            len(resolved_overlaps) < len(overlap_candidates)
            and prior_registry is None
            and _allow_neighbor_lookup
        ):
            saved_gid_map = dict(self._polygon_label_to_gid)
            try:
                for neighbor_frame, neighbor_path in find_neighbor_frame_paths(
                    json_path, max_neighbors=2
                ):
                    try:
                        neighbor_manual = self.read_manual_annotation(
                            int(neighbor_frame),
                            neighbor_path,
                            _allow_neighbor_lookup=False,
                        )
                    except Exception:
                        continue
                    neighbor_prior = self._extract_prior_keypoints_for_gids(
                        neighbor_manual.registry,
                        polygon_gids,
                        unique_label_to_gid,
                    )
                    if not neighbor_prior:
                        continue
                    candidate_resolution = resolve_overlapping_point_instances(
                        polygon_masks=str_polygon_masks,
                        assigned_keypoints=assigned_for_overlap,
                        candidates=overlap_candidates,
                        prior_keypoints=neighbor_prior,
                    )
                    if len(candidate_resolution) > len(resolved_overlaps):
                        resolved_overlaps = candidate_resolution
                        if len(resolved_overlaps) == len(overlap_candidates):
                            break
            finally:
                self._polygon_label_to_gid = saved_gid_map

        if overlap_candidates and effective_policy != "withhold":
            resolved_overlaps = apply_ambiguous_point_policy(
                policy=effective_policy,
                candidates=overlap_candidates,
                resolved=resolved_overlaps,
                assigned_keypoints=assigned_for_overlap,
                source_path=json_path,
            )

        ambiguous_records: List[Dict[str, object]] = []
        keypoints_by_instance: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for rec_idx, kpt_label, x, y, resolved_gid, containing in pending_points:
            final_gid = resolved_gid
            if final_gid is None and rec_idx in resolved_overlaps:
                final_gid = int(resolved_overlaps[rec_idx])
            if final_gid is None:
                if len(containing) >= 2:
                    ambiguous_records.append(
                        {
                            "label": kpt_label,
                            "keypoint_label": kpt_label,
                            "x": float(x),
                            "y": float(y),
                            "containing_instances": list(containing),
                            "json_path": str(json_path),
                        }
                    )
                    logger.warning(
                        "Skipping keypoint '%s' at (%.1f, %.1f) in %s: ambiguous "
                        "overlapping polygon group_ids %s. Add group_id or "
                        "instance_label metadata.",
                        kpt_label,
                        x,
                        y,
                        json_path,
                        containing,
                    )
                continue
            inst_key = str(int(final_gid))
            keypoints_by_instance.setdefault(inst_key, {})[kpt_label] = (x, y)

        registry = InstanceRegistry()
        for gid, _, polygon in polygons:
            inst_key = str(int(gid))
            mask = polygon_masks.get(int(gid))
            inst_state = registry.ensure_instance(inst_key)
            inst_state.group_id = int(gid)
            inst_state.set_mask(bitmap=mask, polygon=polygon)

        for inst_key, points in keypoints_by_instance.items():
            for kpt_label, (x, y) in points.items():
                state = KeypointState(
                    key=f"{inst_key}:{kpt_label}",
                    instance_label=inst_key,
                    label=kpt_label,
                    x=float(x),
                    y=float(y),
                    visible=True,
                    confidence=1.0,
                )
                registry.register_keypoint(state)

        for instance in registry:
            instance.last_updated_frame = int(frame_number)

        return ManualAnnotation(
            frame_number=int(frame_number),
            registry=registry,
            keypoints_by_instance=keypoints_by_instance,
            display_labels=display_labels,
            ambiguous_keypoints=tuple(ambiguous_records),
        )

    def _assign_group_for_point(
        self,
        x: float,
        y: float,
        polygon_masks: Dict[int, np.ndarray],
        polygons: List[Tuple[int, str, List[Tuple[float, float]]]],
    ) -> Optional[int]:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if (
            xi < 0
            or yi < 0
            or xi >= int(self.image_width)
            or yi >= int(self.image_height)
        ):
            return None

        for gid, mask in polygon_masks.items():
            try:
                if bool(mask[yi, xi]):
                    return int(gid)
            except Exception:
                continue

        if len(polygons) == 1:
            return int(polygons[0][0])

        best_gid = None
        best_dist = None
        for gid, _, polygon in polygons:
            if not polygon:
                continue
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            cx = float(sum(xs) / max(1, len(xs)))
            cy = float(sum(ys) / max(1, len(ys)))
            dist = (cx - float(x)) ** 2 + (cy - float(y)) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_gid = int(gid)
        return best_gid
