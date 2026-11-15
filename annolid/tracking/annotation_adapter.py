"""Adapter for converting between LabelMe JSON annotations and domain models."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    set_visibility_flags,
    visibility_from_labelme_shape,
)
from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.tracking.domain import (
    BodyPartState,
    KEYPOINT_DELIMITER,
    MASK_SUFFIX,
    combine_labels,
    InstanceRegistry,
    KeypointState,
)
from annolid.utils.body_part_shapes import (
    build_body_part_flags,
    build_body_part_metadata,
    extract_body_part_label,
    extract_body_part_parent_instance,
    is_body_part_shape_payload,
)
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
from annolid.utils.files import (
    find_manual_seed_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger
from annolid.utils.zone_shapes import is_zone_shape_payload


DEFAULT_KEYPOINT_LABEL = "centroid"


@dataclass(slots=True)
class AnnotationAdapter:
    """Converts between JSON annotations and domain objects."""

    image_height: int
    image_width: int
    description: str = "Cutie+DINO"
    persist_json: bool = True
    mask_description: str = "Cutie"
    ambiguous_point_policy: str = "withhold"
    last_ambiguous_points: List[Dict[str, object]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.ambiguous_point_policy = normalize_ambiguous_point_policy(
            self.ambiguous_point_policy, strict=True
        )

    def load_initial_state(self, annotation_dir: Path) -> Tuple[int, InstanceRegistry]:
        json_files = find_manual_seed_json_files(str(annotation_dir))
        if not json_files:
            raise RuntimeError(
                "No manual PNG+JSON seed pairs found. Provide an initial annotation for the first frame."
            )
        candidates: List[Tuple[float, int, Path]] = []
        for name in json_files:
            path = annotation_dir / name
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            frame_idx = get_frame_number_from_json(name)
            candidates.append((mtime, frame_idx, path))
        if not candidates:
            raise RuntimeError(
                "No labeled JSON files found. Provide an initial annotation for the first frame."
            )
        _, frame_number, latest_json = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        registry = self.read_annotation(latest_json)
        return frame_number, registry

    @staticmethod
    def _extract_prior_keypoints_for_instances(
        source_registry: InstanceRegistry,
        target_registry: InstanceRegistry,
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        prior_by_group: Dict[str, Optional[str]] = {}
        for src_label, src_inst in source_registry.instances.items():
            if src_inst.group_id is None:
                continue
            gkey = str(src_inst.group_id).strip()
            if not gkey:
                continue
            if gkey not in prior_by_group:
                prior_by_group[gkey] = src_label
            elif prior_by_group[gkey] != src_label:
                prior_by_group[gkey] = None

        extracted: Dict[str, List[Tuple[str, float, float]]] = {}
        for dst_label, dst_inst in target_registry.instances.items():
            src_inst = source_registry.instances.get(dst_label)
            if src_inst is None and dst_inst.group_id is not None:
                mapped = prior_by_group.get(str(dst_inst.group_id).strip())
                if mapped:
                    src_inst = source_registry.instances.get(mapped)
            if src_inst is None or not src_inst.keypoints:
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
                extracted[dst_label] = vis_pts
        return extracted

    def read_annotation(
        self,
        json_path: Path,
        *,
        prior_registry: Optional[InstanceRegistry] = None,
        ambiguous_point_policy: Optional[str] = None,
        _allow_neighbor_lookup: bool = True,
    ) -> InstanceRegistry:
        effective_policy = (
            "withhold"
            if not _allow_neighbor_lookup
            else normalize_ambiguous_point_policy(
                ambiguous_point_policy or self.ambiguous_point_policy,
                strict=True,
            )
        )
        payload = load_labelme_json(json_path)
        shapes: Sequence[Dict[str, object]] = payload.get("shapes", [])  # type: ignore[assignment]
        registry = InstanceRegistry()

        # Pre-count polygon base labels so multi-animal scenes sharing a class label
        # (e.g., two "mouse" polygons with group_id=0 and group_id=1) are disambiguated
        # rather than overwriting each other in InstanceRegistry.
        polygon_base_counts: Counter[str] = Counter()
        pending_body_parts: List[Dict[str, object]] = []
        for shape in shapes:
            if is_zone_shape_payload(shape):
                continue
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            if shape_type != "polygon":
                continue
            if is_body_part_shape_payload(shape):
                if shape.get("points"):
                    pending_body_parts.append(shape)
                continue
            raw_label = str(shape.get("label", "")).strip()
            base_instance = self._parse_mask_label(raw_label, shape)
            if base_instance and shape.get("points"):
                polygon_base_counts[base_instance] += 1

        polygon_masks: List[Tuple[str, np.ndarray]] = []
        group_instances: Dict[str, Optional[str]] = {}
        group_candidates: Dict[str, List[str]] = defaultdict(list)
        duplicate_ordinals: Counter[str] = Counter()
        for shape in shapes:
            if is_zone_shape_payload(shape) or is_body_part_shape_payload(shape):
                continue
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            if shape_type != "polygon":
                continue
            base_instance = self._parse_mask_label(label, shape)
            if not base_instance:
                continue
            polygon_points = [
                (float(pt[0]), float(pt[1]))
                for pt in shape.get("points", [])  # type: ignore[index,union-attr]
            ]
            if not polygon_points:
                continue

            raw_group_id = self._normalize_group_id_value(shape.get("group_id"))
            group_key = self._group_id_key(shape)

            instance_label = base_instance
            mask_label_override: Optional[str] = None
            if polygon_base_counts[base_instance] > 1 and not self._flag_instance_label(
                shape
            ):
                suffix = group_key
                if suffix is None or f"{base_instance}_{suffix}" in registry.instances:
                    ordinal = duplicate_ordinals[base_instance]
                    while f"{base_instance}_{ordinal}" in registry.instances:
                        ordinal += 1
                    duplicate_ordinals[base_instance] = ordinal + 1
                    suffix = str(ordinal)
                instance_label = f"{base_instance}_{suffix}"
                mask_label_override = label or base_instance

            polygon = self._sanitize_polygon(polygon_points)
            instance_state = registry.ensure_instance(instance_label)
            instance_state.set_mask(
                bitmap=None,
                polygon=polygon,
            )
            if raw_group_id is not None and instance_state.group_id is None:
                instance_state.group_id = raw_group_id
            if not mask_label_override and label and label != instance_label:
                mask_label_override = label
            if mask_label_override:
                instance_state.mask_label_override = mask_label_override

            polygon_masks.append(
                (instance_label, self.mask_bitmap_from_polygon(polygon))
            )
            if group_key is not None:
                if instance_label not in group_candidates[group_key]:
                    group_candidates[group_key].append(instance_label)
                if group_key not in group_instances:
                    group_instances[group_key] = instance_label
                elif group_instances[group_key] != instance_label:
                    group_instances[group_key] = None

        # Attach intra-instance body-part polygons to their parent InstanceState.
        for bp_shape in pending_body_parts:
            bp_points = [
                (float(pt[0]), float(pt[1]))
                for pt in bp_shape.get("points", [])  # type: ignore[index,union-attr]
            ]
            bp_poly = self._sanitize_polygon(bp_points)
            if not bp_poly:
                continue
            parent_hint = extract_body_part_parent_instance(bp_shape)
            part_label = extract_body_part_label(bp_shape) or "part"
            bp_group_id = self._normalize_group_id_value(bp_shape.get("group_id"))
            bp_group_key = self._group_id_key(bp_shape)

            resolved_parent: Optional[str] = None
            if parent_hint and parent_hint in registry.instances:
                resolved_parent = parent_hint
            elif bp_group_key is not None and group_instances.get(bp_group_key):
                resolved_parent = group_instances[bp_group_key]
            elif polygon_masks:
                bp_bitmap = self.mask_bitmap_from_polygon(bp_poly)
                best_overlap = 0
                for cand_inst, cand_mask in polygon_masks:
                    overlap = int(np.count_nonzero(bp_bitmap & cand_mask))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        resolved_parent = cand_inst
            if not resolved_parent:
                resolved_parent = parent_hint or (
                    next(iter(registry.instances))
                    if len(registry.instances) == 1
                    else "instance"
                )

            bp_conf, _bp_qual = self._extract_confidence_and_quality(bp_shape)
            raw_area_ratio = bp_shape.get("area_ratio")
            if raw_area_ratio is None and isinstance(bp_shape.get("flags"), Mapping):
                raw_area_ratio = bp_shape["flags"].get("area_ratio")  # type: ignore[index]
            area_ratio_val: Optional[float] = None
            if raw_area_ratio is not None and not isinstance(raw_area_ratio, bool):
                try:
                    ar = float(raw_area_ratio)  # type: ignore[arg-type]
                    if math.isfinite(ar):
                        area_ratio_val = ar
                except (TypeError, ValueError):
                    area_ratio_val = None

            raw_bp_label = str(bp_shape.get("label", "")).strip()
            parent_state = registry.ensure_instance(resolved_parent)
            parent_state.upsert_body_part(
                BodyPartState(
                    part_label=part_label,
                    instance_label=resolved_parent,
                    polygon=bp_poly,
                    mask_bitmap=None,
                    confidence=bp_conf,
                    area_ratio=area_ratio_val,
                    group_id=(
                        bp_group_id
                        if bp_group_id is not None
                        else parent_state.group_id
                    ),
                    storage_label_override=(
                        raw_bp_label
                        if raw_bp_label and raw_bp_label != part_label
                        else None
                    ),
                )
            )

        known_polygon_instances = list(registry.instances.keys())
        assigned_for_overlap: Dict[str, List[Tuple[str, float, float]]] = defaultdict(
            list
        )
        overlap_candidates: List[OverlappingPointCandidate] = []
        parsed_points: List[
            Tuple[
                int,
                str,
                Dict[str, object],
                float,
                float,
                str,
                str,
                Optional[int | str],
                Optional[str],
                List[str],
            ]
        ] = []

        for shape in shapes:
            if is_zone_shape_payload(shape):
                continue
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            if shape_type != "point":
                continue
            point = (shape.get("points") or [[None, None]])[0]  # type: ignore[index]
            if point[0] is None or point[1] is None:
                continue
            x, y = float(point[0]), float(point[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            instance_label, keypoint_label = self._split_keypoint_label(label, shape)
            raw_group_id = self._normalize_group_id_value(shape.get("group_id"))
            resolved_instance: Optional[str] = None
            containing: List[str] = []

            if self._has_explicit_keypoint_instance(label, shape):
                resolved_instance = instance_label
            elif known_polygon_instances:
                prefixed = match_instance_prefix(label, known_polygon_instances)
                if prefixed is not None:
                    resolved_instance, remainder = prefixed
                    keypoint_label = self._flag_display_label(shape) or remainder

            if resolved_instance is None:
                group_key = self._group_id_key(shape)
                if group_key is not None and group_instances.get(group_key):
                    resolved_instance = group_instances[group_key]
                else:
                    containing = containing_instances_for_point(x, y, polygon_masks)
                    if group_key is not None and group_candidates.get(group_key):
                        narrowed = [
                            inst
                            for inst in containing
                            if inst in group_candidates[group_key]
                        ]
                        if narrowed:
                            containing = narrowed
                    if len(containing) == 1:
                        resolved_instance = containing[0]

            rec_idx = len(parsed_points)
            if resolved_instance is not None:
                assigned_for_overlap[resolved_instance].append((keypoint_label, x, y))
            elif len(containing) >= 2:
                overlap_candidates.append(
                    OverlappingPointCandidate(
                        index=rec_idx,
                        keypoint_label=keypoint_label,
                        x=x,
                        y=y,
                        containing_instances=tuple(containing),
                    )
                )

            parsed_points.append(
                (
                    rec_idx,
                    label,
                    shape,
                    x,
                    y,
                    instance_label,
                    keypoint_label,
                    raw_group_id,
                    resolved_instance,
                    containing,
                )
            )

        prior_kps: Optional[Dict[str, List[Tuple[str, float, float]]]] = None
        if prior_registry is not None and overlap_candidates:
            prior_kps = self._extract_prior_keypoints_for_instances(
                prior_registry, registry
            )

        resolved_overlaps = resolve_overlapping_point_instances(
            polygon_masks=polygon_masks,
            assigned_keypoints=assigned_for_overlap,
            candidates=overlap_candidates,
            prior_keypoints=prior_kps,
        )

        if (
            len(resolved_overlaps) < len(overlap_candidates)
            and prior_registry is None
            and _allow_neighbor_lookup
        ):
            for _, neighbor_path in find_neighbor_frame_paths(
                json_path, max_neighbors=2
            ):
                try:
                    neighbor_reg = self.read_annotation(
                        neighbor_path, _allow_neighbor_lookup=False
                    )
                except Exception:
                    continue
                neighbor_prior = self._extract_prior_keypoints_for_instances(
                    neighbor_reg, registry
                )
                if not neighbor_prior:
                    continue
                candidate_resolution = resolve_overlapping_point_instances(
                    polygon_masks=polygon_masks,
                    assigned_keypoints=assigned_for_overlap,
                    candidates=overlap_candidates,
                    prior_keypoints=neighbor_prior,
                )
                if len(candidate_resolution) > len(resolved_overlaps):
                    resolved_overlaps = candidate_resolution
                    if len(resolved_overlaps) == len(overlap_candidates):
                        break

        if overlap_candidates and effective_policy != "withhold":
            try:
                resolved_overlaps = apply_ambiguous_point_policy(
                    policy=effective_policy,
                    candidates=overlap_candidates,
                    resolved=resolved_overlaps,
                    assigned_keypoints=assigned_for_overlap,
                    source_path=json_path,
                )
            except Exception as exc:
                if _allow_neighbor_lookup and hasattr(exc, "ambiguous_points"):
                    self.last_ambiguous_points = [
                        dict(item) for item in getattr(exc, "ambiguous_points", ())
                    ]
                raise

        ambiguous_records: List[Dict[str, object]] = []

        for (
            rec_idx,
            label,
            shape,
            x,
            y,
            fallback_instance,
            keypoint_label,
            raw_group_id,
            resolved_instance,
            containing,
        ) in parsed_points:
            final_instance = resolved_instance or resolved_overlaps.get(rec_idx)
            if final_instance:
                instance_label = final_instance
            else:
                instance_label = fallback_instance
                if polygon_masks:
                    if len(containing) >= 2:
                        ambiguous_records.append(
                            {
                                "label": label or keypoint_label,
                                "keypoint_label": keypoint_label,
                                "x": float(x),
                                "y": float(y),
                                "containing_instances": list(containing),
                                "json_path": str(json_path),
                            }
                        )
                        logger.warning(
                            "Keypoint '%s' at (%.1f, %.1f) in %s lies inside "
                            "ambiguous overlapping polygons %s; tracking it "
                            "without mask constraints. Add instance_label "
                            "metadata or a matching group_id.",
                            label or keypoint_label,
                            x,
                            y,
                            json_path,
                            containing,
                        )
                    else:
                        logger.warning(
                            "Keypoint '%s' in %s is not associated with a unique "
                            "polygon; tracking it without mask constraints. Add "
                            "instance_label metadata or a matching group_id.",
                            label or keypoint_label,
                            json_path,
                        )

            if raw_group_id is None and instance_label in registry.instances:
                raw_group_id = registry.instances[instance_label].group_id

            kp_vis = visibility_from_labelme_shape(shape)
            is_visible = bool(shape.get("visible", True)) and (
                kp_vis is None or int(kp_vis) == int(KeypointVisibility.VISIBLE)
            )
            confidence, quality = self._extract_confidence_and_quality(shape)

            key = self._build_unique_key(instance_label, keypoint_label, registry)
            registry.register_keypoint(
                KeypointState(
                    key=key,
                    instance_label=instance_label,
                    label=keypoint_label,
                    x=x,
                    y=y,
                    visible=is_visible,
                    confidence=confidence,
                    quality=quality,
                    storage_label_override=label or None,
                    group_id=raw_group_id,
                )
            )
        if _allow_neighbor_lookup:
            self.last_ambiguous_points = ambiguous_records
        return registry

    def write_annotation(
        self, *, frame_number: int, registry: InstanceRegistry, output_dir: Path
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{output_dir.name}_{frame_number:09d}.json"
        shapes: List[Shape] = []
        for instance in registry:
            for keypoint in instance.keypoints.values():
                kp_flags: Dict[str, object] = {
                    "instance_label": instance.label,
                    "display_label": keypoint.label,
                    "quality": round(float(keypoint.quality), 4),
                    "velocity": [
                        round(float(keypoint.velocity_x), 4),
                        round(float(keypoint.velocity_y), 4),
                    ],
                    "misses": int(keypoint.misses),
                }
                if not keypoint.visible:
                    set_visibility_flags(kp_flags, int(KeypointVisibility.OCCLUDED))
                resolved_gid = (
                    keypoint.group_id
                    if keypoint.group_id is not None
                    else instance.group_id
                )
                shape = Shape(
                    label=keypoint.storage_label,
                    shape_type="point",
                    group_id=resolved_gid,
                    flags=kp_flags,
                    description=self.description,
                    visible=keypoint.visible,
                )
                shape.points = [[float(keypoint.x), float(keypoint.y)]]
                shapes.append(shape)

            if instance.polygon:
                mask_label = instance.mask_label_override or self._mask_label(
                    instance.label
                )
                mask_shape = Shape(
                    label=mask_label,
                    shape_type="polygon",
                    group_id=instance.group_id,
                    flags={"instance_label": instance.label},
                    description=self.mask_description,
                )
                mask_shape.points = [
                    [float(x), float(y)]
                    for x, y in self._sanitize_polygon(instance.polygon)
                ]
                shapes.append(mask_shape)

            for body_part in instance.body_parts.values():
                if not body_part.polygon:
                    continue
                part_shape_label = (
                    body_part.storage_label_override or body_part.part_label
                )
                part_gid = (
                    body_part.group_id
                    if body_part.group_id is not None
                    else instance.group_id
                )
                part_shape = Shape(
                    label=part_shape_label,
                    shape_type="polygon",
                    group_id=part_gid,
                    flags=build_body_part_flags(),
                    description="InstanceBodyPartSeg",
                )
                part_shape.other_data = build_body_part_metadata(
                    parent_instance=instance.label,
                    part_label=body_part.part_label,
                    confidence=body_part.confidence,
                    area_ratio=body_part.area_ratio,
                )
                part_shape.points = [
                    [float(x), float(y)]
                    for x, y in self._sanitize_polygon(body_part.polygon)
                ]
                shapes.append(part_shape)

        save_labels(
            filename=json_path,
            imagePath="",
            label_list=shapes,
            height=self.image_height,
            width=self.image_width,
            save_image_to_json=False,
            persist_json=self.persist_json,
        )
        return json_path

    def _split_keypoint_label(
        self, label: str, shape: Dict[str, object]
    ) -> Tuple[str, str]:
        if MASK_SUFFIX and label.endswith(MASK_SUFFIX):
            label = label[: -len(MASK_SUFFIX)]
        display_label = self._flag_display_label(shape)
        if KEYPOINT_DELIMITER and KEYPOINT_DELIMITER in label:
            instance_label, keypoint_label = label.split(KEYPOINT_DELIMITER, 1)
            instance_label = instance_label or self._flag_instance_label(shape)
            keypoint_label = keypoint_label or display_label or DEFAULT_KEYPOINT_LABEL
            return instance_label, keypoint_label
        flag_label = self._flag_instance_label(shape)
        if flag_label:
            keypoint_label = display_label or label or DEFAULT_KEYPOINT_LABEL
            return flag_label, keypoint_label
        keypoint_label = display_label or label or DEFAULT_KEYPOINT_LABEL
        instance_label = label or keypoint_label
        return instance_label, keypoint_label

    def _has_explicit_keypoint_instance(
        self,
        label: str,
        shape: Dict[str, object],
    ) -> bool:
        if KEYPOINT_DELIMITER and KEYPOINT_DELIMITER in label:
            return True
        return bool(self._flag_instance_label(shape))

    @staticmethod
    def _normalize_group_id_value(value: object) -> Optional[int | str]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return int(value) if value.is_integer() else float(value)
        text = str(value).strip()
        if not text:
            return None
        sign = -1 if text.startswith("-") else 1
        digits = text[1:] if text[:1] in ("+", "-") else text
        if digits.isdigit():
            return int(sign * int(digits))
        return text

    @classmethod
    def _group_id_key(cls, shape: Dict[str, object]) -> Optional[str]:
        normalized = cls._normalize_group_id_value(shape.get("group_id"))
        if normalized is None:
            return None
        text = str(normalized).strip()
        return text or None

    @staticmethod
    def _extract_confidence_and_quality(
        shape: Mapping[str, object],
    ) -> Tuple[float, float]:
        def _find_numeric(keys: Sequence[str]) -> Optional[float]:
            for key in keys:
                val = shape.get(key)
                if val is not None and not isinstance(val, bool):
                    try:
                        num = float(val)  # type: ignore[arg-type]
                        if math.isfinite(num):
                            return float(np.clip(num, 0.0, 1.0))
                    except (TypeError, ValueError):
                        pass
            for container_key in ("flags", "other_data", "otherData"):
                container = shape.get(container_key)
                if isinstance(container, Mapping):
                    for key in keys:
                        val = container.get(key)
                        if val is not None and not isinstance(val, bool):
                            try:
                                num = float(val)
                                if math.isfinite(num):
                                    return float(np.clip(num, 0.0, 1.0))
                            except (TypeError, ValueError):
                                pass
            return None

        quality = _find_numeric(("quality",))
        confidence = _find_numeric(("score", "confidence", "kp_score", "quality"))
        resolved_conf = 1.0 if confidence is None else confidence
        resolved_qual = resolved_conf if quality is None else quality
        return resolved_conf, resolved_qual

    def _instance_containing_point(
        self,
        x: float,
        y: float,
        polygon_masks: Sequence[Tuple[str, np.ndarray]],
    ) -> Optional[str]:
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
        unique = list(dict.fromkeys(containing))
        return unique[0] if len(unique) == 1 else None

    def _parse_mask_label(self, label: str, shape: Dict[str, object]) -> Optional[str]:
        if MASK_SUFFIX and label.endswith(MASK_SUFFIX):
            return label[: -len(MASK_SUFFIX)]
        flag_label = self._flag_instance_label(shape)
        if flag_label:
            return flag_label
        return label or None

    def _mask_label(self, instance_label: str) -> str:
        if MASK_SUFFIX:
            return f"{instance_label}{MASK_SUFFIX}"
        return instance_label

    def _build_key(self, instance_label: str, keypoint_label: str) -> str:
        clean_instance = instance_label or "instance"
        clean_keypoint = keypoint_label or DEFAULT_KEYPOINT_LABEL
        return combine_labels(clean_instance, clean_keypoint)

    def _build_unique_key(
        self,
        instance_label: str,
        keypoint_label: str,
        registry: InstanceRegistry,
    ) -> str:
        primary = self._build_key(instance_label, keypoint_label)
        existing = registry.get_keypoint(primary)
        if existing is None or (
            existing.instance_label == instance_label
            and existing.label == keypoint_label
        ):
            return primary
        clean_instance = instance_label or "instance"
        clean_keypoint = keypoint_label or DEFAULT_KEYPOINT_LABEL
        disambiguated = f"{clean_instance}::{clean_keypoint}"
        suffix = 2
        while True:
            holder = registry.get_keypoint(disambiguated)
            if holder is None or (
                holder.instance_label == instance_label
                and holder.label == keypoint_label
            ):
                return disambiguated
            disambiguated = f"{clean_instance}::{clean_keypoint}#{suffix}"
            suffix += 1

    def _flag_instance_label(self, shape: Dict[str, object]) -> str:
        value = shape.get("instance_label")
        if value:
            return str(value).strip()
        flags = shape.get("flags")
        if isinstance(flags, dict):
            value = flags.get("instance_label")
            return str(value).strip() if value else ""
        return ""

    def _flag_display_label(self, shape: Dict[str, object]) -> str:
        value = shape.get("display_label")
        if value:
            return str(value).strip()
        flags = shape.get("flags")
        if not isinstance(flags, dict):
            return ""
        value = flags.get("display_label")
        return str(value).strip() if value else ""

    def _sanitize_polygon(
        self, polygon: Iterable[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        points = [(float(x), float(y)) for x, y in polygon]
        if not points:
            return []
        if points[0] != points[-1]:
            points.append(points[0])
        return points

    def mask_bitmap_from_polygon(
        self, polygon: Sequence[Tuple[float, float]]
    ) -> np.ndarray:
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        if not polygon:
            return mask
        polygon_array = np.rint(np.array(polygon)).astype(np.int32)
        # Fill the polygon on the mask to reconstruct a bitmap representation.
        cv2 = self._lazy_cv2()
        cv2.fillPoly(mask, [polygon_array], 1)
        return mask.astype(bool)

    def _lazy_cv2(self):  # pragma: no cover - isolated import
        import cv2  # Local import to avoid optional dependency at module import

        return cv2
