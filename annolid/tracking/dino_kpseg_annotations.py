"""Helpers for parsing instance polygons + keypoints for DinoKPSEG tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.utils.annotation_store import load_labelme_json


@dataclass(frozen=True)
class ManualAnnotation:
    frame_number: int
    registry: InstanceRegistry
    keypoints_by_instance: Dict[str, Dict[str, Tuple[float, float]]]
    display_labels: Dict[str, str]


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
    ) -> None:
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.adapter = adapter
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

    def read_manual_annotation(
        self,
        frame_number: int,
        json_path: Path,
    ) -> ManualAnnotation:
        payload = load_labelme_json(json_path)
        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list):
            shapes = []

        polygons: List[Tuple[int, str, List[Tuple[float, float]]]] = []
        used_gids: set[int] = set()
        for shape in shapes:
            if not isinstance(shape, dict):
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
        for gid, raw_label, polygon in polygons:
            display = raw_label or f"instance_{gid}"
            display_labels[str(gid)] = display
            if raw_label:
                self._polygon_label_to_gid[raw_label] = int(gid)
            if polygon:
                polygon_masks[gid] = self.adapter.mask_bitmap_from_polygon(polygon)

        keypoints_by_instance: Dict[str, Dict[str, Tuple[float, float]]] = {}
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

            if gid is None:
                gid = self._assign_group_for_point(x, y, polygon_masks, polygons)
            if gid is None:
                continue
            inst_key = str(int(gid))
            keypoints_by_instance.setdefault(inst_key, {})[label] = (x, y)

        registry = InstanceRegistry()
        for gid, _, polygon in polygons:
            inst_key = str(int(gid))
            mask = polygon_masks.get(int(gid))
            registry.ensure_instance(inst_key).set_mask(bitmap=mask, polygon=polygon)

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
