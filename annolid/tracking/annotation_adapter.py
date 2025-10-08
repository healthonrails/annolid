"""Adapter for converting between LabelMe JSON annotations and domain models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.tracking.domain import (
    KEYPOINT_DELIMITER,
    MASK_SUFFIX,
    combine_labels,
    InstanceRegistry,
    KeypointState,
)
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.annotation_store import load_labelme_json


DEFAULT_KEYPOINT_LABEL = "centroid"


@dataclass(slots=True)
class AnnotationAdapter:
    """Converts between JSON annotations and domain objects."""

    image_height: int
    image_width: int
    description: str = "Cutie+DINO"

    def load_initial_state(self, annotation_dir: Path) -> Tuple[int, InstanceRegistry]:
        json_files = find_manual_labeled_json_files(str(annotation_dir))
        if not json_files:
            raise RuntimeError(
                "No labeled JSON files found. Provide an initial annotation for the first frame.")
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
                "No labeled JSON files found. Provide an initial annotation for the first frame.")
        _, frame_number, latest_json = max(
            candidates, key=lambda item: (item[0], item[1]))
        registry = self.read_annotation(latest_json)
        return frame_number, registry

    def read_annotation(self, json_path: Path) -> InstanceRegistry:
        payload = load_labelme_json(json_path)
        shapes: Sequence[Dict[str, object]] = payload.get(
            "shapes", [])  # type: ignore[assignment]
        registry = InstanceRegistry()
        for shape in shapes:
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).strip() or "point"
            flags = shape.get("flags") or {}

            if shape_type == "point":
                instance_label, keypoint_label = self._split_keypoint_label(
                    label, flags)
                point = (shape.get("points") or [[None, None]])[0]
                if point[0] is None or point[1] is None:
                    continue

                key = self._build_key(instance_label, keypoint_label)
                keypoint_state = KeypointState(
                    key=key,
                    instance_label=instance_label,
                    label=keypoint_label,
                    x=float(point[0]),
                    y=float(point[1]),
                    visible=bool(shape.get("visible", True)),
                )

                registry.register_keypoint(keypoint_state)

            elif shape_type == "polygon":
                instance_label = self._parse_mask_label(label, flags)
                if not instance_label:
                    continue
                polygon_points = [
                    (float(pt[0]), float(pt[1])) for pt in shape.get("points", [])
                ]
                if not polygon_points:
                    continue
                registry.ensure_instance(instance_label).set_mask(
                    bitmap=None,
                    polygon=self._sanitize_polygon(polygon_points),
                )
        return registry

    def write_annotation(self, *, frame_number: int, registry: InstanceRegistry,
                         output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{output_dir.name}_{frame_number:09d}.json"
        shapes: List[Shape] = []
        for instance in registry:
            for keypoint in instance.keypoints.values():
                shape = Shape(
                    label=keypoint.storage_label,
                    shape_type="point",
                    flags={
                        "instance_label": instance.label,
                        "display_label": keypoint.label,
                        "quality": round(float(keypoint.quality), 4),
                        "velocity": [
                            round(float(keypoint.velocity_x), 4),
                            round(float(keypoint.velocity_y), 4),
                        ],
                        "misses": int(keypoint.misses),
                    },
                    description=self.description,
                    visible=keypoint.visible,
                )
                shape.points = [[float(keypoint.x), float(keypoint.y)]]
                shapes.append(shape)

            if instance.polygon:
                mask_shape = Shape(
                    label=self._mask_label(instance.label),
                    shape_type="polygon",
                    flags={"instance_label": instance.label},
                    description="Cutie",
                )
                mask_shape.points = [
                    [float(x), float(y)] for x, y in self._sanitize_polygon(instance.polygon)
                ]
                shapes.append(mask_shape)

        save_labels(
            filename=json_path,
            imagePath="",
            label_list=shapes,
            height=self.image_height,
            width=self.image_width,
            save_image_to_json=False,
        )
        return json_path

    def _split_keypoint_label(self, label: str, flags: Dict[str, object]) -> Tuple[str, str]:
        if MASK_SUFFIX and label.endswith(MASK_SUFFIX):
            label = label[: -len(MASK_SUFFIX)]
        display_label = self._flag_display_label(flags)
        if KEYPOINT_DELIMITER and KEYPOINT_DELIMITER in label:
            instance_label, keypoint_label = label.split(KEYPOINT_DELIMITER, 1)
            instance_label = instance_label or self._flag_instance_label(flags)
            keypoint_label = keypoint_label or display_label or DEFAULT_KEYPOINT_LABEL
            return instance_label, keypoint_label
        flag_label = self._flag_instance_label(flags)
        if flag_label:
            keypoint_label = display_label or label or DEFAULT_KEYPOINT_LABEL
            return flag_label, keypoint_label
        keypoint_label = display_label or label or DEFAULT_KEYPOINT_LABEL
        instance_label = label or keypoint_label
        return instance_label, keypoint_label

    def _parse_mask_label(self, label: str, flags: Dict[str, object]) -> Optional[str]:
        if MASK_SUFFIX and label.endswith(MASK_SUFFIX):
            return label[: -len(MASK_SUFFIX)]
        flag_label = self._flag_instance_label(flags)
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

    def _flag_instance_label(self, flags: Dict[str, object]) -> str:
        value = flags.get("instance_label") if isinstance(
            flags, dict) else None
        return str(value).strip() if value else ""

    def _flag_display_label(self, flags: Dict[str, object]) -> str:
        if not isinstance(flags, dict):
            return ""
        value = flags.get("display_label")
        return str(value).strip() if value else ""

    def _sanitize_polygon(self, polygon: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        points = [(float(x), float(y)) for x, y in polygon]
        if not points:
            return []
        if points[0] != points[-1]:
            points.append(points[0])
        return points

    def mask_bitmap_from_polygon(self, polygon: Sequence[Tuple[float, float]]) -> np.ndarray:
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
