"""Domain models for combined keypoint and mask tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np


KEYPOINT_DELIMITER = ""
MASK_SUFFIX = ""

KeypointPayload = Dict[str, object]
MaskPolygon = List[Tuple[float, float]]


def combine_labels(instance_label: str, keypoint_label: str) -> str:
    """Join instance/keypoint names while avoiding redundant duplicates."""

    instance_label = (instance_label or "").strip()
    keypoint_label = (keypoint_label or "").strip()

    if KEYPOINT_DELIMITER:
        if instance_label and keypoint_label:
            return f"{instance_label}{KEYPOINT_DELIMITER}{keypoint_label}"
        return instance_label or keypoint_label

    if not instance_label:
        return keypoint_label
    if not keypoint_label:
        return instance_label
    if instance_label == keypoint_label:
        return keypoint_label
    return f"{instance_label}{keypoint_label}"


@dataclass(slots=True)
class KeypointState:
    """Represents the most recent state of a tracked keypoint."""

    key: str
    instance_label: str
    label: str
    x: float
    y: float
    visible: bool = True
    confidence: float = 1.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    misses: int = 0
    quality: float = 1.0

    def to_tracker_payload(self) -> KeypointPayload:
        """Convert to the payload expected by the Dino tracker."""
        return {
            "id": self.key,
            "label": self.storage_label,
            "x": float(self.x),
            "y": float(self.y),
            "visible": bool(self.visible),
            "quality": float(self.quality),
        }

    @property
    def storage_label(self) -> str:
        """Stable label used for JSON serialization to avoid collisions."""
        return combine_labels(self.instance_label, self.label)

    def update(
        self,
        *,
        x: float,
        y: float,
        visible: bool = True,
        confidence: Optional[float] = None,
        velocity: Optional[Tuple[float, float]] = None,
        misses: Optional[int] = None,
        quality: Optional[float] = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.visible = bool(visible)
        if confidence is not None:
            self.confidence = float(confidence)
        if velocity is not None:
            self.velocity_x = float(velocity[0])
            self.velocity_y = float(velocity[1])
        if misses is not None:
            self.misses = int(misses)
        if quality is not None:
            self.quality = float(quality)


@dataclass
class InstanceState:
    """Encapsulates keypoints and mask information for a labeled instance."""

    label: str
    keypoints: Dict[str, KeypointState] = field(default_factory=dict)
    mask_bitmap: Optional[np.ndarray] = None
    polygon: Optional[MaskPolygon] = None
    last_updated_frame: Optional[int] = None

    def upsert_keypoint(self, keypoint: KeypointState) -> None:
        self.keypoints[keypoint.key] = keypoint

    def upsert_keypoints(self, keypoints: Sequence[KeypointState]) -> None:
        for kp in keypoints:
            self.upsert_keypoint(kp)

    def set_mask(
        self, *, bitmap: Optional[np.ndarray], polygon: Optional[MaskPolygon]
    ) -> None:
        self.mask_bitmap = bitmap if bitmap is None else np.asarray(bitmap)
        self.polygon = list(polygon) if polygon is not None else None

    def to_tracker_payload(self) -> List[KeypointPayload]:
        return [kp.to_tracker_payload() for kp in self.keypoints.values()]


@dataclass
class InstanceRegistry:
    """State container for all instances in the current tracking session."""

    instances: MutableMapping[str, InstanceState] = field(default_factory=dict)
    _keypoint_index: Dict[str, Tuple[str, str]] = field(
        default_factory=dict, init=False
    )

    def ensure_instance(self, label: str) -> InstanceState:
        if label not in self.instances:
            self.instances[label] = InstanceState(label=label)
        return self.instances[label]

    def register_keypoint(self, keypoint: KeypointState) -> None:
        instance = self.ensure_instance(keypoint.instance_label)
        instance.upsert_keypoint(keypoint)
        self._keypoint_index[keypoint.key] = (instance.label, keypoint.key)

    def register_keypoints(self, keypoints: Sequence[KeypointState]) -> None:
        for keypoint in keypoints:
            self.register_keypoint(keypoint)

    def get_keypoint(self, key: str) -> Optional[KeypointState]:
        location = self._keypoint_index.get(key)
        if not location:
            return None
        instance_label, keypoint_key = location
        return self.instances[instance_label].keypoints.get(keypoint_key)

    def keypoint_payload(self) -> List[KeypointPayload]:
        payload: List[KeypointPayload] = []
        for instance in self.instances.values():
            payload.extend(instance.to_tracker_payload())
        return payload

    def apply_tracker_results(
        self, results: Iterable[KeypointPayload], frame_number: Optional[int] = None
    ) -> None:
        for result in results:
            key = str(result["id"])
            keypoint = self.get_keypoint(key)
            if keypoint is None:
                continue
            velocity = None
            if "velocity" in result and result["velocity"] is not None:
                vx, vy = result["velocity"]
                velocity = (float(vx), float(vy))
            keypoint.update(
                x=result.get("x", keypoint.x),
                y=result.get("y", keypoint.y),
                visible=result.get("visible", keypoint.visible),
                confidence=result.get("confidence"),
                velocity=velocity,
                misses=result.get("misses"),
                quality=result.get("quality"),
            )
            instance = self.instances[keypoint.instance_label]
            instance.last_updated_frame = frame_number

    def instances_with_masks(self) -> List[InstanceState]:
        return [
            instance
            for instance in self.instances.values()
            if instance.polygon is not None
        ]

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.instances.values())
