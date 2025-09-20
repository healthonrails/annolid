"""Manager for orchestrating Cutie VOS mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
)
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.predict import CutieVideoProcessor
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.domain import InstanceRegistry, InstanceState
from annolid.utils.logger import logger


@dataclass(slots=True)
class MaskResult:
    instance_label: str
    mask_bitmap: np.ndarray
    polygon: List[Tuple[float, float]]


class CutieMaskManager:
    """Encapsulates Cutie-specific logic and state management."""

    def __init__(
        self,
        video_path: Path,
        adapter: AnnotationAdapter,
        config: CutieDinoTrackerConfig,
    ) -> None:
        self.video_path = Path(video_path)
        self.adapter = adapter
        self.config = config
        self.enabled = bool(config.use_cutie_tracking)
        self._processor: Optional[CutieVideoProcessor] = None
        self._core: Optional[InferenceCore] = None
        self._device: Optional[str] = None
        self._label_to_value: Dict[str, int] = {}
        self._value_to_label: Dict[int, str] = {}
        self._initialized = False

    def ready(self) -> bool:
        return self.enabled and self._initialized

    def prime(self, frame_number: int, frame: np.ndarray, registry: InstanceRegistry) -> None:
        if not self.enabled:
            logger.debug(
                "CutieMaskManager prime skipped because Cutie is disabled.")
            return
        try:
            self._ensure_core()
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_failure(exc, "Failed to initialize Cutie core")
            return

        mask_values = self._build_initial_mask(registry)
        if not mask_values:
            logger.warning("Cutie prime skipped: no initial mask available.")
            return

        composed_mask = mask_values["mask"]
        mapping = mask_values["mapping"]

        frame_tensor = image_to_torch(frame, device=self._device)
        mask_tensor = index_numpy_to_one_hot_torch(
            composed_mask, len(mapping) + 1).to(self._device)
        try:
            self._core.step(
                frame_tensor,
                mask_tensor[1:],
                idx_mask=False,
                force_permanent=True,
            )
        except Exception as exc:  # pragma: no cover - device level failure
            self._handle_failure(exc, "Cutie prime inference failed")
            return

        self._label_to_value = mapping
        self._value_to_label = {
            value: label for label, value in mapping.items()}
        self._initialized = True
        logger.debug("CutieMaskManager primed with instances: %s",
                     list(mapping.keys()))

    def update_masks(self, frame_number: int, frame: np.ndarray,
                     registry: InstanceRegistry) -> Dict[str, MaskResult]:
        if not self.ready():
            return {}
        frame_tensor = image_to_torch(frame, device=self._device)
        try:
            prediction = self._core.step(frame_tensor)
        except Exception as exc:  # pragma: no cover - device level failure
            self._handle_failure(exc, "Cutie update failed")
            return {}

        mask = torch_prob_to_numpy_mask(prediction)
        results: Dict[str, MaskResult] = {}
        for value, label in self._value_to_label.items():
            binary_mask = mask == value
            if not binary_mask.any():
                continue
            polygon = self._mask_to_polygon(binary_mask)
            results[label] = MaskResult(
                instance_label=label,
                mask_bitmap=binary_mask,
                polygon=polygon,
            )
        return results

    def _ensure_core(self) -> None:
        if self._core is not None:
            return
        use_cpu_only = self.config.cutie_device == "cpu"
        processor = CutieVideoProcessor(
            str(self.video_path),
            mem_every=self.config.cutie_mem_every,
            t_max_value=self.config.cutie_max_mem_frames,
            use_cpu_only=use_cpu_only,
            debug=False,
        )
        self._processor = processor
        self._device = processor.device
        self._core = InferenceCore(processor.cutie, cfg=processor.cfg)

    def _build_initial_mask(self, registry: InstanceRegistry) -> Dict[str, object]:
        composed_mask = np.zeros(
            (self.adapter.image_height, self.adapter.image_width), dtype=np.uint8)
        mapping: Dict[str, int] = {}
        value = 1
        for instance in registry:
            mask_bitmap = self._resolve_instance_mask(instance)
            if mask_bitmap is None or not mask_bitmap.any():
                continue
            composed_mask[mask_bitmap] = value
            mapping[instance.label] = value
            value += 1
        if not mapping:
            return {}
        return {"mask": composed_mask, "mapping": mapping}

    def _resolve_instance_mask(self, instance: InstanceState) -> Optional[np.ndarray]:
        if instance.mask_bitmap is not None:
            mask = instance.mask_bitmap.astype(bool)
        elif instance.polygon is not None:
            mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
        else:
            mask = None
        if mask is None:
            return None
        if self.config.restrict_to_initial_mask and instance.mask_bitmap is not None:
            return mask & instance.mask_bitmap.astype(bool)
        return mask

    def _mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        cv2 = self._lazy_cv2()
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.008 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        if points and points[0] != points[-1]:
            points.append(points[0])
        return points

    def _handle_failure(self, exc: Exception, message: str) -> None:
        logger.error("%s: %s", message, exc)
        self.enabled = False
        self._initialized = False
        if self.config.error_hook:
            self.config.error_hook(exc)

    def _lazy_cv2(self):  # pragma: no cover - isolated import
        import cv2

        return cv2
