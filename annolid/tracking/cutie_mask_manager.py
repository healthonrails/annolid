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
from annolid.segmentation.cutie_vos.predict import CutieCoreVideoProcessor
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
        self._processor: Optional[CutieCoreVideoProcessor] = None
        self._core: Optional[InferenceCore] = None
        self._device: Optional[str] = None
        self._label_to_value: Dict[str, int] = {}
        self._value_to_label: Dict[int, str] = {}
        self._initialized = False
        self._last_results: Dict[str, MaskResult] = {}
        self._mask_miss_counts: Dict[str, int] = {}

    def ready(self) -> bool:
        return self.enabled and self._initialized

    def reset_state(self) -> None:
        """Reset cached Cutie state so a new run starts clean."""
        self._processor = None
        self._core = None
        self._device = None
        self._label_to_value = {}
        self._value_to_label = {}
        self._initialized = False
        self._last_results = {}
        self._mask_miss_counts = {}

    def prime(
        self, frame_number: int, frame: np.ndarray, registry: InstanceRegistry
    ) -> None:
        if not self.enabled:
            logger.debug("CutieMaskManager prime skipped because Cutie is disabled.")
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
        mask_tensor = index_numpy_to_one_hot_torch(composed_mask, len(mapping) + 1).to(
            self._device
        )
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
        self._value_to_label = {value: label for label, value in mapping.items()}
        self._initialized = True
        self._seed_last_results(registry)
        logger.debug("CutieMaskManager primed with instances: %s", list(mapping.keys()))

    def update_masks(
        self, frame_number: int, frame: np.ndarray, registry: InstanceRegistry
    ) -> Dict[str, MaskResult]:
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
            binary_mask, corrected = self._sanitize_full_frame_artifact(
                label, binary_mask
            )
            if corrected:
                logger.warning(
                    "Cutie full-frame mask corrected for '%s' at frame %s.",
                    label,
                    frame_number,
                )
            if bool(
                getattr(self.config, "reject_suspicious_mask_jumps", False)
            ) and self._is_suspicious_mask_jump(label, binary_mask):
                logger.warning(
                    "Cutie mask jump rejected for '%s' at frame %s (possible full-frame artifact).",
                    label,
                    frame_number,
                )
                continue
            polygon = self._mask_to_polygon(binary_mask)
            results[label] = MaskResult(
                instance_label=label,
                mask_bitmap=binary_mask,
                polygon=polygon,
            )

        expected_labels = [instance.label for instance in registry]
        results = self._apply_fallbacks(results, expected_labels)
        return results

    def _apply_fallbacks(
        self,
        results: Dict[str, MaskResult],
        expected_labels: List[str],
    ) -> Dict[str, MaskResult]:
        updated: Dict[str, MaskResult] = {}
        for label, result in results.items():
            updated[label] = result
            self._last_results[label] = result
            self._mask_miss_counts[label] = 0

        kernel_size = max(1, int(self.config.mask_dilation_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        iterations = max(0, int(self.config.mask_dilation_iterations))
        allowed_misses = max(0, int(self.config.max_mask_fallback_frames))

        for label in set(expected_labels):
            if label in updated:
                continue
            previous = self._last_results.get(label)
            if previous is None:
                continue
            misses = self._mask_miss_counts.get(label, 0) + 1
            self._mask_miss_counts[label] = misses
            if misses > allowed_misses:
                continue
            fallback_bitmap = previous.mask_bitmap
            if iterations > 0:
                fallback_bitmap = self._dilate_bitmap(
                    fallback_bitmap, kernel_size, iterations
                )
            polygon = self._mask_to_polygon(fallback_bitmap)
            updated[label] = MaskResult(
                instance_label=label,
                mask_bitmap=fallback_bitmap,
                polygon=polygon,
            )
            self._last_results[label] = updated[label]
        return updated

    def _seed_last_results(self, registry: InstanceRegistry) -> None:
        self._last_results = {}
        self._mask_miss_counts = {}
        for instance in registry:
            mask_bitmap = self._resolve_instance_mask(instance)
            if mask_bitmap is None or not mask_bitmap.any():
                continue
            polygon = instance.polygon or self._mask_to_polygon(mask_bitmap)
            self._last_results[instance.label] = MaskResult(
                instance_label=instance.label,
                mask_bitmap=mask_bitmap,
                polygon=polygon,
            )
            self._mask_miss_counts[instance.label] = 0

    def _ensure_core(self) -> None:
        if self._core is not None:
            return
        use_cpu_only = self.config.cutie_device == "cpu"
        processor = CutieCoreVideoProcessor(
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
            (self.adapter.image_height, self.adapter.image_width), dtype=np.uint8
        )
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

    def _dilate_bitmap(
        self,
        bitmap: np.ndarray,
        kernel_size: int,
        iterations: int,
    ) -> np.ndarray:
        if iterations <= 0:
            return bitmap
        cv2 = self._lazy_cv2()
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(bitmap.astype(np.uint8), kernel, iterations=iterations)
        return dilated.astype(bool)

    def _mask_to_polygon(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        cv2 = self._lazy_cv2()
        mask_uint8 = mask.astype(bool).astype(np.uint8) * 255
        if not mask_uint8.any():
            return []

        nonzero = cv2.findNonZero(mask_uint8)
        if nonzero is None:
            return []
        x, y, w, h = cv2.boundingRect(nonzero)
        pad = 2
        height, width = mask_uint8.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)

        cropped = mask_uint8[y0:y1, x0:x1]
        contours, _ = cv2.findContours(
            cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.003 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        points = [(float(pt[0][0] + x0), float(pt[0][1] + y0)) for pt in approx]
        if points and points[0] != points[-1]:
            points.append(points[0])
        return points

    def _handle_failure(self, exc: Exception, message: str) -> None:
        logger.error("%s: %s", message, exc)
        self.enabled = False
        self._initialized = False
        if self.config.error_hook:
            self.config.error_hook(exc)

    def _sanitize_full_frame_artifact(
        self, label: str, mask: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        frame_area = float(self.adapter.image_height * self.adapter.image_width)
        if frame_area <= 0:
            return mask, False
        current_area = float(np.count_nonzero(mask))
        if current_area <= 0:
            return mask, False
        if (current_area / frame_area) < 0.98:
            return mask, False

        previous = self._last_results.get(label)
        if previous is None or previous.mask_bitmap is None:
            return mask, False

        ys, xs = np.where(previous.mask_bitmap)
        if xs.size == 0 or ys.size == 0:
            return mask, False
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())
        bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
        bbox_ratio = bbox_area / frame_area
        if bbox_ratio <= 0.0 or bbox_ratio >= 0.85:
            return mask, False

        cv2 = self._lazy_cv2()
        mask_u8 = mask.astype(np.uint8)
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return mask, False

        h, w = mask.shape[:2]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        pad_x = max(8, int(bw))
        pad_y = max(8, int(bh))
        rx1 = max(0, int(np.floor(x1)) - pad_x)
        ry1 = max(0, int(np.floor(y1)) - pad_y)
        rx2 = min(w, int(np.ceil(x2)) + pad_x)
        ry2 = min(h, int(np.ceil(y2)) + pad_y)
        if rx2 <= rx1 or ry2 <= ry1:
            return mask, False

        best_id = 0
        best_overlap = 0
        best_area = 0
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            overlap = int(np.count_nonzero(components[ry1:ry2, rx1:rx2] == comp_id))
            if overlap <= 0:
                continue
            if overlap > best_overlap or (overlap == best_overlap and area > best_area):
                best_id = comp_id
                best_overlap = overlap
                best_area = area

        if best_id == 0:
            return mask, False

        cleaned = components == best_id
        cleaned_area = float(np.count_nonzero(cleaned))
        if cleaned_area <= 0:
            return mask, False
        cleaned_ratio = cleaned_area / frame_area
        current_ratio = current_area / frame_area
        if cleaned_ratio >= current_ratio or cleaned_ratio >= 0.98:
            return mask, False
        return cleaned, True

    def _is_suspicious_mask_jump(self, label: str, mask: np.ndarray) -> bool:
        """Detect implausible mask expansions that often become full-frame polygons."""
        frame_area = float(self.adapter.image_height * self.adapter.image_width)
        if frame_area <= 0:
            return False

        current_area = float(np.count_nonzero(mask))
        if current_area <= 0:
            return False
        current_ratio = current_area / frame_area
        if current_ratio < 0.998:
            return False

        touches_all_borders = bool(
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        )
        if not touches_all_borders:
            return False

        previous = self._last_results.get(label)
        if previous is None or previous.mask_bitmap is None:
            return current_ratio >= 0.9995

        previous_area = float(np.count_nonzero(previous.mask_bitmap))
        previous_ratio = previous_area / frame_area if previous_area > 0 else 0.0
        if previous_ratio >= 0.80:
            return False

        growth_ratio = current_ratio / max(previous_ratio, 1.0 / frame_area)
        return growth_ratio >= 1.4

    def _lazy_cv2(self):  # pragma: no cover - isolated import
        import cv2

        return cv2
