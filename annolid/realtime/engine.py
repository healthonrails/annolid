from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from annolid.utils.logger import logger
from annolid.yolo import configure_ultralytics_cache, load_yolo_model, select_backend


_engine_cache: Dict[str, "InferenceEngine"] = {}


def get_engine(model_name_base: str = "yolo11n-seg") -> "InferenceEngine":
    """
    Return a cached InferenceEngine for the requested model.

    The cache key is the resolved weight path chosen by Annolid's shared YOLO
    backend selection logic, ensuring consistent reuse across call sites.
    """
    configure_ultralytics_cache()
    spec = select_backend(model_name_base)
    cache_key = str(spec.weight_path)

    cached = _engine_cache.get(cache_key)
    if cached is not None:
        logger.debug("Returning cached InferenceEngine for %s", cache_key)
        return cached

    logger.info("Creating new InferenceEngine instance for %s", cache_key)
    engine = InferenceEngine(model_name_base)
    _engine_cache[cache_key] = engine
    return engine


class InferenceEngine:
    """Hardware-agnostic realtime inference wrapper around Ultralytics YOLO."""

    def __init__(self, model_name_base: str):
        configure_ultralytics_cache()
        model, spec = load_yolo_model(model_name_base)

        self.model = model
        self.model_path = str(spec.weight_path)
        self.device = spec.device
        self.model_type = spec.backend
        self.class_names = getattr(self.model, "names", None)

        logger.info(
            "InferenceEngine initialized with %s on device '%s'.",
            self.model_type,
            self.device,
        )
        self._warmup()

    def _warmup(self) -> None:
        """Run a single dummy inference to reduce first-frame latency."""
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.infer_single_frame(dummy_image)
        except Exception as exc:
            logger.debug("Warmup inference failed: %s", exc, exc_info=True)

    def infer_single_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]:
        """
        Run inference on a single BGR frame.

        Returns:
            (boxes_xyxy, masks, class_ids, scores), or (None, None, None, None) when no detections.
        """
        try:
            results = self.model(
                frame,
                stream=False,
                verbose=False,
                device=self.device,
            )
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                return None, None, None, None

            boxes = result.boxes.xyxy.cpu().numpy()
            masks = (
                result.masks.data.cpu().numpy()
                if getattr(result, "masks", None) is not None
                else None
            )
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
            scores = result.boxes.conf.cpu().numpy()
            return boxes, masks, class_ids, scores
        except Exception as exc:
            logger.error(
                "An error occurred during single-frame inference: %s",
                exc,
                exc_info=True,
            )
            return None, None, None, None
