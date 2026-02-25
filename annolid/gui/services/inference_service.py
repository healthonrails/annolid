"""
Inference Service for Annolid GUI Application.

Handles AI model inference, result processing, and model management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class InferenceResult(TypedDict):
    model_type: str
    detections: List[Dict[str, Any]]
    masks: List[Any]
    keypoints: List[Any]
    results: Any
    meta: Dict[str, Any]
    error: Optional[str]


class InferenceService:
    """
    Domain service for AI inference operations.

    Provides business logic for running AI models, processing results,
    and managing inference workflows.
    """

    def __init__(self):
        """Initialize the inference service."""
        self._model_cache: Dict[str, Any] = {}
        self._inference_configs: Dict[str, Dict[str, Any]] = {}

    def validate_model_config(
        self, model_config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a model configuration.

        Args:
            model_config: Model configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required_fields = ["identifier", "weight_file"]
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Missing required field: {field}")

        # Validate weight file exists
        weight_file = model_config.get("weight_file", "")
        if weight_file and not Path(weight_file).exists():
            errors.append(f"Weight file does not exist: {weight_file}")

        # Validate model type if specified
        model_type = model_config.get("model_type", "")
        valid_types = ["yolo", "sam", "dino", "efficienttam", "cotracker"]
        if model_type and model_type not in valid_types:
            errors.append(f"Invalid model type: {model_type}")

        return len(errors) == 0, errors

    def prepare_inference_input(
        self, model_type: str, input_data: Any, model_config: Dict[str, Any]
    ) -> Any:
        """
        Prepare input data for inference based on model type.

        Args:
            model_type: Type of model (yolo, sam, etc.)
            input_data: Raw input data
            model_config: Model configuration

        Returns:
            Prepared input data
        """
        try:
            if model_type == "yolo":
                return self._prepare_yolo_input(input_data, model_config)
            elif model_type in ["sam", "sam2", "sam3"]:
                return self._prepare_sam_input(input_data, model_config)
            elif model_type == "dino":
                return self._prepare_dino_input(input_data, model_config)
            else:
                return input_data
        except Exception as e:
            logger.error(f"Failed to prepare inference input: {e}")
            return input_data

    def _prepare_yolo_input(self, input_data: Any, model_config: Dict[str, Any]) -> Any:
        """Prepare input for YOLO models."""
        # YOLO typically expects numpy arrays or PIL images
        if isinstance(input_data, np.ndarray):
            return input_data
        return input_data

    def _prepare_sam_input(self, input_data: Any, model_config: Dict[str, Any]) -> Any:
        """Prepare input for SAM models."""
        # SAM expects specific input formats
        return input_data

    def _prepare_dino_input(self, input_data: Any, model_config: Dict[str, Any]) -> Any:
        """Prepare input for DINO models."""
        # DINO expects specific input formats
        return input_data

    def process_inference_results(
        self,
        model_type: str,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        Process raw inference results into standardized format.

        Args:
            model_type: Type of model that produced results
            raw_results: Raw results from inference
            model_config: Model configuration
            postprocessing_config: Postprocessing configuration

        Returns:
            Processed results in standardized format
        """
        try:
            if model_type == "yolo":
                parsed = self._process_yolo_results(
                    raw_results, model_config, postprocessing_config
                )
            elif model_type in ["sam", "sam2", "sam3"]:
                parsed = self._process_sam_results(
                    raw_results, model_config, postprocessing_config
                )
            elif model_type == "dino":
                parsed = self._process_dino_results(
                    raw_results, model_config, postprocessing_config
                )
            else:
                parsed = self._result_template(
                    model_type=model_type, results=raw_results
                )
            return self.normalize_inference_result(parsed, model_type_hint=model_type)
        except Exception as e:
            logger.error(f"Failed to process inference results: {e}")
            return self.normalize_inference_result(
                self._result_template(
                    model_type=model_type,
                    results=raw_results,
                    error=str(e),
                ),
                model_type_hint=model_type,
            )

    def normalize_inference_result(
        self, payload: Any, model_type_hint: str = ""
    ) -> InferenceResult:
        """Normalize legacy/partial payloads to the stable inference schema."""
        if not isinstance(payload, dict):
            return self._result_template(
                model_type=model_type_hint or "unknown", results=payload
            )

        model_type = str(payload.get("model_type") or model_type_hint or "unknown")
        detections = payload.get("detections")
        masks = payload.get("masks")
        keypoints = payload.get("keypoints")
        meta = payload.get("meta")
        error = payload.get("error")
        results = payload.get("results", payload)

        if not isinstance(detections, list):
            detections = []
        if not isinstance(masks, list):
            masks = []
        if not isinstance(keypoints, list):
            keypoints = []
        if not isinstance(meta, dict):
            meta = {}
        if error is not None and not isinstance(error, str):
            error = str(error)

        return self._result_template(
            model_type=model_type,
            detections=detections,
            masks=masks,
            keypoints=keypoints,
            results=results,
            meta=meta,
            error=error,
        )

    @staticmethod
    def _resolve_class_name(
        class_names: Any,
        class_id: int,
    ) -> str:
        if isinstance(class_names, dict):
            name = class_names.get(class_id, class_names.get(str(class_id), "unknown"))
            return str(name)
        if isinstance(class_names, (list, tuple)):
            if 0 <= class_id < len(class_names):
                return str(class_names[class_id])
        return f"class_{class_id}"

    def _extract_yolo_object_detections(
        self,
        yolo_obj: Any,
        *,
        confidence_threshold: float,
        class_names: Any,
    ) -> List[Dict[str, Any]]:
        boxes = getattr(yolo_obj, "boxes", None)
        if boxes is None:
            return []
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        if xyxy is None or conf is None or cls is None:
            return []

        detections: List[Dict[str, Any]] = []
        for box, score, class_id in zip(xyxy, conf, cls):
            score_value = float(score)
            if score_value < confidence_threshold:
                continue
            class_id_value = int(class_id)
            detections.append(
                {
                    "bbox": box.tolist() if hasattr(box, "tolist") else list(box),
                    "confidence": score_value,
                    "class_id": class_id_value,
                    "class_name": self._resolve_class_name(class_names, class_id_value),
                }
            )
        return detections

    def _extract_yolo_dict_detections(
        self,
        raw_results: Dict[str, Any],
        *,
        confidence_threshold: float,
        class_names: Any,
    ) -> List[Dict[str, Any]]:
        raw_detections = raw_results.get("detections")
        if not isinstance(raw_detections, list):
            return []
        detections: List[Dict[str, Any]] = []
        for item in raw_detections:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < confidence_threshold:
                continue
            class_id = int(item.get("class_id", -1))
            class_name = item.get("class_name")
            if not class_name:
                class_name = self._resolve_class_name(class_names, class_id)
            bbox = item.get("bbox", [])
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            detections.append(
                {
                    "bbox": list(bbox) if isinstance(bbox, Iterable) else [],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": str(class_name),
                }
            )
        return detections

    def _collect_yolo_detections(
        self,
        raw_results: Any,
        *,
        confidence_threshold: float,
        class_names: Any,
    ) -> List[Dict[str, Any]]:
        if isinstance(raw_results, dict):
            detections = self._extract_yolo_dict_detections(
                raw_results,
                confidence_threshold=confidence_threshold,
                class_names=class_names,
            )
            if detections:
                return detections
        if isinstance(raw_results, (list, tuple)):
            all_detections: List[Dict[str, Any]] = []
            for result in raw_results:
                all_detections.extend(
                    self._extract_yolo_object_detections(
                        result,
                        confidence_threshold=confidence_threshold,
                        class_names=class_names,
                    )
                )
            return all_detections
        return self._extract_yolo_object_detections(
            raw_results,
            confidence_threshold=confidence_threshold,
            class_names=class_names,
        )

    @staticmethod
    def _result_template(
        *,
        model_type: str,
        detections: Optional[List[Dict[str, Any]]] = None,
        masks: Optional[List[Any]] = None,
        keypoints: Optional[List[Any]] = None,
        results: Any = None,
        meta: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> InferenceResult:
        return {
            "model_type": str(model_type or ""),
            "detections": detections or [],
            "masks": masks or [],
            "keypoints": keypoints or [],
            "results": results,
            "meta": meta or {},
            "error": error,
        }

    def _process_yolo_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Process YOLO inference results."""
        confidence_threshold = float(model_config.get("confidence_threshold", 0.5))
        class_names = model_config.get("class_names", [])
        detections = self._collect_yolo_detections(
            raw_results,
            confidence_threshold=confidence_threshold,
            class_names=class_names,
        )

        return self._result_template(
            model_type="yolo",
            detections=detections,
            results=raw_results,
            meta={"confidence_threshold": confidence_threshold},
        )

    def _process_sam_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Process SAM inference results."""
        masks: List[Any] = []

        # Process raw results into standardized format
        if isinstance(raw_results, dict) and "masks" in raw_results:
            masks = list(raw_results["masks"] or [])

        return self._result_template(
            model_type="sam",
            masks=masks,
            results=raw_results,
        )

    def _process_dino_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """Process DINO inference results."""
        keypoints: List[Any] = []

        # Process raw results into standardized format
        if isinstance(raw_results, dict) and "keypoints" in raw_results:
            keypoints = list(raw_results["keypoints"] or [])

        return self._result_template(
            model_type="dino",
            keypoints=keypoints,
            results=raw_results,
        )

    def validate_inference_results(
        self,
        results: Dict[str, Any],
        validation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate inference results.

        Args:
            results: Inference results to validate
            validation_config: Validation configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        normalized = self.normalize_inference_result(results)

        if not normalized.get("model_type"):
            errors.append("model_type must be a non-empty string")
        if not isinstance(normalized.get("detections"), list):
            errors.append("detections must be a list")
        if not isinstance(normalized.get("masks"), list):
            errors.append("masks must be a list")
        if not isinstance(normalized.get("keypoints"), list):
            errors.append("keypoints must be a list")
        if not isinstance(normalized.get("meta"), dict):
            errors.append("meta must be a dictionary")
        if normalized.get("error") is not None and not isinstance(
            normalized.get("error"), str
        ):
            errors.append("error must be a string or None")

        return len(errors) == 0, errors

    def merge_inference_results(
        self, results_list: List[Dict[str, Any]], merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """
        Merge multiple inference results.

        Args:
            results_list: List of inference results to merge
            merge_strategy: Strategy for merging ("combine", "average", "max_confidence")

        Returns:
            Merged results
        """
        if not results_list:
            return {}

        try:
            merged = results_list[0].copy()

            for results in results_list[1:]:
                if merge_strategy == "combine":
                    self._merge_combine_strategy(merged, results)
                elif merge_strategy == "average":
                    self._merge_average_strategy(merged, results)
                elif merge_strategy == "max_confidence":
                    self._merge_max_confidence_strategy(merged, results)

            return merged
        except Exception as e:
            logger.error(f"Failed to merge inference results: {e}")
            return results_list[0] if results_list else {}

    def _merge_combine_strategy(
        self, merged: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Combine strategy: append all results."""
        for key, value in results.items():
            if (
                key in merged
                and isinstance(merged[key], list)
                and isinstance(value, list)
            ):
                merged[key].extend(value)
            else:
                merged[key] = value

    def _merge_average_strategy(
        self, merged: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Average strategy: average numeric values."""
        # Implementation for averaging numeric results
        pass

    def _merge_max_confidence_strategy(
        self, merged: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Max confidence strategy: keep highest confidence detections."""
        # Implementation for max confidence merging
        pass

    def filter_results_by_confidence(
        self, results: Dict[str, Any], min_confidence: float
    ) -> Dict[str, Any]:
        """
        Filter inference results by confidence threshold.

        Args:
            results: Inference results to filter
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered results
        """
        try:
            filtered = results.copy()

            if results.get("model_type") == "yolo" and "detections" in results:
                filtered["detections"] = [
                    det
                    for det in results["detections"]
                    if det.get("confidence", 0) >= min_confidence
                ]

            return filtered
        except Exception as e:
            logger.error(f"Failed to filter results by confidence: {e}")
            return results

    def convert_results_to_labelme_format(
        self, results: Dict[str, Any], image_path: str, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Convert inference results to LabelMe annotation format.

        Args:
            results: Inference results
            image_path: Path to the source image
            image_size: Image dimensions (width, height)

        Returns:
            LabelMe formatted annotation data
        """
        try:
            annotation = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [],
                "imagePath": Path(image_path).name,
                "imageData": None,
                "imageWidth": image_size[0],
                "imageHeight": image_size[1],
                "otherData": {},
            }

            model_type = results.get("model_type", "")

            if model_type == "yolo" and "detections" in results:
                for detection in results["detections"]:
                    bbox = detection["bbox"]
                    # Convert bbox [x1, y1, x2, y2] to LabelMe rectangle format
                    points = [
                        [bbox[0], bbox[1]],  # top-left
                        [bbox[2], bbox[3]],  # bottom-right
                    ]

                    shape = {
                        "label": detection["class_name"],
                        "points": points,
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                        "other_data": {
                            "confidence": detection["confidence"],
                            "class_id": detection["class_id"],
                        },
                    }
                    annotation["shapes"].append(shape)

            return annotation
        except Exception as e:
            logger.error(f"Failed to convert results to LabelMe format: {e}")
            return {}

    def get_inference_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for inference results.

        Args:
            results: Inference results to analyze

        Returns:
            Dictionary with statistics
        """
        try:
            stats = {
                "model_type": results.get("model_type", "unknown"),
                "total_detections": 0,
                "average_confidence": 0.0,
                "class_distribution": {},
            }

            if results.get("model_type") == "yolo" and "detections" in results:
                detections = results["detections"]
                stats["total_detections"] = len(detections)

                if detections:
                    confidences = [det["confidence"] for det in detections]
                    stats["average_confidence"] = sum(confidences) / len(confidences)

                    for det in detections:
                        class_name = det["class_name"]
                        stats["class_distribution"][class_name] = (
                            stats["class_distribution"].get(class_name, 0) + 1
                        )

            return stats
        except Exception as e:
            logger.error(f"Failed to calculate inference statistics: {e}")
            return {}
