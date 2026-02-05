"""
Inference Service for Annolid GUI Application.

Handles AI model inference, result processing, and model management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    ) -> Dict[str, Any]:
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
                return self._process_yolo_results(
                    raw_results, model_config, postprocessing_config
                )
            elif model_type in ["sam", "sam2", "sam3"]:
                return self._process_sam_results(
                    raw_results, model_config, postprocessing_config
                )
            elif model_type == "dino":
                return self._process_dino_results(
                    raw_results, model_config, postprocessing_config
                )
            else:
                return {"results": raw_results}
        except Exception as e:
            logger.error(f"Failed to process inference results: {e}")
            return {"results": raw_results, "error": str(e)}

    def _process_yolo_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process YOLO inference results."""
        # Standardize YOLO results format
        processed = {
            "detections": [],
            "model_type": "yolo",
            "confidence_threshold": model_config.get("confidence_threshold", 0.5),
        }

        # Process raw results into standardized format
        if hasattr(raw_results, "boxes") and hasattr(raw_results, "scores"):
            # Assume ultralytics format
            boxes = raw_results.boxes
            for i, (box, score, cls) in enumerate(
                zip(boxes.xyxy, boxes.conf, boxes.cls)
            ):
                if score >= processed["confidence_threshold"]:
                    detection = {
                        "bbox": box.tolist(),
                        "confidence": float(score),
                        "class_id": int(cls),
                        "class_name": model_config.get("class_names", ["unknown"])[
                            int(cls)
                        ],
                    }
                    processed["detections"].append(detection)

        return processed

    def _process_sam_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process SAM inference results."""
        # Standardize SAM results format
        processed = {
            "masks": [],
            "model_type": "sam",
        }

        # Process raw results into standardized format
        if isinstance(raw_results, dict) and "masks" in raw_results:
            processed["masks"] = raw_results["masks"]

        return processed

    def _process_dino_results(
        self,
        raw_results: Any,
        model_config: Dict[str, Any],
        postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process DINO inference results."""
        # Standardize DINO results format
        processed = {
            "keypoints": [],
            "model_type": "dino",
        }

        # Process raw results into standardized format
        if isinstance(raw_results, dict) and "keypoints" in raw_results:
            processed["keypoints"] = raw_results["keypoints"]

        return processed

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

        if not isinstance(results, dict):
            errors.append("Results must be a dictionary")
            return False, errors

        # Check for required fields based on model type
        model_type = results.get("model_type", "")
        if not model_type:
            errors.append("Missing model_type in results")

        # Model-specific validation
        if model_type == "yolo":
            if "detections" not in results:
                errors.append("YOLO results missing detections field")
        elif model_type in ["sam", "sam2", "sam3"]:
            if "masks" not in results:
                errors.append("SAM results missing masks field")
        elif model_type == "dino":
            if "keypoints" not in results:
                errors.append("DINO results missing keypoints field")

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
