"""
Annotation Service for Annolid GUI Application.

Handles annotation file operations, validation, and format conversion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


class AnnotationService:
    """
    Domain service for annotation operations.

    Provides business logic for loading, validating, and converting
    annotation files in various formats.
    """

    def __init__(self):
        """Initialize the annotation service."""
        self._supported_formats = {
            "labelme": [".json"],
            "coco": [".json"],
            "voc": [".xml"],
            "yolo": [".txt"],
        }

    def validate_annotation_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """
        Validate an annotation file.

        Args:
            file_path: Path to the annotation file

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                errors.append("File does not exist")
                return False, errors

            # Check file extension
            if file_path.suffix.lower() not in [
                ext for formats in self._supported_formats.values() for ext in formats
            ]:
                errors.append(f"Unsupported file extension: {file_path.suffix}")

            # Try to parse JSON for JSON formats
            if file_path.suffix.lower() == ".json":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Basic validation for LabelMe format
                    if "shapes" in data:
                        for i, shape in enumerate(data["shapes"]):
                            if "label" not in shape:
                                errors.append(f"Shape {i}: Missing 'label' field")
                            if "points" not in shape:
                                errors.append(f"Shape {i}: Missing 'points' field")
                            if "shape_type" not in shape:
                                errors.append(f"Shape {i}: Missing 'shape_type' field")

                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON format: {e}")
                except Exception as e:
                    errors.append(f"Error reading file: {e}")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def convert_annotation_format(
        self, source_format: str, target_format: str, input_data: Any, **kwargs
    ) -> Any:
        """
        Convert annotation between formats.

        Args:
            source_format: Source format (labelme, coco, etc.)
            target_format: Target format
            input_data: Input annotation data
            **kwargs: Additional conversion parameters

        Returns:
            Converted annotation data
        """
        try:
            if source_format == "labelme" and target_format == "coco":
                return self._labelme_to_coco(input_data, **kwargs)
            elif source_format == "coco" and target_format == "labelme":
                return self._coco_to_labelme(input_data, **kwargs)
            elif source_format == "labelme" and target_format == "yolo":
                return self._labelme_to_yolo(input_data, **kwargs)
            else:
                # Return as-is for unsupported conversions
                logger.warning(
                    f"Unsupported conversion: {source_format} -> {target_format}"
                )
                return input_data

        except Exception as e:
            logger.error(f"Failed to convert annotation: {e}")
            return input_data

    def merge_annotations(
        self, annotations_list: List[Dict[str, Any]], merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """
        Merge multiple annotations.

        Args:
            annotations_list: List of annotations to merge
            merge_strategy: Strategy for merging ("combine", "union", "intersection")

        Returns:
            Merged annotation
        """
        if not annotations_list:
            return {}

        try:
            merged = annotations_list[0].copy()

            for annotation in annotations_list[1:]:
                if merge_strategy == "combine":
                    self._merge_combine_strategy(merged, annotation)
                elif merge_strategy == "union":
                    self._merge_union_strategy(merged, annotation)
                elif merge_strategy == "intersection":
                    self._merge_intersection_strategy(merged, annotation)

            return merged

        except Exception as e:
            logger.error(f"Failed to merge annotations: {e}")
            return annotations_list[0] if annotations_list else {}

    def get_annotation_statistics(
        self, annotation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for annotation data.

        Args:
            annotation_data: Annotation data to analyze

        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                "total_shapes": 0,
                "shape_types": {},
                "labels": {},
                "image_info": {},
            }

            if "shapes" in annotation_data:
                shapes = annotation_data["shapes"]
                stats["total_shapes"] = len(shapes)

                for shape in shapes:
                    # Count shape types
                    shape_type = shape.get("shape_type", "unknown")
                    stats["shape_types"][shape_type] = (
                        stats["shape_types"].get(shape_type, 0) + 1
                    )

                    # Count labels
                    label = shape.get("label", "unknown")
                    stats["labels"][label] = stats["labels"].get(label, 0) + 1

            # Image information
            if "imageWidth" in annotation_data and "imageHeight" in annotation_data:
                stats["image_info"] = {
                    "width": annotation_data["imageWidth"],
                    "height": annotation_data["imageHeight"],
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate annotation statistics: {e}")
            return {}

    def validate_annotation_data(
        self, annotation_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate annotation data structure.

        Args:
            annotation_data: Annotation data to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Check for required fields in LabelMe format
            if "shapes" not in annotation_data:
                errors.append("Missing 'shapes' field")

            if "imagePath" not in annotation_data:
                errors.append("Missing 'imagePath' field")

            shapes = annotation_data.get("shapes", [])
            for i, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    errors.append(f"Shape {i}: Must be a dictionary")
                    continue

                required_fields = ["label", "points", "shape_type"]
                for field in required_fields:
                    if field not in shape:
                        errors.append(f"Shape {i}: Missing '{field}' field")

                # Validate points based on shape type
                shape_type = shape.get("shape_type")
                points = shape.get("points", [])

                if shape_type == "rectangle" and len(points) != 2:
                    errors.append(f"Shape {i}: Rectangle must have exactly 2 points")
                elif shape_type == "polygon" and len(points) < 3:
                    errors.append(f"Shape {i}: Polygon must have at least 3 points")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def _labelme_to_coco(
        self, labelme_data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Convert LabelMe format to COCO format."""
        # Implementation for LabelMe to COCO conversion
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # This is a simplified conversion - real implementation would be more complex
        if "shapes" in labelme_data:
            for i, shape in enumerate(labelme_data["shapes"]):
                annotation = {
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": 1,  # Would need proper category mapping
                    "bbox": self._points_to_bbox(shape.get("points", [])),
                    "area": 0,  # Would need to calculate
                    "iscrowd": 0,
                }
                coco_data["annotations"].append(annotation)

        return coco_data

    def _coco_to_labelme(self, coco_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Convert COCO format to LabelMe format."""
        # Implementation for COCO to LabelMe conversion
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": "",
            "imageData": None,
            "imageWidth": 0,
            "imageHeight": 0,
        }

        # This is a simplified conversion - real implementation would be more complex
        if "annotations" in coco_data:
            for annotation in coco_data["annotations"]:
                bbox = annotation.get("bbox", [])
                if len(bbox) == 4:
                    points = [
                        [bbox[0], bbox[1]],  # top-left
                        [bbox[0] + bbox[2], bbox[1] + bbox[3]],  # bottom-right
                    ]

                    shape = {
                        "label": "object",  # Would need proper category mapping
                        "points": points,
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                    labelme_data["shapes"].append(shape)

        return labelme_data

    def _labelme_to_yolo(self, labelme_data: Dict[str, Any], **kwargs) -> List[str]:
        """Convert LabelMe format to YOLO format."""
        yolo_lines = []

        # Get image dimensions
        img_width = labelme_data.get("imageWidth", 1)
        img_height = labelme_data.get("imageHeight", 1)

        if "shapes" in labelme_data:
            for shape in labelme_data["shapes"]:
                if shape.get("shape_type") == "rectangle":
                    points = shape.get("points", [])
                    if len(points) == 2:
                        # Convert to YOLO format: class_id center_x center_y width height (normalized)
                        x1, y1 = points[0]
                        x2, y2 = points[1]

                        center_x = ((x1 + x2) / 2) / img_width
                        center_y = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        yolo_line = "0 %.6f %.6f %.6f %.6f" % (
                            center_x,
                            center_y,
                            width,
                            height,
                        )
                        yolo_lines.append(yolo_line)

        return yolo_lines

    def _points_to_bbox(self, points: List[List[float]]) -> List[float]:
        """Convert polygon points to bounding box [x, y, width, height]."""
        if len(points) < 2:
            return [0, 0, 0, 0]

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def _merge_combine_strategy(
        self, merged: Dict[str, Any], annotation: Dict[str, Any]
    ) -> None:
        """Combine strategy: append all shapes."""
        if "shapes" in annotation:
            if "shapes" not in merged:
                merged["shapes"] = []
            merged["shapes"].extend(annotation["shapes"])

    def _merge_union_strategy(
        self, merged: Dict[str, Any], annotation: Dict[str, Any]
    ) -> None:
        """Union strategy: merge overlapping shapes."""
        # Implementation for union merging
        self._merge_combine_strategy(merged, annotation)  # Simplified

    def _merge_intersection_strategy(
        self, merged: Dict[str, Any], annotation: Dict[str, Any]
    ) -> None:
        """Intersection strategy: keep only overlapping shapes."""
        # Implementation for intersection merging
        self._merge_combine_strategy(merged, annotation)  # Simplified
