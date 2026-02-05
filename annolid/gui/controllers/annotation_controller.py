"""
Annotation Controller for Annolid GUI Application.

Handles UI interactions and coordinates annotation operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from qtpy import QtCore

from ..interfaces.services import IAnnotationService
from ..services import AnnotationService

logger = logging.getLogger(__name__)


class AnnotationController(QtCore.QObject):
    """
    Controller for annotation-related UI operations.

    Coordinates between the UI and annotation service, handling
    user interactions and business logic orchestration.
    """

    # Signals
    annotation_loaded = QtCore.Signal(dict)  # Emitted when annotation is loaded
    annotation_saved = QtCore.Signal(str)  # Emitted when annotation is saved
    annotation_error = QtCore.Signal(str)  # Emitted on annotation errors
    progress_updated = QtCore.Signal(int, str)  # Progress updates

    def __init__(
        self,
        annotation_service: Optional[IAnnotationService] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Initialize the annotation controller.

        Args:
            annotation_service: Annotation service instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._annotation_service = annotation_service or AnnotationService()
        self._current_annotation: Optional[Dict[str, Any]] = None
        self._current_file_path: Optional[Path] = None

    def load_annotation(self, file_path: Union[str, Path]) -> bool:
        """
        Load an annotation file.

        Args:
            file_path: Path to the annotation file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Validate file exists
            if not file_path.exists():
                self.annotation_error.emit(
                    f"Annotation file does not exist: {file_path}"
                )
                return False

            # Validate file format
            is_valid, errors = self._annotation_service.validate_annotation_file(
                file_path
            )
            if not is_valid:
                error_msg = f"Invalid annotation file: {'; '.join(errors)}"
                self.annotation_error.emit(error_msg)
                return False

            # Load annotation data
            with open(file_path, "r", encoding="utf-8") as f:
                import json

                annotation_data = json.load(f)

            self._current_annotation = annotation_data
            self._current_file_path = file_path

            self.annotation_loaded.emit(annotation_data)
            logger.info(f"Annotation loaded: {file_path}")

            return True

        except Exception as e:
            error_msg = f"Failed to load annotation: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def save_annotation(
        self,
        annotation_data: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Save annotation data to file.

        Args:
            annotation_data: Annotation data to save
            file_path: Optional file path (uses current if not provided)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            save_path = Path(file_path) if file_path else self._current_file_path

            if not save_path:
                self.annotation_error.emit("No file path specified for saving")
                return False

            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save annotation data
            with open(save_path, "w", encoding="utf-8") as f:
                import json

                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            self._current_annotation = annotation_data
            self._current_file_path = save_path

            self.annotation_saved.emit(str(save_path))
            logger.info(f"Annotation saved: {save_path}")

            return True

        except Exception as e:
            error_msg = f"Failed to save annotation: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def create_new_annotation(
        self, image_path: Union[str, Path], image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Create a new empty annotation for an image.

        Args:
            image_path: Path to the image
            image_size: Image dimensions (width, height)

        Returns:
            New annotation data
        """
        try:
            image_path = Path(image_path)

            annotation_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [],
                "imagePath": image_path.name,
                "imageData": None,
                "imageWidth": image_size[0],
                "imageHeight": image_size[1],
                "otherData": {},
            }

            self._current_annotation = annotation_data
            self._current_file_path = None  # No file path yet

            self.annotation_loaded.emit(annotation_data)
            logger.info(f"New annotation created for: {image_path}")

            return annotation_data

        except Exception as e:
            error_msg = f"Failed to create new annotation: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return {}

    def add_shape_to_annotation(self, shape_data: Dict[str, Any]) -> bool:
        """
        Add a shape to the current annotation.

        Args:
            shape_data: Shape data to add

        Returns:
            True if added successfully, False otherwise
        """
        if not self._current_annotation:
            self.annotation_error.emit("No annotation loaded")
            return False

        try:
            # Validate shape data
            required_fields = ["label", "points", "shape_type"]
            for field in required_fields:
                if field not in shape_data:
                    self.annotation_error.emit(
                        f"Missing required field in shape: {field}"
                    )
                    return False

            # Add shape to annotation
            if "shapes" not in self._current_annotation:
                self._current_annotation["shapes"] = []

            self._current_annotation["shapes"].append(shape_data)

            logger.info(
                f"Shape added to annotation: {shape_data.get('label', 'Unknown')}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to add shape: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def remove_shape_from_annotation(self, shape_index: int) -> bool:
        """
        Remove a shape from the current annotation.

        Args:
            shape_index: Index of the shape to remove

        Returns:
            True if removed successfully, False otherwise
        """
        if not self._current_annotation:
            self.annotation_error.emit("No annotation loaded")
            return False

        try:
            shapes = self._current_annotation.get("shapes", [])
            if shape_index < 0 or shape_index >= len(shapes):
                self.annotation_error.emit(f"Invalid shape index: {shape_index}")
                return False

            removed_shape = shapes.pop(shape_index)
            logger.info(
                f"Shape removed from annotation: {removed_shape.get('label', 'Unknown')}"
            )

            return True

        except Exception as e:
            error_msg = f"Failed to remove shape: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def convert_annotation_format(
        self, source_format: str, target_format: str, **kwargs
    ) -> Optional[Any]:
        """
        Convert the current annotation to a different format.

        Args:
            source_format: Source format
            target_format: Target format
            **kwargs: Additional conversion parameters

        Returns:
            Converted annotation data or None on failure
        """
        if not self._current_annotation:
            self.annotation_error.emit("No annotation loaded")
            return None

        try:
            converted = self._annotation_service.convert_annotation_format(
                source_format, target_format, self._current_annotation, **kwargs
            )

            logger.info(f"Annotation converted from {source_format} to {target_format}")
            return converted

        except Exception as e:
            error_msg = f"Failed to convert annotation: {str(e)}"
            self.annotation_error.emit(error_msg)
            logger.error(error_msg)
            return None

    def get_annotation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the current annotation.

        Returns:
            Dictionary with annotation statistics
        """
        if not self._current_annotation:
            return {}

        try:
            shapes = self._current_annotation.get("shapes", [])
            stats = {
                "total_shapes": len(shapes),
                "shape_types": {},
                "labels": {},
            }

            for shape in shapes:
                # Count shape types
                shape_type = shape.get("shape_type", "unknown")
                stats["shape_types"][shape_type] = (
                    stats["shape_types"].get(shape_type, 0) + 1
                )

                # Count labels
                label = shape.get("label", "unknown")
                stats["labels"][label] = stats["labels"].get(label, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Failed to get annotation statistics: {e}")
            return {}

    def validate_current_annotation(self) -> Tuple[bool, List[str]]:
        """
        Validate the current annotation.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self._current_annotation:
            return False, ["No annotation loaded"]

        try:
            # Basic validation
            errors = []

            if "shapes" not in self._current_annotation:
                errors.append("Missing 'shapes' field")

            if "imagePath" not in self._current_annotation:
                errors.append("Missing 'imagePath' field")

            shapes = self._current_annotation.get("shapes", [])
            for i, shape in enumerate(shapes):
                if "label" not in shape:
                    errors.append(f"Shape {i}: Missing 'label' field")
                if "points" not in shape:
                    errors.append(f"Shape {i}: Missing 'points' field")
                if "shape_type" not in shape:
                    errors.append(f"Shape {i}: Missing 'shape_type' field")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def get_current_annotation(self) -> Optional[Dict[str, Any]]:
        """
        Get the current annotation data.

        Returns:
            Current annotation data or None
        """
        return self._current_annotation.copy() if self._current_annotation else None

    def get_current_file_path(self) -> Optional[Path]:
        """
        Get the current annotation file path.

        Returns:
            Current file path or None
        """
        return self._current_file_path

    def has_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes.

        Returns:
            True if there are unsaved changes, False otherwise
        """
        # For now, always return False since we don't track changes
        # This could be enhanced to track modifications
        return False
