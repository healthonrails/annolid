import os
import json
from labelme.utils.shape import shapes_to_label
from typing import Dict, List, Tuple, Any
from annolid.utils.logger import logger


class LabelProcessor:
    EXCLUDED_LABEL_KEYWORDS = {'zone'}

    def __init__(self, label_json_file: str):
        """
        Initialize the LabelProcessor with data from a JSON file.

        Args:
            label_json_file (str): Path to the label JSON file.
        """
        self.data = self.load_label_json(label_json_file)
        self.image_size = self.get_image_size(self.data)
        self.shapes = self.filter_shapes(self.data.get('shapes', []))
        self.label_name_to_value = self.generate_label_mapping(self.shapes)
        self.id_to_label_mapping = {v: k for k,
                                    v in self.label_name_to_value.items()}

    @staticmethod
    def load_label_json(label_json_file: str) -> Dict[str, Any]:
        """Load label data from a JSON file."""
        try:
            with open(label_json_file, 'r') as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            logger.error(f"File not found: {label_json_file}")
            return {}
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON from file: {label_json_file}")
            return {}

    @staticmethod
    def filter_shapes(shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out shapes with excluded keywords in their labels or descriptions."""
        return [
            shape for shape in shapes
            if not any(shape['description'] and keyword in shape['description'].lower() or keyword in shape['label'].lower()
                       for keyword in LabelProcessor.EXCLUDED_LABEL_KEYWORDS)
        ]

    @staticmethod
    def generate_label_mapping(shapes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate a dictionary mapping label names to unique integer values."""
        label_name_to_value = {"_background_": 0}
        for shape in sorted(shapes, key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name not in label_name_to_value:
                label_name_to_value[label_name] = len(label_name_to_value)
        return label_name_to_value

    @staticmethod
    def get_image_size(data: Dict[str, Any]) -> Tuple[int, int]:
        """Extract image size (height, width) from the JSON data."""
        return data.get('imageHeight', 0), data.get('imageWidth', 0)

    def get_mask(self, shape) -> Any:
        """Generate and return the binary mask using the stored shapes and label mapping."""
        mask, _ = shapes_to_label(
            self.image_size, [shape], self.label_name_to_value)
        return mask

    def get_label_mapping(self) -> Dict[str, int]:
        """Return the label name to value mapping."""
        return self.label_name_to_value

    def get_id_to_labels(self) -> Dict[int, str]:
        """Return the ID to label name mapping."""
        return self.id_to_label_mapping

    def convert_shapes_to_annotations(self,ann_frame_idx=0) -> List[Dict[str, Any]]:
        """
        Convert LabelMe shapes to a custom annotations format.

        Returns:
            list: List of annotations with 'type', 'points', 'labels', and 'obj_id'.
        """
        annotations = []
        obj_id = 1  # Starting object ID

        for shape in self.shapes:
            shape_type = shape['shape_type']
            points = shape['points']
            label = shape['label']
            label_value = self.label_name_to_value.get(
                label, 0)  # Default to 0 if label not found

            if shape_type == 'rectangle':
                points = self.convert_rectangle_to_points(points)
                shape_type = 'box'
            elif shape_type == 'polygon':
                _mask = self.get_mask(shape)
                shape_type = 'mask'

            annotation = {
                'type': shape_type,
                shape_type: _mask if shape_type == 'mask' else points,
                'labels': [label_value] * len(points) if shape_type == 'points' else [label_value],
                'obj_id': obj_id,
                'ann_frame_idx': ann_frame_idx,
            }
            annotations.append(annotation)
            obj_id += 1  # Increment obj_id for each shape

        return annotations

    @staticmethod
    def convert_rectangle_to_points(rectangle: List[List[float]]) -> List[float]:
        """
        Convert a rectangle shape to a list of points.

        Args:
            rectangle (list): Rectangle coordinates as [[x_min, y_min], [x_max, y_max]]

        Returns:
            list: List of points as [x_min, y_min, x_max, y_max]
        """
        x_min, y_min, x_max, y_max = [*rectangle[0], *rectangle[1]]
        return [x_min, y_min, x_max, y_max]


if __name__ == '__main__':
    # Path to your LabelMe JSON file
    label_json_file = os.path.expanduser('~/Downloads/mouse/00000.json')
    # Create an instance of the LabelProcessor class with the JSON file
    label_processor = LabelProcessor(label_json_file)
    # Convert shapes to the custom annotations format
    annotations = label_processor.convert_shapes_to_annotations()
    logger.info(f"ID to Labels Mapping:{label_processor.get_id_to_labels()}")
    # Output the annotations
    logger.info(f"Annotations:{annotations}")
