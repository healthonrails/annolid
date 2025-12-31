# How to use it?
# python annolid/main.py --labelme2yolo /path/to/labelme_json_folder/ --val_size 0.1 --test_size 0.1
# Refer to https://docs.ultralytics.com/datasets/pose/#dataset-yaml-format for more details.
import math
import os
from pathlib import Path
import numpy as np
import PIL.Image
import shutil
from typing import Dict, Iterable, List, Optional, Set, Tuple
from random import Random
from collections import OrderedDict
from labelme.utils.image import img_b64_to_arr
try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency
    train_test_split = None
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.annotation.pose_schema import PoseSchema
from annolid.behavior.project_schema import (
    DEFAULT_SCHEMA_FILENAME,
    load_schema as load_project_schema,
)


def point_list_to_numpy_array(point_list: List[str]) -> np.ndarray:
    """
    Given a list of points, this function extends the bounding box
    of the points and returns it as a NumPy array.

    Args:
    - point_list: A list of strings representing the points in (x,y) format.

    Returns:
    - A NumPy array of shape (8,) representing the extended bounding box of the points,
      in the order of [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax].
    """
    # Extract the x and y coordinates from the list of points
    x_coords = [float(point) for point in point_list[::2]]
    y_coords = [float(point) for point in point_list[1::2]]

    # Find the minimum and maximum x and y coordinates
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)

    # Return the extended bounding box as a NumPy array
    return np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])


def find_bbox_from_shape(shape):
    """
    Calculate the bounding box coordinates (cx, cy, width, height) from the labelme shape.

    Args:
        shape (dict): Labelme shape dictionary containing the "points" list.

    Returns:
        tuple: A tuple containing the center coordinates (cx, cy) and the width and height of the bounding box.

    Example:
        shape = {
            "label": "rat",
            "points": [
                [523.4615384615385, 196.15384615384613],
                [515.7692307692307, 217.69230769230768],
                ...
            ],
            "group_id": null,
            "shape_type": "polygon",
            "flags": {}
        }

        cx, cy, width, height = find_bbox_from_shape(shape)
    """
    points = shape["points"]

    # Extract x and y coordinates separately
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Calculate minimum and maximum x and y values
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Calculate center coordinates (cx, cy)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2

    # Calculate width and height
    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    return cx, cy, width, height


class Labelme2YOLO:
    """Class that converts Labelme JSON files to YOLO format.

    Args:
        json_dir (str): The directory that contains the Labelme JSON files.

    Attributes:
        json_file_dir (str): The directory that contains the Labelme JSON files.
        label_to_id_dict (collections.OrderedDict): A dictionary that maps label names to label IDs.

    """

    def __init__(self,
                 json_dir,
                 yolo_dataset_name="YOLO_dataset",
                 include_visibility=False,
                 pose_schema_path: Optional[str] = None,
                 **_ignored_kwargs,
                 ):
        self.json_file_dir = json_dir
        labels, keypoints = self._scan_labels_and_keypoints(self.json_file_dir)
        self.label_to_id_dict = OrderedDict(
            (label, label_id) for label_id, label in enumerate(labels)
        )
        self.yolo_dataset_name = yolo_dataset_name
        self.annotation_type = "segmentation"
        # e.g. [17, 2] or [17, 3] if visibility is included
        self.kpt_shape = None
        self.include_visibility = include_visibility
        if pose_schema_path is None:
            base = Path(self.json_file_dir).expanduser()
            for name in ("pose_schema.json", "pose_schema.yaml", "pose_schema.yml"):
                candidate = base / name
                if candidate.exists():
                    pose_schema_path = str(candidate)
                    break

        self.pose_schema_path = pose_schema_path
        self.pose_schema: Optional[PoseSchema] = None
        if pose_schema_path:
            try:
                self.pose_schema = PoseSchema.load(pose_schema_path)
            except Exception:
                self.pose_schema = None

        if self.pose_schema is None:
            try:
                project_path = Path(
                    self.json_file_dir).expanduser() / DEFAULT_SCHEMA_FILENAME
                if project_path.exists():
                    project_schema = load_project_schema(project_path)
                    embedded = getattr(project_schema, "pose_schema", None)
                    if isinstance(embedded, dict) and embedded:
                        self.pose_schema = PoseSchema.from_dict(embedded)
            except Exception:
                self.pose_schema = None

        if self.pose_schema and self.pose_schema.keypoints:
            merged = list(self.pose_schema.keypoints)
            for kp in keypoints:
                if kp not in merged:
                    merged.append(kp)
            self.keypoint_labels_order = merged
        else:
            self.keypoint_labels_order = list(keypoints)
        if self.keypoint_labels_order:
            dims = 3 if self.include_visibility else 2
            self.kpt_shape = [len(self.keypoint_labels_order), dims]

    def create_yolo_dataset_dirs(self):
        """
        Create necessary directories for YOLO dataset and delete 
        any existing directories with the same name.

        Args:
            None

        Returns:
            None
        """

        # Define label and image directory paths
        self.label_folder = os.path.join(
            self.json_file_dir, f'{self.yolo_dataset_name}/labels/')
        self.image_folder = os.path.join(
            self.json_file_dir, f'{self.yolo_dataset_name}/images/')

        # Define YOLO paths for train, validation, and test directories for both images and labels
        yolo_paths = [
            os.path.join(self.label_folder, 'train'),
            os.path.join(self.label_folder, 'val'),
            os.path.join(self.label_folder, 'test'),
            os.path.join(self.image_folder, 'train'),
            os.path.join(self.image_folder, 'val'),
            os.path.join(self.image_folder, 'test')
        ]

        # Delete existing directories and create new ones
        for yolo_path in yolo_paths:
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            os.makedirs(yolo_path)

    def split_jsons(self, folders, json_names, val_size, test_size):
        """Splits json files into training, validation, and test sets.

        Args:
            folders (list): List of subdirectories for train, val, and test data.
            json_names (list): List of json file names.
            val_size (float): Fraction of input data to be used for validation.
            test_size (float): Fraction of input data to be used for testing.

        Returns:
            Tuple of lists: List of training, validation, and test json file names.

        Raises:
            ValueError: When the folders are specified, but one or more of train, val,
              or test data directories are missing.
        """
        if len(folders) > 0 and 'train' in folders and 'val' in folders and 'test' in folders:
            # If the directories are specified, get the file names from them.
            train_folder = os.path.join(self.json_file_dir, 'train/')
            train_jsons = [train_name + '.json'
                           for train_name in os.listdir(train_folder)
                           if os.path.isdir(os.path.join(train_folder, train_name))]

            val_folder = os.path.join(self.json_file_dir, 'val/')
            val_jsons = [val_name + '.json'
                         for val_name in os.listdir(val_folder)
                         if os.path.isdir(os.path.join(val_folder, val_name))]

            test_folder = os.path.join(self.json_file_dir, 'test/')
            test_jsons = [test_name + '.json'
                          for test_name in os.listdir(test_folder)
                          if os.path.isdir(os.path.join(test_folder, test_name))]

            return train_jsons, val_jsons, test_jsons

        # Randomly split the input data into train, validation, and test sets.
        if train_test_split is not None:
            train_idxs, val_idxs = train_test_split(range(len(json_names)),
                                                    test_size=val_size)
            tmp_train_len = len(train_idxs)
            test_idxs = []
            if test_size is None:
                test_size = 0.0
            if test_size > 1e-8 and tmp_train_len:
                train_subset_indices = list(range(tmp_train_len))
                train_idxs_sub, test_subset = train_test_split(
                    train_subset_indices, test_size=test_size / max(1 - val_size, 1e-8))
                test_idxs = [train_idxs[idx] for idx in test_subset]
                train_idxs = [train_idxs[idx] for idx in train_idxs_sub]
        else:
            total = len(json_names)
            indices = list(range(total))
            rng = Random(0)
            rng.shuffle(indices)
            if val_size is None:
                val_size = 0.0
            if test_size is None:
                test_size = 0.0
            val_count = int(round(total * val_size))
            val_count = min(max(val_count, 0), total)
            remaining = total - val_count
            adjusted_test_fraction = 0.0
            if remaining > 0 and (1 - val_size) > 1e-8:
                adjusted_test_fraction = max(
                    0.0, min(1.0, test_size / (1 - val_size)))
            test_count = int(round(remaining * adjusted_test_fraction))
            test_count = min(max(test_count, 0), remaining)
            val_idxs = indices[:val_count]
            test_idxs = indices[val_count: val_count + test_count]
            train_idxs = indices[val_count + test_count:]

        train_jsons = [json_names[train_idx]
                       for train_idx in train_idxs] if train_idxs else []
        val_jsons = [json_names[val_idx]
                     for val_idx in val_idxs] if val_idxs else []
        test_jsons = [json_names[test_idx]
                      for test_idx in test_idxs] if test_idxs else []

        return train_jsons, val_jsons, test_jsons

    @staticmethod
    def _scan_labels_and_keypoints(json_dir: str) -> Tuple[List[str], List[str]]:
        """Scan annotation directory to collect class and keypoint label order."""
        label_order: List[str] = []
        seen_labels: Set[str] = set()
        keypoint_order: List[str] = []
        seen_keypoints: Set[str] = set()

        if not os.path.isdir(json_dir):
            return label_order, keypoint_order

        for file_name in sorted(os.listdir(json_dir)):
            if not file_name.endswith(".json"):
                continue
            json_path = os.path.join(json_dir, file_name)
            try:
                data = load_labelme_json(json_path)
            except Exception:
                continue
            shapes = data.get("shapes") or []
            polygon_labels: Set[str] = set()
            default_label = Labelme2YOLO._default_instance_label(
                Path(json_path), data)

            for shape in shapes:
                shape_type = (shape.get("shape_type") or "polygon").lower()
                if shape_type == "point":
                    continue
                instance_label = Labelme2YOLO._resolve_instance_label(
                    shape, polygon_labels, default_label=default_label)
                if instance_label:
                    polygon_labels.add(instance_label)
                class_label = Labelme2YOLO._clean_label(
                    shape.get("label")) or instance_label
                if class_label and class_label not in seen_labels:
                    seen_labels.add(class_label)
                    label_order.append(class_label)

            candidate_labels = set(polygon_labels)
            if not candidate_labels and default_label:
                candidate_labels.add(default_label)
            if not candidate_labels and label_order:
                candidate_labels.update(label_order)
            for shape in shapes:
                shape_type = (shape.get("shape_type") or "polygon").lower()
                if shape_type != "point":
                    continue
                instance_label = Labelme2YOLO._resolve_instance_label(
                    shape, candidate_labels, default_label=default_label)
                if instance_label:
                    candidate_labels.add(instance_label)
                if instance_label and instance_label not in seen_labels:
                    seen_labels.add(instance_label)
                    label_order.append(instance_label)
                keypoint_label = Labelme2YOLO._resolve_keypoint_label(
                    shape, instance_label or "")
                if not keypoint_label:
                    keypoint_label = f"kp_{len(keypoint_order)}"
                if keypoint_label and keypoint_label not in seen_keypoints:
                    seen_keypoints.add(keypoint_label)
                    keypoint_order.append(keypoint_label)

            if not polygon_labels and default_label and default_label not in seen_labels:
                seen_labels.add(default_label)
                label_order.append(default_label)

        return label_order, keypoint_order

    @staticmethod
    def map_label_to_id(json_dir: str) -> OrderedDict:
        """
        Build a stable mapping of class labels to integer IDs.

        Parameters:
            json_dir (str): Directory containing Labelme JSON annotations.

        Returns:
            OrderedDict: Maps class label strings to zero-based IDs.
        """
        label_order, _ = Labelme2YOLO._scan_labels_and_keypoints(json_dir)
        return OrderedDict(
            (label, label_id) for label_id, label in enumerate(label_order)
        )

    @staticmethod
    def _clean_label(value: Optional[str]) -> str:
        """Normalize a label value to a trimmed string."""
        return str(value).strip() if value not in (None, "") else ""

    @staticmethod
    def _default_instance_label(json_path: Path,
                                payload: Dict[str, object]) -> str:
        """Determine a fallback instance label based on JSON metadata."""
        flags = payload.get("flags") if isinstance(
            payload.get("flags"), dict) else {}
        flag_label = Labelme2YOLO._clean_label(
            flags.get("instance_label") if flags else None
        )
        if flag_label:
            return flag_label

        stem = json_path.stem
        if "_" in stem:
            prefix = Labelme2YOLO._clean_label(stem.split("_", 1)[0])
            if prefix:
                return prefix

        parent = json_path.parent.name
        parent_label = Labelme2YOLO._clean_label(parent)
        if parent_label:
            return parent_label

        return "object"

    @staticmethod
    def _resolve_instance_label(shape: Dict[str, object],
                                candidate_labels: Optional[Set[str]] = None,
                                default_label: Optional[str] = None) -> str:
        """Infer the instance label for a shape using flags and heuristics."""
        flags = shape.get("flags") or {}
        flag_label = Labelme2YOLO._clean_label(
            flags.get("instance_label") if isinstance(flags, dict) else None
        )
        if flag_label:
            return flag_label

        label = Labelme2YOLO._clean_label(shape.get("label"))
        shape_type = (shape.get("shape_type") or "polygon").lower()

        if shape_type == "point":
            candidates = candidate_labels or set()
            lower_label = label.lower()
            for candidate in sorted(candidates, key=len, reverse=True):
                if lower_label.startswith(candidate.lower()):
                    return candidate
            for delimiter in ("_", "-", ":", "|", " "):
                if delimiter in label:
                    return label.split(delimiter, 1)[0]
            if default_label:
                return default_label
            return label
        return label or (default_label or "")

    @staticmethod
    def _resolve_keypoint_label(shape: Dict[str, object],
                                instance_label: str) -> str:
        """Determine the keypoint label, removing the instance prefix when possible."""
        flags = shape.get("flags") or {}
        if isinstance(flags, dict):
            display_label = Labelme2YOLO._clean_label(
                flags.get("display_label"))
            if display_label:
                return display_label

        label = Labelme2YOLO._clean_label(shape.get("label"))
        if instance_label:
            inst_len = len(instance_label)
            if label.lower().startswith(instance_label.lower()) and inst_len < len(label):
                suffix = label[inst_len:]
                suffix = suffix.lstrip("_-:| ")
                if suffix:
                    return suffix
        for delimiter in ("_", "-", ":", "|"):
            if delimiter in label:
                suffix = label.split(delimiter, 1)[1].strip()
                if suffix:
                    return suffix
        return label

    @staticmethod
    def _derive_visibility(shape: Dict[str, object]) -> Optional[int]:
        """Parse visibility from the shape description if available."""
        description = shape.get("description")
        if isinstance(description, (int, float)):
            return int(description)
        if isinstance(description, str):
            try:
                return int(float(description))
            except ValueError:
                return None
        return None

    @staticmethod
    def _extend_bounds(bounds: Optional[Tuple[float, float, float, float]],
                       x: float,
                       y: float) -> Tuple[float, float, float, float]:
        if bounds is None:
            return float(x), float(y), float(x), float(y)
        min_x, min_y, max_x, max_y = bounds
        return (
            min(min_x, float(x)),
            min(min_y, float(y)),
            max(max_x, float(x)),
            max(max_y, float(y)),
        )

    @staticmethod
    def _bounds_to_cxcywh(bounds: Optional[Tuple[float, float, float, float]],
                          image_width: int,
                          image_height: int) -> List[float]:
        if bounds is None or image_width == 0 or image_height == 0:
            return [0.0, 0.0, 0.0, 0.0]
        min_x, min_y, max_x, max_y = bounds
        width = max(max_x - min_x, 0.0)
        height = max(max_y - min_y, 0.0)
        cx = min_x + width / 2.0
        cy = min_y + height / 2.0
        return [
            cx / image_width,
            cy / image_height,
            width / image_width,
            height / image_height,
        ]

    def _collect_pose_instances(self,
                                shapes: List[Dict[str, object]],
                                default_instance_label: Optional[str] = None) -> Dict[str, Dict[str, object]]:
        """Group shapes by instance to prepare pose annotations."""
        instances: Dict[str, Dict[str, object]] = {}
        polygon_labels: Set[str] = set()

        for shape in shapes:
            shape_type = (shape.get("shape_type") or "polygon").lower()
            if shape_type == "point":
                continue
            instance_label = self._resolve_instance_label(
                shape, polygon_labels, default_label=default_instance_label)
            if not instance_label:
                continue
            polygon_labels.add(instance_label)
            entry = instances.setdefault(
                instance_label,
                {
                    "class_label": self._clean_label(shape.get("label")) or instance_label,
                    "bounds": None,
                    "keypoints": {},
                },
            )
            class_label = self._clean_label(
                shape.get("label")) or entry["class_label"]
            if class_label:
                entry["class_label"] = class_label
            for point in shape.get("points") or []:
                if len(point) < 2:
                    continue
                entry["bounds"] = self._extend_bounds(
                    entry["bounds"], point[0], point[1]
                )

        candidate_labels = polygon_labels or set(instances.keys())
        for shape in shapes:
            shape_type = (shape.get("shape_type") or "polygon").lower()
            if shape_type != "point":
                continue
            instance_label = self._resolve_instance_label(
                shape, candidate_labels, default_label=default_instance_label)
            if not instance_label:
                continue
            entry = instances.setdefault(
                instance_label,
                {
                    "class_label": instance_label,
                    "bounds": None,
                    "keypoints": {},
                },
            )
            points = shape.get("points") or []
            if not points:
                continue
            x, y = points[0][:2]
            entry["bounds"] = self._extend_bounds(entry["bounds"], x, y)
            keypoint_label = self._resolve_keypoint_label(
                shape, instance_label)
            if not keypoint_label:
                keypoint_label = f"kp_{len(entry['keypoints'])}"
            visibility = self._derive_visibility(shape)
            entry["keypoints"][keypoint_label] = {
                "x": float(x),
                "y": float(y),
                "visible": bool(shape.get("visible", True)),
                "visibility": visibility,
            }

        if not polygon_labels and len(instances) > 1:
            target_label = self._clean_label(default_instance_label) or next(
                iter(instances))
            merged = {
                "class_label": target_label,
                "bounds": None,
                "keypoints": {},
            }
            for data in instances.values():
                merged["class_label"] = self._clean_label(
                    data.get("class_label")) or merged["class_label"]
                bounds = data.get("bounds")
                if bounds:
                    min_x, min_y, max_x, max_y = bounds
                    merged["bounds"] = self._extend_bounds(
                        merged["bounds"], min_x, min_y)
                    merged["bounds"] = self._extend_bounds(
                        merged["bounds"], max_x, max_y)
                for label, kp in data["keypoints"].items():
                    merged["keypoints"][label] = kp
            if merged["bounds"] is None:
                merged["bounds"] = (0.0, 0.0, 0.0, 0.0)
            instances = {target_label: merged}

        return {
            label: data for label, data in instances.items() if data["keypoints"]
        }

    def _update_keypoint_order(self, labels: Iterable[str]) -> None:
        added = False
        for label in labels:
            clean_label = self._clean_label(label)
            if clean_label and clean_label not in self.keypoint_labels_order:
                self.keypoint_labels_order.append(clean_label)
                added = True
        if self.keypoint_labels_order:
            dims = 3 if self.include_visibility else 2
            self.kpt_shape = [len(self.keypoint_labels_order), dims]
        if added:
            # Ensure annotation type is updated for downstream checks
            self.annotation_type = "pose"

    def convert(self, val_size, test_size):
        """
        Converts a set of JSON files in Labelme format to YOLO format. Splits the dataset
        into train, validation and test sets, and saves the resulting files in the appropriate
        directories.

        Args:
            val_size (float): The percentage of data to set aside for the validation set.
            test_size (float): The percentage of data to set aside for the test set.
        """
        # Get a list of JSON file names from the input directory
        json_names = [file_name for file_name in os.listdir(self.json_file_dir)
                      if os.path.isfile(os.path.join(self.json_file_dir, file_name)) and
                      file_name.endswith('.json')]
        if not json_names:
            folder = Path(self.json_file_dir)
            store = AnnotationStore.for_frame_path(
                folder / f"{folder.name}_000000000.json")
            if store.store_path.exists():
                json_names = [
                    f"{folder.name}_{frame:09d}.json" for frame in sorted(store.iter_frames())
                ]
        # filter and only keep the JSON files with a image associated with it
        json_names = [json_name for json_name in json_names if
                      os.path.exists(os.path.join(self.json_file_dir, json_name.replace('.json', '.png')))]

        # Get a list of folder names from the input directory
        folders = [file_name for file_name in os.listdir(self.json_file_dir)
                   if os.path.isdir(os.path.join(self.json_file_dir, file_name))]

        # Split the JSON files into train, validation and test sets
        train_jsons, val_jsons, test_jsons = self.split_jsons(
            folders, json_names, val_size, test_size)

        # Create the train and validation directories if they don't exist already
        self.create_yolo_dataset_dirs()

        # Convert labelme object to yolo format object, and save them to files
        # Also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/', 'test/'),
                                          (train_jsons, val_jsons, test_jsons)):

            for json_name in json_names:
                self.json_to_text(target_dir, json_name)

        # Save the dataset configuration file
        self.save_data_yaml()

    def get_yolo_objects(self, json_name, json_data, img_path):
        """Return a list of YOLO formatted objects from a JSON annotation file and image."""
        image_height = json_data['imageHeight']
        image_width = json_data['imageWidth']
        shapes = json_data.get("shapes") or []

        json_path = Path(self.json_file_dir) / json_name
        default_label = self._default_instance_label(json_path, json_data)

        pose_instances = self._collect_pose_instances(
            shapes, default_instance_label=default_label)
        if pose_instances:
            self.annotation_type = "pose"
            for data in pose_instances.values():
                self._update_keypoint_order(data["keypoints"].keys())

            yolo_objects = []
            for instance_label, data in pose_instances.items():
                class_label = data.get("class_label") or instance_label
                if class_label and class_label not in self.label_to_id_dict:
                    self.label_to_id_dict[class_label] = len(
                        self.label_to_id_dict)
                label_id = self.label_to_id_dict.get(
                    class_label or default_label, 0)
                bbox = self._bounds_to_cxcywh(
                    data.get("bounds"), image_width, image_height)
                keypoint_values: List[float] = []
                for kp_label in self.keypoint_labels_order:
                    kp = data["keypoints"].get(kp_label)
                    if kp:
                        x = kp["x"] / image_width if image_width else 0.0
                        y = kp["y"] / image_height if image_height else 0.0
                        if self.include_visibility:
                            visibility = kp.get("visibility")
                            if visibility is None:
                                visibility = 2 if kp.get(
                                    "visible", True) else 1
                            keypoint_values.extend(
                                [x, y, int(visibility)]
                            )
                        else:
                            keypoint_values.extend([x, y])
                    else:
                        if self.include_visibility:
                            keypoint_values.extend([0.0, 0.0, 0])
                        else:
                            keypoint_values.extend([0.0, 0.0])
                yolo_objects.append((label_id, bbox + keypoint_values))
            return yolo_objects

        yolo_objects = []
        for shape in shapes:
            shape_type = (shape.get("shape_type") or "polygon").lower()
            if shape_type == 'circle':
                yolo_obj = self.circle_shape_to_yolo(
                    shape, image_height, image_width)
                yolo_objects.append(yolo_obj)
            elif shape_type == 'point':
                continue
            else:
                yolo_obj = self.scale_points(
                    shape, image_height, image_width)
                yolo_objects.append(yolo_obj)
        return yolo_objects

    def json_to_text(self, target_dir, json_name):
        """
        Converts a single JSON file to YOLO label format text file.

        Args:
            target_dir (str): The directory to save the output files to.
            json_name (str): The name of the JSON file to convert.

        Returns:
            None
        """
        # Get the path to the input JSON file and load its data.
        json_path = os.path.join(self.json_file_dir, json_name)
        json_data = load_labelme_json(json_path)

        # Save the image file in the YOLO format.
        img_path = self.save_or_copy_image(
            json_data, json_name, self.image_folder, target_dir)

        # Get a list of YOLO objects from the JSON data and save the output text file.
        yolo_objects = self.get_yolo_objects(json_name, json_data, img_path)
        self.save_yolo_txt_label_file(json_name, self.label_folder,
                                      target_dir, yolo_objects)

    def scale_points(self,
                     labelme_shape: dict,
                     image_height: int,
                     image_width: int,
                     output_fromat: str = 'polygon',
                     include_visibility: bool = None,
                     ):
        """
        Returns the label_id and scaled points of the given shape object in YOLO format.
        """
        # Use the class default if not explicitly provided
        if include_visibility is None:
            include_visibility = self.include_visibility
        cx, cy, w, h = find_bbox_from_shape(labelme_shape)
        scaled_cxcywh = [cx/image_width,
                         cy/image_height,
                         w/image_width,
                         h/image_height]
        # Extract the points from the shape object
        point_list = labelme_shape['points']
        scaled_points = []
        for point in point_list:
            x = float(point[0]) / image_width
            y = float(point[1]) / image_height
            if include_visibility:
                visibility = labelme_shape.get("description", 1)
                scaled_points.extend([x, y, visibility])
            else:
                scaled_points.extend([x, y])

        # Create an array of zeros with length 2 * len(point_list)
        points = np.zeros(2 * len(point_list))
        # Fill the array with the x and y coordinates of each point in the shape, scaled between 0 and 1
        points[::2] = [float(point[0]) / image_width for point in point_list]
        points[1::2] = [float(point[1]) / image_height for point in point_list]
        if len(points) == 4:
            points = point_list_to_numpy_array(points)
        # # Close the polygon by appending the first point to the end
        # points = np.append(points, [points[0], points[1]])
        # Map the label of the shape to a label_id
        try:
            label_id = self.label_to_id_dict[labelme_shape['label']]
        except KeyError:
            label_id = 0
        # Return the label_id and points as a list
        if output_fromat == 'bbox':
            return label_id, scaled_cxcywh
        elif output_fromat == 'pose':
            return label_id, scaled_cxcywh + scaled_points
        else:
            return label_id,  points.tolist()

    def circle_shape_to_yolo(self, labelme_shape, image_height, image_width):
        """
        Returns a YOLO object for a circle shape.

        Args:
            labelme_shape: A dictionary representing the circle shape.
            image_height: An integer representing the height of the image.
            image_width: An integer representing the width of the image.

        Returns:
            A tuple representing the YOLO object for the circle shape.

        """
        # Calculate the center of the circle.
        cx, cy = labelme_shape['points'][0]

        # Calculate the radius of the circle.
        radius = math.sqrt((cx - labelme_shape['points'][1][0]) ** 2 +
                           (cy - labelme_shape['points'][1][1]) ** 2)

        # Calculate the width and height of the circle.
        w = 2 * radius
        h = 2 * radius

        # Calculate the YOLO coordinates.
        yolo_cx = round(float(cx / image_width), 6)
        yolo_cy = round(float(cy / image_height), 6)
        yolo_w = round(float(w / image_width), 6)
        yolo_h = round(float(h / image_height), 6)

        # Get the label ID.
        label_id = self.label_to_id_dict[labelme_shape['label']]

        # Return the YOLO object as a tuple.
        return label_id, yolo_cx, yolo_cy, yolo_w, yolo_h

    @staticmethod
    def save_or_copy_image(json_data: dict,
                           json_name: str,
                           image_dir_path: str,
                           target_dir: str) -> str:
        """
        Save an image in YOLO format.

        :param json_data: Dictionary containing the data from the json file.
        :param json_name: Name of the json file.
        :param image_dir_path: Path to the directory containing the image data.
        :param target_dir: Target directory to save the image in.
        :return: Path of the saved image.
        """
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(image_dir_path, target_dir, img_name)

        # if the image is not already saved, then save it
        if not os.path.exists(img_path):
            image_data = json_data.get('imageData')
            if image_data:
                img = img_b64_to_arr(image_data)
                PIL.Image.fromarray(img).save(img_path)
            else:
                src_img_path = json_data.get('imagePath') or ""
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, img_path)
        return img_path

    def save_data_yaml(self):
        """Save the dataset information as a YAML file in the new format."""
        # Set the path for the YAML file
        yaml_path = os.path.join(
            self.json_file_dir, f'{self.yolo_dataset_name}/', 'data.yaml')

        # Construct the names section
        names_section = "names:\n"
        for label, label_id in self.label_to_id_dict.items():
            names_section += f"  {label_id}: {label}\n"

        # Write the YAML file content
        with open(yaml_path, 'w+') as yaml_file:
            # Relative path to the dataset
            yaml_file.write(f"path: ../{self.yolo_dataset_name}\n")
            yaml_file.write(f"train: images/train\n")
            yaml_file.write(f"val: images/val\n")
            # Include test set in the YAML
            yaml_file.write(f"test: images/test\n")
            yaml_file.write("\n")  # Add an empty line for better readability
            if self.annotation_type == "pose" and self.kpt_shape:
                # Keypoints
                # kpt_shape: [17, 2] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
                dims = 3 if self.include_visibility else 2
                yaml_file.write(f"kpt_shape: [{self.kpt_shape[0]}, {dims}]\n")
                if self.keypoint_labels_order:
                    # Annolid internal mapping (kept for backwards compatibility)
                    yaml_file.write("kpt_labels:\n")
                    for idx, name in enumerate(self.keypoint_labels_order):
                        yaml_file.write(f"  {idx}: {name}\n")

                    # Ultralytics canonical mapping (per-class)
                    yaml_file.write("\n")
                    yaml_file.write("kpt_names:\n")
                    for class_id in sorted(self.label_to_id_dict.values()):
                        yaml_file.write(f"  {class_id}:\n")
                        for name in self.keypoint_labels_order:
                            yaml_file.write(f"    - {name}\n")

                flip_idx = None
                if self.pose_schema:
                    flip_idx = self.pose_schema.compute_flip_idx(
                        self.keypoint_labels_order)
                if flip_idx:
                    yaml_file.write("\n")
                    yaml_file.write(f"flip_idx: {flip_idx}\n")
                else:
                    yaml_file.write(
                        "#(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face. For example if we assume five keypoints of facial landmark: [left eye, right eye, nose, left mouth, right mouth], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3] (just exchange the left-right index, i.e. 0-1 and 3-4, and do not modify others like nose in this example.)\n")
                    yaml_file.write(
                        "#flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n")

            yaml_file.write(names_section)

    @staticmethod
    def save_yolo_txt_label_file(json_name: str, label_folder_path: str,
                                 target_dir: str,
                                 yolo_objects: List[Tuple[str, List[float]]]) -> None:
        """Saves a list of YOLO objects as a text file in the specified directory.

        Args:
            json_name: The name of the JSON file.
            label_folder_path: The path of the directory where the label file will be saved.
            target_dir: The name of the target directory (e.g. 'train', 'val', 'test').
            yolo_objects: A list of YOLO objects, where each object is a tuple containing the label
                and the normalized coordinates of the bounding box (in the format [x_center, y_center, width, height]).

        Returns:
            None
        """
        txt_path = os.path.join(label_folder_path, target_dir,
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            # Write each YOLO object as a line in the label file
            for label, points in yolo_objects:
                points = [str(item) for item in points]
                yolo_object_line = f"{label} {' '.join(points)}\n"
                f.write(yolo_object_line)
