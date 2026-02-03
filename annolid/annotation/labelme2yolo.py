# How to use it?
# python annolid/main.py --labelme2yolo /path/to/labelme_json_folder/ --val_size 0.1 --test_size 0.1
# Refer to https://docs.ultralytics.com/datasets/pose/#dataset-yaml-format for more details.
import hashlib
import math
import os
import re
from pathlib import Path
import numpy as np
import PIL.Image
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from random import Random
from collections import OrderedDict, defaultdict
from labelme.utils.image import img_b64_to_arr

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency
    train_test_split = None
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.annotation.pose_schema import PoseSchema
from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.core.behavior.spec import DEFAULT_SCHEMA_FILENAME, load_behavior_spec

_SPLIT_NAME_PATTERN = re.compile(
    r"^(train(?:ing)?|val|valid(?:ation)?|test)(?:[_-].+)?$",
    re.IGNORECASE,
)


def _normalize_split_name(name: str) -> Optional[str]:
    token = str(name or "").strip().lower()
    match = _SPLIT_NAME_PATTERN.match(token)
    if not match:
        return None
    base = match.group(1).lower()
    if base.startswith("train"):
        return "train"
    if base.startswith("val") or base.startswith("valid"):
        return "val"
    if base == "test":
        return "test"
    return None


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

    def __init__(
        self,
        json_dir,
        yolo_dataset_name="YOLO_dataset",
        include_visibility=False,
        pose_schema_path: Optional[str] = None,
        recursive: bool = True,
        **_ignored_kwargs,
    ):
        self.json_file_dir = json_dir
        self.recursive = bool(recursive)
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
                self.pose_schema.normalize_prefixed_keypoints()
            except Exception:
                self.pose_schema = None

        if self.pose_schema is None:
            try:
                project_path = (
                    Path(self.json_file_dir).expanduser() / DEFAULT_SCHEMA_FILENAME
                )
                if project_path.exists():
                    project_schema, _ = load_behavior_spec(path=project_path)
                    embedded = getattr(project_schema, "pose_schema", None)
                    if isinstance(embedded, dict) and embedded:
                        self.pose_schema = PoseSchema.from_dict(embedded)
                        self.pose_schema.normalize_prefixed_keypoints()
            except Exception:
                self.pose_schema = None

        if self.pose_schema and getattr(self.pose_schema, "instances", None):
            # When LabelMe point labels are prefixed with instance names
            # (e.g. "intruder_nose"), keep the canonical YOLO keypoint list as the
            # base names ("nose") and rely on per-object grouping for instances.
            normalized_keypoints: List[str] = []
            seen: Set[str] = set()
            for kp in keypoints:
                try:
                    _, base = self.pose_schema.strip_instance_prefix(kp)
                except Exception:
                    base = str(kp or "").strip()
                base = self._clean_label(base)
                if base and base not in seen:
                    seen.add(base)
                    normalized_keypoints.append(base)
            keypoints = normalized_keypoints

        if self.pose_schema and self.pose_schema.keypoints:
            # YOLO pose datasets use a single canonical keypoint list shared across all classes.
            # Even when the pose schema represents multiple "instances" (e.g. multi-animal),
            # keypoints remain the base names (nose/left_ear/...) and instance assignment is
            # handled per-object via grouping.
            schema_keypoints = list(self.pose_schema.keypoints)
            merged = list(schema_keypoints)
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
            self.json_file_dir, f"{self.yolo_dataset_name}/labels/"
        )
        self.image_folder = os.path.join(
            self.json_file_dir, f"{self.yolo_dataset_name}/images/"
        )

        # Define YOLO paths for train, validation, and test directories for both images and labels
        yolo_paths = [
            os.path.join(self.label_folder, "train"),
            os.path.join(self.label_folder, "val"),
            os.path.join(self.label_folder, "test"),
            os.path.join(self.image_folder, "train"),
            os.path.join(self.image_folder, "val"),
            os.path.join(self.image_folder, "test"),
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
        if (
            len(folders) > 0
            and "train" in folders
            and "val" in folders
            and "test" in folders
        ):
            # If the directories are specified, get the file names from them.
            train_folder = os.path.join(self.json_file_dir, "train/")
            train_jsons = [
                train_name + ".json"
                for train_name in os.listdir(train_folder)
                if os.path.isdir(os.path.join(train_folder, train_name))
            ]

            val_folder = os.path.join(self.json_file_dir, "val/")
            val_jsons = [
                val_name + ".json"
                for val_name in os.listdir(val_folder)
                if os.path.isdir(os.path.join(val_folder, val_name))
            ]

            test_folder = os.path.join(self.json_file_dir, "test/")
            test_jsons = [
                test_name + ".json"
                for test_name in os.listdir(test_folder)
                if os.path.isdir(os.path.join(test_folder, test_name))
            ]

            return train_jsons, val_jsons, test_jsons

        if not json_names:
            return [], [], []
        if len(json_names) == 1:
            return [json_names[0]], [], []
        if val_size is None:
            val_size = 0.0
        if test_size is None:
            test_size = 0.0
        if float(val_size) <= 1e-8 and float(test_size) <= 1e-8:
            return list(json_names), [], []

        # Randomly split the input data into train, validation, and test sets.
        if train_test_split is not None:
            try:
                train_idxs, val_idxs = train_test_split(
                    range(len(json_names)),
                    test_size=val_size,
                    random_state=0,
                    shuffle=True,
                )
                tmp_train_len = len(train_idxs)
                test_idxs = []
                if test_size > 1e-8 and tmp_train_len:
                    train_subset_indices = list(range(tmp_train_len))
                    train_idxs_sub, test_subset = train_test_split(
                        train_subset_indices,
                        test_size=test_size / max(1 - val_size, 1e-8),
                        random_state=0,
                        shuffle=True,
                    )
                    test_idxs = [train_idxs[idx] for idx in test_subset]
                    train_idxs = [train_idxs[idx] for idx in train_idxs_sub]
            except ValueError:
                # sklearn can fail for small n (e.g. n_samples=1 with non-zero fractions).
                train_idxs, val_idxs, test_idxs = [], [], []
        else:
            total = len(json_names)
            indices = list(range(total))
            rng = Random(0)
            rng.shuffle(indices)
            val_count = int(round(total * val_size))
            val_count = min(max(val_count, 0), total)
            remaining = total - val_count
            adjusted_test_fraction = 0.0
            if remaining > 0 and (1 - val_size) > 1e-8:
                adjusted_test_fraction = max(0.0, min(1.0, test_size / (1 - val_size)))
            test_count = int(round(remaining * adjusted_test_fraction))
            test_count = min(max(test_count, 0), remaining)
            val_idxs = indices[:val_count]
            test_idxs = indices[val_count : val_count + test_count]
            train_idxs = indices[val_count + test_count :]

        # Ensure we always have at least one training sample when possible.
        if not train_idxs:
            train_idxs = []
        if not val_idxs:
            val_idxs = []
        if not test_idxs:
            test_idxs = []
        if not train_idxs:
            if val_idxs:
                train_idxs.append(val_idxs.pop())
            elif test_idxs:
                train_idxs.append(test_idxs.pop())
            else:
                train_idxs = [0]

        train_jsons = (
            [json_names[train_idx] for train_idx in train_idxs] if train_idxs else []
        )
        val_jsons = [json_names[val_idx] for val_idx in val_idxs] if val_idxs else []
        test_jsons = (
            [json_names[test_idx] for test_idx in test_idxs] if test_idxs else []
        )

        return train_jsons, val_jsons, test_jsons

    @dataclass(frozen=True)
    class _LabelmeItem:
        json_path: Path
        image_path: Optional[Path]
        output_stem: str
        preset_split: Optional[str] = None  # train/val/test or None

    @staticmethod
    def _safe_output_stem(relative_path: Path) -> str:
        raw = relative_path.as_posix()
        if raw.lower().endswith(".json"):
            raw = raw[:-5]
        raw = raw.strip().strip("/")
        if not raw:
            raw = "item"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.replace("/", "__"))
        if len(safe) <= 200:
            return safe
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"{safe[:180]}_{digest}"

    @staticmethod
    def _resolve_image_path(
        json_data: Dict[str, object], *, json_path: Path
    ) -> Optional[Path]:
        from annolid.datasets.labelme_collection import resolve_image_path

        return resolve_image_path(json_path)

    def _iter_labelme_json_paths(self) -> Iterable[Path]:
        root = Path(self.json_file_dir).expanduser()
        if not root.exists():
            return []
        ignore_prefixes = {
            self.yolo_dataset_name,
            "YOLO_dataset",
            "YOLO_pose_vis",
            "annolid_logs",
        }

        def should_ignore(path: Path) -> bool:
            try:
                rel_parts = set(path.relative_to(root).parts)
            except Exception:
                rel_parts = set(path.parts)
            for name in ignore_prefixes:
                if name and name in rel_parts:
                    return True
            return False

        paths = root.rglob("*.json") if self.recursive else root.glob("*.json")
        return (p for p in paths if p.is_file() and not should_ignore(p))

    def _discover_items(self, *, require_image: bool = True) -> List["_LabelmeItem"]:
        root = Path(self.json_file_dir).expanduser().resolve()
        items: List[Labelme2YOLO._LabelmeItem] = []
        for json_path in self._iter_labelme_json_paths():
            try:
                data = load_labelme_json(json_path)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if "shapes" not in data:
                continue
            rel = None
            try:
                rel = json_path.resolve().relative_to(root)
            except Exception:
                rel = Path(json_path.name)
            output_stem = self._safe_output_stem(rel)
            image_path = self._resolve_image_path(data, json_path=json_path)
            if require_image and not image_path and not data.get("imageData"):
                continue

            preset_split = None
            for part in reversed(rel.parts[:-1]):
                split_name = _normalize_split_name(part)
                if split_name:
                    preset_split = split_name
                    break

            items.append(
                Labelme2YOLO._LabelmeItem(
                    json_path=json_path,
                    image_path=image_path,
                    output_stem=output_stem,
                    preset_split=preset_split,
                )
            )
        return items

    @staticmethod
    def _scan_labels_and_keypoints(json_dir: str) -> Tuple[List[str], List[str]]:
        """Scan annotation directory to collect class and keypoint label order."""
        label_order: List[str] = []
        seen_labels: Set[str] = set()
        keypoint_order: List[str] = []
        seen_keypoints: Set[str] = set()

        if not os.path.isdir(json_dir):
            return label_order, keypoint_order

        root = Path(json_dir).expanduser()
        candidates = sorted(root.rglob("*.json"))
        # Avoid scanning YOLO outputs (which can contain JSON metadata) and annolid logs
        # within the provided root directory (but allow roots that live inside those dirs,
        # e.g. staging folders under annolid_logs).

        def should_ignore(path: Path) -> bool:
            try:
                parts = set(path.relative_to(root).parts)
            except Exception:
                parts = set(path.parts)
            if "annolid_logs" in parts:
                return True
            for name in ("YOLO_dataset", "YOLO_pose_vis"):
                if name in parts:
                    return True
            if any(p.startswith("YOLO_") for p in parts):
                return True
            return False

        for json_path in candidates:
            if not json_path.is_file() or should_ignore(json_path):
                continue
            try:
                data = load_labelme_json(str(json_path))
            except Exception:
                continue
            if not isinstance(data, dict) or "shapes" not in data:
                continue
            shapes = data.get("shapes") or []
            polygon_labels: Set[str] = set()
            default_label = Labelme2YOLO._default_instance_label(Path(json_path), data)
            saw_non_point = False

            for shape in shapes:
                shape_type = (shape.get("shape_type") or "polygon").lower()
                if shape_type == "point":
                    continue
                saw_non_point = True
                instance_label = Labelme2YOLO._resolve_instance_label(
                    shape, polygon_labels, default_label=default_label
                )
                if instance_label:
                    polygon_labels.add(instance_label)
                class_label = (
                    Labelme2YOLO._clean_label(shape.get("label")) or instance_label
                )
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
                    shape, candidate_labels, default_label=default_label
                )
                if instance_label:
                    candidate_labels.add(instance_label)
                # Only treat point-derived instance labels as classes when there are no
                # polygon/box shapes. For multi-animal pose with polygons, points should
                # not introduce new class labels.
                if (
                    (not saw_non_point)
                    and instance_label
                    and instance_label not in seen_labels
                ):
                    seen_labels.add(instance_label)
                    label_order.append(instance_label)
                keypoint_label = Labelme2YOLO._resolve_keypoint_label(
                    shape, instance_label or ""
                )
                if not keypoint_label:
                    keypoint_label = f"kp_{len(keypoint_order)}"
                if keypoint_label and keypoint_label not in seen_keypoints:
                    seen_keypoints.add(keypoint_label)
                    keypoint_order.append(keypoint_label)

            if (
                not polygon_labels
                and default_label
                and default_label not in seen_labels
            ):
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
        """Normalize a label value to a trimmed string.

        LabelMe projects sometimes end up with accidental trailing separators in
        labels (e.g. `intruder_`). Strip common separators from the end while
        preserving internal underscores used by keypoint names (e.g. `tail_base`).
        """
        if value in (None, ""):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.rstrip("_-:|")

    @staticmethod
    def _default_instance_label(json_path: Path, payload: Dict[str, object]) -> str:
        """Determine a fallback instance label based on JSON metadata."""
        flags = payload.get("flags") if isinstance(payload.get("flags"), dict) else {}
        flag_label = Labelme2YOLO._clean_label(
            flags.get("instance_label") if flags else None
        )
        if flag_label:
            return flag_label

        meta_flags = payload.get("annolid_flags_meta")
        if isinstance(meta_flags, dict):
            meta_label = Labelme2YOLO._clean_label(meta_flags.get("instance_label"))
            if meta_label:
                return meta_label

        payload_label = Labelme2YOLO._clean_label(payload.get("instance_label"))
        if payload_label:
            return payload_label

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
    def _resolve_instance_label(
        shape: Dict[str, object],
        candidate_labels: Optional[Set[str]] = None,
        default_label: Optional[str] = None,
    ) -> str:
        """Infer the instance label for a shape using flags and heuristics."""
        flags = shape.get("flags") or {}
        flag_label = Labelme2YOLO._clean_label(
            flags.get("instance_label") if isinstance(flags, dict) else None
        )
        if flag_label:
            return flag_label
        payload_label = Labelme2YOLO._clean_label(shape.get("instance_label"))
        if payload_label:
            return payload_label

        label = Labelme2YOLO._clean_label(shape.get("label"))
        shape_type = (shape.get("shape_type") or "polygon").lower()

        if shape_type == "point":
            candidates = candidate_labels or set()
            lower_label = label.lower()
            for candidate in sorted(candidates, key=len, reverse=True):
                if lower_label.startswith(candidate.lower()):
                    return candidate
            # Only treat delimiters as instance/keypoint separators when the instance
            # prefix is known. Underscores are common inside keypoint names (e.g.
            # tail_base, left_ear) and should not be split blindly.
            for delimiter in ("_", "-", ":", "|", " "):
                if delimiter not in label:
                    continue
                prefix = label.split(delimiter, 1)[0]
                if not prefix:
                    continue
                if delimiter != "_":
                    return prefix
                if any(prefix.lower() == cand.lower() for cand in candidates):
                    return prefix
                if default_label and prefix.lower() == default_label.lower():
                    return default_label
            if default_label:
                return default_label
            return label
        return label or (default_label or "")

    @staticmethod
    def _resolve_keypoint_label(shape: Dict[str, object], instance_label: str) -> str:
        """Determine the keypoint label, removing the instance prefix when possible."""
        flags = shape.get("flags") or {}
        if isinstance(flags, dict):
            display_label = Labelme2YOLO._clean_label(flags.get("display_label"))
            if display_label:
                return display_label
        payload_label = Labelme2YOLO._clean_label(shape.get("display_label"))
        if payload_label:
            return payload_label

        label = Labelme2YOLO._clean_label(shape.get("label"))
        if instance_label:
            inst_len = len(instance_label)
            if label.lower().startswith(instance_label.lower()) and inst_len < len(
                label
            ):
                suffix = label[inst_len:]
                # Only strip when the instance/keypoint boundary is explicit.
                if suffix and suffix[0] in "_-:| ":
                    suffix = suffix.lstrip("_-:| ")
                    if suffix:
                        return suffix
        return label

    @staticmethod
    def _derive_visibility(shape: Dict[str, object]) -> Optional[int]:
        """Resolve keypoint visibility from flags/description (YOLO v: 0/1/2)."""
        try:
            return visibility_from_labelme_shape(shape)
        except Exception:
            return None

    @staticmethod
    def _extend_bounds(
        bounds: Optional[Tuple[float, float, float, float]], x: float, y: float
    ) -> Tuple[float, float, float, float]:
        if bounds is None:
            return float(x), float(y), float(x), float(y)
        min_x, min_y, max_x, max_y = bounds
        return (
            min(min_x, float(x)),
            min(min_y, float(y)),
            max(max_x, float(x)),
            max(max_y, float(y)),
        )

    @dataclass(frozen=True)
    class _PoseRegion:
        instance_key: str
        instance_label: str
        shape_type: str
        polygon: Tuple[Tuple[float, float], ...]
        center: Optional[Tuple[float, float]]
        radius: Optional[float]
        area: float
        centroid: Tuple[float, float]

    @staticmethod
    def _normalize_group_id(value: object) -> Optional[object]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except Exception:
                return text
        return None

    @staticmethod
    def _polygon_area_and_centroid(
        points: Sequence[Sequence[float]],
    ) -> Tuple[float, Tuple[float, float]]:
        if len(points) < 3:
            xs = [float(p[0]) for p in points if len(p) >= 2]
            ys = [float(p[1]) for p in points if len(p) >= 2]
            if not xs or not ys:
                return 0.0, (0.0, 0.0)
            return 0.0, (sum(xs) / len(xs), sum(ys) / len(ys))
        area = 0.0
        cx = 0.0
        cy = 0.0
        n = len(points)
        for i in range(n):
            x0, y0 = float(points[i][0]), float(points[i][1])
            x1, y1 = float(points[(i + 1) % n][0]), float(points[(i + 1) % n][1])
            cross = x0 * y1 - x1 * y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        area *= 0.5
        if abs(area) < 1e-12:
            xs = [float(p[0]) for p in points if len(p) >= 2]
            ys = [float(p[1]) for p in points if len(p) >= 2]
            if not xs or not ys:
                return 0.0, (0.0, 0.0)
            return 0.0, (sum(xs) / len(xs), sum(ys) / len(ys))
        cx /= 6.0 * area
        cy /= 6.0 * area
        return abs(area), (cx, cy)

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, polygon: Sequence[Sequence[float]]
    ) -> bool:
        if len(polygon) < 3:
            return False
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    def _region_from_shape(
        self, shape: Dict[str, object], *, instance_key: str, instance_label: str
    ) -> Optional["_PoseRegion"]:
        shape_type = (shape.get("shape_type") or "polygon").lower()
        raw_points = shape.get("points") or []
        if not isinstance(raw_points, list) or not raw_points:
            return None

        polygon: List[Tuple[float, float]] = []
        center: Optional[Tuple[float, float]] = None
        radius: Optional[float] = None
        centroid: Tuple[float, float] = (0.0, 0.0)
        area = 0.0

        if shape_type == "circle" and len(raw_points) >= 2:
            cx, cy = raw_points[0][:2]
            px, py = raw_points[1][:2]
            center = (float(cx), float(cy))
            radius = float(math.hypot(float(px) - float(cx), float(py) - float(cy)))
            area = math.pi * (radius**2)
            centroid = center
            return self._PoseRegion(
                instance_key=instance_key,
                instance_label=instance_label,
                shape_type=shape_type,
                polygon=tuple(),
                center=center,
                radius=radius,
                area=area,
                centroid=centroid,
            )

        if shape_type == "rectangle" and len(raw_points) >= 2:
            (x0, y0), (x1, y1) = raw_points[0][:2], raw_points[1][:2]
            x_min = min(float(x0), float(x1))
            x_max = max(float(x0), float(x1))
            y_min = min(float(y0), float(y1))
            y_max = max(float(y0), float(y1))
            polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        else:
            for pt in raw_points:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                polygon.append((float(pt[0]), float(pt[1])))

        if len(polygon) < 3:
            return None

        area, centroid = self._polygon_area_and_centroid(polygon)
        if area <= 0.0:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 0.0)
        return self._PoseRegion(
            instance_key=instance_key,
            instance_label=instance_label,
            shape_type=shape_type,
            polygon=tuple(polygon),
            center=None,
            radius=None,
            area=area,
            centroid=centroid,
        )

    @staticmethod
    def _bounds_to_cxcywh(
        bounds: Optional[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[float]:
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

    def _collect_pose_instances(
        self,
        shapes: List[Dict[str, object]],
        default_instance_label: Optional[str] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Group shapes by instance to prepare pose annotations."""
        instances: Dict[str, Dict[str, object]] = {}
        polygon_labels: Set[str] = set()
        regions: List[Labelme2YOLO._PoseRegion] = []
        label_instance_keys: Dict[str, List[str]] = defaultdict(list)
        group_to_instance_key: Dict[object, str] = {}
        auto_counter: Dict[str, int] = defaultdict(int)
        explicit_group_id = False
        schema_instances: Set[str] = set()
        if self.pose_schema and getattr(self.pose_schema, "instances", None):
            for inst in self.pose_schema.instances:
                clean = self._clean_label(inst)
                if clean:
                    schema_instances.add(clean)

        def register_instance_key(label: str, key: str) -> None:
            keys = label_instance_keys[label]
            if key not in keys:
                keys.append(key)

        for shape in shapes:
            shape_type = (shape.get("shape_type") or "polygon").lower()
            if shape_type == "point":
                continue
            instance_label = self._resolve_instance_label(
                shape, polygon_labels, default_label=default_instance_label
            )
            if not instance_label:
                continue
            polygon_labels.add(instance_label)

            group_id = self._normalize_group_id(shape.get("group_id"))
            if group_id is not None:
                instance_key = f"{instance_label}#{group_id}"
            else:
                idx = auto_counter[instance_label]
                auto_counter[instance_label] += 1
                instance_key = f"{instance_label}@{idx}"
            register_instance_key(instance_label, instance_key)
            if group_id is not None and group_id not in group_to_instance_key:
                group_to_instance_key[group_id] = instance_key

            entry = instances.setdefault(
                instance_key,
                {
                    "class_label": self._clean_label(shape.get("label"))
                    or instance_label,
                    "instance_label": instance_label,
                    "bounds": None,
                    "keypoints": {},
                },
            )
            class_label = self._clean_label(shape.get("label")) or entry["class_label"]
            if class_label:
                entry["class_label"] = class_label
                entry["instance_label"] = instance_label

            for point in shape.get("points") or []:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                entry["bounds"] = self._extend_bounds(
                    entry["bounds"], point[0], point[1]
                )

            region = self._region_from_shape(
                shape, instance_key=instance_key, instance_label=instance_label
            )
            if region is not None:
                regions.append(region)

        def best_region_for_point(
            x: float, y: float, candidate_keys: Optional[Set[str]] = None
        ) -> Optional[str]:
            if not regions:
                return None
            candidates = [
                r
                for r in regions
                if candidate_keys is None or r.instance_key in candidate_keys
            ]
            if not candidates:
                return None

            inside: List[Labelme2YOLO._PoseRegion] = []
            for region in candidates:
                if (
                    region.shape_type == "circle"
                    and region.center
                    and region.radius is not None
                ):
                    dx = x - region.center[0]
                    dy = y - region.center[1]
                    if (dx * dx + dy * dy) <= (region.radius * region.radius):
                        inside.append(region)
                    continue
                if region.polygon and self._point_in_polygon(x, y, region.polygon):
                    inside.append(region)
            pool = inside or candidates
            # Prefer smallest containing region; fall back to nearest centroid.
            pool = sorted(
                pool,
                key=lambda r: (
                    float(r.area if r.area is not None else 0.0),
                    (x - r.centroid[0]) ** 2 + (y - r.centroid[1]) ** 2,
                ),
            )
            return pool[0].instance_key if pool else None

        candidate_instance_labels = (
            set(label_instance_keys.keys()) | set(polygon_labels) | schema_instances
        )
        for shape in shapes:
            shape_type = (shape.get("shape_type") or "polygon").lower()
            if shape_type != "point":
                continue
            points = shape.get("points") or []
            if not points:
                continue
            x, y = points[0][:2]

            group_id = self._normalize_group_id(shape.get("group_id"))
            instance_key: Optional[str] = None
            if group_id is not None:
                explicit_group_id = True
                instance_key = group_to_instance_key.get(group_id)

            flags = shape.get("flags") or {}
            flagged_instance = self._clean_label(
                flags.get("instance_label") if isinstance(flags, dict) else None
            )
            if not flagged_instance:
                flagged_instance = self._clean_label(shape.get("instance_label"))
            if instance_key is None and flagged_instance:
                keys = label_instance_keys.get(flagged_instance) or []
                candidate_keys = set(keys) if keys else None
                instance_key = best_region_for_point(float(x), float(y), candidate_keys)
                if instance_key is None and keys:
                    instance_key = keys[0]

            if instance_key is None:
                label = self._clean_label(shape.get("label"))
                hinted_instance = ""
                for inst in sorted(candidate_instance_labels, key=len, reverse=True):
                    if label.lower().startswith(inst.lower()):
                        suffix = label[len(inst) :]
                        if suffix and suffix[0] in "_-:| ":
                            hinted_instance = inst
                            break
                candidate_keys = None
                if hinted_instance and hinted_instance in label_instance_keys:
                    candidate_keys = set(label_instance_keys[hinted_instance])
                instance_key = best_region_for_point(float(x), float(y), candidate_keys)

            if instance_key is None and group_id is not None:
                inferred_label = (
                    flagged_instance
                    or hinted_instance
                    or self._clean_label(default_instance_label)
                    or "object"
                )
                instance_key = f"{inferred_label}#{group_id}"
                register_instance_key(inferred_label, instance_key)
                group_to_instance_key[group_id] = instance_key

            if instance_key is None:
                # No polygons/regions: fall back to heuristics based on labels/defaults.
                inferred = self._resolve_instance_label(
                    shape,
                    candidate_instance_labels or None,
                    default_label=default_instance_label,
                )
                if inferred:
                    instance_key = inferred
                    if inferred not in label_instance_keys:
                        register_instance_key(inferred, instance_key)

            if instance_key is None:
                continue

            entry = instances.setdefault(
                instance_key,
                {
                    "class_label": instance_key.split("#", 1)[0].split("@", 1)[0],
                    "instance_label": instance_key.split("#", 1)[0].split("@", 1)[0],
                    "bounds": None,
                    "keypoints": {},
                },
            )
            entry["bounds"] = self._extend_bounds(entry["bounds"], x, y)

            instance_label = self._clean_label(entry.get("instance_label")) or ""
            keypoint_label = self._resolve_keypoint_label(shape, instance_label)
            if not keypoint_label:
                keypoint_label = f"kp_{len(entry['keypoints'])}"
            visibility = self._derive_visibility(shape)
            if visibility is None:
                visibility = int(KeypointVisibility.VISIBLE)
            entry["keypoints"][keypoint_label] = {
                "x": float(x),
                "y": float(y),
                "visible": bool(int(visibility) == int(KeypointVisibility.VISIBLE)),
                "visibility": int(visibility),
            }

        schema_multi = bool(schema_instances) and len(schema_instances) > 1
        if (
            not polygon_labels
            and len(instances) > 1
            and not explicit_group_id
            and not schema_multi
        ):
            target_label = self._clean_label(default_instance_label) or next(
                iter(instances)
            )
            merged = {
                "class_label": target_label,
                "bounds": None,
                "keypoints": {},
            }
            for data in instances.values():
                merged["class_label"] = (
                    self._clean_label(data.get("class_label")) or merged["class_label"]
                )
                bounds = data.get("bounds")
                if bounds:
                    min_x, min_y, max_x, max_y = bounds
                    merged["bounds"] = self._extend_bounds(
                        merged["bounds"], min_x, min_y
                    )
                    merged["bounds"] = self._extend_bounds(
                        merged["bounds"], max_x, max_y
                    )
                for label, kp in data["keypoints"].items():
                    merged["keypoints"][label] = kp
            if merged["bounds"] is None:
                merged["bounds"] = (0.0, 0.0, 0.0, 0.0)
            instances = {target_label: merged}

        return {label: data for label, data in instances.items() if data["keypoints"]}

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
        items = self._discover_items(require_image=True)
        if not items:
            folder = Path(self.json_file_dir)
            store = AnnotationStore.for_frame_path(
                folder / f"{folder.name}_000000000.json"
            )
            if store.store_path.exists():
                json_names = [
                    f"{folder.name}_{frame:09d}.json"
                    for frame in sorted(store.iter_frames())
                ]
                items = [
                    Labelme2YOLO._LabelmeItem(
                        json_path=(folder / name),
                        image_path=None,
                        output_stem=Path(name).stem,
                        preset_split=None,
                    )
                    for name in json_names
                ]

        if not items:
            return

        # If dataset already has split-like directories (e.g. train*, val*, test*),
        # always respect them and ignore val_size/test_size.
        preset = [
            item for item in items if item.preset_split in ("train", "val", "test")
        ]
        if preset:
            train_items = [i for i in items if i.preset_split == "train"]
            val_items = [i for i in items if i.preset_split == "val"]
            test_items = [i for i in items if i.preset_split == "test"]
            train_items.extend(
                i for i in items if i.preset_split not in ("train", "val", "test")
            )
            # Ensure train split is non-empty when possible.
            if not train_items:
                if val_items:
                    train_items.append(val_items.pop())
                elif test_items:
                    train_items.append(test_items.pop())
        else:
            train_items, val_items, test_items = self.split_jsons(
                [], items, val_size, test_size
            )

        # Create the train and validation directories if they don't exist already
        self.create_yolo_dataset_dirs()

        # Convert labelme object to yolo format object, and save them to files
        # Also get image from labelme json file and save them under images folder
        for target_dir, split_items in zip(
            ("train/", "val/", "test/"),
            (train_items, val_items, test_items),
        ):
            for item in split_items:
                self.json_to_text(target_dir, item)

        # Save the dataset configuration file
        self.save_data_yaml()

    def get_yolo_objects(self, json_path: Union[str, Path], json_data, img_path):
        """Return a list of YOLO formatted objects from a JSON annotation file and image."""
        image_height = json_data["imageHeight"]
        image_width = json_data["imageWidth"]
        shapes = json_data.get("shapes") or []

        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = Path(self.json_file_dir) / json_path
        default_label = self._default_instance_label(json_path, json_data)

        pose_instances = self._collect_pose_instances(
            shapes, default_instance_label=default_label
        )
        if pose_instances:
            self.annotation_type = "pose"
            for data in pose_instances.values():
                self._update_keypoint_order(data["keypoints"].keys())

            yolo_objects = []
            for instance_label, data in pose_instances.items():
                class_label = data.get("class_label") or instance_label
                if class_label and class_label not in self.label_to_id_dict:
                    self.label_to_id_dict[class_label] = len(self.label_to_id_dict)
                label_id = self.label_to_id_dict.get(class_label or default_label, 0)
                bbox = self._bounds_to_cxcywh(
                    data.get("bounds"), image_width, image_height
                )
                keypoint_values: List[float] = []
                for kp_label in self.keypoint_labels_order:
                    kp = data["keypoints"].get(kp_label)
                    if kp:
                        x = kp["x"] / image_width if image_width else 0.0
                        y = kp["y"] / image_height if image_height else 0.0
                        if self.include_visibility:
                            visibility = kp.get("visibility")
                            if visibility is None:
                                visibility = 2 if kp.get("visible", True) else 1
                            keypoint_values.extend([x, y, int(visibility)])
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
            if shape_type == "circle":
                yolo_obj = self.circle_shape_to_yolo(shape, image_height, image_width)
                yolo_objects.append(yolo_obj)
            elif shape_type == "point":
                continue
            else:
                yolo_obj = self.scale_points(shape, image_height, image_width)
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
        if isinstance(json_name, Labelme2YOLO._LabelmeItem):
            item = json_name
            json_path = str(item.json_path)
            json_data = load_labelme_json(json_path)
            output_stem = item.output_stem
            img_src = item.image_path or self._resolve_image_path(
                json_data, json_path=item.json_path
            )
        else:
            output_stem = Path(str(json_name)).stem
            json_path = os.path.join(self.json_file_dir, str(json_name))
            json_data = load_labelme_json(json_path)
            img_src = self._resolve_image_path(json_data, json_path=Path(json_path))

        img_path = self.save_or_copy_image(
            json_data,
            output_stem,
            self.image_folder,
            target_dir,
            json_path=json_path,
            source_image_path=str(img_src) if img_src else None,
        )

        yolo_objects = self.get_yolo_objects(Path(json_path), json_data, img_path)
        self.save_yolo_txt_label_file(
            output_stem, self.label_folder, target_dir, yolo_objects
        )

    def scale_points(
        self,
        labelme_shape: dict,
        image_height: int,
        image_width: int,
        output_fromat: str = "polygon",
        include_visibility: bool = None,
    ):
        """
        Returns the label_id and scaled points of the given shape object in YOLO format.
        """
        # Use the class default if not explicitly provided
        if include_visibility is None:
            include_visibility = self.include_visibility
        cx, cy, w, h = find_bbox_from_shape(labelme_shape)
        scaled_cxcywh = [
            cx / image_width,
            cy / image_height,
            w / image_width,
            h / image_height,
        ]
        # Extract the points from the shape object
        point_list = labelme_shape["points"]
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
            label_id = self.label_to_id_dict[labelme_shape["label"]]
        except KeyError:
            label_id = 0
        # Return the label_id and points as a list
        if output_fromat == "bbox":
            return label_id, scaled_cxcywh
        elif output_fromat == "pose":
            return label_id, scaled_cxcywh + scaled_points
        else:
            return label_id, points.tolist()

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
        cx, cy = labelme_shape["points"][0]

        # Calculate the radius of the circle.
        radius = math.sqrt(
            (cx - labelme_shape["points"][1][0]) ** 2
            + (cy - labelme_shape["points"][1][1]) ** 2
        )

        # Calculate the width and height of the circle.
        w = 2 * radius
        h = 2 * radius

        # Calculate the YOLO coordinates.
        yolo_cx = round(float(cx / image_width), 6)
        yolo_cy = round(float(cy / image_height), 6)
        yolo_w = round(float(w / image_width), 6)
        yolo_h = round(float(h / image_height), 6)

        # Get the label ID.
        label_id = self.label_to_id_dict[labelme_shape["label"]]

        # Return the YOLO object as a tuple.
        return label_id, yolo_cx, yolo_cy, yolo_w, yolo_h

    @staticmethod
    def save_or_copy_image(
        json_data: dict,
        output_stem: str,
        image_dir_path: str,
        target_dir: str,
        json_path: Optional[str] = None,
        source_image_path: Optional[str] = None,
    ) -> str:
        """
        Save an image in YOLO format.

        :param json_data: Dictionary containing the data from the json file.
        :param json_name: Name of the json file.
        :param image_dir_path: Path to the directory containing the image data.
        :param target_dir: Target directory to save the image in.
        :return: Path of the saved image.
        """
        output_stem = str(output_stem or "").strip() or "image"
        ext = ".png"
        if source_image_path:
            try:
                src_ext = Path(source_image_path).suffix.lower()
                if src_ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                    ext = src_ext
            except Exception:
                pass
        if ext == ".png":
            image_path_field = json_data.get("imagePath")
            if isinstance(image_path_field, str) and image_path_field.strip():
                try:
                    field_ext = Path(image_path_field).suffix.lower()
                    if field_ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                        ext = field_ext
                except Exception:
                    pass
        img_name = f"{output_stem}{ext}"
        img_path = os.path.join(image_dir_path, target_dir, img_name)

        # if the image is not already saved, then save it
        if not os.path.exists(img_path):
            image_data = json_data.get("imageData")
            if image_data:
                img = img_b64_to_arr(image_data)
                PIL.Image.fromarray(img).save(img_path)
            else:
                src_img_path = json_data.get("imagePath") or ""
                candidates = []
                if source_image_path:
                    candidates.append(source_image_path)
                if src_img_path:
                    candidates.append(src_img_path)
                    if json_path:
                        candidates.append(
                            os.path.join(os.path.dirname(json_path), src_img_path)
                        )
                chosen = None
                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        chosen = candidate
                        break
                if chosen:
                    shutil.copy(chosen, img_path)
        return img_path

    def save_data_yaml(self):
        """Save the dataset information as a YAML file in the new format."""
        # Set the path for the YAML file
        yaml_path = os.path.join(
            self.json_file_dir, f"{self.yolo_dataset_name}/", "data.yaml"
        )

        # Construct the names section
        names_section = "names:\n"
        for label, label_id in self.label_to_id_dict.items():
            names_section += f"  {label_id}: {label}\n"

        # Write the YAML file content
        with open(yaml_path, "w+") as yaml_file:
            # Relative path to the dataset
            yaml_file.write(f"path: ../{self.yolo_dataset_name}\n")
            yaml_file.write("train: images/train\n")
            yaml_file.write("val: images/val\n")
            # Include test set in the YAML
            yaml_file.write("test: images/test\n")
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
                        self.keypoint_labels_order
                    )
                if flip_idx:
                    yaml_file.write("\n")
                    yaml_file.write(f"flip_idx: {flip_idx}\n")
                else:
                    yaml_file.write(
                        "#(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face. For example if we assume five keypoints of facial landmark: [left eye, right eye, nose, left mouth, right mouth], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3] (just exchange the left-right index, i.e. 0-1 and 3-4, and do not modify others like nose in this example.)\n"
                    )
                    yaml_file.write(
                        "#flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
                    )

            yaml_file.write(names_section)

    @staticmethod
    def save_yolo_txt_label_file(
        json_name: str,
        label_folder_path: str,
        target_dir: str,
        yolo_objects: List[Tuple[str, List[float]]],
    ) -> None:
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
        base = str(json_name)
        if base.lower().endswith(".json"):
            base = base[:-5]
        txt_path = os.path.join(label_folder_path, target_dir, f"{base}.txt")

        with open(txt_path, "w+") as f:
            # Write each YOLO object as a line in the label file
            for label, points in yolo_objects:
                points = [str(item) for item in points]
                yolo_object_line = f"{label} {' '.join(points)}\n"
                f.write(yolo_object_line)
