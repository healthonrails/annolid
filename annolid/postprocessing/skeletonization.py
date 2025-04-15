import os
import json
import argparse
import glob
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import medial_axis, skeletonize
from annolid.annotation.labelme2binarymasks import LabelMeProcessor
from sklearn.decomposition import PCA


class ShapeProcessor:
    def __init__(self, binary_mask: np.ndarray) -> None:
        """
        Initialize ShapeProcessor with a binary mask.

        Parameters:
        -----------
        binary_mask : np.ndarray
            A binary mask representing the shape.
        """
        self.binary_mask = binary_mask
        self.skeleton = skeletonize(self.binary_mask)
        # Compute medial axis and its distance transform
        medial, dist = medial_axis(self.binary_mask, return_distance=True)
        self.medial_axis = medial
        self.dist_transform = dist
        self.medial_axis = self.prune_medial_axis(threshold_length=2)

        self.centroid = self.calculate_centroid()
        self.extreme_points = self.calculate_extreme_points_on_medial_axis()
        # Label extreme points if any were found
        self.labeled_points = self.label_extreme_points() if self.extreme_points else {}
        self.ensure_tail_is_farthest()

    def prune_medial_axis(self, threshold_length: int) -> np.ndarray:
        """
        Prune small branches in the medial axis.

        Parameters:
        -----------
        threshold_length : int
            The minimum branch length to keep.

        Returns:
        --------
        np.ndarray:
            The pruned medial axis.
        """
        labeled, _ = label(self.medial_axis)
        sizes = np.bincount(labeled.ravel())
        mask_sizes = sizes >= threshold_length
        mask_sizes[0] = 0  # Ensure that background is removed
        cleaned = mask_sizes[labeled]
        return cleaned

    def calculate_centroid(self) -> Tuple[float, float]:
        """
        Calculate the centroid of the medial axis.

        Returns:
        --------
        Tuple[float, float]:
            The centroid coordinates (x, y).
        """
        y_coords, x_coords = np.nonzero(self.medial_axis)
        if len(x_coords) == 0 or len(y_coords) == 0:
            logging.warning(
                "Medial axis has no nonzero points; using default centroid (0, 0).")
            return (0.0, 0.0)
        return (np.mean(x_coords), np.mean(y_coords))

    def calculate_extreme_points_on_medial_axis(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Calculate the extreme points (leftmost, rightmost, topmost, bottommost) on the medial axis.

        Returns:
        --------
        Optional[Dict[str, Tuple[int, int]]]:
            A dictionary with keys 'leftmost', 'rightmost', 'topmost', 'bottommost' mapped to their coordinates,
            or None if no points are found.
        """
        y_coords, x_coords = np.nonzero(self.medial_axis)
        if len(x_coords) == 0 or len(y_coords) == 0:
            logging.warning("No points found in the medial axis.")
            return None

        points = list(zip(x_coords, y_coords))
        leftmost = min(points, key=lambda p: p[0])
        rightmost = max(points, key=lambda p: p[0])
        topmost = min(points, key=lambda p: p[1])
        bottommost = max(points, key=lambda p: p[1])

        return {"leftmost": leftmost, "rightmost": rightmost, "topmost": topmost, "bottommost": bottommost}

    def distance_to_centroid(self, point: Tuple[float, float]) -> float:
        """
        Compute the Euclidean distance from a point to the centroid.

        Parameters:
        -----------
        point : Tuple[float, float]
            The (x, y) coordinate.

        Returns:
        --------
        float:
            The Euclidean distance.
        """
        return np.linalg.norm(np.array(point) - np.array(self.centroid))

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Compute the Euclidean distance between two points.

        Parameters:
        -----------
        point1 : Tuple[float, float]
            The first point (x, y).
        point2 : Tuple[float, float]
            The second point (x, y).

        Returns:
        --------
        float:
            The Euclidean distance.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calculate_polygon_edge_points(self) -> List[Tuple[int, int]]:
        """
        Calculate all edge points of the polygon defined by the binary mask.

        Returns:
        --------
        List[Tuple[int, int]]:
            A list of (x, y) coordinates of the edge points.
        """
        y_coords, x_coords = np.nonzero(self.binary_mask)
        return list(zip(x_coords, y_coords))

    def distance_to_polygon_edge(self, point: Tuple[float, float]) -> float:
        """
        Calculate the shortest distance from a given point to the polygon edge.

        Parameters:
        -----------
        point : Tuple[float, float]
            The (x, y) coordinate.

        Returns:
        --------
        float:
            The minimum distance.
        """
        edge_points = self.calculate_polygon_edge_points()
        distances = [self.calculate_distance(
            point, (ex, ey)) for ex, ey in edge_points]
        return min(distances) if distances else float('inf')

    def label_extreme_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Label the extreme points of the medial axis as 'head', 'tail', 'nose', 'tailbase', and 'body_center'.

        Returns:
        --------
        Dict[str, Tuple[float, float]]:
            A dictionary with labeled keypoints.
        """
        if not self.extreme_points:
            return {}

        # Determine head (closest to centroid) and tail (farthest from centroid)
        distances = {k: self.distance_to_centroid(
            v) for k, v in self.extreme_points.items()}
        head_label = min(distances, key=distances.get)
        tail_label = max(distances, key=distances.get)

        labels = {
            "head": self.extreme_points[head_label],
            "tail": self.extreme_points[tail_label]
        }

        # Process remaining points for additional labels
        remaining = {k: v for k, v in self.extreme_points.items() if k not in [
            head_label, tail_label]}
        if remaining:
            # For "nose": select the remaining point closest to head, further refined by proximity to edge
            nose_distances = {k: self.calculate_distance(
                v, labels["head"]) for k, v in remaining.items()}
            nose_candidate = min(nose_distances, key=nose_distances.get)
            nose_candidates = {
                k: self.distance_to_polygon_edge(self.extreme_points[k])
                for k, d in nose_distances.items() if d == nose_distances[nose_candidate]
            }
            nose_label = min(nose_candidates, key=nose_candidates.get)
            labels["nose"] = self.extreme_points[nose_label]

            # Use PCA to determine the primary axis to estimate "tailbase"
            y_coords, x_coords = np.nonzero(self.medial_axis)
            coords = np.column_stack((x_coords, y_coords))
            if len(coords) >= 2:
                pca = PCA(n_components=2)
                pca.fit(coords)
                primary_axis = pca.components_[0]
                projections = {k: np.dot(primary_axis, (np.array(v) - np.array(labels["head"])))
                               for k, v in remaining.items()}
                tailbase_label = max(projections, key=projections.get)
                labels["tailbase"] = remaining[tailbase_label]
            else:
                labels["tailbase"] = next(iter(remaining.values()))
        else:
            labels["nose"] = labels["head"]
            labels["tailbase"] = labels["tail"]

        labels["body_center"] = self.centroid
        return labels

    def ensure_tail_is_farthest(self) -> None:
        """
        Re-check and ensure that the labeled 'tail' point is indeed the farthest from the 'head'.
        """
        if "head" not in self.labeled_points or "tail" not in self.labeled_points:
            return

        head_point = self.labeled_points["head"]
        tail_point = self.labeled_points["tail"]
        max_distance = self.calculate_distance(head_point, tail_point)

        for label, point in self.labeled_points.items():
            if label in ["head", "tail"]:
                continue
            dist = self.calculate_distance(head_point, point)
            if dist > max_distance:
                max_distance = dist
                self.labeled_points["tail"] = point

    def visualize(self) -> None:
        """
        Visualize the binary mask, medial axis, and key points with annotations.
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(self.binary_mask, cmap='gray')
        plt.imshow(self.medial_axis, cmap='jet', alpha=0.5)
        for label, point in self.labeled_points.items():
            plt.plot(point[0], point[1], 'o', label=label.capitalize())
        plt.title('Medial Axis and Labeled Points')
        plt.legend()
        plt.show()


def compute_distance_transform(binary_image: np.ndarray) -> np.ndarray:
    """
    Compute the distance transform of a binary image.

    Parameters:
    -----------
    binary_image : np.ndarray
        The binary image.

    Returns:
    --------
    np.ndarray:
        The computed distance transform.
    """
    return distance_transform_edt(binary_image)


def extract_medial_axis(binary_image: np.ndarray) -> np.ndarray:
    """
    Extract the medial axis from a binary image.

    Parameters:
    -----------
    binary_image : np.ndarray
        The binary image.

    Returns:
    --------
    np.ndarray:
        The medial axis.
    """
    medial, _ = medial_axis(binary_image, return_distance=True)
    return medial


def add_key_points_to_labelme_json(json_path: str, instance_names: List[str], is_vis: bool = False) -> None:
    """
    Process a LabelMe JSON file, extract key points for specified instances, and add these as new point annotations.

    Parameters:
    -----------
    json_path : str
        Path to the LabelMe JSON file.
    instance_names : List[str]
        List of instance names to process.
    is_vis : bool, optional
        If True, visualize the medial axis and key points during processing.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {json_path}: {e}")
        return

    mask_converter = LabelMeProcessor(json_path)
    binary_masks = mask_converter.get_all_masks()

    for binary_mask, label in binary_masks:
        if label in instance_names:
            processor = ShapeProcessor(binary_mask)
            if is_vis:
                processor.visualize()
            for point_label, point in processor.labeled_points.items():
                data.setdefault('shapes', []).append({
                    "label": f"{label}_{point_label}",
                    "points": [[int(point[0]), int(point[1])]],
                    "shape_type": "point",
                    "visible": True,
                })

    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save updated JSON file {json_path}: {e}")


def extract_ground_truth_keypoints(json_path: str) -> Dict[str, Any]:
    """
    Extract ground truth keypoints from a LabelMe JSON file.
    Only points whose labels do not include an underscore are extracted.

    Parameters:
    -----------
    json_path : str
        Path to the LabelMe JSON file containing keypoints.

    Returns:
    --------
    Dict[str, Any]:
        Dictionary mapping keypoint labels to their coordinates.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {json_path}: {e}")
        return {}

    ground_truth = {}
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'point' and '_' not in shape.get('label', ''):
            ground_truth[shape['label']] = shape['points'][0]
    return ground_truth


def main(input_folder: str, instance_names: List[str]) -> None:
    """
    Process all LabelMe JSON files in the input folder by adding keypoint annotations.

    Parameters:
    -----------
    input_folder : str
        Directory containing JSON files.
    instance_names : List[str]
        List of instance names to process.
    """
    json_files = glob.glob(os.path.join(
        input_folder, '**/*.json'), recursive=True)
    ground_truth_keypoints = None

    for json_file in json_files:
        base_filename = os.path.basename(json_file)
        if '00000000.json' in base_filename:
            ground_truth_keypoints = extract_ground_truth_keypoints(json_file)
            logging.info(
                f"Extracted ground truth keypoints from {json_file}: {ground_truth_keypoints}")
        else:
            add_key_points_to_labelme_json(json_file, instance_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Add key points to LabelMe JSON files.")
    parser.add_argument(
        "input_folder", help="Path to the folder containing JSON files.")
    parser.add_argument("instance_names", nargs="+",
                        help="List of instance names.")
    args = parser.parse_args()

    main(args.input_folder, args.instance_names)
