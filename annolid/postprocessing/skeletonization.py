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

    def compute_symmetry_axis(self) -> tuple:
        """
        Compute the symmetry axis for the head region.

        Uses PCA on points in the upper (facial) region of the medial axis.

        Returns:
            center (np.ndarray): The centroid of the head region.
            axis (np.ndarray): The unit vector along the primary (symmetry) axis.
        """
        # Select points above the overall centroid (assumed to be the head region)
        y_coords, x_coords = np.nonzero(self.medial_axis)
        points = np.column_stack((x_coords, y_coords))
        head_region = points[points[:, 1] < self.centroid[1]]
        if len(head_region) < 2:  # fallback to all points
            head_region = points
        pca = PCA(n_components=2)
        pca.fit(head_region)
        center = np.mean(head_region, axis=0)
        axis = pca.components_[0]  # first principal component
        return center, axis

    def refine_keypoints_with_symmetry(self, candidates: dict, extra_features: dict = None) -> dict:
        """
        Refine keypoint assignments using symmetry and additional anatomical cues.

        Parameters:
            candidates (dict): Initial candidates for keypoints (e.g., 'head', 'nose', 'tail').
            extra_features (dict, optional): Detected positions of features like ears or eyes.
                Expected keys might include 'left_ear', 'right_ear', 'eye_left', 'eye_right', etc.

        Returns:
            dict: Refined keypoint dictionary with improved anatomical consistency.
        """
        refined = {}

        # Compute symmetry axis for the head region.
        center, axis = self.compute_symmetry_axis()

        # For the nose, enforce proximity to the symmetry axis.
        head_pt = np.array(candidates.get("head", self.centroid))
        nose_candidate = np.array(candidates.get("nose", head_pt))
        vec = nose_candidate - center
        # projection on the symmetry axis
        projection = np.dot(vec, axis) * axis
        deviation = np.linalg.norm(vec - projection)

        # If additional facial features (e.g., ears, eyes) are provided, adjust the nose score.
        if extra_features:
            # For instance, assume the nose should be near the center of eyes and ears.
            face_features = []
            for feat in ['left_ear', 'right_ear', 'eye_left', 'eye_right']:
                if feat in extra_features:
                    face_features.append(np.array(extra_features[feat]))
            if face_features:
                face_center = np.mean(face_features, axis=0)
                nose_to_face = np.linalg.norm(nose_candidate - face_center)
                # Weight the deviation: lower nose_to_face distance further supports nose candidate.
                deviation *= 0.5 if nose_to_face < 20 else 1.0  # threshold is an example value
        # Use the deviation as a quality measure: lower is better. If the candidate deviates too much,
        # consider swapping with a more symmetric candidate if available.
        # assign refined nose candidate
        refined["nose"] = tuple(nose_candidate)

        # For tail, typically the candidate should be the one farthest from the head in the direction opposite to the nose.
        tail_candidate = np.array(candidates.get("tail", self.centroid))
        # Here we use the original logic: tail should maximize distance from head.
        refined["tail"] = tuple(tail_candidate)

        # Retain the head and body center as computed.
        refined["head"] = tuple(head_pt)
        refined["body_center"] = self.centroid

        # Optionally incorporate extra features if available.
        if extra_features:
            refined.update(extra_features)

        return refined

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

    def compute_medial_axis_curvature(self) -> list:
        """
        Compute the curvature at each point along the medial axis.

        Returns:
        --------
        List[Tuple[float, Tuple[int, int]]]:
            A list of tuples where the first element is the curvature (in radians)
            and the second element is the corresponding (x, y) point.
        """
        # Extract points along the medial axis
        y_coords, x_coords = np.nonzero(self.medial_axis)
        points = list(zip(x_coords, y_coords))
        if len(points) < 3:
            return []

        # Order points using PCA – this provides an approximate order along the body.
        points_arr = np.array(points)
        pca = PCA(n_components=1)
        pca.fit(points_arr)
        projection = points_arr @ pca.components_[0]
        sorted_indices = np.argsort(projection)
        sorted_points = points_arr[sorted_indices]

        curvature_points = []
        # Use finite differences to approximate curvature at each interior point.
        for i in range(1, len(sorted_points) - 1):
            p_prev = sorted_points[i - 1]
            p_curr = sorted_points[i]
            p_next = sorted_points[i + 1]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            # Avoid division by zero
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                curvature = 0
            else:
                # Calculate the angle between the segments.
                dot = np.dot(v1, v2)
                cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
                curvature = np.arccos(cos_angle)
            curvature_points.append(
                (curvature, (int(p_curr[0]), int(p_curr[1]))))
        return curvature_points

    def local_thickness(self, point: tuple) -> float:
        """
        Calculate the local thickness at a given point using the distance transform.
        Lower values indicate a thinner region.

        Parameters:
            point (tuple): The (x, y) coordinates.

        Returns:
            float: The local thickness (distance transform value) at the point.
        """
        x, y = int(point[0]), int(point[1])
        # Note: Using [y, x] indexing because numpy arrays are row-major.
        return self.dist_transform[y, x]

    def label_extreme_points(self) -> dict:
        """
        Label the extreme points on the medial axis as 'head', 'nose', 'tail',
        'tailbase', and 'body_center' by incorporating distance, curvature,
        local thickness, and domain-specific anatomical knowledge.

        Returns:
            dict: A dictionary with keys corresponding to anatomical labels
                  and values as the (x, y) coordinates.
        """
        if not self.extreme_points:
            return {}

        # Select head candidate as the point closest to the centroid.
        head_label = min(
            self.extreme_points,
            key=lambda k: self.distance_to_centroid(self.extreme_points[k])
        )
        head_point = self.extreme_points[head_label]

        # For tail: while conventional logic would take the farthest point,
        # we now incorporate the local thickness.
        # The tail should be long (high distance from head) and thin (low local thickness).
        epsilon = 1e-6  # small constant to avoid division by zero
        candidate_scores = {}
        for label, pt in self.extreme_points.items():
            if label == head_label:
                continue
            # The candidate score favors points that are far from the head relative to their thickness.
            distance = self.calculate_distance(pt, head_point)
            thickness = self.local_thickness(pt)
            candidate_scores[label] = distance / (thickness + epsilon)
        tail_label = max(candidate_scores, key=candidate_scores.get)
        tail_point = self.extreme_points[tail_label]

        # Curvature analysis to refine the nose (typically a sharp turning point near the head).
        # assumes this method is defined as before
        curvature_points = self.compute_medial_axis_curvature()
        if curvature_points:
            nose_candidates = [
                cp for cp in curvature_points
                if self.calculate_distance(cp[1], head_point) < self.calculate_distance(tail_point, head_point) * 0.5
            ]
            if nose_candidates:
                nose_point = max(nose_candidates, key=lambda x: x[0])[1]
            else:
                nose_point = head_point
        else:
            nose_point = head_point

        # Approximate tailbase as the midpoint between head and tail.
        tailbase_point = (
            (head_point[0] + tail_point[0]) / 2,
            (head_point[1] + tail_point[1]) / 2
        )

        # Use the geometric centroid as body_center.
        labels = {
            "head": head_point,
            "nose": nose_point,
            "tail": tail_point,
            "tailbase": tailbase_point,
            "body_center": self.centroid
        }
        return labels

    def ensure_tail_is_farthest(self) -> None:
        """
        Confirm that the 'tail' point is indeed the farthest from the head based on the refined assignment.
        """
        if "head" not in self.labeled_points or "tail" not in self.labeled_points:
            return

        head_pt = self.labeled_points["head"]
        current_tail = self.labeled_points["tail"]
        max_dist = self.calculate_distance(head_pt, current_tail)
        for label, pt in self.labeled_points.items():
            if label in ["head", "tail"]:
                continue
            d = self.calculate_distance(head_pt, pt)
            if d > max_dist:
                max_dist = d
                self.labeled_points["tail"] = pt

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
