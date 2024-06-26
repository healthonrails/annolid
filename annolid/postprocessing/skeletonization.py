import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import medial_axis, skeletonize
from annolid.annotation.labelme2binarymasks import LabelMeProcessor
from sklearn.decomposition import PCA


class ShapeProcessor:
    def __init__(self, binary_mask):
        """
        Initialize ShapeProcessor with a binary mask.

        Parameters:
        binary_mask (ndarray): A binary mask representing the shape.
        """
        self.binary_mask = binary_mask
        self.skeleton = skeletonize(self.binary_mask)
        self.medial_axis, self.dist_transform = medial_axis(
            self.binary_mask, return_distance=True)
        self.medial_axis = self.prune_medial_axis(threshold_length=2)
        self.centroid = self.calculate_centroid()
        self.extreme_points = self.calculate_extreme_points_on_medial_axis()
        self.labeled_points = self.label_extreme_points()
        self.ensure_tail_is_farthest()

    def prune_medial_axis(self, threshold_length):
        """
        Prune small branches in the medial axis.

        Parameters:
        threshold_length (int): The length threshold for pruning branches.

        Returns:
        ndarray: The pruned medial axis.
        """
        labeled_skeleton, num_features = label(self.medial_axis)
        sizes = np.bincount(labeled_skeleton.ravel())
        mask_sizes = sizes >= threshold_length
        mask_sizes[0] = 0
        cleaned_skeleton = mask_sizes[labeled_skeleton]
        return cleaned_skeleton

    def calculate_centroid(self):
        """
        Calculate the centroid of the medial axis.

        Returns:
        tuple: Coordinates of the centroid (centroid_x, centroid_y).
        """
        y_coords, x_coords = np.nonzero(self.medial_axis)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return (centroid_x, centroid_y)

    def calculate_extreme_points_on_medial_axis(self):
        """
        Calculate the extreme points (leftmost, rightmost, topmost, bottommost) on the medial axis.

        Returns:
        dict: A dictionary with keys 'leftmost', 'rightmost', 'topmost', 'bottommost' and their respective coordinates.
        """
        y_coords, x_coords = np.nonzero(self.medial_axis)

        if len(y_coords) == 0 or len(x_coords) == 0:
            return None  # Handle case where there are no points in the medial axis

        leftmost = (x_coords[0], y_coords[0])
        rightmost = (x_coords[0], y_coords[0])
        topmost = (x_coords[0], y_coords[0])
        bottommost = (x_coords[0], y_coords[0])

        for x, y in zip(x_coords, y_coords):
            if x < leftmost[0]:
                leftmost = (x, y)
            if x > rightmost[0]:
                rightmost = (x, y)
            if y < topmost[1]:
                topmost = (x, y)
            if y > bottommost[1]:
                bottommost = (x, y)

        return {"leftmost": leftmost, "rightmost": rightmost, "topmost": topmost, "bottommost": bottommost}

    def distance_to_centroid(self, point):
        """
        Calculate the Euclidean distance from a point to the centroid.

        Parameters:
        point (tuple): Coordinates of the point (x, y).

        Returns:
        float: Distance to the centroid.
        """
        return np.sqrt((point[0] - self.centroid[0]) ** 2 + (point[1] - self.centroid[1]) ** 2)

    def calculate_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

        Returns:
        float: Distance between the two points.
        """
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_polygon_edge_distances(self):
        """
        Calculate the coordinates of the polygon edges.

        Returns:
        list: A list of tuples representing the edge points.
        """
        poly_y, poly_x = np.nonzero(self.binary_mask)
        edge_points = list(zip(poly_x, poly_y))
        return edge_points

    def distance_to_polygon_edge(self, point):
        """
        Calculate the minimum distance from a point to the polygon edge.

        Parameters:
        point (tuple): Coordinates of the point (x, y).

        Returns:
        float: Minimum distance to the polygon edge.
        """
        edge_points = self.calculate_polygon_edge_distances()
        distances = [np.sqrt((point[0] - ex) ** 2 + (point[1] - ey) ** 2)
                     for ex, ey in edge_points]
        return min(distances)

    def label_extreme_points(self):
        """
        Label the extreme points on the medial axis as 'head', 'tail', 'nose', 'tailbase', and 'body_center'.

        Returns:
        dict: A dictionary with labeled points.
        """
        # Calculate distances to centroid
        distances = {k: self.distance_to_centroid(
            v) for k, v in self.extreme_points.items()}

        closest_point_label = min(distances, key=distances.get)
        farthest_point_label = max(distances, key=distances.get)

        labels = {}
        labels["head"] = self.extreme_points[closest_point_label]
        labels["tail"] = self.extreme_points[farthest_point_label]

        remaining_points = {k: v for k, v in self.extreme_points.items(
        ) if k not in [closest_point_label, farthest_point_label]}

        # Calculate distances from head to other points
        head_distances = {k: self.calculate_distance(
            v, labels['head']) for k, v in remaining_points.items()}

        # Ensure the "nose" point is near the "head" and close to the border
        min_distance_to_head = min(head_distances.values())
        nose_candidates = [
            point for point, dist in head_distances.items() if dist == min_distance_to_head]
        nose_candidates_dist_to_edge = {point: self.distance_to_polygon_edge(
            self.extreme_points[point]) for point in nose_candidates}
        nose_point_label = min(nose_candidates_dist_to_edge,
                               key=nose_candidates_dist_to_edge.get)

        # Use PCA to determine the primary axis of the shape
        y_coords, x_coords = np.nonzero(self.medial_axis)
        pca = PCA(n_components=2)
        pca.fit(np.column_stack((x_coords, y_coords)))
        primary_axis = pca.components_[0]

        # Find the point closest and farthest to the primary axis
        projected_points = {k: np.dot(primary_axis, np.array(
            v) - np.array(labels['head'])) for k, v in remaining_points.items()}
        tailbase_point_label = max(projected_points, key=projected_points.get)

        labels["nose"] = self.extreme_points[nose_point_label]
        labels["tailbase"] = remaining_points[tailbase_point_label]
        labels["body_center"] = self.centroid

        return labels

    def ensure_tail_is_farthest(self):
        """
        Ensure that the labeled tail point is the farthest from the head point.
        """
        head_point = self.labeled_points["head"]
        tail_point = self.labeled_points["tail"]
        tail_distance = self.calculate_distance(head_point, tail_point)

        # Re-evaluate the points to find the actual farthest point if necessary
        for label, point in self.labeled_points.items():
            if label not in ["head", "tail"]:
                distance = self.calculate_distance(head_point, point)
                if distance > tail_distance:
                    # Update the tail point and the label
                    tail_distance = distance
                    self.labeled_points["tail"] = point

    def visualize(self):
        """
        Visualize the binary mask, medial axis, and labeled points.
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(self.binary_mask, cmap='gray')
        plt.imshow(self.medial_axis, cmap='jet',
                   alpha=0.5)  # Overlay medial axis

        # Plot labeled points on the medial axis
        for label, point in self.labeled_points.items():
            plt.plot(point[0], point[1], 'o', label=label.capitalize())

        plt.title('Medial Axis and Labeled Points')
        plt.legend()
        plt.show()


def compute_distance_transform(binary_image):
    """
    Compute the distance transform of a binary image.

    Parameters:
    binary_image (ndarray): A binary image.

    Returns:
    ndarray: The distance transform of the binary image.
    """
    return distance_transform_edt(binary_image)


def extract_medial_axis(binary_image):
    """
    Extract the medial axis of a binary image.

    Parameters:
    binary_image (ndarray): A binary image.

    Returns:
    ndarray: The medial axis of the binary image.
    """
    medial_axis_result, _ = medial_axis(binary_image, return_distance=True)
    return medial_axis_result


def add_key_points_to_labelme_json(json_path, instance_names, is_vis=False):
    """
    Processes annotations from a LabelMe JSON file, extracts key points, and adds them as annotations.

    Args:
        json_path (str): Path to the LabelMe JSON file.
        instance_names (list): List of instance names to process.

    Returns:
        None: The function directly modifies the JSON file in place.
    """

    # Load the existing JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask_converter = LabelMeProcessor(json_path)
    binary_masks = mask_converter.get_all_masks()

    for bm, label in binary_masks:
        if label in instance_names:
            processor = ShapeProcessor(bm)
            if is_vis:
                processor.visualize()
            # Add key point annotations to the JSON data
            for point_label, point in processor.labeled_points.items():
                data['shapes'].append({
                    "label": f"{label}_{point_label}",
                    "points": [[int(point[0]), int(point[1])]],
                    "shape_type": "point",
                    "visible": True,
                })

    # Save the updated JSON data
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def extract_ground_truth_keypoints(json_path):
    """
    Extract ground truth keypoints from a LabelMe JSON file.

    Parameters:
    json_path (str): Path to the LabelMe JSON file containing ground truth keypoints.

    Returns:
    dict: A dictionary where keys are labels of keypoints and 
    values are the coordinates of those keypoints.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ground_truth_keypoints = {
        shape['label']: shape['points'][0]
        for shape in data['shapes']
    }

    return ground_truth_keypoints


def main(input_folder, instance_names):
    """
    Main function to process annotations and visualize results.

    Parameters:
    - input_folder (str): Path to the folder containing JSON files.
    - instance_names (list): List of instance names.
    """
    # Get all JSON files recursively in the input folder
    json_files = glob.glob(os.path.join(
        input_folder, '**/*.json'), recursive=True)

    # Iterate through each JSON file
    for i, json_file in enumerate(json_files):
        if i % 1000 == 0:
            print('Finding keypoints in ', json_file)
        add_key_points_to_labelme_json(json_file, instance_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add key points to LabelMe JSON files.")
    parser.add_argument(
        "input_folder", help="Path to the folder containing JSON files.")
    parser.add_argument("instance_names", nargs="+",
                        help="List of instance names.")
    args = parser.parse_args()

    main(args.input_folder, args.instance_names)
