import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.draw import polygon_perimeter
from annolid.annotation.labelme2binarymasks import LabelMeProcessor
from skimage.morphology import skeletonize


class ShapeProcessor:
    def __init__(self, binary_mask):
        """
        Initialize ShapeProcessor with a binary mask.

        Parameters:
        binary_mask (ndarray): A binary mask representing the shape.
        """
        self.binary_mask = binary_mask
        self.medial_axis, self.dist_transform = medial_axis(
            self.binary_mask, return_distance=True)
        self.centroid = self.calculate_centroid()
        self.extreme_points = self.calculate_extreme_points_on_medial_axis()
        self.labeled_points = self.label_extreme_points()

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
        distances = {k: self.distance_to_centroid(
            v) for k, v in self.extreme_points.items()}

        closest_point_label = min(distances, key=distances.get)
        farthest_point_label = max(distances, key=distances.get)

        labels = {}
        labels["head"] = self.extreme_points[closest_point_label]
        labels["tail"] = self.extreme_points[farthest_point_label]

        remaining_points = {k: v for k, v in self.extreme_points.items(
        ) if k not in [closest_point_label, farthest_point_label]}

        edge_distances = {k: self.calculate_distance(
            v, labels['head']) for k, v in remaining_points.items()}
        tailbase_point_label = max(edge_distances, key=edge_distances.get)
        nose_point_label = min(edge_distances, key=edge_distances.get)

        labels["nose"] = remaining_points[nose_point_label]
        labels["tailbase"] = remaining_points[tailbase_point_label]
        labels["body_center"] = self.centroid

        return labels

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


def process_annotation(json_path, instance_names):
    """
    Process annotations from a LabelMe JSON file and visualize the shape processing.

    Parameters:
    json_path (str): Path to the LabelMe JSON file.
    instance_names (list): List of instance names to process.
    """
    mask_converter = LabelMeProcessor(json_path)
    binary_masks = mask_converter.get_all_masks()

    for bm, label in binary_masks:
        if label in instance_names:
            processor = ShapeProcessor(bm)
            processor.visualize()
            print("Instance:", label)
            print("Centroid:", processor.centroid)
            print("Labeled Points on Medial Axis:", processor.labeled_points)


def main():
    """
    Main function to process annotations and visualize results.
    """
    json_path = 'mouse_000000000.json'
    # Update with the desired instance names
    instance_names = ['rat', 'mouse']
    process_annotation(json_path, instance_names)


if __name__ == "__main__":
    main()
