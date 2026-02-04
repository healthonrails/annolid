import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from annolid.utils.annotation_store import load_labelme_json


class LabelMeProcessor:
    """
    A class to process and visualize LabelMe JSON annotations.

    Attributes:
        json_path (str): Path to the LabelMe JSON file.
        data (dict): The JSON data loaded from the file.
        image_shape (tuple): The shape of the image (height, width).
        polygons (list): List of tuples containing polygon points and their labels.
    """

    def __init__(self, json_path):
        """
        Initializes the LabelMeProcessor with a path to the JSON file.

        Args:
            json_path (str): Path to the LabelMe JSON file.
        """
        self.json_path = json_path
        self.data = self._load_labelme_json()
        self.image_shape = self._extract_image_shape()
        self.polygons = self._extract_polygons()

    def _load_labelme_json(self):
        """
        Loads the LabelMe JSON data from the specified file.

        Returns:
            dict: The JSON data loaded from the file.
        """
        return load_labelme_json(self.json_path)

    def _extract_image_shape(self):
        """
        Extracts the image shape from the JSON data.

        Returns:
            tuple: The shape of the image (height, width).
        """
        image_height = self.data["imageHeight"]
        image_width = self.data["imageWidth"]
        return (image_height, image_width)

    def _extract_polygons(self):
        """
        Extracts the polygon points and their labels from the JSON data.

        Returns:
            list: List of tuples containing polygon points and their labels.
        """
        polygons = [
            (shape["points"], shape["label"])
            for shape in self.data["shapes"]
            if shape["shape_type"] == "polygon"
        ]
        return polygons

    def create_binary_mask(self, polygon_points):
        """
        Creates a binary mask for the given polygon points.

        Args:
            polygon_points (list): List of points defining the polygon.

        Returns:
            np.ndarray: Binary mask of the polygon.
        """
        mask = np.zeros(self.image_shape, dtype=np.uint8)
        poly_x = [point[0] for point in polygon_points]
        poly_y = [point[1] for point in polygon_points]
        rr, cc = polygon(poly_y, poly_x, self.image_shape)
        mask[rr, cc] = 1
        return mask

    def get_all_masks(self):
        """
        Generates binary masks for all polygons in the JSON data.

        Returns:
            list: List of tuples containing binary masks and their labels.
        """
        masks = [
            (self.create_binary_mask(polygon_points), label)
            for polygon_points, label in self.polygons
        ]
        return masks

    def visualize_masks(self, masks):
        """
        Visualizes all binary masks.

        Args:
            masks (list): List of binary masks to visualize.
        """
        for i, (mask, label) in enumerate(masks):
            plt.figure(figsize=(10, 5))
            plt.imshow(mask, cmap="gray")
            plt.title(f"Binary Mask {i + 1}: {label}")
        plt.show()


# Example usage:
if __name__ == "__main__":
    json_path = "path/to/labelme.json"

    processor = LabelMeProcessor(json_path)
    masks = processor.get_all_masks()
    processor.visualize_masks(masks)
