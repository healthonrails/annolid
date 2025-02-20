# How to use it?
# python annolid/main.py --labelme2yolo /path/to/labelme_json_folder/ --val_size 0.1 --test_size 0.1
# Refer to https://docs.ultralytics.com/datasets/pose/#dataset-yaml-format for more details.
import json
import math
import os
import numpy as np
import PIL.Image
import shutil
from typing import List, Tuple
from collections import OrderedDict
from labelme.utils.image import img_b64_to_arr
from sklearn.model_selection import train_test_split


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
                 include_visibility=False
                 ):
        self.json_file_dir = json_dir
        self.label_to_id_dict = self.map_label_to_id(self.json_file_dir)
        self.yolo_dataset_name = yolo_dataset_name
        self.annotation_type = "segmentation"
        # e.g. [17, 2] or [17, 3] if visibility is included
        self.kpt_shape = None
        self.include_visibility = include_visibility

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
        train_idxs, val_idxs = train_test_split(range(len(json_names)),
                                                test_size=val_size)
        tmp_train_len = len(train_idxs)
        test_idxs = []
        if test_size is None:
            test_size = 0.0
        if test_size > 1e-8:
            train_idxs, test_idxs = train_test_split(
                range(tmp_train_len), test_size=test_size / (1 - val_size))
        train_jsons = [json_names[train_idx] for train_idx in train_idxs]
        val_jsons = [json_names[val_idx] for val_idx in val_idxs]
        test_jsons = [json_names[test_idx] for test_idx in test_idxs]

        return train_jsons, val_jsons, test_jsons

    @staticmethod
    def map_label_to_id(json_dir: str) -> OrderedDict:
        """
        Get a mapping of label names to unique integer IDs.

        Parameters:
        json_dir (str): The path to the directory containing the annotation files.

        Returns:
        OrderedDict: A dictionary mapping label names to unique integer IDs.
        """
        # Initialize an empty set to store unique labels.
        label_set = set()

        # Iterate through all JSON annotation files in the given directory.
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                # Load the annotation data from the JSON file.
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                # Iterate through all label shapes in the annotation data, adding each label name to the label set.
                for shape in data['shapes']:
                    label_set.add(shape['label'])

        # Use an ordered dictionary to map each unique label name to a unique integer ID.
        return OrderedDict([(label, label_id)
                            for label_id, label in enumerate(label_set)])

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

    def get_yolo_objects(self, json_data, img_path):
        """Return a list of YOLO formatted objects from a JSON annotation file and image.

        Args:
            json_data (dict): JSON data from annotation file.
            img_path (str): Path to image file.

        Returns:
            list: A list of YOLO formatted objects, one for each shape in the annotation file.
        """

        yolo_objects = []
        # Get the height, width of the image
        image_height = json_data['imageHeight']
        image_width = json_data['imageWidth']

        keypoints = []

        # Iterate through each shape in the annotation file
        for shape in json_data["shapes"]:
            # labelme circle has 2 points,
            # the first one is circle center,
            # the second point is the end point
            if shape['shape_type'] == 'circle':
                # Convert the circle shape to a YOLO formatted object
                yolo_obj = self.circle_shape_to_yolo(
                    shape, image_height, image_width)
            elif shape['shape_type'] == 'point':
                keypoints.append(shape.get('points')[0])
            else:
                # Convert the shape to a YOLO formatted object
                yolo_obj = self.scale_points(
                    shape, image_height, image_width)

                yolo_objects.append(yolo_obj)

        if len(keypoints) > 0:
            self.kpt_shape = [len(keypoints), 3] if self.include_visibility else [
                len(keypoints), 2]
            self.annotation_type = "pose"
            keypoint_shape = {
                "label": "keypoints",
                "points": keypoints,
                "group_id": None,
                "shape_type": "pose",
                "flags": {},
                "visible": True,
            }
            yolo_obj = self.scale_points(keypoint_shape,
                                         image_height,
                                         image_width,
                                         output_fromat='pose')
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
        json_data = json.load(open(json_path))

        # Save the image file in the YOLO format.
        img_path = self.save_or_copy_image(
            json_data, json_name, self.image_folder, target_dir)

        # Get a list of YOLO objects from the JSON data and save the output text file.
        yolo_objects = self.get_yolo_objects(json_data, img_path)
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
            if json_data['imageData'] is not None:
                img = img_b64_to_arr(json_data['imageData'])
                PIL.Image.fromarray(img).save(img_path)
            else:
                src_img_path = json_data['imagePath']
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
            if self.annotation_type == "pose":
                # Keypoints
                # kpt_shape: [17, 2] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
                dims = 3 if self.include_visibility else 2
                yaml_file.write(f"kpt_shape: [{self.kpt_shape[0]}, {dims}]\n")
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
