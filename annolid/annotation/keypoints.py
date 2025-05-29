import os
import cv2
import numpy as np
import pandas as pd
from annolid.gui.label_file import LabelFile
from annolid.gui.shape import Shape
from annolid.utils.logger import logger
import json
import math


def keypoint_to_polygon_points(center_point,
                               radius=10,
                               num_points=10):
    """
    Generate polygon points based on a given point and radius.

    Args:
    - center_point (list): A list containing the x and y coordinates
      of the center point, e.g., [[x, y]].
    - radius (int): The radius of the circle.
    - num_points (int, optional): The number of points to generate (default is 10).
˝
    Returns:
    - points (list): A list of lists containing the x and y coordinates of
      the polygon points, e.g., [[x1, y1], [x2, y2], ...].
    """
    center_x, center_y = center_point[0]
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append([x, y])
    return points


def format_shape(shape):
    data = shape.other_data.copy()
    data.update({
        'label': shape.label,
        'points': shape.points,
        'group_id': shape.group_id,
        'shape_type': shape.shape_type,
        'flags': shape.flags,
        'visible': shape.visible,
        'description': shape.description,
    })
    return data


def load_existing_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return None


def merge_shapes(new_shapes_list, existing_shapes_list):
    """
    Merges new shapes with existing shapes.
    If a new shape has the same label as an existing shape,
    the existing shape is replaced by the new one.
    Otherwise, the new shape is added.

    Args:
        new_shapes_list (list): List of new shape dictionaries.
        existing_shapes_list (list): List of existing shape dictionaries.

    Returns:
        list: A list of merged shape dictionaries.
    """
    # Create a dictionary of existing shapes for efficient lookup and modification
    # Assuming 'label' is unique or you want to replace based on the first match found
    # If labels are not unique and you need more complex logic, this needs adjustment.
    merged_shapes_dict = {
        shape_data['label']: shape_data for shape_data in existing_shapes_list}

    for new_shape_data in new_shapes_list:
        label = new_shape_data['label']
        # If label exists, replace; otherwise, add.
        merged_shapes_dict[label] = new_shape_data

    # Convert the dictionary back to a list
    return list(merged_shapes_dict.values())


def save_labels(filename, imagePath,
                label_list,
                height,
                width,
                imageData=None,
                otherData=None,
                save_image_to_json=False,
                flags=None,
                caption=None,
                ):
    """Save a list of labeled shapes to a JSON file.

    Args:
        filename (str): JSON file name.
        imagePath (str): Image file path.
        label_list (list): List of labeled shapes.
        height (int): Image height.
        width (int): Image width.
        imageData (optional): Image data. Defaults to None.
        otherData (optional): Other data. Defaults to None.
        save_image_to_json (bool, optional): 
        Whether to save image data to JSON. Defaults to False.
    """
    # Check if a PNG file exists with the same name
    png_filename = os.path.splitext(filename)[0] + ".png"
    json_filename = png_filename.replace('.png', '.json')
    if os.path.exists(png_filename) and os.path.exists(json_filename):
        logger.info(
            """A corresponding PNG file was found. 
            We assume the frame has been manually labeled.
            No changes are needed for the JSON file.""")
        return
    lf = LabelFile()
    shapes = [format_shape(shape) for shape in label_list]

    # Load existing shapes from the JSON file and merge with new shapes
    json_data = load_existing_json(filename)
    existing_shapes = json_data.get('shapes', []) if json_data else []
    if flags is None:
        flags = json_data.get('flags', {}) if json_data else {}

    # shapes.extend(existing_shapes)
    shapes = merge_shapes(shapes, existing_shapes)

    # Load image data if necessary
    if imageData is None and save_image_to_json:
        imageData = LabelFile.load_image_file(imagePath)

    # Set default value for otherData
    if otherData is None:
        otherData = {}

    if flags is not None:
        flags = flags
    else:
        flags = {}

    # Save data to JSON file
    lf.save(
        filename=filename,
        shapes=shapes,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=height,
        imageWidth=width,
        otherData=otherData,
        flags=flags,
        caption=caption,
    )


def to_labelme(img_folder,
               anno_file,
               multiple_animals=True
               ):
    """Convert keypoints format to labelme json files.

    Args:
        img_folder (path): folder where images are located
        anno_file (h5): keypoints annotation file.
        multiple_animals: labeled with multiple animals(default True)
    """
    df = pd.read_hdf(os.path.join(img_folder, anno_file))
    scorer = df.columns.get_level_values(0)[0]
    if multiple_animals:
        individuals = df.columns.get_level_values(1)
        bodyparts = df.columns.get_level_values(2)

        for ind, imname in enumerate(df.index):
            img_path = os.path.join(img_folder, imname)
            image = cv2.imread(img_path)
            ny, nx, nc = np.shape(image)
            image_height = ny
            image_width = nx
            label_list = []
            for idv in set(individuals):
                for b in set(bodyparts):
                    s = Shape(label=f"{idv}_{b}", shape_type='point', flags={})
                    x = df.iloc[ind][scorer][idv][b]['x']
                    y = df.iloc[ind][scorer][idv][b]['y']
                    if np.isfinite(x) and np.isfinite(y):
                        s.addPoint((x, y))
                        label_list.append(s)
            save_labels(img_path.replace('.png', '.json'),
                        img_path, label_list, image_height,
                        image_width)
    else:
        bodyparts = df.columns.get_level_values(1)

        for ind, imname in enumerate(df.index):
            img_path = os.path.join(img_folder, imname)
            print(img_folder, imname)
            image = cv2.imread(img_path)
            try:
                ny, nx, nc = np.shape(image)
            except:
                ny, ny = np.shape(image)
            image_height = ny
            image_width = nx
            label_list = []

            for b in set(bodyparts):
                s = Shape(label=f"{b}", shape_type='point', flags={})
                x = df.iloc[ind][scorer][b]['x']
                y = df.iloc[ind][scorer][b]['y']
                if np.isfinite(x) and np.isfinite(y):
                    s.addPoint((x, y))
                    label_list.append(s)
            save_labels(img_path.replace('.png', '.json'),
                        img_path, label_list, image_height,
                        image_width)


def get_shapes(points,
               label_name,
               scale_factor=224/512):
    """
    Convert 2d points with label name to polygon shapes
    """
    shape = Shape(label=label_name,
                  shape_type='polygon',
                  flags={}
                  )
    for x, y in points * scale_factor:
        if x > 0 and y > 0:
            shape.addPoint((int(y), int(x)))
    return shape


def calc_pck(gt_keypoints, est_keypoints, threshold=0.3):
    """The PCK (Percentage of Correct Keypoints) metric is a common evaluation metric 
    used to assess the accuracy of pose estimation algorithms. 
    It measures the percentage of keypoints (joints) whose projection error is 
    less than a certain threshold (default 0.3). 
    The projection error is the Euclidean distance between 
    the ground truth keypoint location and the estimated keypoint location in the image plane.

    Args:
        gt_keypoints (ndarray): ground truth keypoints (num_keypoints, 2)
        est_keypoints (ndarray): estimated keypoints 
        threshold (float, optional): threshold. Defaults to 0.3.

    Returns:
        float: the PCK@threshold score
    """
    num_keypoints = gt_keypoints.shape[0]
    pck = np.zeros(num_keypoints)
    for i in range(num_keypoints):
        gt_kp = gt_keypoints[i]
        est_kp = est_keypoints[i]
        error = np.linalg.norm(gt_kp - est_kp)
        if error < threshold:
            pck[i] = 1
    return np.mean(pck)
