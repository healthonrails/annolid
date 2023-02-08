import json
import numpy as np


def convert_coco_to_dlc(coco_file: str, dlc_file: str) -> None:
    """
    Convert COCO format outputs to DeepLabCut format.

    Parameters
    ----------
    coco_file : str
        The file name of the COCO format outputs.
    dlc_file : str
        The file name to save the DeepLabCut format outputs.

    Returns
    -------
    None

    """
    # Load the COCO format outputs
    with open(coco_file, "r") as f:
        coco_outputs = json.load(f)

    # Initialize an empty list to store the DeepLabCut format outputs
    dlc_outputs = []

    # Iterate over the annotations in the COCO format outputs
    for annotation in coco_outputs["annotations"]:
        # Extract the bounding box coordinates and label
        x = annotation["bbox"][0]
        y = annotation["bbox"][1]
        w = annotation["bbox"][2]
        h = annotation["bbox"][3]
        label = annotation["category_id"]

        # Compute the center of the bounding box
        cx = x + w / 2
        cy = y + h / 2

        # Add the bounding box coordinates and label to the DLC format outputs
        dlc_outputs.append(
            {"frame": annotation["image_id"], "x": cx, "y": cy, "likelihood": 1, "label": label})

    # Save the DLC format outputs to a file
    with open(dlc_file, "w") as f:
        json.dump(dlc_outputs, f)
