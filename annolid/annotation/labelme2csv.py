import json
import csv
import os
import argparse
from labelme.utils import shape_to_mask
from annolid.utils.shapes import masks_to_bboxes, polygon_center
from annolid.annotation.masks import binary_mask_to_coco_rle


def read_json_file(file_path):
    """
    Read JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def get_frame_number_from_filename(file_name):
    """
    Extract frame number from a JSON file name.

    Parameters:
    - file_name (str): JSON file name.

    Returns:
    - int: Extracted frame number.
    """
    return int(file_name.split('_')[-1].replace('.json', ''))


def convert_shape_to_mask(img_shape, points):
    """
    Convert shape points to a binary mask.

    Parameters:
    - img_shape (tuple): Image shape (width, height).
    - points (list): List of points.

    Returns:
    - ndarray: Binary mask.
    """
    mask = shape_to_mask(img_shape, points, shape_type=None,
                         line_width=10, point_size=5)
    return mask


def convert_json_to_csv(json_folder, csv_file=None):
    """
    Convert JSON files in a folder to annolid CSV format file.

    Parameters:
    - json_folder (str): Path to the folder containing JSON files.
    - csv_file (str): Output CSV file.

    Returns:
    - None
    """
    csv_header = ["frame_number", "x1", "y1", "x2", "y2", "cx", "cy",
                  "instance_name", "class_score", "segmentation", "tracking_id"]

    if csv_file is None:
        csv_file = json_folder + '_tracking.csv'

    with open(csv_file, 'w', newline='') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(csv_header)

        for json_file in os.listdir(json_folder):
            if json_file.endswith(".json"):
                data = read_json_file(os.path.join(json_folder, json_file))
                frame_number = get_frame_number_from_filename(json_file)
                img_height, img_width = data["imageHeight"], data["imageWidth"]
                img_shape = (img_height, img_width)

                for shape in data["shapes"]:
                    instance_name = shape["label"]
                    points = shape["points"]
                    if len(points) > 2:
                        mask = convert_shape_to_mask(img_shape, points)
                        bboxs = masks_to_bboxes(mask[None, :, :])
                        segmentation = binary_mask_to_coco_rle(mask)
                        class_score = 1.0

                        if len(bboxs) > 0:
                            cx, cy = polygon_center(points)
                            x1, y1, x2, y2 = bboxs[0]
                            csv_writer.writerow(
                                [frame_number, x1, y1, x2, y2, cx, cy,
                                 instance_name, class_score,
                                 segmentation, 0])

    print(f"CSV file '{csv_file}' has been created.")


def main():
    """
    Main function to parse command-line arguments and execute the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert JSON files to a CSV file.")
    parser.add_argument("--json_folder", required=True,
                        help="Path to the folder containing JSON files.")
    parser.add_argument("--csv_file", required=False, help="Output CSV file.")

    args = parser.parse_args()
    convert_json_to_csv(args.json_folder, args.csv_file)


if __name__ == "__main__":
    main()
