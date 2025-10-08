import csv
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from labelme.utils import shape_to_mask
from annolid.utils.shapes import masks_to_bboxes, polygon_center
from annolid.annotation.masks import binary_mask_to_coco_rle
from annolid.annotation.keypoints import keypoint_to_polygon_points
from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)


def read_json_file(file_path):
    """
    Read JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: JSON data.
    """
    try:
        return load_labelme_json(file_path)
    except (FileNotFoundError, AnnotationStoreError, ValueError):
        return {}


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


def convert_json_to_csv(json_folder, csv_file=None, progress_callback=None):
    """
    Convert JSON files in a folder to annolid CSV format file.

    Parameters:
    - json_folder (str): Path to the folder containing JSON files.
    - csv_file (str): Output CSV file.
    - progress_callback (function): Callback function to report progress.


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

        json_folder_path = Path(json_folder)
        json_files = sorted(
            f for f in os.listdir(json_folder) if f.endswith('.json')
        )

        store = AnnotationStore.for_frame_path(
            json_folder_path / f"{json_folder_path.name}_000000000.json")
        if store.store_path.exists():
            store_files = [
                f"{json_folder_path.name}_{frame:09d}.json"
                for frame in sorted(store.iter_frames())
            ]
            json_files = sorted(set(json_files) | set(store_files))

        if not json_files:
            return f"No annotation files found in {json_folder}."

        total_files = len(json_files)
        num_processed_files = 0

        for json_file in tqdm(json_files, desc='Converting JSON files', unit='files'):
            json_path = os.path.join(json_folder, json_file)
            data = read_json_file(json_path)
            if not data:
                num_processed_files += 1
                if progress_callback:
                    progress = int((num_processed_files / total_files) * 100)
                    progress_callback(progress)
                continue

            frame_number = get_frame_number_from_filename(json_file)
            img_height = data.get("imageHeight")
            img_width = data.get("imageWidth")
            shapes = data.get("shapes") or []

            if img_height is None or img_width is None:
                num_processed_files += 1
                if progress_callback:
                    progress = int((num_processed_files / total_files) * 100)
                    progress_callback(progress)
                continue

            img_shape = (img_height, img_width)

            for shape in shapes:
                instance_name = shape.get("label")
                points = shape.get("points") or []
                if shape.get('shape_type') == 'point':
                    points = keypoint_to_polygon_points(points)
                    shape['shape_type'] = 'polygon'
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
            num_processed_files += 1
            if progress_callback:
                progress = int((num_processed_files / total_files) * 100)
                progress_callback(progress)

    return str(csv_file)


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
