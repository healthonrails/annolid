import pandas as pd
import json
import os
import cv2
from pathlib import Path


def deeplabcut_to_labelme_json(video_path, output_dir=None, is_multi_animal=False):
    """
    Converts DeepLabCut CSV output (single or multi-animal) to LabelMe JSON files for point annotations.

    Args:
        video_path (str): Path to the video file. The function assumes a CSV file
                           with the same base name exists in the same directory.
        output_dir (str, optional): Directory to save the LabelMe JSON files.
                                     If None, it will create a directory with the same
                                     name as the video file in the same directory.
        is_multi_animal (bool, optional): Set to True if the DeepLabCut analysis is for multi-animal tracking.
                                          Defaults to False (single-animal).
    """

    csv_path = Path(video_path).with_suffix(
        '.csv')  # Infer CSV path from video path
    if not csv_path.exists():
        print(
            f"CSV file not found: {csv_path}. Please ensure a CSV file with the same name as the video exists.")
        return

    # Read multi-index header, frame index as index
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    scorer = df.columns.get_level_values(0)[0]  # Get the scorer name

    if is_multi_animal:
        animal_ids = df.columns.get_level_values('animal').unique().tolist()
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
    else:
        animal_ids = [None]
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()

    # Extract image height and width from the first frame of the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    ret, frame = cap.read()
    if not ret:
        raise Exception(
            f"Could not read the first frame from video: {video_path}")
    image_height, image_width, _ = frame.shape
    cap.release()

    if output_dir is None:
        output_dir = csv_path.with_suffix('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        shapes = []
        for animal_id in animal_ids:
            for bodypart in bodyparts:
                try:
                    if is_multi_animal:
                        # Keep multi-animal as is
                        x_col = (scorer, animal_id, bodypart, 'x')
                        y_col = (scorer, animal_id, bodypart, 'y')
                    else:
                        # Corrected x_col and y_col for single-animal based on KeyError
                        # Try removing 'coords' level
                        x_col = (scorer, bodypart, 'x')
                        # Try removing 'coords' level
                        y_col = (scorer, bodypart, 'y')

                    x = row[x_col]
                    y = row[y_col]

                    if pd.isna(x) or pd.isna(y):
                        continue

                    label = bodypart

                    shape = {
                        "label": label,
                        "points": [[float(x), float(y)]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                        "visible": True,
                    }
                    shapes.append(shape)
                except KeyError as e:  # Capture the exception for debugging
                    # Print full KeyError
                    print(
                        f"KeyError for frame {index}, bodypart {bodypart}, animal {animal_id}: {e}")
                    # Print the keys being tried
                    print(
                        f"  Trying to access column keys: x_col={x_col}, y_col={y_col}")
                    # Print available columns for the row
                    print(f"  Available columns for this row: {row.index}")
                    continue

        image_filename = f""  # 9-digit zero-padded frame number
        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
            "caption": "",
        }

        # 9-digit zero-padded json filename
        output_json_path = os.path.join(output_dir, f"{index:09d}.json")
        with open(output_json_path, 'w') as f:
            json.dump(labelme_json, f, indent=2)

        print(f"Converted frame {index} to LabelMe JSON: {output_json_path}")

    print(f"Conversion complete. LabelMe JSON files saved in '{output_dir}'")


if __name__ == '__main__':
    video_path_single = os.path.expanduser('~/Downloads/92-mouse-2.mp4')
    deeplabcut_to_labelme_json(video_path_single)
    print("Example conversion complete.")
