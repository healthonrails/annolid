import os
import glob
import shutil
from pathlib import Path
from PIL import Image


def detect_resolution(image_path):
    """
    Detects the resolution of the given image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Resolution in the format '{height}p'.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return f"{height}p"


def convert_to_davis_format(input_folder, output_folder):
    """
    Converts video frames and their corresponding masks from a custom format to the DAVIS format.

    The custom format assumes video frames and masks are stored in separate subfolders,
    with masks having filenames ending in '_mask.png'.

    Args:
        input_folder (str): Path to the input folder containing video subfolders.
        output_folder (str): Path to the output folder where the DAVIS formatted data will be saved.
    """
    # Find all video subfolders in the input folder
    video_folders = [f for f in Path(input_folder).iterdir() if f.is_dir()]

    for video_folder in video_folders:
        # Get all mask files in the current video folder
        mask_files = glob.glob(os.path.join(video_folder, '*_mask.png'))
        for mask_file in mask_files:
            mask_file_name = os.path.basename(mask_file)
            img_file_name = mask_file_name.replace('_mask', '')

            # Detect resolution from the image file
            img_path = os.path.join(video_folder, img_file_name)
            resolution = detect_resolution(img_path)

            # Define paths for the DAVIS structure using pathlib
            jpeg_images_folder = Path(
                output_folder) / 'JPEGImages' / resolution
            annotations_folder = Path(
                output_folder) / 'Annotations' / resolution

            # Create directories if they don't exist
            jpeg_images_folder.mkdir(parents=True, exist_ok=True)
            annotations_folder.mkdir(parents=True, exist_ok=True)

            # Extract the base name without extension
            base_name = img_file_name.removesuffix('.png')

            # Extract the video_name (folder name)
            video_name = video_folder.name

            # Extract the frame number (everything after the last underscore)
            frame_number = base_name.rsplit('_', 1)[1]

            # Create directories for the video sequence
            video_jpeg_folder = jpeg_images_folder / video_name
            video_annotation_folder = annotations_folder / video_name
            video_jpeg_folder.mkdir(exist_ok=True)
            video_annotation_folder.mkdir(exist_ok=True)

            # Define source and destination paths using pathlib
            src_image_path = Path(img_path)
            dst_image_path = video_jpeg_folder / f"{frame_number}.jpg"

            src_mask_path = Path(mask_file)
            dst_mask_path = video_annotation_folder / f"{frame_number}.png"

            # Copy image and mask to the destination
            try:
                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_mask_path, dst_mask_path)
            except Exception as e:
                print(f"Error copying files: {e}")

    print(f"Conversion to DAVIS format completed in {output_folder}")


if __name__ == '__main__':
    # Example usage
    input_folder = '/path/to/Freezing'
    output_folder = '/path/to/Freezing/mouse_parts'

    convert_to_davis_format(input_folder, output_folder)
