import os
import glob
import shutil
from pathlib import Path
from PIL import Image
import random


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


def create_davis_structure(output_folder,
                           dataset_name,
                           dataset_year):
    """
    Creates the folder structure for the DAVIS dataset.

    Args:
        output_folder (str): Path to the output folder 
        where the DAVIS structure will be created.
        dataset_name (str): Name of the dataset.
        dataset_year (str): Year of the dataset.
    """
    dataset_folder = Path(output_folder) / dataset_name / dataset_year
    trainval_folder = dataset_folder / 'trainval'
    test_dev_folder = dataset_folder / 'test-dev'

    for subfolder in ['Annotations', 'JPEGImages']:
        for main_folder in [trainval_folder, test_dev_folder]:
            (main_folder / subfolder).mkdir(parents=True, exist_ok=True)

    return trainval_folder, test_dev_folder


def copy_files(mask_file, img_file_name,
               video_folder, resolution,
               output_subfolder):
    """
    Copies image and mask files to the appropriate location in the DAVIS structure.

    Args:
        mask_file (str): Path to the mask file.
        img_file_name (str): Name of the corresponding image file.
        video_folder (str): Path to the video folder containing the files.
        resolution (str): Resolution of the files.
        output_subfolder (str): Path to the output subfolder (trainval or test-dev).
    """
    img_path = os.path.join(video_folder, img_file_name)
    jpeg_images_folder = output_subfolder / 'JPEGImages' / resolution
    annotations_folder = output_subfolder / 'Annotations' / resolution

    # Create directories if they don't exist
    jpeg_images_folder.mkdir(parents=True, exist_ok=True)
    annotations_folder.mkdir(parents=True, exist_ok=True)

    base_name = img_file_name.removesuffix('.png')
    video_name = Path(video_folder).name
    frame_number = base_name.rsplit('_', 1)[1]

    video_jpeg_folder = jpeg_images_folder / video_name
    video_annotation_folder = annotations_folder / video_name
    video_jpeg_folder.mkdir(exist_ok=True)
    video_annotation_folder.mkdir(exist_ok=True)

    src_image_path = Path(img_path)
    dst_image_path = video_jpeg_folder / f"{frame_number}.jpg"
    src_mask_path = Path(mask_file)
    dst_mask_path = video_annotation_folder / f"{frame_number}.png"

    try:
        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_mask_path, dst_mask_path)
    except Exception as e:
        print(f"Error copying files: {e}")


def process_video_folder(video_folder, output_subfolder):
    """
    Processes a single video folder, copying its files 
    to the appropriate location in the DAVIS structure.

    Args:
        video_folder (str): Path to the video folder to process.
        output_subfolder (str): Path to the output subfolder (trainval or test-dev).
    """
    mask_files = glob.glob(os.path.join(video_folder, '*_mask.png'))
    for mask_file in mask_files:
        mask_file_name = os.path.basename(mask_file)
        img_file_name = mask_file_name.replace('_mask', '')

        resolution = detect_resolution(
            os.path.join(video_folder, img_file_name))
        copy_files(mask_file, img_file_name, video_folder,
                   resolution, output_subfolder)


def split_video_folders(video_folders, trainval_ratio=0.8):
    """
    Splits video folders into training and testing sets.

    Args:
        video_folders (list): List of video folder paths.
        trainval_ratio (float): Ratio of the data to be used for training/validation.

    Returns:
        tuple: Two lists of video folders, one for training/validation and one for testing.
    """
    random.shuffle(video_folders)
    split_index = int(len(video_folders) * trainval_ratio)
    trainval_folders = video_folders[:split_index]
    test_dev_folders = video_folders[split_index:]
    return trainval_folders, test_dev_folders


def convert_to_davis_format(input_folder, output_folder,
                            dataset_name='DAVIS',
                            dataset_year='2017',
                            trainval_ratio=0.8):
    """
    Converts video frames and their corresponding masks from a custom format to the DAVIS format.

    Args:
        input_folder (str): Path to the input folder containing video subfolders.
        output_folder (str): Path to the output folder where the DAVIS formatted data will be saved.
        dataset_name (str): Name of the dataset (default is 'DAVIS').
        dataset_year (str): Year of the dataset (default is '2017').
        trainval_ratio (float): Ratio of the data to be used for training/validation.
    """
    trainval_folder, test_dev_folder = create_davis_structure(
        output_folder, dataset_name, dataset_year)
    video_folders = [f for f in Path(input_folder).iterdir() if f.is_dir()]

    trainval_folders, test_dev_folders = split_video_folders(
        video_folders, trainval_ratio)

    for video_folder in trainval_folders:
        process_video_folder(video_folder, trainval_folder)

    for video_folder in test_dev_folders:
        process_video_folder(video_folder, test_dev_folder)

    print(f"Conversion to {dataset_name} format completed in {output_folder}")


if __name__ == '__main__':
    # Example usage
    input_folder = '.'
    output_folder = 'animals_dataset'

    convert_to_davis_format(input_folder, output_folder)
