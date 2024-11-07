import argparse
import cv2
import glob as gb
import os
import subprocess
import platform
import shutil
import glob
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def construct_filename(results_folder,
                       frame_number,
                       extension="png",
                       padding=9):
    """
    Constructs a filename for saving video frame results.

    Args:
        results_folder (Path or str): The folder where video results are stored.
        frame_number (int): The current frame number.
        extension (str): The file extension for the output file (default is "png").
        padding (int): The number of digits to pad the frame number with (default is 9).

    Returns:
        Path: The constructed filename as a Path object.

    Raises:
        TypeError: If results_folder is not a Path or a valid string.
    """
    if not isinstance(results_folder, Path):
        if isinstance(results_folder, str):
            results_folder = Path(results_folder)
        else:
            raise TypeError(
                "results_folder must be a Path object or a string representing a path.")

    return results_folder / f"{str(results_folder.name)}_{frame_number:0{padding}}.{extension}"


def count_recent_json_files(folder_path, last_minute=1):
    """
    Count the number of JSON files saved in the last minute within a folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        last_minute (int, optional): Number of minutes to consider as "recent". Defaults to 1.

    Returns:
        int: The number of JSON files saved in the last minute.
    """
    # Get the current time
    current_time = datetime.now()
    # Define the threshold for "recent" files based on last_minute
    recent_threshold = current_time - timedelta(minutes=last_minute)

    # Use os.scandir for potentially faster directory iteration
    recent_json_count = 0
    with os.scandir(folder_path) as entries:
        for entry in entries:
            # Check for regular files only (avoid directories and special files)
            if entry.is_file() and entry.name.endswith(".json"):
                # Get file modification time without full path construction
                file_mtime = datetime.fromtimestamp(entry.stat().st_mtime)
                # Check for recent files
                if file_mtime >= recent_threshold:
                    recent_json_count += 1

    return recent_json_count


def find_manual_labeled_json_files(folder_path):
    """
    Find manually labeled JSON files that correspond to PNG
      files in the given folder.

    Args:
        folder_path (str): Path to the folder 
        containing PNG and JSON files.

    Returns:
        list: List of manually labeled JSON filenames.
    """
    manually_labeled_files = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        return manually_labeled_files

    # Get the folder name
    folder_name = os.path.basename(folder_path)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        # Check if file is a PNG and contains the folder name
        if filename.endswith('.png') and folder_name in filename:
            # Construct the corresponding JSON filename
            json_filename = filename.replace('.png', '.json')
            # Check if corresponding JSON file exists
            if os.path.exists(os.path.join(folder_path, json_filename)):
                manually_labeled_files.append(json_filename)

    return manually_labeled_files


def get_frame_number_from_json(json_file):
    # Assume json file name pattern as
    # xxxx_000000000.json
    # Split the file name by '_' and '.'
    parts = json_file.split('_')
    # Extract the part between '_' and '.json'
    frame_number_str = parts[-1].split('.')[0]
    # Convert the frame number string to an integer
    frame_number = int(frame_number_str)
    return frame_number


def count_json_files(folder_path):
    """
    Count the number of JSON files in a given folder.

    Args:
    - folder_path (str): The path to the folder containing JSON files.

    Returns:
    - int: The number of JSON files found in the folder.
    """
    # Initialize a counter for the number of JSON files
    json_file_count = 0

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with ".json"
        if filename.endswith('.json'):
            # Increment the JSON file counter
            json_file_count += 1

    # Return the number of JSON files found in the folder
    return json_file_count


def find_most_recent_file(folder_path, file_ext=".json"):
    # List all files in the folder
    if not os.path.exists(folder_path):
        return
    all_files = os.listdir(folder_path)

    # Filter out directories and get file paths
    file_paths = [os.path.join(folder_path, file) for file in all_files if os.path.isfile(
        os.path.join(folder_path, file)) and file.endswith(file_ext)]

    if not file_paths:
        return None  # No files found

    # Get the most recent file based on modification time
    most_recent_file = max(file_paths, key=os.path.getmtime)

    return most_recent_file


def create_tracking_csv_file(frame_numbers,
                             instance_names,
                             cx_values,
                             cy_values,
                             motion_indices,
                             output_file,
                             fps=30
                             ):
    """
    Create or update a tracking CSV file with the specified columns.

    Args:
    - frame_numbers (list): List of frame numbers.
    - instance_names (list): List of instance names.
    - cx_values (list): List of cx values.
    - cy_values (list): List of cy values.
    - motion_indices (list): List of motion indices.
    - output_file (str): Output CSV file name.
    - fps (int): Frames per second for generating timestamps.
    """
    try:
        # Read existing CSV file into a DataFrame if it exists
        existing_data = pd.read_csv(output_file)
    except FileNotFoundError:
        existing_data = pd.DataFrame()

    # Create a DataFrame for new data
    new_data = {
        'frame_number': frame_numbers,
        'instance_name': instance_names,
        'cx': cx_values,
        'cy': cy_values,
        'motion_index': motion_indices,
    }

    # Append new data to the existing DataFrame
    updated_data = pd.concat([existing_data, pd.DataFrame(new_data)])

    # Remove duplicates from the DataFrame
    updated_data.drop_duplicates(inplace=True)

    # Generate timestamps based on frame numbers and FPS
    timestamps_seconds = pd.to_datetime(
        updated_data['frame_number'] / fps, unit='s')
    updated_data['timestamps'] = timestamps_seconds.dt.time

    # Write the updated DataFrame back to the CSV file without including the index column
    updated_data.to_csv(output_file, index=False)


# Function to clone a Git repository


def clone_git_repository(repo_url, destination_path):
    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # Execute the Git clone command
        subprocess.run(["git", "clone", repo_url, destination_path])

        print("Repository cloned successfully.")
    except Exception as e:
        print(f"Failed to clone repository: {e}")

# Function to download a file


def download_file(url, destination_path):
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            file.write(response.content)
            print("Download complete.")
    else:
        print("Failed to download file.")


def open_or_start_file(file_name):
    # macOS
    if platform.system() == 'Darwin':
        subprocess.call(('open', file_name))
    # Windows
    elif platform.system() == 'Windows':
        os.startfile(file_name)
    # linux
    else:
        subprocess.call(('xdg-open', file_name))


def merge_annotation_folders(
        anno_dir='/data/project_folder/',
        img_pattern="*/*/*.png",
        dest_dir='/data/my_dataset'
):
    """
    merge labeled png and json files in different folders for videos

    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    imgs = glob.glob(os.path.join(anno_dir, img_pattern))

    for img in imgs:
        label_json = img.replace('png', 'json')
        # skip json files predicted by models
        if not os.path.exists(label_json):
            continue
        dest_json = os.path.basename(label_json).replace(' ', '_')
        dest_img = os.path.basename(img).replace(' ', '_')
        shutil.copy(img, os.path.join(dest_dir, dest_img))
        shutil.copy(label_json, os.path.join(dest_dir, dest_json))


def get_freezing_results(results_dir,
                         video_name):
    """check and fliter all the output results files from freezing analyzer.

    Args:
        results_dir (str): path to the result folder
        video_name (str): video name without ext

    Returns:
        list: list of results files
    """
    filtered_results = []
    res_files = os.listdir(results_dir)
    for rf in res_files:
        if video_name in rf:
            if '_results.mp4' in rf:
                filtered_results.append(rf)
            if '_tracked.mp4' in rf:
                filtered_results.append(rf)
            if '_motion.csv' in rf:
                filtered_results.append(rf)
            if 'nix.csv' in rf:
                filtered_results.append(rf)
    return filtered_results


def create_cluster_folders(cluster_labels,
                           dest_dir='/data/video_embidings'):
    """create a subfolder for each cluster

    Args:
        cluster_labels (list): unique cluster labels
        dest_dir (str, optional): root dir for clusters. Defaults to '/data/video_embidings'.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for label in cluster_labels:
        label_folder = os.path.join(dest_dir, f'cluster_{label}')
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)


def create_video_from_images(img_folder,
                             video_name,
                             suffix='png',
                             show_height=540,
                             show_width=960,
                             show_fps=20):
    """Create a video from a folder of images.

    Args:
        img_folder (str): Path to the folder containing the images.
        video_name (str): Name of the output video file.
        suffix (str): Suffix of the image files. Default is 'png'.
        show_height (int): Height of the video display window. Default is 540.
        show_width (int): Width of the video display window. Default is 960.
        show_fps (int): Frames per second of the output video. Default is 20.
    """
    saved_img_paths = gb.glob(img_folder + "/*." + suffix)

    fps = show_fps
    size = (show_width, show_height)

    videowriter = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for saved_img_path in sorted(saved_img_paths):
        img = cv2.imread(saved_img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print('Video is finished.')


def create_gif(img_folder: str, gif_name: str,
               suffix: str, show_height: int,
               show_width: int, show_fps: int,
               start_frame: int = 0, end_frame: int = -1) -> None:
    """
    Create a GIF from a sequence of images.

    Parameters:
    -----------
    img_folder : str
        The directory path of the images.
    gif_name : str
        The name of the GIF file to be created.
    suffix : str
        The file extension of the images.
    show_height : int
        The height of the images to be displayed.
    show_width : int
        The width of the images to be displayed.
    show_fps : int
        The frames per second of the GIF.
    start_frame : int, optional
        The starting index of the image sequence. Default is 0.
    end_frame : int, optional
        The ending index of the image sequence. Default is -1, which means to the end of the sequence.

    Returns:
    --------
    None
    """
    import imageio
    # Get the paths of the images
    saved_img_paths = gb.glob(img_folder + "/*." + suffix)

    # Set the frames per second and size of the images
    fps = show_fps
    size = (show_width, show_height)

    # Get the end frame index if it is not specified
    end_frame = end_frame if end_frame > start_frame else len(saved_img_paths)

    # Load the images and resize them to the specified size
    frames = []
    for img_path in sorted(saved_img_paths)[start_frame:end_frame]:
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        frames.append(img)

    # Save the GIF with the specified parameters
    imageio.mimsave(gif_name, frames, 'GIF', duration=1/fps)
    print('GIF is finished.')


def main():
    """Parse command-line arguments and call create_gifs or videos."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_folder', default=None,
                        type=str, help='Path to folder containing the images.')
    parser.add_argument('--video_name', default=None,
                        type=str, help='Name of the output video file.')
    parser.add_argument('--gif_name', default=None, type=str)
    parser.add_argument('--suffix', default='png', type=str)
    parser.add_argument('--show_height', default=270, type=int)
    parser.add_argument('--show_width', default=480, type=int)
    parser.add_argument('--show_fps', default=20, type=int)
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--end_frame', default=-1, type=int)

    args = parser.parse_args()

    if args.video_name:
        create_video_from_images(args.img_folder, args.video_name, args.suffix,
                                 args.show_height, args.show_width, args.show_fps)

    if args.gif_name:
        create_gif(args.img_folder, args.gif_name, args.suffix, args.show_height,
                   args.show_width, args.show_fps, args.start_frame, args.end_frame)


if __name__ == '__main__':
    main()
