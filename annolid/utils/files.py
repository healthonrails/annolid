import argparse
import cv2
import glob as gb
import os
import subprocess
import platform
import shutil
import glob
import imageio


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
        assert os.path.exists(label_json)
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
        img = imageio.imread(img_path)
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
