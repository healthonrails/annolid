import os
import subprocess
import platform
import shutil
import glob


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
