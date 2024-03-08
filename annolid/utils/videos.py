import cv2
import os
import subprocess
import csv
import argparse


def frame_to_timestamp(frames, fps):
    timestamps = []
    for frame in frames:
        timestamp = frame / fps
        timestamps.append(timestamp)
    return timestamps


def collect_video_metadata(input_folder):
    """
    Collects metadata for each video file in the specified input folder.

    Args:
    - input_folder (str): Path to the folder containing video files.

    Returns:
    - metadata (list of dicts): List of dictionaries containing metadata for each video.
    """
    metadata = []

    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return metadata

    video_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print("No video files found in the input folder.")
        return metadata

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cap.get(cv2.CAP_PROP_FOURCC)

        metadata_entry = {
            'video_name': video_file,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'codec': codec
        }

        metadata.append(metadata_entry)

        print(f"Collected metadata for {video_file}")

        cap.release()

    return metadata


def save_metadata_to_csv(metadata, output_csv):
    """
    Saves video metadata to a CSV file.

    Args:
    - metadata (list of dicts): List of dictionaries containing video metadata.
    - output_csv (str): Path to the output CSV file.
    """
    if not metadata:
        print("No metadata to save.")
        return

    fieldnames = ['video_name', 'width',
                  'height', 'fps', 'frame_count', 'codec']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def compress_and_rescale_video(input_folder, output_folder, scale_factor):
    """
    Compresses and rescales video files in the input folder using ffmpeg.

    Args:
    - input_folder (str): Path to the folder containing video files.
    - output_folder (str): Path to the folder for compressed and rescaled video files.
    - scale_factor (float): Scale factor for resizing videos.
    """
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

    if not video_files:
        print("No video files found in the input folder.")
        return

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)

        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale=iw*{scale_factor}:ih*{scale_factor}',
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Compressed and rescaled {video_file} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error compressing and rescaling {video_file}: {e}")


def main(args):
    """
    Examples 
    Suppose you have some .mkv files in a folder named input_videos, 
    and you want to convert them to .mp4 format with a scale factor of 0.5 for resizing.
      Additionally, you want to save the metadata to a CSV file named video_metadata.csv. 
      Here's how you would run the script:
    ```python script.py input_videos --output_folder output_videos - 
        -output_csv video_metadata.csv --scale_factor 0.5```
    In this command:
    input_videos is the input folder containing the .mkv files.
    --output_folder output_videos specifies the output folder 
    where the converted .mp4 files will be saved.
    --output_csv video_metadata.csv specifies the output CSV file path for storing the metadata.
    --scale_factor 0.5 specifies the scale factor for resizing the videos during conversion.

    If you only want to collect metadata without performing the conversion, 
    you can add the --collect_only flag:
    ```python script.py input_videos --output_csv video_metadata.csv --collect_only```
    This command will collect metadata for the .mkv files in the input_videos folder and 
    save it to video_metadata.csv. No conversion will be performed in this case.

    """
    if args.collect_only:
        metadata = collect_video_metadata(args.input_folder)
        if args.output_csv:
            save_metadata_to_csv(metadata, args.output_csv)
    else:
        compress_and_rescale_video(
            args.input_folder, args.output_folder, args.scale_factor)
        if not args.collect_only and args.output_csv:
            metadata = collect_video_metadata(args.output_folder)
            save_metadata_to_csv(metadata, args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compress and rescale video files and collect metadata.')
    parser.add_argument('input_folder', type=str,
                        help='Input folder path containing video files.')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder path for compressed and rescaled video files.')
    parser.add_argument('--output_csv', type=str,
                        help='Output CSV file path for metadata.')
    parser.add_argument('--scale_factor', type=float,
                        default=1.0, help='Scale factor for resizing videos.')
    parser.add_argument('--collect_only', action='store_true',
                        help='Collect metadata only, do not compress and rescale.')

    args = parser.parse_args()
    main(args)
