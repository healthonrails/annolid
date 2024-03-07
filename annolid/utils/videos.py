import os
import subprocess


def convert_mkv_to_mp4(input_folder, output_folder):
    # Ensure the output folder exists, create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .mkv files in the input folder
    mkv_files = [f for f in os.listdir(input_folder) if f.endswith('.mkv')]

    for mkv_file in mkv_files:
        input_path = os.path.join(input_folder, mkv_file)
        output_path = os.path.join(
            output_folder, os.path.splitext(mkv_file)[0] + '.mp4')

        # Run ffmpeg command to convert mkv to mp4 and resize
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', 'scale=iw/2:ih/2',
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Converted {input_path} to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {input_path}: {e}")


if __name__ == "__main__":
    input_folder = input("Enter the input folder path containing .mkv files: ")
    output_folder = input(
        "Enter the output folder path for converted .mp4 files: ")
    convert_mkv_to_mp4(input_folder, output_folder)
