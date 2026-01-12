import os
import cv2
import argparse


def find_tiff_files(folder):
    tiff_files = []
    for filename in os.listdir(folder):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            tiff_files.append(os.path.join(folder, filename))
    return sorted(tiff_files)


def scale_with_contrast(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply histogram equalization
    equ = cv2.equalizeHist(gray)

    # Convert back to BGR if it was originally color
    if len(image.shape) == 3:
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

    return equ


def main(input_folder, output_video):
    # Find and sort TIFF files
    tiff_files = find_tiff_files(input_folder)

    # Open the first TIFF file to get image dimensions
    first_image = cv2.imread(tiff_files[0], cv2.IMREAD_UNCHANGED)
    height, width = first_image.shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 24.0, (width, height))

    # Process each TIFF file and write to video
    for tiff_file in tiff_files:
        # Open and process the image
        image = cv2.imread(tiff_file)
        image_scaled = scale_with_contrast(image)

        # Write the frame to the video
        out.write(image_scaled)

    # Release VideoWriter
    out.release()


if __name__ == '__main__':
    """ python tiff2video.py /Fluo-N2DL-HeLa/01 /Fluo-N2DL-HeLa/01.mp4
    """
    parser = argparse.ArgumentParser(
        description='Convert TIFF files to a video with contrast enhancement')
    parser.add_argument('input_folder', type=str,
                        help='Path to the folder containing TIFF files')
    parser.add_argument('output_video', type=str,
                        help='Path to the output video file (MP4)')
    args = parser.parse_args()

    main(args.input_folder, args.output_video)
