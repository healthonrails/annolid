import os
import re
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def create_figure_from_images(folder, rows=2, cols=3, output_file=None, dpi=300):
    """
    Create and save a figure composed of evenly sampled PNG images from a folder.

    Parameters:
        folder (str): Path to the folder containing PNG images.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        output_file (str): Output file path for the saved figure. If None, defaults to '<folder>_figure.png'.
        dpi (int): Dots per inch (DPI) for the saved figure.
    """
    # Get and sort all PNG files in the folder (case insensitive)
    image_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.png')
    ])

    total_files = len(image_files)
    if total_files == 0:
        raise ValueError("No PNG files found in the specified folder.")

    grid_size = rows * cols

    # Evenly sample images if more than grid cells are available
    if total_files > grid_size:
        indices = np.linspace(0, total_files - 1, grid_size, dtype=int)
        sampled_files = [image_files[i] for i in indices]
    else:
        sampled_files = image_files

    # Open images
    images = [Image.open(img) for img in sampled_files]

    # Create figure and subplots using constrained_layout for a clean, professional look
    fig, axes = plt.subplots(rows, cols, figsize=(
        15, 10), constrained_layout=True)
    axes = axes.flatten()

    # Annotate each subplot with a panel label (either frame number or alphabetical)
    for idx, (ax, img, fname) in enumerate(zip(axes, images, sampled_files)):
        ax.imshow(img)
        ax.axis("off")

        base = os.path.basename(fname)
        # Extract all digit groups from the filename
        digit_groups = re.findall(r'(\d+)', base)
        if digit_groups:
            # Use the last group and convert to int to remove leading zeros
            frame_number = str(int(digit_groups[-1]))
            panel_label = f"Frame: {frame_number}"
        else:
            # fallback to alphabetical labeling
            panel_label = f"({chr(97 + idx)})"

        # Place the panel label in the top-left corner with a background for readability
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='white', verticalalignment='top',
                bbox=dict(facecolor='black', edgecolor='none', pad=3))

    # Hide any remaining unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    # Set default output file name if not provided
    if output_file is None:
        output_file = f"{os.path.basename(os.path.normpath(folder))}_figure.png"

    # Save and display the figure
    plt.savefig(output_file, dpi=dpi)
    plt.show()

    print(f"Figure saved as {output_file}")
    print(f"Total number of PNG files found: {total_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a figure from evenly sampled PNG images in a folder."
    )
    parser.add_argument("folder", help="Folder containing PNG images.")
    parser.add_argument("--rows", type=int, default=2,
                        help="Number of rows in the grid (default: 2).")
    parser.add_argument("--cols", type=int, default=3,
                        help="Number of columns in the grid (default: 3).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the figure.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for the output figure (default: 300).")

    args = parser.parse_args()
    create_figure_from_images(args.folder, args.rows,
                              args.cols, args.output, args.dpi)
