import cv2
import numpy as np
import torch
import sys
import os
import pickle
"""
Image Embeddings Generation for All Slices of a Tiff Stack Data

This script provides functions to load, preprocess, and create embeddings for 
input 3d stack tiff images along three Cartesian directions (X, Y, Z) using a Segment Anything Model (SAM).
 The generated embeddings are saved to a pickle file for further analysis and processing.

Functions:
    1. load_image(input_filepath):
        - Load and preprocess the input tiff stack image.
        - Args:
            - input_filepath (str): Path to the input image file.
        - Returns:
            - img (numpy.ndarray): Preprocessed image data.

    2. create_embeddings(img, output_filepath, sam_checkpoint_path):
        - Create embeddings for the input image along X, Y, and Z directions and save them to a pickle file.
        - Args:
            - img (numpy.ndarray): Preprocessed image data.
            - output_filepath (str): Path to save the output embeddings pickle file.
            - sam_checkpoint_path (str): Path to the SAM model checkpoint.

    3. get_image_slice(img, dimension, index):
        - Extract an image slice along a specified dimension (0 for X, 1 for Y, 2 for Z).
        - Args:
            - img (numpy.ndarray): Preprocessed image data.
            - dimension (int): Dimension along which to extract the slice (0 for X, 1 for Y, 2 for Z).
            - index (int): Index of the slice.
        - Returns:
            - img_slice (numpy.ndarray): Extracted image slice.

    4. save_embeddings(output_filepath, embeddings):
        - Save embeddings to a pickle file.
        - Args:
            - output_filepath (str): Path to save the output embeddings pickle file.
            - embeddings (list): List of embeddings data.

Usage:
    - Import this script into your Python environment.
    - Call the provided functions with your input data to generate and save image embeddings.

Example:
    ```python
    import cv2
    import numpy as np
    import torch
    import sys
    import os
    import pickle
    from segment_anything import sam_model_registry, SamPredictor

    # Specify paths and load input image
    input_filepath = "path/to/input_image.png"
    output_filepath = "path/to/output_embeddings"
    sam_checkpoint_path = "path/to/sam_model_checkpoint.pth"

    img = load_image(input_filepath)
    create_embeddings(img, output_filepath, sam_checkpoint_path)
    ```

Note:
    - Ensure that all required dependencies are installed.
    - Adjust the provided paths and settings to match your specific use case.

Reference:
@article{semeraro2023tomosam,
title={TomoSAM: a 3D Slicer extension using SAM for tomography segmentation},
author={Semeraro, Federico and Quintart, Alexandre and Izquierdo, Sergio Fraile and Ferguson, Joseph C},
journal={arXiv preprint arXiv:2306.08609},
year={2023}
} 

"""


def load_image(input_filepath):
    """Load and preprocess the input image."""
    check, img = cv2.imreadmulti(input_filepath)
    img = np.array(img)

    if not check:
        raise Exception("Image file not found.")
    elif img.ndim > 3 or img.ndim < 2:
        raise Exception("Unsupported image type.")
    elif img.ndim == 2:
        img = img[:, :, np.newaxis]

    print(f"Image dimensions: {img.shape}")
    return img


def create_embeddings(img, output_filepath, sam_checkpoint_path):
    """Create embeddings for an input image and save them to a file."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'segment_anything' is required to create SAM embeddings. "
            "Install it with: pip install \"segment-anything @ git+https://github.com/SysCV/sam-hq.git\""
        ) from exc
    slice_directions = ['x', 'y', 'z']

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    if torch.cuda.is_available():
        sam.to(device="cuda")
    predictor = SamPredictor(sam)

    embeddings = [[], [], []]

    for i, direction in enumerate(slice_directions):
        print(f"\nSlicing along {direction} direction")
        for k in range(img.shape[i]):
            img_slice = get_image_slice(img, i, k)
            sys.stdout.write(
                f"\rCreating embedding for {k + 1}/{img.shape[i]} image")

            predictor.reset_image()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predictor.set_image(
                np.repeat(img_slice[:, :, np.newaxis], 3, axis=2))

            embeddings[i].append({
                'original_size': predictor.original_size,
                'input_size': predictor.input_size,
                'features': predictor.features.to('cpu')
            })

    save_embeddings(output_filepath, embeddings)
    print(f"\nSaved {output_filepath}.pkl")


def get_image_slice(img, dimension, index):
    """Extract an image slice along a specified dimension."""
    if dimension == 0:
        return img[index]
    elif dimension == 1:
        return img[:, index]
    else:
        return img[:, :, index]


def save_embeddings(output_filepath, embeddings):
    """Save embeddings to a pickle file."""
    with open(output_filepath + ".pkl", 'wb') as f:
        pickle.dump(embeddings, f)
