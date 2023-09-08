import cv2
import numpy as np
import torch
import sys
import os
import pickle
from segment_anything import sam_model_registry, SamPredictor
"""
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
