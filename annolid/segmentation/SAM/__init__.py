import os
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import torch
import collections

from annolid.segmentation.SAM.segment_anything import SegmentAnythingModel  # NOQA
"""Labelme segment anything encoder and decorder in onnx format
https://github.com/wkentaro/labelme/blob/main/labelme/ai/__init__.py"""

Model = collections.namedtuple(
    "Model", ["name", "encoder_weight", "decoder_weight"]
)

Weight = collections.namedtuple("Weight", ["url", "md5"])

MODELS = [
    Model(
        name="Segment-Anything (speed)",
        encoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx",  # NOQA
            md5="80fd8d0ab6c6ae8cb7b3bd5f368a752c",
        ),
        decoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx",  # NOQA
            md5="4253558be238c15fc265a7a876aaec82",
        ),
    ),
    Model(
        name="Segment-Anything (balanced)",
        encoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx",  # NOQA
            md5="080004dc9992724d360a49399d1ee24b",
        ),
        decoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx",  # NOQA
            md5="851b7faac91e8e23940ee1294231d5c7",
        ),
    ),
    Model(
        name="Segment-Anything (accuracy)",
        encoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx",  # NOQA
            md5="958b5710d25b198d765fb6b94798f49e",
        ),
        decoder_weight=Weight(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.decoder.onnx",  # NOQA
            md5="a997a408347aa081b17a3ffff9f42a80",
        ),
    ),
]


class SAMModel:
    """
    Wrapper class for the Segment Anything Model (SAM).
    Reference: https://github.com/jvpassarelli/sam-clip-segmentation/blob/main/SAMCLIPInstanceSegmentation.py
    """

    def __init__(self, sam_checkpoint="sam_vit_b_01ec64.pth"):
        """
        Initializes the SAM model by loading the checkpoint file.

        sam_checkpoint (str): Path to the SAM checkpoint file.
        """
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint file not found at path: {sam_checkpoint}")
        self.sam = build_sam(checkpoint=sam_checkpoint)
        if torch.cuda.is_available():
            self.sam.to(device="cuda")

    def generate_masks(self, image_path,
                       stability_score_threshold=.98,
                       predicted_iou_threshold=.98):
        """
        Generates masks using the SAM model for the given image.

        image_path (str): Path to the input image.
        stability_score_threshold (float): Filtering threshold for stability/quality of SAM masks.
        predicted_iou_threshold (float): Threshold for the model's (SAM's) own prediction of quality.

        Returns:
            List: List of SAM masks.
        """
        image = Image.open(image_path)
        image_np = np.asarray(image)

        mask_generator = SamAutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(image_np)

        filtered_masks = []
        for mask in masks:
            if (mask["stability_score"] > stability_score_threshold
                    and mask["predicted_iou"] > predicted_iou_threshold):
                filtered_masks.append(mask)

        return filtered_masks
