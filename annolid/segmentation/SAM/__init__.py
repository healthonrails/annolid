import os
import collections
from typing import TYPE_CHECKING

from PIL import Image
import numpy as np
import torch
"""Labelme segment anything encoder and decorder in onnx format
https://github.com/wkentaro/labelme/blob/main/labelme/ai/__init__.py"""

Model = collections.namedtuple(
    "Model", ["name", "encoder_weight", "decoder_weight"]
)

Weight = collections.namedtuple("Weight", ["url", "md5"])

MODELS = [
    Model(
        name="Segment-Anything (Edge)",
        encoder_weight=Weight(
            url="https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_encoder.onnx",  # NOQA
            md5="e0745d06f3ee9c5e01a667b56a40875b",
        ),
        decoder_weight=Weight(
            url="https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x_decoder.onnx",  # NOQA
            md5="9fe1d5521b4349ab710e9cc970936970",
        ),
    ),
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


if TYPE_CHECKING:  # pragma: no cover
    from annolid.segmentation.SAM.segment_anything import SegmentAnythingModel as SegmentAnythingModel


def __getattr__(name: str):
    if name == "SegmentAnythingModel":
        from annolid.segmentation.SAM.segment_anything import (
            SegmentAnythingModel as _SegmentAnythingModel,
        )

        return _SegmentAnythingModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _require_segment_anything():
    try:
        from segment_anything import build_sam, SamAutomaticMaskGenerator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'segment_anything' is required for SAM features. "
            "Install it with: pip install \"segment-anything @ git+https://github.com/SysCV/sam-hq.git\""
        ) from exc
    return build_sam, SamAutomaticMaskGenerator


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
        build_sam, _ = _require_segment_anything()
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint file not found at path: {sam_checkpoint}")
        self.sam = build_sam(checkpoint=sam_checkpoint)
        if torch.cuda.is_available():
            self.sam.to(device="cuda")

    def generate_masks(self, image_path,
                       stability_score_threshold=.98,
                       predicted_iou_threshold=.98,
                       min_mask_region_area=100,
                       points_per_side=32,
                       crop_n_layers=1,
                       crop_n_points_downscale_factor=2
                       ):
        """
        Generates masks using the SAM model for the given image.

        image_path (str): Path to the input image.
        stability_score_threshold (float): Filtering threshold for stability/quality of SAM masks.
        predicted_iou_threshold (float): Threshold for the model's (SAM's) own prediction of quality.

        Here are several optional tunable parameters in automatic mask generation that control how densely points
        are sampled and what the thresholds are for removing low-quality or duplicate masks.
        Additionally, generation can be automatically run on crops of the image to get improved
        performance on smaller objects, and post-processing can remove stray pixels and holes.

        Example configuration:

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,  # Number of points to sample along each side of the bounding box
            pred_iou_thresh=0.86,  # IOU threshold for removing duplicate masks
            stability_score_thresh=0.92,  # Threshold for removing low-quality masks based on stability score
            crop_n_layers=1,  # Number of layers to use when cropping the image
            crop_n_points_downscale_factor=2,  # Downscale factor applied to the number of points when cropping the image
            min_mask_region_area=100  # Minimum area threshold for removing small regions in the final mask

        Parameters:
        - model: The model used for mask generation.
        - points_per_side: The number of points to sample along each side of the bounding box.
        - pred_iou_thresh: The IOU threshold for removing duplicate masks.
        - stability_score_thresh: The threshold for removing low-quality masks based on the stability score.
        - crop_n_layers: The number of layers to use when cropping the image.
        - crop_n_points_downscale_factor: The downscale factor applied to the number of points when cropping the image.
        - min_mask_region_area: The minimum area threshold for removing small regions in the final mask (requires open-cv for post-processing).

        Note: The provided configuration is just an example.
        You may need to adjust these parameters based on your specific requirements and dataset.

        Returns:
            List: List of SAM masks.
        """
        _, SamAutomaticMaskGenerator = _require_segment_anything()
        image = Image.open(image_path)
        image_np = np.asarray(image)

        mask_generator = SamAutomaticMaskGenerator(model=self.sam,
                                                   points_per_side=points_per_side,
                                                   pred_iou_thresh=predicted_iou_threshold,
                                                   stability_score_thresh=stability_score_threshold,
                                                   crop_n_layers=crop_n_layers,
                                                   crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                                   min_mask_region_area=min_mask_region_area)  # Requires open-cv to run post-processing
        masks = mask_generator.generate(image_np)

        filtered_masks = []
        for mask in masks:
            if (mask["stability_score"] > stability_score_threshold
                    and mask["predicted_iou"] > predicted_iou_threshold):
                filtered_masks.append(mask)

        return filtered_masks
