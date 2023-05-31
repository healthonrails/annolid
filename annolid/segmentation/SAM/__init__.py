import os
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import torch


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
