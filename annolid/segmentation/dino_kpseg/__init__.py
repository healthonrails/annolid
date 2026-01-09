"""DINOv3 keypoint-mask segmentation.

This module provides a lightweight segmentation head trained on top of frozen
DINOv3 dense features. The supervision target is a small circular mask around
each keypoint, producing a per-keypoint "body part" segmentation map.
"""

from .predictor import DinoKPSEGPredictor
from .inference_utils import (
    DinoKPSEGInstanceCrop,
    build_instance_crops,
    mask_bbox,
    predict_on_instance_crops,
)

__all__ = [
    "DinoKPSEGPredictor",
    "DinoKPSEGInstanceCrop",
    "build_instance_crops",
    "mask_bbox",
    "predict_on_instance_crops",
]
