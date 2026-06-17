"""DINOv3 keypoint-mask segmentation with lazy runtime exports.

Metadata modules such as ``defaults`` are import-safe for the GUI. Predictor
and inference helpers load their optional ML dependencies only when requested.
"""

from __future__ import annotations

from typing import Any


_EXPORTS = {
    "DinoKPSEGPredictor": (
        "annolid.segmentation.dino_kpseg.predictor",
        "DinoKPSEGPredictor",
    ),
    "DinoKPSEGInstanceCrop": (
        "annolid.segmentation.dino_kpseg.inference_utils",
        "DinoKPSEGInstanceCrop",
    ),
    "build_instance_crops": (
        "annolid.segmentation.dino_kpseg.inference_utils",
        "build_instance_crops",
    ),
    "mask_bbox": ("annolid.segmentation.dino_kpseg.inference_utils", "mask_bbox"),
    "predict_on_instance_crops": (
        "annolid.segmentation.dino_kpseg.inference_utils",
        "predict_on_instance_crops",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'annolid.segmentation.dino_kpseg' has no attribute {name!r}"
        ) from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
