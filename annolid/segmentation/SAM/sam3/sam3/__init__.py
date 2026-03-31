# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

from importlib import import_module

from ..aliases import ensure_sam3_aliases

ensure_sam3_aliases()

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_video_predictor",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".model_builder", __name__)
    return getattr(module, name)
