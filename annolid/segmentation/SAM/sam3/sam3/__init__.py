# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from ..aliases import ensure_sam3_aliases

ensure_sam3_aliases()

from .model_builder import build_sam3_image_model

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]
