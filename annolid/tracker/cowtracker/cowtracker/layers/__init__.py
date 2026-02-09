# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Network layers and backbone modules for CowTracker."""

from cowtracker.layers.temporal_attention import TemporalSelfAttentionBlock
from cowtracker.layers.video_transformer import (
    MODEL_CONFIGS,
    VisionTransformerVideo,
    FlashAttention3,
    replace_attention_with_flash3,
)
from cowtracker.layers.patch_embed import PatchEmbed

__all__ = [
    "TemporalSelfAttentionBlock",
    "MODEL_CONFIGS",
    "VisionTransformerVideo",
    "FlashAttention3",
    "replace_attention_with_flash3",
    "PatchEmbed",
]
