# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Padding utilities for video preprocessing and postprocessing."""

import torch.nn.functional as F


def compute_padding_params(orig_H, orig_W, inf_H, inf_W, skip_upscaling=False):
    """Compute padding parameters to preserve aspect ratio.

    Args:
        orig_H: Original height
        orig_W: Original width
        inf_H: Inference height
        inf_W: Inference width
        skip_upscaling: If True and scale > 1, skip upscaling and just pad

    Returns:
        Dictionary containing:
            - scale: Scale factor that would be applied (1.0 if skipped)
            - scaled_H, scaled_W: Dimensions after scaling (before padding)
            - pad_top, pad_bottom, pad_left, pad_right: Padding amounts
            - orig_H, orig_W: Original dimensions (for reference)
            - upscaling_skipped: Whether upscaling was skipped
    """
    scale = min(inf_H / orig_H, inf_W / orig_W)

    upscaling_skipped = False
    if skip_upscaling and scale > 1.0:
        scaled_H = orig_H
        scaled_W = orig_W
        upscaling_skipped = True
    else:
        scaled_H = int(orig_H * scale)
        scaled_W = int(orig_W * scale)

    pad_H = inf_H - scaled_H
    pad_W = inf_W - scaled_W

    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left

    return {
        "scale": scale,
        "scaled_H": scaled_H,
        "scaled_W": scaled_W,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "orig_H": orig_H,
        "orig_W": orig_W,
        "upscaling_skipped": upscaling_skipped,
    }


def apply_padding(rgbs, padding_info):
    """Apply padding to input images to reach inference size.

    Args:
        rgbs: Input tensor (T, C, H, W)
        padding_info: Dictionary from compute_padding_params

    Returns:
        Padded tensor (T, C, inf_H, inf_W)
    """
    T, C, H, W = rgbs.shape
    scaled_H = padding_info["scaled_H"]
    scaled_W = padding_info["scaled_W"]

    if (scaled_H, scaled_W) != (H, W):
        rgbs_scaled = F.interpolate(
            rgbs,
            size=(scaled_H, scaled_W),
            mode="bilinear",
            align_corners=False,
        )
    else:
        rgbs_scaled = rgbs

    pad_left = padding_info["pad_left"]
    pad_right = padding_info["pad_right"]
    pad_top = padding_info["pad_top"]
    pad_bottom = padding_info["pad_bottom"]

    rgbs_padded = F.pad(
        rgbs_scaled,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    return rgbs_padded


def remove_padding_and_scale_back(tracks, visibility, confidence, padding_info):
    """Remove padding from model outputs and scale back to original resolution.

    Args:
        tracks: Track predictions (T, inf_H, inf_W, 2)
        visibility: Visibility predictions (T, inf_H, inf_W)
        confidence: Confidence predictions (T, inf_H, inf_W)
        padding_info: Dictionary from compute_padding_params

    Returns:
        Tuple of (tracks, visibility, confidence) scaled to original resolution
    """
    scaled_H = padding_info["scaled_H"]
    scaled_W = padding_info["scaled_W"]
    pad_top = padding_info["pad_top"]
    pad_left = padding_info["pad_left"]
    orig_H = padding_info["orig_H"]
    orig_W = padding_info["orig_W"]

    tracks_unpadded = tracks[
        :, pad_top : pad_top + scaled_H, pad_left : pad_left + scaled_W, :
    ]
    visibility_unpadded = visibility[
        :, pad_top : pad_top + scaled_H, pad_left : pad_left + scaled_W
    ]
    confidence_unpadded = confidence[
        :, pad_top : pad_top + scaled_H, pad_left : pad_left + scaled_W
    ]

    tracks_unpadded = tracks_unpadded.clone()
    tracks_unpadded[:, :, :, 0] -= pad_left
    tracks_unpadded[:, :, :, 1] -= pad_top

    if (scaled_H, scaled_W) != (orig_H, orig_W):
        tracks_permuted = tracks_unpadded.permute(0, 3, 1, 2)
        tracks_scaled = F.interpolate(
            tracks_permuted,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        )
        tracks_final = tracks_scaled.permute(0, 2, 3, 1)

        tracks_final[:, :, :, 0] *= orig_W / scaled_W
        tracks_final[:, :, :, 1] *= orig_H / scaled_H

        visibility_final = F.interpolate(
            visibility_unpadded.unsqueeze(1),
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        confidence_final = F.interpolate(
            confidence_unpadded.unsqueeze(1),
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    else:
        tracks_final = tracks_unpadded
        visibility_final = visibility_unpadded
        confidence_final = confidence_unpadded

    return tracks_final, visibility_final, confidence_final
