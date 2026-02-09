# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CowTracker heads."""

from cowtracker.heads.tracking_head import CowTrackingHead
from cowtracker.heads.feature_extractor import FeatureExtractor

__all__ = ["CowTrackingHead", "FeatureExtractor", "DPTHead"]


def __getattr__(name: str):
    if name == "DPTHead":
        from cowtracker.dependencies import get_vggt_dpt_head_cls

        return get_vggt_dpt_head_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
