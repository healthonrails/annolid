"""Segmentation backends and intra-instance body-part segmentation APIs."""

from __future__ import annotations

from annolid.segmentation.body_parts import (
    BodyFramePose,
    BodyPartPrototype,
    BodyPartSegmentationConfig,
    InstanceBodyPartSegmenter,
    InstanceBodyPartVideoProcessor,
    PartPrediction,
    compute_masked_geodesic_distances,
    estimate_body_frame_pose,
    guided_filter_probabilities,
    parse_frame_instances_and_seed_parts,
)

__all__ = [
    "BodyFramePose",
    "BodyPartPrototype",
    "BodyPartSegmentationConfig",
    "InstanceBodyPartSegmenter",
    "InstanceBodyPartVideoProcessor",
    "PartPrediction",
    "compute_masked_geodesic_distances",
    "estimate_body_frame_pose",
    "guided_filter_probabilities",
    "parse_frame_instances_and_seed_parts",
]
