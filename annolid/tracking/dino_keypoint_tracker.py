"""Cutie + DINO tracker that outputs unified keypoint and mask annotations."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from annolid.annotation.pose_schema import PoseSchema
from annolid.data.videos import CV2Video
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.features.dino_models import (
    DEFAULT_DINO_FEATURE_MODEL_ID,
    resolve_dino_model_from_runtime,
    resolve_dino_model_id,
    set_dino_model_on_runtime,
)
from annolid.features.dinov3_feature_grid import (
    apply_channel_debias_basis,
    coordinate_debias_basis,
    extract_feature_grid,
    normalize_feature_grid,
    svd_positional_basis,
)
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager, MaskResult
from annolid.tracking.dino_keypoint_types import (
    BodyPrior,
    Candidate,
    FrameMotionEstimate,
    KeypointTrack,
    MotionPrior,
    PixelFlowEstimate,
    PixelRefineOptions,
    RefineRegion,
    SupportProbe,
)
from annolid.tracking.domain import InstanceRegistry
from annolid.tracking.gaussian_region import (
    GaussianPixelTemplate,
    extract_gaussian_pixel_template,
    gaussian_smooth_feature_grid,
    match_gaussian_pixel_template,
    pixel_to_continuous_patch,
    refine_patch_region_gaussian,
    sample_subpatch_gaussian_descriptor,
)
from annolid.tracking.frame_skip import (
    build_seeded_frame_index,
    remove_seeded_frame,
    should_skip_finished_frame_between_adjacent_seeded_frames,
)
from annolid.tracking.kpseg_smoothing import KeypointSmoother
from annolid.tracking.video_seed_utils import (
    apply_mask_results_to_registry,
    collect_manual_annotation_frames,
    find_latest_tracked_frame_before,
    is_processor_stop_requested,
    mask_lookup_from_registry,
    normalize_tracking_frame_range,
    report_processor_progress,
)
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger

__all__ = [
    "BodyPrior",
    "Candidate",
    "DinoKeypointTracker",
    "DinoKeypointVideoProcessor",
    "FrameMotionEstimate",
    "KeypointTrack",
    "MotionPrior",
    "PixelFlowEstimate",
    "PixelRefineOptions",
    "RefineRegion",
    "SupportProbe",
]


class DinoKeypointTracker:
    """Patch descriptor tracker with optional mask-aware constraints."""

    def __init__(
        self,
        model_name: str,
        *,
        short_side: int = 768,
        device: Optional[str] = None,
        runtime_config: Optional[CutieDinoTrackerConfig] = None,
        search_radius: int = 2,
        min_similarity: float = 0.2,
        momentum: float = 0.2,
        reference_weight: float = 0.7,
        reference_support_radius: int = 0,
        reference_center_weight: float = 1.0,
    ) -> None:
        self.runtime_config = runtime_config or CutieDinoTrackerConfig()
        model_name = self._resolve_model_name(model_name, self.runtime_config)
        cfg = Dinov3Config(
            model_name=model_name,
            short_side=short_side,
            device=device,
            layers=(-2, -1),
            matryoshka_dim=getattr(
                self.runtime_config, "embedding_matryoshka_dim", None
            ),
        )
        self.extractor = Dinov3FeatureExtractor(cfg)
        self.search_radius = max(1, int(search_radius))
        self.min_similarity = float(min_similarity)
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.reference_weight = float(np.clip(reference_weight, 0.0, 1.0))
        self.reference_support_radius = max(0, int(reference_support_radius))
        self.reference_center_weight = float(np.clip(reference_center_weight, 0.0, 1.0))
        self.mask_descriptor_weight = float(
            np.clip(self.runtime_config.mask_descriptor_weight, 0.0, 1.0)
        )
        self.restrict_to_mask = bool(self.runtime_config.restrict_to_initial_mask)
        self.mask_enforce_position = bool(
            getattr(self.runtime_config, "mask_enforce_position", True)
        )
        snap_radius = getattr(
            self.runtime_config,
            "mask_enforce_snap_radius",
            getattr(self.runtime_config, "mask_enforce_search_radius", 12),
        )
        self.mask_enforce_snap_radius = max(1, int(snap_radius))
        self.mask_enforce_reject_outside = bool(
            getattr(self.runtime_config, "mask_enforce_reject_outside", True)
        )
        self.tracks: Dict[str, KeypointTrack] = {}
        self.patch_size = self.extractor.patch_size
        self.max_misses = 8
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_scale: Tuple[float, float] = (1.0, 1.0)
        self._last_patch_masks: Dict[str, np.ndarray] = {}
        self._mask_miss_counts: Dict[str, int] = {}
        self.max_candidates = max(1, int(self.runtime_config.max_candidate_tracks))
        self._is_fresh_start = False
        self._roi_offset: Tuple[float, float] = (0.0, 0.0)
        self._roi_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._roi_size: Tuple[int, int] = (0, 0)
        self._instance_body_axes: Dict[str, Tuple[float, float]] = {}
        self._instance_ref_span: Dict[str, float] = {}
        self._manual_anchor_codebooks: Dict[str, torch.Tensor] = {}
        self._manual_pixel_templates: Dict[str, GaussianPixelTemplate] = {}
        self._part_shared_descriptors: Dict[str, torch.Tensor] = {}
        self._part_shared_counts: Dict[str, int] = {}
        self._last_norm_feats: Optional[torch.Tensor] = None
        self._last_grid_hw: Optional[Tuple[int, int]] = None
        self._prev_matching_feats: Optional[torch.Tensor] = None
        self._svd_positional_basis_cache: Dict[
            Tuple[str, int, int, int, int], torch.Tensor
        ] = {}
        self._stop_requested = False
        self.keypoint_refine_radius = max(
            0, int(getattr(self.runtime_config, "keypoint_refine_radius", 0))
        )
        self.keypoint_refine_sigma = max(
            1e-4, float(getattr(self.runtime_config, "keypoint_refine_sigma", 1.25))
        )
        self.keypoint_refine_temperature = max(
            1e-4,
            float(getattr(self.runtime_config, "keypoint_refine_temperature", 0.35)),
        )
        self.pixel_refine_options = PixelRefineOptions.from_runtime(self.runtime_config)
        self.reset_state()

    @staticmethod
    def _resolve_model_name(
        model_name: object,
        runtime_config: CutieDinoTrackerConfig,
    ) -> str:
        """
        Resolve the DINO feature backbone at the tracker boundary.

        Explicit non-default constructor values win, while a default/empty
        constructor value defers to the selected runtime config model. This
        keeps GUI Advanced Parameters selections effective even if an older
        caller still passes the historical default model string.
        """
        constructor_model = resolve_dino_model_id(model_name)
        runtime_model = resolve_dino_model_from_runtime(
            runtime_config,
            fallback=constructor_model,
        )
        selected = (
            runtime_model
            if constructor_model == DEFAULT_DINO_FEATURE_MODEL_ID
            else constructor_model
        )
        return set_dino_model_on_runtime(runtime_config, selected)

    def _get_pixel_refine_options(self) -> PixelRefineOptions:
        options = getattr(self, "pixel_refine_options", None)
        if isinstance(options, PixelRefineOptions):
            return options
        runtime_config = getattr(self, "runtime_config", CutieDinoTrackerConfig())
        options = PixelRefineOptions.from_runtime(runtime_config)
        self.pixel_refine_options = options
        return options

    def request_stop(self) -> None:
        """Request cooperative cancellation at the next tracker boundary."""
        self._stop_requested = True

    def is_stopped(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return bool(self._stop_requested)

    def reset_state(self, *, preserve_manual_anchors: bool = False) -> None:
        """Restore tracker state to a clean slate prior to (re)starting."""
        self.tracks.clear()
        self._stop_requested = False
        self.prev_gray = None
        self.prev_scale = (1.0, 1.0)
        self._last_patch_masks = {}
        self._mask_miss_counts = {}

        self._is_fresh_start = False
        self._roi_offset = (0.0, 0.0)
        self._roi_box = (0, 0, 0, 0)
        self._roi_size = (0, 0)
        self._instance_body_axes = {}
        self._instance_ref_span = {}
        if not preserve_manual_anchors:
            self._manual_anchor_codebooks = {}
            self._manual_pixel_templates = {}
        self._part_shared_descriptors = {}
        self._part_shared_counts = {}
        self._last_norm_feats = None
        self._last_grid_hw = None
        self._prev_matching_feats = None

    def start(
        self,
        image: Image.Image,
        registry: InstanceRegistry,
        mask_lookup: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.reset_state(preserve_manual_anchors=True)
        self._is_fresh_start = True  # Signal that the next update is the first
        if self._stop_requested:
            return

        polygons: List[List[Tuple[float, float]]] = []
        for instance in registry:
            if instance.polygon:
                polygons.append([(float(x), float(y)) for x, y in instance.polygon])

        feats, scale_x, scale_y, grid_hw = self._prepare_roi_inputs(
            image=image,
            mask_lookup=mask_lookup,
            polygons=polygons,
        )
        if self._stop_requested:
            return
        matching_feats = self._prepare_matching_feature_grid(feats)
        context_map = self._compute_context_map(matching_feats)
        cropped_masks = self._crop_masks_for_roi(mask_lookup)
        self._last_patch_masks = {}
        self._mask_miss_counts = {}
        mask_cache = self._build_mask_cache(
            matching_feats,
            cropped_masks,
            grid_hw,
            allow_fallback=False,
        )
        default_region_feats, instance_region_feats = (
            self._resolve_instance_region_feats(matching_feats, mask_cache)
        )
        self._prev_matching_feats = default_region_feats.detach()
        body_priors: Dict[str, BodyPrior] = {}
        if float(self.runtime_config.structural_consistency_weight) > 0.0:
            body_priors = self._compute_body_priors(mask_cache)

        frame_array = np.array(image)
        frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        use_pixel_region = bool(
            getattr(self.runtime_config, "pixel_region_refine_enabled", False)
        )
        pixel_region_window = int(
            getattr(self.runtime_config, "pixel_region_window", 11)
        )
        pixel_region_sigma = float(
            getattr(self.runtime_config, "pixel_region_sigma", 3.0)
        )

        self.prev_scale = (scale_x, scale_y)
        self.tracks.clear()
        for instance in registry:
            track_feats = instance_region_feats.get(
                instance.label, default_region_feats
            )
            cache_entry = mask_cache.get(instance.label)
            mask_descriptor = cache_entry.get("descriptor") if cache_entry else None
            patch_mask = cache_entry.get("patch_mask") if cache_entry else None
            for keypoint in instance.keypoints.values():
                patch_rc = self._pixel_to_patch(
                    keypoint.x, keypoint.y, scale_x, scale_y, grid_hw
                )
                base_desc = self._sample_point_descriptor(
                    track_feats,
                    xy=(float(keypoint.x), float(keypoint.y)),
                    patch_rc=patch_rc,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    grid_hw=grid_hw,
                    patch_mask=patch_mask,
                )
                reference_desc = self._reference_descriptor(
                    track_feats, patch_rc, grid_hw, base_desc
                )
                self._update_manual_anchor_codebook(keypoint.key, reference_desc)
                descriptor = self._apply_mask_descriptor(
                    reference_desc.clone(), mask_descriptor
                )
                context_desc = None
                if context_map is not None:
                    context_desc = self._sample_point_descriptor(
                        context_map,
                        xy=(float(keypoint.x), float(keypoint.y)),
                        patch_rc=patch_rc,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        grid_hw=grid_hw,
                        patch_mask=patch_mask,
                    )
                pixel_template = None
                if use_pixel_region:
                    pixel_template = extract_gaussian_pixel_template(
                        frame_gray,
                        (float(keypoint.x), float(keypoint.y)),
                        window_px=pixel_region_window,
                        sigma_px=pixel_region_sigma,
                    )
                    if pixel_template is not None:
                        self._manual_pixel_templates[keypoint.key] = pixel_template
                    else:
                        pixel_template = self._manual_pixel_templates.get(keypoint.key)

                # KeypointTrack always starts with zero velocity.
                track = KeypointTrack(
                    key=keypoint.key,
                    storage_label=keypoint.storage_label,
                    instance_label=instance.label,
                    display_label=keypoint.label,
                    patch_rc=patch_rc,
                    descriptor=descriptor,
                    reference_descriptor=reference_desc.clone(),
                    context_descriptor=context_desc,
                    manual_codebook=self._manual_anchor_codebooks.get(keypoint.key),
                    last_position=(float(keypoint.x), float(keypoint.y)),
                    pixel_template=pixel_template,
                )

                track.appearance_codebook = self._collect_appearance_codebook(
                    track_feats, patch_rc
                )
                if track.appearance_codebook is not None:
                    baseline = torch.matmul(track.appearance_codebook, track.descriptor)
                    track.baseline_similarity = float(baseline.max().item())
                track.support_probes = self._sample_support_probes(
                    track.key,
                    track_feats,
                    patch_rc,
                    grid_hw,
                    patch_mask,
                )
                body_prior = body_priors.get(instance.label)
                if body_prior is not None:
                    track.body_coords = body_prior.signature(
                        (float(patch_rc[0]), float(patch_rc[1]))
                    )
                self._update_part_shared_descriptor(
                    track,
                    reference_desc,
                    confidence=1.0,
                )
                self.tracks[keypoint.key] = track

        self.prev_gray = frame_gray
        self._initialize_structural_refs(registry)
        self._initialize_symmetry_refs(registry)
        self._initialize_instance_spans(registry, mask_lookup)

    def update(
        self,
        image: Image.Image,
        mask_lookup: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        if self._stop_requested:
            return []
        if not self.tracks:
            return []

        prev_roi_box = self._roi_box
        feats, scale_x, scale_y, grid_hw = self._prepare_roi_inputs(
            image=image,
            mask_lookup=mask_lookup,
            polygons=None,
        )
        if self._stop_requested:
            return []
        roi_changed = self._roi_box != prev_roi_box
        grid_h, grid_w = grid_hw
        prev_matching_feats = self._prev_matching_feats
        if (
            roi_changed
            or prev_matching_feats is None
            or tuple(prev_matching_feats.shape[1:]) != tuple(grid_hw)
        ):
            prev_matching_feats = None
        matching_feats = self._prepare_matching_feature_grid(feats)
        context_map = self._compute_context_map(matching_feats)
        inv_scale_x = 1.0 / max(scale_x, 1e-6)
        inv_scale_y = 1.0 / max(scale_y, 1e-6)
        patch_size = float(self.patch_size)
        patch_centers_x = (
            np.arange(grid_w, dtype=np.float32) + 0.5
        ) * patch_size * inv_scale_x + self._roi_offset[0]
        patch_centers_y = (
            np.arange(grid_h, dtype=np.float32) + 0.5
        ) * patch_size * inv_scale_y + self._roi_offset[1]
        cropped_masks = self._crop_masks_for_roi(mask_lookup)
        mask_cache = self._build_mask_cache(
            matching_feats,
            cropped_masks,
            grid_hw,
            allow_fallback=True,
        )
        default_region_feats, instance_region_feats = (
            self._resolve_instance_region_feats(matching_feats, mask_cache)
        )
        self._last_norm_feats = default_region_feats
        self._last_grid_hw = grid_hw
        body_priors: Dict[str, BodyPrior] = {}
        if float(self.runtime_config.structural_consistency_weight) > 0.0:
            body_priors = self._compute_body_priors(mask_cache)
        active_masks: Dict[str, Optional[np.ndarray]] = {}
        if mask_lookup:
            active_masks = {
                label: mask.astype(bool) if mask is not None else None
                for label, mask in mask_lookup.items()
            }

        frame_array = np.array(image)
        frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        prev_gray = self.prev_gray
        dense_flow_cache: Optional[np.ndarray] = None
        dense_flow_computed = False

        def get_dense_flow() -> Optional[np.ndarray]:
            nonlocal dense_flow_cache, dense_flow_computed
            if dense_flow_computed:
                return dense_flow_cache
            dense_flow_computed = True
            if prev_gray is not None and prev_gray.shape == frame_gray.shape:
                dense_flow_cache = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    frame_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
            return dense_flow_cache

        batched_lk_by_track: Optional[Dict[str, Optional[PixelFlowEstimate]]] = None
        if (
            prev_gray is not None
            and len(self.tracks) > 1
            and self._is_default_pixel_flow_method()
        ):
            batched_lk_by_track = self._track_pixel_flow_batch(
                prev_gray, frame_gray, list(self.tracks.values())
            )

        self.prev_gray = frame_gray
        self.prev_scale = (scale_x, scale_y)
        previous_positions = {
            track.key: track.last_position for track in self.tracks.values()
        }
        context_weight = float(
            max(0.0, getattr(self.runtime_config, "context_weight", 0.0))
        )
        appearance_weight = float(
            max(0.0, getattr(self.runtime_config, "appearance_bundle_weight", 0.0))
        )
        baseline_weight = float(
            max(0.0, getattr(self.runtime_config, "baseline_similarity_weight", 0.0))
        )
        struct_weight = float(
            max(
                0.0,
                getattr(self.runtime_config, "structural_consistency_weight", 0.0),
            )
        )
        symmetry_weight = float(
            max(0.0, getattr(self.runtime_config, "symmetry_penalty", 0.0))
        )
        motion_penalty_weight = float(
            max(0.0, getattr(self.runtime_config, "motion_prior_penalty_weight", 0.0))
        )
        support_weight = float(
            max(0.0, getattr(self.runtime_config, "support_probe_weight", 0.0))
        )
        backward_weight = float(
            max(
                0.0,
                getattr(self.runtime_config, "dinov3_backward_consistency_weight", 0.0),
            )
        )
        velocity_smoothing = float(
            np.clip(getattr(self.runtime_config, "velocity_smoothing", 0.0), 0.0, 1.0)
        )
        use_body_prior = struct_weight > 0.0
        use_backward_consistency = (
            backward_weight > 0.0
            and bool(getattr(self.runtime_config, "dinov3_backward_consistency", False))
            and prev_matching_feats is not None
        )
        prev_matching_flat = (
            prev_matching_feats.reshape(prev_matching_feats.shape[0], -1)
            if use_backward_consistency and prev_matching_feats is not None
            else None
        )
        use_pixel_region = bool(
            getattr(self.runtime_config, "pixel_region_refine_enabled", False)
        )
        use_cluster_refine = (
            bool(getattr(self.runtime_config, "keypoint_cluster_refine", False))
            and int(getattr(self.runtime_config, "keypoint_cluster_refine_radius", 0))
            > 0
        )
        tracking_enabled = bool(getattr(self.runtime_config, "tracking_enable", False))
        instance_context: Dict[str, Dict[str, object]] = (
            self._prepare_multi_animal_instance_context(
                previous_positions, active_masks
            )
            if tracking_enabled
            else {}
        )
        effective_min_similarity = (
            max(
                self.min_similarity,
                float(
                    getattr(
                        self.runtime_config,
                        "tracking_instance_match_min_similarity",
                        0.0,
                    )
                ),
            )
            if tracking_enabled
            else self.min_similarity
        )

        track_candidates: Dict[str, List[Candidate]] = {}
        refine_regions: Dict[str, RefineRegion] = {}
        mask_descriptors: Dict[str, Optional[torch.Tensor]] = {}
        mask_patches: Dict[str, Optional[np.ndarray]] = {}
        mask_pixels_by_track: Dict[str, Optional[np.ndarray]] = {}
        base_positions: Dict[str, Tuple[float, float]] = {}
        motion_priors: Dict[str, MotionPrior] = {}
        pixel_flow_by_track: Dict[str, Optional[PixelFlowEstimate]] = {}
        results: List[Dict[str, object]] = []
        for track in self.tracks.values():
            manual_codebook = track.manual_codebook
            if manual_codebook is not None and manual_codebook.device != feats.device:
                manual_codebook = manual_codebook.to(feats.device)
                track.manual_codebook = manual_codebook
            manual_weight = self._manual_anchor_weight(track)
            shared_key, shared_descriptor = self._part_shared_descriptor(
                track, feats.device
            )
            shared_weight = self._part_shared_weight(track, shared_key)
            use_context = (
                context_weight > 0.0
                and context_map is not None
                and track.context_descriptor is not None
            )

            prev_r, prev_c = track.patch_rc
            if (
                roi_changed
                or prev_r < 0
                or prev_r >= grid_h
                or prev_c < 0
                or prev_c >= grid_w
            ):
                prev_r, prev_c = self._pixel_to_patch(
                    track.last_position[0],
                    track.last_position[1],
                    scale_x,
                    scale_y,
                    grid_hw,
                )
                track.patch_rc = (prev_r, prev_c)
            base_x = float(patch_centers_x[prev_c])
            base_y = float(patch_centers_y[prev_r])
            if batched_lk_by_track is not None:
                motion = self._estimate_frame_motion(
                    track,
                    prev_gray=prev_gray,
                    frame_gray=frame_gray,
                    dense_flow=None,
                    dense_flow_provider=get_dense_flow,
                    precomputed_pixel_flow=(True, batched_lk_by_track.get(track.key)),
                )
            else:
                motion = self._estimate_frame_motion(
                    track,
                    prev_gray=prev_gray,
                    frame_gray=frame_gray,
                    dense_flow=None,
                    dense_flow_provider=get_dense_flow,
                )
            pixel_flow_by_track[track.key] = motion.pixel_flow
            predicted_x, predicted_y = motion.predicted_xy
            predicted_r, predicted_c = self._pixel_to_patch(
                predicted_x,
                predicted_y,
                scale_x,
                scale_y,
                grid_hw,
            )

            prior = self._prepare_motion_prior(
                track,
                base_xy=(base_x, base_y),
                predicted_xy=(predicted_x, predicted_y),
                predicted_rc=(predicted_r, predicted_c),
                flow_vec=motion.flow_vec,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            motion_priors[track.key] = prior
            predicted_r, predicted_c = prior.predicted_rc
            radius = prior.radius
            r_min = max(0, predicted_r - radius)
            r_max = min(grid_h - 1, predicted_r + radius)
            c_min = max(0, predicted_c - radius)
            c_max = min(grid_w - 1, predicted_c + radius)
            candidate_list: List[Candidate] = []

            mask_entry = mask_cache.get(track.instance_label)
            patch_mask = mask_entry.get("patch_mask") if mask_entry else None
            mask_descriptor = mask_entry.get("descriptor") if mask_entry else None
            similarity_bonus = (
                mask_entry.get("similarity_bonus", 0.0) if mask_entry else 0.0
            )
            mask_descriptors[track.key] = mask_descriptor
            mask_patches[track.key] = patch_mask
            base_positions[track.key] = (base_x, base_y)
            mask_pixels = active_masks.get(track.instance_label)
            mask_pixels_by_track[track.key] = mask_pixels

            use_appearance = (
                appearance_weight > 0.0 and track.appearance_codebook is not None
            )
            if use_appearance:
                codebook = track.appearance_codebook
                if codebook.device != feats.device:
                    codebook = codebook.to(feats.device)
                    track.appearance_codebook = codebook

            use_struct_penalty = (
                struct_weight > 0.0
                and bool(track.struct_refs)
                and not self._is_fresh_start
            )
            use_symmetry_penalty = symmetry_weight > 0.0
            use_motion_penalty = motion_penalty_weight > 0.0
            use_support = support_weight > 0.0 and bool(track.support_probes)
            if use_context and track.context_descriptor is not None:
                if track.context_descriptor.device != feats.device:
                    track.context_descriptor = track.context_descriptor.to(feats.device)

            track_feats = instance_region_feats.get(
                track.instance_label, default_region_feats
            )
            region_feats = track_feats[:, r_min : r_max + 1, c_min : c_max + 1]
            region_h, region_w = region_feats.shape[1:]
            region_flat = region_feats.reshape(region_feats.shape[0], -1).transpose(
                0, 1
            )
            sims_current = torch.matmul(region_flat, track.descriptor)
            sims = sims_current
            if shared_weight > 0.0 and shared_descriptor is not None:
                sims_shared = torch.matmul(region_flat, shared_descriptor)
                sims = (1.0 - shared_weight) * sims + shared_weight * sims_shared
            if manual_weight > 0.0 and manual_codebook is not None:
                anchor_sims = torch.matmul(region_flat, manual_codebook.transpose(0, 1))
                sims_manual = anchor_sims.max(dim=1).values
                sims = (1.0 - manual_weight) * sims + manual_weight * sims_manual

            mask_region = None
            mask_flat = None
            if patch_mask is not None:
                mask_region = patch_mask[r_min : r_max + 1, c_min : c_max + 1]
                mask_flat = torch.from_numpy(mask_region.reshape(-1)).to(sims.device)
                if self.restrict_to_mask:
                    sims = sims.masked_fill(~mask_flat, float("-inf"))

            region_size = int(sims.numel())
            preselect_k = self._candidate_preselect_count(track, region_size)
            if preselect_k >= region_size:
                candidate_indices = torch.arange(region_size, device=sims.device)
            else:
                candidate_indices = torch.topk(sims, k=preselect_k).indices

            if mask_flat is not None and self.restrict_to_mask:
                if candidate_indices.numel() > 0:
                    valid = mask_flat[candidate_indices]
                    candidate_indices = candidate_indices[valid]

            sims_selected = (
                sims[candidate_indices] if candidate_indices.numel() > 0 else sims[:0]
            )
            sims_current_selected = (
                sims_current[candidate_indices]
                if candidate_indices.numel() > 0
                else sims_current[:0]
            )

            norm_region_flat = region_feats.reshape(region_feats.shape[0], -1)
            candidate_descs = (
                norm_region_flat[:, candidate_indices]
                if candidate_indices.numel() > 0
                else norm_region_flat[:, :0]
            )
            context_scores = None
            if use_context and candidate_indices.numel() > 0:
                context_region_flat = context_map[
                    :, r_min : r_max + 1, c_min : c_max + 1
                ].reshape(context_map.shape[0], -1)
                candidate_contexts = context_region_flat[:, candidate_indices]
                context_scores = torch.matmul(
                    track.context_descriptor, candidate_contexts
                )

            appearance_scores = None
            if use_appearance and candidate_indices.numel() > 0:
                appearance_scores = (
                    torch.matmul(codebook, candidate_descs).max(dim=0).values
                )
                if appearance_weight != 1.0:
                    appearance_scores = appearance_scores * appearance_weight

            candidate_indices_list = candidate_indices.tolist()
            num_sel = len(candidate_indices_list)
            sims_selected_np = (
                sims_selected.detach().cpu().numpy() if num_sel > 0 else None
            )
            sims_current_np = (
                sims_current_selected.detach().cpu().numpy() if num_sel > 0 else None
            )
            appearance_scores_np = (
                appearance_scores.detach().cpu().numpy()
                if appearance_scores is not None and num_sel > 0
                else None
            )
            context_scores_np = (
                context_scores.detach().cpu().numpy()
                if context_scores is not None and num_sel > 0
                else None
            )
            cand_rows_np = (
                np.array(
                    [r_min + int(f // region_w) for f in candidate_indices_list],
                    dtype=np.int64,
                )
                if num_sel > 0
                else np.empty((0,), dtype=np.int64)
            )
            cand_cols_np = (
                np.array(
                    [c_min + int(f % region_w) for f in candidate_indices_list],
                    dtype=np.int64,
                )
                if num_sel > 0
                else np.empty((0,), dtype=np.int64)
            )
            support_scores_np = (
                self._support_scores_batch(
                    track,
                    cand_rows_np,
                    cand_cols_np,
                    track_feats,
                    patch_mask,
                )
                if use_support and num_sel > 0
                else None
            )
            backward_scores_np = (
                self._backward_consistency_scores_batch(
                    candidate_descs,
                    previous_flat=prev_matching_flat,
                    previous_grid_hw=prev_matching_feats.shape[1:],
                    previous_rc=(prev_r, prev_c),
                )
                if use_backward_consistency
                and prev_matching_feats is not None
                and prev_matching_flat is not None
                and num_sel > 0
                else None
            )

            for idx, flat_idx in enumerate(candidate_indices_list):
                local_r = int(flat_idx // region_w)
                local_c = int(flat_idx % region_w)
                if (
                    mask_region is not None
                    and not mask_region[local_r, local_c]
                    and self.restrict_to_mask
                ):
                    continue
                r = int(cand_rows_np[idx])
                c = int(cand_cols_np[idx])
                candidate_sim = float(sims_selected_np[idx])
                candidate_sim_current = float(sims_current_np[idx])
                candidate_score = candidate_sim
                in_mask = False
                if mask_region is not None:
                    in_mask = bool(mask_region[local_r, local_c])
                    if in_mask:
                        candidate_score += similarity_bonus
                candidate_desc = candidate_descs[:, idx]
                candidate_xy = (
                    float(patch_centers_x[c]),
                    float(patch_centers_y[r]),
                )
                if appearance_scores_np is not None:
                    candidate_score += float(appearance_scores_np[idx])
                if context_scores_np is not None:
                    candidate_score += context_weight * float(context_scores_np[idx])
                if use_struct_penalty:
                    candidate_score -= self._structural_penalty(
                        track, candidate_xy, previous_positions
                    )
                if use_body_prior:
                    candidate_score -= self._body_prior_penalty(
                        track,
                        (r, c),
                        body_priors.get(track.instance_label),
                    )
                if use_symmetry_penalty:
                    candidate_score -= self._symmetry_penalty(track, candidate_xy)
                if baseline_weight > 0.0:
                    candidate_score -= baseline_weight * max(
                        0.0, track.baseline_similarity - candidate_sim_current
                    )
                if use_motion_penalty:
                    candidate_score -= self._motion_prior_penalty(
                        track,
                        candidate_xy,
                        motion_priors.get(track.key),
                        similarity=float(candidate_sim),
                    )
                if support_scores_np is not None:
                    candidate_score += float(support_scores_np[idx])
                if backward_scores_np is not None:
                    candidate_score += float(backward_scores_np[idx])
                if tracking_enabled and not self._is_fresh_start:
                    candidate_score -= self._multi_animal_instance_penalty(
                        track, candidate_xy, instance_context
                    )
                candidate_list.append(
                    Candidate(
                        rc=(r, c),
                        xy=candidate_xy,
                        score=candidate_score,
                        similarity=float(candidate_sim),
                        descriptor=candidate_desc,
                    )
                )

            refine_region: Optional[RefineRegion] = None
            if self.keypoint_refine_radius > 0 or use_cluster_refine:
                refine_logits = sims.reshape(region_h, region_w).detach().cpu().numpy()
                if mask_region is not None:
                    refine_logits = refine_logits + mask_region.astype(
                        np.float32
                    ) * float(similarity_bonus)
                valid_mask = None
                if mask_region is not None and self.restrict_to_mask:
                    valid_mask = mask_region.astype(bool, copy=False)
                refine_region = RefineRegion(
                    logits=refine_logits,
                    r_min=r_min,
                    c_min=c_min,
                    valid_mask=valid_mask,
                )

            if tracking_enabled and not self._is_fresh_start and candidate_list:
                candidate_list = self._filter_candidates_by_tracking_constraints(
                    track,
                    candidate_list,
                    previous_positions,
                )
            candidate_list.sort(key=lambda c: c.score, reverse=True)
            track_candidates[track.key] = candidate_list[: self.max_candidates]
            if refine_region is not None:
                refine_regions[track.key] = refine_region

        assignments = self._resolve_assignments(track_candidates)

        for track in self.tracks.values():
            base_x, base_y = base_positions.get(track.key, (0.0, 0.0))
            assignment = assignments.get(track.key)
            mask_descriptor = mask_descriptors.get(track.key)
            patch_mask = mask_patches.get(track.key)
            mask_pixels = mask_pixels_by_track.get(track.key)
            pixel_flow = pixel_flow_by_track.get(track.key)
            assignment_prior = motion_priors.get(track.key)

            quality = float(assignment.similarity) if assignment else -1.0
            if assignment is None or quality < effective_min_similarity:
                track.misses += 1
                visible = False
                x, y = base_x, base_y
                self._handle_track_miss_limits(track, tracking_enabled=tracking_enabled)
            else:
                x, y = assignment.xy
                refine_confidence = 0.0
                cluster_refine_confidence = 0.0
                body_prior_rejected = False
                if self.keypoint_refine_radius > 0:
                    refined_x, refined_y, refine_confidence = (
                        self._refine_keypoint_xy_from_region(
                            center_rc=assignment.rc,
                            fallback_xy=(x, y),
                            refine_region=refine_regions.get(track.key),
                            patch_centers_x=patch_centers_x,
                            patch_centers_y=patch_centers_y,
                        )
                    )
                    x, y = refined_x, refined_y
                if use_cluster_refine:
                    refined_x, refined_y, cluster_refine_confidence = (
                        self._refine_keypoint_xy_from_connected_peak(
                            center_rc=assignment.rc,
                            fallback_xy=(x, y),
                            refine_region=refine_regions.get(track.key),
                            patch_centers_x=patch_centers_x,
                            patch_centers_y=patch_centers_y,
                        )
                    )
                    x, y = refined_x, refined_y
                pixel_refine_confidence = 0.0
                if pixel_flow is not None:
                    refined_x, refined_y, pixel_refine_confidence = (
                        self._apply_pixel_flow_refinement(
                            dino_xy=(x, y),
                            flow_estimate=pixel_flow,
                            patch_px=(
                                assignment_prior.radius_px
                                if assignment_prior is not None
                                else 0.0
                            ),
                            similarity=float(quality),
                        )
                    )
                    x, y = refined_x, refined_y
                if use_pixel_region and track.pixel_template is not None:
                    tpl_x, tpl_y, tpl_conf = (
                        self._apply_pixel_region_template_refinement(
                            frame_gray=frame_gray,
                            template=track.pixel_template,
                            current_xy=(x, y),
                        )
                    )
                    x, y = tpl_x, tpl_y
                    pixel_refine_confidence = max(
                        float(pixel_refine_confidence), float(tpl_conf)
                    )
                candidate_visible = True
                if (
                    self.mask_enforce_position
                    and mask_pixels is not None
                    and mask_pixels.size > 0
                ):
                    x, y, inside_mask = self._enforce_mask_position(
                        x,
                        y,
                        mask_pixels,
                        search_radius=self.mask_enforce_snap_radius,
                    )
                    if not inside_mask and self.mask_enforce_reject_outside:
                        candidate_visible = False

                similarity_conf = 0.0
                if candidate_visible:
                    similarity_conf = float(
                        np.clip(
                            (quality - effective_min_similarity)
                            / max(1e-6, 1.0 - effective_min_similarity),
                            0.0,
                            1.0,
                        )
                    )
                    candidate_visible = self._should_accept_jump(
                        track,
                        candidate_xy=(x, y),
                        prior=assignment_prior,
                        similarity_confidence=similarity_conf,
                    )
                    if (
                        candidate_visible
                        and tracking_enabled
                        and not self._is_fresh_start
                    ):
                        candidate_visible = self._satisfies_tracking_constraints(
                            track,
                            candidate_xy=(x, y),
                            previous_positions=previous_positions,
                            instance_context=instance_context,
                        )
                    if candidate_visible:
                        candidate_rc = self._pixel_to_patch(
                            x,
                            y,
                            scale_x,
                            scale_y,
                            grid_hw,
                        )
                        body_prior_accept = self._should_accept_body_prior(
                            track,
                            candidate_rc=candidate_rc,
                            body_prior=body_priors.get(track.instance_label),
                        )
                        body_prior_rejected = not body_prior_accept
                        candidate_visible = candidate_visible and body_prior_accept

                if not candidate_visible:
                    if body_prior_rejected:
                        # Reject the jump but treat it as a hold at the prior spot.
                        visible = True
                        x, y = track.last_position
                        quality = track.quality
                    else:
                        track.misses += 1
                        visible = False
                        x, y = track.last_position
                        quality = 0.0
                        self._handle_track_miss_limits(
                            track, tracking_enabled=tracking_enabled
                        )
                else:
                    track.misses = 0
                    visible = True
                    track.patch_rc = self._pixel_to_patch(
                        x,
                        y,
                        scale_x,
                        scale_y,
                        grid_hw,
                    )
                    track_feats = instance_region_feats.get(
                        track.instance_label, default_region_feats
                    )
                    new_desc = self._sample_point_descriptor(
                        track_feats,
                        xy=(float(x), float(y)),
                        patch_rc=track.patch_rc,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        grid_hw=grid_hw,
                        patch_mask=patch_mask,
                    )
                    combined_conf = max(
                        similarity_conf,
                        float(refine_confidence),
                        float(cluster_refine_confidence),
                        float(pixel_refine_confidence),
                    )
                    effective_momentum = self.momentum * combined_conf

                    blended = (
                        1.0 - effective_momentum
                    ) * track.descriptor + effective_momentum * new_desc
                    if self.reference_weight > 0.0:
                        effective_reference = self.reference_weight + (
                            (1.0 - self.reference_weight) * (1.0 - combined_conf)
                        )
                        blended = (
                            1.0 - effective_reference
                        ) * blended + effective_reference * track.reference_descriptor
                    blended = self._apply_mask_descriptor(blended, mask_descriptor)
                    track.descriptor = blended / (blended.norm() + 1e-12)
                    if context_map is not None:
                        new_context = self._sample_point_descriptor(
                            context_map,
                            xy=(float(x), float(y)),
                            patch_rc=track.patch_rc,
                            scale_x=scale_x,
                            scale_y=scale_y,
                            grid_hw=grid_hw,
                            patch_mask=patch_mask,
                        )
                        if track.context_descriptor is None:
                            track.context_descriptor = new_context
                        else:
                            blended_context = (
                                (1.0 - effective_momentum) * track.context_descriptor
                                + effective_momentum * new_context
                            )
                            track.context_descriptor = blended_context / (
                                blended_context.norm() + 1e-12
                            )
                    self._update_part_shared_descriptor(
                        track,
                        track.descriptor,
                        confidence=combined_conf,
                    )
                    self._update_appearance_codebook(track, new_desc)
                    body_prior = body_priors.get(track.instance_label)
                    if body_prior is not None:
                        current_body = body_prior.signature(
                            (float(track.patch_rc[0]), float(track.patch_rc[1]))
                        )
                        if track.body_coords is None:
                            track.body_coords = current_body
                        else:
                            alpha = 0.1 * float(np.clip(combined_conf, 0.0, 1.0))
                            track.body_coords = (
                                (1.0 - alpha) * track.body_coords[0]
                                + alpha * current_body[0],
                                (1.0 - alpha) * track.body_coords[1]
                                + alpha * current_body[1],
                                (1.0 - alpha) * track.body_coords[2]
                                + alpha * current_body[2],
                            )
                    if track.appearance_codebook is not None:
                        refreshed = torch.matmul(
                            track.appearance_codebook, track.descriptor
                        )
                        track.baseline_similarity = min(
                            1.0,
                            max(
                                track.baseline_similarity,
                                float(refreshed.max().item()),
                            ),
                        )
                    self._refresh_support_probes(
                        track, track_feats, normalized_feats=track_feats
                    )
                    self._update_support_probe_mask_flags(track, patch_mask)
                    velocity_base_x, velocity_base_y = track.last_position
                    delta_x = x - velocity_base_x
                    delta_y = y - velocity_base_y
                    track.velocity = (
                        (1.0 - velocity_smoothing) * track.velocity[0]
                        + velocity_smoothing * delta_x,
                        (1.0 - velocity_smoothing) * track.velocity[1]
                        + velocity_smoothing * delta_y,
                    )

            quality = max(0.0, quality)
            results.append(
                {
                    "id": track.key,
                    "label": track.storage_label,
                    "x": float(x),
                    "y": float(y),
                    "visible": visible,
                    "instance_label": track.instance_label,
                    "velocity": track.velocity,
                    "misses": track.misses,
                    "quality": quality,
                    "confidence": max(0.0, min(1.0, quality)),
                    "symmetry_partner": track.symmetry_partner,
                    "symmetry_sign": track.symmetry_sign,
                }
            )
            if visible:
                track.last_position = (float(x), float(y))
                track.quality = max(0.0, min(1.0, quality))
            else:
                track.quality = max(0.0, min(1.0, quality))
        self._update_symmetry_midpoints()

        if self._is_fresh_start:
            self._is_fresh_start = False  # Clear the fresh start flag
        self._prev_matching_feats = default_region_feats.detach()
        return results

    def apply_external_corrections(
        self,
        corrections: Dict[str, Tuple[float, float]],
        *,
        blend_alpha: Optional[float] = None,
        clear_cache: bool = True,
    ) -> None:
        """Adjust tracker state to externally corrected keypoints."""
        if not corrections:
            return
        feats = self._last_norm_feats
        grid_hw = self._last_grid_hw
        if feats is None or grid_hw is None:
            return
        scale_x, scale_y = self.prev_scale
        alpha = (
            float(blend_alpha)
            if blend_alpha is not None
            else float(getattr(self.runtime_config, "kpseg_blend_alpha", 1.0))
        )
        alpha = float(max(0.0, min(1.0, alpha)))

        for track_key, coord in corrections.items():
            track = self.tracks.get(track_key)
            if track is None:
                continue
            x, y = float(coord[0]), float(coord[1])
            dx = x - float(track.last_position[0])
            dy = y - float(track.last_position[1])
            if (dx * dx + dy * dy) < 1e-6:
                continue
            rr, cc = self._pixel_to_patch(x, y, scale_x, scale_y, grid_hw)
            if rr < 0 or cc < 0:
                continue
            if rr >= feats.shape[1] or cc >= feats.shape[2]:
                continue
            patch_mask = self._last_patch_masks.get(track.instance_label)
            new_desc = self._sample_point_descriptor(
                feats,
                xy=(x, y),
                patch_rc=(rr, cc),
                scale_x=scale_x,
                scale_y=scale_y,
                grid_hw=grid_hw,
                patch_mask=patch_mask,
            )
            if (
                bool(getattr(self.runtime_config, "pixel_region_refine_enabled", False))
                and self.prev_gray is not None
            ):
                refreshed_tpl = extract_gaussian_pixel_template(
                    self.prev_gray,
                    (x, y),
                    window_px=int(
                        getattr(self.runtime_config, "pixel_region_window", 11)
                    ),
                    sigma_px=float(
                        getattr(self.runtime_config, "pixel_region_sigma", 3.0)
                    ),
                )
                if refreshed_tpl is not None:
                    track.pixel_template = refreshed_tpl
            if alpha >= 1.0:
                blended = new_desc
            else:
                blended = (1.0 - alpha) * track.descriptor + alpha * new_desc
            track.descriptor = self._normalize_descriptor(blended)
            track.patch_rc = (rr, cc)
            track.last_position = (x, y)
            track.velocity = (0.0, 0.0)
            track.misses = 0
            track.quality = max(track.quality, 0.0)
            self._update_appearance_codebook(track, new_desc)

        if clear_cache:
            self._last_norm_feats = None
            self._last_grid_hw = None

    def _refine_keypoint_xy(
        self,
        *,
        center_rc: Tuple[int, int],
        fallback_xy: Tuple[float, float],
        candidate_cloud: List[
            Tuple[Tuple[int, int], Tuple[float, float], float, float]
        ],
    ) -> Tuple[float, float, float]:
        radius = int(self.keypoint_refine_radius)
        if radius <= 0 or not candidate_cloud:
            return fallback_xy[0], fallback_xy[1], 0.0

        sigma = max(1e-4, float(self.keypoint_refine_sigma))
        temperature = max(1e-4, float(self.keypoint_refine_temperature))
        inv_sigma_sq = 1.0 / (sigma * sigma)

        selected: List[Tuple[int, int, Tuple[float, float], float]] = []
        for rc, xy, similarity, _score in candidate_cloud:
            dr = rc[0] - center_rc[0]
            dc = rc[1] - center_rc[1]
            if abs(dr) > radius or abs(dc) > radius:
                continue
            selected.append((dr, dc, xy, float(similarity)))
        if not selected:
            return fallback_xy[0], fallback_xy[1], 0.0

        max_similarity = max(item[3] for item in selected)
        total = 0.0
        sum_x = 0.0
        sum_y = 0.0
        max_weight = 0.0
        for dr, dc, xy, similarity in selected:
            dist_sq = float(dr * dr + dc * dc)
            spatial = math.exp(-0.5 * dist_sq * inv_sigma_sq)
            weight = math.exp((float(similarity) - float(max_similarity)) / temperature)
            w = spatial * weight
            max_weight = max(max_weight, w)
            total += w
            sum_x += w * float(xy[0])
            sum_y += w * float(xy[1])
        if total <= 1e-12:
            return fallback_xy[0], fallback_xy[1], 0.0
        confidence = float(np.clip(max_weight / total, 0.0, 1.0))
        return (sum_x / total, sum_y / total, confidence)

    def _refine_keypoint_xy_from_region(
        self,
        *,
        center_rc: Tuple[int, int],
        fallback_xy: Tuple[float, float],
        refine_region: Optional[RefineRegion],
        patch_centers_x: np.ndarray,
        patch_centers_y: np.ndarray,
    ) -> Tuple[float, float, float]:
        radius = int(self.keypoint_refine_radius)
        if radius <= 0 or refine_region is None:
            return fallback_xy[0], fallback_xy[1], 0.0
        return refine_patch_region_gaussian(
            refine_region.logits,
            center_rc=center_rc,
            r_min=refine_region.r_min,
            c_min=refine_region.c_min,
            valid_mask=refine_region.valid_mask,
            radius=radius,
            sigma=float(self.keypoint_refine_sigma),
            temperature=float(self.keypoint_refine_temperature),
            patch_centers_x=patch_centers_x,
            patch_centers_y=patch_centers_y,
            fallback_xy=fallback_xy,
        )

    def _refine_keypoint_xy_from_connected_peak(
        self,
        *,
        center_rc: Tuple[int, int],
        fallback_xy: Tuple[float, float],
        refine_region: Optional[RefineRegion],
        patch_centers_x: np.ndarray,
        patch_centers_y: np.ndarray,
    ) -> Tuple[float, float, float]:
        radius = int(getattr(self.runtime_config, "keypoint_cluster_refine_radius", 0))
        if radius <= 0 or refine_region is None:
            return fallback_xy[0], fallback_xy[1], 0.0

        logits = refine_region.logits
        if logits.size == 0:
            return fallback_xy[0], fallback_xy[1], 0.0

        local_r = int(center_rc[0] - refine_region.r_min)
        local_c = int(center_rc[1] - refine_region.c_min)
        if (
            local_r < 0
            or local_c < 0
            or local_r >= int(logits.shape[0])
            or local_c >= int(logits.shape[1])
        ):
            return fallback_xy[0], fallback_xy[1], 0.0

        valid_mask = refine_region.valid_mask
        if valid_mask is not None and not bool(valid_mask[local_r, local_c]):
            return fallback_xy[0], fallback_xy[1], 0.0

        r_start = max(0, local_r - radius)
        r_end = min(int(logits.shape[0]) - 1, local_r + radius)
        c_start = max(0, local_c - radius)
        c_end = min(int(logits.shape[1]) - 1, local_c + radius)
        center_score = float(logits[local_r, local_c])
        threshold = center_score - float(
            max(0.0, getattr(self.runtime_config, "keypoint_cluster_refine_drop", 0.1))
        )

        stack = [(local_r, local_c)]
        visited: Set[Tuple[int, int]] = set()
        component: List[Tuple[int, int, float]] = []
        while stack:
            r_idx, c_idx = stack.pop()
            key = (r_idx, c_idx)
            if key in visited:
                continue
            visited.add(key)
            if r_idx < r_start or r_idx > r_end or c_idx < c_start or c_idx > c_end:
                continue
            if valid_mask is not None and not bool(valid_mask[r_idx, c_idx]):
                continue
            score = float(logits[r_idx, c_idx])
            if score < threshold:
                continue
            component.append((r_idx, c_idx, score))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r_idx + dr, c_idx + dc))

        if len(component) <= 1:
            return fallback_xy[0], fallback_xy[1], 0.0

        max_similarity = max(item[2] for item in component)
        temperature = max(1e-4, float(self.keypoint_refine_temperature))
        total = 0.0
        sum_x = 0.0
        sum_y = 0.0
        for r_idx, c_idx, similarity in component:
            global_r = int(refine_region.r_min + r_idx)
            global_c = int(refine_region.c_min + c_idx)
            if (
                global_r < 0
                or global_c < 0
                or global_r >= len(patch_centers_y)
                or global_c >= len(patch_centers_x)
            ):
                continue
            weight = math.exp((float(similarity) - float(max_similarity)) / temperature)
            total += weight
            sum_x += weight * float(patch_centers_x[global_c])
            sum_y += weight * float(patch_centers_y[global_r])
        if total <= 1e-12:
            return fallback_xy[0], fallback_xy[1], 0.0

        max_area = float((2 * radius + 1) * (2 * radius + 1))
        confidence = float(np.clip(len(component) / max_area, 0.0, 1.0))
        return (sum_x / total, sum_y / total, confidence)

    def _backward_consistency_score(
        self,
        candidate_desc: torch.Tensor,
        *,
        previous_feats: torch.Tensor,
        previous_rc: Tuple[int, int],
    ) -> float:
        weight = float(
            max(
                0.0,
                getattr(self.runtime_config, "dinov3_backward_consistency_weight", 0.0),
            )
        )
        if weight <= 0.0 or previous_feats.dim() != 3:
            return 0.0
        if candidate_desc.device != previous_feats.device:
            candidate_desc = candidate_desc.to(previous_feats.device)

        prev_flat = previous_feats.reshape(previous_feats.shape[0], -1)
        if prev_flat.numel() == 0:
            return 0.0
        sims = torch.matmul(candidate_desc, prev_flat)
        best_idx = int(torch.argmax(sims).item())
        best_sim = max(0.0, float(sims[best_idx].item()))
        if best_sim <= 0.0:
            return 0.0

        grid_h, grid_w = previous_feats.shape[1:]
        if grid_h <= 0 or grid_w <= 0:
            return 0.0
        best_r = int(best_idx // grid_w)
        best_c = int(best_idx % grid_w)
        dist = math.hypot(
            float(best_r - previous_rc[0]), float(best_c - previous_rc[1])
        )
        tolerance = max(
            0,
            int(
                getattr(
                    self.runtime_config,
                    "dinov3_backward_consistency_tolerance",
                    1,
                )
            ),
        )
        if dist <= float(tolerance):
            return weight * best_sim

        radius = max(1.0, float(self.search_radius))
        penalty = min(1.0, (dist - float(tolerance)) / radius)
        return -weight * best_sim * penalty

    def _backward_consistency_scores_batch(
        self,
        candidate_descs: torch.Tensor,
        *,
        previous_flat: torch.Tensor,
        previous_grid_hw: Tuple[int, int],
        previous_rc: Tuple[int, int],
    ) -> np.ndarray:
        num_cand = int(candidate_descs.shape[1]) if candidate_descs.ndim == 2 else 0
        if num_cand == 0:
            return np.zeros((0,), dtype=np.float64)
        weight = float(
            max(
                0.0,
                getattr(self.runtime_config, "dinov3_backward_consistency_weight", 0.0),
            )
        )
        grid_h, grid_w = int(previous_grid_hw[0]), int(previous_grid_hw[1])
        if weight <= 0.0 or previous_flat.numel() == 0 or grid_h <= 0 or grid_w <= 0:
            return np.zeros((num_cand,), dtype=np.float64)
        if candidate_descs.device != previous_flat.device:
            candidate_descs = candidate_descs.to(previous_flat.device)

        sims = torch.matmul(candidate_descs.transpose(0, 1), previous_flat)
        best_vals, best_indices = torch.max(sims, dim=1)
        best_sim = np.maximum(0.0, best_vals.detach().cpu().numpy().astype(np.float64))
        best_idx_np = best_indices.detach().cpu().numpy()
        best_r = (best_idx_np // grid_w).astype(np.float64)
        best_c = (best_idx_np % grid_w).astype(np.float64)
        dist = np.hypot(best_r - float(previous_rc[0]), best_c - float(previous_rc[1]))
        tolerance = float(
            max(
                0,
                int(
                    getattr(
                        self.runtime_config,
                        "dinov3_backward_consistency_tolerance",
                        1,
                    )
                ),
            )
        )
        radius = max(1.0, float(self.search_radius))
        penalty = np.minimum(1.0, np.maximum(0.0, dist - tolerance) / radius)
        scores = np.where(
            best_sim <= 0.0,
            0.0,
            np.where(
                dist <= tolerance, weight * best_sim, -weight * best_sim * penalty
            ),
        )
        return scores

    def _is_default_pixel_flow_method(self) -> bool:
        return (
            "_track_pixel_flow" not in self.__dict__
            and getattr(type(self), "_track_pixel_flow", None)
            is DinoKeypointTracker._track_pixel_flow
        )

    def _track_pixel_flow_batch(
        self,
        prev_gray: Optional[np.ndarray],
        frame_gray: np.ndarray,
        tracks: List[KeypointTrack],
    ) -> Dict[str, Optional[PixelFlowEstimate]]:
        out: Dict[str, Optional[PixelFlowEstimate]] = {t.key: None for t in tracks}
        options = self._get_pixel_refine_options()
        if (
            not options.enabled
            or prev_gray is None
            or frame_gray is None
            or prev_gray.shape != frame_gray.shape
        ):
            return out
        height, width = frame_gray.shape[:2]
        valid_tracks: List[Tuple[str, float, float]] = []
        for trk in tracks:
            x, y = float(trk.last_position[0]), float(trk.last_position[1])
            if 0.0 <= x < width and 0.0 <= y < height:
                valid_tracks.append((trk.key, x, y))
        if not valid_tracks:
            return out

        prev_pts = np.array([[[x, y]] for _, x, y in valid_tracks], dtype=np.float32)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            prev_pts,
            None,
            winSize=(options.window, options.window),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if next_pts is None or status is None:
            return out

        next_xy = next_pts.reshape(-1, 2)
        status_flat = status.reshape(-1)
        err_flat = err.reshape(-1) if err is not None else None
        for idx, (key, x, y) in enumerate(valid_tracks):
            if int(status_flat[idx]) != 1:
                continue
            nx = float(next_xy[idx, 0])
            ny = float(next_xy[idx, 1])
            if not (math.isfinite(nx) and math.isfinite(ny)):
                continue
            if not (0.0 <= nx < width and 0.0 <= ny < height):
                continue
            jump = math.hypot(nx - x, ny - y)
            if options.max_jump_px > 0.0 and jump > options.max_jump_px:
                continue
            error = 0.0
            if err_flat is not None:
                error = float(err_flat[idx])
                if not math.isfinite(error):
                    continue
                if options.max_error > 0.0 and error > options.max_error:
                    continue
            out[key] = PixelFlowEstimate(xy=(nx, ny), error=error)
        return out

    def _estimate_frame_motion(
        self,
        track: KeypointTrack,
        *,
        prev_gray: Optional[np.ndarray],
        frame_gray: np.ndarray,
        dense_flow: Optional[np.ndarray] = None,
        dense_flow_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        precomputed_pixel_flow: Optional[
            Tuple[bool, Optional[PixelFlowEstimate]]
        ] = None,
    ) -> FrameMotionEstimate:
        """Estimate the current point without counting the same motion twice.

        A valid per-point LK estimate is already a current-frame position. When
        LK is unavailable, dense flow is sampled at the last emitted point and
        applied once (computing dense flow lazily when ``dense_flow_provider``
        is supplied). Stored velocity is a fallback for frames without a valid
        flow observation, rather than an additional displacement.
        """
        last_x, last_y = (float(value) for value in track.last_position)
        if precomputed_pixel_flow is not None and precomputed_pixel_flow[0]:
            pixel_flow = precomputed_pixel_flow[1]
        else:
            pixel_flow = self._track_pixel_flow(
                prev_gray,
                frame_gray,
                (last_x, last_y),
            )
        if pixel_flow is not None:
            flow_vec = (
                float(pixel_flow.xy[0] - last_x),
                float(pixel_flow.xy[1] - last_y),
            )
            return FrameMotionEstimate(
                predicted_xy=(float(pixel_flow.xy[0]), float(pixel_flow.xy[1])),
                flow_vec=flow_vec,
                pixel_flow=pixel_flow,
            )

        if dense_flow is None and dense_flow_provider is not None:
            dense_flow = dense_flow_provider()
        if dense_flow is not None and dense_flow.ndim >= 3 and dense_flow.shape[2] >= 2:
            height, width = dense_flow.shape[:2]
            sample_x = math.floor(last_x)
            sample_y = math.floor(last_y)
            if 0 <= sample_x < width and 0 <= sample_y < height:
                flow_dx = float(dense_flow[sample_y, sample_x, 0])
                flow_dy = float(dense_flow[sample_y, sample_x, 1])
                if math.isfinite(flow_dx) and math.isfinite(flow_dy):
                    return FrameMotionEstimate(
                        predicted_xy=(last_x + flow_dx, last_y + flow_dy),
                        flow_vec=(flow_dx, flow_dy),
                        pixel_flow=None,
                    )

        velocity_dx, velocity_dy = (float(value) for value in track.velocity)
        return FrameMotionEstimate(
            predicted_xy=(last_x + velocity_dx, last_y + velocity_dy),
            flow_vec=None,
            pixel_flow=None,
        )

    def _track_pixel_flow(
        self,
        prev_gray: Optional[np.ndarray],
        frame_gray: np.ndarray,
        xy: Tuple[float, float],
    ) -> Optional[PixelFlowEstimate]:
        options = self._get_pixel_refine_options()
        if not options.enabled:
            return None
        if prev_gray is None or frame_gray is None:
            return None
        if prev_gray.shape != frame_gray.shape:
            return None
        height, width = frame_gray.shape[:2]
        x, y = float(xy[0]), float(xy[1])
        if not (0.0 <= x < width and 0.0 <= y < height):
            return None

        prev_pts = np.array([[[x, y]]], dtype=np.float32)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            prev_pts,
            None,
            winSize=(options.window, options.window),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
        if next_pts is None or status is None or int(status.reshape(-1)[0]) != 1:
            return None

        nx = float(next_pts.reshape(-1, 2)[0][0])
        ny = float(next_pts.reshape(-1, 2)[0][1])
        if not (math.isfinite(nx) and math.isfinite(ny)):
            return None
        if not (0.0 <= nx < width and 0.0 <= ny < height):
            return None

        jump = math.hypot(nx - x, ny - y)
        if options.max_jump_px > 0.0 and jump > options.max_jump_px:
            return None

        error = 0.0
        if err is not None:
            error = float(err.reshape(-1)[0])
            if not math.isfinite(error):
                return None
            if options.max_error > 0.0 and error > options.max_error:
                return None
        return PixelFlowEstimate(xy=(nx, ny), error=error)

    def _apply_pixel_flow_refinement(
        self,
        *,
        dino_xy: Tuple[float, float],
        flow_estimate: PixelFlowEstimate,
        patch_px: float,
        similarity: float,
    ) -> Tuple[float, float, float]:
        options = self._get_pixel_refine_options()
        weight = float(options.weight)
        if weight <= 0.0:
            return dino_xy[0], dino_xy[1], 0.0

        flow_x, flow_y = flow_estimate.xy
        dino_x, dino_y = float(dino_xy[0]), float(dino_xy[1])
        dist = math.hypot(flow_x - dino_x, flow_y - dino_y)
        gate = max(3.0, float(patch_px) * 1.5)
        if dist > gate:
            return dino_x, dino_y, 0.0

        similarity_conf = float(np.clip((float(similarity) + 1.0) * 0.5, 0.0, 1.0))
        error_conf = options.error_confidence(flow_estimate.error)
        confidence = max(0.0, min(1.0, 0.5 * (similarity_conf + error_conf)))
        alpha = weight * confidence
        refined_x = (1.0 - alpha) * dino_x + alpha * flow_x
        refined_y = (1.0 - alpha) * dino_y + alpha * flow_y
        return float(refined_x), float(refined_y), float(confidence)

    def _apply_pixel_region_template_refinement(
        self,
        *,
        frame_gray: np.ndarray,
        template: GaussianPixelTemplate,
        current_xy: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        weight = float(
            np.clip(getattr(self.runtime_config, "pixel_region_weight", 0.35), 0.0, 1.0)
        )
        if weight <= 0.0:
            return float(current_xy[0]), float(current_xy[1]), 0.0
        search_r = max(
            1, int(getattr(self.runtime_config, "pixel_region_search_radius_px", 4))
        )
        min_zncc = float(
            np.clip(
                getattr(self.runtime_config, "pixel_region_min_zncc", 0.45), 0.0, 0.99
            )
        )
        match = match_gaussian_pixel_template(
            frame_gray,
            template,
            center_xy=current_xy,
            search_radius_px=search_r,
        )
        if match is None:
            return float(current_xy[0]), float(current_xy[1]), 0.0
        zncc_x, zncc_y, peak_zncc = match
        if peak_zncc < min_zncc:
            return float(current_xy[0]), float(current_xy[1]), 0.0
        conf = float(
            np.clip((peak_zncc - min_zncc) / max(1e-6, 1.0 - min_zncc), 0.0, 1.0)
        )
        alpha = weight * (0.4 + 0.6 * conf)
        out_x = (1.0 - alpha) * float(current_xy[0]) + alpha * float(zncc_x)
        out_y = (1.0 - alpha) * float(current_xy[1]) + alpha * float(zncc_y)
        return float(out_x), float(out_y), float(conf)

    def _should_accept_jump(
        self,
        track: KeypointTrack,
        *,
        candidate_xy: Tuple[float, float],
        prior: Optional[MotionPrior],
        similarity_confidence: float,
    ) -> bool:
        if prior is None:
            return True

        soft_radius_px = float(self.runtime_config.motion_prior_soft_radius_px)
        radius_factor = float(self.runtime_config.motion_prior_radius_factor)
        hard_limit = max(soft_radius_px, 1.0) * max(radius_factor, 1.0)
        hard_limit *= 1.0 + 0.5 * max(0, int(track.misses))

        dx = float(candidate_xy[0] - prior.predicted_xy[0])
        dy = float(candidate_xy[1] - prior.predicted_xy[1])
        dist = math.hypot(dx, dy)

        if dist <= hard_limit:
            return True

        # If appearance evidence is strong, allow larger jumps (fast motion/reacquisition).
        if similarity_confidence >= 0.75:
            return True

        # Otherwise, reject large jumps to prevent identity swaps on occlusion frames.
        return False

    def _should_accept_body_prior(
        self,
        track: KeypointTrack,
        *,
        candidate_rc: Tuple[int, int],
        body_prior: Optional[BodyPrior],
    ) -> bool:
        weight = float(max(0.0, self.runtime_config.structural_consistency_weight))
        if weight <= 0.0:
            return True
        if body_prior is None or track.body_coords is None:
            return True

        cand_t, cand_p, cand_th = body_prior.signature(
            (float(candidate_rc[0]), float(candidate_rc[1]))
        )
        ref_t, ref_p, ref_th = track.body_coords
        sigma_t = max(1.0, float(body_prior.sigma_axis))
        sigma_p = max(1.0, float(body_prior.sigma_perp))
        sigma_th = max(1.0, float(body_prior.sigma_thickness))
        dt = (cand_t - ref_t) / sigma_t
        dp = (cand_p - ref_p) / sigma_p
        dth = (cand_th - ref_th) / sigma_th
        d2 = dt * dt + dp * dp + dth * dth
        # 2-sigma gate in body coordinates: prevents keypoints from hopping to
        # a different anatomical region (ear -> tail) when appearance is ambiguous.
        return bool(d2 <= 4.0)

    def _body_prior_penalty(
        self,
        track: KeypointTrack,
        candidate_rc: Tuple[int, int],
        body_prior: Optional[BodyPrior],
    ) -> float:
        weight = float(max(0.0, self.runtime_config.structural_consistency_weight))
        if weight <= 0.0 or body_prior is None or track.body_coords is None:
            return 0.0

        cand_t, cand_p, cand_th = body_prior.signature(
            (float(candidate_rc[0]), float(candidate_rc[1]))
        )
        ref_t, ref_p, ref_th = track.body_coords
        sigma_t = max(1.0, float(body_prior.sigma_axis))
        sigma_p = max(1.0, float(body_prior.sigma_perp))
        sigma_th = max(1.0, float(body_prior.sigma_thickness))
        dt = (cand_t - ref_t) / sigma_t
        dp = (cand_p - ref_p) / sigma_p
        dth = (cand_th - ref_th) / sigma_th
        d2 = dt * dt + dp * dp + dth * dth

        penalty = weight * (1.0 - math.exp(-0.5 * float(d2)))
        if track.struct_refs:
            penalty *= 0.35
        relief = 1.0 / (1.0 + 0.5 * max(0, int(track.misses)))
        return penalty * relief

    def _compute_body_priors(
        self,
        mask_cache: Dict[str, Dict[str, object]],
    ) -> Dict[str, BodyPrior]:
        priors: Dict[str, BodyPrior] = {}
        for label, entry in mask_cache.items():
            patch_mask = entry.get("patch_mask")
            if patch_mask is None:
                continue
            prior = self._compute_body_prior(label=label, patch_mask=patch_mask)
            if prior is not None:
                priors[label] = prior
        return priors

    def _compute_body_prior(
        self,
        *,
        label: str,
        patch_mask: np.ndarray,
    ) -> Optional[BodyPrior]:
        if patch_mask is None or patch_mask.size == 0:
            return None
        ys, xs = np.nonzero(patch_mask)
        if xs.size < 16:
            return None

        mask_u8 = patch_mask.astype(np.uint8) * 255
        thickness_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        thickness_values = thickness_map[patch_mask]
        sigma_thickness = float(np.std(thickness_values))
        if not math.isfinite(sigma_thickness) or sigma_thickness < 1e-6:
            sigma_thickness = float(np.median(thickness_values))
        sigma_thickness = max(1.0, float(sigma_thickness))

        coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        cov = (centered.T @ centered) / float(max(1, coords.shape[0] - 1))
        eigvals, eigvecs = np.linalg.eigh(cov)

        axis = eigvecs[:, 1]
        axis_norm = float(np.linalg.norm(axis) + 1e-6)
        axis = axis / axis_norm
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)

        prev_axis = self._instance_body_axes.get(label)
        if prev_axis is not None:
            dot = float(axis[0] * prev_axis[0] + axis[1] * prev_axis[1])
            if dot < 0.0:
                axis = -axis
                perp = -perp
        self._instance_body_axes[label] = (float(axis[0]), float(axis[1]))

        sigma_axis = float(np.sqrt(max(float(eigvals[1]), 1e-6)))
        sigma_perp = float(np.sqrt(max(float(eigvals[0]), 1e-6)))

        return BodyPrior(
            centroid_rc=(float(centroid[0]), float(centroid[1])),
            axis_rc=(float(axis[0]), float(axis[1])),
            perp_rc=(float(perp[0]), float(perp[1])),
            sigma_axis=max(1.0, sigma_axis),
            sigma_perp=max(1.0, sigma_perp),
            thickness_map=thickness_map.astype(np.float32, copy=False),
            sigma_thickness=float(sigma_thickness),
        )

    @staticmethod
    def _normalize_descriptor(
        descriptor: torch.Tensor,
        *,
        clone: bool = True,
    ) -> torch.Tensor:
        if clone:
            descriptor = descriptor.detach().clone()
        else:
            descriptor = descriptor.detach()
        return descriptor / (descriptor.norm() + 1e-12)

    @staticmethod
    def _normalize_feature_grid(feats: torch.Tensor) -> torch.Tensor:
        return normalize_feature_grid(feats)

    def _prepare_matching_feature_grid(self, feats: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_feature_grid(feats)
        if not bool(getattr(self.runtime_config, "dinov3_positional_debias", False)):
            return normalized
        strength = float(
            np.clip(
                getattr(self.runtime_config, "dinov3_positional_debias_strength", 0.0),
                0.0,
                1.0,
            )
        )
        if strength <= 0.0:
            return normalized
        components = max(
            1,
            int(getattr(self.runtime_config, "dinov3_positional_debias_components", 6)),
        )
        return self._positionally_debias_feature_grid(
            normalized,
            components=components,
            strength=strength,
        )

    def _positionally_debias_feature_grid(
        self,
        feats: torch.Tensor,
        *,
        components: int,
        strength: float,
    ) -> torch.Tensor:
        """Suppress low-dimensional positional DINO responses."""
        if feats.dim() != 3:
            return feats
        channels, grid_h, grid_w = feats.shape
        if channels <= 0 or grid_h * grid_w <= 2:
            return feats
        svd_basis = self._svd_positional_debias_basis(
            feats,
            components=components,
        )
        if svd_basis is not None and svd_basis.numel() > 0:
            return self._apply_channel_debias_basis(
                feats,
                svd_basis,
                strength=strength,
            )

        basis = coordinate_debias_basis(
            grid_h,
            grid_w,
            components=components,
            device=feats.device,
            dtype=feats.dtype,
        )
        if basis is None or basis.numel() == 0:
            return feats

        flat = feats.reshape(channels, -1).transpose(0, 1)
        positional_component = basis @ (basis.transpose(0, 1) @ flat)
        debiased = flat - float(strength) * positional_component
        debiased = debiased.transpose(0, 1).reshape_as(feats)
        return self._normalize_feature_grid(debiased)

    def _svd_positional_debias_basis(
        self,
        feats: torch.Tensor,
        *,
        components: int,
    ) -> Optional[torch.Tensor]:
        """Build an INSID3-style channel basis from a blank DINO response."""
        channels, grid_h, grid_w = feats.shape
        roi_w, roi_h = self._roi_size
        if roi_w <= 0 or roi_h <= 0:
            roi_w = max(1, int(grid_w * self.patch_size))
            roi_h = max(1, int(grid_h * self.patch_size))
        model_key = str(getattr(getattr(self.extractor, "cfg", None), "model_name", ""))
        basis = svd_positional_basis(
            extractor=self.extractor,
            feature_shape=(int(channels), int(grid_h), int(grid_w)),
            patch_size=int(self.patch_size),
            components=int(components),
            cache=self._svd_positional_basis_cache,
            model_key=model_key,
            image_size=(int(roi_w), int(roi_h)),
        )
        if basis is None:
            return None
        return basis.to(device=feats.device, dtype=feats.dtype)

    def _apply_channel_debias_basis(
        self,
        feats: torch.Tensor,
        basis: torch.Tensor,
        *,
        strength: float,
    ) -> torch.Tensor:
        return apply_channel_debias_basis(feats, basis, strength=strength)

    @staticmethod
    def _avg_pool_descriptor_map(
        feats: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        if radius <= 0:
            return feats
        kernel = 2 * radius + 1
        pooled = F.avg_pool2d(
            feats.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )
        return pooled.squeeze(0)

    def _compute_context_map(
        self,
        normalized_feats: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        weight = float(max(0.0, getattr(self.runtime_config, "context_weight", 0.0)))
        radius = int(getattr(self.runtime_config, "context_radius", 0))
        if weight <= 0.0 or radius <= 0:
            return None
        use_gauss = bool(
            getattr(self.runtime_config, "context_gaussian_pooling", False)
        )
        if use_gauss:
            small = gaussian_smooth_feature_grid(
                normalized_feats,
                radius=radius,
                sigma=max(0.6, 0.65 * float(radius)),
                mask_aware=False,
                blend=1.0,
                normalize=False,
            )
        else:
            small = self._avg_pool_descriptor_map(normalized_feats, radius)
        large_radius = int(getattr(self.runtime_config, "context_radius_large", radius))
        if large_radius > radius:
            if use_gauss:
                large = gaussian_smooth_feature_grid(
                    normalized_feats,
                    radius=large_radius,
                    sigma=max(0.8, 0.65 * float(large_radius)),
                    mask_aware=False,
                    blend=1.0,
                    normalize=False,
                )
            else:
                large = self._avg_pool_descriptor_map(normalized_feats, large_radius)
            blend = float(
                np.clip(
                    getattr(self.runtime_config, "context_large_weight", 0.0),
                    0.0,
                    1.0,
                )
            )
            combined = (1.0 - blend) * small + blend * large
        else:
            combined = small
        return self._normalize_feature_grid(combined)

    def _resolve_instance_region_feats(
        self,
        matching_feats: torch.Tensor,
        mask_cache: Dict[str, Dict[str, object]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        radius = max(
            0, int(getattr(self.runtime_config, "region_descriptor_radius", 0))
        )
        blend = float(
            np.clip(
                getattr(self.runtime_config, "region_descriptor_blend", 0.5), 0.0, 1.0
            )
        )
        if radius <= 0 or blend <= 0.0:
            return matching_feats, {}
        sigma = max(
            1e-4, float(getattr(self.runtime_config, "region_descriptor_sigma", 0.85))
        )
        mask_aware = bool(
            getattr(self.runtime_config, "region_descriptor_mask_aware", True)
        )
        default_feats = gaussian_smooth_feature_grid(
            matching_feats,
            radius=radius,
            sigma=sigma,
            patch_mask=None,
            mask_aware=False,
            blend=blend,
            normalize=True,
        )
        per_instance: Dict[str, torch.Tensor] = {}
        if mask_aware and mask_cache:
            for inst_label, entry in mask_cache.items():
                patch_mask = entry.get("patch_mask") if entry else None
                if patch_mask is None:
                    continue
                per_instance[inst_label] = gaussian_smooth_feature_grid(
                    matching_feats,
                    radius=radius,
                    sigma=sigma,
                    patch_mask=patch_mask,
                    mask_aware=True,
                    blend=blend,
                    normalize=True,
                )
        return default_feats, per_instance

    def _sample_point_descriptor(
        self,
        feats: torch.Tensor,
        *,
        xy: Tuple[float, float],
        patch_rc: Tuple[int, int],
        scale_x: float,
        scale_y: float,
        grid_hw: Tuple[int, int],
        patch_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if not bool(getattr(self.runtime_config, "region_subpatch_sampling", False)):
            return feats[:, patch_rc[0], patch_rc[1]].detach().clone()
        cont_rc = pixel_to_continuous_patch(
            xy[0],
            xy[1],
            roi_offset=self._roi_offset,
            roi_size=self._roi_size,
            scale_x=scale_x,
            scale_y=scale_y,
            patch_size=float(self.patch_size),
            grid_hw=grid_hw,
        )
        radius = max(
            1, int(getattr(self.runtime_config, "region_descriptor_radius", 1) or 1)
        )
        sigma = max(
            0.35, float(getattr(self.runtime_config, "region_descriptor_sigma", 0.85))
        )
        mask_aware = bool(
            getattr(self.runtime_config, "region_descriptor_mask_aware", True)
        )
        return sample_subpatch_gaussian_descriptor(
            feats,
            cont_rc=cont_rc,
            radius=radius,
            sigma=sigma,
            patch_mask=patch_mask,
            mask_aware=mask_aware,
        )

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        return extract_feature_grid(self.extractor, image)

    def _prepare_roi_inputs(
        self,
        image: Image.Image,
        mask_lookup: Optional[Dict[str, np.ndarray]],
        polygons: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> Tuple[torch.Tensor, float, float, Tuple[int, int]]:
        width, height = image.size
        roi = self._determine_roi(
            width=width,
            height=height,
            mask_lookup=mask_lookup,
            polygons=polygons,
        )
        if self._roi_box != roi:
            self._last_patch_masks = {}
            self._mask_miss_counts = {}
        left, top, right, bottom = roi
        cropped_image = image.crop((left, top, right, bottom))
        feats = self._extract_features(cropped_image)
        new_h, new_w = self.extractor._compute_resized_hw(*cropped_image.size)
        roi_w = max(1, cropped_image.width)
        roi_h = max(1, cropped_image.height)
        scale_x = new_w / roi_w
        scale_y = new_h / roi_h
        self._roi_offset = (float(left), float(top))
        self._roi_box = roi
        self._roi_size = (roi_w, roi_h)
        return feats, scale_x, scale_y, feats.shape[1:]

    def _determine_roi(
        self,
        width: int,
        height: int,
        mask_lookup: Optional[Dict[str, np.ndarray]],
        polygons: Optional[List[List[Tuple[float, float]]]],
    ) -> Tuple[int, int, int, int]:
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        found = False

        if mask_lookup:
            for mask in mask_lookup.values():
                if mask is None:
                    continue
                ys, xs = np.nonzero(mask)
                if xs.size == 0 or ys.size == 0:
                    continue
                min_x = min(min_x, int(xs.min()))
                min_y = min(min_y, int(ys.min()))
                max_x = max(max_x, int(xs.max()))
                max_y = max(max_y, int(ys.max()))
                found = True

        if not found and polygons:
            for polygon in polygons:
                if not polygon:
                    continue
                xs, ys = zip(*polygon)
                min_x = min(min_x, int(math.floor(min(xs))))
                min_y = min(min_y, int(math.floor(min(ys))))
                max_x = max(max_x, int(math.ceil(max(xs))))
                max_y = max(max_y, int(math.ceil(max(ys))))
                found = True

        if not found:
            prev_left, prev_top, prev_right, prev_bottom = self._roi_box
            if (
                self.tracks
                and 0 <= prev_left < prev_right <= width
                and 0 <= prev_top < prev_bottom <= height
            ):
                return (
                    int(prev_left),
                    int(prev_top),
                    int(prev_right),
                    int(prev_bottom),
                )
            return (0, 0, width, height)

        margin = max(self.patch_size * 2, 16)
        left = max(0, min_x - margin)
        top = max(0, min_y - margin)
        right = min(width, max_x + margin + 1)
        bottom = min(height, max_y + margin + 1)

        min_width = max(1, self.patch_size)
        min_height = max(1, self.patch_size)

        if right - left < min_width:
            center = (left + right) / 2.0
            left = int(max(0, math.floor(center - min_width / 2)))
            right = int(min(width, left + min_width))
        if bottom - top < min_height:
            center = (top + bottom) / 2.0
            top = int(max(0, math.floor(center - min_height / 2)))
            bottom = int(min(height, top + min_height))

        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)

        return (int(left), int(top), int(right), int(bottom))

    def _crop_masks_for_roi(
        self,
        mask_lookup: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        if not mask_lookup:
            return {}
        left, top, right, bottom = self._roi_box
        cropped: Dict[str, np.ndarray] = {}
        for label, mask in mask_lookup.items():
            if mask is None:
                continue
            h, w = mask.shape[:2]
            y0 = max(0, min(top, h))
            y1 = max(0, min(bottom, h))
            x0 = max(0, min(left, w))
            x1 = max(0, min(right, w))
            if y0 >= y1 or x0 >= x1:
                continue
            region = mask[y0:y1, x0:x1]
            if region.size == 0:
                continue
            cropped[label] = region.astype(bool)
        return cropped

    def _pixel_to_patch(
        self,
        x: float,
        y: float,
        scale_x: float,
        scale_y: float,
        grid_hw: Tuple[int, int],
    ) -> Tuple[int, int]:
        grid_h, grid_w = grid_hw
        offset_x, offset_y = self._roi_offset
        roi_w, roi_h = self._roi_size
        local_x = x - offset_x
        local_y = y - offset_y
        if roi_w > 0:
            local_x = min(max(local_x, 0.0), float(roi_w - 1))
        if roi_h > 0:
            local_y = min(max(local_y, 0.0), float(roi_h - 1))
        resized_x = local_x * scale_x
        resized_y = local_y * scale_y
        patch_c = int(resized_x / self.patch_size)
        patch_r = int(resized_y / self.patch_size)
        patch_c = max(0, min(grid_w - 1, patch_c))
        patch_r = max(0, min(grid_h - 1, patch_r))
        return patch_r, patch_c

    def _patch_to_pixel(
        self,
        patch_rc: Tuple[int, int],
        scale_x: float,
        scale_y: float,
    ) -> Tuple[float, float]:
        r, c = patch_rc
        center_resized_x = (c + 0.5) * self.patch_size
        center_resized_y = (r + 0.5) * self.patch_size
        local_x = center_resized_x / scale_x if scale_x else 0.0
        local_y = center_resized_y / scale_y if scale_y else 0.0
        return (
            local_x + self._roi_offset[0],
            local_y + self._roi_offset[1],
        )

    def _initialize_structural_refs(self, registry: InstanceRegistry) -> None:
        for instance in registry:
            keypoints = {kp.key: kp for kp in instance.keypoints.values()}
            for key, kp in keypoints.items():
                track = self.tracks.get(key)
                if track is None:
                    continue
                struct_refs: Dict[str, float] = {}
                for other_key, other in keypoints.items():
                    if other_key == key:
                        continue
                    dist = math.hypot(other.x - kp.x, other.y - kp.y)
                    if dist > 1e-6:
                        struct_refs[other_key] = dist
                track.struct_refs = struct_refs

    def _initialize_symmetry_refs(self, registry: InstanceRegistry) -> None:
        pairs = getattr(self.runtime_config, "symmetry_pairs", ()) or ()
        symmetry_penalty = float(
            getattr(self.runtime_config, "symmetry_penalty", 0.0) or 0.0
        )
        if not pairs and symmetry_penalty <= 0.0:
            return
        lower_pairs = {
            (str(left).strip().lower(), str(right).strip().lower())
            for left, right in pairs
            if str(left).strip() and str(right).strip()
        }
        allow_inferred_lr = bool(
            (not lower_pairs and symmetry_penalty > 0.0)
            or ("left", "right") in lower_pairs
            or ("right", "left") in lower_pairs
            or ("l", "r") in lower_pairs
            or ("r", "l") in lower_pairs
        )

        def _bind_pair(left_track: KeypointTrack, right_track: KeypointTrack) -> None:
            if (
                left_track.key == right_track.key
                or left_track.symmetry_partner is not None
                or right_track.symmetry_partner is not None
            ):
                return
            left_pos = left_track.last_position
            right_pos = right_track.last_position
            axis = (
                left_pos[0] - right_pos[0],
                left_pos[1] - right_pos[1],
            )
            norm = math.hypot(axis[0], axis[1])
            if norm <= 1e-6:
                return
            axis_unit = (axis[0] / norm, axis[1] / norm)
            midpoint = (
                (left_pos[0] + right_pos[0]) / 2.0,
                (left_pos[1] + right_pos[1]) / 2.0,
            )
            left_track.symmetry_axis = axis_unit
            left_track.symmetry_midpoint = midpoint
            left_track.symmetry_sign = 1.0
            left_track.symmetry_partner = right_track.key
            right_track.symmetry_axis = axis_unit
            right_track.symmetry_midpoint = midpoint
            right_track.symmetry_sign = -1.0
            right_track.symmetry_partner = left_track.key

        for instance in registry:
            label_to_track: Dict[str, KeypointTrack] = {}
            lower_to_track: Dict[str, KeypointTrack] = {}
            for kp in instance.keypoints.values():
                track = self.tracks.get(kp.key)
                if track:
                    label_to_track[kp.label] = track
                    lower_to_track.setdefault(kp.label.lower(), track)

            for left_label, right_label in pairs:
                left_track = label_to_track.get(left_label) or lower_to_track.get(
                    str(left_label).strip().lower()
                )
                right_track = label_to_track.get(right_label) or lower_to_track.get(
                    str(right_label).strip().lower()
                )
                if not left_track or not right_track:
                    continue
                _bind_pair(left_track, right_track)

            if allow_inferred_lr and len(label_to_track) >= 2:
                unpaired_labels = [
                    lbl
                    for lbl, trk in label_to_track.items()
                    if trk.symmetry_partner is None
                ]
                for left_label, right_label in PoseSchema.infer_symmetry_pairs(
                    unpaired_labels
                ):
                    left_track = label_to_track.get(left_label)
                    right_track = label_to_track.get(right_label)
                    if left_track and right_track:
                        _bind_pair(left_track, right_track)

    def _structural_penalty(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
        previous_positions: Dict[str, Tuple[float, float]],
    ) -> float:
        weight = float(max(0.0, self.runtime_config.structural_consistency_weight))
        if weight <= 0.0 or not track.struct_refs:
            return 0.0
        if self._is_fresh_start:
            return 0.0

        total = 0.0
        count = 0
        for other_key, ref_distance in track.struct_refs.items():
            other_pos = previous_positions.get(other_key)
            if other_pos is None or ref_distance <= 1e-6:
                continue
            current_distance = math.hypot(
                candidate_xy[0] - other_pos[0],
                candidate_xy[1] - other_pos[1],
            )
            ratio = abs(current_distance - ref_distance) / ref_distance
            total += ratio
            count += 1
        if count == 0:
            return 0.0
        return weight * (total / count)

    def _symmetry_penalty(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
    ) -> float:
        penalty = float(max(0.0, self.runtime_config.symmetry_penalty))
        if penalty <= 0.0:
            return 0.0

        # If a track has been missed, its position is unreliable. The symmetry constraint,
        # which depends on the partner's position, is likely stale. We disable the
        # penalty to allow the tracker to re-acquire the keypoint based on appearance alone.
        # Once found, misses will be reset to 0 and the symmetry logic will resume.
        if track.misses > 0:
            return 0.0

        if self._is_fresh_start:
            # On the first frame after a start/restart, the symmetry references may be inaccurate.
            # We disable the penalty for this single frame to avoid misguiding the tracker.
            return 0.0

        axis = track.symmetry_axis
        midpoint = track.symmetry_midpoint
        if axis is None or midpoint is None:
            return 0.0
        vec = (
            candidate_xy[0] - midpoint[0],
            candidate_xy[1] - midpoint[1],
        )
        magnitude = math.hypot(vec[0], vec[1])
        if magnitude <= 1e-6:
            return 0.0
        dot = vec[0] * axis[0] + vec[1] * axis[1]
        if abs(dot) < 1e-6:
            return penalty * 0.5
        sign = 1.0 if dot >= 0 else -1.0
        if sign == track.symmetry_sign:
            return 0.0
        perp = vec[0] * (-axis[1]) + vec[1] * axis[0]
        severity = min(1.0, abs(perp) / (magnitude + 1e-6))
        return penalty * (1.0 + 0.5 * severity)

    def _appearance_score(
        self, track: KeypointTrack, candidate_desc: torch.Tensor
    ) -> float:
        weight = float(max(0.0, self.runtime_config.appearance_bundle_weight))
        if weight <= 0.0 or track.appearance_codebook is None:
            return 0.0
        sims = torch.matmul(track.appearance_codebook, candidate_desc)
        return float(sims.max().item()) * weight

    def _support_score(
        self,
        track: KeypointTrack,
        candidate_rc: Tuple[int, int],
        feats: torch.Tensor,
        patch_mask: Optional[np.ndarray],
        *,
        normalized_feats: Optional[torch.Tensor] = None,
    ) -> float:
        weight = float(max(0.0, self.runtime_config.support_probe_weight))
        if weight <= 0.0 or not track.support_probes:
            return 0.0
        grid_h, grid_w = feats.shape[1:]
        total_weight = 0.0
        accum = 0.0
        penalties = 0.0
        mask_bonus = float(self.runtime_config.support_probe_mask_bonus)
        for probe in track.support_probes:
            probe_weight = max(1e-6, float(probe.weight))
            total_weight += probe_weight
            sr = candidate_rc[0] + probe.offset_rc[0]
            sc = candidate_rc[1] + probe.offset_rc[1]
            if not (0 <= sr < grid_h and 0 <= sc < grid_w):
                penalties += probe_weight
                continue
            if normalized_feats is None:
                desc = self._normalize_descriptor(feats[:, sr, sc])
            else:
                desc = normalized_feats[:, sr, sc]
            descriptor = probe.descriptor
            if descriptor.device != desc.device:
                descriptor = descriptor.to(desc.device)
                probe.descriptor = descriptor
            sim = torch.dot(desc, descriptor).item()
            accum += probe_weight * sim
            if patch_mask is not None and patch_mask[sr, sc]:
                accum += probe_weight * mask_bonus
        if total_weight <= 0.0:
            return 0.0
        normalized = accum / total_weight
        penalty_ratio = penalties / total_weight if total_weight else 0.0
        return weight * (normalized - penalty_ratio)

    def _support_scores_batch(
        self,
        track: KeypointTrack,
        candidate_rows: np.ndarray,
        candidate_cols: np.ndarray,
        normalized_feats: torch.Tensor,
        patch_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        num_cand = int(candidate_rows.shape[0])
        if num_cand == 0:
            return np.zeros((0,), dtype=np.float64)
        weight = float(max(0.0, self.runtime_config.support_probe_weight))
        probes = track.support_probes
        if weight <= 0.0 or not probes:
            return np.zeros((num_cand,), dtype=np.float64)

        grid_h, grid_w = int(normalized_feats.shape[1]), int(normalized_feats.shape[2])
        probe_weights = np.array(
            [max(1e-6, float(p.weight)) for p in probes], dtype=np.float64
        )
        total_weight = float(np.sum(probe_weights))
        if total_weight <= 0.0:
            return np.zeros((num_cand,), dtype=np.float64)

        offsets_r = np.array([int(p.offset_rc[0]) for p in probes], dtype=np.int64)
        offsets_c = np.array([int(p.offset_rc[1]) for p in probes], dtype=np.int64)
        sr = candidate_rows[:, None] + offsets_r[None, :]
        sc = candidate_cols[:, None] + offsets_c[None, :]
        in_bounds = (sr >= 0) & (sr < grid_h) & (sc >= 0) & (sc < grid_w)

        sr_clamped = np.clip(sr, 0, max(0, grid_h - 1))
        sc_clamped = np.clip(sc, 0, max(0, grid_w - 1))

        feat_dev = normalized_feats.device
        probe_descs_list: List[torch.Tensor] = []
        for p in probes:
            d = p.descriptor
            if d.device != feat_dev:
                d = d.to(feat_dev)
                p.descriptor = d
            probe_descs_list.append(d)
        probe_mat = torch.stack(probe_descs_list, dim=0)  # [P, C]

        sr_t = torch.from_numpy(sr_clamped).to(device=feat_dev, dtype=torch.long)
        sc_t = torch.from_numpy(sc_clamped).to(device=feat_dev, dtype=torch.long)
        sampled = normalized_feats[:, sr_t, sc_t].permute(1, 2, 0)  # [K, P, C]
        sims_np = (
            (sampled * probe_mat.unsqueeze(0)).sum(dim=-1).detach().cpu().numpy()
        ).astype(np.float64)

        mask_bonus = float(self.runtime_config.support_probe_mask_bonus)
        if patch_mask is not None and mask_bonus != 0.0:
            in_mask = in_bounds & patch_mask[sr_clamped, sc_clamped].astype(bool)
            sims_np = sims_np + in_mask.astype(np.float64) * mask_bonus

        valid_sims = np.where(in_bounds, sims_np, 0.0)
        accum = valid_sims @ probe_weights
        penalties = (~in_bounds).astype(np.float64) @ probe_weights
        return weight * ((accum - penalties) / total_weight)

    def _prepare_motion_prior(
        self,
        track: KeypointTrack,
        *,
        base_xy: Tuple[float, float],
        predicted_xy: Tuple[float, float],
        predicted_rc: Tuple[int, int],
        flow_vec: Optional[Tuple[float, float]],
        scale_x: float,
        scale_y: float,
    ) -> MotionPrior:
        min_radius = max(1.0, float(self.runtime_config.motion_search_min_radius))
        max_radius = max(
            min_radius, float(self.runtime_config.motion_search_max_radius)
        )
        tighten = float(np.clip(self.runtime_config.motion_search_tighten, 0.1, 2.0))
        base_radius = max(min_radius, float(self.search_radius) * tighten)

        patch_px_x = self.patch_size / max(scale_x, 1e-6)
        patch_px_y = self.patch_size / max(scale_y, 1e-6)
        patch_px = max(1.0, 0.5 * (patch_px_x + patch_px_y))

        flow_dx = flow_vec[0] if flow_vec else 0.0
        flow_dy = flow_vec[1] if flow_vec else 0.0
        flow_speed = math.hypot(flow_dx, flow_dy)
        velocity_speed = math.hypot(track.velocity[0], track.velocity[1])
        flow_gain = max(0.0, float(self.runtime_config.motion_search_flow_gain))
        velocity_gain = max(0.0, float(self.runtime_config.motion_search_gain))
        motion_px = (flow_speed * flow_gain) + (velocity_speed * velocity_gain)
        effective_patch_px = max(patch_px, 1e-6)
        radius_f = base_radius + (motion_px / effective_patch_px)

        miss_boost = max(0.0, float(self.runtime_config.motion_search_miss_boost))
        radius_f += miss_boost * track.misses
        radius_f = max(min_radius, min(max_radius, radius_f))

        radius = max(1, int(math.ceil(radius_f)))
        radius_px = max(patch_px, radius_f * patch_px)

        confidence = 1.0 / (1.0 + track.misses)
        if flow_vec is None:
            confidence *= 0.5
        confidence *= 0.5 + 0.5 * track.quality
        confidence = float(np.clip(confidence, 0.1, 1.0))

        return MotionPrior(
            predicted_xy=predicted_xy,
            predicted_rc=(int(predicted_rc[0]), int(predicted_rc[1])),
            base_xy=base_xy,
            flow_vec=flow_vec,
            flow_speed=float(flow_speed),
            velocity_speed=float(velocity_speed),
            radius=radius,
            radius_px=radius_px,
            confidence=confidence,
        )

    def _motion_prior_penalty(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
        prior: Optional[MotionPrior],
        *,
        similarity: Optional[float] = None,
    ) -> float:
        weight = float(max(0.0, self.runtime_config.motion_prior_penalty_weight))
        if weight <= 0.0 or prior is None:
            return 0.0
        soft_radius_px = max(
            1.0,
            float(self.runtime_config.motion_prior_soft_radius_px),
        )
        radius_factor = max(1.0, float(self.runtime_config.motion_prior_radius_factor))
        soft_radius_px = max(soft_radius_px, prior.radius_px * radius_factor)

        dx = candidate_xy[0] - prior.predicted_xy[0]
        dy = candidate_xy[1] - prior.predicted_xy[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return 0.0

        scaled = dist / soft_radius_px
        distance_factor = 1.0 - math.exp(-scaled * scaled)

        feature_factor = 1.0
        if similarity is not None:
            similarity_clamped = float(np.clip(similarity, -1.0, 1.0))
            feature_distance = float(np.clip(1.0 - similarity_clamped, 0.0, 2.0))
            feature_factor = 0.5 + 0.25 * feature_distance

        penalty = weight * distance_factor * feature_factor

        miss_relief = 1.0 / (
            1.0
            + track.misses
            * max(0.0, float(self.runtime_config.motion_prior_miss_relief))
        )
        combined = 0.5 * (prior.confidence + miss_relief)

        flow_relief_gain = max(
            0.0, float(getattr(self.runtime_config, "motion_prior_flow_relief", 0.0))
        )
        if flow_relief_gain > 0.0:
            flow_speed = getattr(prior, "flow_speed", 0.0)
            velocity_speed = getattr(prior, "velocity_speed", 0.0)
            speed_sum = max(0.0, float(flow_speed + velocity_speed))
            combined *= 1.0 / (1.0 + speed_sum * flow_relief_gain)

        combined = float(np.clip(combined, 0.1, 1.0))
        return penalty * combined

    def _baseline_penalty(self, track: KeypointTrack, similarity: float) -> float:
        weight = float(max(0.0, self.runtime_config.baseline_similarity_weight))
        if weight <= 0.0:
            return 0.0
        delta = track.baseline_similarity - similarity
        return weight * max(0.0, delta)

    @staticmethod
    def snap_point_to_mask(
        x: float,
        y: float,
        mask: np.ndarray,
        *,
        search_radius: int,
    ) -> Tuple[float, float, bool]:
        height, width = mask.shape

        def clamp_index(value: float, upper: int) -> int:
            return int(min(max(math.floor(value), 0), upper - 1))

        floored_x = math.floor(x)
        floored_y = math.floor(y)
        cx = clamp_index(x, width)
        cy = clamp_index(y, height)
        if floored_x == cx and floored_y == cy and mask[cy, cx]:
            return x, y, True

        best_xy: Optional[Tuple[float, float]] = None

        for radius in range(1, search_radius + 1):
            y_min = max(0, cy - radius)
            y_max = min(height - 1, cy + radius)
            x_min = max(0, cx - radius)
            x_max = min(width - 1, cx + radius)
            region = mask[y_min : y_max + 1, x_min : x_max + 1]
            if not region.any():
                continue
            ys, xs = np.nonzero(region)
            xs = xs + x_min
            ys = ys + y_min
            diff_x = xs.astype(np.float32) - float(x)
            diff_y = ys.astype(np.float32) - float(y)
            dist_sq = diff_x * diff_x + diff_y * diff_y
            idx = int(np.argmin(dist_sq))
            best_xy = (float(xs[idx]), float(ys[idx]))
            break

        if best_xy is not None:
            snapped_x = min(max(best_xy[0] + 0.001, 0.0), width - 1.0)
            snapped_y = min(max(best_xy[1] + 0.001, 0.0), height - 1.0)
            return snapped_x, snapped_y, True

        return x, y, False

    def _enforce_mask_position(
        self,
        x: float,
        y: float,
        mask: np.ndarray,
        *,
        search_radius: int,
    ) -> Tuple[float, float, bool]:
        return self.snap_point_to_mask(x, y, mask, search_radius=search_radius)

    def _collect_appearance_codebook(
        self,
        feats: torch.Tensor,
        patch_rc: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        radius = max(0, int(self.runtime_config.appearance_bundle_radius))
        max_samples = max(1, int(self.runtime_config.appearance_bundle_size))
        grid_h, grid_w = feats.shape[1:]
        descriptors: List[torch.Tensor] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = min(max(patch_rc[0] + dr, 0), grid_h - 1)
                cc = min(max(patch_rc[1] + dc, 0), grid_w - 1)
                desc = self._normalize_descriptor(feats[:, rr, cc])
                descriptors.append(desc)
                if len(descriptors) >= max_samples:
                    break
            if len(descriptors) >= max_samples:
                break
        if not descriptors:
            return None
        return torch.stack(descriptors, dim=0)

    def _sample_support_probes(
        self,
        track_key: str,
        feats: torch.Tensor,
        patch_rc: Tuple[int, int],
        grid_hw: Tuple[int, int],
        patch_mask: Optional[np.ndarray],
    ) -> List[SupportProbe]:
        count = max(0, int(self.runtime_config.support_probe_count))
        if count <= 0:
            return []
        sigma = max(1e-4, float(self.runtime_config.support_probe_sigma))
        radius = max(1, int(self.runtime_config.support_probe_radius))
        mask_only = bool(self.runtime_config.support_probe_mask_only)
        rng = np.random.default_rng(abs(hash(track_key)) & 0xFFFFFFFF)
        probes: List[SupportProbe] = []
        used: Set[Tuple[int, int]] = set()
        grid_h, grid_w = grid_hw
        max_attempts = max(20, count * 20)

        def try_sample(enforce_mask: bool) -> None:
            attempts = 0
            while len(probes) < count and attempts < max_attempts:
                attempts += 1
                offsets = rng.normal(loc=0.0, scale=sigma, size=2)
                dr = int(round(float(offsets[0])))
                dc = int(round(float(offsets[1])))
                if dr == 0 and dc == 0:
                    continue
                if abs(dr) > radius or abs(dc) > radius:
                    continue
                sr = patch_rc[0] + dr
                sc = patch_rc[1] + dc
                if not (0 <= sr < grid_h and 0 <= sc < grid_w):
                    continue
                offset_tuple = (dr, dc)
                if offset_tuple in used:
                    continue
                if patch_mask is not None and enforce_mask and not patch_mask[sr, sc]:
                    continue
                descriptor = self._normalize_descriptor(feats[:, sr, sc])
                distance_sq = (dr / sigma) ** 2 + (dc / sigma) ** 2
                weight = math.exp(-0.5 * distance_sq)
                in_mask = bool(patch_mask is not None and patch_mask[sr, sc])
                probes.append(
                    SupportProbe(
                        offset_rc=offset_tuple,
                        descriptor=descriptor,
                        weight=weight,
                        in_mask=in_mask,
                    )
                )
                used.add(offset_tuple)

        try_sample(mask_only)
        if len(probes) < count and mask_only:
            try_sample(False)

        if len(probes) < count:
            fallback_offsets = [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]
            for dr, dc in fallback_offsets:
                if len(probes) >= count:
                    break
                sr = patch_rc[0] + dr
                sc = patch_rc[1] + dc
                if not (0 <= sr < grid_h and 0 <= sc < grid_w):
                    continue
                if (dr, dc) in used:
                    continue
                if patch_mask is not None and mask_only and not patch_mask[sr, sc]:
                    continue
                descriptor = self._normalize_descriptor(feats[:, sr, sc])
                in_mask = bool(patch_mask is not None and patch_mask[sr, sc])
                probes.append(
                    SupportProbe(
                        offset_rc=(dr, dc),
                        descriptor=descriptor,
                        weight=1.0,
                        in_mask=in_mask,
                    )
                )
                used.add((dr, dc))
        return probes

    def _refresh_support_probes(
        self,
        track: KeypointTrack,
        feats: torch.Tensor,
        *,
        normalized_feats: Optional[torch.Tensor] = None,
    ) -> None:
        if not track.support_probes:
            return
        grid_h, grid_w = feats.shape[1:]
        for probe in track.support_probes:
            sr = track.patch_rc[0] + probe.offset_rc[0]
            sc = track.patch_rc[1] + probe.offset_rc[1]
            if 0 <= sr < grid_h and 0 <= sc < grid_w:
                if normalized_feats is None:
                    probe.descriptor = self._normalize_descriptor(feats[:, sr, sc])
                else:
                    probe.descriptor = normalized_feats[:, sr, sc].detach().clone()

    def _update_support_probe_mask_flags(
        self,
        track: KeypointTrack,
        patch_mask: Optional[np.ndarray],
    ) -> None:
        if not track.support_probes or patch_mask is None:
            return
        grid_h, grid_w = patch_mask.shape
        for probe in track.support_probes:
            sr = track.patch_rc[0] + probe.offset_rc[0]
            sc = track.patch_rc[1] + probe.offset_rc[1]
            probe.in_mask = 0 <= sr < grid_h and 0 <= sc < grid_w and patch_mask[sr, sc]

    def _update_appearance_codebook(
        self,
        track: KeypointTrack,
        descriptor: torch.Tensor,
    ) -> None:
        descriptor = self._normalize_descriptor(descriptor)
        max_samples = max(1, int(self.runtime_config.appearance_bundle_size))
        if track.appearance_codebook is None:
            track.appearance_codebook = descriptor.unsqueeze(0)
            return
        codebook = track.appearance_codebook
        if descriptor.device != codebook.device:
            descriptor = descriptor.to(codebook.device)
        similarity = torch.matmul(codebook, descriptor)
        if bool((similarity > 0.999).any().item()):
            return
        updated = torch.cat([descriptor.unsqueeze(0), codebook], dim=0)
        if updated.shape[0] > max_samples:
            updated = updated[:max_samples]
        track.appearance_codebook = updated

    def _update_manual_anchor_codebook(
        self, track_key: str, descriptor: torch.Tensor
    ) -> None:
        desc = self._normalize_descriptor(descriptor)
        desc = desc.to("cpu")

        codebook = self._manual_anchor_codebooks.get(track_key)
        if codebook is None:
            self._manual_anchor_codebooks[track_key] = desc.unsqueeze(0)
            return

        similarity = torch.matmul(codebook, desc)
        if bool((similarity > 0.999).any().item()):
            return

        updated = torch.cat([desc.unsqueeze(0), codebook], dim=0)
        max_samples = max(1, int(self.runtime_config.appearance_bundle_size))
        if updated.shape[0] > max_samples:
            updated = updated[:max_samples]
        self._manual_anchor_codebooks[track_key] = updated

    def _manual_anchor_weight(self, track: KeypointTrack) -> float:
        if track.manual_codebook is None or self.reference_weight <= 0.0:
            return 0.0
        base = float(np.clip(self.reference_weight, 0.0, 1.0))
        boost = 0.1 * min(5.0, float(max(0, int(track.misses))))
        boost += 0.25 * (1.0 - float(np.clip(track.quality, 0.0, 1.0)))
        return float(np.clip(base + boost, 0.0, 0.95))

    def _part_shared_key(self, track: KeypointTrack) -> Optional[str]:
        label = (track.display_label or "").strip()
        return label or None

    def _part_shared_descriptor(
        self,
        track: KeypointTrack,
        device: torch.device,
    ) -> Tuple[Optional[str], Optional[torch.Tensor]]:
        key = self._part_shared_key(track)
        if not key:
            return None, None
        descriptor = self._part_shared_descriptors.get(key)
        if descriptor is None:
            return key, None
        if descriptor.device != device:
            descriptor = descriptor.to(device)
        return key, descriptor

    def _part_shared_weight(self, track: KeypointTrack, key: Optional[str]) -> float:
        base = float(max(0.0, getattr(self.runtime_config, "part_shared_weight", 0.0)))
        if base <= 0.0 or key is None:
            return 0.0
        count = int(self._part_shared_counts.get(key, 0))
        if count <= 0:
            return 0.0
        quality = float(np.clip(track.quality, 0.0, 1.0))
        boost = 0.1 * min(4.0, float(max(0, track.misses)))
        boost += 0.25 * (1.0 - quality)
        warmup = min(1.0, count / 3.0)
        weight = (base + boost) * warmup
        return float(np.clip(weight, 0.0, 0.95))

    def _update_part_shared_descriptor(
        self,
        track: KeypointTrack,
        descriptor: torch.Tensor,
        *,
        confidence: float,
    ) -> None:
        key = self._part_shared_key(track)
        if not key:
            return
        momentum = float(
            max(0.0, getattr(self.runtime_config, "part_shared_momentum", 0.0))
        )
        if momentum <= 0.0:
            return
        conf = float(np.clip(confidence, 0.0, 1.0))
        if conf < 0.1:
            return
        desc = self._normalize_descriptor(descriptor).to("cpu")
        existing = self._part_shared_descriptors.get(key)
        if existing is None:
            self._part_shared_descriptors[key] = desc
            self._part_shared_counts[key] = 1
            return
        if existing.device != desc.device:
            existing = existing.to(desc.device)
        adaptive = momentum * (0.25 + 0.75 * conf)
        blended = (1.0 - adaptive) * existing + adaptive * desc
        self._part_shared_descriptors[key] = blended / (blended.norm() + 1e-12)
        self._part_shared_counts[key] = self._part_shared_counts.get(key, 1) + 1

    def _candidate_preselect_count(self, track: KeypointTrack, region_size: int) -> int:
        if region_size <= 0:
            return 0
        ratio = float(
            max(
                0.0,
                min(1.0, getattr(self.runtime_config, "candidate_prune_ratio", 1.0)),
            )
        )
        min_candidates = max(
            0, int(getattr(self.runtime_config, "candidate_prune_min", 0))
        )
        if ratio >= 1.0:
            return region_size
        if track.misses > 0:
            return region_size
        quality = float(np.clip(track.quality, 0.0, 1.0))
        dynamic_ratio = ratio + (1.0 - ratio) * (1.0 - quality)
        preselect = int(math.ceil(region_size * dynamic_ratio))
        preselect = max(preselect, min_candidates, self.max_candidates)
        return min(region_size, preselect)

    def _initialize_instance_spans(
        self,
        registry: InstanceRegistry,
        mask_lookup: Optional[Dict[str, np.ndarray]],
    ) -> None:
        self._instance_ref_span = {}
        masks = mask_lookup or {}
        for instance in registry:
            mask = masks.get(instance.label)
            if mask is not None and np.any(mask):
                area = float(np.count_nonzero(mask))
                if area > 0.0:
                    self._instance_ref_span[instance.label] = math.sqrt(area)
                    continue
            pts = [
                (float(kp.x), float(kp.y))
                for kp in instance.keypoints.values()
                if kp.visible and math.isfinite(kp.x) and math.isfinite(kp.y)
            ]
            if len(pts) >= 2:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                rms = math.sqrt(
                    sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / len(pts)
                )
                if rms > 1e-3:
                    self._instance_ref_span[instance.label] = rms

    def _prepare_multi_animal_instance_context(
        self,
        previous_positions: Dict[str, Tuple[float, float]],
        active_masks: Dict[str, Optional[np.ndarray]],
    ) -> Dict[str, Dict[str, object]]:
        tracks_by_inst: Dict[str, List[KeypointTrack]] = {}
        for trk in self.tracks.values():
            tracks_by_inst.setdefault(trk.instance_label, []).append(trk)

        context: Dict[str, Dict[str, object]] = {}
        for inst_label, inst_tracks in tracks_by_inst.items():
            reliable_tracks = [t for t in inst_tracks if t.misses == 0] or inst_tracks
            prev_pts = [
                previous_positions.get(t.key, t.last_position) for t in reliable_tracks
            ]
            prev_cx = sum(p[0] for p in prev_pts) / max(1, len(prev_pts))
            prev_cy = sum(p[1] for p in prev_pts) / max(1, len(prev_pts))

            pred_pts = [
                (
                    previous_positions.get(t.key, t.last_position)[0] + t.velocity[0],
                    previous_positions.get(t.key, t.last_position)[1] + t.velocity[1],
                )
                for t in reliable_tracks
            ]
            pred_cx = sum(p[0] for p in pred_pts) / max(1, len(pred_pts))
            pred_cy = sum(p[1] for p in pred_pts) / max(1, len(pred_pts))

            mask = active_masks.get(inst_label)
            target_centroid = (pred_cx, pred_cy)
            size_ratio_dev = 0.0
            has_mask = False
            if mask is not None and np.any(mask):
                ys, xs = np.nonzero(mask)
                target_centroid = (float(xs.mean()), float(ys.mean()))
                has_mask = True
                ref_span = self._instance_ref_span.get(inst_label, 0.0)
                if ref_span > 1e-3:
                    cur_span = math.sqrt(float(xs.size))
                    size_ratio_dev = abs(cur_span - ref_span) / ref_span

            offsets: Dict[str, Tuple[float, float]] = {}
            offsets_norm: Dict[str, float] = {}
            for trk in inst_tracks:
                px, py = previous_positions.get(trk.key, trk.last_position)
                dx = float(px - prev_cx)
                dy = float(py - prev_cy)
                offsets[trk.key] = (dx, dy)
                offsets_norm[trk.key] = math.hypot(dx, dy)

            context[inst_label] = {
                "target_centroid": target_centroid,
                "offsets": offsets,
                "offsets_norm": offsets_norm,
                "num_tracks": len(inst_tracks),
                "has_mask": has_mask,
                "size_ratio_dev": size_ratio_dev,
            }
        return context

    def _multi_animal_instance_penalty(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
        instance_context: Dict[str, Dict[str, object]],
    ) -> float:
        ctx = instance_context.get(track.instance_label)
        if not ctx:
            return 0.0
        scale_px = max(
            1.0,
            float(
                getattr(
                    self.runtime_config, "tracking_instance_match_distance_px", 150.0
                )
            ),
        )
        centroid_w = max(
            0.0, float(getattr(self.runtime_config, "tracking_centroid_weight", 0.0))
        )
        pose_w = max(
            0.0, float(getattr(self.runtime_config, "tracking_pose_weight", 0.0))
        )
        size_w = max(
            0.0, float(getattr(self.runtime_config, "tracking_size_weight", 0.0))
        )

        target_cx, target_cy = ctx["target_centroid"]  # type: ignore[misc]
        offsets: Dict[str, Tuple[float, float]] = ctx["offsets"]  # type: ignore[assignment]
        expected_dx, expected_dy = offsets.get(track.key, (0.0, 0.0))
        num_tracks = int(ctx.get("num_tracks", 1))  # type: ignore[arg-type]
        has_mask = bool(ctx.get("has_mask", False))

        cand_dx = float(candidate_xy[0] - target_cx)
        cand_dy = float(candidate_xy[1] - target_cy)
        centroid_excess = max(
            0.0, math.hypot(cand_dx, cand_dy) - math.hypot(expected_dx, expected_dy)
        )
        penalty = 0.1 * centroid_w * min(2.0, centroid_excess / scale_px)

        if has_mask or num_tracks >= 2:
            pose_err = math.hypot(cand_dx - expected_dx, cand_dy - expected_dy)
            penalty += 0.1 * pose_w * min(2.0, pose_err / scale_px)

        size_dev = float(ctx.get("size_ratio_dev", 0.0))  # type: ignore[arg-type]
        if size_dev > 0.0 and size_w > 0.0:
            penalty += 0.05 * size_w * min(2.0, size_dev)

        relief = 1.0 / (1.0 + 0.5 * max(0, int(track.misses)))
        return penalty * relief

    def _skeletal_stretch_ratio(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
        previous_positions: Dict[str, Tuple[float, float]],
    ) -> Optional[float]:
        if not track.struct_refs:
            return None
        total = 0.0
        count = 0
        for other_key, ref_distance in track.struct_refs.items():
            if ref_distance <= 1e-6:
                continue
            other_track = self.tracks.get(other_key)
            if other_track is not None and other_track.misses > 0:
                continue
            other_pos = previous_positions.get(other_key)
            if other_pos is None:
                continue
            current_distance = math.hypot(
                candidate_xy[0] - other_pos[0],
                candidate_xy[1] - other_pos[1],
            )
            total += abs(current_distance - ref_distance) / ref_distance
            count += 1
        if count == 0:
            return None
        return total / count

    def _filter_candidates_by_tracking_constraints(
        self,
        track: KeypointTrack,
        candidates: List[Candidate],
        previous_positions: Dict[str, Tuple[float, float]],
    ) -> List[Candidate]:
        max_vel = max(
            1.0,
            float(
                getattr(
                    self.runtime_config, "tracking_constraint_max_velocity_px", 200.0
                )
            ),
        )
        allowed_step = max_vel * max(1, int(track.misses) + 1)
        enforce_skel = bool(
            getattr(self.runtime_config, "tracking_enforce_skeletal_constraints", True)
        ) and bool(track.struct_refs)
        max_rel_dev = 0.25 * max(
            1.0,
            float(
                getattr(
                    self.runtime_config, "tracking_constraint_limb_std_multiplier", 3.0
                )
            ),
        )

        valid: List[Candidate] = []
        last_x, last_y = track.last_position
        for cand in candidates:
            step = math.hypot(cand.xy[0] - last_x, cand.xy[1] - last_y)
            if step > allowed_step:
                continue
            if enforce_skel:
                stretch = self._skeletal_stretch_ratio(
                    track, cand.xy, previous_positions
                )
                if stretch is not None and stretch > max_rel_dev:
                    continue
            valid.append(cand)
        return valid if valid else candidates

    def _satisfies_tracking_constraints(
        self,
        track: KeypointTrack,
        *,
        candidate_xy: Tuple[float, float],
        previous_positions: Dict[str, Tuple[float, float]],
        instance_context: Dict[str, Dict[str, object]],
    ) -> bool:
        max_vel = max(
            1.0,
            float(
                getattr(
                    self.runtime_config, "tracking_constraint_max_velocity_px", 200.0
                )
            ),
        )
        allowed_step = max_vel * max(1, int(track.misses) + 1)
        step_dist = math.hypot(
            candidate_xy[0] - track.last_position[0],
            candidate_xy[1] - track.last_position[1],
        )
        if step_dist > allowed_step:
            return False

        ctx = instance_context.get(track.instance_label)
        if ctx and (bool(ctx.get("has_mask")) or int(ctx.get("num_tracks", 1)) >= 2):  # type: ignore[arg-type]
            target_cx, target_cy = ctx["target_centroid"]  # type: ignore[misc]
            offsets_norm: Dict[str, float] = ctx["offsets_norm"]  # type: ignore[assignment]
            match_dist = max(
                1.0,
                float(
                    getattr(
                        self.runtime_config,
                        "tracking_instance_match_distance_px",
                        150.0,
                    )
                ),
            )
            allowed_inst = max(match_dist, offsets_norm.get(track.key, 0.0) * 1.5) * (
                1.0 + 0.25 * max(0, int(track.misses))
            )
            if (
                math.hypot(candidate_xy[0] - target_cx, candidate_xy[1] - target_cy)
                > allowed_inst
            ):
                return False

        if bool(
            getattr(self.runtime_config, "tracking_enforce_skeletal_constraints", True)
        ) and bool(track.struct_refs):
            max_rel_dev = 0.25 * max(
                1.0,
                float(
                    getattr(
                        self.runtime_config,
                        "tracking_constraint_limb_std_multiplier",
                        3.0,
                    )
                ),
            )
            stretch = self._skeletal_stretch_ratio(
                track, candidate_xy, previous_positions
            )
            if stretch is not None and stretch > max_rel_dev:
                return False

        return True

    def _handle_track_miss_limits(
        self,
        track: KeypointTrack,
        *,
        tracking_enabled: bool,
    ) -> None:
        if track.misses > self.max_misses:
            track.velocity = (0.0, 0.0)
        if tracking_enabled:
            dropout_timeout = max(
                1,
                int(
                    getattr(
                        self.runtime_config,
                        "tracking_instance_dropout_timeout_frames",
                        30,
                    )
                ),
            )
            if track.misses >= dropout_timeout:
                track.velocity = (0.0, 0.0)
                track.descriptor = track.reference_descriptor.detach().clone()

    def _resolve_assignments(
        self, track_candidates: Dict[str, List[Candidate]]
    ) -> Dict[str, Optional[Candidate]]:
        if (
            bool(getattr(self.runtime_config, "tracking_enable", False))
            and str(
                getattr(self.runtime_config, "tracking_matcher_algorithm", "greedy")
            )
            .strip()
            .lower()
            == "hungarian"
        ):
            return self._resolve_assignments_hungarian(track_candidates)

        candidate_pool = {
            key: list(candidates) for key, candidates in track_candidates.items()
        }
        assignments: Dict[str, Optional[Candidate]] = {}
        rc_owner: Dict[Tuple[int, int], str] = {}

        def assign(track_key: str) -> None:
            candidates = candidate_pool.get(track_key, [])
            while candidates:
                candidate = candidates.pop(0)
                rc_key = candidate.rc
                owner = rc_owner.get(rc_key)
                if owner is None:
                    assignments[track_key] = candidate
                    rc_owner[rc_key] = track_key
                    return
                existing_candidate = assignments.get(owner)
                if existing_candidate is None:
                    assignments[track_key] = candidate
                    rc_owner[rc_key] = track_key
                    return
                if candidate.score > existing_candidate.score:
                    assignments[track_key] = candidate
                    rc_owner[rc_key] = track_key
                    assignments.pop(owner, None)
                    assign(owner)
                    return
            assignments[track_key] = None

        order = sorted(
            track_candidates.keys(),
            key=lambda k: track_candidates[k][0].score
            if track_candidates.get(k)
            else float("-inf"),
            reverse=True,
        )
        for track_key in order:
            assign(track_key)
        return assignments

    def _resolve_assignments_hungarian(
        self, track_candidates: Dict[str, List[Candidate]]
    ) -> Dict[str, Optional[Candidate]]:
        track_keys = list(track_candidates.keys())
        if not track_keys:
            return {}

        rc_index: Dict[Tuple[int, int], int] = {}
        by_track_rc: List[Dict[int, Candidate]] = []
        min_score = 0.0
        max_score = 0.0
        has_any = False
        for key in track_keys:
            mapping: Dict[int, Candidate] = {}
            for cand in track_candidates.get(key, []):
                if not math.isfinite(cand.score):
                    continue
                col = rc_index.setdefault(cand.rc, len(rc_index))
                prev = mapping.get(col)
                if prev is None or cand.score > prev.score:
                    mapping[col] = cand
                if not has_any:
                    min_score = max_score = float(cand.score)
                    has_any = True
                else:
                    min_score = min(min_score, float(cand.score))
                    max_score = max(max_score, float(cand.score))
            by_track_rc.append(mapping)

        if not has_any:
            return {key: None for key in track_keys}

        num_tracks = len(track_keys)
        num_cells = len(rc_index)
        span = max(1.0, max_score - min_score)
        unassigned_cost = -min_score + span * float(num_tracks + 2)
        disallowed_cost = unassigned_cost + span * float(num_tracks + 10) * 100.0

        cost = np.full(
            (num_tracks, num_cells + num_tracks), disallowed_cost, dtype=np.float64
        )
        for row_idx, mapping in enumerate(by_track_rc):
            for col_idx, cand in mapping.items():
                cost[row_idx, col_idx] = -float(cand.score)
            cost[row_idx, num_cells + row_idx] = unassigned_cost

        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost)
            assignments: Dict[str, Optional[Candidate]] = {
                key: None for key in track_keys
            }
            for r_idx, c_idx in zip(row_ind.tolist(), col_ind.tolist()):
                if c_idx < num_cells:
                    cand = by_track_rc[r_idx].get(c_idx)
                    if cand is not None:
                        assignments[track_keys[r_idx]] = cand
            return assignments
        except ImportError:
            # Exact minimum-cost assignment fallback when SciPy is unavailable.
            best_total = float("inf")
            best_choice: List[Optional[Candidate]] = [None] * num_tracks
            used_cols: Set[int] = set()
            current_choice: List[Optional[Candidate]] = [None] * num_tracks

            def _search(row_idx: int, running_cost: float) -> None:
                nonlocal best_total, best_choice
                if running_cost >= best_total:
                    return
                if row_idx == num_tracks:
                    best_total = running_cost
                    best_choice = list(current_choice)
                    return
                for col_idx, cand in sorted(
                    by_track_rc[row_idx].items(),
                    key=lambda item: item[1].score,
                    reverse=True,
                ):
                    if col_idx in used_cols:
                        continue
                    used_cols.add(col_idx)
                    current_choice[row_idx] = cand
                    _search(row_idx + 1, running_cost - float(cand.score))
                    current_choice[row_idx] = None
                    used_cols.remove(col_idx)
                current_choice[row_idx] = None
                _search(row_idx + 1, running_cost + unassigned_cost)

            _search(0, 0.0)
            return {track_keys[idx]: best_choice[idx] for idx in range(num_tracks)}

    def _update_symmetry_midpoints(self) -> None:
        pairs = getattr(self.runtime_config, "symmetry_pairs", ())
        symmetry_penalty = float(
            getattr(self.runtime_config, "symmetry_penalty", 0.0) or 0.0
        )
        if not pairs and symmetry_penalty <= 0.0:
            return
        processed: Set[str] = set()
        for track in self.tracks.values():
            partner_key = track.symmetry_partner
            if not partner_key or track.key in processed:
                continue
            partner = self.tracks.get(partner_key)
            processed.add(track.key)
            processed.add(partner_key)
            if partner is None:
                continue
            if track.misses > 0 or partner.misses > 0:
                continue
            axis = (
                track.last_position[0] - partner.last_position[0],
                track.last_position[1] - partner.last_position[1],
            )
            norm = math.hypot(axis[0], axis[1])
            if norm <= 1e-6:
                continue
            axis_unit = (axis[0] / norm, axis[1] / norm)
            prev_axis = track.symmetry_axis
            if (
                prev_axis
                and (axis_unit[0] * prev_axis[0] + axis_unit[1] * prev_axis[1]) < 0
            ):
                axis_unit = (-axis_unit[0], -axis_unit[1])
            midpoint = (
                (track.last_position[0] + partner.last_position[0]) / 2.0,
                (track.last_position[1] + partner.last_position[1]) / 2.0,
            )
            track.symmetry_axis = axis_unit
            track.symmetry_midpoint = midpoint
            partner.symmetry_axis = axis_unit
            partner.symmetry_midpoint = midpoint

    def _reference_descriptor(
        self,
        feats: torch.Tensor,
        patch_rc: Tuple[int, int],
        grid_hw: Tuple[int, int],
        base_desc: torch.Tensor,
    ) -> torch.Tensor:
        reference_desc = base_desc
        if self.reference_support_radius > 0:
            r_min = max(0, patch_rc[0] - self.reference_support_radius)
            r_max = min(grid_hw[0] - 1, patch_rc[0] + self.reference_support_radius)
            c_min = max(0, patch_rc[1] - self.reference_support_radius)
            c_max = min(grid_hw[1] - 1, patch_rc[1] + self.reference_support_radius)
            region = feats[:, r_min : r_max + 1, c_min : c_max + 1]
            region = region.reshape(region.shape[0], -1)
            region_mean = region.mean(dim=1).detach()
            region_mean = region_mean / (region_mean.norm() + 1e-12)
            mix = (
                self.reference_center_weight * base_desc
                + (1.0 - self.reference_center_weight) * region_mean
            )
            reference_desc = mix / (mix.norm() + 1e-12)
        return reference_desc

    def _build_mask_cache(
        self,
        feats: torch.Tensor,
        mask_lookup: Optional[Dict[str, np.ndarray]],
        grid_hw: Tuple[int, int],
        *,
        allow_fallback: bool,
    ) -> Dict[str, Dict[str, object]]:
        cache: Dict[str, Dict[str, object]] = {}
        kernel_size = max(1, int(self.runtime_config.mask_dilation_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        iterations = max(0, int(self.runtime_config.mask_dilation_iterations))
        mask_lookup = mask_lookup or {}

        for label, mask in mask_lookup.items():
            if mask is None:
                continue
            patch_mask = self._mask_to_patch(mask, grid_hw)
            if not patch_mask.any():
                continue
            if iterations > 0:
                patch_mask = self._dilate_patch_mask(
                    patch_mask, kernel_size, iterations
                )
            descriptor = None
            if self.mask_descriptor_weight > 0.0:
                descriptor = self._compute_mask_descriptor(feats, patch_mask)
            cache[label] = {
                "patch_mask": patch_mask,
                "descriptor": descriptor,
                "similarity_bonus": self._mask_similarity_bonus(0),
            }
            self._last_patch_masks[label] = patch_mask
            self._mask_miss_counts[label] = 0

        if not allow_fallback:
            return cache

        allowed_misses = max(0, int(self.runtime_config.max_mask_fallback_frames))
        for label, stored_mask in list(self._last_patch_masks.items()):
            if label in cache:
                continue
            misses = self._mask_miss_counts.get(label, 0) + 1
            self._mask_miss_counts[label] = misses
            if misses > allowed_misses:
                continue
            fallback_mask = stored_mask
            if iterations > 0:
                fallback_mask = self._dilate_patch_mask(
                    fallback_mask, kernel_size, iterations
                )
            descriptor = None
            if self.mask_descriptor_weight > 0.0:
                descriptor = self._compute_mask_descriptor(feats, fallback_mask)
            cache[label] = {
                "patch_mask": fallback_mask,
                "descriptor": descriptor,
                "similarity_bonus": self._mask_similarity_bonus(misses),
            }
        return cache

    def _mask_to_patch(self, mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
        grid_h, grid_w = grid_hw
        resized = cv2.resize(
            mask.astype(np.uint8), (grid_w, grid_h), interpolation=cv2.INTER_NEAREST
        )
        return resized.astype(bool)

    def _dilate_patch_mask(
        self,
        patch_mask: np.ndarray,
        kernel_size: int,
        iterations: int,
    ) -> np.ndarray:
        if iterations <= 0:
            return patch_mask
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(patch_mask.astype(np.uint8), kernel, iterations=iterations)
        return dilated.astype(bool)

    def _compute_mask_descriptor(
        self,
        feats: torch.Tensor,
        patch_mask: np.ndarray,
    ) -> Optional[torch.Tensor]:
        mask_tensor = torch.from_numpy(patch_mask.astype(np.bool_))
        mask_tensor = mask_tensor.to(feats.device)
        flat_mask = mask_tensor.view(-1)
        if not bool(flat_mask.any().item()):
            return None
        flat_feats = feats.view(feats.shape[0], -1)
        masked_feats = flat_feats[:, flat_mask]
        if masked_feats.numel() == 0:
            return None
        descriptor = masked_feats.mean(dim=1)
        return self._normalize_descriptor(descriptor, clone=False)

    def _apply_mask_descriptor(
        self,
        descriptor: torch.Tensor,
        mask_descriptor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask_descriptor is None or self.mask_descriptor_weight <= 0.0:
            return descriptor
        blended = (
            1.0 - self.mask_descriptor_weight
        ) * descriptor + self.mask_descriptor_weight * mask_descriptor
        return blended / (blended.norm() + 1e-12)

    def _mask_similarity_bonus(self, misses: int) -> float:
        base_bonus = float(max(0.0, self.runtime_config.mask_similarity_bonus))
        if base_bonus <= 0.0:
            return 0.0
        if misses <= 0:
            return base_bonus
        decay = max(0.0, 1.0 - 0.25 * (misses - 1))
        return base_bonus * 0.5 * decay


class DinoKeypointVideoProcessor:
    """Video orchestrator coordinating instances, masks, and serialization."""

    allow_force_thread_terminate = False

    def __init__(
        self,
        video_path: str,
        *,
        result_folder: Optional[Path],
        model_name: str,
        short_side: int = 768,
        device: Optional[str] = None,
        runtime_config: Optional[CutieDinoTrackerConfig] = None,
    ) -> None:
        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        first_frame = self.video_loader.get_first_frame()
        if first_frame is None:
            raise RuntimeError("Video contains no frames.")
        self.video_height, self.video_width = first_frame.shape[:2]
        self.total_frames = self.video_loader.total_frames()

        self.video_result_folder = (
            Path(result_folder)
            if result_folder
            else Path(self.video_path).with_suffix("")
        )
        self.video_result_folder.mkdir(parents=True, exist_ok=True)

        self.config = runtime_config or CutieDinoTrackerConfig()
        self.adapter = AnnotationAdapter(
            image_height=self.video_height,
            image_width=self.video_width,
            persist_json=self.config.persist_labelme_json,
            ambiguous_point_policy=str(
                getattr(self.config, "ambiguous_point_policy", "withhold") or "withhold"
            ),
        )
        self.mask_manager = CutieMaskManager(
            Path(self.video_path),
            adapter=self.adapter,
            config=self.config,
        )
        self.tracker = DinoKeypointTracker(
            model_name=model_name,
            short_side=short_side,
            device=device,
            runtime_config=self.config,
        )
        fps = None
        get_fps = getattr(self.video_loader, "get_fps", None)
        if callable(get_fps):
            try:
                fps = float(get_fps())
            except Exception:
                fps = None
        cfg_smoother_fps = getattr(self.config, "tracking_smoother_fps", None)
        if cfg_smoother_fps is not None:
            fps = float(cfg_smoother_fps)
        if not fps or not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        self._smoother: Optional[KeypointSmoother] = None
        if bool(getattr(self.config, "tracking_smoother_enable", False)):
            self._smoother = KeypointSmoother(
                mode=str(
                    getattr(self.config, "tracking_smoother_mode", "one_euro")
                    or "one_euro"
                ),
                fps=float(fps),
                ema_alpha=float(
                    getattr(self.config, "tracking_smoother_ema_alpha", 0.7)
                ),
                min_score=float(
                    getattr(self.config, "tracking_smoother_min_conf", 0.25)
                ),
                one_euro_min_cutoff=float(
                    getattr(self.config, "tracking_smoother_one_euro_min_cutoff", 1.0)
                ),
                one_euro_beta=float(
                    getattr(self.config, "tracking_smoother_one_euro_beta", 0.0)
                ),
                kalman_process_noise=float(
                    getattr(
                        self.config,
                        "tracking_smoother_kalman_process_noise",
                        getattr(self.config, "tracking_smoother_process_noise", 1e-2),
                    )
                ),
                kalman_measurement_noise=float(
                    getattr(
                        self.config,
                        "tracking_smoother_kalman_measurement_noise",
                        getattr(
                            self.config, "tracking_smoother_measurement_noise", 1e-1
                        ),
                    )
                ),
                kalman_gate_sigma=float(
                    getattr(self.config, "tracking_smoother_gate_sigma", 4.5)
                ),
                max_jump_px=float(
                    getattr(self.config, "tracking_smoother_max_jump_px", 0.0)
                ),
            )
        self.pred_worker = None
        self._stop_requested = False

    def set_pred_worker(self, pred_worker) -> None:
        self.pred_worker = pred_worker

    def request_stop(self) -> None:
        """Request cooperative cancellation without killing the Qt worker thread."""
        self._stop_requested = True
        request_tracker_stop = getattr(self.tracker, "request_stop", None)
        if callable(request_tracker_stop):
            request_tracker_stop()

    def allows_force_terminate(self) -> bool:
        """DINO/Transformers/MPS work is not safe to kill with QThread.terminate()."""
        return False

    def _reset_tracker_state(self, *, preserve_manual_anchors: bool = False) -> None:
        reset_state = getattr(self.tracker, "reset_state", None)
        if not callable(reset_state):
            return
        if preserve_manual_anchors:
            try:
                accepts_manual_anchors = (
                    "preserve_manual_anchors"
                    in inspect.signature(reset_state).parameters
                )
            except (TypeError, ValueError):
                accepts_manual_anchors = False
            if accepts_manual_anchors:
                reset_state(preserve_manual_anchors=True)
                return
        reset_state()

    def _seed_smoother_from_registry(self, registry: InstanceRegistry) -> None:
        if self._smoother is None:
            return
        self._smoother.reset()
        for instance in registry:
            for kp in instance.keypoints.values():
                if kp.visible:
                    self._smoother.smooth(
                        kp.key,
                        (float(kp.x), float(kp.y)),
                        score=1.0,
                        mask_ok=True,
                    )

    def _apply_smoother_to_results(
        self,
        tracker_results: List[Dict[str, object]],
        mask_lookup: Dict[str, np.ndarray],
        registry: InstanceRegistry,
    ) -> None:
        if self._smoother is None or not tracker_results:
            return
        for item in tracker_results:
            key = str(item.get("id", ""))
            if not key:
                continue
            kp_state = registry.get_keypoint(key)
            raw_x = float(item.get("x", kp_state.x if kp_state else 0.0))  # type: ignore[arg-type]
            raw_y = float(item.get("y", kp_state.y if kp_state else 0.0))  # type: ignore[arg-type]
            visible = bool(item.get("visible", True))
            confidence = float(item.get("confidence", item.get("quality", 1.0)))  # type: ignore[arg-type]
            mask_ok = visible
            if mask_ok and kp_state is not None:
                inst_mask = mask_lookup.get(kp_state.instance_label)
                if inst_mask is not None and bool(self.config.mask_enforce_position):
                    px = int(round(raw_x))
                    py = int(round(raw_y))
                    mask_ok = (
                        0 <= py < int(inst_mask.shape[0])
                        and 0 <= px < int(inst_mask.shape[1])
                        and bool(inst_mask[py, px])
                    )
            sx, sy = self._smoother.smooth(
                key,
                (raw_x, raw_y),
                score=(confidence if visible else 0.0),
                mask_ok=mask_ok,
            )
            item["x"] = float(sx)
            item["y"] = float(sy)

    def process_video(
        self,
        *,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        step: int = 1,
        pred_worker: Optional[object] = None,
    ) -> str:
        if pred_worker is not None:
            self.set_pred_worker(pred_worker)
        try:
            return self._process_video_impl(
                start_frame=start_frame,
                end_frame=end_frame,
                step=step,
            )
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("DINO tracking failed")
            if self.config.error_hook:
                self.config.error_hook(exc)
            raise

    def _process_video_impl(
        self,
        *,
        start_frame: Optional[int],
        end_frame: Optional[int],
        step: int,
    ) -> str:
        (
            initial_frame,
            registry,
            is_tracked_resume,
            anchor_seed,
        ) = self._resolve_initial_state_with_resume(start_frame=start_frame)
        start_frame, end_frame, step = normalize_tracking_frame_range(
            initial_frame=initial_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
            total_frames=self.total_frames,
        )

        manual_frames = collect_manual_annotation_frames(
            self.video_result_folder,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        manual_frames.pop(initial_frame, None)
        manual_frames = self._filter_valid_manual_frames(manual_frames)
        seeded_frames = build_seeded_frame_index(initial_frame, manual_frames.keys())
        finished_frame_cache: Dict[int, bool] = {}

        self.mask_manager.reset_state()
        self._reset_tracker_state(preserve_manual_anchors=False)

        # When resuming from an intermediate tracked frame (e.g., frame 59 before
        # predicting frame 60+), seed long-term manual anchor codebooks from the
        # originating manual frame first so appearance anchoring stays grounded.
        if is_tracked_resume and anchor_seed is not None:
            anchor_frame, anchor_registry = anchor_seed
            anchor_array = self.video_loader.load_frame(anchor_frame)
            if anchor_array is not None and anchor_registry.keypoint_payload():
                anchor_masks = self._mask_lookup_from_registry(anchor_registry)
                self.tracker.start(
                    Image.fromarray(anchor_array),
                    anchor_registry,
                    anchor_masks,
                )
                self._reset_tracker_state(preserve_manual_anchors=True)

        initial_frame_array = self.video_loader.load_frame(initial_frame)
        if initial_frame_array is None:
            raise RuntimeError("Unable to load the initial frame for tracking.")
        if self._should_stop():
            logger.info("DINO tracking stopped before tracker initialization.")
            return "Cutie + DINO tracking stopped early."

        initial_mask_lookup = self._mask_lookup_from_registry(registry)
        if self.mask_manager.enabled:
            self.mask_manager.prime(initial_frame, initial_frame_array, registry)
        if self._should_stop():
            logger.info("DINO tracking stopped before DINO feature extraction.")
            return "Cutie + DINO tracking stopped early."
        self.tracker.start(
            Image.fromarray(initial_frame_array),
            registry,
            initial_mask_lookup,
        )
        self._seed_smoother_from_registry(registry)
        if self._should_stop():
            logger.info("DINO tracking stopped after tracker initialization.")
            return "Cutie + DINO tracking stopped early."
        if not is_tracked_resume:
            self.adapter.write_annotation(
                frame_number=initial_frame,
                registry=registry,
                output_dir=self.video_result_folder,
            )

        frame_numbers = list(range(start_frame, end_frame + 1, step))
        frames_to_process = [frame for frame in frame_numbers if frame != initial_frame]
        total_steps = max(1, len(frames_to_process))
        processed = 0
        stopped_early = False

        for frame_number in frame_numbers:
            if frame_number == initial_frame:
                continue
            if self._should_stop():
                stopped_early = True
                break

            manual_path = manual_frames.get(frame_number)
            if manual_path is not None:
                resume_result = self._resume_from_manual_annotation(
                    frame_number,
                    manual_path,
                    prior_registry=registry,
                )
                manual_frames.pop(frame_number, None)
                if resume_result is not None:
                    registry, manual_mask_lookup = resume_result
                    self._seed_smoother_from_registry(registry)
                    processed += 1
                    self._report_progress(processed, total_steps)
                    if self.config.analytics_hook:
                        self.config.analytics_hook(
                            {
                                "frame": frame_number,
                                "keypoints": len(registry.keypoint_payload()),
                                "masks": len(manual_mask_lookup),
                            }
                        )
                    continue
                # Invalid manual seed file: process this frame normally and
                # remove it from seeded boundaries for skip decisions.
                remove_seeded_frame(seeded_frames, frame_number)

            if should_skip_finished_frame_between_adjacent_seeded_frames(
                frame_number=frame_number,
                seeded_frames=seeded_frames,
                video_result_folder=self.video_result_folder,
                finished_frame_cache=finished_frame_cache,
            ):
                processed += 1
                self._report_progress(processed, total_steps)
                continue

            frame = self.video_loader.load_frame(frame_number)
            if frame is None:
                logger.warning("Skipping missing frame %s", frame_number)
                continue

            mask_results = self.mask_manager.update_masks(frame_number, frame, registry)
            if self._should_stop():
                stopped_early = True
                break
            if mask_results:
                self._apply_mask_results(registry, mask_results)
            mask_lookup = self._mask_lookup_from_registry(registry)

            tracker_results = self.tracker.update(
                Image.fromarray(frame),
                mask_lookup,
            )
            if self._should_stop():
                stopped_early = True
                break
            if tracker_results:
                self._apply_smoother_to_results(tracker_results, mask_lookup, registry)
                registry.apply_tracker_results(
                    tracker_results, frame_number=frame_number
                )
            self.adapter.write_annotation(
                frame_number=frame_number,
                registry=registry,
                output_dir=self.video_result_folder,
            )

            processed += 1
            self._report_progress(processed, total_steps)
            if self.config.analytics_hook:
                self.config.analytics_hook(
                    {
                        "frame": frame_number,
                        "keypoints": len(tracker_results),
                        "masks": len(mask_lookup),
                    }
                )

        message = "Cutie + DINO tracking completed."
        if stopped_early:
            message = "Cutie + DINO tracking stopped early."
        logger.info(message)
        return message

    def _resolve_initial_state(
        self,
        *,
        start_frame: Optional[int],
    ) -> Tuple[int, InstanceRegistry]:
        seed_frame, registry, _, _ = self._resolve_initial_state_with_resume(
            start_frame=start_frame
        )
        return seed_frame, registry

    def _resolve_initial_state_with_resume(
        self,
        *,
        start_frame: Optional[int],
    ) -> Tuple[
        int,
        InstanceRegistry,
        bool,
        Optional[Tuple[int, InstanceRegistry]],
    ]:
        initial_frame, registry = self.adapter.load_initial_state(
            self.video_result_folder
        )
        if start_frame is None:
            return int(initial_frame), registry, False, None

        try:
            requested = int(start_frame)
        except Exception:
            return int(initial_frame), registry, False, None
        requested = max(0, min(int(requested), int(self.total_frames) - 1))

        requested_path = (
            self.video_result_folder
            / f"{self.video_result_folder.name}_{requested:09d}.json"
        )
        try:
            requested_registry = self.adapter.read_annotation(
                requested_path,
                prior_registry=(
                    registry if int(requested) != int(initial_frame) else None
                ),
            )
            if (
                requested_registry.keypoint_payload()
                or requested_registry.instances_with_masks()
            ):
                logger.info(
                    "Using requested seed frame %s for keypoint tracking.", requested
                )
                return int(requested), requested_registry, False, None
        except Exception:
            pass

        base_frame = int(initial_frame)
        base_registry = registry

        # Fall back to the nearest manual annotated frame at or before requested.
        available_frames: List[int] = []
        for filename in find_manual_labeled_json_files(str(self.video_result_folder)):
            try:
                available_frames.append(int(get_frame_number_from_json(filename)))
            except Exception:
                continue
        if available_frames:
            lower_or_equal = [frame for frame in available_frames if frame <= requested]
            candidate = max(lower_or_equal) if lower_or_equal else min(available_frames)
            if int(candidate) != int(initial_frame):
                candidate_path = (
                    self.video_result_folder
                    / f"{self.video_result_folder.name}_{int(candidate):09d}.json"
                )
                try:
                    candidate_registry = self.adapter.read_annotation(candidate_path)
                    base_frame = int(candidate)
                    base_registry = candidate_registry
                except Exception:
                    pass

        # If resuming further down the video (requested > base_frame + 1), prefer the
        # latest persisted tracked frame right before `requested` so Cutie memory and
        # optical flow do not teleport across dozens of already-tracked frames.
        tracked_candidate = find_latest_tracked_frame_before(
            self.video_result_folder,
            target_frame=requested,
            min_frame=base_frame,
        )
        if tracked_candidate is not None:
            tracked_frame, tracked_path = tracked_candidate
            try:
                tracked_registry = self.adapter.read_annotation(
                    tracked_path,
                    prior_registry=base_registry,
                )
                if (
                    tracked_registry.keypoint_payload()
                    or tracked_registry.instances_with_masks()
                ):
                    logger.info(
                        "Resuming keypoint tracking from persisted frame %s before requested frame %s (manual anchor %s).",
                        tracked_frame,
                        requested,
                        base_frame,
                    )
                    return (
                        int(tracked_frame),
                        tracked_registry,
                        True,
                        (base_frame, base_registry),
                    )
            except Exception:
                logger.debug(
                    "Failed to read persisted frame %s; falling back to manual seed %s.",
                    tracked_path,
                    base_frame,
                    exc_info=True,
                )

        logger.info(
            "Requested seed frame %s unavailable; using nearest manual frame %s.",
            requested,
            int(base_frame),
        )
        return int(base_frame), base_registry, False, None

    def _manual_annotation_frames(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, Path]:
        return collect_manual_annotation_frames(
            self.video_result_folder,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def _filter_valid_manual_frames(
        self, manual_frames: Dict[int, Path]
    ) -> Dict[int, Path]:
        valid: Dict[int, Path] = {}
        for frame_number, manual_path in manual_frames.items():
            try:
                self.adapter.read_annotation(manual_path)
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid manual seed frame %s (%s): %s",
                    frame_number,
                    manual_path,
                    exc,
                )
                continue
            valid[int(frame_number)] = manual_path
        return valid

    def _resume_from_manual_annotation(
        self,
        frame_number: int,
        manual_path: Path,
        *,
        prior_registry: Optional[InstanceRegistry] = None,
    ) -> Optional[Tuple[InstanceRegistry, Dict[str, np.ndarray]]]:
        previous_masks: Dict[str, MaskResult] = {}
        last_results = getattr(self.mask_manager, "_last_results", None)
        if isinstance(last_results, dict) and last_results:
            for label, result in last_results.items():
                if result.mask_bitmap is None:
                    continue
                polygon_copy: Optional[List[Tuple[float, float]]] = None
                if result.polygon:
                    polygon_copy = [(float(x), float(y)) for x, y in result.polygon]
                previous_masks[label] = MaskResult(
                    instance_label=result.instance_label,
                    mask_bitmap=np.array(result.mask_bitmap, copy=True),
                    polygon=polygon_copy or [],
                )

        try:
            registry = self.adapter.read_annotation(
                manual_path,
                prior_registry=prior_registry,
            )
        except Exception as exc:
            logger.warning(
                "Manual resume skipped for frame %s: failed to read %s (%s)",
                frame_number,
                manual_path,
                exc,
            )
            return None

        if previous_masks:
            for instance in registry:
                has_mask = instance.mask_bitmap is not None and bool(
                    np.any(instance.mask_bitmap)
                )
                # Manual annotations hold polygons without a bitmap until
                # _mask_lookup_from_registry rasterizes them. They take
                # precedence over the previous frame's tracked mask.
                if has_mask or instance.polygon:
                    continue
                fallback = previous_masks.get(instance.label)
                if fallback is None:
                    continue
                polygon = (
                    [tuple(point) for point in fallback.polygon]
                    if fallback.polygon
                    else None
                )
                instance.set_mask(
                    bitmap=np.array(fallback.mask_bitmap, copy=True),
                    polygon=polygon,
                )

        frame_array = self.video_loader.load_frame(frame_number)
        if frame_array is None:
            logger.warning(
                "Manual resume skipped for frame %s: frame unavailable",
                frame_number,
            )
            return None

        for instance in registry:
            instance.last_updated_frame = frame_number

        self.mask_manager.reset_state()
        self._reset_tracker_state(preserve_manual_anchors=True)

        mask_lookup = self._mask_lookup_from_registry(registry)
        if self.mask_manager.enabled:
            self.mask_manager.prime(frame_number, frame_array, registry)

        self.tracker.start(
            Image.fromarray(frame_array),
            registry,
            mask_lookup,
        )

        return registry, mask_lookup

    def _apply_mask_results(
        self,
        registry: InstanceRegistry,
        mask_results: Dict[str, MaskResult],
    ) -> None:
        apply_mask_results_to_registry(registry, mask_results)

    def _mask_lookup_from_registry(
        self, registry: InstanceRegistry
    ) -> Dict[str, np.ndarray]:
        return mask_lookup_from_registry(registry, self.adapter)

    def _report_progress(self, processed: int, total: int) -> None:
        report_processor_progress(
            processed=processed,
            total=total,
            pred_worker=self.pred_worker,
            progress_hook=self.config.progress_hook,
        )

    def _should_stop(self) -> bool:
        return is_processor_stop_requested(
            stop_requested=self._stop_requested,
            pred_worker=self.pred_worker,
        )
