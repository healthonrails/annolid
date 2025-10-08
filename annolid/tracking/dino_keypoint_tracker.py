"""Cutie + DINO tracker that outputs unified keypoint and mask annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import torch
from PIL import Image
import math

from annolid.data.videos import CV2Video
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager, MaskResult
from annolid.tracking.domain import InstanceRegistry
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger


@dataclass
class KeypointTrack:
    """Internal state for a single tracked keypoint."""

    key: str
    storage_label: str
    instance_label: str
    display_label: str
    patch_rc: Tuple[int, int]
    descriptor: torch.Tensor
    reference_descriptor: torch.Tensor
    velocity: Tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    struct_refs: Dict[str, float] = field(default_factory=dict)
    last_position: Tuple[float, float] = (0.0, 0.0)
    quality: float = 1.0
    appearance_codebook: Optional[torch.Tensor] = None
    baseline_similarity: float = 1.0
    symmetry_partner: Optional[str] = None
    symmetry_axis: Optional[Tuple[float, float]] = None
    symmetry_midpoint: Optional[Tuple[float, float]] = None
    symmetry_sign: float = 0.0
    support_probes: List["SupportProbe"] = field(default_factory=list)


@dataclass
class SupportProbe:
    offset_rc: Tuple[int, int]
    descriptor: torch.Tensor
    weight: float
    in_mask: bool = False


@dataclass
class Candidate:
    rc: Tuple[int, int]
    xy: Tuple[float, float]
    score: float
    similarity: float
    descriptor: torch.Tensor


@dataclass
class MotionPrior:
    predicted_xy: Tuple[float, float]
    predicted_rc: Tuple[int, int]
    base_xy: Tuple[float, float]
    flow_vec: Optional[Tuple[float, float]]
    flow_speed: float
    velocity_speed: float
    radius: int
    radius_px: float
    confidence: float


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
        cfg = Dinov3Config(
            model_name=model_name,
            short_side=short_side,
            device=device,
        )
        self.extractor = Dinov3FeatureExtractor(cfg)
        self.runtime_config = runtime_config or CutieDinoTrackerConfig()
        self.search_radius = max(1, int(search_radius))
        self.min_similarity = float(min_similarity)
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.reference_weight = float(np.clip(reference_weight, 0.0, 1.0))
        self.reference_support_radius = max(0, int(reference_support_radius))
        self.reference_center_weight = float(
            np.clip(reference_center_weight, 0.0, 1.0))
        self.mask_descriptor_weight = float(
            np.clip(self.runtime_config.mask_descriptor_weight, 0.0, 1.0))
        self.restrict_to_mask = bool(
            self.runtime_config.restrict_to_initial_mask)
        self.mask_enforce_position = bool(
            getattr(self.runtime_config, "mask_enforce_position", True))
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
        self.max_candidates = max(
            1, int(self.runtime_config.max_candidate_tracks))
        self._is_fresh_start = False
        self.reset_state()

    def reset_state(self) -> None:
        """Restore tracker state to a clean slate prior to (re)starting."""
        self.tracks.clear()
        self.prev_gray = None
        self.prev_scale = (1.0, 1.0)
        self._last_patch_masks = {}
        self._mask_miss_counts = {}

        self._is_fresh_start = False

    def start(
        self,
        image: Image.Image,
        registry: InstanceRegistry,
        mask_lookup: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.reset_state()
        self._is_fresh_start = True  # Signal that the next update is the first

        feats = self._extract_features(image)
        new_h, new_w = self.extractor._compute_resized_hw(*image.size)
        scale_x = new_w / image.width
        scale_y = new_h / image.height
        grid_hw = feats.shape[1:]
        self._last_patch_masks = {}
        self._mask_miss_counts = {}
        mask_cache = self._build_mask_cache(
            feats,
            mask_lookup,
            grid_hw,
            allow_fallback=False,
        )

        self.prev_scale = (scale_x, scale_y)
        self.tracks.clear()
        for instance in registry:
            for keypoint in instance.keypoints.values():
                patch_rc = self._pixel_to_patch(
                    keypoint.x, keypoint.y, scale_x, scale_y, grid_hw)
                base_desc = feats[:, patch_rc[0], patch_rc[1]].detach().clone()
                base_desc = base_desc / (base_desc.norm() + 1e-12)
                reference_desc = self._reference_descriptor(
                    feats, patch_rc, grid_hw, base_desc)
                mask_descriptor = None
                patch_mask = None
                cache_entry = mask_cache.get(instance.label)
                if cache_entry:
                    mask_descriptor = cache_entry.get("descriptor")
                    patch_mask = cache_entry.get("patch_mask")
                descriptor = self._apply_mask_descriptor(
                    reference_desc.clone(), mask_descriptor)

                # KeypointTrack always starts with zero velocity.
                track = KeypointTrack(
                    key=keypoint.key,
                    storage_label=keypoint.storage_label,
                    instance_label=instance.label,
                    display_label=keypoint.label,
                    patch_rc=patch_rc,
                    descriptor=descriptor,
                    reference_descriptor=reference_desc.clone(),
                    last_position=(float(keypoint.x), float(keypoint.y)),
                )

                track.appearance_codebook = self._collect_appearance_codebook(
                    feats, patch_rc)
                if track.appearance_codebook is not None:
                    baseline = torch.matmul(
                        track.appearance_codebook, track.descriptor)
                    track.baseline_similarity = float(baseline.max().item())
                track.support_probes = self._sample_support_probes(
                    track.key,
                    feats,
                    patch_rc,
                    grid_hw,
                    patch_mask,
                )
                self.tracks[keypoint.key] = track

        frame_array = np.array(image)
        self.prev_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        self._initialize_structural_refs(registry)
        self._initialize_symmetry_refs(registry)

    def update(
        self,
        image: Image.Image,
        mask_lookup: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, object]]:
        if not self.tracks:
            return []

        feats = self._extract_features(image)
        new_h, new_w = self.extractor._compute_resized_hw(*image.size)
        scale_x = new_w / image.width
        scale_y = new_h / image.height
        grid_h, grid_w = feats.shape[1:]
        mask_cache = self._build_mask_cache(
            feats,
            mask_lookup,
            feats.shape[1:],
            allow_fallback=True,
        )
        active_masks: Dict[str, Optional[np.ndarray]] = {}
        if mask_lookup:
            active_masks = {
                label: mask.astype(bool) if mask is not None else None
                for label, mask in mask_lookup.items()
            }

        frame_array = np.array(image)
        frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        flow = None
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray,
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
        self.prev_gray = frame_gray
        self.prev_scale = (scale_x, scale_y)
        previous_positions = {
            track.key: track.last_position for track in self.tracks.values()
        }

        track_candidates: Dict[str, List[Candidate]] = {}
        mask_descriptors: Dict[str, Optional[torch.Tensor]] = {}
        base_positions: Dict[str, Tuple[float, float]] = {}
        motion_priors: Dict[str, MotionPrior] = {}
        results: List[Dict[str, object]] = []
        for track in self.tracks.values():
            prev_r, prev_c = track.patch_rc
            base_x, base_y = self._patch_to_pixel(
                (prev_r, prev_c), scale_x, scale_y)
            flow_dx = flow_dy = 0.0
            flow_vec_tuple: Optional[Tuple[float, float]] = None
            if flow is not None:
                fy = int(round(base_y))
                fx = int(round(base_x))
                if 0 <= fy < flow.shape[0] and 0 <= fx < flow.shape[1]:
                    flow_vec = flow[fy, fx]
                    flow_dx, flow_dy = float(flow_vec[0]), float(flow_vec[1])
                    flow_vec_tuple = (flow_dx, flow_dy)

            velocity_dx, velocity_dy = track.velocity
            predicted_x = base_x + flow_dx + velocity_dx
            predicted_y = base_y + flow_dy + velocity_dy
            predicted_r, predicted_c = self._pixel_to_patch(
                predicted_x,
                predicted_y,
                scale_x,
                scale_y,
                feats.shape[1:],
            )

            prior = self._prepare_motion_prior(
                track,
                base_xy=(base_x, base_y),
                predicted_xy=(predicted_x, predicted_y),
                predicted_rc=(predicted_r, predicted_c),
                flow_vec=flow_vec_tuple,
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
            mask_descriptor = mask_entry.get(
                "descriptor") if mask_entry else None
            similarity_bonus = mask_entry.get(
                "similarity_bonus", 0.0) if mask_entry else 0.0
            mask_descriptors[track.key] = mask_descriptor
            base_positions[track.key] = (base_x, base_y)
            mask_pixels = active_masks.get(track.instance_label)

            for r in range(r_min, r_max + 1):
                row_vecs = feats[:, r, c_min:c_max + 1]
                sims = torch.matmul(row_vecs.transpose(0, 1), track.descriptor)
                sims_np = sims.detach().cpu().numpy()
                for idx, candidate_sim in enumerate(sims_np):
                    candidate_c = c_min + idx
                    if patch_mask is not None and not patch_mask[r, candidate_c]:
                        if self.restrict_to_mask:
                            continue
                    candidate_score = float(candidate_sim)
                    if patch_mask is not None and patch_mask[r, candidate_c]:
                        candidate_score += similarity_bonus
                    candidate_desc = feats[:, r, candidate_c].detach().clone()
                    candidate_desc = candidate_desc / (
                        candidate_desc.norm() + 1e-12)
                    candidate_xy = self._patch_to_pixel(
                        (r, candidate_c), scale_x, scale_y)
                    candidate_score += self._appearance_score(
                        track, candidate_desc)
                    candidate_score -= self._structural_penalty(
                        track, candidate_xy, previous_positions)
                    candidate_score -= self._symmetry_penalty(
                        track, candidate_xy)
                    candidate_score -= self._baseline_penalty(
                        track, float(candidate_sim))
                    candidate_score -= self._motion_prior_penalty(
                        track,
                        candidate_xy,
                        motion_priors.get(track.key),
                    )
                    candidate_score += self._support_score(
                        track,
                        (r, candidate_c),
                        feats,
                        patch_mask,
                    )
                    candidate_list.append(
                        Candidate(
                            rc=(r, candidate_c),
                            xy=candidate_xy,
                            score=candidate_score,
                            similarity=float(candidate_sim),
                            descriptor=candidate_desc,
                        )
                    )

            candidate_list.sort(key=lambda c: c.score, reverse=True)
            track_candidates[track.key] = candidate_list[: self.max_candidates]

        assignments = self._resolve_assignments(track_candidates)

        for track in self.tracks.values():
            base_x, base_y = base_positions.get(track.key, (0.0, 0.0))
            assignment = assignments.get(track.key)
            mask_descriptor = mask_descriptors.get(track.key)

            quality = float(assignment.similarity) if assignment else -1.0
            if assignment is None or quality < self.min_similarity:
                track.misses += 1
                visible = False
                x, y = base_x, base_y
                if track.misses > self.max_misses:
                    track.velocity = (0.0, 0.0)
            else:
                x, y = assignment.xy
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

                if not candidate_visible:
                    track.misses += 1
                    visible = False
                    x, y = track.last_position
                    quality = 0.0
                else:
                    track.misses = 0
                    visible = True
                    track.patch_rc = self._pixel_to_patch(
                        x,
                        y,
                        scale_x,
                        scale_y,
                        feats.shape[1:],
                    )
                    new_desc = assignment.descriptor.clone()
                    blended = (1.0 - self.momentum) * track.descriptor + \
                        self.momentum * new_desc
                    if self.reference_weight > 0.0:
                        blended = (1.0 - self.reference_weight) * blended + \
                            self.reference_weight * track.reference_descriptor
                    blended = self._apply_mask_descriptor(
                        blended, mask_descriptor)
                    track.descriptor = blended / (blended.norm() + 1e-12)
                    self._update_appearance_codebook(
                        track, assignment.descriptor)
                    if track.appearance_codebook is not None:
                        refreshed = torch.matmul(
                            track.appearance_codebook, track.descriptor)
                        track.baseline_similarity = min(
                            1.0,
                            max(
                                track.baseline_similarity,
                                float(refreshed.max().item()),
                            ),
                        )
                    self._refresh_support_probes(track, feats)
                    self._update_support_probe_mask_flags(track, patch_mask)
                    delta_x = x - base_x
                    delta_y = y - base_y
                    smoothing = float(
                        np.clip(self.runtime_config.velocity_smoothing, 0.0, 1.0))
                    track.velocity = (
                        (1.0 - smoothing) *
                        track.velocity[0] + smoothing * delta_x,
                        (1.0 - smoothing) *
                        track.velocity[1] + smoothing * delta_y,
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
        return results

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        feats = self.extractor.extract(
            image, return_layer="all", normalize=True)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.dim() == 4:  # [L, D, H, W]
            feats = feats[-2:].mean(dim=0)
        return feats

    def _pixel_to_patch(
        self,
        x: float,
        y: float,
        scale_x: float,
        scale_y: float,
        grid_hw: Tuple[int, int],
    ) -> Tuple[int, int]:
        grid_h, grid_w = grid_hw
        resized_x = x * scale_x
        resized_y = y * scale_y
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
        return center_resized_x / scale_x, center_resized_y / scale_y

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
        pairs = getattr(self.runtime_config, "symmetry_pairs", ())
        if not pairs:
            return
        for instance in registry:
            label_to_track: Dict[str, KeypointTrack] = {}
            for kp in instance.keypoints.values():
                track = self.tracks.get(kp.key)
                if track:
                    label_to_track[kp.label] = track
            for left_label, right_label in pairs:
                left_track = label_to_track.get(left_label)
                right_track = label_to_track.get(right_label)
                if not left_track or not right_track:
                    continue
                left_pos = left_track.last_position
                right_pos = right_track.last_position
                axis = (
                    left_pos[0] - right_pos[0],
                    left_pos[1] - right_pos[1],
                )
                norm = math.hypot(axis[0], axis[1])
                if norm <= 1e-6:
                    continue
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

    def _structural_penalty(
        self,
        track: KeypointTrack,
        candidate_xy: Tuple[float, float],
        previous_positions: Dict[str, Tuple[float, float]],
    ) -> float:
        weight = float(
            max(0.0, self.runtime_config.structural_consistency_weight))
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
            desc = feats[:, sr, sc].detach().clone()
            desc = desc / (desc.norm() + 1e-12)
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
        min_radius = max(1.0, float(
            self.runtime_config.motion_search_min_radius))
        max_radius = max(min_radius, float(
            self.runtime_config.motion_search_max_radius))
        tighten = float(
            np.clip(self.runtime_config.motion_search_tighten, 0.1, 2.0))
        base_radius = max(min_radius, float(self.search_radius) * tighten)

        patch_px_x = self.patch_size / max(scale_x, 1e-6)
        patch_px_y = self.patch_size / max(scale_y, 1e-6)
        patch_px = max(1.0, 0.5 * (patch_px_x + patch_px_y))

        flow_dx = flow_vec[0] if flow_vec else 0.0
        flow_dy = flow_vec[1] if flow_vec else 0.0
        flow_speed = math.hypot(flow_dx, flow_dy)
        velocity_speed = math.hypot(track.velocity[0], track.velocity[1])
        flow_gain = max(0.0, float(
            self.runtime_config.motion_search_flow_gain))
        velocity_gain = max(0.0, float(self.runtime_config.motion_search_gain))
        motion_px = (flow_speed * flow_gain) + (velocity_speed * velocity_gain)
        effective_patch_px = max(patch_px, 1e-6)
        radius_f = base_radius + (motion_px / effective_patch_px)

        miss_boost = max(0.0, float(
            self.runtime_config.motion_search_miss_boost))
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
    ) -> float:
        weight = float(
            max(0.0, self.runtime_config.motion_prior_penalty_weight))
        if weight <= 0.0 or prior is None:
            return 0.0
        soft_radius_px = max(
            1.0,
            float(self.runtime_config.motion_prior_soft_radius_px),
        )
        radius_factor = max(1.0, float(
            self.runtime_config.motion_prior_radius_factor))
        soft_radius_px = max(soft_radius_px, prior.radius_px * radius_factor)

        dx = candidate_xy[0] - prior.predicted_xy[0]
        dy = candidate_xy[1] - prior.predicted_xy[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return 0.0

        scaled = dist / soft_radius_px
        penalty = weight * (1.0 - math.exp(-scaled * scaled))

        miss_relief = 1.0 / (
            1.0
            + track.misses
            * max(0.0, float(self.runtime_config.motion_prior_miss_relief))
        )
        combined = 0.5 * (prior.confidence + miss_relief)

        flow_relief_gain = max(
            0.0, float(getattr(self.runtime_config,
                       "motion_prior_flow_relief", 0.0))
        )
        if flow_relief_gain > 0.0:
            flow_speed = getattr(prior, "flow_speed", 0.0)
            velocity_speed = getattr(prior, "velocity_speed", 0.0)
            speed_sum = max(0.0, float(flow_speed + velocity_speed))
            combined *= 1.0 / (1.0 + speed_sum * flow_relief_gain)

        combined = float(np.clip(combined, 0.1, 1.0))
        return penalty * combined

    def _baseline_penalty(self, track: KeypointTrack, similarity: float) -> float:
        weight = float(
            max(0.0, self.runtime_config.baseline_similarity_weight))
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
            region = mask[y_min:y_max + 1, x_min:x_max + 1]
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
                desc = feats[:, rr, cc].detach().clone()
                desc = desc / (desc.norm() + 1e-12)
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
                descriptor = feats[:, sr, sc].detach().clone()
                descriptor = descriptor / (descriptor.norm() + 1e-12)
                distance_sq = (
                    (dr / sigma) ** 2 + (dc / sigma) ** 2
                )
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
                if (
                    patch_mask is not None
                    and mask_only
                    and not patch_mask[sr, sc]
                ):
                    continue
                descriptor = feats[:, sr, sc].detach().clone()
                descriptor = descriptor / (descriptor.norm() + 1e-12)
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
    ) -> None:
        if not track.support_probes:
            return
        grid_h, grid_w = feats.shape[1:]
        for probe in track.support_probes:
            sr = track.patch_rc[0] + probe.offset_rc[0]
            sc = track.patch_rc[1] + probe.offset_rc[1]
            if 0 <= sr < grid_h and 0 <= sc < grid_w:
                descriptor = feats[:, sr, sc].detach().clone()
                descriptor = descriptor / (descriptor.norm() + 1e-12)
                probe.descriptor = descriptor

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
        descriptor = descriptor.detach().clone()
        descriptor = descriptor / (descriptor.norm() + 1e-12)
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

    def _resolve_assignments(
        self, track_candidates: Dict[str, List[Candidate]]
    ) -> Dict[str, Optional[Candidate]]:
        candidate_pool = {
            key: list(candidates)
            for key, candidates in track_candidates.items()
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

    def _update_symmetry_midpoints(self) -> None:
        pairs = getattr(self.runtime_config, "symmetry_pairs", ())
        if not pairs:
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
            if prev_axis and (
                axis_unit[0] * prev_axis[0] + axis_unit[1] * prev_axis[1]
            ) < 0:
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
            r_max = min(grid_hw[0] - 1, patch_rc[0] +
                        self.reference_support_radius)
            c_min = max(0, patch_rc[1] - self.reference_support_radius)
            c_max = min(grid_hw[1] - 1, patch_rc[1] +
                        self.reference_support_radius)
            region = feats[:, r_min:r_max + 1, c_min:c_max + 1]
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
                    patch_mask, kernel_size, iterations)
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

        allowed_misses = max(
            0, int(self.runtime_config.max_mask_fallback_frames))
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
                    fallback_mask, kernel_size, iterations)
            descriptor = None
            if self.mask_descriptor_weight > 0.0:
                descriptor = self._compute_mask_descriptor(
                    feats, fallback_mask)
            cache[label] = {
                "patch_mask": fallback_mask,
                "descriptor": descriptor,
                "similarity_bonus": self._mask_similarity_bonus(misses),
            }
        return cache

    def _mask_to_patch(self, mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
        grid_h, grid_w = grid_hw
        resized = cv2.resize(
            mask.astype(np.uint8), (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
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
        dilated = cv2.dilate(patch_mask.astype(np.uint8),
                             kernel, iterations=iterations)
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
        descriptor = descriptor / (descriptor.norm() + 1e-12)
        return descriptor

    def _apply_mask_descriptor(
        self,
        descriptor: torch.Tensor,
        mask_descriptor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask_descriptor is None or self.mask_descriptor_weight <= 0.0:
            return descriptor
        blended = (
            (1.0 - self.mask_descriptor_weight) * descriptor
            + self.mask_descriptor_weight * mask_descriptor
        )
        return blended / (blended.norm() + 1e-12)

    def _mask_similarity_bonus(self, misses: int) -> float:
        base_bonus = float(
            max(0.0, self.runtime_config.mask_similarity_bonus))
        if base_bonus <= 0.0:
            return 0.0
        if misses <= 0:
            return base_bonus
        decay = max(0.0, 1.0 - 0.25 * (misses - 1))
        return base_bonus * 0.5 * decay


class DinoKeypointVideoProcessor:
    """Video orchestrator coordinating instances, masks, and serialization."""

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

        self.video_result_folder = Path(result_folder) if result_folder else Path(
            self.video_path).with_suffix("")
        self.video_result_folder.mkdir(parents=True, exist_ok=True)

        self.config = runtime_config or CutieDinoTrackerConfig()
        self.adapter = AnnotationAdapter(
            image_height=self.video_height,
            image_width=self.video_width,
            persist_json=self.config.persist_labelme_json,
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
        self.pred_worker = None

    def set_pred_worker(self, pred_worker) -> None:
        self.pred_worker = pred_worker

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
        initial_frame, registry = self.adapter.load_initial_state(
            self.video_result_folder)
        start_frame = initial_frame if start_frame is None else max(
            initial_frame, start_frame)
        if end_frame is None or end_frame < 0:
            end_frame = self.total_frames - 1
        end_frame = min(end_frame, self.total_frames - 1)
        if start_frame >= end_frame:
            end_frame = min(self.total_frames - 1, start_frame + 1)
        step = max(1, abs(step))

        manual_frames = self._manual_annotation_frames(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        manual_frames.pop(initial_frame, None)

        self.mask_manager.reset_state()
        self.tracker.reset_state()

        initial_frame_array = self.video_loader.load_frame(initial_frame)
        if initial_frame_array is None:
            raise RuntimeError(
                "Unable to load the initial frame for tracking.")

        initial_mask_lookup = self._mask_lookup_from_registry(registry)
        if self.mask_manager.enabled:
            self.mask_manager.prime(
                initial_frame, initial_frame_array, registry)
        self.tracker.start(
            Image.fromarray(initial_frame_array),
            registry,
            initial_mask_lookup,
        )
        self.adapter.write_annotation(
            frame_number=initial_frame,
            registry=registry,
            output_dir=self.video_result_folder,
        )

        frame_numbers = list(range(start_frame, end_frame + 1, step))
        frames_to_process = [
            frame for frame in frame_numbers if frame != initial_frame
        ]
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
                )
                manual_frames.pop(frame_number, None)
                if resume_result is not None:
                    registry, manual_mask_lookup = resume_result
                    processed += 1
                    self._report_progress(processed, total_steps)
                    if self.config.analytics_hook:
                        self.config.analytics_hook(
                            {
                                "frame": frame_number,
                                "keypoints": len(
                                    registry.keypoint_payload()),
                                "masks": len(manual_mask_lookup),
                            }
                        )
                    continue

            frame = self.video_loader.load_frame(frame_number)
            if frame is None:
                logger.warning("Skipping missing frame %s", frame_number)
                continue

            mask_results = self.mask_manager.update_masks(
                frame_number, frame, registry)
            if mask_results:
                self._apply_mask_results(registry, mask_results)
            mask_lookup = self._mask_lookup_from_registry(registry)

            tracker_results = self.tracker.update(
                Image.fromarray(frame),
                mask_lookup,
            )
            if tracker_results:
                registry.apply_tracker_results(
                    tracker_results, frame_number=frame_number)
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

    def _manual_annotation_frames(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, Path]:
        manual_files = find_manual_labeled_json_files(
            str(self.video_result_folder))
        mapping: Dict[int, Path] = {}
        for filename in manual_files:
            try:
                frame_idx = get_frame_number_from_json(filename)
            except (ValueError, IndexError):  # pragma: no cover - defensive
                logger.warning(
                    "Skipping manual annotation with unexpected name: %s",
                    filename,
                )
                continue
            if frame_idx < start_frame or frame_idx > end_frame:
                continue
            mapping[frame_idx] = self.video_result_folder / filename
        return mapping

    def _resume_from_manual_annotation(
        self,
        frame_number: int,
        manual_path: Path,
    ) -> Optional[Tuple[InstanceRegistry, Dict[str, np.ndarray]]]:

        previous_masks: Dict[str, MaskResult] = {}
        last_results = getattr(self.mask_manager, "_last_results", None)
        if isinstance(last_results, dict) and last_results:
            for label, result in last_results.items():
                if result.mask_bitmap is None:
                    continue
                polygon_copy: Optional[List[Tuple[float, float]]] = None
                if result.polygon:
                    polygon_copy = [
                        (float(x), float(y)) for x, y in result.polygon
                    ]
                previous_masks[label] = MaskResult(
                    instance_label=result.instance_label,
                    mask_bitmap=np.array(result.mask_bitmap, copy=True),
                    polygon=polygon_copy or [],
                )

        try:
            registry = self.adapter.read_annotation(manual_path)
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
                has_mask = (
                    instance.mask_bitmap is not None
                    and bool(np.any(instance.mask_bitmap))
                )
                if has_mask:
                    continue
                fallback = previous_masks.get(instance.label)
                if fallback is None:
                    continue
                polygon = [
                    tuple(point) for point in fallback.polygon] if fallback.polygon else None
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
        self.tracker.reset_state()

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
        for instance in registry:
            result = mask_results.get(instance.label)
            if not result:
                continue
            instance.set_mask(
                bitmap=result.mask_bitmap,
                polygon=result.polygon,
            )

    def _mask_lookup_from_registry(
            self, registry: InstanceRegistry) -> Dict[str, np.ndarray]:
        lookup: Dict[str, np.ndarray] = {}
        for instance in registry:
            mask = instance.mask_bitmap
            if mask is None and instance.polygon is not None:
                mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
            if mask is not None:
                lookup[instance.label] = mask.astype(bool)
        return lookup

    def _report_progress(self, processed: int, total: int) -> None:
        if total <= 0:
            return
        progress = int(min(100, max(0, (processed / total) * 100)))
        if self.pred_worker is not None:
            self.pred_worker.report_progress(progress)
        if self.config.progress_hook:
            self.config.progress_hook(processed, total)

    def _should_stop(self) -> bool:
        return bool(self.pred_worker and self.pred_worker.is_stopped())
