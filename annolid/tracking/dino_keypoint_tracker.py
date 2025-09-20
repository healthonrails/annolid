"""Cutie + DINO tracker that outputs unified keypoint and mask annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from annolid.data.videos import CV2Video
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager, MaskResult
from annolid.tracking.domain import InstanceRegistry
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
        self.tracks: Dict[str, KeypointTrack] = {}
        self.patch_size = self.extractor.patch_size
        self.max_misses = 8
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_scale: Tuple[float, float] = (1.0, 1.0)

    def start(
        self,
        image: Image.Image,
        registry: InstanceRegistry,
        mask_lookup: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        feats = self._extract_features(image)
        new_h, new_w = self.extractor._compute_resized_hw(*image.size)
        scale_x = new_w / image.width
        scale_y = new_h / image.height
        grid_hw = feats.shape[1:]
        mask_cache = self._build_mask_cache(feats, mask_lookup, grid_hw)

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
                cache_entry = mask_cache.get(instance.label)
                if cache_entry:
                    mask_descriptor = cache_entry.get("descriptor")
                descriptor = self._apply_mask_descriptor(
                    reference_desc.clone(), mask_descriptor)
                self.tracks[keypoint.key] = KeypointTrack(
                    key=keypoint.key,
                    storage_label=keypoint.storage_label,
                    instance_label=instance.label,
                    display_label=keypoint.label,
                    patch_rc=patch_rc,
                    descriptor=descriptor,
                    reference_descriptor=reference_desc.clone(),
                )

        frame_array = np.array(image)
        self.prev_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)

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
            feats, mask_lookup, feats.shape[1:])

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

        results: List[Dict[str, object]] = []
        for track in self.tracks.values():
            prev_r, prev_c = track.patch_rc
            base_x, base_y = self._patch_to_pixel(
                (prev_r, prev_c), scale_x, scale_y)
            flow_dx = flow_dy = 0.0
            if flow is not None:
                fy = int(round(base_y))
                fx = int(round(base_x))
                if 0 <= fy < flow.shape[0] and 0 <= fx < flow.shape[1]:
                    flow_vec = flow[fy, fx]
                    flow_dx, flow_dy = float(flow_vec[0]), float(flow_vec[1])

            predicted_x = base_x + flow_dx + track.velocity[0]
            predicted_y = base_y + flow_dy + track.velocity[1]
            predicted_r, predicted_c = self._pixel_to_patch(
                predicted_x,
                predicted_y,
                scale_x,
                scale_y,
                feats.shape[1:],
            )

            radius = self.search_radius + track.misses
            r_min = max(0, predicted_r - radius)
            r_max = min(grid_h - 1, predicted_r + radius)
            c_min = max(0, predicted_c - radius)
            c_max = min(grid_w - 1, predicted_c + radius)
            best_sim = -1.0
            best_rc = (prev_r, prev_c)

            mask_entry = mask_cache.get(track.instance_label)
            patch_mask = mask_entry.get("patch_mask") if mask_entry else None
            mask_descriptor = mask_entry.get(
                "descriptor") if mask_entry else None

            for r in range(r_min, r_max + 1):
                row_vecs = feats[:, r, c_min:c_max + 1]
                sims = torch.matmul(row_vecs.transpose(0, 1), track.descriptor)
                sims_np = sims.detach().cpu().numpy()
                for idx, candidate_sim in enumerate(sims_np):
                    candidate_c = c_min + idx
                    if patch_mask is not None and not patch_mask[r, candidate_c]:
                        if self.restrict_to_mask:
                            continue
                    if candidate_sim > best_sim:
                        best_sim = float(candidate_sim)
                        best_rc = (r, candidate_c)

            if best_sim < self.min_similarity:
                track.misses += 1
                visible = False
                x, y = base_x, base_y
                if track.misses > self.max_misses:
                    track.velocity = (0.0, 0.0)
            else:
                track.misses = 0
                track.patch_rc = best_rc
                new_desc = feats[:, best_rc[0], best_rc[1]].detach()
                new_desc = new_desc / (new_desc.norm() + 1e-12)
                blended = (1.0 - self.momentum) * track.descriptor + \
                    self.momentum * new_desc
                if self.reference_weight > 0.0:
                    blended = (1.0 - self.reference_weight) * blended + \
                        self.reference_weight * track.reference_descriptor
                blended = self._apply_mask_descriptor(blended, mask_descriptor)
                track.descriptor = blended / (blended.norm() + 1e-12)
                x, y = self._patch_to_pixel(best_rc, scale_x, scale_y)
                delta_x = x - base_x
                delta_y = y - base_y
                track.velocity = (
                    0.6 * track.velocity[0] + 0.4 * delta_x,
                    0.6 * track.velocity[1] + 0.4 * delta_y,
                )
                visible = True

            results.append(
                {
                    "id": track.key,
                    "label": track.storage_label,
                    "x": float(x),
                    "y": float(y),
                    "visible": visible,
                    "instance_label": track.instance_label,
                }
            )
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
    ) -> Dict[str, Dict[str, object]]:
        cache: Dict[str, Dict[str, object]] = {}
        if not mask_lookup:
            return cache
        for label, mask in mask_lookup.items():
            if mask is None:
                continue
            patch_mask = self._mask_to_patch(mask, grid_hw)
            if not patch_mask.any():
                continue
            descriptor = None
            if self.mask_descriptor_weight > 0.0:
                descriptor = self._compute_mask_descriptor(feats, patch_mask)
            cache[label] = {
                "patch_mask": patch_mask,
                "descriptor": descriptor,
            }
        return cache

    def _mask_to_patch(self, mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
        grid_h, grid_w = grid_hw
        resized = cv2.resize(
            mask.astype(np.uint8), (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)

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
        total_steps = max(1, len(frame_numbers) - 1)
        processed = 0
        stopped_early = False

        for frame_number in frame_numbers:
            if frame_number == initial_frame:
                continue
            if self._should_stop():
                stopped_early = True
                break

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
