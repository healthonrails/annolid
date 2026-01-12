"""Cutie + DinoKPSEG tracker that outputs instance polygons + keypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import math

from annolid.annotation.keypoints import save_labels
from annolid.data.videos import CV2Video
from annolid.gui.shape import Shape
from annolid.segmentation.dino_kpseg import DinoKPSEGPredictor
from annolid.segmentation.dino_kpseg.inference_utils import (
    build_instance_crops,
    mask_bbox,
    predict_on_instance_crops,
)
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager, MaskResult
from annolid.tracking.dino_kpseg_annotations import (
    DinoKPSEGAnnotationParser,
    ManualAnnotation,
)
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker
from annolid.tracking.kpseg_smoothing import KeypointSmoother
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger
from PIL import Image


class DinoKPSEGVideoProcessor:
    """Video orchestrator: Cutie polygons + DinoKPSEG keypoints per instance."""

    def __init__(
        self,
        video_path: str,
        *,
        result_folder: Optional[Path],
        kpseg_weights: str | Path,
        tracker_model_name: Optional[str] = None,
        device: Optional[str] = None,
        runtime_config: Optional[CutieDinoTrackerConfig] = None,
        bbox_padding_px: int = 8,
    ) -> None:
        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        first_frame = self.video_loader.get_first_frame()
        if first_frame is None:
            raise RuntimeError("Video contains no frames.")
        self.video_height, self.video_width = first_frame.shape[:2]
        self.total_frames = self.video_loader.total_frames()

        self.video_result_folder = Path(result_folder) if result_folder else Path(
            self.video_path
        ).with_suffix("")
        self.video_result_folder.mkdir(parents=True, exist_ok=True)

        self.config = runtime_config or CutieDinoTrackerConfig()
        # This processor is the "Cutie + DinoKPSEG" pipeline: ensure Cutie is on and
        # keypoints are produced for each tracked polygon.
        self.config.use_cutie_tracking = True
        if str(getattr(self.config, "kpseg_apply_mode", "never") or "never").strip().lower() == "never":
            self.config.kpseg_apply_mode = "always"
        self.config.normalize()
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
        self.predictor = DinoKPSEGPredictor(
            kpseg_weights, device=device)
        self.keypoint_names = list(self.predictor.keypoint_names or [])
        tracker_name = (
            tracker_model_name
            if tracker_model_name
            else getattr(self.predictor.meta, "model_name", None)
        )
        tracker_device = str(
            self.predictor.device) if device is None else device
        self._tracker_name = str(tracker_name)
        self._tracker_device = str(tracker_device)
        self._keypoint_tracker: Optional[DinoKeypointTracker] = None
        self._keypoint_tracker_started = False

        fps = None
        try:
            fps = self.video_loader.get_fps()
        except Exception:
            fps = None
        if self.config.kpseg_smoothing_fps is not None:
            fps = float(self.config.kpseg_smoothing_fps)
        if not fps or float(fps) <= 0:
            fps = 30.0
        self._kpseg_smoother = KeypointSmoother(
            mode=str(self.config.kpseg_smoothing or "none"),
            fps=float(fps),
            ema_alpha=float(self.config.kpseg_smoothing_alpha),
            min_score=float(self.config.kpseg_smoothing_min_score),
            one_euro_min_cutoff=float(self.config.kpseg_one_euro_min_cutoff),
            one_euro_beta=float(self.config.kpseg_one_euro_beta),
            one_euro_d_cutoff=float(self.config.kpseg_one_euro_d_cutoff),
            kalman_process_noise=float(self.config.kpseg_kalman_process_noise),
            kalman_measurement_noise=float(
                self.config.kpseg_kalman_measurement_noise),
        )

        self.pred_worker = None
        self._bbox_padding_px = max(0, int(bbox_padding_px))
        self._instance_display_labels: Dict[str, str] = {}
        self.annotation_parser = DinoKPSEGAnnotationParser(
            image_height=self.video_height,
            image_width=self.video_width,
            adapter=self.adapter,
        )
        self._kpseg_enabled_by_instance: Dict[str, bool] = {}
        self._kpseg_good_streak: Dict[str, int] = {}
        self._kpseg_bad_streak: Dict[str, int] = {}

    def _ensure_keypoint_tracker(self) -> DinoKeypointTracker:
        if self._keypoint_tracker is None:
            self._keypoint_tracker = DinoKeypointTracker(
                model_name=str(self._tracker_name),
                short_side=int(self.predictor.meta.short_side),
                device=str(self._tracker_device),
                runtime_config=self.config,
            )
        return self._keypoint_tracker

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
            logger.exception("DinoKPSEG tracking failed")
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
        manual_seed = self._load_initial_state(self.video_result_folder)
        start_frame = manual_seed.frame_number if start_frame is None else max(
            manual_seed.frame_number, int(start_frame)
        )
        if end_frame is None or int(end_frame) < 0:
            end_frame = self.total_frames - 1
        end_frame = min(int(end_frame), self.total_frames - 1)
        if start_frame >= end_frame:
            end_frame = min(self.total_frames - 1, start_frame + 1)
        step = max(1, abs(int(step)))

        manual_frames = self._manual_annotation_frames(
            start_frame=start_frame,
            end_frame=end_frame,
        )
        manual_frames.pop(manual_seed.frame_number, None)

        self.mask_manager.reset_state()
        self.predictor.reset_state()
        if self._keypoint_tracker is not None:
            self._keypoint_tracker.reset_state()
        self._keypoint_tracker_started = False
        self._kpseg_smoother.reset()

        seed_frame_rgb = self.video_loader.load_frame(manual_seed.frame_number)
        self._instance_display_labels.update(manual_seed.display_labels)
        if self.mask_manager.enabled:
            self.mask_manager.prime(
                manual_seed.frame_number, seed_frame_rgb, manual_seed.registry
            )
        mask_lookup = self._mask_lookup_from_registry(manual_seed.registry)
        has_seed_keypoints = any(
            bool(points) for points in (manual_seed.keypoints_by_instance or {}).values()
        )
        if has_seed_keypoints:
            tracker = self._ensure_keypoint_tracker()
            tracker.start(
                Image.fromarray(seed_frame_rgb),
                manual_seed.registry,
                mask_lookup,
            )
            self._keypoint_tracker_started = True
        self._seed_predictor_from_manual(
            seed_frame_rgb,
            manual_seed.registry,
            manual_seed.keypoints_by_instance,
        )

        frame_numbers = list(range(start_frame, end_frame + 1, step))
        frames_to_process = [
            frame for frame in frame_numbers if frame != manual_seed.frame_number
        ]
        total_steps = max(1, len(frames_to_process))
        processed = 0
        stopped_early = False

        registry = manual_seed.registry
        for frame_number in frame_numbers:
            if frame_number == manual_seed.frame_number:
                continue
            if self._should_stop():
                stopped_early = True
                break

            manual_path = manual_frames.get(frame_number)
            if manual_path is not None:
                resume = self._resume_from_manual_annotation(
                    frame_number, manual_path
                )
                manual_frames.pop(frame_number, None)
                if resume is not None:
                    registry = resume.registry
                    self._instance_display_labels.update(resume.display_labels)
                    frame_rgb = self.video_loader.load_frame(frame_number)
                    if self.mask_manager.enabled:
                        self.mask_manager.prime(
                            frame_number, frame_rgb, registry)
                    mask_lookup = self._mask_lookup_from_registry(registry)
                    has_seed_keypoints = any(
                        bool(points) for points in (resume.keypoints_by_instance or {}).values()
                    )
                    self._keypoint_tracker_started = False
                    if has_seed_keypoints:
                        tracker = self._ensure_keypoint_tracker()
                        tracker.reset_state()
                        tracker.start(
                            Image.fromarray(frame_rgb),
                            registry,
                            mask_lookup,
                        )
                        self._keypoint_tracker_started = True
                    self._seed_predictor_from_manual(
                        frame_rgb, registry, resume.keypoints_by_instance
                    )
                processed += 1
                self._report_progress(processed, total_steps)
                continue

            frame_rgb = self.video_loader.load_frame(frame_number)
            mask_results = self.mask_manager.update_masks(
                frame_number, frame_rgb, registry
            )
            if mask_results:
                self._apply_mask_results(registry, mask_results)

            mask_lookup = self._mask_lookup_from_registry(registry)
            if self._keypoint_tracker_started and self._keypoint_tracker is not None:
                tracker_results = self._keypoint_tracker.update(
                    Image.fromarray(frame_rgb),
                    mask_lookup,
                )
                if tracker_results:
                    registry.apply_tracker_results(
                        tracker_results, frame_number=frame_number
                    )

            frame_bgr = frame_rgb[:, :, ::-1]
            self._maybe_apply_kpseg(
                frame_bgr, registry, frame_number=frame_number)
            self._write_annotation(frame_number, registry)

            processed += 1
            self._report_progress(processed, total_steps)

        message = "Cutie + DinoKPSEG tracking completed."
        if stopped_early:
            message = "Cutie + DinoKPSEG tracking stopped early."
        logger.info(message)
        return message

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

    def _manual_annotation_frames(
        self,
        *,
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, Path]:
        manual_files = find_manual_labeled_json_files(
            str(self.video_result_folder)
        )
        mapping: Dict[int, Path] = {}
        for filename in manual_files:
            path = self.video_result_folder / filename
            try:
                frame = get_frame_number_from_json(filename)
            except Exception:
                continue
            if frame < int(start_frame) or frame > int(end_frame):
                continue
            mapping[int(frame)] = path
        return mapping

    def _load_initial_state(self, annotation_dir: Path) -> ManualAnnotation:
        json_files = find_manual_labeled_json_files(str(annotation_dir))
        if not json_files:
            raise RuntimeError(
                "No labeled JSON files found. Provide an initial polygon annotation for the first frame."
            )
        candidates: List[Tuple[float, int, Path]] = []
        for name in json_files:
            path = annotation_dir / name
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            frame_idx = get_frame_number_from_json(name)
            candidates.append((mtime, int(frame_idx), path))
        if not candidates:
            raise RuntimeError(
                "No labeled JSON files found. Provide an initial polygon annotation for the first frame."
            )
        _, frame_number, latest_json = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        return self.annotation_parser.read_manual_annotation(frame_number, latest_json)

    def _resume_from_manual_annotation(
        self,
        frame_number: int,
        manual_path: Path,
    ) -> Optional[ManualAnnotation]:
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
            manual = self.annotation_parser.read_manual_annotation(
                frame_number, manual_path)
        except Exception as exc:
            logger.warning(
                "Manual resume skipped for frame %s: failed to read %s (%s)",
                frame_number,
                manual_path,
                exc,
            )
            return None

        if previous_masks:
            for instance in manual.registry:
                has_mask = (
                    instance.mask_bitmap is not None
                    and bool(np.any(instance.mask_bitmap))
                )
                if has_mask:
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

        self.mask_manager.reset_state()
        self.predictor.reset_state()
        if self._keypoint_tracker is not None:
            self._keypoint_tracker.reset_state()
        self._keypoint_tracker_started = False
        self._kpseg_smoother.reset()
        return manual

    @staticmethod
    def _normalize_group_id(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):  # bool is also int, so check first
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                return int(raw)
        return None

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
        self,
        registry: InstanceRegistry,
    ) -> Dict[str, np.ndarray]:
        lookup: Dict[str, np.ndarray] = {}
        for instance in registry:
            mask = instance.mask_bitmap
            if mask is None and instance.polygon is not None:
                mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
            if mask is not None:
                lookup[instance.label] = mask.astype(bool)
        return lookup

    def _seed_predictor_from_manual(
        self,
        frame_rgb: np.ndarray,
        registry: InstanceRegistry,
        keypoints_by_instance: Dict[str, Dict[str, Tuple[float, float]]],
    ) -> None:
        if not self.keypoint_names:
            return
        frame_bgr = frame_rgb[:, :, ::-1]
        for instance in registry:
            gid = self._normalize_group_id(instance.label)
            if gid is None:
                continue
            manual_points = keypoints_by_instance.get(instance.label, {})
            if not manual_points:
                continue

            bbox = self._bbox_from_instance_mask(instance)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_mask = self._crop_mask_for_instance(instance, x1, y1, x2, y2)
            pred = self.predictor.predict(
                crop_bgr,
                mask=crop_mask,
                stabilize_lr=False,
                instance_id=int(gid),
            )
            coords = [
                (float(x) + float(x1), float(y) + float(y1))
                for x, y in pred.keypoints_xy
            ]
            scores = [float(s) for s in pred.keypoint_scores]

            for idx, name in enumerate(self.keypoint_names):
                manual_xy = manual_points.get(name)
                if manual_xy is None:
                    continue
                if idx >= len(coords):
                    continue
                coords[idx] = (float(manual_xy[0]), float(manual_xy[1]))
                scores[idx] = 1.0

            self.predictor.seed_instance_state(
                int(gid),
                keypoints_xy=coords,
                keypoint_scores=scores,
            )

    def _bbox_from_instance_mask(
        self,
        instance,
    ) -> Optional[Tuple[int, int, int, int]]:
        mask = instance.mask_bitmap
        if mask is None and instance.polygon is not None:
            mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
        if mask is None:
            return None
        return mask_bbox(
            mask.astype(bool),
            pad_px=int(self._bbox_padding_px),
            image_hw=(int(self.video_height), int(self.video_width)),
        )

    def _crop_mask_for_instance(
        self,
        instance,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[np.ndarray]:
        mask = instance.mask_bitmap
        if mask is None and instance.polygon is not None:
            mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
        if mask is None:
            return None
        return mask[y1:y2, x1:x2]

    def _predict_keypoints(
        self,
        frame_bgr: np.ndarray,
        registry: InstanceRegistry,
        *,
        frame_number: int,
    ) -> None:
        raise NotImplementedError  # replaced by _maybe_apply_kpseg

    def _maybe_apply_kpseg(
        self,
        frame_bgr: np.ndarray,
        registry: InstanceRegistry,
        *,
        frame_number: int,
    ) -> None:
        mode = str(getattr(self.config, "kpseg_apply_mode", "never")
                   or "never").strip().lower()
        if mode not in ("never", "auto", "always"):
            mode = "never"
        if mode == "never":
            return

        if not self.keypoint_names:
            pred = self.predictor.predict(frame_bgr, stabilize_lr=False)
            self.keypoint_names = [str(i)
                                   for i in range(len(pred.keypoints_xy))]
            self.predictor.reset_state()

        corrections: Dict[str, Tuple[float, float]] = {}
        use_mask_gate = bool(getattr(self.config, "kpseg_use_mask_gate", True))
        instance_masks: List[Tuple[int, np.ndarray]] = []
        instance_lookup: Dict[int, object] = {}
        instance_key_lookup: Dict[int, str] = {}

        for instance in registry:
            gid = self._normalize_group_id(instance.label)
            if gid is None:
                continue
            mask = instance.mask_bitmap
            if mask is None and instance.polygon is not None:
                mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
            if mask is None:
                continue
            instance_lookup[int(gid)] = instance
            instance_key_lookup[int(gid)] = str(instance.label)
            instance_masks.append((int(gid), mask))

        crops = build_instance_crops(
            frame_bgr,
            instance_masks,
            pad_px=int(self._bbox_padding_px),
            use_mask_gate=use_mask_gate,
        )
        preds_by_id = {
            int(idx): pred
            for idx, pred in predict_on_instance_crops(
                self.predictor,
                crops,
                stabilize_lr=True,
            )
        }

        smoother_mode = "none"
        if self._kpseg_smoother is not None:
            smoother_mode = str(getattr(self._kpseg_smoother, "mode", "none") or "none").strip().lower()
        use_temporal_correction = smoother_mode not in ("", "none")

        for gid, instance in instance_lookup.items():
            pred = preds_by_id.get(int(gid))
            if pred is None:
                continue
            instance_key = instance_key_lookup.get(int(gid), str(gid))
            enabled = True if mode == "always" else bool(
                self._kpseg_enabled_by_instance.get(instance_key, False)
            )

            coords = [(float(x), float(y)) for x, y in pred.keypoints_xy]
            reliability_ratio = self._kpseg_reliability_ratio(
                instance=instance,
                coords=coords,
                scores=pred.keypoint_scores,
            )

            if mode == "auto":
                target = float(
                    getattr(self.config, "kpseg_reliable_ratio", 0.6))
                min_frames = int(
                    getattr(self.config, "kpseg_min_reliable_frames", 5))
                disable_patience = int(
                    getattr(self.config, "kpseg_disable_patience", 5))

                if not enabled:
                    if reliability_ratio >= target:
                        self._kpseg_good_streak[instance_key] = int(
                            self._kpseg_good_streak.get(instance_key, 0)
                        ) + 1
                    else:
                        self._kpseg_good_streak[instance_key] = 0
                    if int(self._kpseg_good_streak[instance_key]) >= max(1, min_frames):
                        enabled = True
                        self._kpseg_enabled_by_instance[instance_key] = True
                        self._kpseg_bad_streak[instance_key] = 0
                else:
                    if reliability_ratio < target:
                        self._kpseg_bad_streak[instance_key] = int(
                            self._kpseg_bad_streak.get(instance_key, 0)
                        ) + 1
                    else:
                        self._kpseg_bad_streak[instance_key] = 0
                    if int(self._kpseg_bad_streak[instance_key]) >= max(1, disable_patience):
                        enabled = False
                        self._kpseg_enabled_by_instance[instance_key] = False
                        self._kpseg_good_streak[instance_key] = 0
                        self._kpseg_bad_streak[instance_key] = 0

            if not enabled:
                continue

            measured_ok: Optional[List[bool]] = None
            if not use_temporal_correction:
                final_coords, final_scores = self._combine_pred_with_track(
                    instance=instance,
                    coords=coords,
                    scores=pred.keypoint_scores,
                    reliability_ratio=reliability_ratio,
                )
            else:
                measured_ok = []
                mask = instance.mask_bitmap
                if mask is None and instance.polygon is not None:
                    mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
                if not use_mask_gate:
                    mask = None

                min_score = float(getattr(self.config, "kpseg_min_score", 0.25))
                blend_alpha = float(getattr(self.config, "kpseg_blend_alpha", 0.7))
                max_jump = float(getattr(self.config, "kpseg_max_jump_px", 0.0))
                if max_jump <= 0.0:
                    max_jump = self._resolve_max_jump_px(mask)

                fallback_mode = str(getattr(self.config, "kpseg_fallback_mode", "per_keypoint") or "per_keypoint").strip().lower()
                fallback_ratio = float(getattr(self.config, "kpseg_fallback_ratio", 0.5))
                instance_fallback = (
                    fallback_mode == "instance"
                    and reliability_ratio is not None
                    and float(reliability_ratio) < float(fallback_ratio)
                )

                final_coords = []
                final_scores = []
                for idx, (coord, score) in enumerate(zip(coords, pred.keypoint_scores)):
                    label = self.keypoint_names[idx] if idx < len(self.keypoint_names) else str(idx)
                    key = f"{instance.label}:{label}"
                    prev = instance.keypoints.get(key)

                    ok = (not instance_fallback) and float(score) >= min_score
                    if ok and mask is not None:
                        ok = self._point_in_mask(coord, mask)
                    if ok and prev is not None and max_jump > 0.0:
                        dx = float(coord[0]) - float(prev.x)
                        dy = float(coord[1]) - float(prev.y)
                        if (dx * dx + dy * dy) > max_jump * max_jump:
                            ok = False

                    input_coord = coord
                    if ok and prev is not None and blend_alpha < 1.0:
                        x = blend_alpha * float(coord[0]) + (1.0 - blend_alpha) * float(prev.x)
                        y = blend_alpha * float(coord[1]) + (1.0 - blend_alpha) * float(prev.y)
                        input_coord = (float(x), float(y))
                    elif (not ok) and prev is not None:
                        input_coord = (float(prev.x), float(prev.y))

                    smooth_coord = self._kpseg_smoother.smooth(
                        key,
                        input_coord,
                        score=float(score),
                        mask_ok=bool(ok),
                    )
                    final_coords.append(smooth_coord)
                    final_scores.append(float(score))
                    measured_ok.append(bool(ok))

            self.predictor.seed_instance_state(
                int(gid),
                keypoints_xy=final_coords,
                keypoint_scores=final_scores,
            )

            for idx, (kpt_label, (x, y), score) in enumerate(
                zip(self.keypoint_names, final_coords, final_scores)
            ):
                key = f"{instance.label}:{kpt_label}"
                prev = instance.keypoints.get(key)
                if prev is not None:
                    vx = float(x) - float(prev.x)
                    vy = float(y) - float(prev.y)
                    misses = int(prev.misses)
                else:
                    vx, vy = 0.0, 0.0
                    misses = 0
                if measured_ok is not None:
                    if idx < len(measured_ok) and not bool(measured_ok[idx]):
                        misses += 1
                    else:
                        misses = 0
                state = KeypointState(
                    key=key,
                    instance_label=str(instance.label),
                    label=str(kpt_label),
                    x=float(x),
                    y=float(y),
                    visible=True,
                    confidence=float(score),
                    velocity_x=float(vx),
                    velocity_y=float(vy),
                    misses=misses,
                )
                registry.register_keypoint(state)
                corrections[key] = (float(x), float(y))
            instance.last_updated_frame = int(frame_number)

        if (
            corrections
            and bool(getattr(self.config, "kpseg_update_tracker_state", True))
            and mode in ("always", "auto")
        ):
            if self._keypoint_tracker_started and self._keypoint_tracker is not None:
                self._keypoint_tracker.apply_external_corrections(corrections)

    def _kpseg_reliability_ratio(
        self,
        *,
        instance,
        coords: List[Tuple[float, float]],
        scores: List[float],
    ) -> float:
        if not coords or not self.keypoint_names:
            return 0.0
        mask = instance.mask_bitmap
        if mask is None and instance.polygon is not None:
            mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
        if not bool(getattr(self.config, "kpseg_use_mask_gate", True)):
            mask = None
        max_jump = float(getattr(self.config, "kpseg_max_jump_px", 0.0))
        if max_jump <= 0.0:
            max_jump = self._resolve_max_jump_px(mask)
        min_score = float(getattr(self.config, "kpseg_min_score", 0.25))

        reliable = 0
        total = min(len(coords), len(self.keypoint_names))
        if total <= 0:
            return 0.0
        for idx in range(total):
            label = self.keypoint_names[idx]
            key = f"{instance.label}:{label}"
            prev = instance.keypoints.get(key)
            score = float(scores[idx]) if idx < len(scores) else 0.0
            coord = coords[idx]

            ok = score >= min_score
            if ok and mask is not None:
                ok = self._point_in_mask(coord, mask)
            if ok and prev is not None and max_jump > 0:
                dx = float(coord[0]) - float(prev.x)
                dy = float(coord[1]) - float(prev.y)
                ok = (dx * dx + dy * dy) <= max_jump * max_jump
            if ok:
                reliable += 1
        return float(reliable) / float(total)

    def _write_annotation(self, frame_number: int, registry: InstanceRegistry) -> None:
        json_path = self.video_result_folder / \
            f"{self.video_result_folder.name}_{frame_number:09d}.json"
        shapes: List[Shape] = []
        for instance in registry:
            gid = self._normalize_group_id(instance.label)
            group_id = int(gid) if gid is not None else None
            display = self._instance_display_labels.get(
                instance.label, str(instance.label)
            )

            if instance.polygon:
                mask_shape = Shape(
                    label=display,
                    shape_type="polygon",
                    flags={"instance_id": group_id},
                    group_id=group_id,
                    description="Cutie",
                    visible=True,
                )
                mask_shape.points = [
                    [float(x), float(y)] for x, y in instance.polygon
                ]
                shapes.append(mask_shape)

            for keypoint in instance.keypoints.values():
                point_shape = Shape(
                    label=str(keypoint.label),
                    shape_type="point",
                    flags={
                        "score": float(keypoint.confidence),
                        "instance_id": group_id,
                    },
                    group_id=group_id,
                    description="dinokpseg",
                    visible=bool(keypoint.visible),
                )
                point_shape.points = [[float(keypoint.x), float(keypoint.y)]]
                shapes.append(point_shape)

        save_labels(
            filename=str(json_path),
            imagePath="",
            label_list=shapes,
            height=int(self.video_height),
            width=int(self.video_width),
            save_image_to_json=False,
            persist_json=self.config.persist_labelme_json,
        )

    def _combine_pred_with_track(
        self,
        *,
        instance,
        coords: List[Tuple[float, float]],
        scores: List[float],
        reliability_ratio: Optional[float] = None,
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        min_score = float(self.config.kpseg_min_score)
        blend_alpha = float(self.config.kpseg_blend_alpha)
        use_mask_gate = bool(self.config.kpseg_use_mask_gate)
        fallback_to_track = bool(self.config.kpseg_fallback_to_track)
        max_jump = float(self.config.kpseg_max_jump_px)
        fallback_mode = str(
            getattr(self.config, "kpseg_fallback_mode",
                    "per_keypoint") or "per_keypoint"
        ).strip().lower()
        fallback_ratio = float(
            getattr(self.config, "kpseg_fallback_ratio", 0.5))
        instance_fallback = (
            fallback_mode == "instance"
            and reliability_ratio is not None
            and float(reliability_ratio) < float(fallback_ratio)
        )

        mask = instance.mask_bitmap
        if mask is None and instance.polygon is not None:
            mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
        if max_jump <= 0.0:
            max_jump = self._resolve_max_jump_px(mask)

        final_coords: List[Tuple[float, float]] = []
        final_scores: List[float] = []

        for idx, (coord, score) in enumerate(zip(coords, scores)):
            label = (
                self.keypoint_names[idx]
                if idx < len(self.keypoint_names)
                else str(idx)
            )
            key = f"{instance.label}:{label}"
            prev = instance.keypoints.get(key)

            reliable = (not instance_fallback) and float(score) >= min_score
            if reliable and use_mask_gate and mask is not None:
                reliable = self._point_in_mask(coord, mask)

            if reliable and prev is not None and max_jump > 0:
                dx = float(coord[0]) - float(prev.x)
                dy = float(coord[1]) - float(prev.y)
                if (dx * dx + dy * dy) > max_jump * max_jump:
                    reliable = False

            if not reliable and fallback_to_track and prev is not None:
                coord = (float(prev.x), float(prev.y))
                score = float(prev.confidence)
                prev.misses += 1
            elif prev is not None and blend_alpha < 1.0:
                x = blend_alpha * float(coord[0]) + \
                    (1.0 - blend_alpha) * float(prev.x)
                y = blend_alpha * float(coord[1]) + \
                    (1.0 - blend_alpha) * float(prev.y)
                coord = (float(x), float(y))
                prev.misses = 0
            elif prev is not None:
                prev.misses = 0

            final_coords.append((float(coord[0]), float(coord[1])))
            final_scores.append(float(score))

        return final_coords, final_scores

    @staticmethod
    def _point_in_mask(
        coord: Tuple[float, float],
        mask: np.ndarray,
    ) -> bool:
        if mask is None:
            return True
        x, y = int(round(coord[0])), int(round(coord[1]))
        if x < 0 or y < 0:
            return False
        if y >= int(mask.shape[0]) or x >= int(mask.shape[1]):
            return False
        try:
            return bool(mask[y, x])
        except Exception:
            return False

    @staticmethod
    def _resolve_max_jump_px(mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0
        if not np.any(mask):
            return 0.0
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return 0.0
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        diag = math.sqrt(float(w * w + h * h))
        return max(6.0, 0.25 * diag)
