"""Cutie + DinoKPSEG tracker that outputs instance polygons + keypoints."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class _KPSEGInstanceContext:
    gid: int
    instance_key: str
    instance: object
    mask: Optional[np.ndarray]
    max_jump_px: float
    mask_points_xy: Optional[np.ndarray]
    mask_centroid_xy: Optional[Tuple[float, float]]


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

        self.video_result_folder = (
            Path(result_folder)
            if result_folder
            else Path(self.video_path).with_suffix("")
        )
        self.video_result_folder.mkdir(parents=True, exist_ok=True)

        self.config = runtime_config or CutieDinoTrackerConfig()
        # This processor is the "Cutie + DinoKPSEG" pipeline: ensure Cutie is on and
        # keypoints are produced for each tracked polygon.
        self.config.use_cutie_tracking = True
        if (
            str(getattr(self.config, "kpseg_apply_mode", "never") or "never")
            .strip()
            .lower()
            == "never"
        ):
            self.config.kpseg_apply_mode = "always"
        # Cutie + DinoKPSEG contract: keypoints must stay inside instance polygons.
        # Force inside-mask tracking/search and inside-mask kpseg fusion gates.
        self.config.restrict_to_initial_mask = True
        self.config.kpseg_use_mask_gate = True
        self.config.mask_enforce_position = True
        self.config.mask_enforce_reject_outside = True
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
        try:
            self.predictor = DinoKPSEGPredictor(kpseg_weights, device=device)
        except TypeError:
            # Backward compatibility for monkeypatched predictor doubles.
            self.predictor = DinoKPSEGPredictor(kpseg_weights, device=device)
        self.keypoint_names = list(self.predictor.keypoint_names or [])
        tracker_name = (
            tracker_model_name
            if tracker_model_name
            else getattr(self.predictor.meta, "model_name", None)
        )
        tracker_device = str(self.predictor.device) if device is None else device
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
            kalman_measurement_noise=float(self.config.kpseg_kalman_measurement_noise),
        )

        self.pred_worker = None
        self._bbox_padding_px = max(0, int(bbox_padding_px))
        self._instance_display_labels: Dict[str, str] = {}
        self._instance_numeric_id_by_label: Dict[str, int] = {}
        self._instance_label_by_numeric_id: Dict[int, str] = {}
        self._next_instance_numeric_id: int = 0
        self.annotation_parser = DinoKPSEGAnnotationParser(
            image_height=self.video_height,
            image_width=self.video_width,
            adapter=self.adapter,
        )
        self._kpseg_enabled_by_instance: Dict[str, bool] = {}
        self._kpseg_good_streak: Dict[str, int] = {}
        self._kpseg_bad_streak: Dict[str, int] = {}

    def _reset_instance_id_state(self) -> None:
        self._instance_numeric_id_by_label = {}
        self._instance_label_by_numeric_id = {}
        self._next_instance_numeric_id = 0

    def _reserve_instance_numeric_id(
        self,
        *,
        instance_label: str,
        preferred: Optional[int] = None,
    ) -> int:
        label = str(instance_label)
        existing = self._instance_numeric_id_by_label.get(label)
        if existing is not None:
            return int(existing)

        if preferred is not None:
            candidate = int(preferred)
            owner = self._instance_label_by_numeric_id.get(candidate)
            if owner is None or owner == label:
                self._instance_numeric_id_by_label[label] = int(candidate)
                self._instance_label_by_numeric_id[int(candidate)] = label
                self._next_instance_numeric_id = max(
                    int(self._next_instance_numeric_id), int(candidate) + 1
                )
                return int(candidate)

        candidate = int(self._next_instance_numeric_id)
        while candidate in self._instance_label_by_numeric_id:
            candidate += 1
        self._instance_numeric_id_by_label[label] = int(candidate)
        self._instance_label_by_numeric_id[int(candidate)] = label
        self._next_instance_numeric_id = int(candidate) + 1
        return int(candidate)

    def _resolve_instance_numeric_id(self, instance_label: str) -> int:
        preferred = self._normalize_group_id(instance_label)
        return self._reserve_instance_numeric_id(
            instance_label=str(instance_label),
            preferred=(int(preferred) if preferred is not None else None),
        )

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
        manual_seed = self._resolve_initial_seed(start_frame=start_frame)
        start_frame = (
            manual_seed.frame_number
            if start_frame is None
            else max(manual_seed.frame_number, int(start_frame))
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
        self._reset_instance_id_state()

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
            bool(points)
            for points in (manual_seed.keypoints_by_instance or {}).values()
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
                resume = self._resume_from_manual_annotation(frame_number, manual_path)
                manual_frames.pop(frame_number, None)
                if resume is not None:
                    registry = resume.registry
                    self._instance_display_labels.update(resume.display_labels)
                    frame_rgb = self.video_loader.load_frame(frame_number)
                    if self.mask_manager.enabled:
                        self.mask_manager.prime(frame_number, frame_rgb, registry)
                    mask_lookup = self._mask_lookup_from_registry(registry)
                    has_seed_keypoints = any(
                        bool(points)
                        for points in (resume.keypoints_by_instance or {}).values()
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
            self._maybe_apply_kpseg(frame_bgr, registry, frame_number=frame_number)
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
        manual_files = find_manual_labeled_json_files(str(self.video_result_folder))
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

    def _resolve_initial_seed(self, *, start_frame: Optional[int]) -> ManualAnnotation:
        manual_seed = self._load_initial_state(self.video_result_folder)
        if start_frame is None:
            return manual_seed

        try:
            requested = int(start_frame)
        except Exception:
            return manual_seed
        requested = max(0, min(int(requested), int(self.total_frames) - 1))
        requested_path = (
            self.video_result_folder
            / f"{self.video_result_folder.name}_{requested:09d}.json"
        )

        try:
            resolved = self.annotation_parser.read_manual_annotation(
                requested, requested_path
            )
            logger.info(
                "Using requested seed frame %s for DinoKPSEG tracking.", requested
            )
            return resolved
        except Exception:
            pass

        available_frames: List[int] = []
        for filename in find_manual_labeled_json_files(str(self.video_result_folder)):
            try:
                available_frames.append(int(get_frame_number_from_json(filename)))
            except Exception:
                continue
        if available_frames:
            lower_or_equal = [frame for frame in available_frames if frame <= requested]
            candidate = max(lower_or_equal) if lower_or_equal else min(available_frames)
            if int(candidate) != int(manual_seed.frame_number):
                candidate_path = (
                    self.video_result_folder
                    / f"{self.video_result_folder.name}_{int(candidate):09d}.json"
                )
                try:
                    resolved = self.annotation_parser.read_manual_annotation(
                        int(candidate), candidate_path
                    )
                    logger.info(
                        "Requested seed frame %s unavailable; using nearest annotated frame %s.",
                        requested,
                        int(candidate),
                    )
                    return resolved
                except Exception:
                    pass

        logger.info(
            "Requested seed frame %s unavailable; using latest available frame %s.",
            requested,
            int(manual_seed.frame_number),
        )
        return manual_seed

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
                    polygon_copy = [(float(x), float(y)) for x, y in result.polygon]
                previous_masks[label] = MaskResult(
                    instance_label=result.instance_label,
                    mask_bitmap=np.array(result.mask_bitmap, copy=True),
                    polygon=polygon_copy or [],
                )

        try:
            manual = self.annotation_parser.read_manual_annotation(
                frame_number, manual_path
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
            for instance in manual.registry:
                has_mask = instance.mask_bitmap is not None and bool(
                    np.any(instance.mask_bitmap)
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
            gid = self._resolve_instance_numeric_id(str(instance.label))
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

    @staticmethod
    def _keypoint_key(instance_label: str, keypoint_label: str) -> str:
        return f"{instance_label}:{keypoint_label}"

    def _build_kpseg_instance_contexts(
        self,
        registry: InstanceRegistry,
        *,
        use_mask_gate: bool,
    ) -> Tuple[Dict[int, _KPSEGInstanceContext], List[Tuple[int, np.ndarray]]]:
        contexts: Dict[int, _KPSEGInstanceContext] = {}
        instance_masks: List[Tuple[int, np.ndarray]] = []

        for instance in registry:
            gid = self._resolve_instance_numeric_id(str(instance.label))
            mask = instance.mask_bitmap
            if mask is None and instance.polygon is not None:
                mask = self.adapter.mask_bitmap_from_polygon(instance.polygon)
            if mask is None:
                continue

            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                continue

            ys, xs = np.nonzero(mask_bool)
            if xs.size == 0 or ys.size == 0:
                continue

            points_xy = np.column_stack((xs, ys)).astype(np.float32)
            centroid_xy = (float(xs.mean()), float(ys.mean()))
            max_jump_px = float(getattr(self.config, "kpseg_max_jump_px", 0.0))
            if max_jump_px <= 0.0:
                max_jump_px = self._resolve_max_jump_px(mask_bool)

            context = _KPSEGInstanceContext(
                gid=int(gid),
                instance_key=str(instance.label),
                instance=instance,
                mask=(mask_bool if use_mask_gate else None),
                max_jump_px=float(max_jump_px),
                mask_points_xy=points_xy,
                mask_centroid_xy=centroid_xy,
            )
            contexts[int(gid)] = context
            instance_masks.append((int(gid), mask_bool))

        return contexts, instance_masks

    def _project_coord_to_instance_mask(
        self,
        coord: Tuple[float, float],
        context: _KPSEGInstanceContext,
        *,
        fallback_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        mask = context.mask
        if mask is None:
            return (float(coord[0]), float(coord[1]))

        if self._point_in_mask(coord, mask):
            return (float(coord[0]), float(coord[1]))
        if fallback_xy is not None and self._point_in_mask(fallback_xy, mask):
            return (float(fallback_xy[0]), float(fallback_xy[1]))

        points = context.mask_points_xy
        if points is not None and points.size > 0:
            target = np.asarray([float(coord[0]), float(coord[1])], dtype=np.float32)
            deltas = points - target[None, :]
            d2 = np.einsum("ij,ij->i", deltas, deltas)
            best = points[int(np.argmin(d2))]
            return (float(best[0]), float(best[1]))

        if context.mask_centroid_xy is not None:
            return (
                float(context.mask_centroid_xy[0]),
                float(context.mask_centroid_xy[1]),
            )
        return (float(coord[0]), float(coord[1]))

    def _align_predictions_to_schema(
        self,
        *,
        context: _KPSEGInstanceContext,
        coords: List[Tuple[float, float]],
        scores: List[float],
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        aligned_coords: List[Tuple[float, float]] = []
        aligned_scores: List[float] = []

        for idx, label in enumerate(self.keypoint_names):
            key = self._keypoint_key(context.instance_key, str(label))
            prev = context.instance.keypoints.get(key)
            fallback_prev = (float(prev.x), float(prev.y)) if prev is not None else None

            if idx < len(coords):
                coord = (float(coords[idx][0]), float(coords[idx][1]))
                score = float(scores[idx]) if idx < len(scores) else 0.0
            elif prev is not None:
                coord = fallback_prev
                score = float(prev.confidence)
            elif context.mask_centroid_xy is not None:
                coord = (
                    float(context.mask_centroid_xy[0]),
                    float(context.mask_centroid_xy[1]),
                )
                score = 0.0
            else:
                coord = (0.0, 0.0)
                score = 0.0

            coord = self._project_coord_to_instance_mask(
                coord, context, fallback_xy=fallback_prev
            )
            aligned_coords.append(coord)
            aligned_scores.append(score)

        return aligned_coords, aligned_scores

    def _is_kpseg_measurement_reliable(
        self,
        *,
        context: _KPSEGInstanceContext,
        coord: Tuple[float, float],
        score: float,
        prev: Optional[KeypointState],
        min_score: float,
        instance_fallback: bool,
    ) -> bool:
        if instance_fallback:
            return False
        if float(score) < float(min_score):
            return False
        if context.mask is not None and not self._point_in_mask(coord, context.mask):
            return False
        if prev is not None and float(context.max_jump_px) > 0.0:
            dx = float(coord[0]) - float(prev.x)
            dy = float(coord[1]) - float(prev.y)
            if (dx * dx + dy * dy) > float(context.max_jump_px) * float(
                context.max_jump_px
            ):
                return False
        return True

    def _maybe_apply_kpseg(
        self,
        frame_bgr: np.ndarray,
        registry: InstanceRegistry,
        *,
        frame_number: int,
    ) -> None:
        mode = (
            str(getattr(self.config, "kpseg_apply_mode", "never") or "never")
            .strip()
            .lower()
        )
        if mode not in ("never", "auto", "always"):
            mode = "never"
        if mode == "never":
            return

        if not self.keypoint_names:
            pred = self.predictor.predict(frame_bgr, stabilize_lr=False)
            self.keypoint_names = [str(i) for i in range(len(pred.keypoints_xy))]
            self.predictor.reset_state()

        corrections: Dict[str, Tuple[float, float]] = {}
        # Pipeline guarantee: keypoint search/estimation happens inside instance masks.
        use_mask_gate = True
        context_by_gid, instance_masks = self._build_kpseg_instance_contexts(
            registry,
            use_mask_gate=use_mask_gate,
        )
        if not context_by_gid:
            return

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
            smoother_mode = (
                str(getattr(self._kpseg_smoother, "mode", "none") or "none")
                .strip()
                .lower()
            )
        use_temporal_correction = smoother_mode not in ("", "none")
        min_score = float(getattr(self.config, "kpseg_min_score", 0.25))
        blend_alpha = float(getattr(self.config, "kpseg_blend_alpha", 0.7))
        fallback_mode = (
            str(
                getattr(self.config, "kpseg_fallback_mode", "per_keypoint")
                or "per_keypoint"
            )
            .strip()
            .lower()
        )
        fallback_ratio = float(getattr(self.config, "kpseg_fallback_ratio", 0.5))
        fallback_to_track = bool(getattr(self.config, "kpseg_fallback_to_track", True))

        for gid, context in context_by_gid.items():
            pred = preds_by_id.get(int(gid))
            if pred is None:
                continue
            instance = context.instance
            instance_key = context.instance_key
            enabled = (
                True
                if mode == "always"
                else bool(self._kpseg_enabled_by_instance.get(instance_key, False))
            )

            raw_coords = [(float(x), float(y)) for x, y in pred.keypoints_xy]
            raw_scores = [float(s) for s in pred.keypoint_scores]
            coords, scores = self._align_predictions_to_schema(
                context=context,
                coords=raw_coords,
                scores=raw_scores,
            )
            reliability_ratio = self._kpseg_reliability_ratio(
                context=context,
                coords=coords,
                scores=scores,
                min_score=min_score,
            )

            if mode == "auto":
                target = float(getattr(self.config, "kpseg_reliable_ratio", 0.6))
                min_frames = int(getattr(self.config, "kpseg_min_reliable_frames", 5))
                disable_patience = int(
                    getattr(self.config, "kpseg_disable_patience", 5)
                )

                if not enabled:
                    if reliability_ratio >= target:
                        self._kpseg_good_streak[instance_key] = (
                            int(self._kpseg_good_streak.get(instance_key, 0)) + 1
                        )
                    else:
                        self._kpseg_good_streak[instance_key] = 0
                    if int(self._kpseg_good_streak[instance_key]) >= max(1, min_frames):
                        enabled = True
                        self._kpseg_enabled_by_instance[instance_key] = True
                        self._kpseg_bad_streak[instance_key] = 0
                else:
                    if reliability_ratio < target:
                        self._kpseg_bad_streak[instance_key] = (
                            int(self._kpseg_bad_streak.get(instance_key, 0)) + 1
                        )
                    else:
                        self._kpseg_bad_streak[instance_key] = 0
                    if int(self._kpseg_bad_streak[instance_key]) >= max(
                        1, disable_patience
                    ):
                        enabled = False
                        self._kpseg_enabled_by_instance[instance_key] = False
                        self._kpseg_good_streak[instance_key] = 0
                        self._kpseg_bad_streak[instance_key] = 0

            if not enabled:
                continue

            instance_fallback = (
                fallback_mode == "instance"
                and reliability_ratio is not None
                and float(reliability_ratio) < float(fallback_ratio)
            )

            if not use_temporal_correction:
                final_coords, final_scores, measured_ok = self._combine_pred_with_track(
                    context=context,
                    coords=coords,
                    scores=scores,
                    reliability_ratio=reliability_ratio,
                    min_score=min_score,
                    blend_alpha=blend_alpha,
                    fallback_to_track=fallback_to_track,
                    fallback_mode=fallback_mode,
                    fallback_ratio=fallback_ratio,
                )
            else:
                final_coords = []
                final_scores = []
                measured_ok = []
                for idx, (coord, score) in enumerate(zip(coords, scores)):
                    label = (
                        self.keypoint_names[idx]
                        if idx < len(self.keypoint_names)
                        else str(idx)
                    )
                    key = self._keypoint_key(context.instance_key, str(label))
                    prev = instance.keypoints.get(key)
                    prev_xy = (
                        (float(prev.x), float(prev.y)) if prev is not None else None
                    )
                    projected = self._project_coord_to_instance_mask(
                        coord, context, fallback_xy=prev_xy
                    )
                    ok = self._is_kpseg_measurement_reliable(
                        context=context,
                        coord=projected,
                        score=float(score),
                        prev=prev,
                        min_score=min_score,
                        instance_fallback=instance_fallback,
                    )

                    input_coord = projected
                    if ok and prev is not None and blend_alpha < 1.0:
                        x = blend_alpha * float(projected[0]) + (
                            1.0 - blend_alpha
                        ) * float(prev.x)
                        y = blend_alpha * float(projected[1]) + (
                            1.0 - blend_alpha
                        ) * float(prev.y)
                        input_coord = (float(x), float(y))
                    elif (not ok) and prev is not None:
                        input_coord = prev_xy or input_coord

                    smooth_coord = self._kpseg_smoother.smooth(
                        key,
                        input_coord,
                        score=float(score),
                        mask_ok=bool(ok),
                    )
                    smooth_coord = self._project_coord_to_instance_mask(
                        smooth_coord, context, fallback_xy=prev_xy
                    )
                    final_coords.append(smooth_coord)
                    if ok:
                        final_scores.append(float(score))
                    elif prev is not None:
                        final_scores.append(float(prev.confidence))
                    else:
                        final_scores.append(
                            float(max(0.0, min(float(score), min_score)))
                        )
                    measured_ok.append(bool(ok))

            self.predictor.seed_instance_state(
                int(gid),
                keypoints_xy=final_coords,
                keypoint_scores=final_scores,
            )

            for idx, (kpt_label, (x, y), score) in enumerate(
                zip(self.keypoint_names, final_coords, final_scores)
            ):
                key = self._keypoint_key(context.instance_key, str(kpt_label))
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
                    instance_label=str(context.instance_key),
                    label=str(kpt_label),
                    x=float(x),
                    y=float(y),
                    visible=True
                    if context.mask is None
                    else bool(self._point_in_mask((float(x), float(y)), context.mask)),
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
        context: _KPSEGInstanceContext,
        coords: List[Tuple[float, float]],
        scores: List[float],
        min_score: float,
    ) -> float:
        if not coords or not self.keypoint_names:
            return 0.0

        reliable = 0
        total = min(len(coords), len(self.keypoint_names))
        if total <= 0:
            return 0.0
        for idx in range(total):
            label = self.keypoint_names[idx]
            key = self._keypoint_key(context.instance_key, str(label))
            prev = context.instance.keypoints.get(key)
            score = float(scores[idx]) if idx < len(scores) else 0.0
            coord = self._project_coord_to_instance_mask(
                coords[idx],
                context,
                fallback_xy=(float(prev.x), float(prev.y))
                if prev is not None
                else None,
            )
            ok = self._is_kpseg_measurement_reliable(
                context=context,
                coord=coord,
                score=score,
                prev=prev,
                min_score=min_score,
                instance_fallback=False,
            )
            if ok:
                reliable += 1
        return float(reliable) / float(total)

    def _write_annotation(self, frame_number: int, registry: InstanceRegistry) -> None:
        json_path = (
            self.video_result_folder
            / f"{self.video_result_folder.name}_{frame_number:09d}.json"
        )
        shapes: List[Shape] = []
        for instance in registry:
            group_id = int(self._resolve_instance_numeric_id(str(instance.label)))
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
                mask_shape.points = [[float(x), float(y)] for x, y in instance.polygon]
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
        context: _KPSEGInstanceContext,
        coords: List[Tuple[float, float]],
        scores: List[float],
        reliability_ratio: Optional[float] = None,
        min_score: float,
        blend_alpha: float,
        fallback_to_track: bool,
        fallback_mode: str,
        fallback_ratio: float,
    ) -> Tuple[List[Tuple[float, float]], List[float], List[bool]]:
        instance_fallback = (
            fallback_mode == "instance"
            and reliability_ratio is not None
            and float(reliability_ratio) < float(fallback_ratio)
        )

        final_coords: List[Tuple[float, float]] = []
        final_scores: List[float] = []
        measured_ok: List[bool] = []

        for idx, (coord, score) in enumerate(zip(coords, scores)):
            label = (
                self.keypoint_names[idx] if idx < len(self.keypoint_names) else str(idx)
            )
            key = self._keypoint_key(context.instance_key, str(label))
            prev = context.instance.keypoints.get(key)
            prev_xy = (float(prev.x), float(prev.y)) if prev is not None else None
            projected = self._project_coord_to_instance_mask(
                coord, context, fallback_xy=prev_xy
            )
            reliable = self._is_kpseg_measurement_reliable(
                context=context,
                coord=projected,
                score=float(score),
                prev=prev,
                min_score=min_score,
                instance_fallback=instance_fallback,
            )

            if not reliable and fallback_to_track and prev is not None:
                coord = prev_xy or projected
                score = float(prev.confidence)
                prev.misses += 1
            elif reliable and prev is not None and blend_alpha < 1.0:
                x = blend_alpha * float(projected[0]) + (1.0 - blend_alpha) * float(
                    prev.x
                )
                y = blend_alpha * float(projected[1]) + (1.0 - blend_alpha) * float(
                    prev.y
                )
                coord = (float(x), float(y))
                prev.misses = 0
            elif not reliable:
                coord = self._project_coord_to_instance_mask(
                    projected, context, fallback_xy=prev_xy
                )
                score = float(max(0.0, min(float(score), min_score)))
                if prev is not None:
                    prev.misses += 1
            elif prev is not None:
                coord = projected
                prev.misses = 0
            else:
                coord = projected

            final_coords.append((float(coord[0]), float(coord[1])))
            final_scores.append(float(score))
            measured_ok.append(bool(reliable))

        return final_coords, final_scores, measured_ok

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
