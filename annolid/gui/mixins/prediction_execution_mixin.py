from __future__ import annotations

import copy
import functools
import json
import os
import re
from pathlib import Path

import cv2
from qtpy import QtCore, QtWidgets

from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.gui.workers import FlexibleWorker
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
    should_start_predictions_from_frame0,
)
from annolid.utils.logger import logger

PATCH_SIMILARITY_DEFAULT_MODEL = PATCH_SIMILARITY_MODELS[2].identifier


class PredictionExecutionMixin:
    """Prediction worker setup and completion handling."""

    @staticmethod
    def _resolve_cutie_start_frame_from_seed_state(
        current_start_frame: int,
        max_existing_frame: int,
        manual_seed_max: int,
    ) -> int:
        """Resolve CUTIE restart frame using existing outputs + latest manual seed.

        Priority:
        1) Continue after latest existing output frame when available.
        2) Otherwise continue after latest manual seed; keep frame-0 bootstrap when
           frame 0 is the only seed.
        3) Fall back to the caller-proposed current start frame.
        """
        start = max(0, int(current_start_frame))
        if int(max_existing_frame) >= 0:
            return max(start, int(max_existing_frame) + 1)
        if int(manual_seed_max) > 0:
            return max(start, int(manual_seed_max) + 1)
        if int(manual_seed_max) == 0:
            return 0
        return start

    def _cutie_seed_paths(self, frame_index: int) -> tuple[Path, Path]:
        results_dir = Path(self.video_results_folder)
        stem = results_dir.name
        json_path = results_dir / f"{stem}_{int(frame_index):09d}.json"
        return (json_path, json_path.with_suffix(".png"))

    def _has_cutie_seed_frame(self, frame_index: int) -> bool:
        if not self.video_results_folder:
            return False
        try:
            idx = max(0, int(frame_index))
        except Exception:
            return False
        json_path, png_path = self._cutie_seed_paths(idx)
        return json_path.exists() and png_path.exists()

    def _is_cutie_tracking_model(self, model_name: str | None = None) -> bool:
        active = str(
            model_name
            if model_name is not None
            else getattr(self, "_active_prediction_model_name", "")
        ).lower()
        return ("cutie" in active) and ("cotracker" not in active)

    def _queue_cutie_resume_from_frame(self, stalled_frame: int, message: str) -> None:
        if self._prediction_stop_requested:
            return
        if not self.video_results_folder or not self.video_loader:
            return
        if not self._is_cutie_tracking_model():
            return

        try:
            fallback = max(0, int(stalled_frame))
        except Exception:
            fallback = max(0, int(self.frame_number or 0))

        latest_available = fallback
        try:
            latest_available = max(
                fallback,
                int(self._max_predicted_frame_index(Path(self.video_results_folder))),
            )
        except Exception:
            latest_available = fallback

        question = self.tr(
            "Tracking stalled near frame {stalled}.\n\n"
            "{details}\n\n"
            "Resume by saving frame {seed} as a new seed and continue from there?"
        ).format(
            stalled=int(stalled_frame),
            details=str(message or "").strip(),
            seed=int(latest_available),
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Resume Tracking"),
            question,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        self._pending_prediction_resume_frame = int(latest_available)
        logger.info(
            "Queued CUTIE resume from frame %s after stall at frame %s.",
            int(latest_available),
            int(stalled_frame),
        )

    def _materialize_prediction_seed(self, frame_index: int) -> bool:
        if not self.video_results_folder or not self.video_loader:
            return False

        try:
            frame_idx = max(0, int(frame_index))
        except Exception:
            return False

        results_dir = Path(self.video_results_folder)
        stem = results_dir.name
        json_path = results_dir / f"{stem}_{frame_idx:09d}.json"
        png_path = json_path.with_suffix(".png")

        should_write_json = not json_path.exists()
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
                shapes = payload.get("shapes") or []
                should_write_json = len(shapes) == 0
            except Exception:
                should_write_json = True

        if should_write_json:
            try:
                store = AnnotationStore.for_frame_path(json_path)
                record = store.get_frame(frame_idx)
            except Exception:
                record = None
            if not record:
                logger.warning(
                    "Cannot materialize seed for frame %s: no annotation record found.",
                    int(frame_idx),
                )
                return False
            payload = {
                "version": record.get("version"),
                "flags": record.get("flags", {}),
                "shapes": record.get("shapes", []),
                "imagePath": png_path.name,
                "imageHeight": record.get("imageHeight"),
                "imageWidth": record.get("imageWidth"),
            }
            if record.get("caption") is not None:
                payload["caption"] = record["caption"]
            if record.get("imageData") is not None:
                payload["imageData"] = record["imageData"]
            for key, value in (record.get("otherData") or {}).items():
                payload[key] = value
            try:
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with json_path.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, separators=(",", ":"))
            except Exception as exc:
                logger.warning("Failed to write seed JSON %s: %s", json_path, exc)
                return False

        if not png_path.exists():
            try:
                frame_rgb = self.video_loader.load_frame(int(frame_idx))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(png_path), frame_bgr):
                    logger.warning("Failed to write seed image %s", png_path)
                    return False
            except Exception as exc:
                logger.warning("Failed to materialize seed image %s: %s", png_path, exc)
                return False

        try:
            self._refresh_manual_seed_slider_marks(results_dir)
        except Exception:
            logger.debug("Failed to refresh manual seed marks.", exc_info=True)
        return True

    def _max_predicted_frame_index(
        self, folder: Path, *, include_store: bool = True
    ) -> int:
        """Return the maximum predicted frame index found for a results folder.

        Args:
            folder: Results folder containing per-frame JSON files.
            include_store: When True, also consider frames in AnnotationStore.
                Use False when we need strictly on-disk JSON availability
                (for example after users manually deleted prediction files).
        """
        folder = Path(folder)
        prefixed_pattern = re.compile(r"_(\d{9,})\.json$")
        bare_pattern = re.compile(r"^(\d{9,})\.json$")

        max_frame = -1
        try:
            for name in os.listdir(folder):
                if not name.endswith(".json"):
                    continue
                match = None
                if folder.name in name:
                    match = prefixed_pattern.search(name)
                if match is None:
                    match = bare_pattern.match(name)
                if match is None:
                    continue
                try:
                    idx = int(float(match.group(1)))
                except Exception:
                    continue
                if idx > max_frame:
                    max_frame = idx
        except Exception:
            pass

        if include_store:
            try:
                store = AnnotationStore.for_frame_path(
                    folder / f"{folder.name}_000000000.json"
                )
                if store.store_path.exists():
                    for idx in store.iter_frames():
                        if int(idx) > max_frame:
                            max_frame = int(idx)
            except Exception:
                pass

        return int(max_frame)

    def _dino_kpseg_gui_inference_config(self) -> dict:
        settings = getattr(self, "settings", None)
        if settings is None:
            return {
                "tta_hflip": False,
                "tta_merge": "mean",
                "min_keypoint_score": 0.0,
                "stabilize_lr": True,
            }
        try:
            tta_hflip = bool(settings.value("dino_kpseg/tta_hflip", False, type=bool))
            tta_merge = (
                str(settings.value("dino_kpseg/tta_merge", "mean", type=str) or "mean")
                .strip()
                .lower()
            )
            if tta_merge not in {"mean", "max"}:
                tta_merge = "mean"
            min_score = float(
                settings.value("dino_kpseg/min_keypoint_score", 0.0, type=float)
            )
        except Exception:
            tta_hflip = False
            tta_merge = "mean"
            min_score = 0.0
        return {
            "tta_hflip": bool(tta_hflip),
            "tta_merge": str(tta_merge),
            "min_keypoint_score": max(0.0, float(min_score)),
            "stabilize_lr": True,
        }

    def predict_from_next_frame(self, to_frame=60):
        model_config, model_identifier, model_weight = self._resolve_model_identity()
        _ = model_config
        model_name = model_identifier or model_weight
        text_prompt = self._current_text_prompt()
        self._active_prediction_model_name = str(model_name or "")
        self._skip_tracking_csv_overwrite_for_keypoint_round = bool(
            self._is_dino_keypoint_model(model_name, model_weight)
            or self._is_dino_kpseg_model(model_name, model_weight)
        )
        if self.pred_worker and self.stop_prediction_flag:
            self.stop_prediction()
            return
        elif len(self.canvas.shapes) <= 0 and not (
            self._is_yolo_model(model_name, model_weight)
            or self._is_dino_kpseg_tracker_model(model_name, model_weight)
            or self._is_dino_kpseg_model(model_name, model_weight)
            or (
                self.sam3_manager.is_sam3_model(model_name, model_weight)
                and text_prompt
            )
        ):
            QtWidgets.QMessageBox.about(
                self, "No Shapes or Labeled Frames", "Please label this frame"
            )
            return

        if self.video_file:
            self._prediction_stop_requested = False
            self._prediction_auto_continue_to_end = False
            self._prediction_target_end_frame = None
            self._prediction_chunk_to_frame = int(to_frame)
            self._prediction_run_start_frame = None

            if self._is_dino_kpseg_tracker_model(model_name, model_weight):
                from annolid.tracking.dino_kpseg_tracker import DinoKPSEGVideoProcessor

                resolved = self._resolve_dino_kpseg_weight(model_weight)
                if resolved is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Cutie + DINO Keypoint Segmentation"),
                        self.tr(
                            "No DinoKPSEG checkpoint found. Train a model first or select a valid checkpoint."
                        ),
                    )
                    return
                fresh_tracker_config = copy.deepcopy(self.tracker_runtime_config)
                try:
                    self.video_processor = DinoKPSEGVideoProcessor(
                        video_path=self.video_file,
                        result_folder=self.video_results_folder,
                        kpseg_weights=resolved,
                        device=None,
                        runtime_config=fresh_tracker_config,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to initialize Cutie + DINO keypoint segmentation: %s",
                        exc,
                        exc_info=True,
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Cutie + DINO Keypoint Segmentation"),
                        self.tr(
                            "Failed to initialize tracker:\n%(error)s\n\n"
                            "If using RADIO features, install optional dependency:\n"
                            "pip install open-clip-torch"
                        )
                        % {"error": str(exc)},
                    )
                    return
            elif self._is_dino_keypoint_model(model_name, model_weight):
                from annolid.tracking.dino_keypoint_tracker import (
                    DinoKeypointVideoProcessor,
                )

                dino_model = (
                    self.patch_similarity_model or PATCH_SIMILARITY_DEFAULT_MODEL
                )
                fresh_tracker_config = copy.deepcopy(self.tracker_runtime_config)
                try:
                    self.video_processor = DinoKeypointVideoProcessor(
                        video_path=self.video_file,
                        result_folder=self.video_results_folder,
                        model_name=dino_model,
                        short_side=768,
                        device=None,
                        runtime_config=fresh_tracker_config,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to initialize DINO keypoint tracker '%s': %s",
                        dino_model,
                        exc,
                        exc_info=True,
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("DINO Keypoint Tracker"),
                        self.tr(
                            "Failed to initialize model '%(model)s'.\n\n%(error)s\n\n"
                            "If you selected a RADIO model, install its optional dependency:\n"
                            "pip install open-clip-torch"
                        )
                        % {"model": dino_model, "error": str(exc)},
                    )
                    return
            elif self._is_efficienttam_model(model_name, model_weight):
                from annolid.segmentation.SAM.sam_v2 import process_video_efficienttam

                model_key = (
                    Path(model_weight).stem if model_weight else "efficienttam_s"
                )
                logger.info(
                    "Using EfficientTAM model '%s' for video '%s'",
                    model_key,
                    self.video_file,
                )
                self.video_processor = functools.partial(
                    process_video_efficienttam,
                    video_path=self.video_file,
                    model_key=model_key,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                )
            elif self.sam2_manager.is_sam2_model(model_name, model_weight):
                processor = self.sam2_manager.build_video_processor(
                    model_name=model_name,
                    model_weight=model_weight,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                )
                if processor is None:
                    return
                self.video_processor = processor
            elif self.sam3_manager.is_sam3_model(model_name, model_weight):
                processor = self.sam3_manager.build_video_processor(
                    model_name=model_name,
                    model_weight=model_weight,
                    text_prompt=text_prompt,
                )
                if processor is None:
                    return
                self.video_processor = processor
            elif self._is_yolo_model(model_name, model_weight):
                from annolid.segmentation.yolos import InferenceProcessor

                weight_lower = (model_weight or "").lower()
                is_prompt_free_yoloe = "yoloe" in weight_lower and (
                    "-pf." in weight_lower
                    or weight_lower.endswith("-pf")
                    or "-pf_" in weight_lower
                )

                visual_prompts = {}
                class_names = None
                prompt_class_names = None
                yoloe_text_prompt = True
                if not is_prompt_free_yoloe:
                    visual_prompts = self.extract_visual_prompts_from_canvas()
                    if visual_prompts:
                        logger.info(
                            "Extracted visual prompts for YOLOE: %s", visual_prompts
                        )

                    if visual_prompts and "yoloe" in weight_lower:
                        prompt_class_names = (
                            list(self.class_mapping.keys())
                            if hasattr(self, "class_mapping")
                            else None
                        )
                        yoloe_text_prompt = False
                    else:
                        text_prompt = self.aiRectangle._aiRectanglePrompt.text()
                        class_names = [
                            p.strip() for p in text_prompt.split(",") if p.strip()
                        ]
                        if class_names:
                            logger.info(
                                "Extracted class names from text prompt: %s",
                                class_names,
                            )
                else:
                    visual_prompts = {}
                    class_names = None
                    prompt_class_names = None
                    yoloe_text_prompt = True
                pose_keypoint_names = None
                pose_schema_path = None
                if getattr(self, "_pose_schema", None) is not None and getattr(
                    self._pose_schema, "keypoints", None
                ):
                    pose_keypoint_names = list(self._pose_schema.keypoints)
                if getattr(self, "_pose_schema_path", None):
                    pose_schema_path = self._pose_schema_path
                self.video_processor = InferenceProcessor(
                    model_name=model_weight,
                    model_type="yolo",
                    class_names=class_names,
                    keypoint_names=pose_keypoint_names,
                    pose_schema_path=pose_schema_path,
                    yoloe_text_prompt=bool(yoloe_text_prompt),
                    prompt_class_names=prompt_class_names,
                )
            elif self._is_dino_kpseg_model(model_name, model_weight):
                from annolid.segmentation.yolos import InferenceProcessor

                pose_keypoint_names = None
                pose_schema_path = None
                if getattr(self, "_pose_schema", None) is not None and getattr(
                    self._pose_schema, "keypoints", None
                ):
                    pose_keypoint_names = list(self._pose_schema.keypoints)
                if getattr(self, "_pose_schema_path", None):
                    pose_schema_path = self._pose_schema_path

                resolved = self._resolve_dino_kpseg_weight(model_weight)
                if not resolved:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("DINO Keypoint Segmentation"),
                        self.tr(
                            "No DinoKPSEG checkpoint found.\n\n"
                            "Train a DinoKPSEG model first (Train Models → DINO KPSEG), "
                            "or ensure the best checkpoint exists under your runs folder."
                        ),
                    )
                    return

                try:
                    resolved_path = str(Path(resolved).expanduser().resolve())
                    dino_infer_cfg = self._dino_kpseg_gui_inference_config()
                    cached = getattr(self, "_dinokpseg_inference_processor", None)
                    if (
                        cached is not None
                        and getattr(cached, "model_type", "").lower() == "dinokpseg"
                        and str(getattr(cached, "model_name", "")) == resolved_path
                    ):
                        cached.keypoint_names = pose_keypoint_names or getattr(
                            cached, "keypoint_names", None
                        )
                        try:
                            cached.set_dino_kpseg_inference_config(dino_infer_cfg)
                        except Exception:
                            pass
                        self.video_processor = cached
                    else:
                        self.video_processor = InferenceProcessor(
                            model_name=resolved_path,
                            model_type="dinokpseg",
                            keypoint_names=pose_keypoint_names,
                            pose_schema_path=pose_schema_path,
                            dino_kpseg_inference_config=dino_infer_cfg,
                        )
                        self._dinokpseg_inference_processor = self.video_processor
                except Exception as exc:
                    logger.error(
                        "Failed to load DINO keypoint segmentation model '%s': %s",
                        model_weight,
                        exc,
                        exc_info=True,
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("DINO Keypoint Segmentation"),
                        self.tr(f"Failed to load model:\n{exc}"),
                    )
                    return
            else:
                from annolid.motion.optical_flow import optical_flow_settings_from
                from annolid.segmentation.cutie_vos.runtime import (
                    build_tracking_video_processor,
                )

                flow_settings = optical_flow_settings_from(self.optical_flow_manager)
                processor_kwargs = dict(
                    save_image_to_disk=False,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                    t_max_value=self.t_max_value,
                    use_cpu_only=self.use_cpu_only,
                    auto_recovery_missing_instances=self.auto_recovery_missing_instances,
                    save_video_with_color_mask=self.save_video_with_color_mask,
                    **flow_settings,
                    results_folder=str(self.video_results_folder)
                    if self.video_results_folder
                    else None,
                )
                self.video_processor = build_tracking_video_processor(
                    video_path=self.video_file,
                    model_name=model_name,
                    **processor_kwargs,
                )
                if self._is_cowtracker_model(model_name, model_weight):
                    # In the predict-next workflow, advance CowTracker windows one
                    # frame at a time to reduce temporal jumps.
                    self.video_processor.stride = 1
            if getattr(self, "seg_pred_thread", None) is not None:
                try:
                    if self.seg_pred_thread.isRunning():
                        logger.warning(
                            "Prediction thread already running; stop it before starting a new run."
                        )
                        self.stop_prediction()
                        return
                except RuntimeError:
                    self.seg_pred_thread = None

            old_thread = getattr(self, "seg_pred_thread", None)
            if isinstance(old_thread, QtCore.QThread):
                try:
                    if not old_thread.isRunning():
                        old_thread.deleteLater()
                except RuntimeError:
                    pass

            self.seg_pred_thread = QtCore.QThread(self)
            stop_when_lost_tracking_instance = (
                self.stepSizeWidget.occclusion_checkbox.isChecked()
                or self.automatic_pause_enabled
            )
            inference_step = 1
            inference_start_frame = max(0, int(self.frame_number or 0) + 1)
            inference_end_frame = None
            inference_skip_existing = True
            forced_start_frame = getattr(self, "_prediction_forced_start_frame", None)
            has_forced_start = forced_start_frame is not None
            if has_forced_start:
                try:
                    inference_start_frame = max(0, int(forced_start_frame))
                except Exception:
                    has_forced_start = False
            is_point_tracking_model = bool(
                self._is_cotracker_model(model_name, model_weight)
                or self._is_cowtracker_model(model_name, model_weight)
                or self._is_dino_kpseg_tracker_model(model_name, model_weight)
                or self._is_dino_keypoint_model(model_name, model_weight)
            )
            if self.video_results_folder and not has_forced_start:
                try:
                    results_folder = Path(self.video_results_folder)
                    # Refresh existing predicted marks before a resumed run so users
                    # can see completed sections immediately.
                    self._scan_prediction_folder(str(results_folder))
                    manual_seed_max = -1
                    try:
                        for name in find_manual_labeled_json_files(str(results_folder)):
                            frame_idx = int(get_frame_number_from_json(name))
                            manual_seed_max = max(manual_seed_max, frame_idx)
                    except Exception:
                        manual_seed_max = -1

                    max_existing = self._max_predicted_frame_index(
                        results_folder, include_store=False
                    )
                    if self._is_cutie_tracking_model(model_name):
                        # CUTIE should not scan/predict before the first manual seed.
                        # Resume from existing predictions when available, otherwise
                        # continue after the latest manual seed (multi-seed safe).
                        inference_start_frame = (
                            self._resolve_cutie_start_frame_from_seed_state(
                                int(inference_start_frame),
                                int(max_existing),
                                int(manual_seed_max),
                            )
                        )
                    elif manual_seed_max >= 0:
                        inference_start_frame = max(
                            int(inference_start_frame), int(manual_seed_max) + 1
                        )
                    elif (
                        not is_point_tracking_model
                        and should_start_predictions_from_frame0(results_folder)
                    ):
                        inference_start_frame = 0
                    else:
                        if max_existing >= int(inference_start_frame):
                            inference_start_frame = int(max_existing) + 1
                except Exception:
                    pass
            elif has_forced_start:
                logger.info(
                    "Prediction restart forced from seed frame: %s",
                    int(inference_start_frame),
                )
            if self.num_frames and int(inference_start_frame) >= int(self.num_frames):
                try:
                    self.frame_number = int(self.num_frames) - 1
                except Exception:
                    pass
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Prediction Complete"),
                    self.tr(
                        "All frames already have predictions. Delete or prune later "
                        "predictions if you want to rerun from an earlier frame."
                    ),
                )
                return

            # CUTIE resume robustness:
            # if we plan to continue from frame N, ensure frame N-1 is a real
            # seed (PNG+JSON). Otherwise CUTIE may fall back to an older seed.
            if (
                self._is_cutie_tracking_model(model_name)
                and int(inference_start_frame) > 1
            ):
                resume_seed_frame = int(inference_start_frame) - 1
                if not self._has_cutie_seed_frame(resume_seed_frame):
                    # Try to auto-materialize the seed from stored predictions.
                    _ = self._materialize_prediction_seed(resume_seed_frame)
                if not self._has_cutie_seed_frame(resume_seed_frame):
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        self.tr("Missing Resume Seed"),
                        self.tr(
                            "To resume from frame {start}, CUTIE needs a seed at frame {seed}, "
                            "but only older seeds are available.\n\n"
                            "Do you want to jump to frame {seed} now, review/edit, save, and continue?"
                        ).format(
                            start=int(inference_start_frame),
                            seed=int(resume_seed_frame),
                        ),
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if answer == QtWidgets.QMessageBox.Yes:
                        try:
                            self.set_frame_number(int(resume_seed_frame))
                            if self.seekbar is not None:
                                self.seekbar.setValue(int(resume_seed_frame))
                        except Exception:
                            pass
                        return

            if self.step_size < 0:
                end_frame = self.num_frames + self.step_size
            else:
                end_frame = (int(inference_start_frame) - 1) + to_frame * self.step_size
            if self._is_cutie_tracking_model(model_name):
                # CUTIE already processes seed-defined segments sequentially inside a
                # single run. Avoid GUI-level chunk restarts that make progress look
                # stalled and repeatedly restart from earlier seeds.
                end_frame = self.num_frames - 1
            if self._is_cotracker_model(model_name, model_weight):
                # CoTracker can stream windows efficiently to the end of the video.
                end_frame = self.num_frames - 1
            if end_frame >= self.num_frames:
                end_frame = self.num_frames - 1
            self._prediction_run_start_frame = int(inference_start_frame)
            if self._is_cotracker_model(
                model_name, model_weight
            ) or self._is_cowtracker_model(model_name, model_weight):
                # For CoTracker/CoWTracker, continue launching chunks until prediction
                # reaches the true last frame.
                self._prediction_auto_continue_to_end = True
                self._prediction_target_end_frame = int(self.num_frames) - 1
            elif self._is_cutie_tracking_model(model_name):
                # CUTIE runs to the target end in a single worker run.
                self._prediction_auto_continue_to_end = False
                self._prediction_target_end_frame = int(end_frame)
            else:
                self._prediction_target_end_frame = int(end_frame)
            watch_start_frame = int(self.frame_number or 0)
            if self._is_dino_kpseg_tracker_model(model_name, model_weight):
                end_frame = self.num_frames - 1
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.process_video,
                    start_frame=int(inference_start_frame),
                    end_frame=end_frame,
                    step=1,
                    pred_worker=None,
                )
                self.video_processor.set_pred_worker(self.pred_worker)
                self.pred_worker._kwargs["pred_worker"] = self.pred_worker
                watch_start_frame = int(inference_start_frame)
            elif self._is_dino_keypoint_model(model_name, model_weight):
                end_frame = self.num_frames - 1
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.process_video,
                    start_frame=int(inference_start_frame),
                    end_frame=end_frame,
                    step=1,
                    pred_worker=None,
                )
                self.video_processor.set_pred_worker(self.pred_worker)
                self.pred_worker._kwargs["pred_worker"] = self.pred_worker
                watch_start_frame = int(inference_start_frame)
            elif self._is_efficienttam_model(model_name, model_weight):
                frame_idx = max(int(inference_start_frame), 0)
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor,
                    frame_idx=frame_idx,
                )
                watch_start_frame = int(frame_idx)
            elif self.sam2_manager.is_sam2_model(model_name, model_weight):
                frame_idx = max(int(inference_start_frame), 0)
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor,
                    frame_idx=frame_idx,
                )
                watch_start_frame = int(frame_idx)
            elif self.sam3_manager.is_sam3_model(model_name, model_weight):
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor,
                )
                watch_start_frame = int(self.frame_number or 0)
            elif self._is_dino_kpseg_model(model_name, model_weight):
                # Second-pass keypoint enrichment should update existing polygon-only
                # annotations instead of skipping already-labeled frames.
                try:
                    if self.video_results_folder:
                        existing_max = self._max_predicted_frame_index(
                            Path(self.video_results_folder)
                        )
                        if int(existing_max) >= 0:
                            inference_start_frame = 0
                            inference_skip_existing = False
                except Exception:
                    pass
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.run_inference,
                    source=self.video_file,
                    start_frame=int(inference_start_frame),
                    end_frame=inference_end_frame,
                    step=int(inference_step),
                    skip_existing=bool(inference_skip_existing),
                    save_pose_bbox=self._save_pose_bbox,
                )
                watch_start_frame = int(inference_start_frame)
            elif self._is_yolo_model(model_name, model_weight):
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.run_inference,
                    source=self.video_file,
                    visual_prompts=visual_prompts if visual_prompts else None,
                    start_frame=int(inference_start_frame),
                    end_frame=inference_end_frame,
                    step=int(inference_step),
                    skip_existing=True,
                    save_pose_bbox=self._save_pose_bbox,
                )
                watch_start_frame = int(inference_start_frame)
            else:
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.process_video_frames,
                    start_frame=int(inference_start_frame),
                    end_frame=end_frame,
                    step=self.step_size,
                    is_cutie=False
                    if (
                        self._is_cotracker_model(model_name, model_weight)
                        or self._is_cowtracker_model(model_name, model_weight)
                    )
                    else True,
                    mem_every=self.step_size,
                    point_tracking=(
                        self._is_cotracker_model(model_name, model_weight)
                        or self._is_cowtracker_model(model_name, model_weight)
                    ),
                    has_occlusion=stop_when_lost_tracking_instance,
                )
                self.video_processor.set_pred_worker(self.pred_worker)
                watch_start_frame = int(inference_start_frame)

            if self.video_results_folder:
                try:
                    self._setup_prediction_folder_watcher(
                        str(self.video_results_folder),
                        start_frame=int(watch_start_frame),
                    )
                except Exception:
                    logger.debug(
                        "Failed to start prediction progress watcher.", exc_info=True
                    )
            # Forced restart is a one-shot hint set by prediction cleanup actions.
            self._prediction_forced_start_frame = None
            logger.info("Prediction started from frame: %s", int(watch_start_frame))
            self.stepSizeWidget.predict_button.setText("Stop")
            self.stepSizeWidget.predict_button.setStyleSheet(
                "background-color: red; color: white;"
            )
            self.stop_prediction_flag = True
            self.pred_worker.moveToThread(self.seg_pred_thread)
            self.seg_pred_thread.started.connect(
                self.pred_worker.run, QtCore.Qt.QueuedConnection
            )
            self.pred_worker.result_signal.connect(
                self.lost_tracking_instance, QtCore.Qt.QueuedConnection
            )
            self.pred_worker.progress_signal.connect(
                self._update_progress_bar, QtCore.Qt.QueuedConnection
            )
            self.pred_worker.finished_signal.connect(
                self.predict_is_ready, QtCore.Qt.QueuedConnection
            )
            self.pred_worker.finished_signal.connect(
                self.seg_pred_thread.quit, QtCore.Qt.QueuedConnection
            )
            self.pred_worker.finished_signal.connect(
                self.pred_worker.deleteLater, QtCore.Qt.QueuedConnection
            )
            self.seg_pred_thread.finished.connect(
                self._cleanup_prediction_worker, QtCore.Qt.QueuedConnection
            )
            self.seg_pred_thread.finished.connect(
                self.seg_pred_thread.deleteLater, QtCore.Qt.QueuedConnection
            )
            self.seg_pred_thread.start()

    def lost_tracking_instance(self, message):
        if message is None or "#" not in str(message):
            return
        message, current_frame_index = message.split("#")
        try:
            stalled_frame = int(float(current_frame_index))
        except Exception:
            stalled_frame = int(self.frame_number or 0)
        if "missing instance(s)" in message:
            if self._is_cutie_tracking_model():
                try:
                    self._queue_cutie_resume_from_frame(stalled_frame, str(message))
                except Exception:
                    logger.debug(
                        "Failed to queue CUTIE resume workflow.",
                        exc_info=True,
                    )
            else:
                QtWidgets.QMessageBox.information(self, "Stop early", message)
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;"
        )
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False

    def predict_is_ready(self, message):
        pending_resume_frame = getattr(self, "_pending_prediction_resume_frame", None)
        queue_point_tracker_resume = False
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;"
        )
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False
        try:
            if isinstance(message, Exception):
                logger.exception("Prediction worker failed", exc_info=message)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Prediction failed",
                    f"Prediction failed with error:\n{message}",
                )
                return
            message_text = ""
            stop_from_message = False
            if isinstance(message, tuple):
                if message:
                    message_text = str(message[0])
                if len(message) > 1 and isinstance(message[1], bool):
                    stop_from_message = message[1]
            elif message is not None:
                message_text = str(message)

            if (
                message_text.startswith("Stopped")
                or "missing instance(s)" in message_text
            ):
                stop_from_message = True

            if message_text and "last frame" in message_text:
                stop_from_message = True
                QtWidgets.QMessageBox.information(self, "Stop early", message_text)
            if self._prediction_stop_requested or stop_from_message:
                logger.info(
                    "Prediction stopped early; skipping tracking CSV conversion."
                )
            else:
                if self.video_loader is not None:
                    max_predicted = -1
                    try:
                        if self.video_results_folder:
                            max_predicted = self._max_predicted_frame_index(
                                Path(self.video_results_folder)
                            )
                    except Exception:
                        max_predicted = -1
                    expected_end_frame = int(
                        getattr(self, "_prediction_target_end_frame", -1)
                    )
                    if expected_end_frame < 0:
                        expected_end_frame = int(self.num_frames or 0) - 1
                    logger.info(
                        "Predicted frames available: max_frame=%s of end_frame=%s",
                        int(max_predicted),
                        int(expected_end_frame),
                    )
                    auto_continue_to_end = bool(
                        getattr(self, "_prediction_auto_continue_to_end", False)
                    )
                    run_start_frame = int(
                        getattr(self, "_prediction_run_start_frame", -1)
                    )
                    if (
                        expected_end_frame >= 0
                        and int(max_predicted) >= int(expected_end_frame)
                        and int(max_predicted) >= 0
                    ):
                        skip_csv = bool(
                            getattr(
                                self,
                                "_skip_tracking_csv_overwrite_for_keypoint_round",
                                False,
                            )
                        )
                        if skip_csv:
                            logger.info(
                                "Keypoint round completed; refreshing tracking CSV only (skip tracked CSV write)."
                            )
                            self.convert_json_to_tracked_csv(
                                include_tracked_output=False,
                                force_rewrite_tracking_csv=True,
                            )
                        else:
                            self.convert_json_to_tracked_csv()
                    else:
                        if (
                            auto_continue_to_end
                            and expected_end_frame >= 0
                            and int(max_predicted) >= 0
                            and int(max_predicted) < int(expected_end_frame)
                        ):
                            # Keep advancing point-tracker inference in chunks
                            # until the true end frame is reached.
                            if (
                                run_start_frame >= 0
                                and int(max_predicted) < run_start_frame
                            ):
                                logger.warning(
                                    "Prediction made no forward progress (max=%d, start=%d); stopping auto-continue.",
                                    int(max_predicted),
                                    int(run_start_frame),
                                )
                            else:
                                try:
                                    self.frame_number = int(max_predicted)
                                except Exception:
                                    pass
                                queue_point_tracker_resume = True
                        logger.info("Prediction has not reached target end frame yet.")
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
        self.reset_predict_button()
        self._finalize_prediction_progress("Manual prediction worker finished.")
        if queue_point_tracker_resume and not self._prediction_stop_requested:
            chunk_to_frame = int(getattr(self, "_prediction_chunk_to_frame", 60))
            QtCore.QTimer.singleShot(
                0, lambda: self.predict_from_next_frame(to_frame=chunk_to_frame)
            )
            self._prediction_stop_requested = False
            return
        if pending_resume_frame is not None:
            try:
                self._pending_prediction_resume_frame = None
            except Exception:
                pass
            should_resume = (
                (not self._prediction_stop_requested)
                and self.stop_prediction_flag is False
                and self._is_cutie_tracking_model()
            )
            if should_resume:
                if self._materialize_prediction_seed(int(pending_resume_frame)):
                    try:
                        self.frame_number = int(pending_resume_frame)
                    except Exception:
                        pass
                    QtCore.QTimer.singleShot(
                        0, lambda: self.predict_from_next_frame(to_frame=60)
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Resume Failed"),
                        self.tr(
                            "Could not save a seed at frame {frame}. "
                            "Please save that frame manually, then run prediction again."
                        ).format(frame=int(pending_resume_frame)),
                    )
        self._prediction_stop_requested = False
        self._skip_tracking_csv_overwrite_for_keypoint_round = False

    def reset_predict_button(self):
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;"
        )
