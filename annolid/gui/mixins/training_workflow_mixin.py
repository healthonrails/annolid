from __future__ import annotations

import contextlib
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Mapping, Optional, Sequence

from qtpy import QtCore, QtWidgets

from annolid.utils.runs import allocate_run_dir, shared_runs_root


class TrainingWorkflowMixin:
    """Frame selection/extraction and training workflow dialogs."""

    def _sync_label_queue_slider_marks(self, frames: Sequence[int]) -> None:
        """Render ``label_queue`` marks on the active video seekbar."""
        seekbar = getattr(self, "seekbar", None)
        if seekbar is None:
            return
        from annolid.gui.widgets.video_slider import VideoSliderMark

        total_frames = getattr(self, "num_frames", None)
        valid = sorted(
            {
                int(f)
                for f in (frames or ())
                if int(f) >= 0 and (total_frames is None or int(f) < int(total_frames))
            }
        )
        seekbar.blockSignals(True)
        try:
            seekbar.removeMarksByType("label_queue")
            for fidx in valid:
                seekbar.addMark(
                    VideoSliderMark(mark_type="label_queue", val=fidx), update=False
                )
        finally:
            seekbar.blockSignals(False)
        seekbar._update_visual_positions()
        seekbar.update()

    def _show_label_queue_dock(self) -> None:
        dock = getattr(self, "label_queue_dock", None)
        if dock is not None:
            dock.setVisible(True)
            dock.show()
            dock.raise_()

    def _queue_frames_for_labeling(
        self,
        frame_indices: Sequence[int],
        *,
        video_path: Optional[str | Path] = None,
        task: str = "general",
        algo: str = "custom",
        feature_backend: str = "none",
        reasons_by_frame: Optional[Mapping[int, str]] = None,
        scores_by_frame: Optional[Mapping[int, float]] = None,
        role: str = "custom",
        jump_to_first: bool = True,
    ) -> None:
        """Populate the interactive Label Queue dock with an explicit set of video frames."""
        target_video = video_path or getattr(self, "video_file", None)
        if not target_video:
            self.statusBar().showMessage(
                self.tr("Open a video to queue frames for labeling."), 4000
            )
            return
        from annolid.services.frame_selection import (
            build_selection_result_from_indices,
        )

        ann_dir = getattr(self, "video_results_folder", None) or getattr(
            self, "annotation_dir", None
        )
        result = build_selection_result_from_indices(
            target_video,
            frame_indices,
            task=task,
            algo=algo,
            feature_backend=feature_backend,
            annotation_dir=ann_dir,
            reasons_by_frame=reasons_by_frame,
            scores_by_frame=scores_by_frame,
            role=role,
            persist_queue_manifest=True,
        )
        self._activate_frame_selection_result(
            result, open_output_in_files=False, jump_to_first=jump_to_first
        )

    def _activate_frame_selection_result(
        self,
        result,
        *,
        open_output_in_files: bool = True,
        jump_to_first: bool = True,
    ) -> None:
        """Apply a completed ``FrameSelectionResult`` to the main window UI."""
        if not result.selected_items:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("No Frames Selected"),
                self.tr("No matching frames were selected with the current settings."),
            )
            return

        if result.extract_images and result.output_dir:
            out_dir = str(result.output_dir)
            self.annotation_dir = out_dir
            self.statusBar().showMessage(
                self.tr("Extracted %s frames (%s · %s) to %s")
                % (
                    len(result.saved_paths),
                    result.algo,
                    result.feature_backend,
                    out_dir,
                ),
                6000,
            )
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Frames Extracted"),
                self.tr("Extracted %s frame(s) (%s, %s) to:\n%s")
                % (
                    len(result.saved_paths),
                    result.algo,
                    result.feature_backend,
                    out_dir,
                ),
            )
            if open_output_in_files and hasattr(self, "importDirImages"):
                self.importDirImages(out_dir)
            return

        # Zero-extraction workflow: queue selected frames directly inside the video!
        current_video = getattr(self, "video_file", None)
        target_video = Path(result.video_path).expanduser().resolve()
        if (
            not current_video
            or Path(str(current_video)).expanduser().resolve() != target_video
        ):
            if hasattr(self, "openVideo"):
                self.openVideo(
                    from_video_list=True,
                    video_path=str(target_video),
                    programmatic_call=True,
                )

        queue_widget = getattr(self, "frame_selection_queue_widget", None)
        if queue_widget is not None:
            queue_widget.set_selection_result(result, persist=True)
            self._show_label_queue_dock()
            if jump_to_first:
                queue_widget.jump_to_next_unlabeled()
        else:
            self._sync_label_queue_slider_marks(result.selected_indices)
            if (
                jump_to_first
                and result.selected_indices
                and hasattr(self, "set_frame_number")
            ):
                self.set_frame_number(int(result.selected_indices[0]))

        cache_note = " (cached embeddings)" if result.embedding_cache_hit else ""
        self.statusBar().showMessage(
            self.tr(
                "Queued %s %s frames (%s · %s%s) — label directly in video without extracting PNGs."
            )
            % (
                len(result.selected_items),
                result.task,
                result.algo,
                result.feature_backend,
                cache_note,
            ),
            6000,
        )

    def frames(self):
        """Select or extract diverse frames for pose/segmentation labeling."""
        from annolid.gui.widgets.extract_frame_dialog import (
            ExtractFrameDialog,
            FrameSelectionWorker,
        )

        active_worker = getattr(self, "_frame_selection_worker", None)
        if active_worker is not None and active_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Frame Selection Running"),
                self.tr(
                    "A frame selection job is already running. Please wait or cancel it."
                ),
            )
            return

        slider_sel = None
        seekbar = getattr(self, "seekbar", None)
        if seekbar is not None and hasattr(seekbar, "hasSelection"):
            try:
                if seekbar.hasSelection():
                    slider_sel = seekbar.getSelection()
            except Exception:
                slider_sel = None

        ann_dir = getattr(self, "video_results_folder", None) or getattr(
            self, "annotation_dir", None
        )
        dlg = ExtractFrameDialog(
            self.video_file,
            self,
            annotation_dir=ann_dir,
            owner_window=self,
            total_frames=getattr(self, "num_frames", None),
            fps=getattr(self, "fps", None),
            slider_selection=slider_sel,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        video_file = dlg.video_file
        if not video_file:
            return
        config = dlg.build_config()

        progress_dlg = QtWidgets.QProgressDialog(
            self.tr("Selecting frames…"),
            self.tr("Cancel"),
            0,
            100,
            self,
        )
        progress_dlg.setWindowTitle(
            self.tr("Extracting Frames")
            if config.extract_images
            else self.tr("Selecting Frames for Label Queue")
        )
        progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setAutoClose(False)
        progress_dlg.setAutoReset(False)
        progress_dlg.setValue(0)
        progress_dlg.show()

        worker = FrameSelectionWorker(video_file, config, parent=self)
        self._frame_selection_worker = worker
        self._frame_selection_progress_dlg = progress_dlg

        def _cleanup() -> None:
            dlg_ref = getattr(self, "_frame_selection_progress_dlg", None)
            if dlg_ref is progress_dlg:
                self._frame_selection_progress_dlg = None
            with contextlib.suppress(Exception):
                progress_dlg.close()
                progress_dlg.deleteLater()
            if getattr(self, "_frame_selection_worker", None) is worker:
                self._frame_selection_worker = None
            worker.deleteLater()

        def _on_progress(pct: int, msg: str) -> None:
            with contextlib.suppress(RuntimeError):
                progress_dlg.setValue(int(pct))
                progress_dlg.setLabelText(str(msg))

        def _on_finished(result) -> None:
            _cleanup()
            self._activate_frame_selection_result(
                result,
                open_output_in_files=bool(config.open_output_in_files),
                jump_to_first=True,
            )

        def _on_failed(err_msg: str) -> None:
            _cleanup()
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Frame Selection Failed"),
                str(err_msg),
            )
            self.statusBar().showMessage(
                self.tr("Frame selection failed: %s") % err_msg, 5000
            )

        def _on_cancelled() -> None:
            _cleanup()
            self.statusBar().showMessage(self.tr("Frame selection cancelled."), 4000)

        def _on_cancel_clicked() -> None:
            worker.stop()
            with contextlib.suppress(RuntimeError):
                progress_dlg.setLabelText(self.tr("Cancelling…"))

        progress_dlg.canceled.connect(_on_cancel_clicked)
        worker.progress.connect(_on_progress)
        worker.finishedResult.connect(_on_finished)
        worker.failed.connect(_on_failed)
        worker.cancelled.connect(_on_cancelled)
        worker.start()

    def tracks(self):
        """Open the inference wizard for tracking and batch inference."""
        if self.tracking_controller.is_tracking_busy():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Tracking Busy"),
                self.tr("Another tracking job is already active. Please wait."),
            )
            return

        self.open_inference_wizard()

    def models(self):
        """Train a model with the provided dataset and selected params."""
        from annolid.gui.widgets.train_model_dialog import TrainModelDialog

        from annolid.segmentation.dino_kpseg import defaults as dino_defaults

        dlg = TrainModelDialog()
        config_file = None
        out_dir = None
        max_iterations = 2000
        batch_size = 8
        model_path = None

        if dlg.exec_():
            config_file = dlg.config_file
            batch_size = dlg.batch_size
            algo = dlg.algo
            out_dir = dlg.out_dir
            max_iterations = dlg.max_iterations
            model_path = dlg.trained_model
            epochs = dlg.epochs
            image_size = dlg.image_size
            yolo_model_file = dlg.yolo_model_file
            yolo_device = getattr(dlg, "yolo_device", None)
            yolo_plots = getattr(dlg, "yolo_plots", False)
            yolo_resume = bool(getattr(dlg, "yolo_resume", False))
            yolo_train_overrides = (
                dlg.get_yolo_train_overrides()
                if hasattr(dlg, "get_yolo_train_overrides")
                else {}
            )
            dino_model_name = getattr(dlg, "dino_model_name", None)
            dino_short_side = getattr(dlg, "dino_short_side", dino_defaults.SHORT_SIDE)
            dino_layers = getattr(dlg, "dino_layers", dino_defaults.LAYERS)
            dino_radius_px = getattr(dlg, "dino_radius_px", dino_defaults.RADIUS_PX)
            dino_hidden_dim = getattr(dlg, "dino_hidden_dim", dino_defaults.HIDDEN_DIM)
            dino_head_type = getattr(dlg, "dino_head_type", dino_defaults.HEAD_TYPE)
            dino_attn_heads = getattr(dlg, "dino_attn_heads", dino_defaults.ATTN_HEADS)
            dino_attn_layers = getattr(
                dlg, "dino_attn_layers", dino_defaults.ATTN_LAYERS
            )
            dino_lr_pair_loss_weight = getattr(
                dlg, "dino_lr_pair_loss_weight", dino_defaults.LR_PAIR_LOSS_WEIGHT
            )
            dino_lr_pair_margin_px = getattr(
                dlg, "dino_lr_pair_margin_px", dino_defaults.LR_PAIR_MARGIN_PX
            )
            dino_lr_side_loss_weight = getattr(
                dlg, "dino_lr_side_loss_weight", dino_defaults.LR_SIDE_LOSS_WEIGHT
            )
            dino_lr_side_loss_margin = getattr(
                dlg, "dino_lr_side_loss_margin", dino_defaults.LR_SIDE_LOSS_MARGIN
            )
            dino_obj_loss_weight = getattr(
                dlg, "dino_obj_loss_weight", dino_defaults.OBJ_LOSS_WEIGHT
            )
            dino_box_loss_weight = getattr(
                dlg, "dino_box_loss_weight", dino_defaults.BOX_LOSS_WEIGHT
            )
            dino_inst_loss_weight = getattr(
                dlg, "dino_inst_loss_weight", dino_defaults.INST_LOSS_WEIGHT
            )
            dino_multitask_aux_warmup_epochs = getattr(
                dlg,
                "dino_multitask_aux_warmup_epochs",
                dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
            )
            dino_lr = getattr(dlg, "dino_lr", dino_defaults.LR)
            dino_threshold = getattr(dlg, "dino_threshold", dino_defaults.THRESHOLD)
            dino_bce_type = getattr(dlg, "dino_bce_type", dino_defaults.BCE_TYPE)
            dino_focal_alpha = getattr(
                dlg, "dino_focal_alpha", dino_defaults.FOCAL_ALPHA
            )
            dino_focal_gamma = getattr(
                dlg, "dino_focal_gamma", dino_defaults.FOCAL_GAMMA
            )
            dino_coord_warmup_epochs = getattr(
                dlg, "dino_coord_warmup_epochs", dino_defaults.COORD_WARMUP_EPOCHS
            )
            dino_radius_schedule = getattr(
                dlg, "dino_radius_schedule", dino_defaults.RADIUS_SCHEDULE
            )
            dino_radius_start_px = getattr(
                dlg, "dino_radius_start_px", dino_defaults.RADIUS_START_PX
            )
            dino_radius_end_px = getattr(
                dlg, "dino_radius_end_px", dino_defaults.RADIUS_END_PX
            )
            dino_overfit_n = getattr(dlg, "dino_overfit_n", 0)
            dino_cache_features = getattr(dlg, "dino_cache_features", True)
            dino_patience = getattr(
                dlg, "dino_patience", dino_defaults.EARLY_STOP_PATIENCE
            )
            dino_min_delta = getattr(
                dlg, "dino_min_delta", dino_defaults.EARLY_STOP_MIN_DELTA
            )
            dino_min_epochs = getattr(
                dlg, "dino_min_epochs", dino_defaults.EARLY_STOP_MIN_EPOCHS
            )
            dino_best_metric = getattr(
                dlg, "dino_best_metric", dino_defaults.BEST_METRIC
            )
            dino_early_stop_metric = getattr(
                dlg, "dino_early_stop_metric", dino_defaults.EARLY_STOP_METRIC
            )
            dino_pck_weighted_weights = getattr(
                dlg, "dino_pck_weighted_weights", dino_defaults.PCK_WEIGHTED_WEIGHTS
            )
            dino_augment_enabled = getattr(
                dlg, "dino_augment_enabled", dino_defaults.AUGMENT_ENABLED
            )
            dino_hflip_prob = getattr(dlg, "dino_hflip_prob", dino_defaults.HFLIP)
            dino_degrees = getattr(dlg, "dino_degrees", dino_defaults.DEGREES)
            dino_translate = getattr(dlg, "dino_translate", dino_defaults.TRANSLATE)
            dino_scale = getattr(dlg, "dino_scale", dino_defaults.SCALE)
            dino_brightness = getattr(dlg, "dino_brightness", dino_defaults.BRIGHTNESS)
            dino_contrast = getattr(dlg, "dino_contrast", dino_defaults.CONTRAST)
            dino_saturation = getattr(dlg, "dino_saturation", dino_defaults.SATURATION)
            dino_seed = getattr(dlg, "dino_seed", -1)
            dino_tb_add_graph = getattr(dlg, "dino_tb_add_graph", False)
            dino_tb_projector = getattr(
                dlg, "dino_tb_projector", dino_defaults.TB_PROJECTOR
            )
            dino_tb_projector_split = getattr(
                dlg, "dino_tb_projector_split", dino_defaults.TB_PROJECTOR_SPLIT
            )
            dino_tb_projector_max_images = getattr(
                dlg,
                "dino_tb_projector_max_images",
                dino_defaults.TB_PROJECTOR_MAX_IMAGES,
            )
            dino_tb_projector_max_patches = getattr(
                dlg,
                "dino_tb_projector_max_patches",
                dino_defaults.TB_PROJECTOR_MAX_PATCHES,
            )
            dino_tb_projector_per_image_per_keypoint = getattr(
                dlg,
                "dino_tb_projector_per_image_per_keypoint",
                dino_defaults.TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT,
            )
            dino_tb_projector_pos_threshold = getattr(
                dlg,
                "dino_tb_projector_pos_threshold",
                dino_defaults.TB_PROJECTOR_POS_THRESHOLD,
            )
            dino_tb_projector_crop_px = getattr(
                dlg, "dino_tb_projector_crop_px", dino_defaults.TB_PROJECTOR_CROP_PX
            )
            dino_tb_projector_sprite_border_px = getattr(
                dlg,
                "dino_tb_projector_sprite_border_px",
                dino_defaults.TB_PROJECTOR_SPRITE_BORDER_PX,
            )
            dino_tb_projector_add_negatives = getattr(
                dlg,
                "dino_tb_projector_add_negatives",
                dino_defaults.TB_PROJECTOR_ADD_NEGATIVES,
            )
            dino_tb_projector_neg_threshold = getattr(
                dlg,
                "dino_tb_projector_neg_threshold",
                dino_defaults.TB_PROJECTOR_NEG_THRESHOLD,
            )
            dino_tb_projector_negatives_per_image = getattr(
                dlg,
                "dino_tb_projector_negatives_per_image",
                dino_defaults.TB_PROJECTOR_NEGATIVES_PER_IMAGE,
            )
            dino_auto_safe_mode = getattr(dlg, "dino_auto_safe_mode", True)
            dino_workers = getattr(dlg, "dino_workers", 0)
            dino_max_cpu_threads = getattr(dlg, "dino_max_cpu_threads", 8)
            dino_max_interop_threads = getattr(dlg, "dino_max_interop_threads", 1)
            dino_auto_report_enabled = getattr(dlg, "dino_auto_report_enabled", True)
            dino_auto_report_split = getattr(dlg, "dino_auto_report_split", "test")
            dino_auto_report_fallback_to_val = getattr(
                dlg, "dino_auto_report_fallback_to_val", True
            )
            dino_auto_report_thresholds = getattr(
                dlg, "dino_auto_report_thresholds", "4,8,16"
            )
            dino_auto_report_per_keypoint = getattr(
                dlg, "dino_auto_report_per_keypoint", False
            )
            dino_auto_report_paper = getattr(dlg, "dino_auto_report_paper", True)
            dino_freeze_bn = getattr(dlg, "dino_freeze_bn", None)

        if config_file is None:
            return
        yolo_manager = (
            self.ensure_yolo_training_manager()
            if hasattr(self, "ensure_yolo_training_manager")
            else getattr(self, "yolo_training_manager", None)
        )
        dino_manager = (
            self.ensure_dino_kpseg_training_manager()
            if hasattr(self, "ensure_dino_kpseg_training_manager")
            else getattr(self, "dino_kpseg_training_manager", None)
        )
        if algo == "YOLO":
            if yolo_manager is None:
                return
            expected_task = yolo_manager._infer_expected_task(
                yolo_model_file,
                model_path=model_path,
            )
            data_config = yolo_manager.prepare_data_config(
                config_file,
                expected_task=expected_task,
            )
            if data_config is None:
                return
            yolo_manager.start_training(
                yolo_model_file=yolo_model_file,
                model_path=model_path,
                data_config_path=data_config,
                epochs=epochs,
                image_size=image_size,
                batch_size=batch_size,
                device=yolo_device,
                plots=yolo_plots,
                train_overrides=yolo_train_overrides,
                out_dir=out_dir,
                resume=yolo_resume,
            )

        elif algo == "DINO KPSEG":
            if dino_manager is None:
                return
            dino_data_format = getattr(dlg, "dino_data_format", "auto")
            data_config = dino_manager.prepare_data_config(
                config_file, data_format=str(dino_data_format or "auto")
            )
            if data_config is None:
                return
            dino_manager.start_training(
                data_config_path=data_config,
                data_format=str(dino_data_format or "auto"),
                out_dir=out_dir,
                model_name=str(dino_model_name or ""),
                short_side=int(dino_short_side),
                layers=str(dino_layers or dino_defaults.LAYERS),
                radius_px=float(dino_radius_px),
                hidden_dim=int(dino_hidden_dim),
                lr=float(dino_lr),
                epochs=int(epochs),
                batch_size=int(batch_size),
                threshold=float(dino_threshold),
                bce_type=str(dino_bce_type or dino_defaults.BCE_TYPE),
                focal_alpha=float(dino_focal_alpha),
                focal_gamma=float(dino_focal_gamma),
                coord_warmup_epochs=int(dino_coord_warmup_epochs),
                radius_schedule=str(
                    dino_radius_schedule or dino_defaults.RADIUS_SCHEDULE
                ),
                radius_start_px=float(dino_radius_start_px),
                radius_end_px=float(dino_radius_end_px),
                overfit_n=int(dino_overfit_n),
                device=yolo_device,
                cache_features=bool(dino_cache_features),
                head_type=str(dino_head_type or dino_defaults.HEAD_TYPE),
                attn_heads=int(dino_attn_heads),
                attn_layers=int(dino_attn_layers),
                lr_pair_loss_weight=float(dino_lr_pair_loss_weight),
                lr_pair_margin_px=float(dino_lr_pair_margin_px),
                lr_side_loss_weight=float(dino_lr_side_loss_weight),
                lr_side_loss_margin=float(dino_lr_side_loss_margin),
                obj_loss_weight=float(dino_obj_loss_weight),
                box_loss_weight=float(dino_box_loss_weight),
                inst_loss_weight=float(dino_inst_loss_weight),
                multitask_aux_warmup_epochs=int(dino_multitask_aux_warmup_epochs),
                early_stop_patience=int(dino_patience),
                early_stop_min_delta=float(dino_min_delta),
                early_stop_min_epochs=int(dino_min_epochs),
                best_metric=str(dino_best_metric or dino_defaults.BEST_METRIC),
                early_stop_metric=str(
                    dino_early_stop_metric or dino_defaults.EARLY_STOP_METRIC
                ),
                pck_weighted_weights=str(
                    dino_pck_weighted_weights or dino_defaults.PCK_WEIGHTED_WEIGHTS
                ),
                augment=bool(dino_augment_enabled),
                hflip=float(dino_hflip_prob),
                degrees=float(dino_degrees),
                translate=float(dino_translate),
                scale=float(dino_scale),
                brightness=float(dino_brightness),
                contrast=float(dino_contrast),
                saturation=float(dino_saturation),
                seed=(int(dino_seed) if int(dino_seed) >= 0 else None),
                tb_add_graph=bool(dino_tb_add_graph),
                tb_projector=bool(dino_tb_projector),
                tb_projector_split=str(
                    dino_tb_projector_split or dino_defaults.TB_PROJECTOR_SPLIT
                ),
                tb_projector_max_images=int(dino_tb_projector_max_images),
                tb_projector_max_patches=int(dino_tb_projector_max_patches),
                tb_projector_per_image_per_keypoint=int(
                    dino_tb_projector_per_image_per_keypoint
                ),
                tb_projector_pos_threshold=float(dino_tb_projector_pos_threshold),
                tb_projector_crop_px=int(dino_tb_projector_crop_px),
                tb_projector_sprite_border_px=int(dino_tb_projector_sprite_border_px),
                tb_projector_add_negatives=bool(dino_tb_projector_add_negatives),
                tb_projector_neg_threshold=float(dino_tb_projector_neg_threshold),
                tb_projector_negatives_per_image=int(
                    dino_tb_projector_negatives_per_image
                ),
                auto_safe_mode=bool(dino_auto_safe_mode),
                workers=int(dino_workers),
                max_cpu_threads=int(dino_max_cpu_threads),
                max_interop_threads=int(dino_max_interop_threads),
                auto_report_enabled=bool(dino_auto_report_enabled),
                auto_report_split=str(dino_auto_report_split or "test"),
                auto_report_fallback_to_val=bool(dino_auto_report_fallback_to_val),
                auto_report_thresholds=str(dino_auto_report_thresholds or "4,8,16"),
                auto_report_per_keypoint=bool(dino_auto_report_per_keypoint),
                auto_report_paper=bool(dino_auto_report_paper),
                freeze_bn=dino_freeze_bn,
            )

        elif algo == "YOLACT":
            try:
                import torch
            except ImportError:
                torch = None
            if torch is None or not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(
                    self,
                    "GPU or PyTorch unavailable",
                    "PyTorch with CUDA support is required to train YOLACT models.",
                )
                return

            proc = subprocess.Popen(
                [
                    "annolid-train",
                    f"--config={config_file}",
                    f"--batch_size={batch_size}",
                ]
            )
            if not hasattr(self, "_active_training_subprocesses"):
                self._active_training_subprocesses = []
            self._active_training_subprocesses.append(proc)

            if out_dir is None:
                out_runs_dir = shared_runs_root()
            else:
                out_runs_dir = Path(out_dir) / Path(config_file).name / "runs"

            out_runs_dir.mkdir(exist_ok=True, parents=True)
            QtWidgets.QMessageBox.about(
                self,
                "Started",
                f"Results are in folder: \
                                         {str(out_runs_dir)}",
            )

        elif algo == "MaskRCNN":
            # Build the CLI command so MaskRCNN training runs in a subprocess,
            # consistent with YOLO and YOLACT. Uses torchvision Mask R-CNN.
            dataset_dir = str(Path(config_file).parent)
            if out_dir is None:
                run_dir = allocate_run_dir(
                    task="maskrcnn",
                    model="train",
                    runs_root=shared_runs_root(),
                )
            else:
                run_dir = Path(out_dir).expanduser().resolve()
                run_dir.mkdir(exist_ok=True, parents=True)

            cmd = [
                "annolid-run",
                "train",
                "maskrcnn_detectron2",
                f"--dataset-dir={dataset_dir}",
                f"--max-iterations={max_iterations}",
                f"--batch-size={batch_size}",
                f"--output-dir={run_dir}",
                "--num-workers=0",
            ]
            if model_path:
                cmd.append(f"--weights={model_path}")

            proc = self._launch_monitored_cli_training(
                cmd=cmd,
                task="maskrcnn",
                model="maskrcnn_resnet50_fpn",
                run_dir=run_dir,
                total_steps=int(max_iterations),
                step_unit="iteration",
                missing_cli_title="annolid-run CLI not found",
                missing_cli_hint="annolid-run train maskrcnn_detectron2",
            )
            if proc is None:
                return
            self.statusBar().showMessage(
                self.tr(
                    "Mask R-CNN training started — monitoring in Training Dashboard."
                )
            )

        elif algo == "Keypoint RCNN":
            # Keypoint R-CNN training: torchvision keypointrcnn_resnet50_fpn.
            dataset_dir = str(Path(config_file).parent)
            if out_dir is None:
                run_dir = allocate_run_dir(
                    task="keypoint_rcnn",
                    model="train",
                    runs_root=shared_runs_root(),
                )
            else:
                run_dir = Path(out_dir).expanduser().resolve()
                run_dir.mkdir(exist_ok=True, parents=True)

            cmd = [
                "annolid-run",
                "train",
                "keypoint_rcnn",
                f"--dataset-dir={dataset_dir}",
                f"--max-iterations={max_iterations}",
                f"--batch-size={batch_size}",
                f"--output-dir={run_dir}",
                "--num-workers=0",
            ]
            if model_path:
                cmd.append(f"--weights={model_path}")

            proc = self._launch_monitored_cli_training(
                cmd=cmd,
                task="keypoint_rcnn",
                model="keypointrcnn_resnet50_fpn",
                run_dir=run_dir,
                total_steps=int(max_iterations),
                step_unit="iteration",
                missing_cli_title="annolid-run CLI not found",
                missing_cli_hint="annolid-run train keypoint_rcnn",
            )
            if proc is None:
                return
            self.statusBar().showMessage(
                self.tr(
                    "Keypoint R-CNN training started — monitoring in Training Dashboard."
                )
            )

    def _launch_monitored_cli_training(
        self,
        *,
        cmd: list[str],
        task: str,
        model: str,
        run_dir: Path,
        total_steps: int | None = None,
        step_unit: str = "iteration",
        missing_cli_title: str = "annolid-run CLI not found",
        missing_cli_hint: str = "annolid-run train",
    ) -> subprocess.Popen | None:
        """Spawn a CLI training subprocess, log to ``train.log``, and register in the Training Dashboard."""
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd_line = shlex.join(str(tok) for tok in cmd)
        with contextlib.suppress(OSError):
            (run_dir / "command.txt").write_text(cmd_line + "\n", encoding="utf-8")

        log_path = run_dir / "train.log"
        log_fh = None
        try:
            log_fh = log_path.open("a", encoding="utf-8")
            log_fh.write("Command:\n" + cmd_line + "\n\n")
            log_fh.flush()
        except OSError:
            log_fh = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh if log_fh is not None else None,
                stderr=subprocess.STDOUT if log_fh is not None else None,
            )
        except FileNotFoundError:
            if log_fh is not None:
                with contextlib.suppress(Exception):
                    log_fh.close()
            QtWidgets.QMessageBox.critical(
                self,
                missing_cli_title,
                f"Could not launch '{missing_cli_hint}'.\n"
                "Please ensure the annolid-run CLI is on your PATH.\n\n"
                "Try: pip install -e .",
            )
            return None

        if log_fh is not None:

            def _close_log_when_done(p: subprocess.Popen, fh) -> None:
                with contextlib.suppress(Exception):
                    p.wait()
                with contextlib.suppress(Exception):
                    fh.flush()
                    fh.close()

            threading.Thread(
                target=_close_log_when_done,
                args=(proc, log_fh),
                name=f"annolid-{task}-log-close",
                daemon=True,
            ).start()

        if not hasattr(self, "_active_training_subprocesses"):
            self._active_training_subprocesses = []
        self._active_training_subprocesses = [
            p for p in self._active_training_subprocesses if p.poll() is None
        ]
        self._active_training_subprocesses.append(proc)

        open_dashboard = getattr(self, "open_training_dashboard", None)
        if callable(open_dashboard):
            try:
                dialog = open_dashboard(initial_tab="monitor")
                dashboard = getattr(dialog, "dashboard", None)
                if dashboard is not None and hasattr(
                    dashboard, "register_subprocess_run"
                ):
                    dashboard.register_subprocess_run(
                        task=task,
                        model=model,
                        run_dir=run_dir,
                        process=proc,
                        total_steps=total_steps,
                        step_unit=step_unit,
                    )
            except Exception:
                pass
        return proc
