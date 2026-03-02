from __future__ import annotations

import subprocess
from pathlib import Path

from qtpy import QtWidgets

from annolid.data import videos
from annolid.gui.widgets import ExtractFrameDialog
from annolid.gui.widgets import ProgressingWindow
from annolid.gui.widgets import TrainModelDialog
from annolid.gui.workers import FlexibleWorker
from annolid.segmentation.dino_kpseg import defaults as dino_defaults
from annolid.utils.runs import shared_runs_root

try:
    import torch
except ImportError:
    torch = None


class TrainingWorkflowMixin:
    """Frame extraction and training workflow dialogs."""

    def frames(self):
        """Extract frames based on the selected algos."""
        dlg = ExtractFrameDialog(self.video_file)
        video_file = None
        out_dir = None

        if dlg.exec_():
            video_file = dlg.video_file
            num_frames = dlg.num_frames
            algo = dlg.algo
            out_dir = dlg.out_dir
            start_seconds = dlg.start_sconds
            end_seconds = dlg.end_seconds
            sub_clip = isinstance(start_seconds, int) and isinstance(end_seconds, int)

        if video_file is None:
            return
        out_frames_gen = videos.extract_frames(
            video_file,
            num_frames=num_frames,
            algo=algo,
            out_dir=out_dir,
            sub_clip=sub_clip,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )

        pw = ProgressingWindow(out_frames_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        if out_dir is None:
            out_frames_dir = str(Path(video_file).resolve().with_suffix(""))
        else:
            out_frames_dir = str(Path(out_dir) / Path(video_file).name)

        if start_seconds is not None and end_seconds is not None:
            out_frames_dir = f"{out_frames_dir}_{start_seconds}_{end_seconds}"
        out_frames_dir = f"{out_frames_dir}_{algo}"

        self.annotation_dir = out_frames_dir

        QtWidgets.QMessageBox.about(
            self,
            "Finished",
            f"Done! Results are in folder: \
                                         {out_frames_dir}",
        )
        self.statusBar().showMessage(self.tr("Finished extracting frames."))
        self.importDirImages(out_frames_dir)

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

        if config_file is None:
            return
        if algo == "YOLO":
            data_config = self.yolo_training_manager.prepare_data_config(config_file)
            if data_config is None:
                return
            self.yolo_training_manager.start_training(
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
            )

        elif algo == "DINO KPSEG":
            dino_data_format = getattr(dlg, "dino_data_format", "auto")
            data_config = self.dino_kpseg_training_manager.prepare_data_config(
                config_file, data_format=str(dino_data_format or "auto")
            )
            if data_config is None:
                return
            self.dino_kpseg_training_manager.start_training(
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
            )

        elif algo == "YOLACT":
            if torch is None or not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(
                    self,
                    "GPU or PyTorch unavailable",
                    "PyTorch with CUDA support is required to train YOLACT models.",
                )
                return

            subprocess.Popen(
                [
                    "annolid-train",
                    f"--config={config_file}",
                    f"--batch_size={batch_size}",
                ]
            )

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
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor

            dataset_dir = str(Path(config_file).parent)
            segmentor = Segmentor(
                dataset_dir,
                out_dir or str(shared_runs_root()),
                max_iterations=max_iterations,
                batch_size=batch_size,
                model_pth_path=model_path,
            )
            out_runs_dir = segmentor.out_put_dir
            try:
                self.seg_train_thread.start()
                train_worker = FlexibleWorker(task_function=segmentor.train)
                train_worker.moveToThread(self.seg_train_thread)
                train_worker.start_signal.connect(train_worker.run)
                train_worker.start_signal.emit()
            except Exception:
                segmentor.train()

            QtWidgets.QMessageBox.about(
                self,
                "Started.",
                f"Training in background... \
                                        Results will be saved to folder: \
                                         {str(out_runs_dir)} \
                                         Please do not close Annolid GUI.",
            )
            self.statusBar().showMessage(self.tr("Training..."))
