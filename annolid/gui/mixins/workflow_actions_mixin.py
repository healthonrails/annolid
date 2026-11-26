from __future__ import annotations

import os
import shutil
from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.annotation import coco2labelme, labelme2coco
from annolid.gui.widgets.progressing_dialog import ProgressingWindow
from annolid.gui.widgets.training_dashboard import TrainingDashboardDialog
from annolid.gui.widgets.convert_coco_dialog import ConvertCOODialog
from annolid.gui.widgets.convert_coco2labelme_dialog import ConvertCOCO2LabelMeDialog
from annolid.gui.widgets.glitter2_dialog import Glitter2Dialog
from annolid.utils.logger import logger


class WorkflowActionsMixin:
    """Dataset/training/post-processing action helpers."""

    def coco(self):
        """Convert Labelme annotations to COCO format."""
        coco_dlg = ConvertCOODialog(annotation_dir=self.annotation_dir)
        output_dir = None
        labels_file = None
        input_anno_dir = None
        num_train_frames = 0.7
        output_mode = "segmentation"
        pose_schema_path = None
        if coco_dlg.exec_():
            input_anno_dir = coco_dlg.annotation_dir
            labels_file = coco_dlg.label_list_text
            output_dir = coco_dlg.out_dir
            num_train_frames = coco_dlg.num_train_frames
            output_mode = coco_dlg.output_mode
            pose_schema_path = getattr(coco_dlg, "pose_schema_path", None)
        else:
            return

        if input_anno_dir is None:
            QtWidgets.QMessageBox.about(
                self,
                "No input file or directory",
                "Please select an annotation file or directory first.",
            )
            return

        if output_dir is None:
            export_dir = Path(input_anno_dir).parent / (
                Path(input_anno_dir).name + "_coco_dataset"
            )
        else:
            export_dir = Path(output_dir)
        self.output_dir = str(export_dir)

        label_gen = labelme2coco.convert(
            str(input_anno_dir),
            output_annotated_dir=str(export_dir),
            labels_file=labels_file,
            train_valid_split=num_train_frames,
            output_mode=output_mode,
            pose_schema_path=pose_schema_path,
        )
        pw = ProgressingWindow(label_gen)
        pw.exec_()
        if (
            getattr(pw, "runner_thread", None) is not None
            and pw.runner_thread.isRunning()
        ):
            pw.runner_thread.requestInterruption()
            pw.runner_thread.quit()
            pw.runner_thread.wait(2000)

        self.statusBar().showMessage(self.tr("%s ...") % "converting")
        QtWidgets.QMessageBox.about(
            self,
            "Finished",
            (
                f"COCO export finished.\n\n"
                f"Output: {str(export_dir)}\n"
                f"Train split: {int(round(float(num_train_frames) * 100))}%\n"
                f"Mode: {output_mode}"
            ),
        )
        self.statusBar().showMessage(self.tr("%s Done.") % "converting")
        try:
            shutil.make_archive(
                str(export_dir),
                "zip",
                str(export_dir.parent),
                export_dir.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create COCO zip archive for %s: %s", export_dir, exc
            )

    def coco_to_labelme(self):
        """Convert a COCO annotations directory into a LabelMe dataset."""
        start_dir = str(
            self.annotation_dir or self.output_dir or self.lastOpenDir or Path.home()
        )
        dlg = ConvertCOCO2LabelMeDialog(self, start_dir=start_dir)
        if not dlg.exec_() or dlg.config is None:
            return
        cfg = dlg.config

        try:
            summary = coco2labelme.convert_coco_annotations_dir_to_labelme_dataset(
                cfg.annotations_dir,
                output_dir=cfg.output_dir,
                images_dir=cfg.images_dir,
                recursive=cfg.recursive,
                link_mode=cfg.link_mode,
                write_pose_schema=bool(getattr(cfg, "write_pose_schema", True)),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "COCO to LabelMe", f"Conversion failed:\n{exc}"
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "COCO to LabelMe",
            (
                f"Processed JSON files: {summary.get('json_files_total', 0)}\n"
                f"Converted images: {summary.get('converted_images', 0)} / "
                f"{summary.get('images_total', 0)}\n"
                f"Copied/linked images: {summary.get('copied_images', 0)}\n"
                f"Missing images: {summary.get('missing_images', 0)}\n"
                f"Total shapes: {summary.get('shapes_total', 0)}\n\n"
                f"Output dataset: {cfg.output_dir}"
            ),
        )

    def open_training_dashboard(
        self,
        *,
        initial_tab: str | None = None,
        run_dir: str | Path | None = None,
        start_tensorboard: bool = False,
        open_projector: bool = False,
    ) -> TrainingDashboardDialog:
        """Open (or raise) the unified Training Monitor & Visualization Dashboard."""
        dialog = getattr(self, "_training_dashboard_dialog", None)
        if dialog is None:
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
            dialog = TrainingDashboardDialog(settings=self.settings, parent=None)
            if yolo_manager is not None:
                dialog.dashboard.register_training_manager(yolo_manager)
            if dino_manager is not None:
                dialog.dashboard.register_training_manager(dino_manager)
            dialog.finished.connect(
                lambda *_: setattr(self, "_training_dashboard_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_training_dashboard_dialog", None)
            )
            self._training_dashboard_dialog = dialog

        try:
            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        if run_dir is not None:
            try:
                dialog.dashboard.select_run_dir(run_dir)
            except Exception:
                pass
        else:
            try:
                dialog.dashboard._refresh_runs_list()
            except Exception:
                pass

        if initial_tab:
            try:
                dialog.dashboard.select_tab(initial_tab)
            except Exception:
                pass

        if start_tensorboard or open_projector:
            try:
                dialog.dashboard.start_tensorboard_for_current_scope(
                    open_projector=bool(open_projector),
                    switch_to_tab=True,
                )
            except Exception:
                pass

        return dialog

    def visualization(self):
        """Open the unified Training Monitor & Visualization Dashboard."""
        return self.open_training_dashboard(initial_tab="monitor")

    @QtCore.Slot(object)
    def _show_training_dashboard_for_training(self, payload: object) -> None:
        """Auto-open the training dashboard window when training starts."""
        dialog = self.open_training_dashboard(initial_tab="monitor")
        try:
            dialog.dashboard._on_training_started(payload)
        except Exception:
            pass

    def open_colab_studio(
        self,
        *,
        initial_tab: str = "auth",
        training_config: dict | None = None,
        inference_config: dict | None = None,
    ):
        """Open the Google Colab Studio dialog for auth, training, and inference."""
        from annolid.gui.widgets.colab_dialog import ColabStudioDialog

        dialog = getattr(self, "_colab_studio_dialog", None)
        if dialog is None:
            dataset_path = None
            video_file = str(getattr(self, "video_file", "") or "").strip() or None
            if video_file:
                candidate = Path(video_file).with_suffix("") / "data.yaml"
                if candidate.is_file():
                    dataset_path = str(candidate)
            if dataset_path is None and getattr(self, "annotation_dir", None):
                ann_dir = Path(str(self.annotation_dir))
                if (ann_dir / "data.yaml").is_file():
                    dataset_path = str(ann_dir / "data.yaml")
                elif ann_dir.is_dir():
                    dataset_path = str(ann_dir)

            dialog = ColabStudioDialog(
                initial_dataset_path=dataset_path,
                initial_video_path=video_file,
                initial_tab=initial_tab,
                auto_refresh_account=not bool(os.environ.get("PYTEST_CURRENT_TEST")),
                parent=self,
            )
            open_dir_fn = getattr(self, "importDirImages", None)
            if callable(open_dir_fn):
                dialog.openPredictionsRequested.connect(open_dir_fn)
            if hasattr(dialog, "openTrainingDashboardRequested"):
                dialog.openTrainingDashboardRequested.connect(
                    lambda run_dir: self.open_training_dashboard(
                        initial_tab="monitor", run_dir=run_dir
                    )
                )
            dialog.finished.connect(
                lambda *_: setattr(self, "_colab_studio_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_colab_studio_dialog", None)
            )
            self._colab_studio_dialog = dialog
        else:
            dialog.select_tab(initial_tab)

        if training_config:
            dialog.prefill_training_config(dict(training_config))
        if inference_config:
            dialog.prefill_inference_config(dict(inference_config))

        dialog.show()
        dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def train_on_colab(self):
        """Open Google Colab Studio (backward-compatible entry point for menu/toolbar)."""
        return self.open_colab_studio(initial_tab="train")

    def open_sim2real_studio(self, *, project_dir: str | None = None):
        """Open Sim-to-Real Studio for example-video matching, 3D Mouse/Fly simulation, and Colab iteration."""
        from annolid.gui.widgets.sim2real_dialog import Sim2RealStudioDialog

        dialog = getattr(self, "_sim2real_studio_dialog", None)
        if dialog is None:
            video_file = str(getattr(self, "video_file", "") or "").strip() or None
            initial_videos = [video_file] if video_file else None
            colab_dlg = getattr(self, "_colab_studio_dialog", None)
            colab_svc = (
                getattr(colab_dlg, "_service", None) if colab_dlg is not None else None
            )
            dialog = Sim2RealStudioDialog(
                project_dir=project_dir,
                initial_video_paths=initial_videos,
                colab_service=colab_svc,
                parent=self,
            )
            dialog.finished.connect(
                lambda *_: setattr(self, "_sim2real_studio_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_sim2real_studio_dialog", None)
            )
            self._sim2real_studio_dialog = dialog

        dialog.show()
        dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def quality_control(self):
        from annolid.gui.widgets.quality_control_dialog import QualityControlDialog
        from annolid.postprocessing.quality_control import TracksResults

        video_file = None
        tracking_results = None
        skip_num_frames = None
        qc_dialog = QualityControlDialog()

        if qc_dialog.exec_():
            video_file = qc_dialog.video_file
            tracking_results = qc_dialog.tracking_results
            skip_num_frames = qc_dialog.skip_num_frames
        else:
            return

        if video_file is None or tracking_results is None:
            QtWidgets.QMessageBox.about(
                self,
                "No input video or tracking results",
                "Please select both a video file and tracking results first.",
            )
            return
        out_dir = (
            f"{str(Path(video_file).with_suffix(''))}{self._pred_res_folder_suffix}"
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        trs = TracksResults(video_file, tracking_results)
        label_json_gen = trs.to_labelme_json(str(out_dir), skip_frames=skip_num_frames)

        try:
            if label_json_gen is not None:
                pwj = ProgressingWindow(label_json_gen)
                pwj.exec_()
                if (
                    getattr(pwj, "runner_thread", None) is not None
                    and pwj.runner_thread.isRunning()
                ):
                    trs._is_running = False
                    pwj.running_submitted.emit("stopped")
                    pwj.runner_thread.requestInterruption()
                    pwj.runner_thread.quit()
                    pwj.runner_thread.wait(2000)
        except Exception:
            logger.debug("Quality control runner cleanup failed", exc_info=True)
        finally:
            self.importDirImages(str(out_dir))

    def glitter2(self):
        """
        overlay the predicted masks and bboxes on the inference video
        and convert the nix format for editing with glitter2 package
        https://github.com/matham/glitter2
        """
        video_file = None
        tracking_results = None
        out_nix_csv_file = None
        zone_info_json = None
        score_threshold = None
        motion_threshold = None
        pretrained_model = None
        subject_names = None

        g_dialog = Glitter2Dialog()
        if g_dialog.exec_():
            video_file = g_dialog.video_file
            tracking_results = g_dialog.tracking_results
            out_nix_csv_file = g_dialog.out_nix_csv_file
            zone_info_json = g_dialog.zone_info_json
            score_threshold = g_dialog.score_threshold
            motion_threshold = g_dialog.motion_threshold
            pretrained_model = g_dialog.pretrained_model
            subject_names = g_dialog.subject_names
            behavior_names = g_dialog.behaviors
        else:
            return

        if video_file is None or tracking_results is None:
            QtWidgets.QMessageBox.about(
                self,
                "No input video or tracking results",
                "Please select both a video file and tracking results first.",
            )
            return

        if out_nix_csv_file is None:
            out_nix_csv_file = tracking_results.replace(".csv", "_nix.csv")

        from annolid.postprocessing.glitter import tracks2nix

        tracks2nix(
            video_file,
            tracking_results,
            out_nix_csv_file,
            zone_info=zone_info_json,
            score_threshold=score_threshold,
            motion_threshold=motion_threshold,
            pretrained_model=pretrained_model,
            subject_names=subject_names,
            behavior_names=behavior_names,
        )

    def run_sam3d_reconstruction(self):
        manager = getattr(self, "sam3d_manager", None)
        if manager is None and hasattr(self, "ensure_sam3d_manager"):
            try:
                manager = self.ensure_sam3d_manager()
            except Exception:
                manager = None
        if manager is not None:
            manager.run_sam3d_reconstruction()
