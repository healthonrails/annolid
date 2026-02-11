from __future__ import annotations

import shutil
import webbrowser
from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.annotation import coco2labelme, labelme2coco
from annolid.gui.tensorboard import VisualizationWindow, ensure_tensorboard
from annolid.gui.widgets import ProgressingWindow, TrainingDashboardDialog
from annolid.gui.widgets import QualityControlDialog
from annolid.gui.widgets import ConvertCOODialog
from annolid.gui.widgets import ConvertCOCO2LabelMeDialog
from annolid.gui.widgets import Glitter2Dialog
from annolid.postprocessing.glitter import tracks2nix
from annolid.postprocessing.quality_control import TracksResults
from annolid.utils.runs import shared_runs_root


class WorkflowActionsMixin:
    """Dataset/training/post-processing action helpers."""

    def coco(self):
        """Convert Labelme annotations to COCO format."""
        coco_dlg = ConvertCOODialog(annotation_dir=self.annotation_dir)
        output_dir = None
        labels_file = None
        input_anno_dir = None
        num_train_frames = 0.7
        if coco_dlg.exec_():
            input_anno_dir = coco_dlg.annotation_dir
            labels_file = coco_dlg.label_list_text
            output_dir = coco_dlg.out_dir
            num_train_frames = coco_dlg.num_train_frames
        else:
            return

        if input_anno_dir is None:
            QtWidgets.QMessageBox.about(
                self,
                "No input file or directory",
                "Please check and open the                                          files or directories.",
            )
            return

        if output_dir is None:
            self.output_dir = Path(input_anno_dir).parent / (
                Path(input_anno_dir).name + "_coco_dataset"
            )
        else:
            self.output_dir = output_dir

        if labels_file is None:
            labels_file = str(self.here.parent / "annotation" / "labels_custom.txt")

        label_gen = labelme2coco.convert(
            str(input_anno_dir),
            output_annotated_dir=str(self.output_dir),
            labels_file=labels_file,
            train_valid_split=num_train_frames,
        )
        pw = ProgressingWindow(label_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        self.statusBar().showMessage(self.tr("%s ...") % "converting")
        QtWidgets.QMessageBox.about(
            self,
            "Finished",
            f"Done! Results are in folder:                                             {str(self.output_dir)}",
        )
        self.statusBar().showMessage(self.tr("%s Done.") % "converting")
        try:
            shutil.make_archive(
                str(self.output_dir),
                "zip",
                self.output_dir.parent,
                self.output_dir.stem,
            )
        except Exception:
            print("Failed to create the zip file")

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

    def visualization(self):
        try:
            process, url = ensure_tensorboard(
                log_dir=shared_runs_root(), preferred_port=6006, host="127.0.0.1"
            )
            self._tensorboard_process = process
            webbrowser.open(url)
        except Exception:
            vdlg = VisualizationWindow()
            if vdlg.exec_():
                pass

    @QtCore.Slot(object)
    def _show_training_dashboard_for_training(self, payload: object) -> None:
        """Auto-open the training dashboard window when training starts."""
        dialog = getattr(self, "_training_dashboard_dialog", None)
        if dialog is None:
            dialog = TrainingDashboardDialog(settings=self.settings, parent=None)
            dialog.dashboard.register_training_manager(self.yolo_training_manager)
            dialog.dashboard.register_training_manager(self.dino_kpseg_training_manager)
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

        try:
            dialog.dashboard._on_training_started(payload)
        except Exception:
            pass

    def train_on_colab(self):
        url = "https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb"
        webbrowser.open(url)

    def quality_control(self):
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
                "Please check and open the                                          files.",
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
                if pwj.exec_():
                    trs._is_running = False
                    pwj.running_submitted.emit("stopped")
                    pwj.runner_thread.terminate()
                    pwj.runner_thread.quit()
        except Exception:
            pass
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
                "Please check and open the                                          files.",
            )
            return

        if out_nix_csv_file is None:
            out_nix_csv_file = tracking_results.replace(".csv", "_nix.csv")

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
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager.run_sam3d_reconstruction()
