from __future__ import annotations

import functools
import webbrowser
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.pose_schema import PoseSchema
from annolid.core.behavior.spec import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    save_behavior_spec,
)
from annolid.gui.widgets import LabelCollectionDialog, SystemInfoDialog
from annolid.gui.widgets.convert_deeplabcut_dialog import ConvertDLCDialog
from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog
from annolid.gui.widgets.convert_sleap_dialog import ConvertSleapDialog
from annolid.gui.widgets.downsample_videos_dialog import VideoRescaleWidget
from annolid.gui.widgets.extract_keypoints_dialog import ExtractShapeKeyPointsDialog
from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog
from annolid.gui.window_base import newAction
from annolid.postprocessing.video_timestamp_annotator import process_directory
from annolid.utils.logger import logger


class ToolingDialogsMixin:
    """UI actions and utility dialogs for tools, conversion, and schemas."""

    def _setup_canvas_screenshot_action(self):
        action = functools.partial(newAction, self)
        self.save_canvas_screenshot_action = action(
            self.tr("Save Canvas Image"),
            self._save_canvas_screenshot,
            "Ctrl+Shift+I",
            "Save Canvas Image",
            self.tr("Save the current canvas as a PNG image."),
            enabled=True,
        )
        self.menus.file.addAction(self.save_canvas_screenshot_action)

    def _setup_open_pdf_action(self):
        action = functools.partial(newAction, self)
        self.open_pdf_action = action(
            self.tr("Open &PDF..."),
            self.pdf_import_widget.open_pdf,
            None,
            "open",
            self.tr("Convert PDF pages to images and load them"),
            enabled=True,
        )
        file_menu = getattr(self.menus, "file", None)
        if file_menu is not None:
            actions = file_menu.actions()
            target_action = None
            for act in actions:
                text = act.text() if act is not None else ""
                if text and "Open Dir" in text:
                    target_action = act
                    break
            if target_action:
                file_menu.insertAction(target_action, self.open_pdf_action)
            else:
                file_menu.addAction(self.open_pdf_action)

    def _setup_label_collection_action(self) -> None:
        action = functools.partial(newAction, self)
        self.collect_labels_action = action(
            self.tr("Collect &Labels..."),
            self._open_label_collection_dialog,
            None,
            "open",
            self.tr("Index labeled PNG/JSON pairs into a central dataset JSONL file."),
            enabled=True,
        )
        file_menu = getattr(self.menus, "file", None)
        if file_menu is not None:
            file_menu.addAction(self.collect_labels_action)

    def _open_label_collection_dialog(self) -> None:
        dlg = LabelCollectionDialog(settings=self.settings, parent=self)
        dlg.exec_()

    def _set_active_view(self, mode: str = "canvas") -> None:
        if mode == "pdf" and getattr(self, "pdf_manager", None) is not None:
            viewer = self.pdf_manager.pdf_widget()
            if viewer is not None:
                pdf_index = self._viewer_stack.indexOf(viewer)
                if pdf_index != -1:
                    self._viewer_stack.setCurrentIndex(pdf_index)
                    return
        if mode == "threejs" and getattr(self, "threejs_manager", None) is not None:
            viewer = self.threejs_manager.viewer_widget()
            if viewer is not None:
                three_index = self._viewer_stack.indexOf(viewer)
                if three_index != -1:
                    self._viewer_stack.setCurrentIndex(three_index)
                    return
        self._viewer_stack.setCurrentIndex(0)

    def show_pdf_in_viewer(self, pdf_path: str) -> None:
        if self.pdf_manager is not None:
            self.pdf_manager.show_pdf_in_viewer(pdf_path)

    @QtCore.Slot(str)
    def _apply_pdf_selection_to_caption(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self.caption_widget is None:
            self.openCaption()
        if self.caption_widget is not None:
            self.caption_widget.set_caption(cleaned)

    def _save_canvas_screenshot(self):
        self.canvas_screenshot_widget.save_canvas_screenshot(filename=self.filename)

    def downsample_videos(self):
        video_downsample_widget = VideoRescaleWidget()
        video_downsample_widget.exec_()

    def run_optical_flow_tool(self):
        if getattr(self, "optical_flow_manager", None) is not None:
            return self.optical_flow_manager.run_tool()

    def configure_optical_flow_settings(self):
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.configure_tool()

    def trigger_gap_analysis(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Video File to Analyze"),
            self.lastOpenDir,
            self.tr("Video Files (*.mp4 *.avi *.mov *.mkv)"),
        )

        if not video_path:
            return

        video_file = Path(video_path)
        if not video_file.is_file():
            QtWidgets.QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected video file does not exist:\n{video_path}",
            )
            return

        json_dir = video_file.with_suffix("")
        if not json_dir.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Results Not Found",
                f"Could not find the associated tracking results directory:\n{json_dir}\n\n"
                "Please ensure tracking has been run for this video.",
            )
            return

        try:
            self.statusBar().showMessage(
                self.tr(f"Analyzing {video_file.name}, please wait...")
            )
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            from annolid.postprocessing.tracking_reports import (
                find_tracking_gaps,
                generate_reports,
            )

            gaps = find_tracking_gaps(video_file)
            md_filepath = generate_reports(gaps, video_file)

            QtWidgets.QApplication.restoreOverrideCursor()
            report_path = str(md_filepath)
            self.statusBar().showMessage(self.tr("Gap analysis complete."), 5000)

            reply = QtWidgets.QMessageBox.information(
                self,
                "Analysis Complete",
                f"A tracking gap report has been saved to:\n{report_path}\n\nWould you like to open it now?",
                QtWidgets.QMessageBox.Open | QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Open,
            )
            if reply == QtWidgets.QMessageBox.Open:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(report_path))

        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            logger.error(f"An error occurred during gap analysis: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Analysis Error", f"An unexpected error occurred:\n\n{e}"
            )
            self.statusBar().showMessage(self.tr("Gap analysis failed."), 5000)

    def _add_real_time_stamps(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select folder to annotate"), str(Path.home())
        )
        if not folder:
            return
        try:
            process_directory(Path(folder))
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Done"),
                f"{self.tr('All CSVs have been updated in:')}\n{folder}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Error"),
                f"{self.tr('Failed to add real-time stamps:')}\n{e}",
            )

    def convert_sleap_h5_to_labelme(self):
        convert_sleap_h5_widget = ConvertSleapDialog()
        convert_sleap_h5_widget.exec_()

    def convert_deeplabcut_csv_to_labelme(self):
        convert_deeplabcut_widget = ConvertDLCDialog()
        convert_deeplabcut_widget.exec_()

    def convert_labelme2yolo_format(self):
        from annolid.gui.widgets import convert_labelme2yolo

        convert_labelme2yolo_widget = convert_labelme2yolo.YOLOConverterWidget()
        convert_labelme2yolo_widget.exec_()

    def open_pose_schema_dialog(self):
        from annolid.gui.widgets.pose_schema_dialog import PoseSchemaDialog

        keypoints = sorted(
            {
                getattr(shape, "label", None)
                for shape in getattr(self.canvas, "shapes", []) or []
                if str(getattr(shape, "shape_type", "")).lower() == "point"
                and getattr(shape, "label", None)
            }
        )
        if not keypoints:
            try:
                keypoints = [
                    self.uniqLabelList.item(i).text().strip()
                    for i in range(self.uniqLabelList.count())
                    if self.uniqLabelList.item(i).text().strip()
                ]
            except Exception:
                keypoints = []

        start_dir = (
            str(self.video_results_folder)
            if getattr(self, "video_results_folder", None)
            else getattr(self, "outputDir", None)
            or getattr(self, "lastOpenDir", None)
            or str(Path.home())
        )

        default_path = None
        for candidate in ("pose_schema.json", "pose_schema.yaml", "pose_schema.yml"):
            maybe = Path(start_dir) / candidate
            if maybe.exists():
                default_path = str(maybe)
                break
        if default_path is None:
            default_path = str(Path(start_dir) / "pose_schema.json")

        schema = self._pose_schema
        if (
            schema is None
            and self.project_schema
            and getattr(self.project_schema, "pose_schema", None)
        ):
            try:
                schema = PoseSchema.from_dict(self.project_schema.pose_schema)  # type: ignore[arg-type]
            except Exception:
                schema = None

        dlg = PoseSchemaDialog(
            keypoints=keypoints or None,
            schema=schema,
            schema_path=default_path,
            parent=self,
        )
        if not dlg.exec_():
            return

        try:
            path = dlg.schema_path or default_path
            if not path:
                path = str(Path(start_dir) / "pose_schema.json")
            dlg.schema.save(path)
            self._pose_schema_path = path
            self._pose_schema = dlg.schema
            self._persist_pose_schema_to_project_schema(dlg.schema, path)
            try:
                self.canvas.setPoseSchema(self._pose_schema)
            except Exception:
                pass
            QtWidgets.QMessageBox.information(
                self,
                "Pose Schema Saved",
                f"Pose schema saved to:\n{path}\n\n"
                "Use this file in LabelMe→YOLO conversion to generate flip_idx.",
            )
        except Exception as exc:
            logger.error("Failed to save pose schema: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Save Failed", f"Failed to save pose schema:\n{exc}"
            )

    def _persist_pose_schema_to_project_schema(
        self, schema: PoseSchema, schema_path: str
    ) -> None:
        project_schema = self.project_schema or default_behavior_spec()
        project_path = self.project_schema_path
        if project_path is None:
            if self.video_file:
                project_path = (
                    Path(self.video_file).with_suffix("") / DEFAULT_SCHEMA_FILENAME
                )
            else:
                project_path = Path.cwd() / DEFAULT_SCHEMA_FILENAME

        try:
            project_path.parent.mkdir(parents=True, exist_ok=True)
            stored_path = schema_path
            try:
                stored_path = str(
                    Path(schema_path)
                    .resolve()
                    .relative_to(project_path.parent.resolve())
                )
            except Exception:
                stored_path = schema_path

            project_schema.pose_schema_path = stored_path
            project_schema.pose_schema = schema.to_dict()
            save_behavior_spec(project_schema, project_path)
            self.project_schema = project_schema
            self.project_schema_path = project_path
        except Exception:
            logger.debug(
                "Failed to persist pose schema into project schema.", exc_info=True
            )

    def extract_and_save_shape_keypoints(self):
        extract_shape_keypoints_dialog = ExtractShapeKeyPointsDialog()
        extract_shape_keypoints_dialog.exec_()

    def place_preference_analyze(self):
        place_preference_analyze_widget = TrackingAnalyzerDialog()
        place_preference_analyze_widget.exec_()

    def place_preference_analyze_auto(self):
        if self.video_file is not None:
            analyzer_dialog = TrackingAnalyzerDialog()
            analyzer_dialog.run_analysis_without_gui(
                self.video_file, self.zone_path, self.fps
            )

    def convert_labelme_json_to_csv(self):
        convert_labelme_json_to_csv_widget = LabelmeJsonToCsvDialog()
        convert_labelme_json_to_csv_widget.exec_()

    def about_annolid_and_system_info(self):
        about_annolid_dialog = SystemInfoDialog()
        about_annolid_dialog.exec_()

    def tutorial(self):
        url = "https://github.com/healthonrails/annolid/tree/main/docs/tutorials"
        webbrowser.open(url)
