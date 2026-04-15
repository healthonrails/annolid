from __future__ import annotations

import functools
import webbrowser
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation.pose_schema import PoseSchema
from annolid.domain import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    save_behavior_spec,
)
from annolid.gui.keypoint_catalog import (
    extract_labels_from_uniq_label_list,
    merge_keypoint_lists,
)
from annolid.gui.widgets import (
    BatchRelabelDialog,
    IdentityGovernorDialog,
    LabelCollectionDialog,
    LogManagerDialog,
    SystemInfoDialog,
)
from annolid.gui.widgets.tracking_stats_dashboard_dialog import (
    TrackingStatsDashboardDialog,
)
from annolid.gui.widgets.convert_deeplabcut_dialog import ConvertDLCDialog
from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog
from annolid.gui.widgets.convert_sleap_dialog import ConvertSleapDialog
from annolid.gui.widgets.downsample_videos_dialog import VideoRescaleWidget
from annolid.gui.widgets.extract_keypoints_dialog import ExtractShapeKeyPointsDialog
from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog
from annolid.gui.cursor_utils import set_widget_busy_cursor
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

    def _setup_log_manager_action(self) -> None:
        action = functools.partial(newAction, self)
        self.manage_logs_action = action(
            self.tr("Manage &Logs..."),
            self._open_log_manager_dialog,
            None,
            "open",
            self.tr("Open Annolid logs manager."),
            enabled=True,
        )
        file_menu = getattr(self.menus, "file", None)
        if file_menu is not None:
            file_menu.addAction(self.manage_logs_action)

    def _open_label_collection_dialog(self) -> None:
        dlg = LabelCollectionDialog(settings=self.settings, parent=self)
        dlg.exec_()

    def _open_log_manager_dialog(self) -> None:
        dlg = LogManagerDialog(parent=self)
        dlg.exec_()

    def _set_active_view(self, mode: str = "canvas") -> None:
        logger.debug("Switching active view to: %s", mode)
        # Toggle main toolbar visibility.
        if hasattr(self, "tools"):
            self.tools.setVisible(mode == "canvas")

        if mode == "canvas":
            self.set_unrelated_docks_visible(True)
            self._viewer_stack.setCurrentIndex(0)
            return

        stack = getattr(self, "_viewer_stack", None)
        if stack is None:
            logger.warning("Cannot switch view to %s: _viewer_stack is missing.", mode)
            return

        if mode == "pdf":
            pdf_manager = getattr(self, "pdf_manager", None)
            if pdf_manager is not None:
                self.set_unrelated_docks_visible(False)
                viewer = pdf_manager.pdf_widget()
                if viewer is not None:
                    idx = stack.indexOf(viewer)
                    if idx != -1:
                        stack.setCurrentIndex(idx)
                        return
                    logger.warning("PDF viewer widget not found in stack.")
            else:
                logger.warning("pdf_manager is missing, cannot switch to PDF view.")

        if mode == "web":
            web_manager = getattr(self, "web_manager", None)
            if web_manager is not None:
                self.set_unrelated_docks_visible(False)
                viewer = web_manager.viewer_widget()
                if viewer is not None:
                    idx = stack.indexOf(viewer)
                    if idx != -1:
                        stack.setCurrentIndex(idx)
                        return
                    logger.warning("Web viewer widget not found in stack.")

        if mode == "threejs":
            threejs_manager = getattr(self, "threejs_manager", None)
            if threejs_manager is not None:
                self.set_unrelated_docks_visible(False)
                viewer = threejs_manager.viewer_widget()
                if viewer is not None:
                    idx = stack.indexOf(viewer)
                    if idx != -1:
                        stack.setCurrentIndex(idx)
                        return
                    logger.warning("ThreeJS viewer widget not found in stack.")

        logger.warning(
            "Fallback: Switching to default canvas view (mode was: %s)", mode
        )
        stack.setCurrentIndex(0)

    def show_pdf_in_viewer(self, pdf_path: str) -> None:
        if self.pdf_manager is not None:
            self.pdf_manager.show_pdf_in_viewer(pdf_path)

    def show_web_in_viewer(self, url: str) -> bool:
        if self.web_manager is not None:
            return bool(self.web_manager.show_url_in_viewer(url))
        return False

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
        video_downsample_widget = VideoRescaleWidget(
            initial_video_path=getattr(self, "video_file", None)
        )
        video_downsample_widget.exec_()

    def open_zone_manager(self):
        dock = getattr(self, "zone_dock", None)
        if dock is None:
            return
        refresh = getattr(dock, "refresh_from_current_canvas", None)
        if callable(refresh):
            refresh()
        dock.show()
        dock.raise_()
        dock.activateWindow()

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
            set_widget_busy_cursor(self, True)

            from annolid.postprocessing.tracking_reports import (
                find_tracking_gaps,
                generate_reports,
            )

            gaps = find_tracking_gaps(video_file)
            md_filepath = generate_reports(gaps, video_file)
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
            logger.error(f"An error occurred during gap analysis: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Analysis Error", f"An unexpected error occurred:\n\n{e}"
            )
            self.statusBar().showMessage(self.tr("Gap analysis failed."), 5000)
        finally:
            set_widget_busy_cursor(self, False)

    def open_tracking_stats_dashboard(self) -> None:
        """Open cross-video tracking stats analysis/visualization dashboard."""
        try:
            dialog = getattr(self, "_tracking_stats_dashboard_dialog", None)
            if dialog is None:
                initial_root = None
                if getattr(self, "video_results_folder", None):
                    initial_root = Path(self.video_results_folder).resolve()
                elif getattr(self, "annotation_dir", None):
                    initial_root = Path(self.annotation_dir).resolve()
                elif getattr(self, "lastOpenDir", None):
                    initial_root = Path(str(self.lastOpenDir)).resolve()
                else:
                    initial_root = Path.cwd()
                dialog = TrackingStatsDashboardDialog(
                    initial_root_dir=initial_root,
                    parent=self,
                )
                dialog.finished.connect(
                    lambda *_: setattr(self, "_tracking_stats_dashboard_dialog", None)
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(self, "_tracking_stats_dashboard_dialog", None)
                )
                self._tracking_stats_dashboard_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception as exc:
            logger.debug("Could not open tracking stats dashboard: %s", exc)

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

    def batch_rename_shape_labels(self) -> None:
        start_dir = (
            getattr(self, "annotation_dir", None)
            or getattr(self, "lastOpenDir", None)
            or str(Path.cwd())
        )
        dlg = BatchRelabelDialog(initial_root=start_dir, parent=self)
        dlg.exec_()

        # If dashboard is open, refresh to reflect bulk relabel updates.
        try:
            dialog = getattr(self, "_labeling_progress_dashboard_dialog", None)
            dashboard = getattr(dialog, "dashboard", None)
            if dashboard is not None:
                dashboard.refresh_stats()
        except Exception:
            logger.debug(
                "Failed to refresh dashboard after batch relabel.", exc_info=True
            )

    def open_identity_governor_dialog(self) -> None:
        try:
            dialog = getattr(self, "_identity_governor_dialog", None)
            if dialog is None:
                initial_root = None
                if getattr(self, "video_results_folder", None):
                    initial_root = Path(self.video_results_folder).resolve()
                elif getattr(self, "annotation_dir", None):
                    initial_root = Path(self.annotation_dir).resolve()
                elif getattr(self, "lastOpenDir", None):
                    initial_root = Path(str(self.lastOpenDir)).resolve()
                else:
                    initial_root = Path.cwd()

                dialog = IdentityGovernorDialog(
                    initial_annotation_dir=initial_root,
                    parent=self,
                )
                dialog.finished.connect(
                    lambda *_: setattr(self, "_identity_governor_dialog", None)
                )
                dialog.destroyed.connect(
                    lambda *_: setattr(self, "_identity_governor_dialog", None)
                )
                self._identity_governor_dialog = dialog

            dialog.show()
            dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception as exc:
            logger.debug("Could not open identity governor dialog: %s", exc)

    def open_pose_schema_dialog(self):
        point_labels = [
            getattr(shape, "label", None)
            for shape in getattr(self.canvas, "shapes", []) or []
            if str(getattr(shape, "shape_type", "")).lower() == "point"
            and getattr(shape, "label", None)
        ]
        uniq_labels = extract_labels_from_uniq_label_list(
            getattr(self, "uniqLabelList", None)
        )
        seq_labels = []
        try:
            widget = getattr(self, "keypoint_sequence_widget", None)
            if widget is not None:
                seq_labels = list(getattr(widget, "keypoint_order", lambda: [])() or [])
        except Exception:
            seq_labels = []
        keypoints = merge_keypoint_lists(seq_labels, point_labels, uniq_labels)

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
        widget = getattr(self, "keypoint_sequence_widget", None)
        dock = getattr(self, "keypoint_sequence_dock", None)
        if widget is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Pose Manager Unavailable",
                "Keypoint sequencer dock is not available in this window.",
            )
            return

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

        schema_path = getattr(self, "_pose_schema_path", None) or default_path
        widget.set_pose_schema(schema, schema_path)
        widget.load_keypoints_from_labels(keypoints)
        if schema is None and schema_path:
            try:
                widget.load_schema_from_path(str(schema_path), quiet=True)
            except Exception:
                pass

        if dock is not None:
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass

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
        existing_dialog = getattr(self, "_zone_analysis_dialog", None)
        if isinstance(existing_dialog, TrackingAnalyzerDialog):
            existing_dialog.apply_session_context()
            existing_dialog.show()
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return

        place_preference_analyze_widget = TrackingAnalyzerDialog(
            parent=self,
            video_path=str(getattr(self, "video_file", "") or "").strip() or None,
            zone_path=str(getattr(self, "zone_path", "") or "").strip() or None,
            fps=getattr(self, "fps", None),
        )
        place_preference_analyze_widget.setModal(False)
        place_preference_analyze_widget.setWindowModality(QtCore.Qt.NonModal)
        place_preference_analyze_widget.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._zone_analysis_dialog = place_preference_analyze_widget
        place_preference_analyze_widget.destroyed.connect(
            lambda *_: setattr(self, "_zone_analysis_dialog", None)
        )
        place_preference_analyze_widget.show()
        place_preference_analyze_widget.raise_()
        place_preference_analyze_widget.activateWindow()

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
