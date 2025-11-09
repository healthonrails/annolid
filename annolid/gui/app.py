# Enable CPU fallback for unsupported MPS ops
import os  # noqa
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # noqa

import csv
import re
import os.path as osp
import time
import html
import shutil
import sys
import hashlib
import json
import io
import copy
import logging
from PIL import ImageQt, Image, ImageDraw
import pandas as pd
import numpy as np
import torch
import cv2
import imgviz
from pathlib import Path
from datetime import datetime
import functools
import subprocess

from labelme.ai import MODELS
from qtpy import QtCore
from qtpy.QtCore import Qt, Slot, Signal
from qtpy import QtWidgets
from qtpy import QtGui
from labelme import PY2
from labelme import QT5
from qtpy.QtCore import QFileSystemWatcher
from pycocotools import mask as maskUtils

from annolid.gui.widgets.video_manager import VideoManagerWidget
from annolid.gui.workers import (
    FlexibleWorker,
    LoadFrameThread,
    PerceptionProcessWorker,
    RealtimeSubscriberWorker,
)
from annolid.gui.shape import Shape, MaskShape
from labelme.app import MainWindow
from labelme.utils import newAction
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import LabelListWidgetItem
from labelme import utils
from annolid.utils.logger import logger
from annolid.utils.files import count_json_files
from annolid.utils.annotation_store import AnnotationStore
from labelme.widgets import ToolBar
from annolid.gui.label_file import LabelFileError
from annolid.gui.label_file import LabelFile
from annolid.gui.widgets.canvas import Canvas
from annolid.annotation import labelme2coco
from annolid.data import videos
from annolid.behavior.project_schema import (
    DEFAULT_SCHEMA_FILENAME,
    ProjectSchema,
    default_schema,
    find_schema_near_video,
    save_schema as save_project_schema,
    load_schema as load_project_schema,
    validate_schema as validate_project_schema,
)
from annolid.behavior.time_budget import (
    compute_time_budget,
    format_category_summary,
    format_time_budget_table,
    summarize_by_category,
    TimeBudgetComputationError,
    write_time_budget_csv,
)
from annolid.gui.widgets.project_dialog import ProjectDialog
from annolid.gui.widgets.behavior_controls import BehaviorControlsWidget
from annolid.gui.widgets import ExtractFrameDialog
from annolid.gui.widgets import ConvertCOODialog
from annolid.gui.widgets import TrainModelDialog
from annolid.gui.widgets import Glitter2Dialog
from annolid.gui.widgets import QualityControlDialog
from annolid.gui.widgets import TrackDialog
from annolid.gui.widgets import SystemInfoDialog
from annolid.gui.widgets import FlagTableWidget
from annolid.postprocessing.glitter import tracks2nix
from annolid.postprocessing.quality_control import TracksResults
from annolid.gui.widgets import ProgressingWindow
import webbrowser
import atexit
from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark
from annolid.gui.widgets.step_size_widget import StepSizeWidget
from annolid.gui.widgets.downsample_videos_dialog import VideoRescaleWidget
from annolid.gui.widgets.convert_sleap_dialog import ConvertSleapDialog
from annolid.gui.widgets.convert_deeplabcut_dialog import ConvertDLCDialog
from annolid.gui.widgets.extract_keypoints_dialog import ExtractShapeKeyPointsDialog
from annolid.gui.widgets import CanvasScreenshotWidget
from annolid.gui.widgets.pdf_import_widget import PdfImportWidget
from annolid.gui.widgets import RealtimeControlWidget
from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog
from annolid.gui.widgets.youtube_dialog import YouTubeVideoDialog
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor
from annolid.annotation import labelme2csv
from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog
from annolid.data.videos import get_video_files
from annolid.data.audios import AudioLoader
from annolid.gui.widgets.caption import CaptionWidget
from annolid.gui.widgets.florence2_widget import (
    Florence2Request,
    Florence2Widget,
)
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.gui.model_manager import AIModelManager
from annolid.gui.widgets.shape_dialog import ShapePropagationDialog
from annolid.postprocessing.video_timestamp_annotator import process_directory
from annolid.gui.widgets.segment_editor import SegmentEditorDialog
from annolid.vision.florence_2 import (
    Florence2Predictor,
    Florence2Result,
    create_shapes_from_mask_dict,
    process_nth_frame_from_video,
)
import contextlib
import socket
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from annolid.jobs.tracking_jobs import TrackingSegment
from annolid.gui.dino_patch_service import (
    DinoPatchRequest,
    DinoPatchSimilarityService,
    DinoPCARequest,
    DinoPCAMapService,
)
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import DinoKeypointVideoProcessor
from annolid.gui.behavior_controller import BehaviorController, BehaviorEvent
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget
from annolid.gui.tensorboard import start_tensorboard, VisualizationWindow
from annolid.realtime.perception import Config as RealtimeConfig
from annolid.gui.yolo_training_manager import YOLOTrainingManager
from annolid.gui.cli import parse_cli
from annolid.gui.application import create_qapp
from annolid.gui.controllers import (
    DinoController,
    FlagsController,
    MenuController,
    TrackingController,
    TrackingDataController,
)


__appname__ = 'Annolid'
__version__ = "1.3.0"

LABEL_COLORMAP = imgviz.label_colormap(value=200)

PATCH_SIMILARITY_DEFAULT_MODEL = PATCH_SIMILARITY_MODELS[2].identifier


def _hex_to_rgb(color: str) -> Optional[Tuple[int, int, int]]:
    color = color.strip()
    if not color.startswith("#"):
        return None
    hex_value = color[1:]
    if len(hex_value) == 6:
        try:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return (r, g, b)
        except ValueError:
            return None
    return None


class AnnolidWindow(MainWindow):
    """Annolid Main Window based on Labelme.
    """

    live_annolid_frame_updated = Signal(
        int, str)  # For modeless dialogs if any

    def __init__(self,
                 config=None
                 ):

        self.config = config
        tracker_cfg = dict((self.config or {}).get("tracker", {}) or {})
        tracker_fields = set(CutieDinoTrackerConfig.__dataclass_fields__)
        unsupported_tracker_keys = set(tracker_cfg.keys()) - tracker_fields
        if unsupported_tracker_keys:
            logger.warning(
                "Ignoring unsupported tracker config keys: %s",
                sorted(unsupported_tracker_keys),
            )
        tracker_kwargs = {k: v for k,
                          v in tracker_cfg.items() if k in tracker_fields}
        self.tracker_runtime_config = CutieDinoTrackerConfig(**tracker_kwargs)
        super(AnnolidWindow, self).__init__(config=self.config)

        # self.flag_dock.setVisible(True)
        self.flag_widget.close()
        self.flag_widget = None
        self.label_dock.setVisible(True)
        self.shape_dock.setVisible(True)
        self.file_dock.setVisible(True)

        self.csv_thread = None
        self.csv_worker = None
        self._last_tracking_csv_path = None
        self._csv_conversion_queue = []
        self._florence_worker = None
        self._florence_thread = None
        self._florence_predictors: Dict[str,
                                        Florence2Predictor] = {}
        self._running_florence_request: Optional[Florence2Request] = None
        self.florence_widget: Optional[Florence2Widget] = None
        self.florence_dock: Optional[QtWidgets.QDockWidget] = None

        self.tracking_controller = TrackingController(self)

        # Create the Video Manager Widget
        self.video_manager_widget = VideoManagerWidget()
        self.video_manager_widget.video_selected.connect(self._load_video)
        # Connect the close video signal
        self.video_manager_widget.close_video_requested.connect(self.closeFile)
        self.video_manager_widget.output_folder_ready.connect(
            self.handle_extracted_frames)
        self.video_manager_widget.json_saved.connect(
            self.video_manager_widget.update_json_column)

        self.video_manager_widget.track_all_worker_created.connect(
            self.tracking_controller.register_track_all_worker)

        # Create the Dock Widget
        self.video_dock = QtWidgets.QDockWidget("Video List", self)
        # Set a unique objectName
        self.video_dock.setObjectName('videoListDock')
        self.video_dock.setWidget(self.video_manager_widget)
        self.video_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                                    QtWidgets.QDockWidget.DockWidgetClosable |
                                    QtWidgets.QDockWidget.DockWidgetFloatable)

        # Add the Dock Widget to the Main Window
        self.addDockWidget(Qt.RightDockWidgetArea, self.video_dock)

        self.here = Path(__file__).resolve().parent
        self.settings = QtCore.QSettings("Annolid", 'Annolid')
        self._df = None
        self._df_deeplabcut = None
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_columns = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        self._df_deeplabcut_multi_animal = False
        self._df_deeplabcut_multi_animal = False
        self.label_stats = {}
        self.shape_hash_ids = {}
        self.changed_json_stats = {}
        self._pred_res_folder_suffix = '_tracking_results_labelme'
        self.ai_model_manager = AIModelManager(
            parent=self,
            combo=self._selectAiModelComboBox,
            settings=self.settings,
            base_config=self._config,
            canvas_getter=lambda: getattr(self, "canvas", None),
        )
        self.yolo_training_manager = YOLOTrainingManager(self)
        self.frame_number = 0
        self.video_loader = None
        self.video_file = None
        self.isPlaying = False
        self.event_type = None
        self._time_stamp = ''
        self.behavior_controller = BehaviorController(self._get_rgb_by_label)
        self.project_schema: Optional[ProjectSchema] = None
        self.project_schema_path: Optional[Path] = None
        self.behavior_controller.configure_from_schema(self.project_schema)
        self.annotation_dir = None
        self.step_size = 5
        self.stepSizeWidget = StepSizeWidget(5)
        self.prev_shapes = None
        self.pred_worker = None
        self.video_processor = None
        self._active_subject_name: Optional[str] = None
        self._behavior_modifier_state: Dict[str, Set[str]] = {}
        self.realtime_perception_worker = None
        self.realtime_subscriber_worker = None
        self.realtime_running = False
        self._realtime_connect_address = None
        self._realtime_shapes = []
        self.realtime_log_enabled = False
        self.realtime_log_fp = None
        self.realtime_log_path = None
        self.zone_path = None
        # Initialize a flag to control thread termination
        self.stop_prediction_flag = False
        self.epsilon_for_polygon = 2.0
        self.automatic_pause_enabled = False
        self.t_max_value = 5
        self.use_cpu_only = False
        self.save_video_with_color_mask = False
        self.auto_recovery_missing_instances = False
        self.compute_optical_flow = True
        self.playButton = None
        self.saveButton = None
        # Create progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._current_video_defined_segments: List[TrackingSegment] = []
        self.menu_controller = MenuController(self)
        self.menu_controller.setup()

        self.prediction_progress_watcher = None
        self.last_known_predicted_frame = -1  # Track the latest frame seen
        self.prediction_start_timestamp = 0.0
        self._prediction_progress_mark = None

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            sam=self._config["sam"]
        )
        self.canvas.zoomRequest.connect(self.zoomRequest)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.canvas)
        scrollArea.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scrollArea.verticalScrollBar(),
            Qt.Horizontal: scrollArea.horizontalScrollBar(),
        }

        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)

        self.flag_widget = FlagTableWidget()
        self.flag_dock.setWidget(self.flag_widget)
        self.flags_controller = FlagsController(
            window=self,
            widget=self.flag_widget,
            config_path=self.here.parent.resolve() / 'configs' / 'behaviors.yaml',
        )
        self.flags_controller.initialize()

        self.dino_controller = DinoController(self)
        self.dino_controller.initialize()

        self.tracking_data_controller = TrackingDataController(self)

        # Behavior event log dock
        self.behavior_log_widget = BehaviorEventLogWidget(self)
        self.behavior_log_widget.jumpToFrame.connect(
            self._jump_to_frame_from_log)
        self.behavior_log_widget.undoRequested.connect(
            self.undo_last_behavior_event)
        self.behavior_log_widget.clearRequested.connect(
            self._clear_behavior_events_from_log)

        self.behavior_log_dock = QtWidgets.QDockWidget("Behavior Log", self)
        self.behavior_log_dock.setObjectName('behaviorLogDock')
        self.behavior_log_dock.setWidget(self.behavior_log_widget)
        self.behavior_log_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                                           QtWidgets.QDockWidget.DockWidgetClosable |
                                           QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.behavior_log_dock)

        self.behavior_controls_widget = BehaviorControlsWidget(self)
        self.behavior_controls_widget.subjectChanged.connect(
            self._on_active_subject_changed)
        self.behavior_controls_widget.modifierToggled.connect(
            self._on_modifier_toggled)
        self.behavior_controls_dock = QtWidgets.QDockWidget(
            "Behavior Controls", self)
        self.behavior_controls_dock.setObjectName('behaviorControlsDock')
        self.behavior_controls_dock.setWidget(self.behavior_controls_widget)
        self.behavior_controls_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea,
                           self.behavior_controls_dock)
        self.tabifyDockWidget(
            self.behavior_log_dock, self.behavior_controls_dock)

        self.realtime_control_dialog = QtWidgets.QDialog(self)
        self.realtime_control_dialog.setWindowTitle(
            self.tr("Realtime Control"))
        self.realtime_control_dialog.setModal(False)
        self.realtime_control_widget = RealtimeControlWidget(
            parent=self.realtime_control_dialog,
            config=self._config,
        )
        self.realtime_control_widget.start_requested.connect(
            self._handle_realtime_start_request)
        self.realtime_control_widget.stop_requested.connect(
            self.stop_realtime_inference)
        dialog_layout = QtWidgets.QVBoxLayout(self.realtime_control_dialog)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        dialog_layout.addWidget(self.realtime_control_widget)
        self.realtime_control_dialog.resize(420, 560)
        self.realtime_control_widget.set_status_text(self.tr("Realtime idle."))

        self.setCentralWidget(scrollArea)

        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        # Restore application settings.
        self.recentFiles = self.settings.value("recentFiles", []) or []
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.move(position)

        self.video_results_folder = None
        self.seekbar = None
        self.audio_widget = None
        self.audio_dock = None
        self._audio_loader: Optional[AudioLoader] = None
        self._suppress_audio_seek = False
        self.caption_widget = None

        self.frame_worker = QtCore.QThread()
        self.frame_loader = LoadFrameThread()
        self.seg_pred_thread = QtCore.QThread()
        self.seg_train_thread = QtCore.QThread()
        self.destroyed.connect(self.clean_up)
        self.stepSizeWidget.valueChanged.connect(self.update_step_size)
        self.stepSizeWidget.predict_button.pressed.connect(
            self.predict_from_next_frame)
        atexit.register(self.clean_up)
        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.ai_model_manager.initialize()

        self.canvas_screenshot_widget = CanvasScreenshotWidget(
            canvas=self.canvas, here=Path(__file__).resolve().parent)
        self.pdf_import_widget = PdfImportWidget(self)
        self._setup_canvas_screenshot_action()
        self._setup_open_pdf_action()

        self.populateModeActions()

    @Slot()
    def _open_segment_editor_dialog(self):  # Largely the same
        if not self.video_file or self.fps is None or self.num_frames is None:
            QtWidgets.QMessageBox.information(
                self, "No Video Loaded", "Please load a video first.")
            return

        initial_segment_dicts = [s.to_dict()
                                 for s in self._current_video_defined_segments]

        dialog = SegmentEditorDialog(
            active_video_path=Path(self.video_file), active_video_fps=self.fps,
            active_video_total_frames=self.num_frames, current_annolid_frame=self.frame_number,
            initial_segments_data=initial_segment_dicts,
            annolid_config=self.config,  # Pass Annolid's main config to dialog
            parent=self
        )

        # NEW: Connect to the dialog's signal that provides the worker instance
        dialog.tracking_initiated.connect(
            self.tracking_controller.start_tracking)

        # Optional: For modeless live updates (if SegmentEditorDialog becomes modeless)
        # self.live_annolid_frame_updated.connect(dialog.update_live_annolid_frame_info)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:  # User clicked "OK"
            self._current_video_defined_segments = dialog.get_defined_segments()
            logger.info(
                f"Segment Editor OK. {len(self._current_video_defined_segments)} segments stored.")
            self._save_segments_for_active_video()  # Persist
            if self.caption_widget is not None:
                self.caption_widget.set_video_segments(
                    self._current_video_defined_segments
                )
        else:  # User clicked "Cancel" or closed dialog
            logger.info("Segment Editor Cancelled/Closed.")

        dialog.deleteLater()

    def is_tracking_busy(self) -> bool:
        return self.tracking_controller.is_tracking_busy()

    def _setup_canvas_screenshot_action(self):
        """Sets up the 'Save Canvas Image' action."""
        action = functools.partial(newAction, self)
        self.save_canvas_screenshot_action = action(
            self.tr("Save Canvas Image"),
            self._save_canvas_screenshot,
            'Ctrl+Shift+I',  # Shortcut
            "Save Canvas Image",
            self.tr("Save the current canvas as a PNG image."),
            enabled=True
        )
        self.menus.file.addAction(self.save_canvas_screenshot_action)

    def _setup_open_pdf_action(self):
        """Adds an 'Open PDF' entry to the File menu."""
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
            file_menu.addAction(self.open_pdf_action)

    def _save_canvas_screenshot(self):
        """ Calls CanvasScreenshotWidget and passes in the current filename"""
        self.canvas_screenshot_widget.save_canvas_screenshot(
            filename=self.filename)

    def _grounding_sam(self):
        """
        Handles the text prompt inputs for grouding DINO SAM. 
        The function:
        1. Toggles the drawing mode to 'grounding_sam'.
        2. Processes the text prompt to determine if it contains flags or other commands.
        3. Loads flags if specified in the prompt.
        4. Initiates an AI-based prediction for the rectangle on the canvas if no flags are found.

        The expected format for flags in the prompt is 'flags:flag1,flag2,...'.
        If the prompt is empty or invalid, an informational log is recorded.

        Returns:
            None
        """
        self.toggleDrawMode(False, createMode="grounding_sam")
        prompt_text = self.aiRectangle._aiRectanglePrompt.text().lower()

        # Check if the prompt text is empty or too short
        if len(prompt_text) < 1:
            logger.info(f"Invalid text prompt '{prompt_text}'")
            return

        # Check if the prompt starts with 'flags:' and contains flags separated by commas
        if prompt_text.startswith('flags:'):
            flags = {k.strip(): False for k in prompt_text.replace(
                'flags:', '').split(',') if len(k.strip()) > 0}
            if len(flags.keys()) > 0:
                self.flags_controller.apply_prompt_flags(flags)
            else:
                self.flags_controller.clear_flags()
        else:
            self.canvas.predictAiRectangle(prompt_text)

    def update_step_size(self, value):
        self.step_size = value
        self.stepSizeWidget.set_value(self.step_size)

    def downsample_videos(self):
        video_downsample_widget = VideoRescaleWidget()
        video_downsample_widget.exec_()

    def trigger_gap_analysis(self):
        """
        Prompts the user to select a video file, then runs the gap analysis
        synchronously and asks to open the generated report.
        """
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Video File to Analyze"),
            self.lastOpenDir,  # Start in the last used directory
            self.tr("Video Files (*.mp4 *.avi *.mov *.mkv)")
        )

        if not video_path:
            # User canceled the dialog
            return

        video_file = Path(video_path)
        if not video_file.is_file():
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", f"The selected video file does not exist:\n{video_path}")
            return

        json_dir = video_file.with_suffix('')
        if not json_dir.is_dir():
            QtWidgets.QMessageBox.warning(self, "Results Not Found",
                                          f"Could not find the associated tracking results directory:\n{json_dir}\n\n"
                                          "Please ensure tracking has been run for this video.")
            return

        # --- Execute the analysis directly ---
        try:
            # 1. Provide feedback to the user that something is happening
            self.statusBar().showMessage(
                self.tr(f"Analyzing {video_file.name}, please wait..."))
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

            # 2. Import and run the core logic
            from annolid.postprocessing.tracking_reports import find_tracking_gaps, generate_reports

            gaps = find_tracking_gaps(video_file)
            # This function now also saves the files
            md_filepath = generate_reports(gaps, video_file)

            # 3. Restore cursor and show completion message
            QtWidgets.QApplication.restoreOverrideCursor()
            report_path = str(md_filepath)

            self.statusBar().showMessage(self.tr("Gap analysis complete."), 5000)

            reply = QtWidgets.QMessageBox.information(
                self,
                "Analysis Complete",
                f"A tracking gap report has been saved to:\n{report_path}\n\nWould you like to open it now?",
                QtWidgets.QMessageBox.Open | QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Open
            )
            if reply == QtWidgets.QMessageBox.Open:
                QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl.fromLocalFile(report_path))

        except Exception as e:
            # Catch any errors from the analysis and report them gracefully
            QtWidgets.QApplication.restoreOverrideCursor()
            logger.error(
                f"An error occurred during gap analysis: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(self, "Analysis Error",
                                           f"An unexpected error occurred:\n\n{e}")
            self.statusBar().showMessage(self.tr("Gap analysis failed."), 5000)

    def _add_real_time_stamps(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select folder to annotate"),
            str(Path.home())
        )
        if not folder:
            return
        try:
            process_directory(Path(folder))
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Done"),
                f"{self.tr('All CSVs have been updated in:')}\n{folder}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Error"),
                f"{self.tr('Failed to add real-time stamps:')}\n{e}"
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
                self.video_file, self.zone_path, self.fps)

    def convert_labelme_json_to_csv(self):
        convert_labelme_json_to_csv_widget = LabelmeJsonToCsvDialog()
        convert_labelme_json_to_csv_widget.exec_()

    def about_annolid_and_system_info(self):
        about_annolid_dialog = SystemInfoDialog()
        about_annolid_dialog.exec_()

    def openAudio(self):
        from annolid.gui.widgets.audio import AudioWidget
        if not self.video_file:
            return

        if self._audio_loader is None:
            self._configure_audio_for_video(self.video_file, self.fps)

        if self._audio_loader is None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Audio"),
                self.tr("No audio track available for this video."),
            )
            return

        self.audio_widget = AudioWidget(
            self.video_file, audio_loader=self._audio_loader
        )
        self.audio_dock = QtWidgets.QDockWidget(self.tr("Audio"), self)
        self.audio_dock.setObjectName("Audio")
        self.audio_dock.setWidget(self.audio_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.audio_dock)

    def _configure_audio_for_video(
        self, video_path: Optional[str], fps: Optional[float]
    ) -> None:
        """Prepare audio playback for the active video if an audio track exists."""
        self._release_audio_loader()

        if not video_path:
            return

        effective_fps = fps if fps and fps > 0 else 29.97
        try:
            self._audio_loader = AudioLoader(video_path, fps=effective_fps)
        except Exception as exc:
            logger.debug(
                "Skipping audio playback for %s: %s",
                video_path,
                exc,
            )
            self._audio_loader = None

    def _release_audio_loader(self) -> None:
        """Stop and discard any cached audio loader."""
        if self._audio_loader is None:
            return

        with contextlib.suppress(Exception):
            self._audio_loader.stop()
        self._audio_loader = None

    def _active_audio_loader(self) -> Optional[AudioLoader]:
        """Return the audio loader currently associated with playback."""
        if self.audio_widget and self.audio_widget.audio_loader:
            return self.audio_widget.audio_loader
        return self._audio_loader

    def _update_audio_playhead(self, frame_number: int) -> None:
        """Align cached audio playback position with the given frame number."""
        audio_loader = self._active_audio_loader()
        if not audio_loader:
            return

        set_playhead = getattr(audio_loader, "set_playhead_frame", None)
        if callable(set_playhead):
            try:
                set_playhead(frame_number)
            except Exception as exc:
                logger.debug(
                    "Failed to align audio playhead for frame %s: %s",
                    frame_number,
                    exc,
                )
            return

        frame_to_sample = getattr(audio_loader, "_frame_to_sample_index", None)
        if callable(frame_to_sample) and hasattr(audio_loader, "_playhead_sample"):
            try:
                audio_loader._playhead_sample = frame_to_sample(frame_number)
            except Exception as exc:
                logger.debug(
                    "Failed fallback audio playhead update for frame %s: %s",
                    frame_number,
                    exc,
                )

    def openCaption(self):
        # Caption dock (created but initially hidden)
        self.caption_dock = QtWidgets.QDockWidget(self.tr("Caption"), self)
        self.caption_dock.setObjectName("Caption")
        self.caption_widget = CaptionWidget()
        self.caption_widget.set_canvas(self.canvas)
        self.caption_dock.setWidget(self.caption_widget)
        self.caption_dock.installEventFilter(self.caption_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.caption_dock)

        self.caption_widget.charInserted.connect(
            self.setDirty)      # Mark as dirty
        self.caption_widget.charDeleted.connect(
            self.setDirty)      # Mark as dirty
        self.caption_widget.captionChanged.connect(
            self.canvas.setCaption)  # Update canvas
        self.caption_widget.imageGenerated.connect(
            self.display_generated_image)

    def openFlorence2(self):
        """Open or show the Florence-2 dock widget."""
        if self.florence_dock is not None:
            if self.florence_dock.isHidden():
                self.florence_dock.show()
            self.florence_dock.raise_()
            return

        self.florence_widget = Florence2Widget(self)
        self.florence_widget.runFrameRequested.connect(
            self._handle_florence_frame_request)
        self.florence_widget.runVideoRequested.connect(
            self._handle_florence_video_request)

        dock = QtWidgets.QDockWidget(self.tr("Florence-2"), self)
        dock.setObjectName("Florence2Dock")
        dock.setWidget(self.florence_widget)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        dock.destroyed.connect(lambda *_: setattr(self, "florence_dock", None))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.florence_dock = dock

    @QtCore.Slot(str)
    def display_generated_image(self, image_path: str) -> None:
        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Image generation failed"),
                self.tr("Could not load generated image:\n%s") % image_path,
            )
            return

        self.canvas.loadPixmap(pixmap, clear_shapes=True)
        try:
            self.imageData = imgviz.io.imread(image_path)
        except Exception:
            self.imageData = None

        self.imagePath = image_path
        self.filename = os.path.basename(image_path)
        self.statusBar().showMessage(
            self.tr("Generated image loaded: %s") % image_path
        )

    def _get_florence_predictor(self, model_name: str) -> Florence2Predictor:
        predictor = self._florence_predictors.get(model_name)
        if predictor is None:
            predictor = Florence2Predictor(model_name=model_name)
            self._florence_predictors[model_name] = predictor
        return predictor

    def _handle_florence_frame_request(self, request: Florence2Request) -> None:
        image = self._get_pil_image_from_state()
        if image is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No frame available"),
                self.tr("Load a frame before running Florence-2."),
            )
            return
        self._start_florence_job(request, image=image)

    def _handle_florence_video_request(self, request: Florence2Request) -> None:
        if not self.video_file:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("No video loaded"),
                self.tr("Open a video before processing it with Florence-2."),
            )
            return
        self._start_florence_job(request, video_path=self.video_file)

    def _start_florence_job(
        self,
        request: Florence2Request,
        *,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
    ) -> None:
        if self._florence_thread and self._florence_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Florence-2 busy"),
                self.tr("Please wait for the current Florence-2 task to finish."),
            )
            return

        predictor = self._get_florence_predictor(request.model_name)
        self._florence_worker = FlexibleWorker(
            task_function=self._execute_florence_job,
            predictor=predictor,
            request=request,
            image=image.copy() if image is not None else None,
            video_path=video_path,
        )
        self._florence_thread = QtCore.QThread()
        self._florence_worker.moveToThread(self._florence_thread)
        self._florence_worker.start_signal.connect(self._florence_worker.run)
        self._florence_worker.result_signal.connect(
            self._handle_florence_result)
        self._florence_worker.finished_signal.connect(
            self._handle_florence_finished)
        self._florence_worker.finished_signal.connect(
            self._florence_thread.quit)
        self._florence_worker.finished_signal.connect(
            self._florence_worker.deleteLater)
        self._florence_thread.finished.connect(self._clear_florence_worker)
        self._florence_thread.finished.connect(
            self._florence_thread.deleteLater)

        self._running_florence_request = request
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage(self.tr("Running Florence-2…"))
        self._florence_thread.start()
        QtCore.QTimer.singleShot(
            0, lambda: self._florence_worker.start_signal.emit())

    @staticmethod
    def _execute_florence_job(
        *,
        predictor: Florence2Predictor,
        request: Florence2Request,
        image: Optional[Image.Image] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[Florence2Request, Optional[Florence2Result]]:
        if request.target == "frame":
            if image is None:
                raise ValueError(
                    "Florence-2 frame request missing image data.")
            result = predictor.predict(
                image,
                text_input=request.text_input,
                segmentation_task=request.segmentation_task,
                include_caption=request.include_caption,
                caption_task=request.caption_task,
            )
            return request, result

        if video_path is None:
            raise ValueError(
                "Florence-2 video request requires an open video file.")

        process_nth_frame_from_video(
            video_path,
            request.every_n or 1,
            predictor,
            segmentation_task=request.segmentation_task,
            text_input=request.text_input,
            caption_task=request.caption_task,
            description=request.description,
        )
        return request, None

    def _handle_florence_result(
        self,
        payload: Tuple[Florence2Request, Optional[Florence2Result]],
    ) -> None:
        request, result = payload
        if request.target != "frame" or not isinstance(result, Florence2Result):
            return

        shapes = create_shapes_from_mask_dict(
            result.mask_dict, description=request.description
        )

        if request.replace_existing:
            preserved_shapes = [
                shape.copy()
                for shape in self.canvas.shapes
                if getattr(shape, "description", None)
                != request.description
            ]
            shapes_to_load = preserved_shapes + shapes
            self.loadShapes(shapes_to_load, replace=True)
        else:
            self.loadShapes(shapes, replace=False)

        if shapes:
            self.setDirty()
            self.statusBar().showMessage(
                self.tr("Florence-2 added %d shape(s).") % len(shapes),
                5000,
            )
        elif not result.caption:
            self.statusBar().showMessage(
                self.tr("Florence-2 did not return any shapes."), 5000
            )

        if result.caption:
            self._apply_florence_caption(result.caption)
            if not shapes:
                self.statusBar().showMessage(
                    self.tr("Florence-2 caption updated."), 5000
                )

    def _handle_florence_finished(self, outcome: object) -> None:
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

        request = self._running_florence_request
        self._running_florence_request = None

        if isinstance(outcome, Exception):
            logger.error("Florence-2 job failed: %s", outcome, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Florence-2 Error"),
                self.tr("Failed to run Florence-2:\n%s") % str(outcome),
            )
            self.statusBar().showMessage(self.tr("Florence-2 failed."), 5000)
            return

        if request and request.target == "video":
            self.statusBar().showMessage(
                self.tr("Florence-2 video processing complete."), 5000
            )
        elif request and request.target == "frame":
            # Success message already handled when shapes were added.
            pass
        else:
            self.statusBar().showMessage(
                self.tr("Florence-2 finished."), 3000)

    def _clear_florence_worker(self) -> None:
        self._florence_thread = None
        self._florence_worker = None

    def _apply_florence_caption(self, caption: str) -> None:
        if not caption:
            return

        self.canvas.setCaption(caption)
        if self.caption_widget is None:
            self.openCaption()
        if self.caption_widget is not None:
            self.caption_widget.set_caption(caption)
            if self.filename:
                self.caption_widget.set_image_path(self.filename)
        self.setDirty()

    def set_advanced_params(self):
        advanced_params_dialog = AdvancedParametersDialog(
            self,
            tracker_config=self.tracker_runtime_config,
        )
        if advanced_params_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.epsilon_for_polygon = advanced_params_dialog.get_epsilon_value()
        self.automatic_pause_enabled = advanced_params_dialog.is_automatic_pause_enabled()
        self.t_max_value = advanced_params_dialog.get_t_max_value()
        self.use_cpu_only = advanced_params_dialog.is_cpu_only_enabled()
        self.save_video_with_color_mask = advanced_params_dialog.is_save_video_with_color_mask_enabled()
        self.auto_recovery_missing_instances = advanced_params_dialog.is_auto_recovery_missing_instances_enabled()
        self.compute_optical_flow = advanced_params_dialog.is_compute_optiocal_flow_enabled()

        tracker_settings = advanced_params_dialog.get_tracker_settings()
        for key, value in tracker_settings.items():
            setattr(self.tracker_runtime_config, key, value)

        logger.info("Computing optical flow is %s .",
                    self.compute_optical_flow)
        logger.info("Set epsilon for polygon to : %s",
                    self.epsilon_for_polygon)

    def segmentAnything(self,):
        try:
            self.toggleDrawMode(False, createMode="polygonSAM")
            self.canvas.loadSamPredictor()
        except ImportError as e:
            print(e)

    def populateModeActions(self):
        tool, menu = self.actions.tool, self.actions.menu
        self.tools.clear()
        utils.addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        utils.addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            self.actions.editMode,
        )
        utils.addActions(self.menus.edit, actions + self.actions.editMenu)

    def scrollRequest(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        bar = self.scrollBars[orientation]
        value = bar.value() + bar.singleStep() * units
        self.setScroll(orientation, value)

    def resetState(self):
        self.labelList.clear()
        self.filename = None
        self.imagePath = None
        self.imageData = None
        self.labelFile = None
        self.otherData = None
        self.canvas.resetState()

    def playVideo(self, isPlaying=False):
        self.isPlaying = isPlaying
        if self.video_loader is None:
            return
        if self.isPlaying:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.openNextImg)
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.play(start_frame=self.frame_number)
            if self.fps is not None and self.fps > 0:
                self.timer.start(int(1000/self.fps))
            else:
                # 10 to 50 milliseconds are normal real time
                # playback
                self.timer.start(20)
        else:
            self.timer.stop()
            # Stop audio playback when video playback stops
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.stop()

    def startPlaying(self):
        self.playVideo(isPlaying=True)

    def stopPlaying(self):
        self.playVideo(isPlaying=False)

    def toggleDrawMode(self, edit=True, createMode="polygon"):

        draw_actions = {
            "polygon": self.actions.createMode,
            "rectangle": self.actions.createRectangleMode,
            "circle": self.actions.createCircleMode,
            "point": self.actions.createPointMode,
            "line": self.actions.createLineMode,
            "linestrip": self.actions.createLineStripMode,
            "ai_polygon": self.actions.createAiPolygonMode,
            "ai_mask": self.actions.createAiMaskMode,
            "polygonSAM": self.createPolygonSAMMode,
            "grouding_sam": self.createGroundingSAMMode,
        }

        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            for draw_action in draw_actions.values():
                draw_action.setEnabled(True)
        else:
            for draw_mode, draw_action in draw_actions.items():
                draw_action.setEnabled(createMode != draw_mode)
        self.actions.editMode.setEnabled(not edit)

    def closeFile(self, _value=False, *, suppress_tracking_prompt=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.dino_controller.deactivate_patch_similarity()
        self.dino_controller.deactivate_pca_map()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)
        self.uniqLabelList.clear()
        # clear the file list
        self.fileListWidget.clear()
        # if self.video_loader is not None:
        self.video_loader = None
        self.num_frames = None
        self.video_file = None
        if self.caption_widget is not None:
            self.caption_widget.set_video_context(None, None, None)
            self.caption_widget.set_video_segments([])
        self._release_audio_loader()
        if self.audio_widget:
            self.audio_widget.set_audio_loader(None)
            self.audio_widget.close()
        self.audio_widget = None
        if self.audio_dock:
            self.audio_dock.close()
        self.audio_dock = None
        self.annotation_dir = None
        if self.seekbar is not None:
            self.statusBar().removeWidget(self.seekbar)
            if self.saveButton is not None:
                self.statusBar().removeWidget(self.saveButton)
            if self.playButton is not None:
                self.statusBar().removeWidget(self.playButton)
            self.behavior_controller.attach_slider(None)
            self.seekbar = None
        self._df = None
        self._df_deeplabcut = None
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_columns = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        self.label_stats = {}
        self.shape_hash_ids = {}
        self.changed_json_stats = {}
        self._pred_res_folder_suffix = '_tracking_results_labelme'
        self.frame_number = 0
        self.step_size = 5
        self.video_results_folder = None
        self.behavior_controller.clear()
        self.behavior_log_widget.clear()
        self.isPlaying = False
        self._time_stamp = ''
        self.saveButton = None
        self.playButton = None
        self.timer = None
        self.filename = None
        self.canvas.pixmap = None
        self.event_type = None
        self.stepSizeWidget = StepSizeWidget()
        self.prev_shapes = None
        self.pred_worker = None
        self.stop_prediction_flag = False
        self.imageData = None
        self._behavior_modifier_state.clear()
        self._active_subject_name = None
        if hasattr(self, "behavior_controls_widget"):
            self.behavior_controls_widget.set_modifier_states(
                [],
                allowed=self._modifier_ids_from_schema(),
            )
            self.behavior_controls_widget.set_category_badge(None, None)
            self.behavior_controls_widget.show_warning(None)
        # 3D viewer menu remains enabled; no action needed here
        self._stop_frame_loader()
        self.frame_loader = LoadFrameThread()
        if self.video_processor is not None and hasattr(self.video_processor, "cutie_processor"):
            self.video_processor.cutie_processor = None
        self.video_processor = None
        self.fps = None
        self.only_json_files = False
        self._stop_prediction_folder_watcher()
        # Clear "predicted" marks from the slider when file is closed
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")

        if self.tracking_controller.is_tracking_busy():
            if suppress_tracking_prompt or self.tracking_controller.is_track_all_running():
                logger.info(
                    "Skipping tracking stop prompt while batch processing is active.")
            else:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Tracking in Progress",
                    "Stop tracking and close video?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self.tracking_controller.stop_active_worker()
                    # It's better to wait for worker's finished signal before truly closing,
                    # but for simplicity here, we'll proceed.
                else:
                    return  # Don't close

        # if self.video_file and self._current_video_defined_segments: # Auto-save on close?
        #     self._save_segments_for_active_video()

        super().closeFile(_value)  # Call parent's closeFile

        self.open_segment_editor_action.setEnabled(False)
        self._current_video_defined_segments = []
        logger.info("File closed in AnnolidWindow.")

    def _update_frame_display_and_emit_update(self):
        self._emit_live_frame_update()

    # ------------------------------------------------------------------
    # Realtime inference helpers
    # ------------------------------------------------------------------

    def _handle_realtime_start_request(self,
                                       realtime_config: RealtimeConfig,
                                       extras: Dict[str, Any]):
        self._show_realtime_control_dialog()
        if self.realtime_perception_worker is not None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Realtime Inference"),
                self.tr("A realtime session is already running."),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.tr("Realtime session already running."))
            return

        # Prevent duplicate publisher binding by probing the address first.
        publisher = realtime_config.publisher_address
        if publisher:
            try:
                with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    sock.settimeout(0.5)
                    host, port = self._resolve_tcp_endpoint(publisher)
                    bind_result = sock.connect_ex((host, port))
                    if bind_result == 0:
                        raise RuntimeError(
                            self.tr("Publisher port %1 is already in use.").replace("%1", str(port)))
            except RuntimeError:
                message = self.tr(
                    "Publisher address %1 is already in use.").replace("%1", publisher)
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Realtime Inference"),
                    message,
                )
                self.realtime_control_widget.set_running(False)
                self.realtime_control_widget.set_status_text(message)
                return
            except Exception:
                # If we cannot determine the state we proceed; ZMQ will raise if needed.
                pass

        try:
            self.start_realtime_inference(realtime_config, extras)
        except Exception as exc:
            logger.error("Failed to start realtime inference: %s", exc,
                         exc_info=True)
            self.realtime_running = False
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Realtime Inference"),
                self.tr("Unable to start realtime inference: %s") % str(exc),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.tr("Failed to start realtime inference."))

    def _show_realtime_control_dialog(self):
        self.realtime_control_dialog.show()
        self.realtime_control_dialog.raise_()
        self.realtime_control_dialog.activateWindow()

    def _prepare_realtime_log_path(self, requested_path: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if requested_path:
            path = Path(requested_path).expanduser()
            if path.is_dir() or not path.suffix:
                path = path / f"realtime_{timestamp}.ndjson"
        else:
            path = Path.home() / "annolid_realtime_logs" / \
                f"realtime_{timestamp}.ndjson"
        return path.resolve()

    def _resolve_tcp_endpoint(self, address: str) -> Tuple[str, int]:
        if not address.startswith("tcp://"):
            raise ValueError(f"Unsupported address format: {address}")
        host_port = address[len("tcp://"):].strip()
        if host_port.startswith("*:"):
            host = "127.0.0.1"
            port_part = host_port[2:]
        elif host_port.count(":") >= 1:
            host, port_part = host_port.rsplit(":", 1)
            if host in ("*", "0.0.0.0"):
                host = "127.0.0.1"
        else:
            raise ValueError(f"Invalid tcp address: {address}")
        port = int(port_part)
        return host, port

    def _decode_mask(self, mask_data, width: int, height: int):
        if not mask_data:
            return None

        encoding = (mask_data.get("encoding") or "").lower()

        try:
            if encoding in {"coco_rle", "rle"}:
                counts = mask_data.get("counts")
                if counts is None:
                    return None
                rle = {
                    "size": mask_data.get("size") or [height, width],
                    "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
                }
                mask = maskUtils.decode(rle)
                if mask is None:
                    return None
                if mask.shape[1] != width or mask.shape[0] != height:
                    mask = cv2.resize(mask.astype(np.uint8),
                                      (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                return mask.astype(bool)

            if encoding == "polygon":
                points = np.array(mask_data.get("points")
                                  or [], dtype=np.float32)
                if points.size == 0:
                    return None
                pts = points.copy()
                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                return mask.astype(bool)

            if encoding == "bitmap":
                data = mask_data.get("data")
                if isinstance(data, list):
                    arr = np.array(data, dtype=np.uint8)
                    if arr.size == width * height:
                        return arr.reshape((height, width)).astype(bool)

        except Exception as exc:
            logger.debug("Failed to decode realtime mask: %s",
                         exc, exc_info=True)

        return None

    def start_realtime_inference(self,
                                 realtime_config: RealtimeConfig,
                                 extras: Dict[str, Any]):
        self.realtime_control_widget.set_running(True)
        self._realtime_connect_address = extras.get(
            "subscriber_address", "tcp://127.0.0.1:5555")

        self.realtime_running = True
        self._realtime_shapes = []
        self.realtime_log_fp = None
        self.realtime_log_path = None
        self.realtime_log_enabled = bool(extras.get("log_enabled", False))

        status_message = self.tr("Realtime inference starting with %s") \
            % realtime_config.model_base_name

        if self.realtime_log_enabled:
            try:
                log_path = self._prepare_realtime_log_path(
                    extras.get("log_path", ""))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.realtime_log_fp = open(log_path, "a", encoding="utf-8")
                self.realtime_log_path = log_path
                logger.info(
                    "Realtime detections will be logged to %s", log_path)
                status_message += f" (logging to {log_path})"
            except Exception as exc:
                logger.error("Failed to open realtime NDJSON log: %s", exc,
                             exc_info=True)
                self.realtime_log_fp = None
                self.realtime_log_path = None
                self.realtime_log_enabled = False
                status_message += self.tr(" (logging disabled)")
        else:
            self.realtime_log_fp = None
            self.realtime_log_path = None

        self.statusBar().showMessage(status_message)
        self.realtime_control_widget.set_status_text(status_message)

        self.realtime_perception_worker = PerceptionProcessWorker(
            config=realtime_config,
            parent=self,
        )
        self.realtime_perception_worker.error.connect(
            self._on_realtime_error)
        self.realtime_perception_worker.stopped.connect(
            self._on_realtime_stopped)
        self.realtime_perception_worker.start()

        self.realtime_subscriber_worker = RealtimeSubscriberWorker(
            self._realtime_connect_address)
        self.realtime_subscriber_worker.frame_received.connect(
            self._on_realtime_frame)
        self.realtime_subscriber_worker.status_received.connect(
            self._on_realtime_status)
        self.realtime_subscriber_worker.error.connect(
            self._on_realtime_error)
        self.realtime_subscriber_worker.start()

    def _convert_detections_to_shapes(self,
                                      detections: List[dict],
                                      width: int,
                                      height: int) -> List[Shape]:
        shapes: List[Shape] = []
        if not detections:
            return shapes

        for detection in detections:
            label = str(detection.get("behavior", "") or "")
            base_color_rgb = self._get_rgb_by_label(label) or (0, 255, 0)
            base_color = QtGui.QColor(
                int(base_color_rgb[0]),
                int(base_color_rgb[1]),
                int(base_color_rgb[2]),
                255
            )
            fill_color = QtGui.QColor(
                base_color.red(),
                base_color.green(),
                base_color.blue(),
                60
            )

            bbox = detection.get("bbox_normalized") or []
            if len(bbox) != 4:
                bbox = None

            rect_shape = None
            if bbox:
                x1 = max(0.0, min(1.0, float(bbox[0]))) * width
                y1 = max(0.0, min(1.0, float(bbox[1]))) * height
                x2 = max(0.0, min(1.0, float(bbox[2]))) * width
                y2 = max(0.0, min(1.0, float(bbox[3]))) * height

                if x2 > x1 and y2 > y1:
                    rect_shape = Shape(
                        label=label,
                        shape_type="rectangle",
                        flags={"source": "realtime"},
                        description="realtime",
                    )
                    rect_shape.points = [
                        QtCore.QPointF(x1, y1),
                        QtCore.QPointF(x2, y2),
                    ]
                    rect_shape.point_labels = [1, 1]
                    rect_shape.fill = True
                    rect_shape.line_color = QtGui.QColor(base_color)
                    rect_shape.fill_color = QtGui.QColor(fill_color)
                    rect_shape.select_line_color = QtGui.QColor(
                        255, 255, 255, 255)
                    rect_shape.select_fill_color = QtGui.QColor(
                        base_color.red(),
                        base_color.green(),
                        base_color.blue(),
                        160
                    )
                    rect_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0))
                    rect_shape.other_data["source"] = "realtime"
                    rect_shape.other_data["frame_timestamp"] = detection.get(
                        "timestamp")
                    shapes.append(rect_shape)

            mask_data = detection.get("mask")
            if mask_data:
                mask = self._decode_mask(mask_data, width, height)
                if mask is not None:
                    mask_shape = MaskShape(
                        label=label,
                        flags={"source": "realtime"},
                        description="realtime_mask",
                    )
                    mask_shape.mask_color = np.array(
                        [base_color.red(), base_color.green(),
                         base_color.blue(), 64],
                        dtype=np.uint8
                    )
                    mask_shape.boundary_color = np.array(
                        [base_color.red(), base_color.green(),
                         base_color.blue(), 180],
                        dtype=np.uint8
                    )
                    mask_shape.mask = mask
                    mask_shape.scale = 1.0
                    mask_shape.other_data = dict(
                        rect_shape.other_data if rect_shape else {})
                    mask_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0))
                    shapes.append(mask_shape)

            keypoints = detection.get("keypoints")
            if keypoints:
                try:
                    kp_array = np.array(
                        keypoints, dtype=np.float32).reshape(-1, 2)
                except ValueError:
                    kp_array = np.array(keypoints, dtype=np.float32)
                    if kp_array.ndim == 1:
                        kp_array = kp_array.reshape(-1, 2)
                if kp_array.size > 0 and kp_array.shape[1] == 2:
                    points_shape = Shape(
                        label=f"{label}_keypoints",
                        shape_type="points",
                        flags={"source": "realtime"},
                        description="realtime_keypoints",
                    )
                    points_shape.points = []
                    points_shape.point_labels = []
                    for point in kp_array:
                        px = max(0.0, min(1.0, float(point[0]))) * width
                        py = max(0.0, min(1.0, float(point[1]))) * height
                        points_shape.points.append(QtCore.QPointF(px, py))
                        points_shape.point_labels.append(1)
                    points_shape.line_color = QtGui.QColor(base_color)
                    points_shape.vertex_fill_color = QtGui.QColor(
                        base_color.red(),
                        base_color.green(),
                        base_color.blue(),
                        255
                    )
                    shapes.append(points_shape)

        return shapes

    @QtCore.Slot(object, dict, list)
    def _on_realtime_frame(self, qimage, metadata, detections):
        if not self.realtime_running:
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.canvas.loadPixmap(pixmap, clear_shapes=False)
        shapes = self._convert_detections_to_shapes(
            detections, pixmap.width(), pixmap.height())
        if hasattr(self.canvas, "setRealtimeShapes"):
            self.canvas.setRealtimeShapes(shapes)
        self._realtime_shapes = shapes

        if self.realtime_log_fp:
            try:
                record = {
                    "timestamp": time.time(),
                    "frame_metadata": metadata,
                    "detections": detections,
                }
                json.dump(record, self.realtime_log_fp)
                self.realtime_log_fp.write("\n")
                self.realtime_log_fp.flush()
            except Exception as exc:
                logger.error("Failed to write realtime NDJSON record: %s",
                             exc, exc_info=True)
                try:
                    self.realtime_log_fp.close()
                except Exception:
                    pass
                self.realtime_log_fp = None
                self.realtime_log_path = None
                self.realtime_log_enabled = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Realtime detections for frame %s: %d",
                         metadata.get("frame_index"),
                         len(detections))

        self.canvas.update()

        frame_index = metadata.get("frame_index")
        detection_count = len(shapes)
        self.statusBar().showMessage(
            self.tr("Realtime frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?",
               detection_count))
        self.realtime_control_widget.set_status_text(
            self.tr("Frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?",
               detection_count))

    @QtCore.Slot(dict)
    def _on_realtime_status(self, status):
        if not isinstance(status, dict):
            return
        event_name = status.get("event") or "status"
        message = self.tr("Realtime %s: %s") % (
            event_name,
            status.get("recording_state", status.get("message", "")),
        )
        self.statusBar().showMessage(message)
        self.realtime_control_widget.set_status_text(message)

    @QtCore.Slot(str)
    def _on_realtime_error(self, message: str):
        logger.error("Realtime error: %s", message)
        self.realtime_control_widget.set_status_text(
            self.tr("Realtime error: %s") % message)
        QtWidgets.QMessageBox.critical(
            self,
            self.tr("Realtime Inference Error"),
            str(message),
        )
        self.stop_realtime_inference()

    @QtCore.Slot()
    def _on_realtime_stopped(self):
        self.realtime_perception_worker = None
        self._finalize_realtime_shutdown()

    def _shutdown_realtime_subscriber(self):
        if self.realtime_subscriber_worker is not None:
            self.realtime_subscriber_worker.stop()
            self.realtime_subscriber_worker.wait(500)
            self.realtime_subscriber_worker = None

    def _finalize_realtime_shutdown(self):
        self._shutdown_realtime_subscriber()
        self.realtime_running = False
        self.realtime_perception_worker = None
        self.realtime_control_widget.set_running(False)
        if hasattr(self.canvas, "setRealtimeShapes"):
            self.canvas.setRealtimeShapes([])
        self._realtime_shapes = []
        if self.realtime_log_fp:
            try:
                self.realtime_log_fp.flush()
                self.realtime_log_fp.close()
            except Exception:
                pass
            self.realtime_log_fp = None
            self.realtime_log_path = None
        message = self.tr("Realtime inference stopped.")
        self.statusBar().showMessage(message)
        self.realtime_control_widget.set_status_text(message)

    def stop_realtime_inference(self):
        if self.realtime_perception_worker is None and not self.realtime_running:
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.tr("Realtime inference stopped."))
            return

        self.realtime_control_widget.set_stopping()
        self.realtime_control_widget.set_status_text(
            self.tr("Stopping realtime inference…"))
        self.statusBar().showMessage(self.tr("Stopping realtime inference…"))
        self._shutdown_realtime_subscriber()

        worker = self.realtime_perception_worker
        if worker is not None:
            worker.request_stop()
            return

        # Nothing running, finalize immediately.
        self._finalize_realtime_shutdown()

    def closeEvent(self, event):
        try:
            self.stop_realtime_inference()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.error("Error stopping realtime inference on exit: %s",
                         exc, exc_info=True)
        super().closeEvent(event)

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName("%sToolBar" % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            utils.addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar

    def scanAllImages(self, folderPath):
        extensions = [
            ".%s" % fmt.data().decode().lower()
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        extensions.append('.json')
        self.only_json_files = True

        images = []
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = osp.join(root, file)
                    if self.only_json_files and not file.lower().endswith('.json'):
                        self.only_json_files = False
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images

    def _addItem(self, filename, label_file):
        item = QtWidgets.QListWidgetItem(filename)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        is_checked = False
        if QtCore.QFile.exists(label_file):
            if LabelFile.is_label_file(label_file):
                is_checked = True
        elif self._annotation_store_has_frame(label_file):
            is_checked = True

        item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
        if not self.fileListWidget.findItems(filename, Qt.MatchExactly):
            self.fileListWidget.addItem(item)

    def _getLabelFile(self, filename):
        label_file = osp.splitext(filename)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(label_file)
            label_file = osp.join(self.output_dir, label_file_without_path)
        return label_file

    def tutorial(self):
        url = "https://github.com/healthonrails/annolid/tree/main/docs/tutorials"  # NOQA
        webbrowser.open(url)

    def removeSelectedPoint(self):
        self.canvas.removeSelectedPoint()
        self.canvas.update()
        if self.canvas.hShape and not self.canvas.hShape.points:
            self.canvas.deleteShape(self.canvas.hShape)
            self.remLabels([self.canvas.hShape])
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)

    def importDirImages(self, dirpath, pattern=None, load=True):
        self.actions.openNextImg.setEnabled(True)
        self.actions.openPrevImg.setEnabled(True)

        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.annotation_dir = dirpath
        self.filename = None
        self.fileListWidget.clear()
        for filename in self.scanAllImages(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = self._getLabelFile(filename)

            if not filename.endswith('.json') or self.only_json_files:
                self._addItem(filename, label_file)
        self.openNextImg(load=load)

    def _get_rgb_by_label(self, label):
        """
        Return an (R, G, B) tuple for a given label.

        If auto mode is active, compute the color by hashing the normalized label and adding
        an optional shift offset. Otherwise, if manual mapping is provided use that mapping;
        if neither is available, fall back to a default shape color.
        """
        schema = getattr(self, "project_schema", None)
        if schema is not None:
            behavior = schema.behavior_map().get(label)
            if behavior is None:
                behavior = next(
                    (beh for beh in schema.behaviors if beh.name.lower() == label.lower()),
                    None,
                )
            if behavior is not None and behavior.category_id:
                category = schema.category_map().get(behavior.category_id)
                if category and category.color:
                    rgb = _hex_to_rgb(category.color)
                    if rgb is not None:
                        return rgb

        config = self._config
        if config.get("shape_color") == "auto":
            # Normalize label (strip whitespace and lowercase)
            normalized_label = label.strip().lower()
            # Compute MD5 hash for reproducibility.
            hash_digest = hashlib.md5(
                normalized_label.encode("utf-8")).hexdigest()
            hash_int = int(hash_digest, 16)
            # Get a shift offset (default to 0 if not provided).
            shift_offset = config.get("shift_auto_shape_color", 0)
            # Calculate index within LABEL_COLORMAP.
            index = (hash_int + shift_offset) % len(LABEL_COLORMAP)
            # Convert the NumPy array color to a Python tuple.
            return (int(LABEL_COLORMAP[index][0]),
                    int(LABEL_COLORMAP[index][1]),
                    int(LABEL_COLORMAP[index][2]))
        elif (
            config.get("shape_color") == "manual"
            and config.get("label_colors")
            and label in config["label_colors"]
        ):
            return config["label_colors"][label]
        elif config.get("default_shape_color"):
            return config["default_shape_color"]

    def _update_shape_color(self, shape):

        if not self.uniqLabelList.findItemByLabel(shape.label):
            item = self.uniqLabelList.createItemFromLabel(shape.label)
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(item, shape.label, rgb)
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        if not shape.visible:
            shape.vertex_fill_color = QtGui.QColor(r, g, b, 0)
        else:
            shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)
        return r, g, b

    def addLabel(self, shape):
        if shape.group_id is None:
            text = str(shape.label)
        else:
            text = "{} ({})".format(shape.label, shape.group_id)
        shape_points_hash = hash(
            str(sorted(shape.points, key=lambda point: point.x())))
        self.shape_hash_ids[shape_points_hash] = self.shape_hash_ids.get(
            shape_points_hash, 0) + 1
        if self.shape_hash_ids[shape_points_hash] <= 1:
            self.label_stats[text] = self.label_stats.get(text, 0) + 1
        label_list_item = LabelListWidgetItem(text, shape)
        self.labelList.addItem(label_list_item)
        item = self.uniqLabelList.findItemByLabel(shape.label)
        if item is None:
            item = self.uniqLabelList.createItemFromLabel(
                shape.label
            )
            self.uniqLabelList.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(
                item, f"{shape.label} [{self.label_stats.get(text,0)} instance]", rgb)
        else:
            rgb = self._get_rgb_by_label(shape.label)
            self.uniqLabelList.setItemLabel(
                item, f"{shape.label} [{self.label_stats.get(text,0)} instances]", rgb)

        self.labelDialog.addLabelHistory(str(shape.label))
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

        r, g, b = self._update_shape_color(shape)
        label_list_item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                html.escape(text), r, g, b
            )
        )

    def propagateSelectedShape(self):
        """
        Triggered when the user selects "Propagate Selected Shape" from the context menu.
        Uses the currently selected shape in the canvas.
        """
        from annolid.gui.widgets.shape_dialog import ShapePropagationDialog
        if not self.canvas.selectedShapes:
            QtWidgets.QMessageBox.information(
                self, "No Shape Selected", "Please select a shape first.")
            return

        # For simplicity, take the first selected shape.
        selected_shape = self.canvas.selectedShapes[0]

        current_frame = self.frame_number  # AnnolidWindow's current frame attribute
        max_frame = self.num_frames - 1      # Total number of frames

        # Create the dialog, explicitly passing self (the main window) and canvas.
        dialog = ShapePropagationDialog(
            self.canvas, self, current_frame, max_frame, parent=self)

        # Optionally, preselect the shape in the dialog's list.
        for i in range(dialog.shape_list.count()):
            item = dialog.shape_list.item(i)
            if item.data(QtCore.Qt.UserRole) == selected_shape:
                dialog.shape_list.setCurrentRow(i)
                break

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage("Shape propagation completed.")
        else:
            self.statusBar().showMessage("Shape propagation canceled.")

    def editLabel(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
        if not item:
            item = self.currentItem()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        text, flags, group_id, description = self.labelDialog.popUp(
            text=str(shape.label), flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
        )
        if text is None:
            return
        if not self.validateLabel(text):
            self.errorMessage(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description

        r, g, b = self._update_shape_color(shape)

        if shape.group_id is None:
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    shape.label, r, g, b))
        else:
            item.setText("{} ({})".format(shape.label, shape.group_id))
        self.setDirty()
        if not self.uniqLabelList.findItemByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def _saveImageFile(self, filename):
        image_filename = filename.replace('.json', '.png')
        if self.imageData is None:
            return image_filename
        try:
            if not self.imageData.save(image_filename):
                logger.warning(f"Failed to save seed image: {image_filename}")
        except Exception as exc:
            logger.warning(
                f"Exception while saving seed image {image_filename}: {exc}")
        return image_filename

    def _get_current_model_config(self):
        """Return the ModelConfig for the currently selected model, if any."""
        return self.ai_model_manager.get_current_model()

    def get_current_model_weight_file(self) -> str:
        """
        Returns the weight file associated with the currently selected model.
        If no matching model is found, returns a default fallback weight file.
        """
        return self.ai_model_manager.get_current_weight()

    def _resolve_model_identity(self):
        model_config = self._get_current_model_config()
        identifier = model_config.identifier if model_config else None
        weight = model_config.weight_file if model_config else None
        if identifier is None and weight is None:
            fallback = self.get_current_model_weight_file()
            identifier = fallback
            weight = fallback
        return model_config, identifier or "", weight or ""

    @staticmethod
    def _is_cotracker_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return identifier == "cotracker" or weight == "cotracker.pt"

    @staticmethod
    def _is_dino_keypoint_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return identifier == "dinov3_keypoint_tracker" or weight == "dino_keypoint_tracker"

    @staticmethod
    def _is_yolo_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        if "yolo" in identifier or "yolo" in weight:
            return True

        non_yolo_keywords = (
            "sam",
            "dinov",
            "dino",
            "cotracker",
            "cutie",
            "efficientvit",
            "mediapipe",
            "maskrcnn",
        )
        if any(keyword in identifier for keyword in non_yolo_keywords):
            return False
        if any(keyword in weight for keyword in non_yolo_keywords):
            return False

        # Many custom YOLO exports rely on generic names such as "best.pt".
        yolo_extensions = (".pt", ".pth", ".onnx", ".engine", ".mlpackage")
        if weight.endswith(yolo_extensions):
            return True

        return False

    @staticmethod
    def _is_sam2_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return "sam2_hiera" in identifier or "sam2_hiera" in weight

    def _resolve_sam2_model_config(self, identifier: str, weight: str) -> str:
        """
        Resolve the SAM2 config file name based on the selected identifier or weight.
        Falls back to the small hierarchy config if nothing matches.
        """
        key = f"{identifier or ''}|{weight or ''}".lower()
        if "hiera_l" in key:
            return "sam2.1_hiera_l.yaml"
        if "hiera_s" in key:
            return "sam2.1_hiera_s.yaml"
        return "sam2.1_hiera_s.yaml"

    def _resolve_sam2_checkpoint_path(self, weight: str) -> Optional[str]:
        """
        Try to resolve the absolute checkpoint path for SAM2 models.
        Returns None to use the default download location when the file is not found.
        """
        if not weight:
            return None

        weight = weight.strip()
        if not weight:
            return None

        weight_path = Path(weight)
        if weight_path.exists():
            return str(weight_path)

        checkpoints_dir = (
            Path(__file__).resolve().parent.parent
            / "segmentation"
            / "SAM"
            / "segment-anything-2"
            / "checkpoints"
        )

        candidate = checkpoints_dir / weight_path.name
        if candidate.exists():
            return str(candidate)

        lower_name = weight_path.name.lower()
        fallback_names = []
        if "hiera_l" in lower_name:
            fallback_names.extend(
                ["sam2_hiera_large.pt", "sam2.1_hiera_large.pt"]
            )
        elif "hiera_s" in lower_name:
            fallback_names.extend(
                ["sam2_hiera_small.pt", "sam2.1_hiera_small.pt"]
            )

        for fallback_name in fallback_names:
            fallback_candidate = checkpoints_dir / fallback_name
            if fallback_candidate.exists():
                return str(fallback_candidate)

        return None

    def stop_prediction(self):
        # Emit the stop signal to signal the prediction thread to stop
        self.pred_worker.stop_signal.emit()
        self.seg_pred_thread.quit()
        self.seg_pred_thread.wait()
        self.stepSizeWidget.predict_button.setText(
            "Pred")  # Change button text
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")

        self.stop_prediction_flag = False
        logger.info(f"Prediction was stopped.")

    def extract_visual_prompts_from_canvas(self) -> dict:
        """
        Extract visual prompts from canvas rectangle shapes.

        This function iterates over all shapes on the canvas, selects those that
        are rectangles, and constructs:
        - A list of bounding boxes [x1, y1, x2, y2].
        - A list of class indices for each bounding box.

        It also builds/updates self.class_mapping where each unique label found
        is mapped to an integer. These labels will be used as class names for YOLOE.

        Returns:
            dict: A dictionary with keys "bboxes" and "cls" containing lists.
                Returns an empty dict if no valid rectangle shapes are found.
        """
        bboxes = []
        cls_list = []
        # Build or update the mapping using all rectangle shapes with a valid label.
        labels = {shape.label for shape in self.canvas.shapes
                  if shape.shape_type == 'rectangle' and shape.label}
        if labels:
            # Create a sorted mapping so that the order is predictable.
            self.class_mapping = {label: idx for idx,
                                  label in enumerate(sorted(labels))}
        else:
            self.class_mapping = {}

        for shape in self.canvas.shapes:
            if shape.shape_type != 'rectangle':
                continue
            if not shape.points or len(shape.points) < 2:
                continue

            # Compute bounding box coordinates.
            xs = [pt.x() if hasattr(pt, "x") else pt[0] for pt in shape.points]
            ys = [pt.y() if hasattr(pt, "y") else pt[1] for pt in shape.points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            bboxes.append([x1, y1, x2, y2])

            # Use the class mapping to get the index.
            cls_idx = self.class_mapping.get(shape.label, 0)
            cls_list.append(cls_idx)

        if not bboxes:
            logger.info(
                "No rectangle shapes found on canvas for visual prompts.")
            return {}

        # Convert arrays to plain Python lists to avoid pop() errors in YOLOE.
        return {"bboxes": bboxes, "cls": cls_list}

    def predict_from_next_frame(self, to_frame=60):
        """
        Updated prediction routine that extracts visual prompts from the canvas.
        If the current model supports visual prompts (e.g. YOLOE), the prompts are extracted
        from the canvas rectangle shapes and passed to the inference module.
        """
        model_config, model_identifier, model_weight = self._resolve_model_identity()
        model_name = model_identifier or model_weight
        if self.pred_worker and self.stop_prediction_flag:
            self.stop_prediction()
            return
        elif len(self.canvas.shapes) <= 0 and not self._is_yolo_model(model_name, model_weight):
            QtWidgets.QMessageBox.about(self,
                                        "No Shapes or Labeled Frames",
                                        "Please label this frame")
            return

        if self.video_file:

            if self.video_results_folder:  # video_results_folder is Path object
                self._setup_prediction_folder_watcher(
                    str(self.video_results_folder))

            if self._is_dino_keypoint_model(model_name, model_weight):
                dino_model = self.patch_similarity_model or PATCH_SIMILARITY_DEFAULT_MODEL
                # Instead of passing a reference to the shared config object,
                # pass a deep copy. This ensures every tracking run starts with
                # a pristine configuration, free from any mutations made by
                # a previous run. This is the key to a true "start from scratch".
                fresh_tracker_config = copy.deepcopy(
                    self.tracker_runtime_config)

                self.video_processor = DinoKeypointVideoProcessor(
                    video_path=self.video_file,
                    result_folder=self.video_results_folder,
                    model_name=dino_model,
                    short_side=768,
                    device=None,
                    runtime_config=fresh_tracker_config,
                )
            elif self._is_sam2_model(model_name, model_weight):
                from annolid.segmentation.SAM.sam_v2 import process_video
                sam2_config = self._resolve_sam2_model_config(
                    model_name, model_weight
                )
                sam2_checkpoint = self._resolve_sam2_checkpoint_path(
                    model_weight
                )
                logger.info(
                    "Using SAM2 config '%s' with checkpoint '%s'",
                    sam2_config,
                    sam2_checkpoint if sam2_checkpoint else "auto-download",
                )
                self.video_processor = functools.partial(
                    process_video,
                    video_path=self.video_file,
                    checkpoint_path=sam2_checkpoint,
                    model_config=sam2_config,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                )
            elif self._is_yolo_model(model_name, model_weight):
                from annolid.segmentation.yolos import InferenceProcessor
                # Instead of using a hard-coded prompt, extract visual prompts from canvas.
                visual_prompts = self.extract_visual_prompts_from_canvas()
                # Optionally, log the mapping
                logger.info(f"Extracted visual prompts: {visual_prompts}")
                # For YOLO models, you might also pass class names if needed.
                # Here, class_names could be the sorted keys of self.class_mapping.
                class_names = list(self.class_mapping.keys()) if hasattr(
                    self, "class_mapping") else None
                if len(class_names) < 1:
                    text_prompt = self.aiRectangle._aiRectanglePrompt.text().lower()
                    class_names = text_prompt.split(",")
                    logger.info(
                        f"Extracted class names from text prompt: {class_names}")
                self.video_processor = InferenceProcessor(model_name=model_weight,
                                                          model_type="yolo",
                                                          class_names=class_names)
            else:
                from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor
                self.video_processor = VideoProcessor(
                    self.video_file,
                    model_name=model_name,
                    save_image_to_disk=False,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                    t_max_value=self.t_max_value,
                    use_cpu_only=self.use_cpu_only,
                    auto_recovery_missing_instances=self.auto_recovery_missing_instances,
                    save_video_with_color_mask=self.save_video_with_color_mask,
                    compute_optical_flow=self.compute_optical_flow,
                    results_folder=str(self.video_results_folder)
                    if self.video_results_folder else None,
                )
            if not self.seg_pred_thread.isRunning():
                self.seg_pred_thread = QtCore.QThread()
            self.seg_pred_thread.start()
            # Determine end_frame
            # step_size is -1, i.e., predict to the end
            if self.step_size < 0:
                end_frame = self.num_frames + self.step_size
            else:
                end_frame = self.frame_number + to_frame * self.step_size
            if end_frame >= self.num_frames:
                end_frame = self.num_frames - 1
            stop_when_lost_tracking_instance = (self.stepSizeWidget.occclusion_checkbox.isChecked()
                                                or self.automatic_pause_enabled)
            if self._is_dino_keypoint_model(model_name, model_weight):
                # Run the Cutie + DINO tracker over the full video by default.
                end_frame = self.num_frames - 1
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.process_video,
                    start_frame=self.frame_number,
                    end_frame=end_frame,
                    step=1,
                    pred_worker=None,
                )
                self.video_processor.set_pred_worker(self.pred_worker)
                self.pred_worker._kwargs["pred_worker"] = self.pred_worker
            elif self._is_sam2_model(model_name, model_weight):
                frame_idx = max(self.frame_number, 0)
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor,
                    frame_idx=frame_idx,
                )
            elif self._is_yolo_model(model_name, model_weight):
                # Pass visual_prompts to run_inference if extracted successfully.
                self.pred_worker = FlexibleWorker(
                    task_function=lambda: self.video_processor.run_inference(
                        source=self.video_file,
                        visual_prompts=visual_prompts if visual_prompts else None
                    )
                )
            else:
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.process_video_frames,
                    start_frame=self.frame_number+1,
                    end_frame=end_frame,
                    step=self.step_size,
                    is_cutie=False if self._is_cotracker_model(
                        model_name, model_weight) else True,
                    mem_every=self.step_size,
                    point_tracking=self._is_cotracker_model(
                        model_name, model_weight),
                    has_occlusion=stop_when_lost_tracking_instance,
                )
                self.video_processor.set_pred_worker(self.pred_worker)
            self.frame_number += 1
            logger.info(f"Prediction started from frame: {self.frame_number}")
            self.stepSizeWidget.predict_button.setText("Stop")
            self.stepSizeWidget.predict_button.setStyleSheet(
                "background-color: red; color: white;")
            self.stop_prediction_flag = True
            self.pred_worker.moveToThread(self.seg_pred_thread)
            self.pred_worker.start_signal.connect(self.pred_worker.run)
            self.pred_worker.result_signal.connect(self.lost_tracking_instance)
            self.pred_worker.finished_signal.connect(self.predict_is_ready)
            self.seg_pred_thread.finished.connect(self.seg_pred_thread.quit)
            self.pred_worker.start_signal.emit()

    def lost_tracking_instance(self, message):
        if message is None or "#" not in str(message):
            return
        message, current_frame_index = message.split("#")
        current_frame_index = int(current_frame_index)
        if "missing instance(s)" in message:
            QtWidgets.QMessageBox.information(
                self, "Stop early",
                message
            )
        self.stepSizeWidget.predict_button.setText(
            "Pred")  # Change button text
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False

    def predict_is_ready(self, messege):
        self.stepSizeWidget.predict_button.setText(
            "Pred")  # Change button text
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False
        try:
            if messege is not None and "last frame" in str(messege):
                QtWidgets.QMessageBox.information(
                    self, "Stop early",
                    messege
                )
            else:
                if self.video_loader is not None:
                    json_count = count_json_files(self.video_results_folder)
                    store_count = self._annotation_store_frame_count()
                    total_predicted = max(json_count, store_count)
                    expected_total_frames = self.num_frames or total_predicted
                    missing_frames = max(
                        0, expected_total_frames - total_predicted)
                    logger.info(
                        f"Predicted frames available: json={json_count}, store={store_count}, total={total_predicted} of {self.num_frames}")
                    if total_predicted >= max(1, expected_total_frames - 1):
                        self.convert_json_to_tracked_csv()
                    elif total_predicted > 0:
                        logger.info(
                            "Prediction finished with %d missing frame(s); generating CSV with available results.",
                            missing_frames
                        )
                        self.convert_json_to_tracked_csv()
                    else:
                        logger.info(
                            "Prediction finished without any frames to convert; skipping CSV generation."
                        )
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
        self.reset_predict_button()

    def reset_predict_button(self):
        """Reset the predict button text and style"""
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")

    def loadFlags(self, flags):
        """Delegate flag loading to the flags controller."""
        self.flags_controller.load_flags(flags)

    @property
    def pinned_flags(self):
        if hasattr(self, "flags_controller"):
            return self.flags_controller.pinned_flags
        return getattr(self, "_pending_pinned_flags", {})

    @pinned_flags.setter
    def pinned_flags(self, value):
        if hasattr(self, "flags_controller"):
            self.flags_controller.set_flags(value or {}, persist=False)
        else:
            self.__dict__["_pending_pinned_flags"] = value or {}

    def _refresh_behavior_overlay(self) -> None:
        """Synchronize canvas label and flag widget with timeline behaviors."""
        active_behaviors = self.behavior_controller.active_behaviors(
            self.frame_number)

        # Preserve any user-specified flags that aren't managed by the behavior controller.
        current_flags: Dict[str, bool] = {}
        table = self.flag_widget._table
        for row in range(table.rowCount()):
            name_widget = table.cellWidget(row, FlagTableWidget.COLUMN_NAME)
            value_widget = table.cellWidget(row, FlagTableWidget.COLUMN_ACTIVE)
            if isinstance(name_widget, QtWidgets.QLineEdit) and isinstance(value_widget, QtWidgets.QCheckBox):
                name = name_widget.text().strip()
                if name:
                    current_flags[name] = value_widget.isChecked()

        for behavior in sorted(self.behavior_controller.behavior_names):
            current_flags[behavior] = behavior in active_behaviors

        if current_flags:
            self.loadFlags(current_flags)
        else:
            text = ",".join(sorted(active_behaviors)
                            ) if active_behaviors else None
            self.canvas.setBehaviorText(text)

    def _get_pil_image_from_state(self) -> Image.Image | None:
        """
        Safely converts the stored image data (self.imageData) to a standard
        PIL.Image.Image object, regardless of its current Qt-related type.

        This method centralizes the conversion logic to ensure robustness and
        avoids code duplication.

        Returns:
            A PIL.Image.Image object in RGB format, or None if conversion fails.
        """
        if self.imageData is None:
            return None

        # 0. Handle raw bytes (from a loaded JSON with embedded data)
        if isinstance(self.imageData, bytes):
            try:
                # Create an in-memory binary stream and open it with Pillow
                pil_image = Image.open(io.BytesIO(self.imageData))
            except Exception as e:
                logger.error(f"Failed to load PIL Image from bytes: {e}")
                return None

        # 1. Check if it's already the target type
        elif isinstance(self.imageData, Image.Image):
            pil_image = self.imageData
        # 2. Check for the most common case: a QImage from the video loader
        elif isinstance(self.imageData, tuple(filter(None, (
            QtGui.QImage,
            getattr(ImageQt, "ImageQt", None),
        )))):
            qimage = QtGui.QImage(self.imageData)
            image_bytes = self._qimage_to_bytes(qimage)
            if image_bytes is None:
                logger.error("Failed to serialize QImage to bytes for saving.")
                return None
            try:
                with io.BytesIO(image_bytes) as buffer:
                    pil_image = Image.open(buffer)
                    pil_image.load()
            except Exception as e:
                logger.error(f"Failed to convert QImage to PIL Image: {e}")
                return None
        else:
            logger.warning(
                f"self.imageData is of an unexpected type ({type(self.imageData)}). "
                "Cannot convert to PIL.Image for saving."
            )
            return None

        # 4. Ensure the final image is in a standard format (RGB)
        if pil_image.mode != 'RGB':
            return pil_image.convert('RGB')

        return pil_image

    def saveLabels(self, filename, save_image_data=True):
        lf = LabelFile()
        has_zone_shapes = False

        def format_shape(s):
            data = s.other_data.copy()
            if s.description and 'zone' in s.description.lower():
                has_zone_shapes = True
            if len(s.points) <= 1:
                s.shape_type = 'point'
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    visible=s.visible,
                    description=s.description
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        if self.flag_widget:
            # Retrieve the flags as a dictionary
            flags = {_flag: True for _flag in self.flag_widget._get_existing_flag_names(
            ) if self.is_behavior_active(self.frame_number, _flag)}

        if self.canvas.current_behavior_text is not None:
            behaviors = self.canvas.current_behavior_text.split(",")
            for behavior in behaviors:
                if len(behavior) > 0:
                    flags[behavior] = True
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = None
            if save_image_data and self._config["store_data"]:
                # Call the dedicated helper method to get a clean PIL image
                pil_image_to_save = self._get_pil_image_from_state()

                # If the conversion was successful, get the byte data
                if pil_image_to_save:
                    imageData = utils.img_pil_to_data(pil_image_to_save)

            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=Path(imagePath).name,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                caption=self.canvas.getCaption(),
            )
            if has_zone_shapes:
                self.zone_path = filename

            self.labelFile = lf
            items = self.fileListWidget.findItems(
                self.imagePath, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            # Emit VideoManagerWidget's json_saved signal if shapes are present and video_file is set
            if shapes and self.video_file:
                self.video_manager_widget.json_saved.emit(
                    self.video_file, filename)
                logger.debug(
                    f"Emitted VideoManagerWidget.json_saved for video: {self.video_file}, JSON: {filename}")
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def _saveFile(self, filename):
        json_saved = self.saveLabels(filename)
        if filename and json_saved:
            self.changed_json_stats[filename] = self.changed_json_stats.get(
                filename, 0) + 1
            if (self.changed_json_stats[filename] >= 1
                    and self._pred_res_folder_suffix in filename):

                changed_folder = str(Path(filename).parent)
                if '_edited' not in changed_folder:
                    changed_folder = Path(changed_folder + '_edited')
                else:
                    changed_folder = Path(changed_folder)
                if not changed_folder.exists():
                    changed_folder.mkdir(exist_ok=True, parents=True)
                changed_filename = changed_folder / Path(filename).name
                _ = self.saveLabels(str(changed_filename))
                _ = self._saveImageFile(
                    str(changed_filename).replace('.json', '.png'))
            image_filename = self._saveImageFile(filename)
            self.imageList.append(image_filename)
            self.addRecentFile(filename)
            label_file = self._getLabelFile(filename)
            self._addItem(image_filename, label_file)

            if self.caption_widget is not None:
                self.caption_widget.set_image_path(image_filename)

            self.setClean()

    def getLabelFile(self):
        if str(self.filename).lower().endswith(".json"):
            label_file = str(self.filename)
        else:
            label_file = osp.splitext(str(self.filename))[0] + ".json"

        return label_file

    def popLabelListMenu(self, point):
        try:
            self.menus.labelList.exec_(self.labelList.mapToGlobal(point))
        except AttributeError:
            return

    def setDirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            if self.filename:
                label_file = osp.splitext(self.filename)[0] + ".json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(
                        self.output_dir, label_file_without_path)
                self.saveLabels(label_file)
                self.saveFile()
                return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = self.getTitle(clean=False)
        self.setWindowTitle(title)

    def getTitle(self, clean=True):
        title = __appname__
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
            # Inform behavior widget of the live frame index for quick selection in dialogs
            if getattr(self.caption_widget, "behavior_widget", None) is not None:
                try:
                    self.caption_widget.behavior_widget.set_current_frame(self.frame_number)
                except Exception:
                    pass
        _filename = os.path.basename(self.filename)
        if self.video_loader:
            if self.frame_number:
                self._time_stamp = convert_frame_number_to_time(
                    self.frame_number, self.fps)
                if clean:
                    title = f"{title}-Video Timestamp:{self._time_stamp}|Events:{self.behavior_controller.events_count}"
                    title = f"{title}|Frame_number:{self.frame_number}"
                else:
                    title = f"{title}|Video Timestamp:{self._time_stamp}"

                    title = f"{title}|Frame_number:{self.frame_number}*"
            else:
                if clean:
                    title = "{} - {}".format(title, _filename)
                else:
                    title = "{} - {}*".format(title, _filename)
        return title

    def deleteAllFuturePredictions(self):
        """
        Delete all future prediction files except manually labeled ones.
        This version uses robust parsing to handle varied filename formats.
        """
        if not self.video_loader or not self.video_results_folder:
            return

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return

        deleted_files = 0
        protected_frames: Set[int] = set()

        logger.info(f"Scanning for future predictions in: {prediction_folder}")

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)

            # If the filename doesn't match our expected pattern, skip it safely.
            if not match:
                logger.debug(
                    "Skipping file with unexpected name format: %s",
                    prediction_path.name,
                )
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse frame number from file: %s",
                    prediction_path.name,
                )
                continue

            image_file_png = prediction_path.with_suffix(".png")
            image_file_jpg = prediction_path.with_suffix(".jpg")
            is_manually_saved = image_file_png.exists() or image_file_jpg.exists()
            if is_manually_saved:
                protected_frames.add(frame_number)

            is_future_frame = frame_number > self.frame_number

            if is_future_frame and not is_manually_saved:
                try:
                    prediction_path.unlink()
                    deleted_files += 1
                except OSError as e:
                    logger.error(
                        "Failed to delete file %s: %s", prediction_path, e
                    )

        store_removed = 0
        try:
            store = AnnotationStore.for_frame_path(
                prediction_folder / f"{prediction_folder.name}_000000000.json"
            )
            store_removed = store.remove_frames_after(
                self.frame_number, protected_frames=protected_frames
            )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s future prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("prediction_progress")
            self.last_known_predicted_frame = -1
            self.prediction_start_timestamp = 0.0
            if hasattr(self, "_update_progress_bar"):
                self._update_progress_bar(0)
            try:
                self._scan_prediction_folder(str(prediction_folder))
            except Exception as exc:  # pragma: no cover - GUI safeguard
                logger.debug(
                    "Failed to rescan prediction folder after deletion: %s", exc
                )
        else:
            logger.info(
                "No future prediction files or store records required removal."
            )

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "Or all the predicted label files from the next frame, "
            "what would you like to do?"
        )
        msg_box = mb(self)
        msg_box.setIcon(mb.Warning)
        msg_box.setText(msg)
        msg_box.setStandardButtons(mb.No | mb.Yes | mb.YesToAll)
        msg_box.setDefaultButton(mb.No)
        answer = msg_box.exec_()

        if answer == mb.No:
            return
        elif answer == mb.YesToAll:
            # Handle the case to delete all predictions from now on
            self.deleteAllFuturePredictions()
        else:
            label_file = self.getLabelFile()
            if osp.exists(label_file):
                os.remove(label_file)
                img_file = label_file.replace('.json', '.png')
                if osp.exists(img_file):
                    os.remove(img_file)
                logger.info("Label file is removed: {}".format(label_file))

                item = self.fileListWidget.currentItem()
                if item:
                    item.setCheckState(Qt.Unchecked)

                self.resetState()

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            title = self.getTitle()
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def save_labels(self):
        """Save the labels into a selected text file.
        """
        file_name, extension = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labels file",
            str(self.here.parent / 'annotation'),
            filter='*.txt'
        )

        # return if user cancels the operation
        if len(file_name) < 1:
            return

        if Path(file_name).is_file() or Path(file_name).parent.is_dir():
            labels_text_list = ['__ignore__', '_background_']
            for i in range(self.uniqLabelList.count()):
                # get all the label names from the unique list
                label_name = self.uniqLabelList.item(
                    i).data(QtCore.Qt.UserRole)
                labels_text_list.append(label_name)

            with open(file_name, 'w') as lt:
                for ltl in labels_text_list:
                    lt.writelines(str(ltl)+'\n')

    def frames(self):
        """Extract frames based on the selected algos.
        """
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
            sub_clip = isinstance(
                start_seconds, int) and isinstance(end_seconds, int)

        if video_file is None:
            return
        out_frames_gen = videos.extract_frames(
            video_file,
            num_frames=num_frames,
            algo=algo,
            out_dir=out_dir,
            sub_clip=sub_clip,
            start_seconds=start_seconds,
            end_seconds=end_seconds
        )

        pw = ProgressingWindow(out_frames_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        if out_dir is None:
            out_frames_dir = str(
                Path(video_file).resolve().with_suffix(''))
        else:
            out_frames_dir = str(Path(out_dir) / Path(video_file).name)

        if start_seconds is not None and end_seconds is not None:
            out_frames_dir = f"{out_frames_dir}_{start_seconds}_{end_seconds}"
        out_frames_dir = f"{out_frames_dir}_{algo}"

        self.annotation_dir = out_frames_dir

        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                         {out_frames_dir}")
        self.statusBar().showMessage(
            self.tr(f"Finshed extracting frames."))
        self.importDirImages(out_frames_dir)

    def convert_json_to_tracked_csv(self):
        """
        Convert JSON annotations to a tracked CSV file and handle the progress using a separate thread.
        """
        if not self.video_file:
            QtWidgets.QMessageBox.warning(
                self, "Missing Video File", "No video file selected.")
            return

        video_file = self.video_file
        out_folder = Path(video_file).with_suffix('')

        if not out_folder or not out_folder.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "No Predictions Found",
                "Help Annolid achieve precise predictions by labeling a frame. Your input is valuable!"
            )
            return

        csv_output_path = out_folder.parent / \
            f"{out_folder.name}_tracking.csv"

        if getattr(self, "csv_thread", None) and self.csv_thread and self.csv_thread.isRunning():
            job = (out_folder, csv_output_path)
            if job not in self._csv_conversion_queue:
                self._csv_conversion_queue.append(job)
                self.statusBar().showMessage("Queued tracking CSV conversion...", 3000)
            return

        self._start_csv_conversion(out_folder, csv_output_path)

    def _start_csv_conversion(self, out_folder: Path, csv_output_path: Path):
        """Kick off a background CSV conversion job for the given folder."""
        self._initialize_progress_bar()
        self._last_tracking_csv_path = str(csv_output_path)
        self.statusBar().showMessage(
            f"Generating tracking CSV: {csv_output_path.name}", 3000)

        try:
            self.csv_worker = FlexibleWorker(
                task_function=labelme2csv.convert_json_to_csv,
                json_folder=str(out_folder),
                csv_file=str(csv_output_path),
                progress_callback=self._csv_worker_progress_proxy
            )
            self.csv_thread = QtCore.QThread()

            # Move the worker to the thread and connect signals
            self.csv_worker.moveToThread(self.csv_thread)
            self._connect_worker_signals()

            # Safely start the thread and worker signal
            self.csv_thread.start()
            QtCore.QTimer.singleShot(
                0, lambda: self.csv_worker.start_signal.emit())

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}")
            try:
                if hasattr(self, "progress_bar") and self.progress_bar.isVisible():
                    self.statusBar().removeWidget(self.progress_bar)
            except Exception:
                pass
            self.csv_worker = None
            self.csv_thread = None
            self._last_tracking_csv_path = None

            if self._csv_conversion_queue:
                next_out, next_csv = self._csv_conversion_queue.pop(0)
                self._start_csv_conversion(next_out, next_csv)

    def _on_csv_conversion_finished(self, result=None):
        """Handle cleanup and user feedback after CSV conversion completes."""
        try:
            if hasattr(self, "progress_bar") and self.progress_bar.isVisible():
                self.statusBar().removeWidget(self.progress_bar)
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
        except Exception:
            pass

        if isinstance(result, str) and result.startswith("No annotation"):
            QtWidgets.QMessageBox.information(
                self,
                "Tracking CSV",
                result
            )
            self._last_tracking_csv_path = None
            self._cleanup_csv_worker()
            return

        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(
                self,
                "Tracking CSV Error",
                f"Failed to generate tracking CSV:\n{result}"
            )
            self._cleanup_csv_worker()
            return

        csv_path = getattr(self, "_last_tracking_csv_path", None)
        if csv_path:
            path_obj = Path(csv_path)
            if path_obj.exists():
                QtWidgets.QMessageBox.information(
                    self,
                    "Tracking Complete",
                    f"Review the file at:\n{csv_path}"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Tracking CSV Missing",
                    f"Expected tracking file was not found:\n{csv_path}\n"
                    "Please try saving again."
                )
            self._last_tracking_csv_path = None
        self._cleanup_csv_worker()

    def _initialize_progress_bar(self):
        """Initialize the progress bar and add it to the status bar."""
        self.progress_bar.setValue(0)
        self.statusBar().addWidget(self.progress_bar)

    def _csv_worker_progress_proxy(self, progress):
        """Route worker progress updates through thread-safe signal emission."""
        worker = getattr(self, "csv_worker", None)
        if worker is not None:
            worker.report_progress(progress)

    def _update_progress_bar(self, progress):
        """Update the progress bar's value."""
        self.progress_bar.setValue(progress)

    # method to hide progress bar
    def _finalize_prediction_progress(self, message=""):
        logger.info(f"Prediction finalization: {message}")
        if hasattr(self, 'progress_bar') and self.progress_bar.isVisible():
            self.statusBar().removeWidget(self.progress_bar)
        self._stop_prediction_folder_watcher()
        # Clear "predicted" marks from the slider
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")  # Use the new method

        # Reset button state (already in predict_is_ready and lost_tracking_instance)
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False  # This flag is specific to AnnolidWindow

    def _setup_prediction_folder_watcher(self, folder_path_to_watch):
        if self.prediction_progress_watcher is None:
            self.prediction_progress_watcher = QFileSystemWatcher(self)
            self.prediction_progress_watcher.directoryChanged.connect(
                self._handle_prediction_folder_change
            )
            # You can also watch for file additions specifically if directoryChanged is too broad
            # self.prediction_progress_watcher.fileChanged.connect(...)

        # Remove any existing paths
        if self.prediction_progress_watcher.directories():
            self.prediction_progress_watcher.removePaths(
                self.prediction_progress_watcher.directories())

        if osp.isdir(folder_path_to_watch):
            self.prediction_start_timestamp = time.time()
            self.prediction_progress_watcher.addPath(str(folder_path_to_watch))
            logger.info(
                f"Prediction progress watcher started for: {folder_path_to_watch}")
            # Initial scan when watcher starts
            self._scan_prediction_folder(folder_path_to_watch)
        else:
            logger.warning(
                f"Cannot watch non-existent folder: {folder_path_to_watch}")

    def _scan_prediction_folder(self, folder_path):
        """
        Scans a folder for prediction JSONs and adds corresponding marks to the slider.

        This version is optimized to handle a very large number of files by
        dynamically decimating the visual markers to prevent GUI freezing, while
        still providing accurate progress feedback.
        """
        if not self.video_loader or self.num_frames is None or self.num_frames == 0:
            return
        if not self.seekbar:
            return

        try:
            path = Path(folder_path)
            # Use a robust regex to extract frame numbers from filenames
            json_pattern = re.compile(r'_(\d{9,})\.json$')

            # --- 1. Efficiently Scan and Parse All Relevant Frame Numbers ---
            all_frame_nums = []
            for f_name in os.listdir(path):
                # The check `self.video_results_folder.name in f_name` is kept for consistency
                if f_name.endswith(".json") and self.video_results_folder.name in f_name:
                    file_path = path / f_name
                    if self.prediction_start_timestamp:
                        try:
                            if file_path.stat().st_mtime < self.prediction_start_timestamp:
                                continue
                        except FileNotFoundError:
                            continue
                    match = json_pattern.search(f_name)
                    if match:
                        try:
                            # Convert via float to handle cases like "123.0"
                            frame_num = int(float(match.group(1)))
                            all_frame_nums.append(frame_num)
                        except (ValueError, IndexError):
                            continue  # Skip malformed numbers

            if not all_frame_nums:
                store = AnnotationStore.for_frame_path(
                    path / f"{path.name}_000000000.json")
                if store.store_path.exists():
                    all_frame_nums = sorted(store.iter_frames())
                if not all_frame_nums:
                    return

            all_frame_nums.sort()
            num_total_frames = len(all_frame_nums)

            # --- 2. Dynamic Marker Decimation Logic ---
            frames_to_mark = []
            # Define the threshold at which we start thinning the markers
            DECIMATION_THRESHOLD = 2000

            if num_total_frames < DECIMATION_THRESHOLD:
                # If the number of files is manageable, mark all of them
                frames_to_mark = all_frame_nums
            else:
                # If there are too many files, apply the decimation strategy
                step = 100 if num_total_frames > 10000 else 20
                # Add every Nth frame
                frames_to_mark = all_frame_nums[::step]

                # IMPORTANT: Always ensure the very last frame is included to show completion
                if all_frame_nums[-1] not in frames_to_mark:
                    frames_to_mark.append(all_frame_nums[-1])

            # --- 3. Update the GUI Efficiently ---
            if not frames_to_mark:
                return
            # Get existing markers once to avoid repeated calls inside the loop
            existing_predicted_vals = {
                mark.val for mark in self.seekbar.getMarks() if mark.mark_type == "predicted"
            }

            # Block signals to prevent the UI from trying to update thousands of times
            self.seekbar.blockSignals(True)

            new_marks_added = False
            for frame_num in frames_to_mark:
                if 0 <= frame_num < self.num_frames and frame_num not in existing_predicted_vals:
                    pred_mark = VideoSliderMark(
                        mark_type="predicted", val=frame_num)
                    self.seekbar.addMark(pred_mark)
                    new_marks_added = True

            # Re-enable signals and force a single repaint if we added anything
            self.seekbar.blockSignals(False)
            if new_marks_added:
                self.seekbar.update()

            # Update the progress bar and slider position to the latest actual frame
            latest_frame = all_frame_nums[-1]
            self.last_known_predicted_frame = max(
                self.last_known_predicted_frame, latest_frame)

            if self.num_frames > 0:
                progress = int(
                    (self.last_known_predicted_frame / self.num_frames) * 100)
                self._update_progress_bar(progress)

            # The original code moved the slider on every found frame.
            # This is not ideal for user experience. We will move it only once to the latest frame.
            if 0 <= latest_frame < self.num_frames:
                self.seekbar.removeMarksByType("prediction_progress")
                progress_mark = VideoSliderMark(
                    mark_type="prediction_progress",
                    val=latest_frame
                )
                self.seekbar.addMark(progress_mark)
                self._prediction_progress_mark = progress_mark
                if self.frame_number != latest_frame:
                    self.set_frame_number(latest_frame)
                self.seekbar.setValue(latest_frame)

        except Exception as e:
            logger.error(
                f"Error scanning prediction folder for slider marks: {e}", exc_info=True)

    @QtCore.Slot(str)
    def _handle_prediction_folder_change(self, path):
        logger.debug(f"Prediction folder changed: {path}. Re-scanning.")
        self._scan_prediction_folder(path)

    def _stop_prediction_folder_watcher(self):
        if self.prediction_progress_watcher:
            if self.prediction_progress_watcher.directories():
                self.prediction_progress_watcher.removePaths(
                    self.prediction_progress_watcher.directories())
            # self.prediction_progress_watcher.directoryChanged.disconnect(self._handle_prediction_folder_change)
            # self.prediction_progress_watcher = None # Or just keep it around
            logger.info("Prediction progress watcher stopped.")
        self.last_known_predicted_frame = -1  # Reset
        self.prediction_start_timestamp = 0.0
        if self.seekbar:
            self.seekbar.removeMarksByType("prediction_progress")
        self._prediction_progress_mark = None

    def _connect_worker_signals(self):
        """Connect worker signals to their respective slots safely."""
        worker = self.csv_worker
        thread = self.csv_thread

        worker.start_signal.connect(worker.run)
        worker.finished_signal.connect(
            lambda _result: self.place_preference_analyze_auto())

        worker.finished_signal.connect(self._on_csv_conversion_finished)
        worker.finished_signal.connect(thread.quit)
        worker.finished_signal.connect(worker.deleteLater)
        thread.finished.connect(self._cleanup_csv_worker)
        thread.finished.connect(thread.deleteLater)

        worker.progress_signal.connect(self._update_progress_bar)
        self.seekbar.removeMarksByType("predicted")  # Clear previous marks

    def _cleanup_csv_worker(self):
        """Clear references once the CSV conversion thread has fully finished."""
        try:
            if getattr(self, "csv_thread", None) and isinstance(self.csv_thread, QtCore.QThread):
                if self.csv_thread.isRunning():
                    return
        except Exception:
            pass
        self.csv_thread = None
        self.csv_worker = None

        if self._csv_conversion_queue:
            next_out, next_csv = self._csv_conversion_queue.pop(0)
            self._start_csv_conversion(next_out, next_csv)

    def tracks(self):
        """
        Track animals using the trained models for videos in a folder.
        The tracking results CSV files will be saved on the disk.
        """

        dlg = TrackDialog()
        config_file = None
        out_dir = None
        score_threshold = 0.15
        algo = None
        video_folder = None
        model_path = None
        top_k = 100
        video_multiframe = 1
        display_mask = False

        if self.video_file is not None:
            self.video_folder = str(Path(self.video_file).parent)
            dlg.inputVideoFileLineEdit.setText(self.video_folder)

        if dlg.exec_():
            config_file = dlg.config_file
            score_threshold = dlg.score_threshold
            algo = dlg.algo
            out_dir = dlg.out_dir
            video_folder = dlg.video_folder
            model_path = dlg.trained_model

        if video_folder is None:
            return

        # Get all video files in the folder
        video_files = get_video_files(video_folder)

        if not video_files:
            QtWidgets.QMessageBox.about(
                self, "No videos found", "No video files found in the specified folder.")
            return

        from annolid.inference.predict import Segmentor
        dataset_dir = str(Path(config_file).parent)
        segmentor = Segmentor(dataset_dir, model_path, score_threshold)

        for video_file in video_files:
            out_video_file = f"tracked_{Path(video_file).name}"

            if config_file is None and algo != "Predictions":
                return

            # Convert annolid predicted json files to a single tracked csv file
            if algo == 'Predictions':
                self.video_folder = video_folder
                self.convert_json_to_tracked_csv()

            if algo == 'Detectron2':
                try:
                    import detectron2
                except ImportError:
                    QtWidgets.QMessageBox.about(
                        self, "Detectron2 is not installed", "Please check the docs and install it.")
                    return
                out_folder = Path(video_file).with_suffix('')
                if out_folder.exists():
                    QtWidgets.QMessageBox.about(
                        self, f"Your folder {str(out_folder)} is not empty.", "Please backup your data or rename it.")
                # TODO: What's the best way to run multiple videos on different CPU, GPU, and MPS devices?

                # try:
                #     self.pred_worker = FlexibleWorker(
                #         function=segmentor.on_video, video_path=video_file)
                #     self.pred_worker.moveToThread(self.seg_pred_thread)
                #     self.pred_worker.start.connect(self.pred_worker.run)
                #     self.pred_worker.start.emit()
                #     self.pred_worker.finished.connect(
                #         self.seg_pred_thread.quit)
                #     logger.info(f"Start segmenting video {video_file}")
                #     out_result_dir = Path(video_file).with_suffix('')
                # except Exception as e:
                #     logger.info(e)
                logger.info(f"Start segmenting video {video_file}")
                out_result_dir = segmentor.on_video(video_file)
                # QtWidgets.QMessageBox.about(
                #     self, "Running", f"Results will be saved to folder: {out_result_dir} Please do not close Annolid GUI")
                self.importDirImages(out_result_dir)
                self.statusBar().showMessage(self.tr(f"Tracking..."))

            if algo == 'YOLACT':
                if not torch.cuda.is_available():
                    QtWidgets.QMessageBox.about(
                        self, "No GPU available", "At least one GPU is required to train models.")
                    return

                subprocess.Popen([
                    'annolid-track',
                    f'--trained_model={model_path}',
                    f'--config={config_file}',
                    f'--score_threshold={score_threshold}',
                    f'--top_k={top_k}',
                    f'--video_multiframe={video_multiframe}',
                    f'--video={video_file}|{out_video_file}',
                    '--mot',
                    f'--display_mask={display_mask}'
                ])

                if out_dir is None:
                    out_runs_dir = Path(__file__).parent.parent / 'runs'
                else:
                    out_runs_dir = Path(out_dir) / \
                        Path(config_file).name / 'runs'

                out_runs_dir.mkdir(exist_ok=True, parents=True)

                QtWidgets.QMessageBox.about(
                    self, "Started", f"Results are in folder: {str(out_runs_dir)}")
                self.statusBar().showMessage(self.tr(f"Tracking..."))

        QtWidgets.QMessageBox.about(
            self, "Completed", "Batch processing of videos is complete.")
        self.statusBar().showMessage(
            self.tr(f"Tracking completed for all videos in the folder."))

    def models(self):
        """
        Train a model with the provided dataset and selected params
        """

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

        if config_file is None:
            return
        if algo == 'YOLO':
            data_config = self.yolo_training_manager.prepare_data_config(
                config_file)
            if data_config is None:
                return
            self.yolo_training_manager.start_training(
                yolo_model_file=yolo_model_file,
                model_path=model_path,
                data_config_path=data_config,
                epochs=epochs,
                image_size=image_size,
                out_dir=out_dir,
            )

        elif algo == 'YOLACT':
            # start training models
            if not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(self,
                                            "Not GPU available",
                                            "At least one GPU  is required to train models.")
                return

            subprocess.Popen(['annolid-train',
                              f'--config={config_file}',
                              f'--batch_size={batch_size}'])

            if out_dir is None:
                out_runs_dir = Path(__file__).parent.parent / 'runs'
            else:
                out_runs_dir = Path(out_dir) / Path(config_file).name / 'runs'

            out_runs_dir.mkdir(exist_ok=True, parents=True)
            process = start_tensorboard(log_dir=out_runs_dir)
            QtWidgets.QMessageBox.about(self,
                                        "Started",
                                        f"Results are in folder: \
                                         {str(out_runs_dir)}")

        elif algo == 'MaskRCNN':
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor
            dataset_dir = str(Path(config_file).parent)
            segmentor = Segmentor(dataset_dir, out_dir,
                                  max_iterations=max_iterations,
                                  batch_size=batch_size,
                                  model_pth_path=model_path
                                  )
            out_runs_dir = segmentor.out_put_dir
            process = start_tensorboard(log_dir=out_runs_dir)
            try:
                self.seg_train_thread.start()
                train_worker = FlexibleWorker(task_function=segmentor.train)
                train_worker.moveToThread(self.seg_train_thread)
                train_worker.start_signal.connect(train_worker.run)
                train_worker.start_signal.emit()
            except Exception:
                segmentor.train()

            QtWidgets.QMessageBox.about(self,
                                        "Started.",
                                        f"Training in background... \
                                        Results will be saved to folder: \
                                         {str(out_runs_dir)} \
                                         Please do not close Annolid GUI."
                                        )
            self.statusBar().showMessage(
                self.tr(f"Training..."))

    def handle_uniq_label_list_selection_change(self):
        selected_items = self.uniqLabelList.selectedItems()
        if self.seekbar is None:
            return
        if selected_items:
            self.add_highlighted_mark()
        else:
            self.add_highlighted_mark(mark_type='event_end',
                                      color='red')

    def _estimate_recording_time(self, frame_number: int) -> Optional[float]:
        """Approximate the recording timestamp (seconds) for a frame."""
        if self.fps and self.fps > 0:
            return frame_number / float(self.fps)
        # Fallback to NTSC-like default if FPS is not known
        return frame_number / 29.97 if frame_number is not None else None

    def record_behavior_event(
        self,
        behavior: str,
        event_label: str,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        trial_time: Optional[float] = None,
        subject: Optional[str] = None,
        modifiers: Optional[Iterable[str]] = None,
        highlight: bool = True,
    ) -> Optional[BehaviorEvent]:
        if frame_number is None:
            frame_number = self.frame_number
        if timestamp is None:
            timestamp = self._estimate_recording_time(frame_number)
        if trial_time is None:
            trial_time = timestamp

        auto_subject = False
        if subject is None:
            subject = self._subject_from_selected_shape()
            auto_subject = subject is not None
        if subject is None:
            subject = self._current_subject_name()

        if modifiers is None:
            modifiers = self._selected_modifiers_for_behavior(behavior)
        if not modifiers:
            modifiers = self._default_modifiers_for_behavior(behavior)

        category_label: Optional[str] = None
        if self.project_schema:
            behavior_def = self.project_schema.behavior_map().get(behavior)
            if behavior_def:
                if behavior_def.category_id:
                    category = self.project_schema.category_map().get(
                        behavior_def.category_id)
                    if category:
                        category_label = category.name or category.id

        event = self.behavior_controller.record_event(
            behavior,
            event_label,
            frame_number,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            modifiers=modifiers,
            category=category_label,
            highlight=highlight,
        )
        if event is None:
            logger.warning(
                "Unrecognized behavior event label '%s' for '%s'.",
                event_label,
                behavior,
            )
            return None

        if auto_subject:
            self.statusBar().showMessage(
                self.tr(
                    "Auto-selected subject '%s' from polygon selection") % subject,
                2500,
            )

        self.pinned_flags.setdefault(behavior, False)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.append_event(event, fps=fps_for_log)
        return event

    def _populate_behavior_controls_from_schema(
        self, schema: Optional[ProjectSchema]
    ) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return

        self._behavior_modifier_state.clear()

        if schema is None:
            self.behavior_controls_widget.clear()
            self._active_subject_name = None
            return

        stored_subject = self._active_subject_name or self.settings.value(
            "behavior/last_subject", type=str)
        subjects = list(schema.subjects)
        self.behavior_controls_widget.set_subjects(
            subjects, selected=stored_subject)
        self._active_subject_name = self.behavior_controls_widget.current_subject()

        self.behavior_controls_widget.set_modifiers(list(schema.modifiers))
        for behavior in schema.behaviors:
            if behavior.modifier_ids:
                self._behavior_modifier_state[behavior.code] = set(
                    behavior.modifier_ids)

        self.behavior_controls_widget.set_category_badge(None, None)
        self.behavior_controls_widget.set_modifier_states(
            [],
            allowed=self._modifier_ids_from_schema(),
        )
        self.behavior_controls_widget.show_warning(None)
        self._update_modifier_controls_for_behavior(self.event_type)

    def _modifier_ids_from_schema(self) -> Set[str]:
        if not self.project_schema:
            return set()
        return {
            modifier.id
            for modifier in self.project_schema.modifiers
            if modifier.id
        }

    def _allowed_modifiers_for_behavior(self, behavior: Optional[str]) -> Set[str]:
        if not self.project_schema:
            return set()
        schema_modifiers = self._modifier_ids_from_schema()
        if not behavior:
            return schema_modifiers
        behavior_def = self.project_schema.behavior_map().get(behavior)
        if behavior_def and behavior_def.modifier_ids:
            return {modifier_id for modifier_id in behavior_def.modifier_ids if modifier_id in schema_modifiers}
        return schema_modifiers

    def _update_behavior_conflict_warning(self, behavior: Optional[str]) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return
        if not behavior or not self.project_schema:
            self.behavior_controls_widget.show_warning(None)
            return

        behavior_def = self.project_schema.behavior_map().get(behavior)
        if not behavior_def or not behavior_def.exclusive_with:
            self.behavior_controls_widget.show_warning(None)
            return

        active_conflicts = [
            name
            for name, value in (self.flags_controller.pinned_flags or {}).items()
            if value and name in behavior_def.exclusive_with and name != behavior
        ]
        if active_conflicts:
            message = self.tr(
                "Behavior '%s' excludes %s. Stop them before recording."
            ) % (
                behavior,
                ", ".join(sorted(active_conflicts)),
            )
            self.behavior_controls_widget.show_warning(message)
            self.statusBar().showMessage(message, 4000)
        else:
            self.behavior_controls_widget.show_warning(None)

    def _update_modifier_controls_for_behavior(self, behavior: Optional[str]) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return

        schema = self.project_schema
        if schema is None:
            self.behavior_controls_widget.clear()
            return

        allowed_modifiers = self._allowed_modifiers_for_behavior(behavior)
        if not behavior:
            self.behavior_controls_widget.set_category_badge(None, None)
            self.behavior_controls_widget.set_modifier_states(
                [],
                allowed=allowed_modifiers,
            )
            self.behavior_controls_widget.show_warning(None)
            return

        behavior_def = schema.behavior_map().get(behavior)
        category_label: Optional[str] = None
        category_color: Optional[str] = None
        if behavior_def and behavior_def.category_id:
            category = schema.category_map().get(behavior_def.category_id)
            if category:
                category_label = category.name or category.id
                category_color = category.color

        selected_modifiers = self._behavior_modifier_state.get(behavior)
        if selected_modifiers is None:
            if behavior_def and behavior_def.modifier_ids:
                selected_modifiers = set(
                    modifier_id for modifier_id in behavior_def.modifier_ids if modifier_id in allowed_modifiers)
                self._behavior_modifier_state[behavior] = set(
                    selected_modifiers)
            else:
                selected_modifiers = set()
        else:
            selected_modifiers = {
                modifier_id for modifier_id in selected_modifiers if modifier_id in allowed_modifiers}
            self._behavior_modifier_state[behavior] = set(selected_modifiers)
            if not selected_modifiers and behavior_def and behavior_def.modifier_ids:
                selected_modifiers = {
                    modifier_id for modifier_id in behavior_def.modifier_ids if modifier_id in allowed_modifiers}
                self._behavior_modifier_state[behavior] = set(
                    selected_modifiers)

        self.behavior_controls_widget.set_category_badge(
            category_label,
            category_color,
        )
        self.behavior_controls_widget.set_modifier_states(
            selected_modifiers,
            allowed=allowed_modifiers,
        )
        self._update_behavior_conflict_warning(behavior)

    def _on_active_subject_changed(self, subject_name: str) -> None:
        subject_name = subject_name.strip()
        if subject_name:
            self._active_subject_name = subject_name
            self.settings.setValue("behavior/last_subject", subject_name)

    def _on_modifier_toggled(self, modifier_id: str, state: bool) -> None:
        behavior = self.event_type
        if not behavior:
            return
        modifier_set = self._behavior_modifier_state.setdefault(
            behavior, set())
        if state:
            modifier_set.add(modifier_id)
        else:
            modifier_set.discard(modifier_id)

    def _subject_from_selected_shape(self) -> Optional[str]:
        selected = getattr(self.canvas, "selectedShapes", None)
        if not selected:
            return None
        label = selected[0].label
        if not label:
            return None
        label_name = str(label).strip()
        if not label_name:
            return None
        if self.project_schema:
            candidates = {
                subj.name: subj.name for subj in self.project_schema.subjects}
            candidates.update(
                {subj.id: subj.name or subj.id for subj in self.project_schema.subjects})
            lowered = {key.lower(): value for key, value in candidates.items()}
            match = lowered.get(label_name.lower())
            if match:
                return match
        return label_name

    def _current_subject_name(self) -> str:
        if self._active_subject_name:
            return self._active_subject_name
        if hasattr(self, "behavior_controls_widget"):
            subject = self.behavior_controls_widget.current_subject()
            if subject:
                self._active_subject_name = subject
                return subject
        if self.project_schema and self.project_schema.subjects:
            subject = self.project_schema.subjects[0]
            return subject.name or subject.id or "Subject 1"
        return "Subject 1"

    def _selected_modifiers_for_behavior(self, behavior: Optional[str]) -> List[str]:
        if not behavior:
            return []
        selected = self._behavior_modifier_state.get(behavior)
        if selected:
            return list(selected)
        defaults = self._default_modifiers_for_behavior(behavior)
        if defaults:
            allowed = self._allowed_modifiers_for_behavior(behavior)
            if allowed:
                defaults = [
                    modifier_id for modifier_id in defaults if modifier_id in allowed]
        if defaults:
            self._behavior_modifier_state[behavior] = set(defaults)
            return list(defaults)
        return []

    def _default_modifiers_for_behavior(self, behavior: Optional[str]) -> List[str]:
        if not behavior or not self.project_schema:
            return []
        behavior_def = self.project_schema.behavior_map().get(behavior)
        if behavior_def and behavior_def.modifier_ids:
            return list(behavior_def.modifier_ids)
        return []

    def _jump_to_frame_from_log(self, frame: int) -> None:
        if self.seekbar is None or self.num_frames is None:
            return
        target = max(0, min(frame, self.num_frames - 1))
        if self.seekbar.value() != target:
            self.seekbar.setValue(target)
        else:
            self.set_frame_number(target)

    def undo_last_behavior_event(self) -> None:
        event = self.behavior_controller.pop_last_event()
        if event is None:
            return
        self.behavior_log_widget.remove_event(event.mark_key)
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        self.canvas.setBehaviorText(None)

    def _clear_behavior_events_from_log(self) -> None:
        if not self.behavior_controller.events_count:
            self.behavior_log_widget.clear()
            return
        self.behavior_controller.clear_behavior_data()
        self.behavior_log_widget.clear()
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        self.canvas.setBehaviorText(None)
        if self.pinned_flags:
            for behavior in list(self.pinned_flags.keys()):
                self.pinned_flags[behavior] = False
            self.loadFlags(self.pinned_flags)

    def add_highlighted_mark(self, val=None,
                             mark_type=None,
                             color=None,
                             init_load=False):
        """Add a non-behavior highlight mark to the slider."""
        if self.seekbar is None:
            return None

        frame_val = self.frame_number if val is None else int(val)
        return self.behavior_controller.add_generic_mark(
            frame_val,
            mark_type=mark_type,
            color=color,
            init_load=init_load,
        )

    def remove_highlighted_mark(self):
        if self.seekbar is None:
            return

        removed_behavior_keys: List[Tuple[int, str, str]] = []

        if self.behavior_controller.highlighted_mark is not None:
            if self.isPlaying:
                self.togglePlay()
            removed = self.behavior_controller.remove_highlighted_mark()
            if removed:
                if removed[0] == "behavior":
                    removed_behavior_keys.append(
                        removed[1])  # type: ignore[index]
                if self.event_type in self.pinned_flags:
                    self.pinned_flags[self.event_type] = False
        elif self.seekbar.isMarkedVal(self.frame_number):
            removed = self.behavior_controller.remove_marks_at_value(
                self.frame_number)
            for kind, key in removed:
                if kind == "behavior":
                    removed_behavior_keys.append(key)  # type: ignore[arg-type]
            if removed and self.event_type in self.pinned_flags:
                self.pinned_flags[self.event_type] = False
        else:
            current_val = self.seekbar.value()
            removed_any = False
            local_removed_keys: List[Tuple[int, str, str]] = []
            for mark in list(self.seekbar.getMarks()):
                if mark.val == current_val:
                    result = self.behavior_controller.remove_mark_instance(
                        mark)
                    removed_any = removed_any or bool(result)
                    if result and result[0] == "behavior":
                        # type: ignore[arg-type]
                        local_removed_keys.append(result[1])
            if removed_any:
                self.seekbar.setTickMarks()
                if self.event_type in self.pinned_flags:
                    self.pinned_flags[self.event_type] = False
                removed_behavior_keys.extend(local_removed_keys)

        for key in removed_behavior_keys:
            self.behavior_log_widget.remove_event(key)
        self.canvas.setBehaviorText(None)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.seekbar is not None:
            if event.key() == Qt.Key_Right:
                next_pos = self.seekbar.value()+1
                self.seekbar.setValue(next_pos)
                self.seekbar.valueChanged.emit(next_pos)
            elif event.key() == Qt.Key_Left:
                prev_pos = self.seekbar.value()-1
                self.seekbar.setValue(prev_pos)
                self.seekbar.valueChanged.emit(prev_pos)
            elif event.key() == Qt.Key_0:
                self.seekbar.setValue(0)
            elif event.key() == Qt.Key_Space:
                self.togglePlay()
            elif event.key() == Qt.Key_S:
                if self.event_type is None:
                    self.add_highlighted_mark(
                        self.frame_number,
                        mark_type=self._config['events']["start"],
                    )
                else:
                    self.record_behavior_event(
                        self.event_type, "start", frame_number=self.frame_number)
            elif event.key() == Qt.Key_E:
                if self.event_type is None:
                    self.add_highlighted_mark(
                        self.frame_number,
                        mark_type=self._config['events']["end"],
                        color='red',
                    )
                else:
                    self.record_behavior_event(
                        self.event_type, "end", frame_number=self.frame_number)
                    self.flags_controller.end_flag(
                        self.event_type, record_event=False)
            elif event.key() == Qt.Key_R:
                self.remove_highlighted_mark()
            elif event.key() == Qt.Key_Q:
                self.seekbar.setValue(self.seekbar._val_max)
            elif event.key() == Qt.Key_1 or event.key() == Qt.Key_I:
                self.update_step_size(1)
            elif event.key() == Qt.Key_2 or event.key() == Qt.Key_F:
                self.update_step_size(self.step_size + 10)
            elif event.key() == Qt.Key_B:
                self.update_step_size(self.step_size - 10)
            elif event.key() == Qt.Key_M:
                self.update_step_size(self.step_size - 1)
            elif event.key() == Qt.Key_P:
                self.update_step_size(self.step_size + 1)
            else:
                event.ignore()

    def saveTimestampList(self):
        # Open file dialog to get file path
        default_timestamp_csv_file = str(
            os.path.dirname(self.filename)) + '_timestamps.csv'
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDefaultSuffix('.csv')
        file_path, _ = file_dialog.getSaveFileName(self, "Save Timestamps",
                                                   default_timestamp_csv_file,
                                                   "CSV files (*.csv)")

        if file_path:
            rows = self.behavior_controller.export_rows(
                timestamp_fallback=lambda evt: self._estimate_recording_time(
                    evt.frame)
            )
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Trial time', 'Recording time',
                                'Subject', 'Behavior', 'Event'])
                for row in rows:
                    writer.writerow(row)

            QtWidgets.QMessageBox.information(
                self, "Timestamps saved", "Timestamps saved successfully!")

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        return super().keyReleaseEvent(event)

    def show_behavior_time_budget_dialog(self) -> None:
        """Summarise recorded behavior events using the time-budget report."""
        rows = self.behavior_controller.export_rows(
            timestamp_fallback=lambda evt: self._estimate_recording_time(
                evt.frame)
        )
        if not rows:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Behavior Time Budget"),
                self.tr("No behavior events are available to summarise."),
            )
            return

        data_rows = []
        local_warnings = []
        for trial_time, recording_time, subject, behavior, event_label in rows:
            if recording_time is None:
                local_warnings.append(
                    self.tr(
                        "Skipping '%s' event for behavior '%s' because no timestamp could be determined."
                    )
                    % (event_label or "?", behavior or "?")
                )
                continue

            data_rows.append(
                {
                    "trial time": trial_time,
                    "recording time": recording_time,
                    "subject": (subject or "").strip(),
                    "behavior": (behavior or "").strip(),
                    "event": (event_label or "").strip(),
                }
            )

        if not data_rows:
            message = self.tr(
                "No timestamped events remain after filtering out rows without timing information."
            )
            if local_warnings:
                message += "\n\n" + \
                    self.tr("Warnings:\n") + "\n".join(local_warnings)
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Behavior Time Budget"),
                message,
            )
            return

        try:
            summary, compute_warnings = compute_time_budget(data_rows)
        except TimeBudgetComputationError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Behavior Time Budget"),
                self.tr("Unable to compute the time budget:\n%s") % exc,
            )
            return

        warnings: List[str] = local_warnings + compute_warnings
        schema_for_dialog = self.project_schema
        category_summary: List[Tuple[str, float, int]] = []
        if schema_for_dialog is not None and summary:
            try:
                category_summary = summarize_by_category(
                    summary, schema_for_dialog)
            except Exception as exc:
                logger.warning("Failed to summarize categories: %s", exc)
                category_summary = []

        if not summary:
            message = self.tr(
                "No completed start/end pairs were found for the current behavior events."
            )
            if warnings:
                message += "\n\n" + \
                    self.tr("Warnings:\n") + "\n".join(warnings)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Behavior Time Budget"),
                message,
            )
            return

        report_text = format_time_budget_table(
            summary, schema=schema_for_dialog)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Behavior Time Budget"))
        layout = QtWidgets.QVBoxLayout(dialog)

        report_view = QtWidgets.QPlainTextEdit()
        report_view.setReadOnly(True)
        fixed_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont)
        report_view.setFont(fixed_font)
        report_view.setPlainText(report_text)
        layout.addWidget(report_view)

        if warnings:
            warning_label = QtWidgets.QLabel(self.tr("Warnings:"))
            warning_label.setStyleSheet("font-weight: bold;")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

            warning_view = QtWidgets.QPlainTextEdit()
            warning_view.setReadOnly(True)
            warning_view.setPlainText("\n".join(warnings))
            warning_view.setMaximumHeight(140)
            warning_view.setStyleSheet("background-color: #fff4e5;")
            layout.addWidget(warning_view)

        if schema_for_dialog is not None and category_summary:
            category_label = QtWidgets.QLabel(self.tr("Category Summary:"))
            category_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(category_label)
            category_view = QtWidgets.QPlainTextEdit()
            category_view.setReadOnly(True)
            category_view.setFont(QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.FixedFont))
            category_view.setPlainText(
                format_category_summary(category_summary))
            category_view.setMaximumHeight(160)
            layout.addWidget(category_view)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        save_button = button_box.addButton(
            self.tr("Save CSV…"), QtWidgets.QDialogButtonBox.ActionRole
        )

        def _save_csv() -> None:
            default_name = "behavior_time_budget.csv"
            default_path = str(Path(self.video_file).with_suffix(
                ".time_budget.csv")) if self.video_file else default_name
            path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Save Time-Budget CSV"),
                default_path,
                self.tr("CSV files (*.csv)"),
            )
            if not path_str:
                return
            try:
                output_path = Path(path_str)
                write_time_budget_csv(
                    summary, output_path, schema=schema_for_dialog)
                if schema_for_dialog is not None and category_summary:
                    category_path = output_path.with_name(
                        output_path.stem + "_categories" + output_path.suffix
                    )
                    with category_path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(
                            ["Category", "TotalSeconds", "Occurrences"])
                        for name, total, occurrences in category_summary:
                            writer.writerow(
                                [name, f"{total:.6f}", occurrences])
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Behavior Time Budget"),
                    self.tr("Failed to save CSV:\n%s") % exc,
                )
            else:
                self.statusBar().showMessage(
                    self.tr(
                        "Time-budget exported to %s") % Path(path_str).name, 4000
                )

        save_button.clicked.connect(_save_csv)
        layout.addWidget(button_box)

        dialog.resize(720, 520)
        dialog.exec_()

    def togglePlay(self):
        if self.isPlaying:
            self.stopPlaying()
            self.update_step_size(1)
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.playButton.setText("Play")
        else:
            self.startPlaying()
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
            self.playButton.setText("Pause")

    def set_frame_number(self, frame_number):
        if frame_number >= self.num_frames or frame_number < 0:
            return
        self.frame_number = frame_number
        self._update_audio_playhead(frame_number)
        if self.isPlaying and not self._suppress_audio_seek:
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.play(start_frame=frame_number)
        self.filename = self.video_results_folder / \
            f"{str(self.video_results_folder.name)}_{self.frame_number:09}.png"
        self.current_frame_time_stamp = self.video_loader.get_time_stamp()
        if self.frame_loader is not None:
            self.frame_loader.request(frame_number)
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)

    def load_tracking_results(self, cur_video_folder, video_filename):
        self.tracking_data_controller.load_tracking_results(
            Path(cur_video_folder), video_filename
        )

    def is_behavior_active(self, frame_number, behavior):
        """Checks if a behavior is active at a given frame."""
        return self.behavior_controller.is_behavior_active(frame_number, behavior)

    def _load_deeplabcut_results(
        self,
        frame_number: int,
        is_multi_animal: Optional[bool] = None,
    ) -> None:
        """
        Load DeepLabCut tracking results for a given frame and convert them into shape objects.

        This method extracts x, y coordinates for each body part and, if applicable, for each animal.
        It then creates shape objects for visualization.

        Args:
            frame_number (int): The index of the frame to extract tracking data from.
            is_multi_animal (bool, optional): Force multi-animal parsing. Auto-detected when None.

        Notes:
            - Assumes self._df_deeplabcut is a multi-index Pandas DataFrame.
            - Multi-animal mode expects an 'animal' level in the column index.
            - Logs warnings for missing data instead of failing.

        Raises:
            KeyError: If expected columns are missing.
            Exception: For unexpected errors during shape extraction.
        """
        if self._df_deeplabcut is None or self._df_deeplabcut.empty:
            return

        if is_multi_animal is None:
            is_multi_animal = getattr(
                self, "_df_deeplabcut_multi_animal", False)

        try:
            row = self._df_deeplabcut.loc[frame_number]
        except KeyError:
            if 0 <= frame_number < len(self._df_deeplabcut.index):
                row = self._df_deeplabcut.iloc[frame_number]
            else:
                logger.debug(
                    "Frame %s is outside the DeepLabCut table bounds (%s rows).",
                    frame_number,
                    len(self._df_deeplabcut.index),
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Unexpected error accessing DeepLabCut frame %s: %s",
                frame_number,
                exc,
            )
            return

        shapes = []
        try:
            if self._df_deeplabcut_scorer is None:
                self._df_deeplabcut_scorer = self._df_deeplabcut.columns.get_level_values(
                    0)[0]

            if self._df_deeplabcut_animal_ids is None:
                if is_multi_animal:
                    self._df_deeplabcut_animal_ids = self._df_deeplabcut.columns.get_level_values(
                        'animal').unique()
                else:
                    self._df_deeplabcut_animal_ids = [
                        None]  # Single-animal mode

            if self._df_deeplabcut_bodyparts is None:
                self._df_deeplabcut_bodyparts = self._df_deeplabcut.columns.get_level_values(
                    'bodyparts').unique()

            for animal_id in self._df_deeplabcut_animal_ids:
                for bodypart in self._df_deeplabcut_bodyparts:
                    x_col = (self._df_deeplabcut_scorer, animal_id, bodypart, 'x') if is_multi_animal else (
                        self._df_deeplabcut_scorer, bodypart, 'x')
                    y_col = (self._df_deeplabcut_scorer, animal_id, bodypart, 'y') if is_multi_animal else (
                        self._df_deeplabcut_scorer, bodypart, 'y')

                    x, y = row.get(x_col, None), row.get(y_col, None)
                    if pd.notna(x) and pd.notna(y):
                        shape = Shape(label=bodypart,
                                      shape_type='point', visible=True)
                        shape.addPoint((x, y))
                        shapes.append(shape)

        except KeyError as e:
            logger.warning(f"Missing columns in DeepLabCut results: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading DeepLabCut results: {e}")

        self.loadShapes(shapes)

    def _configure_project_schema_for_video(self, video_path: str) -> None:
        """Load optional project schema metadata located near the video."""
        schema_path = find_schema_near_video(Path(video_path))
        schema: Optional[ProjectSchema] = None
        if schema_path:
            try:
                schema = load_project_schema(schema_path)
                warnings = validate_project_schema(schema)
                for warning in warnings:
                    logger.warning("Schema warning (%s): %s",
                                   schema_path.name, warning)
            except Exception as exc:
                logger.error("Failed to load project schema %s: %s",
                             schema_path, exc)
        if schema_path:
            self.project_schema_path = schema_path
        else:
            default_path = Path(video_path).with_suffix(
                "") / DEFAULT_SCHEMA_FILENAME
            self.project_schema_path = default_path
        if schema is None:
            logger.debug(
                "No project schema found near %s; using default configuration.",
                video_path,
            )
            schema = default_schema()
        self.project_schema = schema
        self.behavior_controller.configure_from_schema(schema)
        self._populate_behavior_controls_from_schema(schema)
        self._update_modifier_controls_for_behavior(self.event_type)

    def _load_behavior(self, behavior_csv_file: str) -> None:
        """Load behavior events from CSV and populate the slider timeline.

        Args:
            behavior_csv_file (str): Path to the CSV file containing behavior data.
        """
        df_behaviors = pd.read_csv(behavior_csv_file)
        required_columns = {"Recording time", "Event", "Behavior"}

        if not required_columns.issubset(df_behaviors.columns):
            # Fall back to DeepLabCut multi-index CSVs.
            del df_behaviors
            if not self._load_deeplabcut_table(behavior_csv_file):
                logger.debug(
                    "Skipped loading '%s' because it is neither a behavior log nor a DeepLabCut export.",
                    Path(behavior_csv_file).name,
                )
            return

        rows: List[Tuple[float, float, Optional[str], str, str]] = []

        for _, row in df_behaviors.iterrows():
            raw_timestamp = row.get("Recording time")
            event_label = str(row.get("Event"))
            behavior = str(row.get("Behavior"))
            raw_subject = row.get("Subject")
            raw_trial_time = row.get("Trial time")

            try:
                timestamp_value = float(raw_timestamp)
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to convert timestamp '%s' for behavior '%s'.",
                    raw_timestamp,
                    behavior,
                )
                continue

            trial_time_value: Optional[float]
            try:
                trial_time_value = float(raw_trial_time) if raw_trial_time is not None and pd.notna(
                    raw_trial_time) else None
            except (TypeError, ValueError):
                trial_time_value = None

            subject_value = None
            if raw_subject is not None and pd.notna(raw_subject):
                subject_value = str(raw_subject)

            rows.append((
                trial_time_value,
                timestamp_value,
                subject_value,
                behavior,
                event_label,
            ))

        fps = self.fps if self.fps and self.fps > 0 else 29.97

        def time_to_frame(time_value: float) -> int:
            return int(round(time_value * fps))

        self.behavior_controller.load_events_from_rows(
            rows,
            time_to_frame=time_to_frame,
        )
        self.behavior_controller.attach_slider(self.seekbar)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.set_events(
            list(self.behavior_controller.iter_events()),
            fps=fps_for_log,
        )
        self.pinned_flags.update(
            {behavior: False for behavior in self.behavior_controller.behavior_names})

    def _load_deeplabcut_table(self, behavior_csv_file: str) -> bool:
        """Load DeepLabCut tracking results stored as a multi-index CSV.

        Returns:
            bool: True if DeepLabCut data was successfully loaded, False otherwise.
        """
        try:
            df_dlc = pd.read_csv(
                behavior_csv_file,
                header=[0, 1, 2],
                index_col=0,
            )
        except (ValueError, pd.errors.ParserError) as exc:
            logger.debug(
                "Skipping %s: not a DeepLabCut multi-index CSV (%s).",
                Path(behavior_csv_file).name,
                exc,
            )
            self._df_deeplabcut = None
            self._df_deeplabcut_columns = None
            self._df_deeplabcut_scorer = None
            self._df_deeplabcut_bodyparts = None
            self._df_deeplabcut_animal_ids = None
            self._df_deeplabcut_multi_animal = False
            return False

        nlevels = df_dlc.columns.nlevels
        if nlevels == 4:
            expected_names = ["scorer", "animal", "bodyparts", "coords"]
            self._df_deeplabcut_multi_animal = True
        elif nlevels == 3:
            expected_names = ["scorer", "bodyparts", "coords"]
            self._df_deeplabcut_multi_animal = False
        else:
            logger.debug(
                "Skipping %s: expected 3 or 4 column levels, found %s.",
                Path(behavior_csv_file).name,
                nlevels,
            )
            self._df_deeplabcut = None
            self._df_deeplabcut_columns = None
            self._df_deeplabcut_scorer = None
            self._df_deeplabcut_bodyparts = None
            self._df_deeplabcut_animal_ids = None
            self._df_deeplabcut_multi_animal = False
            return False

        df_dlc.columns = df_dlc.columns.set_names(expected_names)

        index_numeric = pd.to_numeric(df_dlc.index, errors="coerce")
        if index_numeric.isna().any():
            logger.debug(
                "DeepLabCut table %s has non-numeric frame index; using positional indices.",
                Path(behavior_csv_file).name,
            )
            df_dlc.reset_index(drop=True, inplace=True)
        else:
            df_dlc.index = index_numeric.astype(int)

        self._df_deeplabcut = df_dlc
        self._df_deeplabcut_columns = df_dlc.columns
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        return True

    def open_project_schema_dialog(self) -> None:
        """Open the schema editor dialog and persist changes."""
        schema = self.project_schema or default_schema()
        dialog = ProjectDialog(schema, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_schema = dialog.get_schema()
            self.project_schema = new_schema
            self.behavior_controller.configure_from_schema(new_schema)
            self._populate_behavior_controls_from_schema(new_schema)
            self._update_modifier_controls_for_behavior(self.event_type)

            target_path = self.project_schema_path
            if target_path is None:
                default_dir = Path(self.video_file).with_suffix(
                    "") if self.video_file else Path.cwd()
                default_dir.mkdir(parents=True, exist_ok=True)
                default_path = default_dir / DEFAULT_SCHEMA_FILENAME
                path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    self.tr("Save Project Schema"),
                    str(default_path),
                    self.tr("Schema Files (*.json *.yaml *.yml)"),
                )
                if not path_str:
                    return
                target_path = Path(path_str)
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                save_project_schema(new_schema, target_path)
                self.project_schema_path = target_path
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Project Schema"),
                    self.tr("Failed to save schema:\n%s") % exc,
                )
            else:
                self.statusBar().showMessage(
                    self.tr("Project schema saved to %s") % target_path.name,
                    4000,
                )

    def _load_labels(self, labels_csv_file):
        """Load labels from the given CSV file."""
        self._df = pd.read_csv(labels_csv_file)
        self._df.rename(columns={'Unnamed: 0': 'frame_number'}, inplace=True)

    def _load_video(self, video_path):
        """Open a video for annotation frame by frame."""
        if not video_path:
            return
        self.openVideo(from_video_list=True, video_path=video_path)

    def open_youtube_video(self):
        """Launch the YouTube download dialog and open the selected video."""
        dialog = YouTubeVideoDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.downloaded_path:
            self.openVideo(from_video_list=True, video_path=str(dialog.downloaded_path))

    def handle_extracted_frames(self, dirpath):
        self.importDirImages(dirpath)

    def openVideo(self, _value=False,
                  from_video_list=False,
                  video_path=None,
                  programmatic_call=False
                  ):
        """open a video for annotaiton frame by frame

        Args:
            _value (bool, optional):  Defaults to False.
        """
        if not programmatic_call and (self.dirty or self.video_loader is not None):
            message_box = QtWidgets.QMessageBox()
            message_box.setWindowTitle(
                "Unsaved Changes or Closing the Existing Video")
            message_box.setText("The existing video will be closed,\n"
                                "and any unsaved changes may be lost.\n"
                                "Do you want to continue and open the new video?")
            message_box.setStandardButtons(
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            choice = message_box.exec()

            if choice == QtWidgets.QMessageBox.Ok:
                self.closeFile()
            elif choice == QtWidgets.QMessageBox.Cancel:
                return  # Cancel operation

        if not from_video_list:
            video_path = Path(self.filename).parent if self.filename else "."
            formats = ["*.*"]
            filters = self.tr(f"Video files {formats[0]}")
            video_filename = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr(f"{__appname__} - Choose Video"),
                str(video_path),
                filters,
            )
            if QT5:
                video_filename, _ = video_filename
        else:
            video_filename = video_path

        video_filename = str(video_filename)
        self.stepSizeWidget.setEnabled(True)

        if video_filename:
            cur_video_folder = Path(video_filename).parent
            self.video_results_folder = Path(video_filename).with_suffix('')

            self.video_results_folder.mkdir(
                exist_ok=True,
                parents=True
            )
            self.annotation_dir = self.video_results_folder
            self.video_file = video_filename
            try:
                suffix_lower = Path(video_filename).suffix.lower()
                # Support TIFF stacks by treating them as videos (one slice per frame)
                if suffix_lower in {'.tif', '.tiff'} or video_filename.lower().endswith(('.ome.tif', '.ome.tiff')):
                    self.video_loader = videos.TiffStackVideo(video_filename)
                else:
                    self.video_loader = videos.CV2Video(video_filename)
            except Exception:
                QtWidgets.QMessageBox.about(self,
                                            "Not a valid media file",
                                            f"Please check and open a valid video or TIFF stack file.")
                self.video_file = None
                self.video_loader = None
                return
            # 3D viewer menu is always available regardless of media type
            self._configure_project_schema_for_video(video_filename)
            self.fps = self.video_loader.get_fps()
            self.num_frames = self.video_loader.total_frames()
            self.behavior_log_widget.set_fps(self.fps)
            if self.caption_widget is not None:
                self.caption_widget.set_video_context(
                    video_filename,
                    self.fps,
                    self.num_frames,
                )
            self._configure_audio_for_video(self.video_file, self.fps)
            if self.seekbar:
                self.statusBar().removeWidget(self.seekbar)
            if self.playButton:
                self.statusBar().removeWidget(self.playButton)
            if self.saveButton:
                self.statusBar().removeWidget(self.saveButton)
            self.seekbar = VideoSlider()
            self.behavior_controller.attach_slider(self.seekbar)
            self.seekbar.input_value.returnPressed.connect(self.jump_to_frame)
            self.seekbar.keyPress.connect(self.keyPressEvent)
            self.seekbar.keyRelease.connect(self.keyReleaseEvent)
            logger.info(f"Working on video:{self.video_file}.")
            logger.info(
                f"FPS: {self.fps}, Total number of frames: {self.num_frames}")

            self.seekbar.valueChanged.connect(lambda f: self.set_frame_number(
                self.seekbar.value()
            ))

            self.seekbar.setMinimum(0)
            self.seekbar.setMaximum(self.num_frames-1)
            self.seekbar.setEnabled(True)
            self.seekbar.resizeEvent()
            self.seekbar.setTooltipCallable(self.tooltip_callable)
            # Create a play button
            self.playButton = QtWidgets.QPushButton("Play", self)
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.playButton.clicked.connect(self.togglePlay)
            # create save button
            self.saveButton = QtWidgets.QPushButton("Save Timestamps", self)
            self.saveButton.clicked.connect(self.saveTimestampList)
            # Add the playback controls as permanent widgets so they stay visible
            self.statusBar().addPermanentWidget(self.playButton)
            self.statusBar().addPermanentWidget(self.seekbar, stretch=1)
            self.statusBar().addPermanentWidget(self.saveButton)

            # load the first frame
            self.set_frame_number(self.frame_number)

            self.actions.openNextImg.setEnabled(True)

            self.actions.openPrevImg.setEnabled(True)

            self.frame_loader.video_loader = self.video_loader

            self.frame_loader.moveToThread(self.frame_worker)

            if not self.frame_worker.isRunning():
                self.frame_worker.start(
                    priority=QtCore.QThread.IdlePriority)

            self.frame_loader.res_frame.connect(
                lambda qimage: self.image_to_canvas(
                    qimage, self.filename, self.frame_number)
            )
            # go over all the tracking csv files
            # use the first matched file with video name
            # and segmentation
            self.load_tracking_results(cur_video_folder, video_filename)

            if self.filename:  # Video successfully loaded
                self.open_segment_editor_action.setEnabled(True)
                # Load persisted segments for the new video
                self._load_segments_for_active_video()
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments(
                        self._current_video_defined_segments
                    )
                if not programmatic_call:
                    self._emit_live_frame_update()
                logger.info(
                    f"Video '{self.filename}' loaded. Segment definition enabled.")
            else:
                self.open_segment_editor_action.setEnabled(False)
                self._current_video_defined_segments = []
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments([])

    def jump_to_frame(self):
        """
        Jump to the specified frame number.

        Retrieves the frame number from the input field, validates it, and sets the frame
        number if it is within the valid range.

        If the entered frame number is not a valid integer or is out of range, it logs
        an informational message and notifies the user with a warning dialog.
        """
        try:
            input_frame_number = int(self.seekbar.input_value.text())
            if 0 <= input_frame_number < self.num_frames:
                self.set_frame_number(input_frame_number)
            else:
                logger.info(
                    f"Frame number {input_frame_number} is out of range.")
                # Notify the user about the error, for instance:
                QtWidgets.QMessageBox.warning(self, "Invalid Frame Number",
                                              f"{input_frame_number} is out of range.")
        except ValueError:
            logger.info(
                f"Invalid input: {self.seekbar.input_value.text()} is not a valid frame number.")
            QtWidgets.QMessageBox.warning(self, "Invalid Input",
                                          f"'{self.seekbar.input_value.text()}' is not a valid frame number.")
        except Exception as e:
            logger.error(f"Error while jumping to frame: {e}")

    def tooltip_callable(self, val):
        if self.behavior_controller.highlighted_mark is not None and self.frame_number == val:
            return f"Frame:{val},Time:{convert_frame_number_to_time(val)}"
        return ''

    def image_to_canvas(self, qimage, filename, frame_number):
        self.resetState()
        self.canvas.setEnabled(True)
        self.canvas.setPatchSimilarityOverlay(None)
        self._deactivate_pca_map()
        if isinstance(filename, str):
            filename = Path(filename)
        self.imagePath = str(filename.parent)
        self.filename = str(filename)
        self.image = qimage
        self.imageData = qimage
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage))
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()

        video_file_key_for_zoom = str(
            self.video_file) if self.video_file else str(self.filename)
        if not self._config["keep_prev_scale"]:
            # If "Keep Previous Scale" is OFF, always try to fit the new frame.
            self.adjustScale(initial=True)
        elif video_file_key_for_zoom in self.zoom_values:
            # If "Keep Previous Scale" is ON and we have a saved zoom state for this video, restore it.
            self.zoomMode = self.zoom_values[video_file_key_for_zoom][0]
            # setZoom updates widget and calls paintCanvas
            self.setZoom(self.zoom_values[video_file_key_for_zoom][1])
        else:
            # If "Keep Previous Scale" is ON, but no saved state for this video yet
            #  (e.g., first frame of this video loaded in session)
            # OR if it's the very first image/frame loaded in the entire application session.
            # is_first_image_ever_in_session = not self.zoom_values # Original labelme logic for this
            self.adjustScale(initial=True)

        # Store the current zoom state for this video file, so if we navigate away and back,
        # it can be restored (if keep_prev_scale is on)
        if video_file_key_for_zoom:  # Ensure we have a valid key
            self.zoom_values[video_file_key_for_zoom] = (
                self.zoomMode, self.zoomWidget.value())

        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness constrast values
        # dialog = BrightnessContrastDialog(
        #     imageData,
        #     self.onNewBrightnessContrast,
        #     parent=self,
        # )
        # brightness, contrast = self.brightnessContrast_values.get(
        #     self.filename, (None, None)
        # )
        # if self._config["keep_prev_brightness"] and self.recentFiles:
        #     brightness, _ = self.brightnessContrast_values.get(
        #         self.recentFiles[0], (None, None)
        #     )

    @staticmethod
    def _qimage_to_bytes(qimage: QtGui.QImage, fmt: str = "PNG"):
        """Serialize a QImage into raw bytes compatible with LabelMe utilities."""
        if qimage is None or qimage.isNull():
            return None

        buffer = QtCore.QBuffer()
        if not buffer.open(QtCore.QIODevice.WriteOnly):
            logger.warning("Unable to open buffer for QImage serialization.")
            return None

        succeeded = qimage.save(buffer, fmt)
        buffer.close()

        if not succeeded:
            logger.warning("Failed to serialize QImage to %s", fmt)
            return None

        return bytes(buffer.data())

    def brightnessContrast(self, value):
        """Run brightness/contrast dialog, converting QImage lazily when needed."""
        restore_image = None
        converted = False

        if isinstance(self.imageData, QtGui.QImage):
            image_bytes = self._qimage_to_bytes(self.imageData)
            if image_bytes is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Brightness/Contrast Unavailable"),
                    self.tr("Unable to prepare image data for adjustment."),
                )
                return
            restore_image = self.imageData
            self.imageData = image_bytes
            converted = True

        try:
            return super().brightnessContrast(value)
        finally:
            if converted:
                self.imageData = restore_image

    def adjustScale(self, initial=False):
        """Safely adjust zoom while handling cases with no active pixmap."""
        canvas_pixmap = getattr(self.canvas, "pixmap", None)
        if canvas_pixmap is None or canvas_pixmap.isNull():
            logger.debug("adjustScale skipped: canvas pixmap not ready.")
            return

        if not getattr(self, "filename", None):
            logger.debug("adjustScale skipped: no active filename.")
            return

        frame_number = getattr(self, "frame_number", None)
        filename = getattr(self, "filename", None)
        if frame_number is None or filename is None:
            logger.debug("adjustScale skipped: missing frame context.")
            return

        super().adjustScale(initial=initial)
        # if self._config["keep_prev_contrast"] and self.recentFiles:
        #     _, contrast = self.brightnessContrast_values.get(
        #         self.recentFiles[0], (None, None)
        #     )
        # if brightness is not None:
        #     dialog.slider_brightness.setValue(brightness)
        # if contrast is not None:
        #     dialog.slider_contrast.setValue(contrast)
        # self.brightnessContrast_values[self.filename] = (brightness, contrast)
        # if brightness is not None or contrast is not None:
        #     dialog.onNewValue(None)
        self.paintCanvas()
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        self.loadPredictShapes(frame_number, filename)
        if self._df_deeplabcut is not None:
            self._load_deeplabcut_results(frame_number)
        self._refresh_behavior_overlay()
        return True

    # ------------------------------------------------------------------
    # Patch similarity (DINO) integration
    # ------------------------------------------------------------------
    def _toggle_patch_similarity_tool(self, checked=False):
        self.dino_controller.toggle_patch_similarity(checked)

    # ------------------------------------------------------------------
    # PCA feature map (DINO) integration
    # ------------------------------------------------------------------
    def _toggle_pca_map_tool(self, checked=False):
        self.dino_controller.toggle_pca_map(checked)

    def _deactivate_pca_map(self):
        self.dino_controller.deactivate_pca_map()

    def _request_pca_map(self) -> None:
        self.dino_controller.request_pca_map()

    def _open_patch_similarity_settings(self):
        self.dino_controller.open_patch_similarity_settings()

    def _open_pca_map_settings(self):
        self.dino_controller.open_pca_map_settings()

    # ---------------------------------------------------------------
    # 3D Viewer
    # ---------------------------------------------------------------
    def open_3d_viewer(self):
        """Open Annolid's built-in 3D stack viewer.

        If a TIFF stack is already open in the main UI, it will be used.
        Otherwise, prompt the user to select a TIFF file.
        """
        tiff_path = None
        try:
            from annolid.data import videos as _videos_mod
            if isinstance(self.video_loader, _videos_mod.TiffStackVideo) and self.video_file:
                tiff_path = str(self.video_file)
        except Exception:
            pass

        if not tiff_path:
            # Prompt user to select a TIFF/OME-TIFF
            start_dir = str(Path(self.filename).parent) if getattr(self, "filename", None) else "."
            filters = self.tr("TIFF files (*.tif *.tiff *.ome.tif *.ome.tiff);;All files (*.*)")
            res = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Choose TIFF Stack"),
                start_dir,
                filters,
            )
            if isinstance(res, tuple):
                tiff_path = res[0]
            else:
                tiff_path = res
            if not tiff_path:
                return

        # Prefer true 3D (VTK) if available, else fallback to slice/MIP viewer
        vtk_missing = False
        try:
            from annolid.gui.widgets.vtk_volume_viewer import VTKVolumeViewerDialog  # type: ignore
            dlg = VTKVolumeViewerDialog(tiff_path, parent=self)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return
        except ModuleNotFoundError:
            vtk_missing = True
        except ImportError:
            vtk_missing = True
        except Exception:
            # Any other VTK/runtime error will fall back silently
            pass

        try:
            from annolid.gui.widgets.volume_viewer import VolumeViewerDialog
            dlg = VolumeViewerDialog(tiff_path, parent=self)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("3D Viewer"),
                self.tr(f"Unable to open 3D viewer: {e}"),
            )
            return

        # If VTK was missing, offer install guidance (non-blocking info)
        if vtk_missing:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("True 3D Rendering (Optional)"),
                self.tr(
                    "For interactive 3D volume rendering, install VTK:\n\n"
                    "Conda:  conda install -c conda-forge vtk\n"
                    "Pip:    pip install vtk\n\n"
                    "You are currently using the built-in slice/MIP viewer."
                ),
            )

    def _on_pca_map_started(self):
        self.statusBar().showMessage(self.tr("Computing PCA feature map…"))

    def _on_pca_map_finished(self, payload: dict) -> None:
        if not self.pca_map_action.isChecked():
            return
        overlay = payload.get("overlay_rgba")
        self.canvas.setPCAMapOverlay(overlay)
        cluster_labels = payload.get("cluster_labels") or []
        if cluster_labels:
            labels_text = ", ".join(cluster_labels)
            message = self.tr("PCA clustering ready (%s)") % labels_text
        else:
            message = self.tr("PCA feature map ready.")
        self.statusBar().showMessage(message, 4000)

    def _on_pca_map_error(self, message: str) -> None:
        self.canvas.setPCAMapOverlay(None)
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("PCA Feature Map"),
            message,
        )
        self._deactivate_pca_map()

    def _open_pca_map_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("PCA Feature Map Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(self.pca_map_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(self.pca_map_alpha)

        cluster_spin = QtWidgets.QSpinBox(dialog)
        cluster_spin.setRange(0, 32)
        cluster_spin.setValue(
            max(0, int(getattr(self, "pca_map_clusters", 0))))

        layout.addRow(self.tr("Model"), model_combo)
        layout.addRow(self.tr("Overlay opacity"), alpha_spin)
        layout.addRow(self.tr("Cluster count"), cluster_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.pca_map_model = model_combo.currentData()
            self.pca_map_alpha = alpha_spin.value()
            self.pca_map_clusters = cluster_spin.value()
            self.settings.setValue("pca_map/model", self.pca_map_model)
            self.settings.setValue("pca_map/alpha", self.pca_map_alpha)
            self.settings.setValue("pca_map/clusters", self.pca_map_clusters)
            self.statusBar().showMessage(
                self.tr("PCA feature map preferences updated."),
                3000,
            )
            if self.pca_map_action.isChecked():
                self._request_pca_map()

    def _stop_frame_loader(self):
        """Tear down the frame loader safely from its owning thread."""
        loader = getattr(self, "frame_loader", None)
        if loader is None:
            return

        # Hold reference locally in case self.frame_loader is reassigned elsewhere
        old_loader = loader
        try:
            target_thread = old_loader.thread()
            current_thread = QtCore.QThread.currentThread()
            if target_thread is None or not target_thread.isRunning():
                if target_thread is not current_thread:
                    try:
                        old_loader.moveToThread(current_thread)
                    except RuntimeError:
                        logger.debug(
                            "Unable to move frame loader to current thread during shutdown.",
                            exc_info=True,
                        )
                old_loader.shutdown()
            elif target_thread is current_thread:
                old_loader.shutdown()
            else:
                QtCore.QMetaObject.invokeMethod(
                    old_loader,
                    "shutdown",
                    QtCore.Qt.BlockingQueuedConnection,
                )
        except RuntimeError:
            logger.debug("Frame loader already cleaned up.", exc_info=True)
        finally:
            if self.frame_loader is old_loader:
                self.frame_loader = None

    def clean_up(self):
        def quit_and_wait(thread, message):
            if thread is not None:
                try:
                    thread.quit()
                    thread.wait()
                except RuntimeError:
                    logger.info(message)

        self._stop_frame_loader()
        quit_and_wait(self.frame_worker, "Thank you!")
        quit_and_wait(self.seg_train_thread, "See you next time!")
        quit_and_wait(self.seg_pred_thread, "Bye!")
        if hasattr(self, "yolo_training_manager") and self.yolo_training_manager:
            self.yolo_training_manager.cleanup()

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            description = shape.get("description", "")
            other_data = shape["other_data"]
            if "visible" in shape:
                visible = shape["visible"]
            else:
                visible = True

            if not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
                visible=visible
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data
            s.append(shape)

        self.loadShapes(s)
        return s

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        for shape in shapes:
            if not isinstance(shape.points[0], QtCore.QPointF):
                shape.points = [QtCore.QPointF(x, y)
                                for x, y in shape.points]
            self.addLabel(shape)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        try:
            caption = self.labelFile.get_caption() if self.labelFile else None
        except AttributeError:
            caption = None
        if caption is not None and len(caption) > 0:
            if self.caption_widget is None:
                self.openCaption()
            self.caption_widget.set_caption(
                caption)  # Update caption widget
            self.caption_widget.set_image_path(self.filename)

    def update_flags_from_file(self, label_file):
        """Update flags from label file with proper validation and error handling.

        Args:
            label_file: LabelFile object containing flags
        """
        if not hasattr(label_file, 'flags'):
            logger.warning("Label file has no flags attribute")
            return

        try:
            # Validate flags from file
            if isinstance(label_file.flags, dict):
                # Deep copy to avoid modifying original
                new_flags = label_file.flags.copy()
                flags_in_frame = ','.join(new_flags.keys())
                self.canvas.setBehaviorText(flags_in_frame)
                _existing_flags = self.flag_widget._get_existing_flag_names()
                for _flag in _existing_flags:
                    if _flag not in new_flags:
                        new_flags[_flag] = False
                self.flag_widget.loadFlags(new_flags)
            else:
                logger.error(f"Invalid flags format: {type(label_file.flags)}")

        except Exception as e:
            logger.error(f"Error updating flags: {e}")

    def _annotation_store_has_frame(self, label_json_file: str) -> bool:
        """Return True if the annotation store contains a record for the given label path."""
        try:
            path = Path(label_json_file)
            frame_number = AnnotationStore.frame_number_from_path(path)
            if frame_number is None:
                return False
            store = AnnotationStore.for_frame_path(path)
            if not store.store_path.exists():
                return False
            return store.get_frame(frame_number) is not None
        except Exception:
            return False

    def _annotation_store_frame_count(self) -> int:
        """Return the number of frames currently stored in the annotation store."""
        if not self.video_results_folder:
            return 0
        try:
            store = AnnotationStore.for_frame_path(
                self.video_results_folder /
                f"{self.video_results_folder.name}_000000000.json"
            )
            if not store.store_path.exists():
                return 0
            return len(list(store.iter_frames()))
        except Exception:
            return 0

    def _iter_frame_label_candidates(self, frame_number: int, frame_path: Optional[Path]) -> list[Path]:
        """Return possible annotation paths for a given frame."""
        candidates: list[Path] = []

        def _append_candidate(path: Optional[Path]) -> None:
            if path is None:
                return
            if path not in candidates:
                candidates.append(path)

        if frame_path is not None:
            frame_path = Path(frame_path)
            if frame_path.suffix.lower() == ".json":
                _append_candidate(frame_path)
            else:
                _append_candidate(frame_path.with_suffix(".json"))

            stem = frame_path.stem
            if "_" in stem:
                alt_name = f"{stem.split('_')[-1]}.json"
                _append_candidate(frame_path.parent / alt_name)

        frame_tag = f"{int(frame_number):09}"
        base_dir: Optional[Path] = None
        if frame_path is not None:
            base_dir = frame_path.parent
        if self.video_results_folder:
            base_dir = self.video_results_folder

        if base_dir is not None:
            if self.video_results_folder:
                stem_name = self.video_results_folder.name
            elif frame_path is not None:
                stem_name = frame_path.stem.rsplit("_", 1)[0]
            else:
                stem_name = base_dir.name

            if stem_name:
                _append_candidate(base_dir /
                                  f"{stem_name}_{frame_tag}.json")
            _append_candidate(base_dir / f"{frame_tag}.json")

        if self.video_results_folder:
            pred_dir = self.video_results_folder / self._pred_res_folder_suffix
            if pred_dir.exists():
                stem_name = self.video_results_folder.name
                _append_candidate(pred_dir / f"{stem_name}_{frame_tag}.json")
                _append_candidate(pred_dir / f"{frame_tag}.json")

        if self.annotation_dir:
            annot_dir = Path(self.annotation_dir)
            stem_name = annot_dir.name
            _append_candidate(annot_dir / f"{stem_name}_{frame_tag}.json")
            _append_candidate(annot_dir / f"{frame_tag}.json")

        return candidates

    def loadPredictShapes(self, frame_number, filename):
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(filename)

        frame_path = Path(filename) if filename else None
        label_candidates = self._iter_frame_label_candidates(
            frame_number, frame_path)

        seen_candidates: set[Path] = set()
        label_loaded = False
        for candidate in label_candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)

            candidate_exists = candidate.exists()
            candidate_in_store = self._annotation_store_has_frame(candidate)
            if not candidate_exists and not candidate_in_store:
                continue

            try:
                label_file = LabelFile(
                    str(candidate),
                    is_video_frame=True,
                )
            except LabelFileError as exc:
                logger.error(
                    "Failed to load label file %s: %s",
                    candidate,
                    exc,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Unexpected error loading label file %s: %s",
                    candidate,
                    exc,
                )
                continue

            self.labelFile = label_file
            self.canvas.setBehaviorText(None)
            self.loadLabels(label_file.shapes)
            self.update_flags_from_file(label_file)
            if len(self.canvas.current_behavior_text) > 1 and 'other' not in self.canvas.current_behavior_text.lower():
                self.add_highlighted_mark(
                    self.frame_number, mark_type=self.canvas.current_behavior_text)
            caption = label_file.get_caption()
            if caption is not None and len(caption) > 0:
                if self.caption_widget is None:
                    self.openCaption()
                self.caption_widget.set_caption(caption)
            label_loaded = True
            break

        if label_loaded:
            return

        if self._df is not None and (frame_path is None or not frame_path.exists()):
            df_cur = self._df[self._df.frame_number == frame_number]
            frame_label_list = []
            pd.options.mode.chained_assignment = None
            for row in df_cur.to_dict(orient='records'):
                if 'x1' not in row:
                    row['x1'] = 2
                    row['y1'] = 2
                    row['x2'] = 4
                    row['y2'] = 4
                    row['class_score'] = 1
                    df_cur.drop('frame_number', axis=1, inplace=True)
                    try:
                        instance_names = df_cur.apply(lambda row:
                                                      df_cur.columns[[i for i in range(
                                                          len(row)) if row[i] > 0][0]
                                                      ],
                                                      axis=1).tolist()
                        row['instance_name'] = '_'.join(instance_names)
                    except IndexError:
                        row['instance_name'] = 'unknown'
                    row['segmentation'] = None
                pred_label_list = pred_dict_to_labelme(row)
                frame_label_list += pred_label_list

            self.loadShapes(frame_label_list)

        # No label file or prediction available; clear caption if any.
        if not label_loaded and self.caption_widget is not None:
            self.caption_widget.set_caption("")

    def openNextImg(self, _value=False, load=True):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            # If the current cursor is at the end of the video,
            # reset it to start from the beginning.
            if self.frame_number >= self.num_frames:
                self.frame_number = 0
                self.set_frame_number(self.frame_number)

            if self.frame_number < self.num_frames:
                if self.step_size + self.frame_number <= self.num_frames:
                    self.frame_number += self.step_size
                else:
                    self.frame_number += 1
            else:
                self.frame_number = self.num_frames
                self.togglePlay()
            self._suppress_audio_seek = True
            try:
                self.set_frame_number(self.frame_number)
                # update the seekbar value
                self.seekbar.setValue(self.frame_number)
            finally:
                self._suppress_audio_seek = False
            self.uniqLabelList.itemSelectionChanged.connect(
                self.handle_uniq_label_list_selection_change)

        elif len(self.imageList) <= 0:
            return
        else:
            filename = None
            if self.filename is None:
                filename = self.imageList[0]
            else:
                currIndex = self.imageList.index(self.filename)
                if currIndex + 1 < len(self.imageList):
                    filename = self.imageList[currIndex + 1]
                else:
                    filename = self.imageList[-1]
            self.filename = filename

            if self.filename and load:
                self.loadFile(self.filename)
                if self.caption_widget is not None:
                    self.caption_widget.set_image_path(self.filename)

        self._config["keep_prev"] = keep_prev
        self._update_frame_display_and_emit_update()

    def openPrevImg(self, _value=False):
        keep_prev = self._config["keep_prev"]
        if Qt.KeyboardModifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
            self._config["keep_prev"] = True

        if not self.mayContinue():
            return

        if self.video_loader is not None:
            if self.frame_number > 1:
                if self.frame_number - self.step_size >= 1:
                    self.frame_number -= self.step_size
                else:
                    self.frame_number -= 1
            else:
                self.frame_number = 0
            self.set_frame_number(self.frame_number)
            # update the seekbar value
            self.seekbar.setValue(self.frame_number)

        elif len(self.imageList) <= 0:
            return
        else:
            if self.filename is None:
                return
            currIndex = self.imageList.index(self.filename)
            if currIndex - 1 >= 0:
                filename = self.imageList[currIndex - 1]
                if filename:
                    self.loadFile(filename)
                    if self.caption_widget is not None:
                        self.caption_widget.set_image_path(self.filename)

        self._config["keep_prev"] = keep_prev
        self._update_frame_display_and_emit_update()

    def _emit_live_frame_update(self):
        if self.filename and self.frame_number is not None and hasattr(self, '_time_stamp'):
            self.live_annolid_frame_updated.emit(
                self.frame_number, self._time_stamp or "")

    def _save_segments_for_active_video(self):
        if not self.video_file or not hasattr(self, '_current_video_defined_segments'):
            return
        segments_as_dicts = [s.to_dict()
                             for s in self._current_video_defined_segments]
        # Use Path(self.video_file) to ensure it's a Path object
        sidecar_path = Path(self.video_file).with_suffix(
            Path(self.video_file).suffix + ".segments.json")
        try:
            with open(sidecar_path, 'w') as f:
                json.dump(segments_as_dicts, f, indent=2)
            logger.info(
                f"Saved {len(segments_as_dicts)} segments to {sidecar_path}")
        except Exception as e:
            logger.error(f"Failed to save segments to {sidecar_path}: {e}")

    def _load_segments_for_active_video(self):
        self._current_video_defined_segments = []  # Clear first
        if not self.video_file or not self.fps:
            return

        sidecar_path = Path(self.video_file).with_suffix(
            Path(self.video_file).suffix + ".segments.json")
        if sidecar_path.exists():
            try:
                with open(sidecar_path, 'r') as f:
                    segment_dicts = json.load(f)
                loaded_segments = []
                for s_dict in segment_dicts:
                    try:  # Ensure current video context is used, even if stored differently
                        # Force current video path
                        s_dict['video_path'] = str(self.video_file)
                        # Force current FPS
                        s_dict['fps'] = self.fps
                        loaded_segments.append(
                            TrackingSegment.from_dict(s_dict))
                    except Exception as e:
                        logger.error(
                            f"Error creating TrackingSegment from dict {s_dict}: {e}")
                self._current_video_defined_segments = loaded_segments
                logger.info(
                    f"Loaded {len(self._current_video_defined_segments)} segments from {sidecar_path}")
            except Exception as e:
                logger.error(
                    f"Failed to load segments from {sidecar_path}: {e}")
        if self.caption_widget is not None:
            self.caption_widget.set_video_segments(
                self._current_video_defined_segments
            )

    # --- Handler for Tracking Initiated by SegmentEditorDialog ---

    def _get_tracking_device(self) -> torch.device:  # Centralized device selection
        # More sophisticated logic could go here (e.g., user settings)
        if self.config.get('use_cpu_only', False):
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def set_tracking_ui_state(self, is_tracking: bool) -> None:
        self.open_segment_editor_action.setEnabled(
            not is_tracking and bool(self.video_file))
        if hasattr(self.actions, 'open'):
            self.actions.open.setEnabled(not is_tracking)
        if hasattr(self.actions, 'openDir'):
            self.actions.openDir.setEnabled(not is_tracking)
        if hasattr(self.actions, 'openVideo'):
            self.actions.openVideo.setEnabled(not is_tracking)
        if hasattr(self, 'video_manager_widget') and hasattr(self.video_manager_widget, 'track_all_button'):
            self.video_manager_widget.track_all_button.setEnabled(
                not is_tracking)
        logger.info(
            "AnnolidWindow UI state for tracking: %s",
            "ACTIVE" if is_tracking else "IDLE",
        )

    def coco(self):
        """
        Convert Labelme annotations to COCO format.
        """
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
            QtWidgets.QMessageBox.about(self,
                                        "No input file or directory",
                                        f"Please check and open the  \
                                        files or directories.")
            return

        if output_dir is None:
            self.output_dir = Path(input_anno_dir).parent / \
                (Path(input_anno_dir).name + '_coco_dataset')

        else:
            self.output_dir = output_dir

        if labels_file is None:
            labels_file = str(self.here.parent / 'annotation' /
                              'labels_custom.txt')

        label_gen = labelme2coco.convert(
            str(input_anno_dir),
            output_annotated_dir=str(self.output_dir),
            labels_file=labels_file,
            train_valid_split=num_train_frames
        )
        pw = ProgressingWindow(label_gen)
        if pw.exec_():
            pw.runner_thread.terminate()

        self.statusBar().showMessage(self.tr("%s ...") % "converting")
        QtWidgets.QMessageBox.about(self,
                                    "Finished",
                                    f"Done! Results are in folder: \
                                            {str(self.output_dir)}")
        self.statusBar().showMessage(self.tr("%s Done.") % "converting")
        try:
            shutil.make_archive(str(self.output_dir),
                                'zip', self.output_dir.parent, self.output_dir.stem)
        except:
            print("Failed to create the zip file")

    def visualization(self):
        try:
            url = 'http://localhost:6006/'
            process = start_tensorboard(tensorboard_url=url)
            webbrowser.open(url)
        except Exception:
            vdlg = VisualizationWindow()
            if vdlg.exec_():
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
            QtWidgets.QMessageBox.about(self,
                                        "No input video or tracking results",
                                        f"Please check and open the  \
                                        files.")
            return
        out_dir = f"{str(Path(video_file).with_suffix(''))}{self._pred_res_folder_suffix}"
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        trs = TracksResults(video_file, tracking_results)
        label_json_gen = trs.to_labelme_json(str(out_dir),
                                             skip_frames=skip_num_frames)

        try:
            if label_json_gen is not None:
                pwj = ProgressingWindow(label_json_gen)
                if pwj.exec_():
                    trs._is_running = False
                    pwj.running_submitted.emit('stopped')
                    pwj.runner_thread.terminate()
                    pwj.runner_thread.quit()
        except Exception:
            pass
        finally:
            self.importDirImages(str(out_dir))

    def newShape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        if self.canvas.createMode == 'grounding_sam':
            self.labelList.clearSelection()
            shapes = [
                shape for shape in self.canvas.shapes
                if shape.description == 'grounding_sam']
            shape = shapes.pop()
            self.addLabel(shape)
            self.actions.editMode.setEnabled(True)
            self.actions.undoLastPoint.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.setDirty()
        else:
            items = self.uniqLabelList.selectedItems()
            text = None
            if items:
                text = items[0].data(Qt.UserRole)
            flags = {}
            group_id = None
            description = ""
            if self._config["display_label_popup"] or not text:
                previous_text = self.labelDialog.edit.text()
                text, flags, group_id, description = self.labelDialog.popUp(
                    text)
                if not text:
                    self.labelDialog.edit.setText(previous_text)
            if text and not self.validateLabel(text):
                self.errorMessage(
                    self.tr("Invalid label"),
                    self.tr("Invalid label '{}' with validation type '{}'").format(
                        text, self._config["validate_label"]
                    ),
                )
                text = ""
            if text:
                self.labelList.clearSelection()
                shapes = self.canvas.setLastLabel(text, flags)
                for shape in shapes:
                    shape.group_id = group_id
                    shape.description = description
                    self.addLabel(shape)
                self.actions.editMode.setEnabled(True)
                self.actions.undoLastPoint.setEnabled(False)
                self.actions.undo.setEnabled(True)
                self.setDirty()
            else:
                self.canvas.undoLastLine()
                self.canvas.shapesBackups.pop()

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
        behaviors = None

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
            QtWidgets.QMessageBox.about(self,
                                        "No input video or tracking results",
                                        f"Please check and open the  \
                                        files.")
            return

        if out_nix_csv_file is None:
            out_nix_csv_file = tracking_results.replace('.csv', '_nix.csv')

        tracks2nix(
            video_file,
            tracking_results,
            out_nix_csv_file,
            zone_info=zone_info_json,
            score_threshold=score_threshold,
            motion_threshold=motion_threshold,
            pretrained_model=pretrained_model,
            subject_names=subject_names,
            behavior_names=behavior_names
        )


def main(argv=None):
    config, _, version_requested = parse_cli(argv)
    if version_requested:
        print(__version__)
        return 0

    qt_args = sys.argv if argv is None else [sys.argv[0], *argv]
    app = create_qapp(qt_args)

    app.setApplicationName(__appname__)
    annolid_icon = QtGui.QIcon(
        str(Path(__file__).resolve().parent / "icons/icon_annolid.png"))
    app.setWindowIcon(annolid_icon)
    win = AnnolidWindow(config=config)
    logger.info("Qt config file: %s" % win.settings.fileName())

    win.show()
    win.raise_()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
