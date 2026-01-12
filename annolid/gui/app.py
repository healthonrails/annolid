from __future__ import annotations

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
from PIL import ImageQt, Image, ImageDraw
import pandas as pd
import numpy as np
import imgviz
import yaml
from pathlib import Path
import functools
import subprocess
try:
    import torch
except ImportError:  # PyTorch is optional for lighter desktop bundles
    torch = None

from labelme.ai import MODELS
from qtpy import QtCore
from qtpy.QtCore import Qt, Slot, Signal
from qtpy import QtWidgets
from qtpy import QtGui
from labelme import PY2
from labelme import QT5
from annolid.gui.widgets.video_manager import VideoManagerWidget
from annolid.gui.workers import (
    FlexibleWorker,
    LoadFrameThread,
)
from annolid.gui.shape import Shape
from labelme.app import MainWindow
from labelme.utils import newAction
from labelme.widgets import LabelListWidgetItem
from labelme import utils
from annolid.utils.logger import logger
from annolid.utils.files import (
    count_json_files,
    should_start_predictions_from_frame0,
)
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
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image
from annolid.gui.widgets import ExtractFrameDialog
from annolid.gui.widgets import ConvertCOODialog
from annolid.gui.widgets import TrainModelDialog
from annolid.gui.widgets import Glitter2Dialog
from annolid.gui.widgets import QualityControlDialog
from annolid.gui.widgets import TrackDialog
from annolid.gui.widgets import SystemInfoDialog
from annolid.gui.widgets import FlagTableWidget
from annolid.gui.widgets import LabelCollectionDialog
from annolid.postprocessing.glitter import tracks2nix
from annolid.postprocessing.quality_control import TracksResults
from annolid.gui.widgets import ProgressingWindow
from annolid.gui.widgets import TrainingDashboardDialog
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
from annolid.gui.widgets.pdf_manager import PdfManager
from annolid.gui.widgets.depth_manager import DepthManager
from annolid.gui.widgets.sam3d_manager import Sam3DManager
from annolid.gui.widgets.sam2_manager import Sam2Manager
from annolid.gui.widgets.sam3_manager import Sam3Manager
from annolid.gui.widgets.optical_flow_manager import OpticalFlowManager
from annolid.gui.widgets.realtime_manager import RealtimeManager
from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog
from annolid.gui.widgets.youtube_dialog import YouTubeVideoDialog
from annolid.postprocessing.quality_control import pred_dict_to_labelme
import io

from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.annotation import labelme2csv
from annolid.annotation.pose_schema import PoseSchema
from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    keypoint_visibility_from_shape_object,
    set_keypoint_visibility_on_shape_object,
)
from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog
from annolid.data.videos import get_video_files
from annolid.data.audios import AudioLoader
from annolid.gui.widgets.caption import CaptionWidget
from annolid.gui.widgets.florence2_widget import Florence2DockWidget
from annolid.gui.widgets.image_editing_widget import ImageEditingDockWidget
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.gui.model_manager import AIModelManager
from annolid.postprocessing.video_timestamp_annotator import process_directory
from annolid.gui.widgets.segment_editor import SegmentEditorDialog
import contextlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from annolid.jobs.tracking_jobs import TrackingSegment

from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import DinoKeypointVideoProcessor
from annolid.tracking.dino_kpseg_tracker import DinoKPSEGVideoProcessor
from annolid.gui.behavior_controller import BehaviorController, BehaviorEvent
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget
from annolid.gui.tensorboard import ensure_tensorboard, start_tensorboard, VisualizationWindow
from annolid.utils.runs import find_latest_checkpoint, shared_runs_root
from annolid.realtime.perception import Config as RealtimeConfig
from annolid.gui.yolo_training_manager import YOLOTrainingManager
from annolid.gui.dino_kpseg_training_manager import DinoKPSEGTrainingManager
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
__version__ = "1.3.3"

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
        self._prediction_stop_requested = False
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
        self._show_pose_edges = self.settings.value(
            "pose/show_edges", True, type=bool)
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
        self.dino_kpseg_training_manager = DinoKPSEGTrainingManager(self)
        self._training_dashboard_dialog = None
        self.yolo_training_manager.training_started.connect(
            self._show_training_dashboard_for_training
        )
        self.dino_kpseg_training_manager.training_started.connect(
            self._show_training_dashboard_for_training
        )
        self.frame_number = 0
        self.video_loader = None
        self.video_file = None
        self.isPlaying = False
        self.event_type = None
        self._time_stamp = ''
        self.behavior_controller = BehaviorController(self._get_rgb_by_label)
        self.project_schema: Optional[ProjectSchema] = None
        self.project_schema_path: Optional[Path] = None
        self._pose_schema_path: Optional[str] = None
        self._pose_schema: Optional[PoseSchema] = None
        self.behavior_controller.configure_from_schema(self.project_schema)
        self.annotation_dir = None
        self.step_size = 5
        self.stepSizeWidget = StepSizeWidget(5)
        self.prev_shapes = None
        self.pred_worker = None
        self.video_processor = None
        self._active_subject_name: Optional[str] = None
        self._behavior_modifier_state: Dict[str, Set[str]] = {}
        self.zone_path = None
        # Initialize a flag to control thread termination
        self.stop_prediction_flag = False
        self.epsilon_for_polygon = 2.0
        self.automatic_pause_enabled = False
        self.t_max_value = 5
        self.use_cpu_only = False
        self.save_video_with_color_mask = False
        self.auto_recovery_missing_instances = False
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
        self._prediction_start_frame = None
        self._prediction_existing_store_frames = set()
        self._prediction_existing_json_frames = set()

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            sam=self._config["sam"]
        )
        try:
            self.canvas.setShowPoseEdges(self._show_pose_edges)
        except Exception:
            pass
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.setContentsMargins(0, 0, 0, 0)
        self._viewer_stack.addWidget(self.canvas)
        self.pdf_manager = PdfManager(self, self._viewer_stack)
        self.depth_manager = DepthManager(self)
        self.optical_flow_manager = OpticalFlowManager(self)
        self.sam3d_manager = Sam3DManager(self)
        self.sam2_manager = Sam2Manager(self)
        self.sam3_manager = Sam3Manager(self)
        self.realtime_manager = RealtimeManager(self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self._viewer_stack)
        scrollArea.setWidgetResizable(True)
        scrollArea.setAlignment(Qt.AlignCenter)
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
        self._setup_label_collection_action()

        self.populateModeActions()

    def paintCanvas(self):
        """Update zoom and redraw the viewer.

        LabelMe's default implementation calls `self.canvas.adjustSize()`, which
        works when the canvas is directly inside the scroll area. Annolid embeds
        the canvas inside `self._viewer_stack`, so resizing the canvas can leave
        it top-aligned within the stacked widget and show a large empty area.
        """
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.updateGeometry()
        if getattr(self, "_viewer_stack", None) is not None:
            self._viewer_stack.updateGeometry()
            self._viewer_stack.adjustSize()
        else:
            self.canvas.adjustSize()
        self.canvas.update()

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
        """Adds a 'Collect Labels' entry to the File menu."""
        action = functools.partial(newAction, self)
        self.collect_labels_action = action(
            self.tr("Collect &Labels..."),
            self._open_label_collection_dialog,
            None,
            "open",
            self.tr(
                "Index labeled PNG/JSON pairs into a central dataset JSONL file."),
            enabled=True,
        )
        file_menu = getattr(self.menus, "file", None)
        if file_menu is not None:
            file_menu.addAction(self.collect_labels_action)

    def _open_label_collection_dialog(self) -> None:
        dlg = LabelCollectionDialog(settings=self.settings, parent=self)
        dlg.exec_()

    def _set_active_view(self, mode: str = "canvas") -> None:
        """Switch the central view between the canvas and PDF viewer."""
        if mode == "pdf" and getattr(self, "pdf_manager", None) is not None:
            viewer = self.pdf_manager.pdf_widget()
            if viewer is not None:
                pdf_index = self._viewer_stack.indexOf(viewer)
                if pdf_index != -1:
                    self._viewer_stack.setCurrentIndex(pdf_index)
                    return
        self._viewer_stack.setCurrentIndex(0)

    def show_pdf_in_viewer(self, pdf_path: str) -> None:
        """Load a PDF into the viewer and display it in place of the canvas."""
        if self.pdf_manager is not None:
            self.pdf_manager.show_pdf_in_viewer(pdf_path)

    @QtCore.Slot(str)
    def _apply_pdf_selection_to_caption(self, text: str) -> None:
        """Send selected PDF text into the caption widget for TTS."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self.caption_widget is None:
            self.openCaption()
        if self.caption_widget is not None:
            self.caption_widget.set_caption(cleaned)

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

        # Check whether the user requested CountGD in the UI (optional).
        use_countgd = False
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_useCountGDCheckbox"
            ):
                use_countgd = self.aiRectangle._useCountGDCheckbox.isChecked()
        except Exception:
            use_countgd = False

        # Check if the prompt starts with 'flags:' and contains flags separated by commas
        if prompt_text.startswith('flags:'):
            flags = {k.strip(): False for k in prompt_text.replace(
                'flags:', '').split(',') if len(k.strip()) > 0}
            if len(flags.keys()) > 0:
                self.flags_controller.apply_prompt_flags(flags)
            else:
                self.flags_controller.clear_flags()
        else:
            self.canvas.predictAiRectangle(
                prompt_text, use_countgd=use_countgd)

    def _current_text_prompt(self) -> Optional[str]:
        """Return the current AI text prompt (trimmed) if available."""
        prompt = None
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_aiRectanglePrompt"
            ):
                widget = self.aiRectangle._aiRectanglePrompt
                if widget:
                    prompt = widget.text().strip() or None
        except Exception:
            prompt = None
        return prompt

    def update_step_size(self, value):
        self.step_size = value
        self.stepSizeWidget.set_value(self.step_size)

    def downsample_videos(self):
        video_downsample_widget = VideoRescaleWidget()
        video_downsample_widget.exec_()

    def run_optical_flow_tool(self):
        """Open the optical-flow tool dialog and run flow."""
        if getattr(self, "optical_flow_manager", None) is not None:
            return self.optical_flow_manager.run_tool()

    def configure_optical_flow_settings(self):
        """Open the optical-flow settings dialog without starting processing."""
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.configure_tool()

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

    def open_pose_schema_dialog(self):
        """Define keypoint order + symmetry pairs and save a pose schema file."""
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
        if schema is None and self.project_schema and getattr(self.project_schema, "pose_schema", None):
            try:
                schema = PoseSchema.from_dict(
                    self.project_schema.pose_schema)  # type: ignore[arg-type]
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

    def _persist_pose_schema_to_project_schema(self, schema: PoseSchema, schema_path: str) -> None:
        """Store pose schema metadata inside `project.annolid.json` by default."""
        project_schema = self.project_schema or default_schema()
        project_path = self.project_schema_path
        if project_path is None:
            if self.video_file:
                project_path = Path(self.video_file).with_suffix(
                    "") / DEFAULT_SCHEMA_FILENAME
            else:
                project_path = Path.cwd() / DEFAULT_SCHEMA_FILENAME

        try:
            project_path.parent.mkdir(parents=True, exist_ok=True)
            # Prefer storing a relative path when the schema lives alongside the project schema.
            stored_path = schema_path
            try:
                stored_path = str(
                    Path(schema_path).resolve().relative_to(
                        project_path.parent.resolve())
                )
            except Exception:
                stored_path = schema_path

            project_schema.pose_schema_path = stored_path
            project_schema.pose_schema = schema.to_dict()
            save_project_schema(project_schema, project_path)
            self.project_schema = project_schema
            self.project_schema_path = project_path
        except Exception:
            logger.debug(
                "Failed to persist pose schema into project schema.", exc_info=True)

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
            start_dir = getattr(self, "lastOpenDir", None) or str(Path.home())
            audio_widget, audio_filename = AudioWidget.create_from_dialog(
                parent=self,
                start_dir=start_dir,
                caption=self.tr(f"{__appname__} - Choose Audio"),
                error_title=self.tr("Audio"),
                error_message=self.tr(
                    "Unable to load the selected audio file."),
            )
            if not audio_widget or not audio_filename:
                return

            self.lastOpenDir = str(Path(audio_filename).parent)
            if self.audio_dock:
                self.audio_dock.close()
            self.audio_dock = None
            if self.audio_widget:
                self.audio_widget.close()
            self.audio_widget = None
            self._release_audio_loader()

            self.audio_widget = audio_widget

            self.audio_dock = QtWidgets.QDockWidget(self.tr("Audio"), self)
            self.audio_dock.setObjectName("Audio")
            self.audio_dock.setWidget(self.audio_widget)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.audio_dock)
            self.audio_dock.visibilityChanged.connect(
                self._on_audio_dock_visibility_changed
            )
            return

        if self.audio_dock:
            self.audio_dock.close()
        self.audio_dock = None
        if self.audio_widget:
            self.audio_widget.close()
        self.audio_widget = None

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
        self.audio_dock.visibilityChanged.connect(
            self._on_audio_dock_visibility_changed
        )

    def _on_audio_dock_visibility_changed(self, visible: bool) -> None:
        if visible:
            return
        QtCore.QTimer.singleShot(0, self._cleanup_audio_ui)

    def _cleanup_audio_ui(self) -> None:
        """Close the audio dock/widget and release any associated audio loader."""
        if getattr(self, "_cleaning_audio_ui", False):
            return
        self._cleaning_audio_ui = True
        try:
            if self.audio_widget and getattr(self.audio_widget, "audio_loader", None):
                with contextlib.suppress(Exception):
                    self.audio_widget.audio_loader.stop()
            if self.audio_widget:
                with contextlib.suppress(Exception):
                    self.audio_widget.close()
            self.audio_widget = None
            if self.audio_dock:
                with contextlib.suppress(Exception):
                    self.audio_dock.close()
                with contextlib.suppress(Exception):
                    self.audio_dock.deleteLater()
            self.audio_dock = None
            self._release_audio_loader()
        finally:
            self._cleaning_audio_ui = False

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
        dock = getattr(self, "florence_dock", None)
        if dock is None:
            dock = Florence2DockWidget(self)
            dock.destroyed.connect(
                lambda *_: setattr(self, "florence_dock", None))
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.florence_dock = dock

        if isinstance(dock, Florence2DockWidget):
            dock.show_or_raise()
        else:
            if dock.isHidden():
                dock.show()
            dock.raise_()

    def openImageEditing(self):
        """Open or show the Image Editing dock widget."""
        dock = getattr(self, "image_editing_dock", None)
        if dock is None:
            dock = ImageEditingDockWidget(self)
            dock.destroyed.connect(
                lambda *_: setattr(self, "image_editing_dock", None)
            )
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.image_editing_dock = dock

        if isinstance(dock, ImageEditingDockWidget):
            dock.show_or_raise()
        else:
            if dock.isHidden():
                dock.show()
            dock.raise_()

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

    def set_advanced_params(self):
        sam3_defaults = self.sam3_manager.dialog_defaults(self._config)
        advanced_params_dialog = AdvancedParametersDialog(
            self,
            tracker_config=self.tracker_runtime_config,
            sam3_runtime=sam3_defaults,
        )
        # Seed current optical-flow settings into the dialog
        of_manager = getattr(self, "optical_flow_manager", None)
        advanced_params_dialog.compute_optical_flow_checkbox.setChecked(
            bool(getattr(of_manager, "compute_optical_flow", True))
        )
        advanced_params_dialog.optical_flow_backend = getattr(
            of_manager, "optical_flow_backend", "farneback"
        )
        backend_val = str(advanced_params_dialog.optical_flow_backend).lower()
        if "raft" in backend_val:
            backend_idx = 2
        elif "torch" in backend_val:
            backend_idx = 1
        else:
            backend_idx = 0
        advanced_params_dialog.optical_flow_backend_combo.setCurrentIndex(
            backend_idx)
        if advanced_params_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.epsilon_for_polygon = advanced_params_dialog.get_epsilon_value()
        self.automatic_pause_enabled = advanced_params_dialog.is_automatic_pause_enabled()
        self.t_max_value = advanced_params_dialog.get_t_max_value()
        self.use_cpu_only = advanced_params_dialog.is_cpu_only_enabled()
        self.save_video_with_color_mask = advanced_params_dialog.is_save_video_with_color_mask_enabled()
        self.auto_recovery_missing_instances = advanced_params_dialog.is_auto_recovery_missing_instances_enabled()
        if of_manager is not None:
            of_manager.set_compute_optical_flow(
                advanced_params_dialog.is_compute_optiocal_flow_enabled()
            )
            of_manager.set_backend(
                advanced_params_dialog.get_optical_flow_backend()
            )

        tracker_settings = advanced_params_dialog.get_tracker_settings()
        for key, value in tracker_settings.items():
            setattr(self.tracker_runtime_config, key, value)

        self.sam3_manager.apply_dialog_results(
            advanced_params_dialog, self._config
        )

        of_manager = getattr(self, "optical_flow_manager", None)
        logger.info(
            "Computing optical flow is %s .",
            getattr(of_manager, "compute_optical_flow", True),
        )
        logger.info("Set epsilon for polygon to : %s",
                    self.epsilon_for_polygon)

    def segmentAnything(self,):
        self.toggleDrawMode(False, createMode="polygonSAM")
        self.canvas.loadSamPredictor()
        if not getattr(self.canvas, "sam_predictor", None):
            error = getattr(self.canvas, "_sam_last_load_error", None)
            if error == "missing_dependency":
                QtWidgets.QMessageBox.information(
                    self,
                    "Segment Anything not installed",
                    "Install Segment Anything first:\n\n"
                    "  pip install git+https://github.com/facebookresearch/segment-anything.git",
                )
            elif error == "no_pixmap":
                QtWidgets.QMessageBox.information(
                    self,
                    "No image loaded",
                    "Open an image first, then enable Segment Anything.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Segment Anything unavailable",
                    "SAM predictor was not initialized.",
                )
            self.toggleDrawMode(True)

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
        self._set_active_view("canvas")
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
        self._depth_ndjson_records = {}
        try:
            if self.canvas:
                self.canvas.setDepthPreviewOverlay(None)
        except Exception:
            pass
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.clear()
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
            self.seekbar.removeMarksByType("predicted_existing")

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
    # Realtime inference (delegated to RealtimeManager)
    # ------------------------------------------------------------------

    def _show_realtime_control_dialog(self):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.show_control_dialog()

    def _handle_realtime_start_request(
        self, realtime_config: RealtimeConfig, extras: Dict[str, Any]
    ):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager._handle_realtime_start_request(
                realtime_config, extras
            )

    def start_realtime_inference(
        self, realtime_config: RealtimeConfig, extras: Dict[str, Any]
    ):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.start_realtime_inference(
                realtime_config, extras
            )

    def stop_realtime_inference(self):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.stop_realtime_inference()

    def closeEvent(self, event):
        try:
            self.stop_realtime_inference()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.error("Error stopping realtime inference on exit: %s",
                         exc, exc_info=True)
        try:
            if getattr(self, "sam3_manager", None):
                self.sam3_manager.close_session()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            logger.warning("Error closing SAM3 session on exit: %s", exc)
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
        def marker_for_shape(shape_obj: Shape) -> str:
            if str(getattr(shape_obj, "shape_type", "") or "").lower() != "point":
                return "●"
            visibility = keypoint_visibility_from_shape_object(shape_obj)
            return "○" if visibility == int(KeypointVisibility.OCCLUDED) else "●"

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
            '{} <font color="#{:02x}{:02x}{:02x}">{}</font>'.format(
                html.escape(text), r, g, b, marker_for_shape(shape)
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
        # Ensure flags values are booleans to satisfy LabelMe's setChecked API
        shape_flags = shape.flags or {}
        safe_flags = {k: bool(v) for k, v in shape_flags.items()}
        text, flags, group_id, description = self.labelDialog.popUp(
            text=str(shape.label), flags=safe_flags,
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

        base_text = (
            str(shape.label)
            if shape.group_id is None
            else "{} ({})".format(shape.label, shape.group_id)
        )
        marker = "●"
        if str(getattr(shape, "shape_type", "") or "").lower() == "point":
            visibility = keypoint_visibility_from_shape_object(shape)
            marker = "○" if visibility == int(
                KeypointVisibility.OCCLUDED) else "●"
        item.setText(
            '{} <font color="#{:02x}{:02x}{:02x}">{}</font>'.format(
                html.escape(base_text), r, g, b, marker
            )
        )
        self.setDirty()
        if not self.uniqLabelList.findItemByLabel(shape.label):
            item = QtWidgets.QListWidgetItem()
            item.setData(Qt.UserRole, shape.label)
            self.uniqLabelList.addItem(item)

    def _selected_shapes_for_keypoint_visibility(self) -> list[Shape]:
        shapes = list(getattr(self.canvas, "selectedShapes", None) or [])
        if shapes:
            return shapes
        try:
            item = self.currentItem()
        except Exception:
            item = None
        if isinstance(item, LabelListWidgetItem):
            shape = item.shape()
            if shape is not None:
                return [shape]
        return []

    def _refresh_label_list_items_for_shapes(self, shapes: list[Shape]) -> None:
        if not shapes:
            return
        target_ids = {id(s) for s in shapes}
        for item in self.labelList:
            if not isinstance(item, LabelListWidgetItem):
                continue
            shape = item.shape()
            if shape is None or id(shape) not in target_ids:
                continue
            r, g, b = self._update_shape_color(shape)
            base_text = (
                str(shape.label)
                if shape.group_id is None
                else "{} ({})".format(shape.label, shape.group_id)
            )
            marker = "●"
            if str(getattr(shape, "shape_type", "") or "").lower() == "point":
                visibility = keypoint_visibility_from_shape_object(shape)
                marker = "○" if visibility == int(
                    KeypointVisibility.OCCLUDED) else "●"
            item.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">{}</font>'.format(
                    html.escape(base_text), r, g, b, marker
                )
            )

    def set_selected_keypoint_visibility(self, visible: bool) -> None:
        shapes = [
            s for s in self._selected_shapes_for_keypoint_visibility()
            if str(getattr(s, "shape_type", "") or "").lower() == "point"
        ]
        if not shapes:
            self.statusBar().showMessage(
                "Select one or more keypoint (point) shapes first."
            )
            return
        target = KeypointVisibility.VISIBLE if visible else KeypointVisibility.OCCLUDED
        for shape in shapes:
            set_keypoint_visibility_on_shape_object(shape, int(target))
        self._refresh_label_list_items_for_shapes(shapes)
        self.canvas.update()
        self.setDirty()

    def toggle_selected_keypoint_visibility(self) -> None:
        shapes = [
            s for s in self._selected_shapes_for_keypoint_visibility()
            if str(getattr(s, "shape_type", "") or "").lower() == "point"
        ]
        if not shapes:
            self.statusBar().showMessage(
                "Select one or more keypoint (point) shapes first."
            )
            return
        for shape in shapes:
            current = keypoint_visibility_from_shape_object(shape)
            target = (
                KeypointVisibility.VISIBLE
                if current == int(KeypointVisibility.OCCLUDED)
                else KeypointVisibility.OCCLUDED
            )
            set_keypoint_visibility_on_shape_object(shape, int(target))
        self._refresh_label_list_items_for_shapes(shapes)
        self.canvas.update()
        self.setDirty()

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

    def _save_ai_mask_renders(self, image_filename: str) -> None:
        """
        Persist a masked version of the current frame (and per-mask cutouts)
        where everything outside the AI mask shapes is painted black.
        """
        if not image_filename or self.labelList is None:
            return

        def _qimage_to_np(qimage_obj):
            try:
                return convert_qt_image_to_rgb_cv_image(qimage_obj).copy()
            except Exception:
                return None

        base_image = None
        try:
            if isinstance(self.imageData, QtGui.QImage):
                base_image = _qimage_to_np(self.imageData)
            elif isinstance(self.imageData, (bytes, bytearray)):
                base_image = utils.img_data_to_arr(
                    self.imageData).copy()
            elif isinstance(self.imageData, np.ndarray):
                base_image = np.asarray(self.imageData).copy()
        except Exception as exc:
            logger.warning(
                f"Unable to convert image for AI mask export: {exc}")

        if base_image is None:
            canvas_pixmap = getattr(self.canvas, "pixmap", None)
            if canvas_pixmap is not None and not canvas_pixmap.isNull():
                base_image = _qimage_to_np(canvas_pixmap.toImage())

        if base_image is None:
            logger.debug(
                "Skipping AI mask render save: unsupported image data.")
            return

        mask_shapes = []

        def _maybe_add_shape(shape):
            if (
                shape is not None
                and getattr(shape, "shape_type", None) == "mask"
                and getattr(shape, "mask", None) is not None
                and len(getattr(shape, "points", [])) >= 1
            ):
                mask_shapes.append(shape)

        try:
            if self.labelList:
                for item in self.labelList:
                    shape_obj = item.shape() if item is not None else None
                    _maybe_add_shape(shape_obj)
        except Exception as exc:
            logger.warning(
                f"Failed to collect AI mask shapes from label list: {exc}")

        if not mask_shapes and getattr(self.canvas, "shapes", None):
            for shape in self.canvas.shapes:
                _maybe_add_shape(shape)

        if not mask_shapes:
            return

        def paste_mask(mask_arr, top_left_point, canvas):
            mask_arr = np.asarray(mask_arr).astype(np.uint8)
            if mask_arr.size == 0:
                return
            if mask_arr.ndim > 2:
                mask_arr = mask_arr[..., 0]
            x1 = max(int(round(top_left_point.x())), 0)
            y1 = max(int(round(top_left_point.y())), 0)
            if x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
                return
            h, w = mask_arr.shape[:2]
            x2 = min(x1 + w, canvas.shape[1])
            y2 = min(y1 + h, canvas.shape[0])
            if x2 <= x1 or y2 <= y1:
                return
            crop_w = x2 - x1
            crop_h = y2 - y1
            canvas[y1:y2, x1:x2] = np.maximum(
                canvas[y1:y2, x1:x2],
                mask_arr[:crop_h, :crop_w].astype(np.uint8),
            )

        combined_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        for shape in mask_shapes:
            paste_mask(shape.mask, shape.points[0], combined_mask)

        if not combined_mask.any():
            return

        stem = Path(image_filename).stem
        base_dir = Path(image_filename).parent

        def save_masked_image(mask_array, suffix):
            masked_image = np.zeros_like(base_image)
            mask_bool = mask_array.astype(bool)
            masked_image[mask_bool] = base_image[mask_bool]
            if suffix:
                out_name = f"{stem}_{suffix}_mask.png"
            else:
                out_name = f"{stem}_mask.png"
            out_path = base_dir / out_name
            try:
                Image.fromarray(masked_image).save(str(out_path))
            except Exception as exc:
                logger.warning(
                    f"Failed to save AI mask render {out_path}: {exc}")

        # Save combined masked frame.
        save_masked_image(combined_mask, "")

        # Save per-shape masked frames to make individual cutouts.
        for idx, shape in enumerate(mask_shapes):
            per_mask = np.zeros_like(combined_mask)
            paste_mask(shape.mask, shape.points[0], per_mask)
            if not per_mask.any():
                continue
            safe_label = re.sub(
                r"[^0-9A-Za-z_-]", "_", shape.label or ""
            )
            if not safe_label:
                safe_label = f"mask_{idx+1}"
            save_masked_image(per_mask, f"{safe_label}")

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

    def _resolve_dino_kpseg_weight(self, model_weight: str) -> Optional[str]:
        raw = str(model_weight or "").strip()
        if raw:
            try:
                p = Path(raw).expanduser()
                if p.is_absolute() and p.exists():
                    return str(p.resolve())
                if not p.is_absolute():
                    resolved = p.resolve()
                    if resolved.exists():
                        return str(resolved)
            except Exception:
                pass

        try:
            saved = self.settings.value(
                "ai/dino_kpseg_last_best", "", type=str)
        except Exception:
            saved = ""
        if saved:
            try:
                p = Path(saved).expanduser().resolve()
                if p.exists():
                    return str(p)
            except Exception:
                pass

        try:
            latest = find_latest_checkpoint(task="dino_kpseg", model="train")
            if latest is not None:
                return str(latest)
        except Exception:
            pass

        return None

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
    def _is_dino_kpseg_tracker_model(identifier: str, weight: str) -> bool:
        key = f"{identifier or ''} {weight or ''}".lower()
        return (
            identifier.lower() == "dino_kpseg_tracker"
            or "dino_kpseg_tracker" in key
            or "dinokpseg_tracker" in key
        )

    @staticmethod
    def _is_dino_kpseg_model(identifier: str, weight: str) -> bool:
        identifier = (identifier or "").lower()
        weight = (weight or "").lower()
        key = f"{identifier} {weight}"
        if "dino_kpseg_tracker" in key or "dinokpseg_tracker" in key:
            return False
        return (
            identifier == "dino_kpseg"
            or "dino_kpseg" in key
            or "dinokpseg" in key
            or "kpseg" in key
        )

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
    def _is_efficienttam_model(identifier: str, weight: str) -> bool:
        """
        Detect EfficientTAM models based on identifier/weight strings.
        """
        key = f"{identifier or ''} {weight or ''}".lower()
        return "efficienttam" in key

    def stop_prediction(self):
        worker = getattr(self, "pred_worker", None)
        thread = getattr(self, "seg_pred_thread", None)

        # Update UI immediately and request a cooperative stop (non-blocking).
        self._prediction_stop_requested = True
        self.stop_prediction_flag = False
        try:
            self.stepSizeWidget.predict_button.setText("Stopping...")
            self.stepSizeWidget.predict_button.setStyleSheet(
                "background-color: orange; color: white;")
            self.stepSizeWidget.predict_button.setEnabled(False)
        except Exception:
            pass

        if worker is None:
            self._finalize_prediction_progress("Stop requested (no worker).")
            return

        try:
            if hasattr(worker, "request_stop"):
                worker.request_stop()
            else:
                worker.stop_signal.emit()
        except Exception:
            logger.debug("Failed to signal prediction worker stop.",
                         exc_info=True)

        try:
            if thread is not None and hasattr(thread, "requestInterruption"):
                thread.requestInterruption()
        except Exception:
            pass

        try:
            if thread is not None:
                thread.quit()
        except Exception:
            pass

        # If a non-cooperative task ignores stop requests, don't hang the UI;
        # attempt a best-effort force-stop after a short grace period.
        self._force_stop_thread_ref = thread
        QtCore.QTimer.singleShot(8000, self._force_stop_prediction_thread)
        logger.info("Prediction stop requested.")

    def _force_stop_prediction_thread(self):
        """Last-resort termination for stuck prediction threads."""
        thread = getattr(self, "seg_pred_thread", None)
        if getattr(self, "_force_stop_thread_ref", None) is not thread:
            return
        worker = getattr(self, "pred_worker", None)
        if thread is None or not isinstance(thread, QtCore.QThread):
            return
        if not thread.isRunning():
            return

        logger.warning(
            "Prediction thread did not stop in time; terminating as a last resort.")
        try:
            thread.terminate()
            thread.wait(2000)
        except Exception:
            logger.debug("Failed to terminate prediction thread.",
                         exc_info=True)
        try:
            if worker is not None:
                worker.deleteLater()
        except Exception:
            pass
        try:
            thread.deleteLater()
        except Exception:
            pass
        self.pred_worker = None
        self.seg_pred_thread = None
        self._force_stop_thread_ref = None
        self._finalize_prediction_progress("Prediction force-stopped.")

    def _cleanup_prediction_worker(self):
        """Clear references once the prediction thread has fully finished."""
        try:
            thread = getattr(self, "seg_pred_thread", None)
            if isinstance(thread, QtCore.QThread) and thread.isRunning():
                return
        except Exception:
            pass
        self.pred_worker = None
        self.seg_pred_thread = None
        self._force_stop_thread_ref = None

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

    def _max_predicted_frame_index(self, folder: Path) -> int:
        """Return the maximum frame index present in the prediction folder.

        Supports both "<folder>_000000123.json" and legacy "000000123.json" files,
        plus AnnotationStore frames if present.
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

    def predict_from_next_frame(self, to_frame=60):
        """
        Updated prediction routine that extracts visual prompts from the canvas.
        If the current model supports visual prompts (e.g. YOLOE), the prompts are extracted
        from the canvas rectangle shapes and passed to the inference module.
        """
        model_config, model_identifier, model_weight = self._resolve_model_identity()
        model_name = model_identifier or model_weight
        text_prompt = self._current_text_prompt()
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
            QtWidgets.QMessageBox.about(self,
                                        "No Shapes or Labeled Frames",
                                        "Please label this frame")
            return

        if self.video_file:

            self._prediction_stop_requested = False

            if self._is_dino_kpseg_tracker_model(model_name, model_weight):
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
                fresh_tracker_config = copy.deepcopy(
                    self.tracker_runtime_config)
                self.video_processor = DinoKPSEGVideoProcessor(
                    video_path=self.video_file,
                    result_folder=self.video_results_folder,
                    kpseg_weights=resolved,
                    device=None,
                    runtime_config=fresh_tracker_config,
                )
            elif self._is_dino_keypoint_model(model_name, model_weight):
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
            elif self._is_efficienttam_model(model_name, model_weight):
                from annolid.segmentation.SAM.sam_v2 import (
                    process_video_efficienttam,
                )

                model_key = Path(
                    model_weight).stem if model_weight else "efficienttam_s"
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
                pose_keypoint_names = None
                pose_schema_path = None
                if getattr(self, "_pose_schema", None) is not None and getattr(self._pose_schema, "keypoints", None):
                    # Keep a single canonical keypoint list; instances are represented
                    # via per-object grouping (track id / group_id), not name prefixes.
                    pose_keypoint_names = list(self._pose_schema.keypoints)
                if getattr(self, "_pose_schema_path", None):
                    pose_schema_path = self._pose_schema_path
                self.video_processor = InferenceProcessor(model_name=model_weight,
                                                          model_type="yolo",
                                                          class_names=class_names,
                                                          keypoint_names=pose_keypoint_names,
                                                          pose_schema_path=pose_schema_path)
            elif self._is_dino_kpseg_model(model_name, model_weight):
                from annolid.segmentation.yolos import InferenceProcessor

                pose_keypoint_names = None
                pose_schema_path = None
                if getattr(self, "_pose_schema", None) is not None and getattr(self._pose_schema, "keypoints", None):
                    # Keep a single canonical keypoint list; DinoKPSEG predicts one
                    # set per instance crop and instances are separated by group_id.
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
                    cached = getattr(
                        self, "_dinokpseg_inference_processor", None)
                    if (
                        cached is not None
                        and getattr(cached, "model_type", "").lower() == "dinokpseg"
                        and str(getattr(cached, "model_name", "")) == resolved_path
                    ):
                        # Reuse the heavy DinoKPSEGPredictor (DINOv3 backbone) across repeated runs.
                        cached.keypoint_names = pose_keypoint_names or getattr(
                            cached, "keypoint_names", None
                        )
                        self.video_processor = cached
                    else:
                        self.video_processor = InferenceProcessor(
                            model_name=resolved_path,
                            model_type="dinokpseg",
                            keypoint_names=pose_keypoint_names,
                            pose_schema_path=pose_schema_path,
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
                from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor
                from annolid.motion.optical_flow import optical_flow_settings_from
                flow_settings = optical_flow_settings_from(
                    self.optical_flow_manager)
                self.video_processor = VideoProcessor(
                    self.video_file,
                    model_name=model_name,
                    save_image_to_disk=False,
                    epsilon_for_polygon=self.epsilon_for_polygon,
                    t_max_value=self.t_max_value,
                    use_cpu_only=self.use_cpu_only,
                    auto_recovery_missing_instances=self.auto_recovery_missing_instances,
                    save_video_with_color_mask=self.save_video_with_color_mask,
                    **flow_settings,
                    results_folder=str(self.video_results_folder)
                    if self.video_results_folder else None,
                )
            if getattr(self, "seg_pred_thread", None) is not None:
                try:
                    if self.seg_pred_thread.isRunning():
                        logger.warning(
                            "Prediction thread already running; stop it before starting a new run.")
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
            inference_step = 1
            inference_start_frame = max(0, int(self.frame_number or 0) + 1)
            inference_end_frame = None  # default: run to end for YOLO/DinoKPSEG inference
            if self.video_results_folder:
                try:
                    results_folder = Path(self.video_results_folder)
                    if should_start_predictions_from_frame0(results_folder):
                        inference_start_frame = 0
                    else:
                        max_existing = self._max_predicted_frame_index(
                            results_folder
                        )
                        if max_existing >= int(inference_start_frame):
                            inference_start_frame = int(max_existing) + 1
                except Exception:
                    pass
            watch_start_frame = int(self.frame_number or 0)
            if self._is_dino_kpseg_tracker_model(model_name, model_weight):
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
                watch_start_frame = int(self.frame_number or 0)
            elif self._is_dino_keypoint_model(model_name, model_weight):
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
                watch_start_frame = int(self.frame_number or 0)
            elif self._is_efficienttam_model(model_name, model_weight):
                frame_idx = max(self.frame_number, 0)
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor,
                    frame_idx=frame_idx,
                )
                watch_start_frame = int(frame_idx)
            elif self.sam2_manager.is_sam2_model(model_name, model_weight):
                frame_idx = max(self.frame_number, 0)
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
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.run_inference,
                    source=self.video_file,
                    start_frame=int(inference_start_frame),
                    end_frame=inference_end_frame,
                    step=int(inference_step),
                    skip_existing=True,
                )
                watch_start_frame = int(inference_start_frame)
            elif self._is_yolo_model(model_name, model_weight):
                # Pass visual_prompts to run_inference if extracted successfully.
                self.pred_worker = FlexibleWorker(
                    task_function=self.video_processor.run_inference,
                    source=self.video_file,
                    visual_prompts=visual_prompts if visual_prompts else None,
                    start_frame=int(inference_start_frame),
                    end_frame=inference_end_frame,
                    step=int(inference_step),
                    skip_existing=True,
                )
                watch_start_frame = int(inference_start_frame)
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
                watch_start_frame = int(self.frame_number + 1)

            if self.video_results_folder:  # video_results_folder is Path object
                try:
                    self._setup_prediction_folder_watcher(
                        str(self.video_results_folder),
                        start_frame=int(watch_start_frame),
                    )
                except Exception:
                    logger.debug(
                        "Failed to start prediction progress watcher.", exc_info=True
                    )
            try:
                self.frame_number = int(watch_start_frame)
            except Exception:
                pass
            logger.info("Prediction started from frame: %s",
                        int(watch_start_frame))
            self.stepSizeWidget.predict_button.setText("Stop")
            self.stepSizeWidget.predict_button.setStyleSheet(
                "background-color: red; color: white;")
            self.stop_prediction_flag = True
            self.pred_worker.moveToThread(self.seg_pred_thread)
            self.seg_pred_thread.started.connect(
                self.pred_worker.run, QtCore.Qt.QueuedConnection)
            self.pred_worker.result_signal.connect(
                self.lost_tracking_instance, QtCore.Qt.QueuedConnection)
            self.pred_worker.finished_signal.connect(
                self.predict_is_ready, QtCore.Qt.QueuedConnection)
            self.pred_worker.finished_signal.connect(
                self.seg_pred_thread.quit, QtCore.Qt.QueuedConnection)
            self.pred_worker.finished_signal.connect(
                self.pred_worker.deleteLater, QtCore.Qt.QueuedConnection)
            self.seg_pred_thread.finished.connect(
                self._cleanup_prediction_worker, QtCore.Qt.QueuedConnection)
            self.seg_pred_thread.finished.connect(
                self.seg_pred_thread.deleteLater, QtCore.Qt.QueuedConnection)
            self.seg_pred_thread.start()

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
            if isinstance(messege, Exception):
                logger.exception("Prediction worker failed", exc_info=messege)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Prediction failed",
                    f"Prediction failed with error:\n{messege}",
                )
                return
            message_text = ""
            stop_from_message = False
            if isinstance(messege, tuple):
                if messege:
                    message_text = str(messege[0])
                if len(messege) > 1 and isinstance(messege[1], bool):
                    stop_from_message = messege[1]
            elif messege is not None:
                message_text = str(messege)

            if message_text.startswith("Stopped") or "missing instance(s)" in message_text:
                stop_from_message = True

            if message_text and "last frame" in message_text:
                stop_from_message = True
                QtWidgets.QMessageBox.information(
                    self, "Stop early",
                    message_text
                )
            if self._prediction_stop_requested or stop_from_message:
                logger.info(
                    "Prediction stopped early; skipping tracking CSV conversion.")
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
                    logger.info(
                        "Predicted frames available: max_frame=%s of %s",
                        int(max_predicted),
                        int(self.num_frames or 0),
                    )
                    if (
                        self.num_frames
                        and int(max_predicted) >= int(self.num_frames) - 1
                        and int(max_predicted) >= 0
                    ):
                        self.convert_json_to_tracked_csv()
                    else:
                        logger.info(
                            "Prediction did not reach the last frame; skipping tracking CSV conversion."
                        )
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
        self.reset_predict_button()
        self._finalize_prediction_progress(
            "Manual prediction worker finished.")
        self._prediction_stop_requested = False

    def reset_predict_button(self):
        """Reset the predict button text and style"""
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")

    def loadFlags(self, flags):
        """Delegate flag loading to the flags controller."""
        from annolid.utils.labelme_flags import sanitize_labelme_flags

        self.flags_controller.load_flags(sanitize_labelme_flags(flags))

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
            self._auto_collect_labelme_pair(filename, image_filename)
            self._save_ai_mask_renders(image_filename)
            self.imageList.append(image_filename)
            self.addRecentFile(filename)
            label_file = self._getLabelFile(filename)
            self._addItem(image_filename, label_file)

            if self.caption_widget is not None:
                self.caption_widget.set_image_path(image_filename)

            self.setClean()

    def _auto_collect_labelme_pair(self, json_path: str, image_path: str) -> None:
        index_file = os.environ.get("ANNOLID_LABEL_INDEX_FILE", "").strip()
        if not index_file:
            index_file = (
                (self.config or {}).get("label_index_file")
                or self.settings.value("dataset/label_index_file", "", type=str)
            )
        if not index_file:
            dataset_root = (
                os.environ.get("ANNOLID_LABEL_COLLECTION_DIR")
                or (self.config or {}).get("label_collection_dir")
                or self.settings.value("dataset/label_collection_dir", "", type=str)
            )
            if not dataset_root:
                return
            try:
                from annolid.datasets.labelme_collection import default_label_index_path
            except Exception:
                index_file = str(Path(dataset_root) /
                                 "annolid_logs" / "annolid_dataset.jsonl")
            else:
                index_file = str(default_label_index_path(Path(dataset_root)))

        include_empty_value = os.environ.get(
            "ANNOLID_LABEL_INDEX_INCLUDE_EMPTY", "0").strip().lower()
        include_empty = include_empty_value in {"1", "true", "yes", "on"}

        try:
            from annolid.datasets.labelme_collection import index_labelme_pair

            index_labelme_pair(
                json_path=Path(json_path),
                index_file=Path(index_file),
                image_path=Path(image_path) if image_path else None,
                include_empty=include_empty,
                source="annolid_gui",
            )
        except Exception as exc:
            logger.warning(
                "Auto label indexing failed for %s: %s", json_path, exc)

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
                    self.caption_widget.behavior_widget.set_current_frame(
                        self.frame_number)
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
                self.seekbar.removeMarksByType("predicted_existing")
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

    def _collect_seed_frames(self, prediction_folder: Path) -> Set[int]:
        seed_frames: Set[int] = set()
        pattern = re.compile(r"(\d+)(?=\.(png|jpg|jpeg)$)", re.IGNORECASE)
        for path in prediction_folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            match = pattern.search(path.name)
            if not match:
                continue
            try:
                seed_frames.add(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return seed_frames

    def deletePredictionsFromSeedToNext(self):
        """
        Delete predicted frames starting from the current seed frame up to the
        next seed frame (exclusive), keeping manual labels intact.
        """
        if not self.video_loader or not self.video_results_folder:
            return False, None, None

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return False, None, None

        seed_frames = sorted(self._collect_seed_frames(prediction_folder))
        protected_frames: Set[int] = set(seed_frames)

        current_seed = None
        if seed_frames:
            for seed in seed_frames:
                if seed <= self.frame_number:
                    current_seed = seed
                else:
                    break
        if current_seed is None:
            current_seed = self.frame_number

        next_seed = None
        for seed in seed_frames:
            if seed > current_seed:
                next_seed = seed
                break

        if next_seed is not None and next_seed - 1 < current_seed:
            return False, current_seed, next_seed

        deleted_files = 0
        logger.info(
            "Deleting predictions from seed frame %s to %s.",
            current_seed,
            next_seed if next_seed is not None else "end",
        )

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)
            if not match:
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                continue

            if frame_number < current_seed:
                continue
            if next_seed is not None and frame_number >= next_seed:
                continue
            if frame_number in protected_frames:
                continue

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
            store_removed = store.remove_frames_in_range(
                current_seed,
                next_seed - 1 if next_seed is not None else None,
                protected_frames=protected_frames,
            )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("predicted_existing")
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
                "No predicted files required removal for the current seed range."
            )

        return bool(deleted_files or store_removed), current_seed, next_seed

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "Or delete predicted label files from the current seed frame "
            "to the next seed frame. "
            "What would you like to do?"
        )
        msg_box = mb(self)
        msg_box.setIcon(mb.Warning)
        msg_box.setText(msg)
        msg_box.setInformativeText(
            self.tr("Yes: delete the current label file. "
                    "Yes to All: delete predicted frames for the current seed range.")
        )
        msg_box.setStandardButtons(mb.No | mb.Yes | mb.YesToAll)
        msg_box.setDefaultButton(mb.No)
        answer = msg_box.exec_()

        if answer == mb.No:
            return
        elif answer == mb.YesToAll:
            removed, start_seed, next_seed = self.deletePredictionsFromSeedToNext()
            if removed:
                msg = self.tr(
                    "Delete all remaining predicted frames after this seed range?"
                )
                follow_up = mb.question(
                    self,
                    self.tr("Delete All Predictions"),
                    msg,
                    mb.Yes | mb.No,
                    mb.No,
                )
                if follow_up == mb.Yes:
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
            tracked_csv_path = out_folder.parent / \
                f"{out_folder.name}_tracked.csv"
            self.csv_worker = FlexibleWorker(
                task_function=labelme2csv.convert_json_to_csv,
                json_folder=str(out_folder),
                csv_file=str(csv_output_path),
                tracked_csv_file=str(tracked_csv_path),
                fps=self.fps,
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

        if result == "Stopped":
            self._cleanup_csv_worker()
            return

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
            try:
                worker.report_progress(progress)
            except RuntimeError:
                logger.debug(
                    "CSV progress update skipped (worker deleted).",
                    exc_info=True,
                )

    def _update_progress_bar(self, progress):
        """Update the progress bar's value."""
        self.progress_bar.setValue(progress)

    # method to hide progress bar
    def _finalize_prediction_progress(self, message=""):
        logger.info(f"Prediction finalization: {message}")
        if hasattr(self, 'progress_bar') and self.progress_bar.isVisible():
            self.statusBar().removeWidget(self.progress_bar)
        self._stop_prediction_folder_watcher()
        # Clear prediction-related marks from the slider
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")  # Use the new method
            self.seekbar.removeMarksByType("predicted_existing")
            self.seekbar.removeMarksByType("prediction_progress")
            self._prediction_progress_mark = None

        # Reset button state (already in predict_is_ready and lost_tracking_instance)
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False  # This flag is specific to AnnolidWindow

    def _setup_prediction_folder_watcher(self, folder_path_to_watch, *, start_frame: int | None = None):
        if self.prediction_progress_watcher is None:
            self.prediction_progress_watcher = QtCore.QTimer(self)
            self.prediction_progress_watcher.timeout.connect(
                self._handle_prediction_folder_change
            )

        if osp.isdir(folder_path_to_watch):
            self.prediction_progress_folder = folder_path_to_watch
            self.prediction_start_timestamp = time.time()
            if start_frame is None:
                start_frame = int(
                    self.frame_number) if self.frame_number is not None else 0
            self._prediction_start_frame = max(0, int(start_frame))
            self._prediction_existing_store_frames = set()
            self._prediction_existing_json_frames = set()
            path = Path(folder_path_to_watch)
            prefixed_pattern = re.compile(r'_(\d{9,})\.json$')
            bare_pattern = re.compile(r'^(\d{9,})\.json$')
            try:
                for f_name in os.listdir(path):
                    if not f_name.endswith(".json"):
                        continue
                    match = None
                    if path.name in f_name:
                        match = prefixed_pattern.search(f_name)
                    if match is None:
                        match = bare_pattern.match(f_name)
                    if match:
                        try:
                            frame_num = int(float(match.group(1)))
                            self._prediction_existing_json_frames.add(
                                frame_num)
                        except (ValueError, IndexError):
                            continue
            except OSError as exc:
                logger.debug(
                    "Failed to read existing prediction JSONs in %s: %s",
                    path,
                    exc,
                )
            try:
                store = AnnotationStore.for_frame_path(
                    path / f"{path.name}_000000000.json"
                )
                if store.store_path.exists():
                    self._prediction_existing_store_frames = set(
                        store.iter_frames())
            except Exception:
                self._prediction_existing_store_frames = set()
            self.prediction_progress_watcher.start(1000)  # Poll every 1000 ms
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
            prefixed_pattern = re.compile(r'_(\d{9,})\.json$')
            bare_pattern = re.compile(r'^(\d{9,})\.json$')
            prediction_active = bool(self.prediction_start_timestamp)

            # --- 1. Efficiently Scan and Parse All Relevant Frame Numbers ---
            all_frame_nums_set: set[int] = set()
            for f_name in os.listdir(path):
                # The check `self.video_results_folder.name in f_name` is kept for consistency
                if not f_name.endswith(".json"):
                    continue
                match = None
                if self.video_results_folder.name in f_name:
                    match = prefixed_pattern.search(f_name)
                if match is None:
                    match = bare_pattern.match(f_name)
                if match is None:
                    continue
                file_path = path / f_name
                if self.prediction_start_timestamp:
                    try:
                        if file_path.stat().st_mtime < self.prediction_start_timestamp:
                            continue
                    except FileNotFoundError:
                        continue
                try:
                    # Convert via float to handle cases like "123.0"
                    frame_num = int(float(match.group(1)))
                    all_frame_nums_set.add(frame_num)
                except (ValueError, IndexError):
                    continue  # Skip malformed numbers

            all_frame_nums: list[int] = []
            if not all_frame_nums_set:
                store = AnnotationStore.for_frame_path(
                    path / f"{path.name}_000000000.json")
                if store.store_path.exists():
                    store_frames = sorted(store.iter_frames())
                    if prediction_active and self._prediction_existing_store_frames:
                        store_frames = [
                            frame for frame in store_frames
                            if frame not in self._prediction_existing_store_frames
                        ]
                    all_frame_nums = store_frames
            else:
                all_frame_nums = sorted(all_frame_nums_set)
            existing_frame_set = set()
            if prediction_active:
                existing_frame_set.update(
                    self._prediction_existing_json_frames)
                if self._prediction_existing_store_frames:
                    existing_frame_set.update(
                        self._prediction_existing_store_frames)
                if all_frame_nums:
                    existing_frame_set.difference_update(all_frame_nums)
            existing_frame_nums = sorted(existing_frame_set)

            num_total_frames = len(all_frame_nums)

            # --- 2. Dynamic Marker Decimation Logic ---
            # Define the threshold at which we start thinning the markers
            DECIMATION_THRESHOLD = 2000

            def decimate_frames(frame_nums):
                if not frame_nums:
                    return []
                if len(frame_nums) < DECIMATION_THRESHOLD:
                    return frame_nums
                step = 100 if len(frame_nums) > 10000 else 20
                decimated = frame_nums[::step]
                # Always ensure the very last frame is included to show completion
                if frame_nums[-1] not in decimated:
                    decimated.append(frame_nums[-1])
                return decimated

            frames_to_mark = decimate_frames(all_frame_nums)
            existing_frames_to_mark = decimate_frames(existing_frame_nums)

            if not frames_to_mark and not existing_frames_to_mark:
                if prediction_active and self._prediction_start_frame is not None:
                    start_frame = self._prediction_start_frame
                    if 0 <= start_frame < self.num_frames:
                        self.seekbar.removeMarksByType("prediction_progress")
                        progress_mark = VideoSliderMark(
                            mark_type="prediction_progress",
                            val=start_frame
                        )
                        self.seekbar.addMark(progress_mark)
                        self._prediction_progress_mark = progress_mark
                        if self.frame_number != start_frame:
                            self.set_frame_number(start_frame)
                        self.seekbar.setValue(start_frame)
                        self._update_progress_bar(0)
                return

            # --- 3. Update the GUI Efficiently ---
            # Get existing markers once to avoid repeated calls inside the loop
            existing_predicted_vals = {
                mark.val for mark in self.seekbar.getMarks() if mark.mark_type == "predicted"
            }
            existing_existing_vals = {
                mark.val for mark in self.seekbar.getMarks() if mark.mark_type == "predicted_existing"
            }

            # Block signals to prevent the UI from trying to update thousands of times
            self.seekbar.blockSignals(True)

            new_marks_added = False
            for frame_num in existing_frames_to_mark:
                if 0 <= frame_num < self.num_frames:
                    if frame_num in existing_existing_vals or frame_num in existing_predicted_vals:
                        continue
                    existing_mark = VideoSliderMark(
                        mark_type="predicted_existing", val=frame_num
                    )
                    self.seekbar.addMark(existing_mark)
                    existing_existing_vals.add(frame_num)
                    new_marks_added = True
            for frame_num in frames_to_mark:
                if 0 <= frame_num < self.num_frames:
                    if frame_num in existing_predicted_vals:
                        continue
                    if frame_num in existing_existing_vals:
                        for mark in self.seekbar.getMarksAtVal(frame_num):
                            if mark.mark_type == "predicted_existing":
                                self.seekbar.removeMark(mark)
                        existing_existing_vals.discard(frame_num)
                    pred_mark = VideoSliderMark(
                        mark_type="predicted", val=frame_num)
                    self.seekbar.addMark(pred_mark)
                    existing_predicted_vals.add(frame_num)
                    new_marks_added = True

            # Re-enable signals and force a single repaint if we added anything
            self.seekbar.blockSignals(False)
            if new_marks_added:
                self.seekbar.update()

            # Update the progress bar and slider position to the latest actual frame
            if all_frame_nums:
                latest_frame = all_frame_nums[-1]
                if prediction_active:
                    self.last_known_predicted_frame = latest_frame
                else:
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
            elif prediction_active and self._prediction_start_frame is not None:
                start_frame = self._prediction_start_frame
                if 0 <= start_frame < self.num_frames:
                    self.seekbar.removeMarksByType("prediction_progress")
                    progress_mark = VideoSliderMark(
                        mark_type="prediction_progress",
                        val=start_frame
                    )
                    self.seekbar.addMark(progress_mark)
                    self._prediction_progress_mark = progress_mark
                    if self.frame_number != start_frame:
                        self.set_frame_number(start_frame)
                    self.seekbar.setValue(start_frame)
                    self._update_progress_bar(0)

        except Exception as e:
            logger.error(
                f"Error scanning prediction folder for slider marks: {e}", exc_info=True)

    @QtCore.Slot()
    def _handle_prediction_folder_change(self):
        path = self.video_results_folder
        if path:
            logger.debug(f"Scanning prediction folder: {path}.")
            self._scan_prediction_folder(str(path))

    def _stop_prediction_folder_watcher(self):
        if self.prediction_progress_watcher:
            self.prediction_progress_watcher.stop()
            logger.info("Prediction progress watcher stopped.")
        self.prediction_progress_folder = None
        self.last_known_predicted_frame = -1  # Reset
        self.prediction_start_timestamp = 0.0
        self._prediction_start_frame = None
        self._prediction_existing_store_frames = set()
        self._prediction_existing_json_frames = set()
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
        self.seekbar.removeMarksByType("predicted_existing")

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

    def _stop_csv_worker(self):
        """Request a graceful stop of any active CSV conversion."""
        if self._csv_conversion_queue:
            self._csv_conversion_queue.clear()

        worker = getattr(self, "csv_worker", None)
        if worker is not None:
            try:
                worker.request_stop()
            except RuntimeError:
                logger.debug("CSV worker already deleted.", exc_info=True)

        thread = getattr(self, "csv_thread", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait(2000)
            except RuntimeError:
                logger.debug("CSV thread already cleaned up.", exc_info=True)

        self._cleanup_csv_worker()

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
                if torch is None or not torch.cuda.is_available():
                    QtWidgets.QMessageBox.about(
                        self,
                        "GPU or PyTorch unavailable",
                        "PyTorch with CUDA support is required to run YOLACT tracking.",
                    )
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
            yolo_device = getattr(dlg, "yolo_device", None)
            yolo_plots = getattr(dlg, "yolo_plots", False)
            yolo_train_overrides = dlg.get_yolo_train_overrides() if hasattr(
                dlg, "get_yolo_train_overrides") else {}
            dino_model_name = getattr(dlg, "dino_model_name", None)
            dino_short_side = getattr(dlg, "dino_short_side", 768)
            dino_layers = getattr(dlg, "dino_layers", "-1")
            dino_radius_px = getattr(dlg, "dino_radius_px", 6.0)
            dino_hidden_dim = getattr(dlg, "dino_hidden_dim", 128)
            dino_head_type = getattr(dlg, "dino_head_type", "conv")
            dino_attn_heads = getattr(dlg, "dino_attn_heads", 4)
            dino_attn_layers = getattr(dlg, "dino_attn_layers", 1)
            dino_lr_pair_loss_weight = getattr(
                dlg, "dino_lr_pair_loss_weight", 0.0)
            dino_lr_pair_margin_px = getattr(
                dlg, "dino_lr_pair_margin_px", 0.0)
            dino_lr_side_loss_weight = getattr(
                dlg, "dino_lr_side_loss_weight", 0.0)
            dino_lr_side_loss_margin = getattr(
                dlg, "dino_lr_side_loss_margin", 0.0)
            dino_lr = getattr(dlg, "dino_lr", 0.002)
            dino_threshold = getattr(dlg, "dino_threshold", 0.4)
            dino_bce_type = getattr(dlg, "dino_bce_type", "bce")
            dino_focal_alpha = getattr(dlg, "dino_focal_alpha", 0.25)
            dino_focal_gamma = getattr(dlg, "dino_focal_gamma", 2.0)
            dino_coord_warmup_epochs = getattr(
                dlg, "dino_coord_warmup_epochs", 0)
            dino_radius_schedule = getattr(dlg, "dino_radius_schedule", "none")
            dino_radius_start_px = getattr(
                dlg, "dino_radius_start_px", dino_radius_px)
            dino_radius_end_px = getattr(
                dlg, "dino_radius_end_px", dino_radius_px)
            dino_overfit_n = getattr(dlg, "dino_overfit_n", 0)
            dino_cache_features = getattr(dlg, "dino_cache_features", True)
            dino_patience = getattr(dlg, "dino_patience", 0)
            dino_min_delta = getattr(dlg, "dino_min_delta", 0.0)
            dino_min_epochs = getattr(dlg, "dino_min_epochs", 0)
            dino_best_metric = getattr(dlg, "dino_best_metric", "pck@8px")
            dino_early_stop_metric = getattr(
                dlg, "dino_early_stop_metric", "auto")
            dino_pck_weighted_weights = getattr(
                dlg, "dino_pck_weighted_weights", "1,1,1,1")
            dino_augment_enabled = getattr(dlg, "dino_augment_enabled", False)
            dino_hflip_prob = getattr(dlg, "dino_hflip_prob", 0.5)
            dino_degrees = getattr(dlg, "dino_degrees", 0.0)
            dino_translate = getattr(dlg, "dino_translate", 0.0)
            dino_scale = getattr(dlg, "dino_scale", 0.0)
            dino_brightness = getattr(dlg, "dino_brightness", 0.0)
            dino_contrast = getattr(dlg, "dino_contrast", 0.0)
            dino_saturation = getattr(dlg, "dino_saturation", 0.0)
            dino_seed = getattr(dlg, "dino_seed", -1)
            dino_tb_add_graph = getattr(dlg, "dino_tb_add_graph", False)
            dino_tb_projector = getattr(dlg, "dino_tb_projector", True)
            dino_tb_projector_split = getattr(
                dlg, "dino_tb_projector_split", "val")
            dino_tb_projector_max_images = getattr(
                dlg, "dino_tb_projector_max_images", 64)
            dino_tb_projector_max_patches = getattr(
                dlg, "dino_tb_projector_max_patches", 4000)
            dino_tb_projector_per_image_per_keypoint = getattr(
                dlg, "dino_tb_projector_per_image_per_keypoint", 3
            )
            dino_tb_projector_pos_threshold = getattr(
                dlg, "dino_tb_projector_pos_threshold", 0.35)
            dino_tb_projector_crop_px = getattr(
                dlg, "dino_tb_projector_crop_px", 96)
            dino_tb_projector_sprite_border_px = getattr(
                dlg, "dino_tb_projector_sprite_border_px", 3)
            dino_tb_projector_add_negatives = getattr(
                dlg, "dino_tb_projector_add_negatives", False)
            dino_tb_projector_neg_threshold = getattr(
                dlg, "dino_tb_projector_neg_threshold", 0.02)
            dino_tb_projector_negatives_per_image = getattr(
                dlg, "dino_tb_projector_negatives_per_image", 6
            )

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
                batch_size=batch_size,
                device=yolo_device,
                plots=yolo_plots,
                train_overrides=yolo_train_overrides,
                out_dir=out_dir,
            )

        elif algo == "DINO KPSEG":
            data_config = self.dino_kpseg_training_manager.prepare_data_config(
                config_file)
            if data_config is None:
                return
            self.dino_kpseg_training_manager.start_training(
                data_config_path=data_config,
                out_dir=out_dir,
                model_name=str(dino_model_name or ""),
                short_side=int(dino_short_side),
                layers=str(dino_layers or "-1"),
                radius_px=float(dino_radius_px),
                hidden_dim=int(dino_hidden_dim),
                lr=float(dino_lr),
                epochs=int(epochs),
                batch_size=int(batch_size),
                threshold=float(dino_threshold),
                bce_type=str(dino_bce_type or "bce"),
                focal_alpha=float(dino_focal_alpha),
                focal_gamma=float(dino_focal_gamma),
                coord_warmup_epochs=int(dino_coord_warmup_epochs),
                radius_schedule=str(dino_radius_schedule or "none"),
                radius_start_px=float(dino_radius_start_px),
                radius_end_px=float(dino_radius_end_px),
                overfit_n=int(dino_overfit_n),
                device=yolo_device,
                cache_features=bool(dino_cache_features),
                head_type=str(dino_head_type or "conv"),
                attn_heads=int(dino_attn_heads),
                attn_layers=int(dino_attn_layers),
                lr_pair_loss_weight=float(dino_lr_pair_loss_weight),
                lr_pair_margin_px=float(dino_lr_pair_margin_px),
                lr_side_loss_weight=float(dino_lr_side_loss_weight),
                lr_side_loss_margin=float(dino_lr_side_loss_margin),
                early_stop_patience=int(dino_patience),
                early_stop_min_delta=float(dino_min_delta),
                early_stop_min_epochs=int(dino_min_epochs),
                best_metric=str(dino_best_metric or "pck@8px"),
                early_stop_metric=str(dino_early_stop_metric or "auto"),
                pck_weighted_weights=str(
                    dino_pck_weighted_weights or "1,1,1,1"),
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
                tb_projector_split=str(dino_tb_projector_split or "val"),
                tb_projector_max_images=int(dino_tb_projector_max_images),
                tb_projector_max_patches=int(dino_tb_projector_max_patches),
                tb_projector_per_image_per_keypoint=int(
                    dino_tb_projector_per_image_per_keypoint),
                tb_projector_pos_threshold=float(
                    dino_tb_projector_pos_threshold),
                tb_projector_crop_px=int(dino_tb_projector_crop_px),
                tb_projector_sprite_border_px=int(
                    dino_tb_projector_sprite_border_px),
                tb_projector_add_negatives=bool(
                    dino_tb_projector_add_negatives),
                tb_projector_neg_threshold=float(
                    dino_tb_projector_neg_threshold),
                tb_projector_negatives_per_image=int(
                    dino_tb_projector_negatives_per_image),
            )

        elif algo == 'YOLACT':
            # start training models
            if torch is None or not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(
                    self,
                    "GPU or PyTorch unavailable",
                    "PyTorch with CUDA support is required to train YOLACT models.",
                )
                return

            subprocess.Popen(['annolid-train',
                              f'--config={config_file}',
                              f'--batch_size={batch_size}'])

            if out_dir is None:
                out_runs_dir = shared_runs_root()
            else:
                out_runs_dir = Path(out_dir) / Path(config_file).name / 'runs'

            out_runs_dir.mkdir(exist_ok=True, parents=True)
            process = start_tensorboard(log_dir=shared_runs_root())
            QtWidgets.QMessageBox.about(self,
                                        "Started",
                                        f"Results are in folder: \
                                         {str(out_runs_dir)}")

        elif algo == 'MaskRCNN':
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor
            dataset_dir = str(Path(config_file).parent)
            segmentor = Segmentor(dataset_dir, out_dir or str(shared_runs_root()),
                                  max_iterations=max_iterations,
                                  batch_size=batch_size,
                                  model_pth_path=model_path
                                  )
            out_runs_dir = segmentor.out_put_dir
            process = start_tensorboard(log_dir=shared_runs_root())
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

    def _on_frame_loaded(self, frame_idx: int, qimage: QtGui.QImage) -> None:
        """Render a frame only if it matches the latest requested index."""
        current = getattr(self, "frame_number", None)
        if current is not None and frame_idx != current:
            logger.debug(
                "Dropping stale frame %s (current=%s)", frame_idx, current
            )
            return
        frame_path = self._frame_image_path(frame_idx)
        self.image_to_canvas(qimage, frame_path, frame_idx)

    def _frame_image_path(self, frame_number: int) -> Path:
        if self.video_results_folder:
            return self.video_results_folder / \
                f"{str(self.video_results_folder.name)}_{frame_number:09}.png"
        if getattr(self, "filename", None):
            try:
                return Path(self.filename)
            except Exception:
                pass
        return Path()

    def set_frame_number(self, frame_number):
        if frame_number >= self.num_frames or frame_number < 0:
            return
        self.frame_number = frame_number
        self._update_audio_playhead(frame_number)
        if self.isPlaying and not self._suppress_audio_seek:
            audio_loader = self._active_audio_loader()
            if audio_loader:
                audio_loader.play(start_frame=frame_number)
        self.filename = str(self._frame_image_path(frame_number))
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
        self._configure_pose_schema_from_project()

    def _configure_pose_schema_from_project(self) -> None:
        schema = self.project_schema
        self._pose_schema = None
        self._pose_schema_path = None
        if schema is None:
            return

        embedded = getattr(schema, "pose_schema", None)
        schema_path_value = getattr(schema, "pose_schema_path", None)
        if embedded and isinstance(embedded, dict):
            try:
                self._pose_schema = PoseSchema.from_dict(embedded)
            except Exception:
                self._pose_schema = None

        if schema_path_value:
            try:
                p = Path(schema_path_value)
                if not p.is_absolute() and self.project_schema_path:
                    p = self.project_schema_path.parent / p
                self._pose_schema_path = str(p)
            except Exception:
                self._pose_schema_path = str(schema_path_value)

        try:
            self.canvas.setPoseSchema(self._pose_schema)
        except Exception:
            pass

    def toggle_pose_edges_display(self, checked: bool = False) -> None:
        """Toggle skeleton edge overlay for pose keypoints."""
        self._show_pose_edges = bool(checked)
        try:
            self.settings.setValue("pose/show_edges", self._show_pose_edges)
        except Exception:
            pass
        try:
            self.canvas.setShowPoseEdges(self._show_pose_edges)
        except Exception:
            pass

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
            self.openVideo(from_video_list=True,
                           video_path=str(dialog.downloaded_path))

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
            # If an audio-only dock is open, close it before switching to a video.
            self._cleanup_audio_ui()
            cur_video_folder = Path(video_filename).parent
            self.video_results_folder = Path(video_filename).with_suffix('')

            self.video_results_folder.mkdir(
                exist_ok=True,
                parents=True
            )
            self.annotation_dir = self.video_results_folder
            self.video_file = video_filename
            if getattr(self, "depth_manager", None) is not None:
                self.depth_manager.load_depth_ndjson_records()
            if getattr(self, "optical_flow_manager", None) is not None:
                self.optical_flow_manager.load_records(video_filename)
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

            # Configure frame loader before requesting any frames.
            self.frame_loader.video_loader = self.video_loader
            self.frame_loader.moveToThread(self.frame_worker)
            self.frame_loader.res_frame.connect(
                self._on_frame_loaded
            )
            if not self.frame_worker.isRunning():
                self.frame_worker.start(
                    priority=QtCore.QThread.IdlePriority)

            # load the first frame
            self.set_frame_number(self.frame_number)

            self.actions.openNextImg.setEnabled(True)

            self.actions.openPrevImg.setEnabled(True)
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
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.canvas.loadPixmap(pixmap)
        try:
            frame_rgb = convert_qt_image_to_rgb_cv_image(qimage).copy()
        except Exception:
            frame_rgb = None
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.update_overlay_for_frame(
                frame_number, frame_rgb)
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.update_overlay_for_frame(
                frame_number, frame_rgb)
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
        # Ensure shapes for the current frame are loaded using the emitted frame index.
        try:
            self.loadPredictShapes(frame_number, filename)
        except Exception:
            logger.debug("Failed to load shapes for frame %s", frame_number,
                         exc_info=True)
        # Refresh behavior and flag states for the newly loaded frame.
        self._refresh_behavior_overlay()
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
        if self._df_deeplabcut is not None:
            self._load_deeplabcut_results(frame_number)
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
            # Prompt user to select a 3D volume source (TIFF/NIfTI/DICOM)
            start_dir = str(Path(self.filename).parent) if getattr(
                self, "filename", None) else "."
            filters = self.tr(
                "3D sources (*.tif *.tiff *.ome.tif *.ome.tiff *.nii *.nii.gz *.dcm *.dicom *.ima *.IMA *.ply *.csv *.xyz *.stl *.STL *.obj *.OBJ *.zarr *.zarr.json *.zgroup);;All files (*.*)"
            )
            dialog = QtWidgets.QFileDialog(
                self, self.tr("Choose 3D Volume (TIFF/NIfTI/DICOM/Zarr)"))
            dialog.setDirectory(start_dir)
            dialog.setNameFilter(filters)
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
            paths: list[str] = []
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                paths = dialog.selectedFiles()
            if not paths:
                # Optionally select a folder (DICOM or Zarr)
                folder = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    self.tr("Choose Volume Folder (DICOM/Zarr)"),
                    start_dir,
                    QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.ReadOnly,
                )
                if folder:
                    paths = [folder]
                else:
                    return
            if paths:
                def _normalize_volume_selection(raw: str) -> str:
                    try:
                        p = Path(raw)
                        if p.is_file():
                            if p.name.lower() == "zarr.json":
                                return str(p.parent)
                            if (p.parent / ".zarray").exists():
                                return str(p.parent)
                        cur = p
                        for _ in range(3):
                            if cur.name.lower().endswith(".zarr") or (cur / ".zarray").exists() or (cur / "zarr.json").exists():
                                return str(cur)
                            cur = cur.parent
                    except Exception:
                        pass
                    return raw

                tiff_path = _normalize_volume_selection(paths[0])

        # Prefer true 3D (VTK) if available, else fallback to slice/MIP viewer
        vtk_missing = False
        vtk_error = None
        try:
            from annolid.gui.widgets.vtk_volume_viewer import VTKVolumeViewerDialog  # type: ignore
            dlg = VTKVolumeViewerDialog(tiff_path, parent=self)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return
        except ModuleNotFoundError as exc:
            vtk_error = exc
            vtk_missing = True
        except ImportError as exc:
            vtk_error = exc
            vtk_missing = True
        except Exception as exc:
            # Any other VTK/runtime error; keep error for messaging
            vtk_error = exc

        # Decide on fallback only for raster volumes; point clouds require VTK
        try:
            suffix = Path(tiff_path).suffix.lower() if tiff_path else ''
            name_lower = Path(tiff_path).name.lower() if tiff_path else ''
        except Exception:
            suffix = ''
            name_lower = ''

        point_cloud_suffixes = {'.ply', '.csv', '.xyz'}
        mesh_suffixes = {'.stl', '.obj'}
        requires_vtk = suffix in point_cloud_suffixes or suffix in mesh_suffixes

        # Re-check VTK availability independently of the viewer import error
        def _vtk_available() -> tuple[bool, str | None]:
            try:
                try:
                    import vtkmodules  # noqa: F401
                except Exception:
                    import vtk  # noqa: F401
                # Also ensure Qt interactor exists
                from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: F401
                return True, None
            except Exception as exc:
                return False, str(exc)

        _ok, _probe = _vtk_available()
        vtk_missing = not _ok

        if requires_vtk:
            # No raster fallback; inform user about VTK requirement
            if vtk_missing:
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Mesh/Point Cloud Viewer Requires VTK"),
                    self.tr(
                        "PLY/CSV/XYZ point clouds and STL/OBJ meshes require VTK with Qt support.\n\n"
                        f"Details: {_probe or 'Unknown import error'}\n\n"
                        "Conda:  conda install -c conda-forge vtk\n"
                        "Pip:    pip install vtk"
                    ),
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Mesh/Point Cloud Viewer"),
                    self.tr("Failed to open the VTK mesh/point cloud viewer.\n%s") % (
                        str(vtk_error) if vtk_error else ""
                    ),
                )
            return

        # Fallback to slice/MIP raster viewer for TIFF and similar
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
                    "For interactive 3D volume rendering, install VTK with Qt support.\n\n"
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
        self._stop_csv_worker()
        quit_and_wait(self.frame_worker, "Thank you!")
        quit_and_wait(self.seg_train_thread, "See you next time!")
        quit_and_wait(self.seg_pred_thread, "Bye!")
        if hasattr(self, "yolo_training_manager") and self.yolo_training_manager:
            self.yolo_training_manager.cleanup()
        if hasattr(self, "dino_kpseg_training_manager") and self.dino_kpseg_training_manager:
            self.dino_kpseg_training_manager.cleanup()
        try:
            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is not None:
                dialog.close()
        except Exception:
            pass
        try:
            from annolid.gui.tensorboard import stop_tensorboard

            stop_tensorboard()
        except Exception:
            pass

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

        self._set_active_view("canvas")

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

        self._set_active_view("canvas")

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

    def _get_tracking_device(self):
        # More sophisticated logic could go here (e.g., user settings)
        if self.config.get('use_cpu_only', False) or torch is None:
            return "cpu" if torch is None else torch.device("cpu")
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
            process, url = ensure_tensorboard(
                log_dir=shared_runs_root(), preferred_port=6006, host="127.0.0.1")
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
            dialog = TrainingDashboardDialog(
                settings=self.settings, parent=None)
            dialog.dashboard.register_training_manager(
                self.yolo_training_manager)
            dialog.dashboard.register_training_manager(
                self.dino_kpseg_training_manager)
            dialog.finished.connect(
                lambda *_: setattr(self, "_training_dashboard_dialog", None)
            )
            dialog.destroyed.connect(
                lambda *_: setattr(self, "_training_dashboard_dialog", None)
            )
            self._training_dashboard_dialog = dialog

        try:
            dialog.show()
            dialog.setWindowState(dialog.windowState() &
                                  ~QtCore.Qt.WindowMinimized)
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
                    text
                )
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

    # ---------------------------------------------------------------
    # SAM 3D Objects (optional, post-processing)
    # ---------------------------------------------------------------
    def run_sam3d_reconstruction(self):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager.run_sam3d_reconstruction()

    def _handle_sam3d_finished(self, result, *, worker_thread: QtCore.QThread):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager._handle_sam3d_finished(
                result, worker_thread=worker_thread
            )

    def configure_sam3d_settings(self):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager.configure_sam3d_settings()

    def run_video_depth_anything(self):
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.run_video_depth_anything()

    def configure_video_depth_settings(self):
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.configure_video_depth_settings()

    def _handle_depth_preview(self, payload: object) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._handle_depth_preview(payload)

    # Optical-flow previews and overlays are handled by OpticalFlowTool.

    def _set_depth_preview_frame(self, frame_index: int) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._set_depth_preview_frame(frame_index)

    def _depth_ndjson_path(self) -> Optional[Path]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._depth_ndjson_path()
        return None

    def _load_depth_ndjson_records(self) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.load_depth_ndjson_records()

    def _build_depth_overlay(self, frame_rgb: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._build_depth_overlay(frame_rgb, depth_map)
        return depth_map

    def _restore_canvas_frame(self) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._restore_canvas_frame()

    def _load_depth_overlay_from_json(
        self, json_path: Path, frame_rgb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        return None

    def _load_depth_overlay_from_record(
        self, record: Dict[str, object], frame_rgb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._load_depth_overlay_from_record(record, frame_rgb)
        return None

    def _current_frame_rgb(self) -> Optional[np.ndarray]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._current_frame_rgb()
        return None

    def _update_depth_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.update_overlay_for_frame(
                frame_number, frame_rgb)

    def _handle_video_depth_finished(
        self,
        result,
        *,
        output_dir: str,
        worker_thread: QtCore.QThread,
    ) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._handle_video_depth_finished(
                result, output_dir=output_dir, worker_thread=worker_thread
            )


def main(argv=None, *, config=None):
    """
    Launch the Annolid GUI.

    When ``config`` is provided (e.g., by a lightweight launcher), CLI parsing
    is skipped to avoid duplicate work.
    """
    if config is None:
        config, _, version_requested = parse_cli(argv)
    else:
        # Assume caller already decided whether to show version/help
        version_requested = False

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
