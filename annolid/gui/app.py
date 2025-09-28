# Enable CPU fallback for unsupported MPS ops
import os  # noqa
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # noqa

import re
import csv
import os.path as osp
import time
import yaml
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
import torch
import codecs
import imgviz
import argparse
from pathlib import Path
import functools
import requests
import subprocess

from labelme.ai import MODELS
from qtpy import QtCore
from qtpy.QtCore import Qt, Slot, Signal
from qtpy import QtWidgets
from qtpy import QtGui
from labelme import PY2
from labelme import QT5
from qtpy.QtCore import QFileSystemWatcher

from annolid.gui.widgets.video_manager import VideoManagerWidget
from annolid.gui.workers import FlexibleWorker, LoadFrameThread
from annolid.gui.shape import Shape
from labelme.app import MainWindow
from labelme.utils import newAction
from labelme.widgets import BrightnessContrastDialog
from labelme.widgets import LabelListWidgetItem
from labelme import utils
from annolid.utils.logger import logger
from annolid.utils.files import count_json_files
from labelme.widgets import ToolBar
from annolid.gui.label_file import LabelFileError
from annolid.gui.label_file import LabelFile
from annolid.configs import get_config
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.text_prompt import AiRectangleWidget
from annolid.annotation import labelme2coco
from annolid.data import videos
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
from annolid.gui.widgets import RecordingWidget
from annolid.gui.widgets import CanvasScreenshotWidget
from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor
from annolid.annotation import labelme2csv
from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.gui.widgets.place_preference_dialog import TrackingAnalyzerDialog
from annolid.data.videos import get_video_files
from annolid.gui.widgets.caption import CaptionWidget
from annolid.gui.models_registry import MODEL_REGISTRY, PATCH_SIMILARITY_MODELS
from annolid.gui.widgets.shape_dialog import ShapePropagationDialog
from annolid.postprocessing.video_timestamp_annotator import process_directory
from annolid.gui.widgets.segment_editor import SegmentEditorDialog
from annolid.jobs.tracking_worker import TrackingWorker
from typing import Dict, List, Optional, Tuple
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


__appname__ = 'Annolid'
__version__ = "1.2.2"

LABEL_COLORMAP = imgviz.label_colormap(value=200)

PATCH_SIMILARITY_DEFAULT_MODEL = PATCH_SIMILARITY_MODELS[2].identifier


def start_tensorboard(log_dir=None,
                      tensorboard_url='http://localhost:6006'):

    process = None
    if log_dir is None:
        here = Path(__file__).parent
        log_dir = here.parent.resolve() / "runs" / "logs"
    try:
        r = requests.get(tensorboard_url)
    except requests.exceptions.ConnectionError:
        process = subprocess.Popen(
            ['tensorboard', f'--logdir={str(log_dir)}'])
        time.sleep(8)
    return process


class VisualizationWindow(QtWidgets.QDialog):

    def __init__(self):
        super(VisualizationWindow, self).__init__()
        from qtpy.QtWebEngineWidgets import QWebEngineView
        self.setWindowTitle("Visualization Tensorboard")
        self.process = start_tensorboard()
        self.browser = QWebEngineView()
        self.browser.setUrl(QtCore.QUrl(self.tensorboar_url))
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.browser)
        self.setLayout(vbox)
        self.show()

    def closeEvent(self, event):
        if self.process is not None:
            time.sleep(3)
            self.process.kill()
        event.accept()


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
            self._connect_track_all_signals)

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
        action = functools.partial(newAction, self)
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
        self.video_loader = None
        self.video_file = None
        self.isPlaying = False
        self.event_type = None
        self._time_stamp = ''
        self.behavior_controller = BehaviorController(self._get_rgb_by_label)
        self.annotation_dir = None
        self.step_size = 5
        self.stepSizeWidget = StepSizeWidget(5)
        self.prev_shapes = None
        self.pred_worker = None
        self.video_processor = None
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
        self.pinned_flags = {}
        # Create progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._current_video_defined_segments: List[TrackingSegment] = []
        self.active_tracking_worker: Optional[TrackingWorker] = None
        self._setup_custom_menu_actions()

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

        self.flag_widget = FlagTableWidget()  # Replace QListWidget with FlagTableWidget
        self.flag_dock.setWidget(self.flag_widget)  # Set the widget to dock
        self.flag_widget.startButtonClicked.connect(
            self.handle_flag_start_button)
        self.flag_widget.endButtonClicked.connect(
            self.handle_flag_end_button)
        self.flag_widget.flagToggled.connect(
            self.handle_flag_toggled)
        self.flag_widget.flagsSaved.connect(self.handle_flags_saved)
        self.flag_widget.rowSelected.connect(self.handle_row_selected)

        self.handle_flags_saved()

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

        self.setCentralWidget(scrollArea)

        self.createPolygonSAMMode = action(
            self.tr("AI Polygons"),
            self.segmentAnything,
            icon="objects",
            tip=self.tr("Start creating polygons with segment anything"),
        )

        createAiPolygonMode = action(
            self.tr("Create AI-Polygon"),
            lambda: self.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            self.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )

        createAiPolygonMode.changed.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText()
            )
            if self.canvas.createMode == "ai_polygon"
            else None
        )

        self.createGroundingSAMMode = action(
            self.tr("Create GroundingSAM"),
            lambda: self.toggleDrawMode(False, createMode="grounding_sam"),
            None,
            "objects",
            self.tr("Start using grounding_sam"),
            enabled=False,
        )

        open_video = action(
            self.tr("&Open Video"),
            self.openVideo,
            None,
            "Open Video",
            self.tr("Open video")
        )

        advance_params = action(
            self.tr("&Advanced Parameters"),
            self.set_advanced_params,
            None,
            "Advanced Parameters",
            self.tr("Advanced Parameters")
        )
        open_video.setIcon(QtGui.QIcon(
            str(
                self.here / "icons/open_video.png"
            )
        ))

        open_audio = action(
            self.tr("&Open Audio"),
            self.openAudio,
            None,
            "Open Audio",
            self.tr("Open Audio")
        )

        open_caption = action(
            self.tr("&Open Caption"),
            self.openCaption,
            None,
            "Open Caption",
            self.tr("Open Caption")
        )

        downsample_video = action(
            self.tr("&Downsample Videos"),
            self.downsample_videos,
            None,
            "Downsample Videos",
            self.tr("Downsample Videos")
        )

        tracking_reports = action(
            self.tr("&Tracking Reports"),
            self.trigger_gap_analysis,
            None,
            "Tracking Reports",
            self.tr("Generate tracking reports for the selected video")
        )

        convert_csv = action(
            self.tr("&Save CSV"),
            self.convert_labelme_json_to_csv,
            None,
            "Save CSV",
            self.tr("Save CSV")
        )

        extract_shape_keypoints = action(
            self.tr("&Extract Shape Keypoints"),
            self.extract_and_save_shape_keypoints,
            None,
            "Extract Shape Keypoints",
            self.tr("Extract Shape Keypoints")
        )

        convert_sleap = action(
            self.tr("&Convert SLEAP h5 to labelme"),
            self.convert_sleap_h5_to_labelme,
            None,
            "Convert SLEAP h5 to labelme",
            self.tr("Convert SLEAP h5 to labelme")
        )

        convert_deeplabcut = action(
            self.tr("&Convert DeepLabCut CSV to labelme"),
            self.convert_deeplabcut_csv_to_labelme,
            None,
            "Convert DeepLabCut CSV to labelme",
            self.tr("Convert DeepLabCut CSV to labelme")
        )

        convert_labelme2yolo_format = action(
            self.tr("&Convert Labelme to YOLO format"),
            self.convert_labelme2yolo_format,
            None,
            "Convert Labelme to YOLO format",
            self.tr("Convert Labelme to YOLO format")
        )

        place_perference = action(
            self.tr("&Place Preference"),
            self.place_preference_analyze,
            None,
            "Place Preference",
            self.tr("Place Preference")
        )

        about_annolid = action(
            self.tr("&About Annolid"),
            self.about_annolid_and_system_info,
            None,
            "About Annolid",
            self.tr("About Annolid")
        )

        step_size = QtWidgets.QWidgetAction(self)
        step_size.setIcon(QtGui.QIcon(
            str(
                self.here / "icons/fast_forward.png"
            )
        ))

        step_size.setDefaultWidget(self.stepSizeWidget)

        self.stepSizeWidget.setWhatsThis(
            self.tr(
                "Step for the next or prev image. e.g. 30"
            )
        )
        self.stepSizeWidget.setEnabled(False)

        coco = action(
            self.tr("&COCO format"),
            self.coco,
            'Ctrl+C+O',
            "coco",
            self.tr("Convert to COCO format"),
        )

        coco.setIcon(QtGui.QIcon(str(
            self.here / "icons/coco.png")))

        save_labeles = action(
            self.tr("&Save labels"),
            self.save_labels,
            'Ctrl+Shift+L',
            'Save Labels',
            self.tr("Save labels to txt file")
        )

        save_labeles.setIcon(QtGui.QIcon(
            str(self.here/"icons/label_list.png")
        ))

        frames = action(
            self.tr("&Extract frames"),
            self.frames,
            'Ctrl+Shift+E',
            "Extract frames",
            self.tr("Extract frames frome a video"),
        )

        models = action(
            self.tr("&Train models"),
            self.models,
            "Ctrl+Shift+T",
            "Train models",
            self.tr("Train neural networks")
        )
        models.setIcon(QtGui.QIcon(str(
            self.here / "icons/models.png")))

        frames.setIcon(QtGui.QIcon(str(
            self.here / "icons/extract_frames.png")))

        tracks = action(
            self.tr("&Track Animals"),
            self.tracks,
            "Ctrl+Shift+O",
            "Track Animals",
            self.tr("Track animals and Objects")
        )

        tracks.setIcon(QtGui.QIcon(str(
            self.here / 'icons/track.png'
        )))

        glitter2 = action(
            self.tr("&Glitter2"),
            self.glitter2,
            "Ctrl+Shift+G",
            self.tr("Convert to Glitter2 nix format")
        )

        glitter2.setIcon(QtGui.QIcon(str(
            self.here / 'icons/glitter2_logo.png'
        )))

        quality_control = action(
            self.tr("&Quality Control"),
            self.quality_control,
            "Ctrl+Shift+G",
            self.tr("Convert to tracking results to labelme format")
        )

        quality_control.setIcon(QtGui.QIcon(str(
            self.here / 'icons/quality_control.png'
        )))

        visualization = action(
            self.tr("&Visualization"),
            self.visualization,
            'Ctrl+Shift+V',
            "Visualization",
            self.tr("Visualization results"),
        )

        colab = action(
            self.tr("&Open in Colab"),
            self.train_on_colab,
            icon="Colab",
            tip=self.tr("Open in Colab"),
        )

        colab.setIcon(QtGui.QIcon(str(
            self.here / "icons/colab.png")))
        shortcuts = self._config["shortcuts"]
        self.shortcuts = shortcuts

        visualization.setIcon(QtGui.QIcon(str(
            self.here / "icons/visualization.png")))

        self.aiRectangle = AiRectangleWidget()
        self.aiRectangle._aiRectanglePrompt.returnPressed.connect(
            self._grounding_sam
        )

        self.recording_widget = RecordingWidget(self.canvas)

        self.patch_similarity_action = newAction(
            self,
            self.tr("Patch Similarity"),
            self._toggle_patch_similarity_tool,
            icon="visualization",
            tip=self.tr(
                "Click on the frame to generate a DINO patch similarity heatmap"),
        )
        self.patch_similarity_action.setCheckable(True)
        self.patch_similarity_action.setIcon(
            QtGui.QIcon(str(self.here / "icons/visualization.png")))

        self.patch_similarity_settings_action = newAction(
            self,
            self.tr("Patch Similarity Settings…"),
            self._open_patch_similarity_settings,
            tip=self.tr(
                "Choose model and overlay opacity for patch similarity"),
        )

        self.pca_map_action = newAction(
            self,
            self.tr("PCA Feature Map"),
            self._toggle_pca_map_tool,
            icon="visualization",
            tip=self.tr(
                "Toggle a PCA-colored DINO feature map overlay for the current frame",
            ),
        )
        self.pca_map_action.setCheckable(True)
        self.pca_map_action.setIcon(
            QtGui.QIcon(str(self.here / "icons/visualization.png")))

        self.pca_map_settings_action = newAction(
            self,
            self.tr("PCA Feature Map Settings…"),
            self._open_pca_map_settings,
            tip=self.tr("Choose model and overlay opacity for the PCA map"),
        )

        # Create the QAction with the new label
        add_stamps_action = newAction(
            self,
            self.tr("Add Real-Time Stamps…"),       # menu text
            self._add_real_time_stamps,             # our slot handler
            icon="timestamp",                       # ensure icons/timestamp.png exists
            tip=self.tr("Populate CSVs with true frame timestamps")
        )

        _action_tools = list(self.actions.tool)
        _action_tools.insert(0, frames)
        _action_tools.insert(1, open_video)
        _action_tools.insert(2, step_size)
        _action_tools.append(self.aiRectangle.aiRectangleAction)
        _action_tools.append(tracks)
        _action_tools.append(glitter2)
        _action_tools.append(coco)
        _action_tools.append(models)
        _action_tools.append(self.createPolygonSAMMode)
        _action_tools.append(save_labeles)
        _action_tools.append(quality_control)
        _action_tools.append(colab)
        _action_tools.append(visualization)
        _action_tools.append(self.patch_similarity_action)
        _action_tools.append(self.pca_map_action)
        _action_tools.append(self.recording_widget.record_action)

        self.actions.tool = tuple(_action_tools)
        self.tools.clear()

        utils.addActions(self.tools, self.actions.tool)
        utils.addActions(self.menus.file, (open_video,))
        utils.addActions(self.menus.file, (open_audio,))
        utils.addActions(self.menus.file, (open_caption,))
        utils.addActions(self.menus.file, (colab,))
        utils.addActions(self.menus.file, (save_labeles,))
        utils.addActions(self.menus.file, (coco,))
        utils.addActions(self.menus.file, (frames,))
        utils.addActions(self.menus.file, (models,))
        utils.addActions(self.menus.file, (tracks,))
        utils.addActions(self.menus.file, (quality_control,))
        utils.addActions(self.menus.file, (downsample_video,))
        utils.addActions(self.menus.file, (tracking_reports,))

        # Insert it under File
        self.menus.file.addSeparator()
        utils.addActions(self.menus.file, (convert_csv,))
        utils.addActions(self.menus.file, (extract_shape_keypoints,))
        utils.addActions(self.menus.file, (convert_deeplabcut,))
        utils.addActions(self.menus.file, (convert_sleap,))
        utils.addActions(self.menus.file, (convert_labelme2yolo_format,))
        utils.addActions(self.menus.file, (place_perference,))
        utils.addActions(self.menus.file, (add_stamps_action,))

        utils.addActions(self.menus.file, (advance_params,))

        utils.addActions(self.menus.view, (glitter2,))
        utils.addActions(self.menus.view, (visualization,))
        utils.addActions(self.menus.view, (self.patch_similarity_action,))
        utils.addActions(self.menus.view, (self.pca_map_action,))
        utils.addActions(
            self.menus.view,
            (self.patch_similarity_settings_action, self.pca_map_settings_action),
        )

        utils.addActions(self.menus.help, (about_annolid,))

        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        self.settings = QtCore.QSettings("Annolid", 'Annolid')
        # Restore application settings.
        self.recentFiles = self.settings.value("recentFiles", []) or []
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        state = self.settings.value("window/state", QtCore.QByteArray())
        self.move(position)

        # Patch similarity preferences
        self.patch_similarity_model = str(
            self.settings.value(
                "patch_similarity/model",
                PATCH_SIMILARITY_DEFAULT_MODEL,
            )
        )
        self.patch_similarity_alpha = float(
            self.settings.value("patch_similarity/alpha", 0.55)
        )
        self.patch_similarity_alpha = min(
            max(self.patch_similarity_alpha, 0.05), 1.0)
        self.patch_similarity_service = DinoPatchSimilarityService(self)
        self.patch_similarity_service.started.connect(
            self._on_patch_similarity_started)
        self.patch_similarity_service.finished.connect(
            self._on_patch_similarity_finished)
        self.patch_similarity_service.error.connect(
            self._on_patch_similarity_error)

        self.pca_map_model = str(
            self.settings.value(
                "pca_map/model",
                self.patch_similarity_model or PATCH_SIMILARITY_DEFAULT_MODEL,
            )
        )
        self.pca_map_alpha = float(
            self.settings.value("pca_map/alpha", 0.65)
        )
        self.pca_map_alpha = min(max(self.pca_map_alpha, 0.05), 1.0)
        self.pca_map_clusters = int(
            self.settings.value("pca_map/clusters", 0)
        )
        if self.pca_map_clusters < 0:
            self.pca_map_clusters = 0
        self.pca_map_service = DinoPCAMapService(self)
        self.pca_map_service.started.connect(self._on_pca_map_started)
        self.pca_map_service.finished.connect(self._on_pca_map_finished)
        self.pca_map_service.error.connect(self._on_pca_map_error)

        self.video_results_folder = None
        self.seekbar = None
        self.audio_widget = None
        self.audio_dock = None
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

        self._selectAiModelComboBox.clear()
        self.custom_ai_model_names = [
            model.display_name for model in MODEL_REGISTRY]
        model_names = [model.name for model in MODELS] + \
            self.custom_ai_model_names
        self._selectAiModelComboBox.addItems(model_names)
        # Set EdgeSAM as default
        if self._config["ai"]["default"] in model_names:
            model_index = model_names.index(self._config["ai"]["default"])
        else:
            logger.warning(
                "Default AI model is not found: %r",
                self._config["ai"]["default"],
            )
            model_index = 0
        self._selectAiModelComboBox.setCurrentIndex(model_index)
        self._selectAiModelComboBox.currentIndexChanged.connect(
            lambda: self.canvas.initializeAiModel(
                name=self._selectAiModelComboBox.currentText(),
                _custom_ai_models=self.custom_ai_model_names,
            )
            if self.canvas.createMode in ["ai_polygon", "ai_mask"]
            else None
        )

        self.canvas_screenshot_widget = CanvasScreenshotWidget(
            canvas=self.canvas, here=Path(__file__).resolve().parent)
        self._setup_canvas_screenshot_action()

        self.populateModeActions()

    def _setup_custom_menu_actions(self):
        self.open_segment_editor_action = newAction(
            self, self.tr("Define Video Segments..."),
            self._open_segment_editor_dialog, shortcut="Ctrl+Alt+S",
            tip=self.tr("Define tracking segments for the current video")
        )
        self.open_segment_editor_action.setEnabled(False)
        if not hasattr(self.menus, 'video_tools'):
            self.menus.video_tools = self.menuBar().addMenu(self.tr("&Video Tools"))
        utils.addActions(self.menus.video_tools,
                         (self.open_segment_editor_action,))

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
            self._handle_tracking_initiated_by_dialog)

        # Optional: For modeless live updates (if SegmentEditorDialog becomes modeless)
        # self.live_annolid_frame_updated.connect(dialog.update_live_annolid_frame_info)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:  # User clicked "OK"
            self._current_video_defined_segments = dialog.get_defined_segments()
            logger.info(
                f"Segment Editor OK. {len(self._current_video_defined_segments)} segments stored.")
            self._save_segments_for_active_video()  # Persist
        else:  # User clicked "Cancel" or closed dialog
            logger.info("Segment Editor Cancelled/Closed.")

        # try: self.live_annolid_frame_updated.disconnect(dialog.update_live_annolid_frame_info)
        # except TypeError: pass
        dialog.deleteLater()

    # --- Modified/New Slot to handle tracking started by the dialog ---

    @Slot(TrackingWorker, Path)  # worker_instance, video_path_processed
    def _handle_tracking_initiated_by_dialog(self, worker_instance: TrackingWorker, video_path: Path):
        if self.active_tracking_worker and self.active_tracking_worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "Tracking Busy",
                                          "Dialog initiated tracking, but Annolid already has an active worker. This shouldn't happen if dialog checks.")
            # Potentially stop the new worker if necessary, or log error
            worker_instance.stop()  # Stop the worker the dialog created if we can't handle it
            worker_instance.wait(1000)  # Wait a bit for it to stop
            return

        logger.info(
            f"AnnolidWindow: Tracking initiated by SegmentEditorDialog for {video_path.name}")
        self.active_tracking_worker = worker_instance
        # The worker is already started by the dialog. AnnolidWindow just needs to connect UI signals.
        # Connect progress, finished, error, UI updates
        self._connect_signals_to_active_worker(self.active_tracking_worker)

    # Helper method (public for dialog to check)

    def is_tracking_busy(self) -> bool:
        return bool(self.active_tracking_worker and self.active_tracking_worker.isRunning())

    @Slot(str)
    def _on_tracking_job_finished(self, completion_message: str):
        QtWidgets.QMessageBox.information(
            self, "Tracking Job Complete", completion_message)
        self.statusBar().showMessage(completion_message, 5000)
        self._set_tracking_ui_state(is_tracking=False)

        # Disconnect signals from the finished worker
        if self.active_tracking_worker:
            try:
                # Attempt to disconnect all relevant signals
                self.active_tracking_worker.progress.disconnect(
                    self._update_main_status_progress)
                self.active_tracking_worker.finished.disconnect(
                    self._on_tracking_job_finished)
                self.active_tracking_worker.error.disconnect(
                    self._on_tracking_job_error)
                if hasattr(self.active_tracking_worker, 'video_job_started'):
                    self.active_tracking_worker.video_job_started.disconnect(
                        self._handle_tracking_video_started_ui_update)
                if hasattr(self.active_tracking_worker, 'video_job_finished'):
                    self.active_tracking_worker.video_job_finished.disconnect(
                        self._handle_tracking_video_finished_ui_update)
            except (TypeError, RuntimeError) as e:
                logger.debug(
                    f"Error disconnecting signals from finished worker (may have already been disconnected or was never connected): {e}")

            # If the worker's parent was None (as set in dialog), and AnnolidWindow is meant to manage its lifetime
            # after receiving it, you might consider worker.deleteLater() here,
            # but only if AnnolidWindow is truly taking ownership beyond just signal connection.
            # For now, let's assume the worker cleans itself up or its parent (if set in dialog) handles it.
            self.active_tracking_worker = None  # CRITICAL: Clear the reference

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

    def _save_canvas_screenshot(self):
        """ Calls CanvasScreenshotWidget and passes in the current filename"""
        self.canvas_screenshot_widget.save_canvas_screenshot(
            filename=self.filename)

    def handle_flags_saved(self, flags={}):
        default_config = self.here.parent.resolve() / 'configs' / 'behaviors.yaml'

        # Load the existing configuration from the YAML file
        try:
            with open(default_config, 'r') as file:
                config_data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            config_data = {}

        # Append the pinned flags to the existing configuration
        if 'pinned_flags' not in config_data:
            config_data['pinned_flags'] = []
        config_data['pinned_flags'].extend(flags)  # Append the new flags
        pinned_flags = {
            pinned_flag: False for pinned_flag in config_data['pinned_flags']}

        # Save the updated configuration back to the YAML file
        with open(default_config, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False)

        # Update the pinned_flags attribute and load the flags
        self.pinned_flags = pinned_flags
        self.loadFlags(pinned_flags)

    def handle_row_selected(self, flag_name: str):
        self.event_type = flag_name

    def _connect_track_all_signals(self, worker_instance):
        """Connects to signals from a newly created TrackAllWorker instance."""
        if worker_instance:
            worker_instance.video_processing_started.connect(
                self._handle_track_all_video_started)
            worker_instance.video_processing_finished.connect(
                self._handle_track_all_video_finished)

    @Slot(str, str)
    def _handle_track_all_video_started(self, video_path, output_folder_path):
        logger.info(f"TrackAll: Starting processing for {video_path}")
        self.closeFile()  # Close any currently open video
        # 1. Load the video into the canvas
        self.openVideo(from_video_list=True,  # Ensure this is set to True
                       video_path=video_path,
                       programmatic_call=True)
        # 2. Setup the prediction folder watcher for this video's output_folder
        #    Ensure self.video_results_folder is correctly set by openVideo to output_folder_path
        #    or pass output_folder_path directly.
        if self.video_file == video_path and self.video_results_folder == Path(output_folder_path):
            logger.info(
                f"TrackAll: Setting up watcher for {output_folder_path}")
            self._setup_prediction_folder_watcher(output_folder_path)
            # Initialize progress bar for predictions (if you have one for single video)
            if hasattr(self, 'progress_bar'):
                self._initialize_progress_bar()  # Make sure this is suitable
        else:
            logger.warning(
                f"TrackAll: Video {video_path} not properly loaded or output folder mismatch. Cannot start watcher.")
            logger.warning(
                f"Current video_file: {self.video_file}, expected: {video_path}")
            logger.warning(
                f"Current video_results_folder: {self.video_results_folder}, expected: {output_folder_path}")

    @Slot(str)
    def _handle_track_all_video_finished(self, video_path):
        logger.info(f"TrackAll: Finished processing for {video_path}")
        # Clean up the watcher and progress for this specific video
        # Check if this is the video we were actually watching
        current_video_name_in_watcher = ""
        if self.prediction_progress_watcher and self.prediction_progress_watcher.directories():
            current_video_name_in_watcher = Path(
                self.prediction_progress_watcher.directories()[0]).name

        if Path(video_path).stem == current_video_name_in_watcher:
            self._finalize_prediction_progress(
                f"Automated tracking for {Path(video_path).name} complete.")
        else:
            logger.info(
                f"TrackAll: Video {video_path} finished, but watcher was on {current_video_name_in_watcher} or not active.")

    def get_active_flags(self, flags):
        active_flags = []
        for _flag in flags:
            if flags[_flag]:
                active_flags.append(_flag)
        return active_flags

    def get_current_behavior_text(self, flags):
        active_flags = self.get_active_flags(flags)
        return ','.join(active_flags)

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
                self.pinned_flags = flags
                self.loadFlags(flags)
            else:
                # # If there is no string after 'flags'
                # in the text prompt, clear the flag widget
                self.flag_widget.clear()
        else:
            self.canvas.predictAiRectangle(prompt_text)

    def update_step_size(self, value):
        self.step_size = value
        self.stepSizeWidget.set_value(self.step_size)

    def handle_flag_start_button(self, flag_name):

        if self.seekbar:
            self.record_behavior_event(
                flag_name, "start", frame_number=self.frame_number)
        self.canvas.setBehaviorText(flag_name)
        self.event_type = flag_name
        self.pinned_flags[flag_name] = True

    def handle_flag_toggled(self, flag_name, state):
        if state:
            self.handle_flag_start_button(flag_name)
        else:
            self.handle_flag_end_button(flag_name)

    def handle_flag_end_button(self, flag_name, record_event: bool = True):
        if self.seekbar and record_event:
            self.record_behavior_event(
                flag_name, "end", frame_number=self.frame_number)
        self.event_type = flag_name
        self.canvas.setBehaviorText("")
        self.pinned_flags[flag_name] = False

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
        if self.video_file:
            self.audio_widget = AudioWidget(self.video_file)
            self.audio_dock = QtWidgets.QDockWidget(self.tr("Audio"), self)
            self.audio_dock.setObjectName("Audio")
            self.audio_dock.setWidget(self.audio_widget)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.audio_dock)

    def openCaption(self):
        # Caption dock (created but initially hidden)
        self.caption_dock = QtWidgets.QDockWidget(self.tr("Caption"), self)
        self.caption_dock.setObjectName("Caption")
        self.caption_widget = CaptionWidget()
        self.caption_dock.setWidget(self.caption_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.caption_dock)

        self.caption_widget.charInserted.connect(
            self.setDirty)      # Mark as dirty
        self.caption_widget.charDeleted.connect(
            self.setDirty)      # Mark as dirty
        self.caption_widget.captionChanged.connect(
            self.canvas.setCaption)  # Update canvas

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
            if self.audio_widget is not None and self.audio_widget.audio_loader is not None:
                self.audio_widget.audio_loader.play()
            if self.fps is not None and self.fps > 0:
                self.timer.start(int(1000/self.fps))
            else:
                # 10 to 50 milliseconds are normal real time
                # playback
                self.timer.start(20)
        else:
            self.timer.stop()
            # Stop audio playback when video playback stops
            if self.audio_widget is not None and self.audio_widget.audio_loader is not None:
                self.audio_widget.audio_loader.stop()

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

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self._deactivate_patch_similarity()
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
        if self.audio_widget:
            self.audio_widget.audio_loader = None
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
        self.frame_loader = LoadFrameThread()
        if self.video_processor is not None:
            self.video_processor.cutie_processor = None
        self.video_processor = None
        self.fps = None
        self.only_json_files = False
        self._stop_prediction_folder_watcher()
        # Clear "predicted" marks from the slider when file is closed
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")

        if self.active_tracking_worker and self.active_tracking_worker.isRunning():
            reply = QtWidgets.QMessageBox.question(self, "Tracking in Progress",
                                                   "Stop tracking and close video?",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.active_tracking_worker.stop()
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
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)
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
        current_text = self._selectAiModelComboBox.currentText()
        return next(
            (m for m in MODEL_REGISTRY if m.display_name == current_text), None)

    def get_current_model_weight_file(self) -> str:
        """
        Returns the weight file associated with the currently selected model.
        If no matching model is found, returns a default fallback weight file.
        """
        model = self._get_current_model_config()
        return model.weight_file if model is not None else "Segment-Anything (Edge)"

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
        return "yolo" in identifier or "yolo" in weight

    @staticmethod
    def _is_sam2_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return "sam2_hiera" in identifier or "sam2_hiera" in weight

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
                self.video_processor = process_video
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
                    num_json_files = count_json_files(
                        self.video_results_folder)
                    logger.info(
                        f"Number of predicted frames: {num_json_files} in total {self.num_frames}")
                    if num_json_files >= self.num_frames:
                        # convert json labels to csv file
                        self.convert_json_to_tracked_csv()
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
        self.reset_predict_button()

    def reset_predict_button(self):
        """Reset the predict button text and style"""
        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;")

    def loadFlags(self, flags):
        """ Loads flags using FlagTableWidget's loadFlags method """
        behave_text = self.get_current_behavior_text(flags)
        self.canvas.setBehaviorText(behave_text)
        self.flag_widget.loadFlags(flags)

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
        elif isinstance(self.imageData, QtGui.QImage):
            pil_image = ImageQt.fromqimage(self.imageData)
        # 3. Handle the special ImageQt.Image wrapper type
        elif isinstance(self.imageData, ImageQt.ImageQt):
            pil_image = self.imageData
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

        prediction_folder = self.video_results_folder
        deleted_files = 0

        logger.info(f"Scanning for future predictions in: {prediction_folder}")

        for filename_str in os.listdir(prediction_folder):
            prediction_file_path = os.path.join(
                prediction_folder, filename_str)

            # Skip directories and non-JSON files
            if not os.path.isfile(prediction_file_path) or not filename_str.endswith('.json'):
                continue

            # Instead of fragile splitting, use a regular expression to find the number.
            # This regex looks for a sequence of digits at the end of the filename, right before ".json".
            match = re.search(r'(\d+)(?=\.json$)', filename_str)

            # If the filename doesn't match our expected pattern, skip it safely.
            if not match:
                logger.debug(
                    f"Skipping file with unexpected name format: {filename_str}")
                continue

            try:
                # The part of the string matched by the regex
                frame_number_str = match.group(1)
                # Convert to float first to handle potential decimals, then to int.
                frame_number = int(float(frame_number_str))
            except (ValueError, IndexError):
                # This handles cases where the regex matches but the string is still invalid (rare).
                logger.warning(
                    f"Could not parse frame number from file: {filename_str}")
                continue

            is_future_frame = frame_number > self.frame_number

            # The logic to check for a manually saved file seems to be based on an accompanying .png.
            # Let's make that check more robust as well.
            image_file_png = prediction_file_path.replace('.json', '.png')
            image_file_jpg = prediction_file_path.replace('.json', '.jpg')
            is_manually_saved = os.path.exists(
                image_file_png) or os.path.exists(image_file_jpg)

            if is_future_frame and not is_manually_saved:
                try:
                    os.remove(prediction_file_path)
                    deleted_files += 1
                except OSError as e:
                    logger.error(
                        f"Failed to delete file {prediction_file_path}: {e}")

        logger.info(
            f"{deleted_files} future prediction(s) were removed, excluding manually labeled files."
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

        self._initialize_progress_bar()

        try:
            self.worker = FlexibleWorker(
                task_function=labelme2csv.convert_json_to_csv,
                json_folder=str(out_folder),
                progress_callback=self._update_progress_bar
            )
            self.thread = QtCore.QThread()

            # Move the worker to the thread and connect signals
            self.worker.moveToThread(self.thread)
            self._connect_worker_signals()

            # Safely start the thread and worker signal
            self.thread.start()
            # Emit in a thread-safe way
            QtCore.QTimer.singleShot(
                0, lambda: self.worker.start_signal.emit())

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}")
        finally:
            self.statusBar().removeWidget(self.progress_bar)

    def _initialize_progress_bar(self):
        """Initialize the progress bar and add it to the status bar."""
        self.progress_bar.setValue(0)
        self.statusBar().addWidget(self.progress_bar)

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
        self.worker.start_signal.connect(self.worker.run)
        self.worker.finished_signal.connect(self.place_preference_analyze_auto)

        # Ensure cleanup happens in the right thread
        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.finished_signal.connect(lambda: self.worker.deleteLater())
        self.thread.finished.connect(lambda: self.thread.deleteLater())

        self.worker.finished_signal.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "Tracking Complete",
                f"Review the file at: {Path(self.video_file).with_suffix('')}_tracked.csv"
            )
        )
        self.worker.progress_signal.connect(self._update_progress_bar)
        self.seekbar.removeMarksByType("predicted")  # Clear previous marks

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
            from ultralytics import YOLO
            try:
                model = YOLO(yolo_model_file)  # Load the model from YAML

                if model_path:
                    try:
                        model.load(model_path)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self, "Error", f"Failed to load trained model: {e}")
                        return

                results = model.train(
                    data=config_file,
                    epochs=epochs,
                    imgsz=image_size,
                    project=out_dir if out_dir else None
                )
                self.statusBar().showMessage(self.tr(f"Training..."))

                QtWidgets.QMessageBox.information(
                    self, "Training Completed", "YOLO model training completed successfully!")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Training Error", f"An error occurred during training: {e}")

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

    def record_behavior_event(self,
                              behavior: str,
                              event_label: str,
                              frame_number: Optional[int] = None,
                              timestamp: Optional[float] = None,
                              trial_time: Optional[float] = None,
                              subject: Optional[str] = None,
                              highlight: bool = True) -> Optional[BehaviorEvent]:
        if frame_number is None:
            frame_number = self.frame_number
        if timestamp is None:
            timestamp = self._estimate_recording_time(frame_number)
        if trial_time is None:
            trial_time = timestamp
        if subject is None:
            subject = "Subject 1"

        event = self.behavior_controller.record_event(
            behavior,
            event_label,
            frame_number,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            highlight=highlight,
        )
        if event is None:
            logger.warning(
                "Unrecognized behavior event label '%s' for '%s'.",
                event_label,
                behavior,
            )
            return None
        self.pinned_flags.setdefault(behavior, False)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.append_event(event, fps=fps_for_log)
        return event

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
                    self.handle_flag_end_button(
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
        self.filename = self.video_results_folder / \
            f"{str(self.video_results_folder.name)}_{self.frame_number:09}.png"
        self.current_frame_time_stamp = self.video_loader.get_time_stamp()
        self.frame_loader.request(frame_number)
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)

    def load_tracking_results(self, cur_video_folder, video_filename):
        """
        Loads various tracking and behavior data from standardized CSV files
        in the video's directory.
        """
        self.behavior_controller.clear()
        self.behavior_log_widget.clear()
        self.pinned_flags = {}
        self._df = None  # Reset dataframe

        video_name = Path(video_filename).stem

        # --- Define Standardized Filenames ---
        # This makes the logic explicit and robust.
        main_tracking_file = cur_video_folder / f"{video_name}_tracking.csv"
        timestamps_file = cur_video_folder / f"{video_name}_timestamps.csv"
        labels_file_path = cur_video_folder / f"{video_name}_labels.csv"

        # --- Load the Main Tracking Results File ---
        # We look for one specific file. No more ambiguity with 'tracking' in the name.
        if main_tracking_file.is_file():
            try:
                logger.info(
                    f"Loading main tracking data from: {main_tracking_file}")
                df = pd.read_csv(main_tracking_file)
                # Ensure the 'frame_number' column exists, which is critical.
                if 'frame_number' not in df.columns and 'Unnamed: 0' in df.columns:
                    df.rename(
                        columns={'Unnamed: 0': 'frame_number'}, inplace=True)

                if 'frame_number' in df.columns:
                    self._df = df
                else:
                    logger.warning(
                        f"'{main_tracking_file}' is missing the required 'frame_number' column.")

            except Exception as e:
                logger.error(
                    f"Error loading main tracking file {main_tracking_file}: {e}")

        # --- Load Behavior/Timestamp Data ---
        # Check for the standardized timestamp/behavior file.
        if timestamps_file.is_file():
            logger.info(f"Loading behavior data from: {timestamps_file}")
            self._load_behavior(timestamps_file)

        # --- Load Other Data Types (like labels) ---
        if labels_file_path.is_file():
            logger.info(f"Loading labels data from: {labels_file_path}")
            self._load_labels(labels_file_path)

    def is_behavior_active(self, frame_number, behavior):
        """Checks if a behavior is active at a given frame."""
        return self.behavior_controller.is_behavior_active(frame_number, behavior)

    def _load_deeplabcut_results(self, frame_number: int, is_multi_animal: bool = False):
        """
        Load DeepLabCut tracking results for a given frame and convert them into shape objects.

        This method extracts x, y coordinates for each body part and, if applicable, for each animal.
        It then creates shape objects for visualization.

        Args:
            frame_number (int): The index of the frame to extract tracking data from.
            is_multi_animal (bool, optional): Whether the dataset contains multiple animals.
                Defaults to False.

        Notes:
            - Assumes self._df_deeplabcut is a multi-index Pandas DataFrame.
            - Multi-animal mode expects an 'animal' level in the column index.
            - Logs warnings for missing data instead of failing.

        Raises:
            KeyError: If expected columns are missing.
            Exception: For unexpected errors during shape extraction.
        """
        if self._df_deeplabcut is None:
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
            row = self._df_deeplabcut.loc[frame_number]

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

    def _load_behavior(self, behavior_csv_file: str) -> None:
        """Load behavior events from CSV and populate the slider timeline.

        Args:
            behavior_csv_file (str): Path to the CSV file containing behavior data.
        """
        # Load the CSV file into a DataFrame
        df_behaviors = pd.read_csv(behavior_csv_file)

        rows: List[Tuple[float, float, str, str, str]] = []

        for _, row in df_behaviors.iterrows():
            try:
                raw_timestamp = row["Recording time"]
                event_label = str(row["Event"])
                behavior = str(row["Behavior"])
                raw_subject = row.get("Subject")
                raw_trial_time = row.get("Trial time")
            except KeyError:
                del df_behaviors
                self._df_deeplabcut = pd.read_csv(
                    behavior_csv_file, header=[0, 1, 2], index_col=0)
                return

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

    def _load_labels(self, labels_csv_file):
        """Load labels from the given CSV file."""
        self._df = pd.read_csv(labels_csv_file)
        self._df.rename(columns={'Unnamed: 0': 'frame_number'}, inplace=True)

    def _load_video(self, video_path):
        """Open a video for annotation frame by frame."""
        if not video_path:
            return
        self.openVideo(from_video_list=True, video_path=video_path)

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
                self.video_loader = videos.CV2Video(video_filename)
            except Exception:
                QtWidgets.QMessageBox.about(self,
                                            "Not a valid video file",
                                            f"Please check and open a correct video file.")
                self.video_file = None
                self.video_loader = None
                return
            self.fps = self.video_loader.get_fps()
            self.num_frames = self.video_loader.total_frames()
            self.behavior_log_widget.set_fps(self.fps)
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
            # Add the play button to the status bar
            self.statusBar().addWidget(self.playButton)
            self.statusBar().addWidget(self.seekbar, stretch=1)
            self.statusBar().addWidget(self.saveButton)

            # load the first frame
            self.set_frame_number(self.frame_number)

            self.actions.openNextImg.setEnabled(True)

            self.actions.openPrevImg.setEnabled(True)

            self.frame_loader.video_loader = self.video_loader

            self.frame_loader.moveToThread(self.frame_worker)

            self.frame_worker.start(priority=QtCore.QThread.IdlePriority)

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
                if not programmatic_call:
                    self._emit_live_frame_update()
                logger.info(
                    f"Video '{self.filename}' loaded. Segment definition enabled.")
            else:
                self.open_segment_editor_action.setEnabled(False)
                self._current_video_defined_segments = []

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
        # imageData = ImageQt.fromqimage(qimage)
        # Save imageData as PIL Image to speed up frame loading by 10x
        self.imageData = qimage
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(qimage))
        flags: Dict[str, bool] = {}
        active_behaviors = self.behavior_controller.active_behaviors(
            self.frame_number)
        if active_behaviors:
            self.canvas.setBehaviorText(",".join(sorted(active_behaviors)))
        else:
            self.canvas.setBehaviorText(None)

        current_text = self.canvas.current_behavior_text or ""
        current_text_set = {item.strip()
                            for item in current_text.split(",") if item.strip()}
        for behavior in sorted(self.behavior_controller.behavior_names):
            flags[behavior] = behavior in current_text_set

        self.loadFlags(flags)
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
        return True

    # ------------------------------------------------------------------
    # Patch similarity (DINO) integration
    # ------------------------------------------------------------------
    def _toggle_patch_similarity_tool(self, checked=False):
        state = bool(checked) if isinstance(
            checked, bool) else self.patch_similarity_action.isChecked()
        if not state:
            self._deactivate_patch_similarity()
            return

        if self.canvas.pixmap is None or self.canvas.pixmap.isNull():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Patch Similarity"),
                self.tr(
                    "Load an image or video frame before starting patch similarity."),
            )
            self.patch_similarity_action.setChecked(False)
            return

        if not self.patch_similarity_model:
            self._open_patch_similarity_settings()
            if not self.patch_similarity_model:
                self.patch_similarity_action.setChecked(False)
                return

        self._deactivate_pca_map()
        self.canvas.enablePatchSimilarityMode(self._request_patch_similarity)
        self.statusBar().showMessage(
            self.tr("Patch similarity active – click on the frame to query patches."),
            5000,
        )

    def _deactivate_patch_similarity(self):
        if hasattr(self, "patch_similarity_action"):
            self.patch_similarity_action.setChecked(False)
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.disablePatchSimilarityMode()
            self.canvas.setPatchSimilarityOverlay(None)

    def _grab_current_frame_image(self):
        if self.canvas.pixmap is None or self.canvas.pixmap.isNull():
            return None
        qimage = self.canvas.pixmap.toImage().convertToFormat(
            QtGui.QImage.Format_RGBA8888)
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        array = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (qimage.height(), qimage.width(), 4))
        return Image.fromarray(array, mode="RGBA").convert("RGB")

    def _request_patch_similarity(self, x: int, y: int) -> None:
        if self.patch_similarity_service.is_busy():
            self.statusBar().showMessage(
                self.tr("Patch similarity is already running…"), 2000)
            return

        pil_image = self._grab_current_frame_image()
        if pil_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Patch Similarity"),
                self.tr("Failed to access the current frame."),
            )
            self._deactivate_patch_similarity()
            return

        self.canvas.setPatchSimilarityOverlay(None)
        request = DinoPatchRequest(
            image=pil_image,
            click_xy=(int(x), int(y)),
            model_name=self.patch_similarity_model,
            short_side=768,
            device=None,
            alpha=float(self.patch_similarity_alpha),
        )
        if not self.patch_similarity_service.request(request):
            self.statusBar().showMessage(
                self.tr("Patch similarity is already running…"), 2000)

    def _on_patch_similarity_started(self):
        self.statusBar().showMessage(
            self.tr("Computing patch similarity…"))

    def _on_patch_similarity_finished(self, payload: dict) -> None:
        overlay = payload.get("overlay_rgba")
        self.canvas.setPatchSimilarityOverlay(overlay)
        self.statusBar().showMessage(
            self.tr("Patch similarity ready."),
            4000,
        )

    def _on_patch_similarity_error(self, message: str) -> None:
        self.canvas.setPatchSimilarityOverlay(None)
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("Patch Similarity"),
            message,
        )
        self._deactivate_patch_similarity()

    def _open_patch_similarity_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Patch Similarity Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(self.patch_similarity_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(self.patch_similarity_alpha)

        layout.addRow(self.tr("Model"), model_combo)
        layout.addRow(self.tr("Overlay opacity"), alpha_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.patch_similarity_model = model_combo.currentData()
            self.patch_similarity_alpha = alpha_spin.value()
            self.settings.setValue(
                "patch_similarity/model", self.patch_similarity_model)
            self.settings.setValue(
                "patch_similarity/alpha", self.patch_similarity_alpha)
            self.statusBar().showMessage(
                self.tr("Patch similarity model updated."),
                3000,
            )

    # ------------------------------------------------------------------
    # PCA feature map (DINO) integration
    # ------------------------------------------------------------------
    def _toggle_pca_map_tool(self, checked=False):
        state = bool(checked) if isinstance(
            checked, bool) else self.pca_map_action.isChecked()
        if not state:
            self._deactivate_pca_map()
            return

        if self.canvas.pixmap is None or self.canvas.pixmap.isNull():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("PCA Feature Map"),
                self.tr(
                    "Load an image or video frame before generating a PCA map."),
            )
            self.pca_map_action.setChecked(False)
            return

        if not self.pca_map_model:
            self._open_pca_map_settings()
            if not self.pca_map_model:
                self.pca_map_action.setChecked(False)
                return

        self._request_pca_map()

    def _deactivate_pca_map(self):
        if hasattr(self, "pca_map_action"):
            self.pca_map_action.setChecked(False)
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setPCAMapOverlay(None)

    def _request_pca_map(self) -> None:
        if self.pca_map_service.is_busy():
            self.statusBar().showMessage(
                self.tr("PCA map is already running…"), 2000)
            return

        self.canvas.setPCAMapOverlay(None)
        pil_image = self._grab_current_frame_image()
        if pil_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("PCA Feature Map"),
                self.tr("Failed to access the current frame."),
            )
            self._deactivate_pca_map()
            return

        mask_bool = None
        selected_polygons = [
            shape
            for shape in getattr(self.canvas, "selectedShapes", [])
            if getattr(shape, "shape_type", "") == "polygon" and len(shape.points) >= 3
        ]
        if selected_polygons:
            mask_img = Image.new("L", pil_image.size, 0)
            draw = ImageDraw.Draw(mask_img)
            for polygon in selected_polygons:
                coords = [(float(pt.x()), float(pt.y()))
                          for pt in polygon.points]
                draw.polygon(coords, fill=255)
            mask_bool = np.array(mask_img) > 0

        cluster_k = self.pca_map_clusters if getattr(
            self, "pca_map_clusters", 0) > 1 else None
        request = DinoPCARequest(
            image=pil_image,
            model_name=self.pca_map_model,
            short_side=768,
            device=None,
            output_size="input",
            components=3,
            clip_percentile=1.0,
            alpha=float(self.pca_map_alpha),
            mask=mask_bool,
            cluster_k=cluster_k,
        )
        if not self.pca_map_service.request(request):
            self.statusBar().showMessage(
                self.tr("PCA map is already running…"), 2000)

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

    def clean_up(self):
        def quit_and_wait(thread, message):
            if thread is not None:
                try:
                    thread.quit()
                    thread.wait()
                except RuntimeError:
                    logger.info(message)

        quit_and_wait(self.frame_worker, "Thank you!")
        quit_and_wait(self.seg_train_thread, "See you next time!")
        quit_and_wait(self.seg_pred_thread, "Bye!")

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

    def loadPredictShapes(self, frame_number, filename):
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(filename)

        label_json_file = str(filename).replace(".png", ".json")
        # try to load json files generated by SAM2 like 000000000.json
        if not Path(label_json_file).exists():
            label_json_file = os.path.join(os.path.dirname(label_json_file),
                                           os.path.basename(label_json_file).split('_')[-1])
        if self._df is not None and not Path(filename).exists():
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

        if Path(label_json_file).exists():
            try:
                self.labelFile = LabelFile(label_json_file,
                                           is_video_frame=True)
                if self.labelFile:
                    self.canvas.setBehaviorText(None)
                    self.loadLabels(self.labelFile.shapes)
                    self.update_flags_from_file(self.labelFile)
                    if len(self.canvas.current_behavior_text) > 1 and 'other' not in self.canvas.current_behavior_text.lower():
                        self.add_highlighted_mark(self.frame_number,
                                                  mark_type=self.canvas.current_behavior_text)
                    caption = self.labelFile.get_caption()
                    if caption is not None and len(caption) > 0:
                        if self.caption_widget is None:
                            self.openCaption()
                        self.caption_widget.set_caption(caption)
            except Exception as e:
                print(e)

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
            self.set_frame_number(self.frame_number)
            # update the seekbar value
            self.seekbar.setValue(self.frame_number)
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

    # --- Handler for Tracking Initiated by SegmentEditorDialog ---

    # worker_instance, video_path_processed by dialog
    @Slot(TrackingWorker, Path)
    def _handle_tracking_initiated_by_dialog(self, worker_instance: TrackingWorker, video_path: Path):
        if self.active_tracking_worker and self.active_tracking_worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self, "Tracking Busy", "Another tracking job is already active. Please wait.")
            # Stop the newly created worker if we can't handle it
            worker_instance.stop()
            worker_instance.wait(500)  # Give it a moment to stop
            worker_instance.deleteLater()  # Schedule for deletion
            return

        logger.info(
            f"AnnolidWindow: Tracking initiated by SegmentEditorDialog for {video_path.name}")
        self.active_tracking_worker = worker_instance
        self._connect_signals_to_active_worker(self.active_tracking_worker)
        self._set_tracking_ui_state(is_tracking=True)
        # Worker is already started by the dialog.

    # Helper method for dialog to check (or internal check)
    def is_tracking_busy(self) -> bool:
        return bool(self.active_tracking_worker and self.active_tracking_worker.isRunning())

    # --- Generic Signal Connection & Handling for the active_tracking_worker ---
    def _connect_signals_to_active_worker(self, worker_instance: Optional[TrackingWorker]):
        if not worker_instance:
            logger.warning(
                "Attempted to connect signals to a null worker instance.")
            return

        # Disconnect from any previous worker to avoid duplicate signal handling
        # This logic needs to be robust if self.active_tracking_worker could be something else
        # For now, assume only one active_tracking_worker at a time.
        # Note: If the previous worker was already deleted or cleaned up, disconnect might raise error.
        previous_worker = getattr(self, '_previous_connected_worker', None)
        if previous_worker and previous_worker != worker_instance:
            try:
                previous_worker.progress.disconnect(
                    self._update_main_status_progress)
                previous_worker.finished.disconnect(
                    self._on_tracking_job_finished)
                previous_worker.error.disconnect(self._on_tracking_job_error)
                if hasattr(previous_worker, 'video_job_started'):
                    previous_worker.video_job_started.disconnect(
                        self._handle_tracking_video_started_ui_update)
                if hasattr(previous_worker, 'video_job_finished'):
                    previous_worker.video_job_finished.disconnect(
                        self._handle_tracking_video_finished_ui_update)
                logger.debug(
                    f"Disconnected signals from previous worker: {previous_worker.__class__.__name__}")
            except (TypeError, RuntimeError) as e:
                logger.debug(
                    f"Error disconnecting signals from previous worker (might be okay): {e}")

        self._previous_connected_worker = worker_instance  # Keep track for next disconnect

        worker_instance.progress.connect(self._update_main_status_progress)
        worker_instance.finished.connect(self._on_tracking_job_finished)
        worker_instance.error.connect(self._on_tracking_job_error)

        # video_job_started/finished are crucial for UI updates per video
        if hasattr(worker_instance, 'video_job_started'):
            worker_instance.video_job_started.connect(
                self._handle_tracking_video_started_ui_update)
        if hasattr(worker_instance, 'video_job_finished'):
            worker_instance.video_job_finished.connect(
                self._handle_tracking_video_finished_ui_update)

        logger.info(
            f"AnnolidWindow: Connected UI signals for worker: {worker_instance.__class__.__name__}")

    def _get_tracking_device(self) -> torch.device:  # Centralized device selection
        # More sophisticated logic could go here (e.g., user settings)
        if self.config.get('use_cpu_only', False):
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # --- Slots for worker signals ---
    @Slot(int, str)
    def _update_main_status_progress(self, percentage: int, message: str):
        self.statusBar().showMessage(f"{message} ({percentage}%)", 4000)

    @Slot(str)
    def _on_tracking_job_finished(self, completion_message: str):
        QtWidgets.QMessageBox.information(
            self, "Tracking Job Complete", completion_message)
        self.statusBar().showMessage(completion_message, 5000)
        self._set_tracking_ui_state(is_tracking=False)

        worker_that_finished = self.sender()  # Get the worker that emitted the signal
        if self.active_tracking_worker == worker_that_finished:
            # Disconnect signals before clearing reference or deleting
            try:
                self.active_tracking_worker.progress.disconnect(
                    self._update_main_status_progress)
                self.active_tracking_worker.finished.disconnect(
                    self._on_tracking_job_finished)
                self.active_tracking_worker.error.disconnect(
                    self._on_tracking_job_error)
                if hasattr(self.active_tracking_worker, 'video_job_started'):
                    self.active_tracking_worker.video_job_started.disconnect(
                        self._handle_tracking_video_started_ui_update)
                if hasattr(self.active_tracking_worker, 'video_job_finished'):
                    self.active_tracking_worker.video_job_finished.disconnect(
                        self._handle_tracking_video_finished_ui_update)
            except (TypeError, RuntimeError):
                logger.debug("Error disconnecting from finished worker.")

            # If the worker's parent was None (dialog created it this way), schedule for deletion
            if self.active_tracking_worker.parent() is None:
                self.active_tracking_worker.deleteLater()
                logger.info("Scheduled dialog-created worker for deletion.")
            self.active_tracking_worker = None
        elif worker_that_finished:  # Some other worker finished
            worker_that_finished.deleteLater()  # If it's not the main one, clean it up too
            logger.info(
                f"An external worker ({worker_that_finished.__class__.__name__}) finished and was scheduled for deletion.")

    @Slot(str)
    def _on_tracking_job_error(self, error_message: str):
        QtWidgets.QMessageBox.critical(
            self, "Tracking Job Error", error_message)
        self.statusBar().showMessage(
            f"Error: {error_message}", 0)  # Persistent
        self._set_tracking_ui_state(is_tracking=False)
        worker_that_errored = self.sender()
        if self.active_tracking_worker == worker_that_errored:
            if self.active_tracking_worker.parent() is None:
                self.active_tracking_worker.deleteLater()
            self.active_tracking_worker = None
        elif worker_that_errored:
            worker_that_errored.deleteLater()

    @Slot(str, str)
    def _handle_tracking_video_started_ui_update(self, video_path_str: str, output_folder_str: str):
        logger.info(
            f"AnnolidWindow UI: Job started for video {video_path_str}")
        if self.filename != video_path_str:  # If the worker is processing a video not currently on canvas
            logger.info(
                f"Worker started on {video_path_str}, but canvas shows {self.filename}. Opening programmatically.")
            # This ensures the canvas shows what the worker is processing for marker updates
            self.openVideo(from_video_list=True,
                           video_path=video_path_str,
                           programmatic_call=True)

        # Now, self.filename should be video_path_str
        if self.video_file == video_path_str:  # self.video_file is usually set by openVideo/loadFile
            # self.video_results_folder is also set by openVideo/loadFile for labelme
            # For consistency, let's ensure it matches output_folder_str or update it
            expected_results_folder = Path(output_folder_str)
            if self.video_results_folder != expected_results_folder:
                logger.warning(
                    f"Mismatch in video_results_folder. Expected: {expected_results_folder}, Have: {self.video_results_folder}. Forcing update.")
                # Ensure watcher uses correct folder
                self.video_results_folder = expected_results_folder

            self._setup_prediction_folder_watcher(
                str(output_folder_str))  # Start watching for JSONs
            # self._initialize_progress_bar() # If you have a per-file progress bar in labelme
        else:
            logger.error(f"Critical: Mismatch after attempting to open video for tracking. "
                         f"Current: {self.video_file}, Expected by worker: {video_path_str}.")

    @Slot(str)
    def _handle_tracking_video_finished_ui_update(self, video_path_str: str):
        logger.info(
            f"AnnolidWindow UI: Job finished for video {video_path_str}")
        # Clean up UI specific to this video (e.g., marker watcher)
        # Only finalize if this video_path_str matches what the watcher is currently on
        current_watched_folder_path_str = ""
        if self.prediction_progress_watcher and self.prediction_progress_watcher.directories():
            current_watched_folder_path_str = self.prediction_progress_watcher.directories()[
                0]

        if Path(video_path_str).with_suffix('') == Path(current_watched_folder_path_str):
            self._finalize_prediction_progress(
                f"GUI finalized for {Path(video_path_str).name}.")
        else:
            logger.info(
                f"GUI: Video {video_path_str} finished, but watcher was on {current_watched_folder_path_str} or not active for UI updates.")

    def _set_tracking_ui_state(self, is_tracking: bool):
        self.open_segment_editor_action.setEnabled(
            not is_tracking and bool(self.video_file))
        # Add other UI elements to disable/enable
        # For example, file opening actions from labelme's self.actions
        if hasattr(self.actions, 'open'):
            self.actions.open.setEnabled(not is_tracking)
        if hasattr(self.actions, 'openDir'):
            self.actions.openDir.setEnabled(not is_tracking)
        if hasattr(self.actions, 'openVideo'):
            self.actions.openVideo.setEnabled(
                not is_tracking)  # Your main video open

        # If VideoManagerWidget is present and has its own track all button
        if hasattr(self, 'video_manager_widget') and hasattr(self.video_manager_widget, 'track_all_button'):
            self.video_manager_widget.track_all_button.setEnabled(
                not is_tracking)

        logger.info(
            f"AnnolidWindow UI state for tracking: {'ACTIVE' if is_tracking else 'IDLE'}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="show version"
    )
    # config for the gui
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--labels',
        default=argparse.SUPPRESS,
        help="comma separated list of labels or file containing labels"
    )

    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )

    default_config_file = str(Path.home() / '.labelmerc')
    parser.add_argument(
        '--config',
        dest="config",
        default=default_config_file,
        help=f"config file or yaml format string default {default_config_file}"
    )

    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    if hasattr(args, "labels"):
        if Path(args.labels).is_file():
            with codecs.open(args.labels,
                             'r', encoding='utf-8'
                             ) as f:
                args.labels = [line.strip()
                               for line in f if line.strip()
                               ]
        else:
            args.labels = [
                line for line in args.labels.split(',')
                if line
            ]

    config_from_args = args.__dict__
    config_from_args.pop("version")
    config_file_or_yaml = config_from_args.pop("config")

    config = get_config(config_file_or_yaml, config_from_args)

    app = QtWidgets.QApplication(sys.argv)

    app.setApplicationName(__appname__)
    annolid_icon = QtGui.QIcon(
        str(Path(__file__).resolve().parent / "icons/icon_annolid.png"))
    app.setWindowIcon(annolid_icon)
    win = AnnolidWindow(config=config)
    logger.info("Qt config file: %s" % win.settings.fileName())

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
