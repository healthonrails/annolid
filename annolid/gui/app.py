from __future__ import annotations

# Enable CPU fallback for unsupported MPS ops
import os  # noqa

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # noqa
# Windows: mitigate OpenMP runtime conflicts (e.g. PyTorch + ONNX Runtime).
if os.name == "nt":  # noqa
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa

import sys
from pathlib import Path

from qtpy import QtCore
from qtpy.QtCore import Qt, Signal
from qtpy import QtWidgets
from qtpy import QtGui
from annolid.gui.window_base import (
    AnnolidWindowBase,
)
from annolid.gui.widgets.video_manager import VideoManagerWidget
from annolid.gui.workers import (
    FlexibleWorker,
    LoadFrameThread,
)
from annolid.utils.logger import configure_logging, logger
from annolid.gui.widgets.canvas import Canvas
from annolid.core.behavior.spec import (
    ProjectSchema,
)
from annolid.gui.widgets.behavior_controls import BehaviorControlsWidget
from annolid.gui.widgets import FlagTableWidget
from annolid.gui.widgets import AnnolidLabelDialog
import atexit
from annolid.gui.widgets.step_size_widget import StepSizeWidget
from annolid.gui.widgets import CanvasScreenshotWidget
from annolid.gui.widgets.pdf_import_widget import PdfImportWidget
from annolid.gui.widgets.pdf_manager import PdfManager
from annolid.gui.widgets.depth_manager import DepthManager
from annolid.gui.widgets.sam3d_manager import Sam3DManager
from annolid.gui.widgets.sam2_manager import Sam2Manager
from annolid.gui.widgets.sam3_manager import Sam3Manager
from annolid.gui.widgets.optical_flow_manager import OpticalFlowManager
from annolid.gui.widgets.realtime_manager import RealtimeManager

from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.model_manager import AIModelManager
from typing import Any, Dict, List, Optional, Set
from annolid.jobs.tracking_jobs import TrackingSegment

from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.gui.behavior_controller import BehaviorController
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget
from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget
from annolid.gui.widgets.timeline_panel import TimelinePanel
from annolid.gui.yolo_training_manager import YOLOTrainingManager
from annolid.gui.dino_kpseg_training_manager import DinoKPSEGTrainingManager
from annolid.gui.cli import parse_cli
from annolid.gui.application import create_qapp
from annolid.gui.controllers import (
    AnnotationController,
    DinoController,
    FlagsController,
    InferenceController,
    MenuController,
    ProjectController,
    TrackingController,
    TrackingDataController,
    VideoController,
)
from annolid.gui.managers import SettingsManager
from annolid.gui.mixins import (
    AnnolidWindowMixinBundle,
)
from annolid.gui.theme import apply_modern_theme, apply_light_theme, apply_dark_theme
from annolid.version import __version__


__appname__ = "Annolid"


class AnnolidWindow(AnnolidWindowMixinBundle, AnnolidWindowBase):
    """Annolid main window built on AnnolidWindowBase."""

    live_annolid_frame_updated = Signal(int, str)  # For modeless dialogs if any

    def __init__(self, config=None):
        self.config = config
        tracker_cfg = dict((self.config or {}).get("tracker", {}) or {})
        tracker_fields = set(CutieDinoTrackerConfig.__dataclass_fields__)
        unsupported_tracker_keys = set(tracker_cfg.keys()) - tracker_fields
        if unsupported_tracker_keys:
            logger.warning(
                "Ignoring unsupported tracker config keys: %s",
                sorted(unsupported_tracker_keys),
            )
        tracker_kwargs = {k: v for k, v in tracker_cfg.items() if k in tracker_fields}
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
            self.handle_extracted_frames
        )
        self.video_manager_widget.json_saved.connect(
            self.video_manager_widget.update_json_column
        )

        self.video_manager_widget.track_all_worker_created.connect(
            self.tracking_controller.register_track_all_worker
        )

        # Create the Dock Widget
        self.video_dock = QtWidgets.QDockWidget("Video List", self)
        # Set a unique objectName
        self.video_dock.setObjectName("videoListDock")
        self.video_dock.setWidget(self.video_manager_widget)
        self.video_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Add the Dock Widget to the Main Window
        self.addDockWidget(Qt.RightDockWidgetArea, self.video_dock)

        self.here = Path(__file__).resolve().parent
        self.settings_manager = SettingsManager("Annolid", "Annolid")
        self.settings = self.settings_manager.qt_settings
        ui_settings = self.settings_manager.get_ui_settings()
        pose_settings = self.settings_manager.get_pose_settings()
        self._agent_mode_enabled = bool(ui_settings.get("agent_mode", True))
        self._show_embedding_search = bool(
            ui_settings.get("show_embedding_search", False)
        )
        self._show_pose_edges = bool(pose_settings.get("show_edges", True))
        self._show_pose_bboxes = bool(pose_settings.get("show_bbox", True))
        self._save_pose_bbox = bool(pose_settings.get("save_bbox", True))
        self.annotation_controller = AnnotationController(parent=self)
        self.video_controller = VideoController(parent=self)
        self.inference_controller = InferenceController(parent=self)
        self.project_controller = ProjectController(parent=self)
        self._agent_run_config: Dict[str, Any] = {}
        self._agent_thread: Optional[QtCore.QThread] = None
        self._agent_worker: Optional[FlexibleWorker] = None
        self._agent_progress_dialog: Optional[QtWidgets.QProgressDialog] = None
        self._df = None
        self._df_deeplabcut = None
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_columns = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        self._df_deeplabcut_multi_animal = False
        self.label_stats = {}
        self.shape_hash_ids = {}
        self._noSelectionSlot = False
        self.changed_json_stats = {}
        self._pred_res_folder_suffix = "_tracking_results_labelme"
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
        self._time_stamp = ""
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
        self.progress_bar.setVisible(False)
        self._progress_bar_owner: Optional[str] = None

        self._current_video_defined_segments: List[TrackingSegment] = []
        self.menu_controller = MenuController(self)
        self.menu_controller.setup()

        # In-tree replacement for LabelMe's label dialog.
        self.labelDialog = AnnolidLabelDialog(parent=self, config=self._config)

        self.prediction_progress_watcher = None
        self.last_known_predicted_frame = -1  # Track the latest frame seen
        self.prediction_start_timestamp = 0.0
        self._prediction_progress_mark = None
        self._prediction_start_frame = None
        self._prediction_existing_store_frames = set()
        self._prediction_existing_json_frames = set()
        self._prediction_store_path = None
        self._prediction_store_baseline_size = 0
        self._prediction_appended_frames = set()
        self._follow_prediction_progress = True

        self.canvas = self.labelList.canvas = Canvas(
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            crosshair=self._config["canvas"]["crosshair"],
            sam=self._config["sam"],
        )
        try:
            self.canvas.setShowPoseEdges(self._show_pose_edges)
        except Exception:
            pass
        try:
            self.canvas.setShowPoseBBoxes(self._show_pose_bboxes)
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
        self._setup_label_list_connections()
        self._setup_file_list_connections()

        # Ensure all drawing/edit mode actions work without relying on LabelMe.
        self._setup_drawing_mode_actions()

        self.flag_widget = FlagTableWidget()
        self.flag_dock.setWidget(self.flag_widget)
        self.flags_controller = FlagsController(
            window=self,
            widget=self.flag_widget,
            config_path=self.here.parent.resolve() / "configs" / "behaviors.yaml",
        )
        self.flags_controller.initialize()

        # Ensure flag_dock is visible and raised (shown as the active tab)
        self.flag_dock.setVisible(True)
        self.flag_dock.raise_()

        self.dino_controller = DinoController(self)
        self.dino_controller.initialize()

        self.tracking_data_controller = TrackingDataController(self)

        # Behavior event log dock
        self.behavior_log_widget = BehaviorEventLogWidget(
            self, color_getter=self._get_rgb_by_label
        )
        self.behavior_log_widget.jumpToFrame.connect(self._jump_to_frame_from_log)
        self.behavior_log_widget.undoRequested.connect(self.undo_last_behavior_event)
        self.behavior_log_widget.clearRequested.connect(
            self._clear_behavior_events_from_log
        )
        self.behavior_log_widget.behaviorSelected.connect(
            self._show_behavior_event_details
        )
        self.behavior_log_widget.editRequested.connect(
            self._edit_behavior_event_from_log
        )
        self.behavior_log_widget.deleteRequested.connect(
            self._delete_behavior_event_from_log
        )
        self.behavior_log_widget.confirmRequested.connect(
            self._confirm_behavior_event_from_log
        )
        self.behavior_log_widget.rejectRequested.connect(
            self._reject_behavior_event_from_log
        )

        self.behavior_log_dock = QtWidgets.QDockWidget("Behavior Log", self)
        self.behavior_log_dock.setObjectName("behaviorLogDock")
        self.behavior_log_dock.setWidget(self.behavior_log_widget)
        self.behavior_log_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.behavior_log_dock)

        self._behavior_event_detail_dialog = None

        self.embedding_search_widget = EmbeddingSearchWidget(self)
        self.embedding_search_widget.jumpToFrame.connect(self._jump_to_frame_from_log)
        self.embedding_search_widget.statusMessage.connect(
            lambda msg: self.statusBar().showMessage(msg, 4000)
        )
        self.embedding_search_widget.labelFramesRequested.connect(
            self._label_frames_from_search
        )
        self.embedding_search_widget.markFramesRequested.connect(
            self._mark_similar_frames_from_search
        )
        self.embedding_search_widget.clearMarkedFramesRequested.connect(
            self._clear_similar_frame_marks
        )
        self.embedding_search_dock = QtWidgets.QDockWidget("Embedding Search", self)
        self.embedding_search_dock.setObjectName("embeddingSearchDock")
        self.embedding_search_dock.setWidget(self.embedding_search_widget)
        self.embedding_search_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.embedding_search_dock)
        try:
            self.tabifyDockWidget(self.video_dock, self.embedding_search_dock)
            self.video_dock.show()
            self.video_dock.raise_()
        except Exception:
            pass
        self._apply_agent_mode(self._agent_mode_enabled)

        self.behavior_controls_widget = BehaviorControlsWidget(self)
        self.behavior_controls_widget.subjectChanged.connect(
            self._on_active_subject_changed
        )
        self.behavior_controls_widget.modifierToggled.connect(self._on_modifier_toggled)
        self.behavior_controls_dock = QtWidgets.QDockWidget("Behavior Controls", self)
        self.behavior_controls_dock.setObjectName("behaviorControlsDock")
        self.behavior_controls_dock.setWidget(self.behavior_controls_widget)
        self.behavior_controls_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.behavior_controls_dock)
        self.tabifyDockWidget(self.behavior_log_dock, self.behavior_controls_dock)
        self.behavior_log_dock.raise_()

        self.timeline_panel = TimelinePanel(self)
        self.timeline_panel.frameSelected.connect(self._jump_to_frame_from_log)
        self.timeline_panel.set_behavior_controller(
            self.behavior_controller, color_getter=self._get_rgb_by_label
        )
        self.timeline_panel.set_timestamp_provider(self._estimate_recording_time)
        self.timeline_panel.set_behavior_catalog(
            provider=self._timeline_behavior_catalog,
            adder=self._timeline_add_behavior,
        )
        try:
            self.flag_widget.flagsSaved.connect(
                self.timeline_panel.refresh_behavior_catalog
            )
            self.flag_widget.rowSelected.connect(
                self.timeline_panel.set_active_behavior
            )
            self.flag_widget.rowSelected.connect(
                lambda _name: self.timeline_panel.refresh_behavior_catalog()
            )
            self.flag_widget.flagToggled.connect(
                lambda _name, _state: self.timeline_panel.refresh_behavior_catalog()
            )
        except Exception:
            pass
        self.timeline_dock = QtWidgets.QDockWidget("Timeline", self)
        self.timeline_dock.setObjectName("timelineDock")
        self.timeline_dock.setWidget(self.timeline_panel)
        self.timeline_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.timeline_dock)
        self._setup_timeline_view_toggle()
        # Only show the timeline when a video is opened and the user enables it.
        self._apply_timeline_dock_visibility(video_open=False)
        self._apply_fixed_dock_sizes()

        self.setCentralWidget(scrollArea)

        self.statusBar().showMessage(self.tr("%s started.") % __appname__)
        self.statusBar().show()
        self.setWindowTitle(__appname__)
        # Restore application settings.
        self.recentFiles = self.settings.value("recentFiles", []) or []
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        size = self.settings.value("window/size", QtCore.QSize(1600, 900))
        if isinstance(position, QtCore.QPoint):
            self.move(position)
        if (
            isinstance(size, QtCore.QSize)
            and size.width() > 200
            and size.height() > 200
        ):
            self.resize(size)
        self._window_state_save_timer = QtCore.QTimer(self)
        self._window_state_save_timer.setSingleShot(True)
        self._window_state_save_timer.setInterval(300)
        self._window_state_save_timer.timeout.connect(self._persist_window_geometry)
        self._fit_window_applied_video_key: Optional[str] = None

        self.video_results_folder = None
        self.seekbar = None
        self.audio_widget = None
        self.audio_dock = None
        self._audio_loader = None
        self._suppress_audio_seek = False
        self.caption_widget = None

        self.frame_worker = QtCore.QThread()
        self.frame_loader = LoadFrameThread()
        self.seg_pred_thread = QtCore.QThread()
        self.seg_train_thread = QtCore.QThread()
        self.destroyed.connect(self.clean_up)
        self.stepSizeWidget.valueChanged.connect(self.update_step_size)
        self.stepSizeWidget.predict_button.pressed.connect(self.predict_from_next_frame)
        atexit.register(self.clean_up)
        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        # Prefer Cutie as the GUI default for the AI model dropdown.
        # This is a UX default only; users can pick a different model at runtime.
        self.ai_model_manager.initialize(default_selection="Cutie")

        self.canvas_screenshot_widget = CanvasScreenshotWidget(
            canvas=self.canvas, here=Path(__file__).resolve().parent
        )
        self.pdf_import_widget = PdfImportWidget(self)
        self._setup_canvas_screenshot_action()
        self._setup_open_pdf_action()
        self._setup_label_collection_action()

        self.populateModeActions()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        return super().keyReleaseEvent(event)


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

    configure_logging()

    qt_args = sys.argv if argv is None else [sys.argv[0], *argv]
    app = create_qapp(qt_args)

    # Apply global theme preference only if user explicitly selected one.
    try:
        settings = QtCore.QSettings("Annolid", "Annolid")
        theme_choice = str(settings.value("ui/theme", "") or "")
        if theme_choice == "modern":
            apply_modern_theme(app)
        elif theme_choice == "dark":
            apply_dark_theme(app)
        elif theme_choice == "light":
            apply_light_theme(app)
        # If theme_choice is empty or unknown, do not apply any custom theme.
    except Exception:
        pass

    app.setApplicationName(__appname__)
    annolid_icon = QtGui.QIcon(
        str(Path(__file__).resolve().parent / "icons/icon_annolid.png")
    )
    app.setWindowIcon(annolid_icon)
    win = AnnolidWindow(config=config)
    logger.info("Qt config file: %s" % win.settings.fileName())

    win.show()
    win.raise_()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
