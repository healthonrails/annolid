from __future__ import annotations
# ruff: noqa: E402

# Enable CPU fallback for unsupported MPS ops
import os  # noqa

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # noqa
# Windows: mitigate OpenMP runtime conflicts (e.g. PyTorch + ONNX Runtime).
if os.name == "nt":  # noqa
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa

import sys
from pathlib import Path
from annolid.infrastructure.runtime import (
    configure_qt_runtime,
)

try:
    configure_qt_runtime()
except Exception as exc:
    if sys.platform == "darwin":
        print(
            f"Warning: Failed to initialize Qt runtime config: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )

from qtpy import QtCore
from qtpy.QtCore import Signal
from qtpy import QtWidgets
from qtpy import QtGui
from annolid.gui.window_base import (
    AnnolidWindowBase,
)
from annolid.gui.workers import (
    FlexibleWorker,
    LoadFrameThread,
)
from annolid.utils.logger import configure_logging, logger
from annolid.gui.widgets.canvas import Canvas
from annolid.domain import ProjectSchema
from annolid.gui.widgets import AnnolidLabelDialog
import atexit
from annolid.gui.widgets.step_size_widget import StepSizeWidget
from annolid.gui.widgets import CanvasScreenshotWidget
from annolid.gui.widgets.pdf_import_widget import PdfImportWidget
from annolid.gui.features import (
    GuiFeatureDeps,
    setup_annotation_feature,
    setup_search_feature,
    setup_timeline_feature,
    setup_video_feature,
    setup_viewers_feature,
)

from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.model_manager import AIModelManager
from annolid.gui.models_registry import (
    get_runtime_model_registry,
    validate_model_registry_entries,
)
from typing import Any, Dict, List, Optional, Set
from annolid.jobs.tracking_jobs import TrackingSegment

from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.gui.behavior_controller import BehaviorController
from annolid.gui.keypoint_catalog import extract_labels_from_uniq_label_list
from annolid.gui.yolo_training_manager import YOLOTrainingManager
from annolid.gui.dino_kpseg_training_manager import DinoKPSEGTrainingManager
from annolid.gui.cli import parse_cli
from annolid.infrastructure.runtime import create_qapp, sanitize_qt_plugin_env
from annolid.gui.controllers import (
    AnnotationController,
    DinoController,
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
    status_message_requested = Signal(str, int)

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
        self.status_message_requested.connect(self._show_status_message)

        # self.flag_dock.setVisible(True)
        self.flag_widget.close()
        self.flag_widget = None
        self.label_dock.setVisible(True)
        self.shape_dock.setVisible(True)
        self.file_dock.setVisible(True)
        self._other_docks_states: Dict[QtWidgets.QDockWidget, bool] = {}

        self.csv_thread = None
        self.csv_worker = None
        self._last_tracking_csv_path = None
        self._csv_conversion_queue = []
        self._prediction_stop_requested = False
        self.florence_dock: Optional[QtWidgets.QDockWidget] = None
        self.tracking_controller = TrackingController(self)
        feature_deps = GuiFeatureDeps(
            window=self,
            status_message=self.post_status_message,
        )
        self.feature_states: Dict[str, object] = {}

        self.feature_states["video"] = setup_video_feature(feature_deps)

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
        self._validate_model_registry_startup()
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
        self.videomt_mask_threshold = 0.5
        self.videomt_logit_threshold = -2.0
        self.videomt_seed_iou_threshold = 0.01
        self.videomt_window = 8
        self.videomt_input_height = 0
        self.videomt_input_width = 0
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
        self.feature_states["viewers"] = setup_viewers_feature(feature_deps)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.selectionChanged.connect(self.large_image_view.set_selected_shapes)
        self.large_image_view.selectionChanged.connect(self.canvas.selectShapes)
        self.large_image_view.shapeMoved.connect(self.canvas.storeShapes)
        self.large_image_view.shapeMoved.connect(self.setDirty)
        self.large_image_view.newShape.connect(self.newShape)
        self.large_image_view.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.vertexSelected.connect(self.actions.removePoint.setEnabled)
        self._setup_label_list_connections()
        self._setup_file_list_connections()

        self.feature_states["annotation"] = setup_annotation_feature(feature_deps)

        # Ensure all drawing/edit mode actions work without relying on LabelMe.
        self._setup_drawing_mode_actions()

        self.dino_controller = DinoController(self)
        self.dino_controller.initialize()

        self.tracking_data_controller = TrackingDataController(self)

        self._behavior_event_detail_dialog = None
        self.feature_states["search"] = setup_search_feature(feature_deps)
        self._apply_agent_mode(self._agent_mode_enabled)
        self.feature_states["timeline"] = setup_timeline_feature(feature_deps)
        self.setCentralWidget(self._main_scroll_area)

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
        if hasattr(self, "setupVectorOverlayDock"):
            self.setupVectorOverlayDock()
        self._setup_canvas_screenshot_action()
        self._setup_open_pdf_action()
        self._setup_label_collection_action()
        self._setup_log_manager_action()

        self.populateModeActions()
        QtCore.QTimer.singleShot(0, self._restore_last_worked_file_if_available)
        QtCore.QTimer.singleShot(0, self._startup_annolid_bot)

    @QtCore.Slot(str, int)
    def _show_status_message(self, message: str, timeout: int = 4000) -> None:
        self.statusBar().showMessage(str(message or ""), int(timeout or 0))

    def post_status_message(self, message: str, timeout: int = 4000) -> None:
        self.status_message_requested.emit(str(message or ""), int(timeout or 0))

    def _setup_keypoint_sequence_quick_toggle(self) -> None:
        """Add quick toggle action (toolbar + shortcut) for keypoint sequencing."""
        shortcut = self._shortcut("toggle_keypoint_sequence") or "Ctrl+Shift+K"
        action = QtWidgets.QAction(self.tr("Keypoint Sequence"), self)
        action.setCheckable(True)
        action.setChecked(
            bool(self.keypoint_sequence_widget.enable_checkbox.isChecked())
        )
        action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setStatusTip(
            self.tr("Enable/disable sequential keypoint labeling for point clicks")
        )
        action.setToolTip(
            self.tr("Toggle keypoint sequencer on/off (shortcut: %s)") % shortcut
        )
        action.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))

        action.toggled.connect(self.keypoint_sequence_widget.enable_checkbox.setChecked)
        self.keypoint_sequence_widget.enable_checkbox.toggled.connect(action.setChecked)

        self.toggle_keypoint_sequence_action = action
        try:
            self.menus.view.addAction(action)
        except Exception:
            pass
        try:
            self.tools.add_stacked_action(
                action,
                "Keypoint\nSequence",
                width=58,
                min_height=68,
                icon_size=QtCore.QSize(32, 32),
            )
        except Exception:
            self.tools.addAction(action)

    def _setup_keypoint_sequence_label_sync(self) -> None:
        """Keep sequencer keypoints merged with Labels dock entries."""
        uniq = getattr(self, "uniqLabelList", None)
        if uniq is None:
            return
        try:
            model = uniq.model()
        except Exception:
            model = None
        if model is not None:
            try:
                model.rowsInserted.connect(
                    self._sync_keypoint_sequencer_from_labels_dock
                )
                model.rowsRemoved.connect(
                    self._sync_keypoint_sequencer_from_labels_dock
                )
                model.modelReset.connect(self._sync_keypoint_sequencer_from_labels_dock)
            except Exception:
                pass
        try:
            self._sync_keypoint_sequencer_from_labels_dock()
        except Exception:
            pass

    def _sync_keypoint_sequencer_from_labels_dock(self, *args) -> None:
        _ = args
        widget = getattr(self, "keypoint_sequence_widget", None)
        uniq = getattr(self, "uniqLabelList", None)
        if widget is None or uniq is None:
            return
        labels = extract_labels_from_uniq_label_list(uniq)
        if labels:
            widget.load_keypoints_from_labels(labels)

    def _on_keypoint_sequence_schema_changed(self, schema, schema_path: str) -> None:
        if schema is None:
            return
        self._pose_schema = schema
        if schema_path:
            self._pose_schema_path = str(schema_path)
        try:
            self.canvas.setPoseSchema(schema)
        except Exception:
            pass
        if schema_path:
            try:
                self._persist_pose_schema_to_project_schema(schema, str(schema_path))
            except Exception:
                pass

    def _validate_model_registry_startup(self) -> None:
        try:
            registry = get_runtime_model_registry(
                config=self._config,
                settings=self.settings,
            )
            is_valid, errors, warnings = validate_model_registry_entries(registry)
            for message in warnings:
                logger.warning("Model registry warning: %s", message)
            if not is_valid:
                for message in errors:
                    logger.error("Model registry error: %s", message)
        except Exception as exc:
            logger.warning("Model registry startup validation failed: %s", exc)

    def _startup_annolid_bot(self) -> None:
        """Start Annolid Bot when the main window opens."""
        if os.environ.get("ANNOLID_DISABLE_BOT_AUTOSTART"):
            return
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
            return
        manager = getattr(self, "ai_chat_manager", None)
        if manager is None:
            return
        try:
            manager.initialize_annolid_bot(start_visible=False)
            self.statusBar().showMessage(self.tr("Annolid Bot ready."), 3000)
        except Exception as exc:
            logger.warning("Failed to auto-start Annolid Bot: %s", exc)

    def _restore_last_worked_file_if_available(self) -> None:
        """Cache last worked file from settings; restore only after opening its folder."""
        # Keep automated tests deterministic; avoid opening user files in CI/test runs.
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        self._pending_last_worked_file = ""
        try:
            enabled = bool(
                self.settings.value("session/restore_last_worked_file", True, type=bool)
            )
        except Exception:
            enabled = True
        if not enabled:
            return
        try:
            last_file = str(
                self.settings.value("session/last_worked_file", "", type=str) or ""
            ).strip()
        except Exception:
            last_file = ""
        if not last_file:
            return
        last_path = Path(last_file).expanduser()
        if not last_path.exists() or not last_path.is_file():
            return
        self._pending_last_worked_file = str(last_path)

    def _pending_last_worked_file_for_directory(self, directory: str) -> str:
        """Return pending restore file if it is contained by the opened directory."""
        pending = str(getattr(self, "_pending_last_worked_file", "") or "").strip()
        if not pending:
            return ""
        pending_path = Path(pending).expanduser()
        if not pending_path.exists() or not pending_path.is_file():
            self._pending_last_worked_file = ""
            return ""
        try:
            dir_path = Path(directory).expanduser().resolve()
            pending_path.resolve().relative_to(dir_path)
        except Exception:
            return ""
        return str(pending_path)

    def _clear_pending_last_worked_file(self) -> None:
        self._pending_last_worked_file = ""

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        return super().keyReleaseEvent(event)

    def set_unrelated_docks_visible(
        self, visible: bool, exclude: Optional[List[QtWidgets.QDockWidget]] = None
    ) -> None:
        """Hide all docks except excluded ones for distraction-free viewing."""
        if not visible:
            # Only save currently visible docks if we are hiding.
            # Don't overwrite states if we're already in a "hidden" mode.
            if not self._other_docks_states:
                for dock in self.findChildren(QtWidgets.QDockWidget):
                    if exclude and dock in exclude:
                        continue
                    if dock.isVisible():
                        self._other_docks_states[dock] = True
                        dock.hide()
        else:
            # Restore previously hidden docks.
            restore_states = list(self._other_docks_states.items())
            # Clear first so re-entrant visibility changes can store fresh state
            # without being wiped after this restore pass.
            self._other_docks_states.clear()
            for dock, was_visible in restore_states:
                if was_visible:
                    try:
                        dock.show()
                        dock.raise_()
                    except (RuntimeError, Exception):
                        continue


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

    try:
        configure_logging(enable_file_logging=True)
    except TypeError:
        configure_logging()

    # OpenCV may reset Qt plugin env vars during import; sanitize again right
    # before QApplication is constructed so Qt does not resolve cv2 plugins.
    sanitize_qt_plugin_env(os.environ)

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
