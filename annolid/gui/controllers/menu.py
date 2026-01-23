import functools
from typing import Dict, TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.theme import (
    apply_modern_theme,
    apply_light_theme,
    apply_dark_theme,
    refresh_app_styles,
)

from labelme import utils
from labelme.utils import newAction

from annolid.gui.widgets.text_prompt import AiRectangleWidget
from annolid.gui.widgets import RecordingWidget


if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class MenuController:
    """Manage Annolid main window menus, tool actions, and related widgets."""

    def __init__(self, window: "AnnolidWindow") -> None:
        self._window = window
        self._action_factory = functools.partial(newAction, window)
        self._actions: Dict[str, QtWidgets.QAction] = {}

    def setup(self) -> None:
        """Create actions, populate menus/toolbars, and configure custom menus."""
        self._ensure_all_menus()
        self._create_core_actions()
        self._populate_tools_and_menus()
        self._reorder_top_menus()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_all_menus(self) -> None:
        """Create all custom menus for organized, user-friendly navigation."""
        w = self._window
        action = self._action_factory

        # Video Tools menu
        if not hasattr(w.menus, "video_tools"):
            w.menus.video_tools = QtWidgets.QMenu(w.tr("&Video Tools"), w)
        w.open_segment_editor_action = action(
            w.tr("Define Video Segments..."),
            w._open_segment_editor_dialog,
            shortcut="Ctrl+Alt+S",
            tip=w.tr("Define tracking segments for the current video"),
        )
        w.open_segment_editor_action.setEnabled(False)

        # AI & Models menu - central hub for machine learning
        if not hasattr(w.menus, "ai_models"):
            w.menus.ai_models = QtWidgets.QMenu(w.tr("&AI && Models"), w)

        # Analysis menu - reports and visualization
        if not hasattr(w.menus, "analysis"):
            w.menus.analysis = QtWidgets.QMenu(w.tr("&Analysis"), w)

        # Convert menu - format conversions
        if not hasattr(w.menus, "convert"):
            w.menus.convert = QtWidgets.QMenu(w.tr("&Convert"), w)

        # Settings menu
        if not hasattr(w.menus, "settings"):
            w.menus.settings = QtWidgets.QMenu(w.tr("&Settings"), w)

    def _create_core_actions(self) -> None:
        w = self._window
        here = w.here
        registry = self._actions

        w.createPolygonSAMMode = self._action_factory(
            w.tr("AI Polygons"),
            w.segmentAnything,
            icon="objects",
            tip=w.tr("Start creating polygons with segment anything"),
        )

        create_ai_polygon_mode = self._action_factory(
            w.tr("Create AI-Polygon"),
            lambda: w.toggleDrawMode(False, createMode="ai_polygon"),
            None,
            "objects",
            w.tr("Start drawing ai_polygon. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        create_ai_polygon_mode.changed.connect(
            lambda: w.canvas.initializeAiModel(
                name=w._selectAiModelComboBox.currentText(),
                _custom_ai_models=w.ai_model_manager.custom_model_names,
            )
            if w.canvas.createMode == "ai_polygon"
            else None
        )
        w.actions.createAiPolygonMode = create_ai_polygon_mode

        w.createGroundingSAMMode = self._action_factory(
            w.tr("Create GroundingSAM"),
            lambda: w.toggleDrawMode(False, createMode="grounding_sam"),
            None,
            "objects",
            w.tr("Start using grounding_sam"),
            enabled=False,
        )
        w.actions.createGroundingSAMMode = w.createGroundingSAMMode

        step_size = QtWidgets.QWidgetAction(w)
        step_size.setIcon(QtGui.QIcon(str(here / "icons/fast_forward.png")))
        step_size.setDefaultWidget(w.stepSizeWidget)
        w.stepSizeWidget.setWhatsThis(w.tr("Step for the next or prev image. e.g. 30"))
        w.stepSizeWidget.setEnabled(False)
        registry["step_size"] = step_size

        simple_specs = [
            {
                "name": "open_video",
                "text": w.tr("&Open Video"),
                "slot": w.openVideo,
                "tip": w.tr("Open video"),
                "icon_path": here / "icons/open_video.png",
            },
            {
                "name": "open_youtube_video",
                "text": w.tr("Open &YouTube Video"),
                "slot": w.open_youtube_video,
                "tip": w.tr("Download a YouTube video and open it in Annolid"),
            },
            {
                "name": "advance_params",
                "text": w.tr("&Advanced Parameters"),
                "slot": w.set_advanced_params,
                "tip": w.tr("Advanced Parameters"),
            },
            {
                "name": "open_audio",
                "text": w.tr("&Open Audio"),
                "slot": w.openAudio,
                "tip": w.tr("Open Audio"),
            },
            {
                "name": "open_caption",
                "text": w.tr("&Open Caption"),
                "slot": w.openCaption,
                "tip": w.tr("Open Caption"),
            },
            {
                "name": "open_florence2",
                "text": w.tr("Florence-&2 Assistant"),
                "slot": w.openFlorence2,
                "tip": w.tr(
                    "Florence-2 captioning and segmentation panel for the current project"
                ),
                "icon_name": "objects",
            },
            {
                "name": "open_image_editing",
                "text": w.tr("Image &Editing…"),
                "slot": w.openImageEditing,
                "tip": w.tr(
                    "Generate/edit images with Diffusers or stable-diffusion.cpp (supports Qwen-Image GGUF presets)"
                ),
                "icon_name": "objects",
            },
            {
                "name": "downsample_video",
                "text": w.tr("&Downsample Videos"),
                "slot": w.downsample_videos,
                "tip": w.tr("Downsample Videos"),
            },
            {
                "name": "run_optical_flow",
                "text": w.tr("Run &Optical Flow..."),
                "slot": w.run_optical_flow_tool,
                "tip": w.tr(
                    "Run optical flow with saved settings, preview on canvas, and optionally export stats"
                ),
            },
            {
                "name": "tracking_reports",
                "text": w.tr("&Tracking Reports"),
                "slot": w.trigger_gap_analysis,
                "tip": w.tr("Generate tracking reports for the selected video"),
            },
            {
                "name": "behavior_time_budget",
                "text": w.tr("&Behavior Time Budget"),
                "slot": w.show_behavior_time_budget_dialog,
                "tip": w.tr("Summarise recorded behavior events"),
            },
            {
                "name": "run_agent",
                "text": w.tr("Run &Agent Analysis…"),
                "slot": w.open_agent_run_dialog,
                "tip": w.tr("Configure and launch the unified agent pipeline"),
                "icon_name": "visualization",
            },
            # New streamlined wizard actions
            {
                "name": "new_project_wizard",
                "text": w.tr("&New Project…"),
                "slot": w.open_new_project_wizard,
                "shortcut": "Ctrl+Shift+N",
                "tip": w.tr("Create a new annotation project with guided wizard"),
            },
            {
                "name": "export_dataset_wizard",
                "text": w.tr("&Export Dataset…"),
                "slot": w.open_export_dataset_wizard,
                "tip": w.tr("Export annotations to COCO, YOLO, or JSONL format"),
            },
            {
                "name": "training_wizard",
                "text": w.tr("Training &Wizard…"),
                "slot": w.open_training_wizard,
                "tip": w.tr("Step-by-step training configuration wizard"),
            },
            {
                "name": "inference_wizard",
                "text": w.tr("&Inference Wizard…"),
                "slot": w.open_inference_wizard,
                "tip": w.tr("Run inference on videos with trained models"),
            },
            {
                "name": "project_schema",
                "text": w.tr("Project &Schema"),
                "slot": w.open_project_schema_dialog,
                "tip": w.tr("Edit categories, modifiers, and behaviors"),
            },
            {
                "name": "convert_csv",
                "text": w.tr("&Save CSV"),
                "slot": w.convert_labelme_json_to_csv,
                "tip": w.tr("Save CSV"),
            },
            {
                "name": "extract_shape_keypoints",
                "text": w.tr("&Extract Shape Keypoints"),
                "slot": w.extract_and_save_shape_keypoints,
                "tip": w.tr("Extract Shape Keypoints"),
            },
            {
                "name": "convert_sleap",
                "text": w.tr("&Convert SLEAP h5 to labelme"),
                "slot": w.convert_sleap_h5_to_labelme,
                "tip": w.tr("Convert SLEAP h5 to labelme"),
            },
            {
                "name": "convert_deeplabcut",
                "text": w.tr("&Convert DeepLabCut CSV to labelme"),
                "slot": w.convert_deeplabcut_csv_to_labelme,
                "tip": w.tr("Convert DeepLabCut CSV to labelme"),
            },
            {
                "name": "convert_labelme2yolo_format",
                "text": w.tr("&Convert Labelme to YOLO format"),
                "slot": w.convert_labelme2yolo_format,
                "tip": w.tr("Convert Labelme to YOLO format"),
            },
            {
                "name": "pose_schema",
                "text": w.tr("Pose &Schema (Keypoints)"),
                "slot": w.open_pose_schema_dialog,
                "tip": w.tr("Define keypoint order, symmetry, and edges"),
            },
            {
                "name": "place_preference",
                "text": w.tr("&Place Preference"),
                "slot": w.place_preference_analyze,
                "tip": w.tr("Place Preference"),
            },
            {
                "name": "about_annolid",
                "text": w.tr("&About Annolid"),
                "slot": w.about_annolid_and_system_info,
                "tip": w.tr("About Annolid"),
            },
            {
                "name": "coco",
                "text": w.tr("&COCO format"),
                "slot": w.coco,
                "shortcut": "Ctrl+C+O",
                "tip": w.tr("Convert to COCO format"),
                "icon_path": here / "icons/coco.png",
            },
            {
                "name": "save_labels",
                "text": w.tr("&Save labels"),
                "slot": w.save_labels,
                "shortcut": "Ctrl+Shift+L",
                "tip": w.tr("Save labels to txt file"),
                "icon_path": here / "icons/label_list.png",
            },
            {
                "name": "frames",
                "text": w.tr("&Extract frames"),
                "slot": w.frames,
                "shortcut": "Ctrl+Shift+E",
                "tip": w.tr("Extract frames from a video"),
                "icon_path": here / "icons/extract_frames.png",
            },
            {
                "name": "models",
                "text": w.tr("&Train models"),
                "slot": w.models,
                "shortcut": "Ctrl+Shift+T",
                "tip": w.tr("Train neural networks"),
                "icon_path": here / "icons/models.png",
            },
            {
                "name": "tracks",
                "text": w.tr("&Track Animals"),
                "slot": w.tracks,
                "shortcut": "Ctrl+Shift+O",
                "tip": w.tr("Track animals and Objects"),
                "icon_path": here / "icons/track.png",
            },
            {
                "name": "glitter2",
                "text": w.tr("&Glitter2"),
                "slot": w.glitter2,
                "shortcut": "Ctrl+Shift+G",
                "tip": w.tr("Convert to Glitter2 nix format"),
                "icon_path": here / "icons/glitter2_logo.png",
            },
            {
                "name": "quality_control",
                "text": w.tr("&Quality Control"),
                "slot": w.quality_control,
                "shortcut": "Ctrl+Shift+Q",
                "tip": w.tr("Convert to tracking results to labelme format"),
                "icon_path": here / "icons/quality_control.png",
            },
            {
                "name": "visualization",
                "text": w.tr("&Visualization"),
                "slot": w.visualization,
                "shortcut": "Ctrl+Shift+V",
                "tip": w.tr("Visualization results"),
                "icon_path": here / "icons/visualization.png",
            },
            {
                "name": "toggle_pose_edges",
                "text": w.tr("Show Pose &Edges"),
                "slot": w.toggle_pose_edges_display,
                "tip": w.tr("Show/hide pose skeleton edges on the canvas"),
                "checkable": True,
                "checked": bool(getattr(w, "_show_pose_edges", False)),
            },
            {
                "name": "toggle_pose_bbox_display",
                "text": w.tr("Show Pose &BBoxes"),
                "slot": w.toggle_pose_bbox_display,
                "tip": w.tr("Show/hide pose bounding boxes on the canvas"),
                "checkable": True,
                "checked": bool(getattr(w, "_show_pose_bboxes", True)),
            },
            {
                "name": "toggle_pose_bbox_save",
                "text": w.tr("Save Pose &BBoxes"),
                "slot": w.toggle_pose_bbox_saving,
                "tip": w.tr("Save pose bounding boxes for YOLO pose inference"),
                "checkable": True,
                "checked": bool(getattr(w, "_save_pose_bbox", True)),
            },
            {
                "name": "video_depth_anything",
                "text": w.tr("Video Depth Anything..."),
                "slot": w.run_video_depth_anything,
                "tip": w.tr("Estimate depth for a video with Video-Depth-Anything"),
                "icon_name": "visualization",
            },
            {
                "name": "sam3d_reconstruct",
                "text": w.tr("Reconstruct 3D (SAM 3D)..."),
                "slot": w.run_sam3d_reconstruction,
                "tip": w.tr("Generate a 3D Gaussian splat from video"),
                "icon_name": "objects",
            },
            {
                "name": "optical_flow_settings",
                "text": w.tr("Optical Flow Settings..."),
                "slot": w.configure_optical_flow_settings,
                "tip": w.tr("Configure optical flow backend, overlays, and outputs"),
            },
            {
                "name": "depth_settings",
                "text": w.tr("Depth Settings..."),
                "slot": w.configure_video_depth_settings,
                "tip": w.tr("Configure Video-Depth-Anything defaults"),
            },
            {
                "name": "sam3d_settings",
                "text": w.tr("SAM 3D Settings..."),
                "slot": w.configure_sam3d_settings,
                "tip": w.tr("Configure SAM 3D repository, checkpoints, and Python env"),
            },
            {
                "name": "colab",
                "text": w.tr("&Open in Colab"),
                "slot": w.train_on_colab,
                "tip": w.tr("Open in Colab"),
                "icon_path": here / "icons/colab.png",
            },
            {
                "name": "add_stamps_action",
                "text": w.tr("Add Real-Time Stamps…"),
                "slot": w._add_real_time_stamps,
                "tip": w.tr("Populate CSVs with true frame timestamps"),
                "icon_name": "timestamp",
            },
        ]

        for spec in simple_specs:
            action = self._create_action_from_spec(spec)
            registry[spec["name"]] = action

        w.shortcuts = w._config["shortcuts"]

        w.aiRectangle = AiRectangleWidget()
        w.aiRectangle._aiRectanglePrompt.returnPressed.connect(w._grounding_sam)
        w.recording_widget = RecordingWidget(lambda: w.canvas)

        w.patch_similarity_action = self._action_factory(
            w.tr("Patch Similarity"),
            w._toggle_patch_similarity_tool,
            icon="visualization",
            tip=w.tr("Click on the frame to generate a DINO patch similarity heatmap"),
        )
        w.patch_similarity_action.setCheckable(True)
        w.patch_similarity_action.setIcon(
            QtGui.QIcon(str(here / "icons/visualization.png"))
        )

        w.patch_similarity_settings_action = self._action_factory(
            w.tr("Patch Similarity Settings…"),
            w._open_patch_similarity_settings,
            tip=w.tr("Choose model and overlay opacity for patch similarity"),
        )

        w.pca_map_action = self._action_factory(
            w.tr("PCA Feature Map"),
            w._toggle_pca_map_tool,
            icon="visualization",
            tip=w.tr(
                "Toggle a PCA-colored DINO feature map overlay for the current frame"
            ),
        )
        w.pca_map_action.setCheckable(True)
        w.pca_map_action.setIcon(QtGui.QIcon(str(here / "icons/visualization.png")))

        w.pca_map_settings_action = self._action_factory(
            w.tr("PCA Feature Map Settings…"),
            w._open_pca_map_settings,
            tip=w.tr("Choose model and overlay opacity for the PCA map"),
        )

        w.realtime_control_action = self._action_factory(
            w.tr("Realtime Control…"),
            w._show_realtime_control_dialog,
            icon="fast_forward",
            tip=w.tr("Configure and launch realtime inference"),
        )
        w.realtime_control_action.setIcon(
            QtGui.QIcon(str(here / "icons/fast_forward.png"))
        )

        registry["create_ai_polygon_mode"] = create_ai_polygon_mode

        # 3D Viewer action
        w.open_3d_viewer_action = self._action_factory(
            w.tr("3D Viewer…"),
            w.open_3d_viewer,
            tip=w.tr("View and explore 3D TIFF image stacks"),
        )

        # ------------------------------------------------------------------
        # Pose/keypoint annotation helpers (visible vs occluded)
        # ------------------------------------------------------------------
        w.toggle_keypoint_visibility_action = self._action_factory(
            w.tr("Toggle Keypoint Visibility"),
            w.toggle_selected_keypoint_visibility,
            shortcut="Ctrl+Alt+I",
            tip=w.tr("Toggle selected keypoint(s) between visible and occluded"),
        )
        w.mark_keypoint_visible_action = self._action_factory(
            w.tr("Mark Keypoint Visible"),
            lambda: w.set_selected_keypoint_visibility(True),
            shortcut="Ctrl+Alt+V",
            tip=w.tr("Mark selected keypoint(s) as visible (v=2)"),
        )
        w.mark_keypoint_occluded_action = self._action_factory(
            w.tr("Mark Keypoint Occluded"),
            lambda: w.set_selected_keypoint_visibility(False),
            shortcut="Ctrl+Alt+O",
            tip=w.tr("Mark selected keypoint(s) as occluded (v=1)"),
        )

        try:
            w.actions.editMenu = tuple(getattr(w.actions, "editMenu", ())) + (
                None,
                w.toggle_keypoint_visibility_action,
                w.mark_keypoint_visible_action,
                w.mark_keypoint_occluded_action,
            )
        except Exception:
            pass

        try:
            w.actions.menu = tuple(getattr(w.actions, "menu", ())) + (
                None,
                w.toggle_keypoint_visibility_action,
                w.mark_keypoint_visible_action,
                w.mark_keypoint_occluded_action,
            )
        except Exception:
            pass

        try:
            label_list_menu = getattr(w.menus, "labelList", None)
            if label_list_menu is not None:
                label_list_menu.addSeparator()
                label_list_menu.addAction(w.toggle_keypoint_visibility_action)
                label_list_menu.addAction(w.mark_keypoint_visible_action)
                label_list_menu.addAction(w.mark_keypoint_occluded_action)
        except Exception:
            pass

    def _create_action_from_spec(self, spec: dict) -> QtWidgets.QAction:
        action = self._action_factory(
            spec["text"],
            spec["slot"],
            spec.get("shortcut"),
            spec.get("icon_name"),
            spec.get("tip"),
            checkable=spec.get("checkable", False),
            enabled=spec.get("enabled", True),
            checked=spec.get("checked", False),
        )
        icon_path = spec.get("icon_path")
        if icon_path:
            action.setIcon(QtGui.QIcon(str(icon_path)))
        action.setIconText(self._format_tool_button_text(action.text()))
        return action

    @staticmethod
    def _format_tool_button_text(text: str) -> str:
        base = text.replace("&", "").strip()
        if not base or "\n" in base:
            return base
        parts = base.split()
        if len(parts) >= 2:
            first, rest = parts[0], " ".join(parts[1:])
            return f"{first}\n{rest}"
        return base

    def _populate_tools_and_menus(self) -> None:
        w = self._window
        actions = self._actions

        tool_actions = list(w.actions.tool)
        tool_actions.insert(0, actions["frames"])
        tool_actions.insert(1, actions["open_video"])
        tool_actions.insert(2, actions["step_size"])
        tool_actions.append(w.aiRectangle.aiRectangleAction)
        tool_actions.append(actions["tracks"])
        tool_actions.append(actions["glitter2"])
        tool_actions.append(actions["coco"])
        tool_actions.append(actions["models"])
        tool_actions.append(w.createPolygonSAMMode)
        tool_actions.append(actions["save_labels"])
        tool_actions.append(actions["quality_control"])
        tool_actions.append(actions["colab"])
        tool_actions.append(actions["video_depth_anything"])
        tool_actions.append(actions["sam3d_reconstruct"])
        tool_actions.append(actions["visualization"])
        tool_actions.append(actions["open_florence2"])
        tool_actions.append(actions["open_image_editing"])
        tool_actions.append(w.patch_similarity_action)
        tool_actions.append(w.pca_map_action)
        tool_actions.append(w.recording_widget.record_action)
        w.actions.tool = tuple(tool_actions)

        w.tools.clear()
        w.tools.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        w.tools.setIconSize(QtCore.QSize(32, 32))
        utils.addActions(w.tools, w.actions.tool)
        for action in w.actions.tool:
            button = w.tools.widgetForAction(action)
            if isinstance(button, QtWidgets.QToolButton):
                button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
                button.setIconSize(QtCore.QSize(32, 32))
                text = self._format_tool_button_text(action.text())
                button.setText(text)

        # ============================================================
        # FILE MENU - Open, Save, Export operations only
        # ============================================================
        file_sections = [
            (
                actions["new_project_wizard"],
                actions["open_video"],
                actions["open_youtube_video"],
            ),
            (
                actions["open_audio"],
                actions["open_caption"],
            ),
            (
                actions["save_labels"],
                actions["export_dataset_wizard"],
            ),
        ]
        self._add_menu_sections(w.menus.file, file_sections)

        # ============================================================
        # VIDEO TOOLS MENU - Video processing and manipulation
        # ============================================================
        video_sections = [
            (
                w.open_segment_editor_action,
                actions["frames"],
                actions["downsample_video"],
            ),
            (
                actions["run_optical_flow"],
                actions["video_depth_anything"],
                actions["add_stamps_action"],
            ),
        ]
        self._add_menu_sections(w.menus.video_tools, video_sections)

        # ============================================================
        # AI & MODELS MENU - Machine learning hub
        # ============================================================
        ai_sections = [
            # Streamlined wizards first
            (
                actions["training_wizard"],
                actions["inference_wizard"],
            ),
            # Legacy training dialog
            (
                actions["models"],
                actions["colab"],
            ),
            # AI-assisted annotation tools
            (
                w.createPolygonSAMMode,
                actions["open_florence2"],
                actions["open_image_editing"],
            ),
            # Tracking and quality
            (
                actions["tracks"],
                actions["quality_control"],
                w.realtime_control_action,
            ),
        ]
        self._add_menu_sections(w.menus.ai_models, ai_sections)

        # ============================================================
        # ANALYSIS MENU - Reports and visualization
        # ============================================================
        analysis_sections = [
            (
                actions["tracking_reports"],
                actions["behavior_time_budget"],
                actions["run_agent"],
                actions["place_preference"],
            ),
            (actions["visualization"],),
        ]
        self._add_menu_sections(w.menus.analysis, analysis_sections)

        # ============================================================
        # VIEW MENU - Display toggles and 3D visualization
        # ============================================================
        view_sections = [
            (
                actions["toggle_pose_edges"],
                actions["toggle_pose_bbox_display"],
                w.patch_similarity_action,
                w.pca_map_action,
            ),
            (
                w.open_3d_viewer_action,
                actions["sam3d_reconstruct"],
            ),
            (actions["glitter2"],),
        ]
        self._add_menu_sections(w.menus.view, view_sections)

        # ============================================================
        # CONVERT MENU - Format conversion utilities
        # ============================================================
        convert_sections = [
            (
                actions["convert_labelme2yolo_format"],
                actions["coco"],
            ),
            (
                actions["convert_deeplabcut"],
                actions["convert_sleap"],
            ),
            (
                actions["convert_csv"],
                actions["extract_shape_keypoints"],
            ),
        ]
        self._add_menu_sections(w.menus.convert, convert_sections)

        # ============================================================
        # SETTINGS MENU - Configuration dialogs
        # ============================================================
        settings_sections = [
            (
                actions["advance_params"],
                actions["project_schema"],
                actions["pose_schema"],
            ),
            (actions["toggle_pose_bbox_save"],),
            (
                actions["optical_flow_settings"],
                actions["depth_settings"],
                actions["sam3d_settings"],
            ),
            (
                w.patch_similarity_settings_action,
                w.pca_map_settings_action,
            ),
        ]

        # Themes submenu: allow switching between UI themes (stored in QSettings)
        try:
            # Default to empty (no theme) so users must opt-in to custom themes
            theme_setting = str(w.settings.value("ui/theme", "") or "")
            themes_menu = QtWidgets.QMenu(w.tr("&Themes"), w)
            theme_group = QtWidgets.QActionGroup(w)
            theme_group.setExclusive(True)

            # If empty, the user has not selected a custom theme (use system)
            system_checked = theme_setting == ""

            light_checked = theme_setting == "light"

            def _set_light(checked=False, _w=w):
                _w.settings.setValue("ui/theme", "light")
                try:
                    app = QtWidgets.QApplication.instance()
                    apply_light_theme(app)
                    refresh_app_styles(app)
                except Exception:
                    pass

            dark_checked = theme_setting == "dark"

            def _set_dark(checked=False, _w=w):
                _w.settings.setValue("ui/theme", "dark")
                try:
                    app = QtWidgets.QApplication.instance()
                    apply_dark_theme(app)
                    refresh_app_styles(app)
                except Exception:
                    pass

            modern_checked = theme_setting == "modern"

            def _set_modern(checked=False, _w=w):
                _w.settings.setValue("ui/theme", "modern")
                try:
                    app = QtWidgets.QApplication.instance()
                    apply_modern_theme(app)
                    refresh_app_styles(app)
                except Exception:
                    pass

            def _set_system(checked=False, _w=w):
                # Clear the theme setting so the app uses the system/default style
                _w.settings.setValue("ui/theme", "")
                try:
                    app = QtWidgets.QApplication.instance()
                    # Remove any custom stylesheet; keep the platform style
                    app.setStyleSheet("")
                    refresh_app_styles(app)
                except Exception:
                    pass

            light_action = self._action_factory(
                w.tr("Light"),
                _set_light,
                None,
                None,
                w.tr("Use a light (OS-like) theme for the application"),
                checkable=True,
                checked=light_checked,
            )
            dark_action = self._action_factory(
                w.tr("Dark"),
                _set_dark,
                None,
                None,
                w.tr("Use a dark (OS-like) theme for the application"),
                checkable=True,
                checked=dark_checked,
            )
            modern_action = self._action_factory(
                w.tr("Modern"),
                _set_modern,
                None,
                None,
                w.tr("Use the modern accent theme for the application"),
                checkable=True,
                checked=modern_checked,
            )

            # System/default (no custom theme) should be first
            system_action = self._action_factory(
                w.tr("System"),
                _set_system,
                None,
                None,
                w.tr("Use the system/default style (no custom theme)"),
                checkable=True,
                checked=system_checked,
            )

            theme_group.addAction(system_action)
            theme_group.addAction(light_action)
            theme_group.addAction(dark_action)
            theme_group.addAction(modern_action)

            themes_menu.addAction(system_action)
            themes_menu.addAction(light_action)
            themes_menu.addAction(dark_action)
            themes_menu.addAction(modern_action)

            # Insert the themes submenu as its own section in Settings
            settings_sections.insert(0, (themes_menu.menuAction(),))
        except Exception:
            pass
        self._add_menu_sections(w.menus.settings, settings_sections)

        # ============================================================
        # HELP MENU
        # ============================================================
        utils.addActions(w.menus.help, (actions["about_annolid"],))

    @staticmethod
    def _add_menu_sections(menu: QtWidgets.QMenu, sections) -> None:
        """Add grouped actions to a menu with separators between sections."""
        if menu is None:
            return
        total = len(sections)
        for idx, section in enumerate(sections):
            utils.addActions(menu, tuple(section))
            if idx < total - 1:
                menu.addSeparator()

    def _reorder_top_menus(self) -> None:
        """Keep top-level menu order consistent and professional."""
        w = self._window
        bar = w.menuBar()
        # Professional menu order: File, Edit, View, then domain menus, then Settings/Help
        desired_names = [
            "file",
            "edit",
            "view",
            "video_tools",
            "ai_models",
            "analysis",
            "convert",
            "settings",
            "help",
        ]
        menus = []
        for name in desired_names:
            menu = getattr(w.menus, name, None)
            if menu is not None:
                menus.append(menu)

        # Remove existing occurrences so we can re-add in the desired order.
        for action in list(bar.actions()):
            menu = action.menu()
            if menu is not None and menu in menus:
                bar.removeAction(action)

        # Re-add in the requested order.
        for menu in menus:
            bar.addMenu(menu)
