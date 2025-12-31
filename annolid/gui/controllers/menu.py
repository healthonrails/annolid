import functools
from typing import Dict, TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets

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
        self._ensure_video_tools_menu()
        self._ensure_settings_menu()
        self._create_core_actions()
        self._populate_tools_and_menus()
        self._reorder_top_menus()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_video_tools_menu(self) -> None:
        w = self._window
        action = self._action_factory
        if not hasattr(w.menus, "video_tools"):
            w.menus.video_tools = w.menuBar().addMenu(w.tr("&Video Tools"))
        w.open_segment_editor_action = action(
            w.tr("Define Video Segments..."),
            w._open_segment_editor_dialog,
            shortcut="Ctrl+Alt+S",
            tip=w.tr("Define tracking segments for the current video"),
        )
        w.open_segment_editor_action.setEnabled(False)
        utils.addActions(w.menus.video_tools, (w.open_segment_editor_action,))

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
        w.stepSizeWidget.setWhatsThis(
            w.tr("Step for the next or prev image. e.g. 30")
        )
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
                "name": "downsample_video",
                "text": w.tr("&Downsample Videos"),
                "slot": w.downsample_videos,
                "tip": w.tr("Downsample Videos"),
            },
            {
                "name": "run_optical_flow",
                "text": w.tr("Run &Optical Flow..."),
                "slot": w.run_optical_flow_tool,
                "tip": w.tr("Run optical flow with saved settings, preview on canvas, and optionally export stats"),
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
                "tip": w.tr("Extract frames frome a video"),
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
                "shortcut": "Ctrl+Shift+G",
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
                "tip": w.tr("Generate a 3D Gaussian splat using SAM 3D Objects"),
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
        w.aiRectangle._aiRectanglePrompt.returnPressed.connect(
            w._grounding_sam)
        w.recording_widget = RecordingWidget(lambda: w.canvas)

        w.patch_similarity_action = self._action_factory(
            w.tr("Patch Similarity"),
            w._toggle_patch_similarity_tool,
            icon="visualization",
            tip=w.tr(
                "Click on the frame to generate a DINO patch similarity heatmap"
            ),
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
        w.pca_map_action.setIcon(
            QtGui.QIcon(str(here / "icons/visualization.png"))
        )

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
            w.tr("Open 3D Viewer…"),
            w.open_3d_viewer,
            tip=w.tr("Open a 3D viewer for TIFF stacks"),
        )

    def _ensure_settings_menu(self) -> None:
        """Create a Settings menu (positioned before View when available)."""
        w = self._window
        if hasattr(w.menus, "settings"):
            return
        settings_menu = QtWidgets.QMenu(w.tr("&Settings"), w)
        view_menu = getattr(w.menus, "view", None)
        if view_menu is not None:
            w.menuBar().insertMenu(view_menu.menuAction(), settings_menu)
        else:
            w.menuBar().addMenu(settings_menu)
        w.menus.settings = settings_menu

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
            first, rest = parts[0], ' '.join(parts[1:])
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
        file_sections = [
            (
                actions["open_video"],
                actions["open_youtube_video"],
                actions["open_audio"],
                actions["open_caption"],
                actions["open_florence2"],
                actions["colab"],
            ),
            (
                actions["save_labels"],
                actions["frames"],
                actions["models"],
                actions["tracks"],
                actions["quality_control"],
                actions["downsample_video"],
            ),
            (
                actions["tracking_reports"],
                actions["behavior_time_budget"],
                actions["project_schema"],
                actions["pose_schema"],
                actions["place_preference"],
                actions["add_stamps_action"],
            ),
            (
                actions["convert_csv"],
                actions["extract_shape_keypoints"],
                actions["convert_labelme2yolo_format"],
                actions["convert_deeplabcut"],
                actions["convert_sleap"],
                actions["coco"],
            ),
        ]
        self._add_menu_sections(w.menus.file, file_sections)

        view_sections = [
            (
                actions["glitter2"],
                actions["video_depth_anything"],
                actions["run_optical_flow"],
                actions["sam3d_reconstruct"],
            ),
            (
                actions["visualization"],
                actions["toggle_pose_edges"],
                w.patch_similarity_action,
                w.pca_map_action,
                w.open_3d_viewer_action,
            ),
            (w.realtime_control_action,),
        ]
        self._add_menu_sections(w.menus.view, view_sections)

        settings_actions = [
            actions["advance_params"],
            actions["optical_flow_settings"],
            actions["depth_settings"],
            actions["sam3d_settings"],
            w.patch_similarity_settings_action,
            w.pca_map_settings_action,
        ]
        self._add_menu_sections(w.menus.settings, [settings_actions])

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
        desired_names = ["file", "edit", "view",
                         "video_tools", "settings", "help"]
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
