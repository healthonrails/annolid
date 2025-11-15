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
        self._create_core_actions()
        self._populate_tools_and_menus()

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
                "name": "video_depth_anything",
                "text": w.tr("Video Depth Anything..."),
                "slot": w.run_video_depth_anything,
                "tip": w.tr("Estimate depth for a video with Video-Depth-Anything"),
                "icon_name": "visualization",
            },
            {
                "name": "depth_settings",
                "text": w.tr("Depth Settings..."),
                "slot": w.configure_video_depth_settings,
                "tip": w.tr("Configure Video-Depth-Anything defaults"),
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
        tool_actions.append(actions["visualization"])
        tool_actions.append(actions["depth_settings"])
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
        utils.addActions(w.menus.file, (actions["open_video"],))
        utils.addActions(w.menus.file, (actions["open_youtube_video"],))
        utils.addActions(w.menus.file, (actions["open_audio"],))
        utils.addActions(w.menus.file, (actions["open_caption"],))
        utils.addActions(w.menus.file, (actions["open_florence2"],))
        utils.addActions(w.menus.file, (actions["colab"],))
        utils.addActions(w.menus.file, (actions["save_labels"],))
        utils.addActions(w.menus.file, (actions["coco"],))
        utils.addActions(w.menus.file, (actions["frames"],))
        utils.addActions(w.menus.file, (actions["models"],))
        utils.addActions(w.menus.file, (actions["tracks"],))
        utils.addActions(w.menus.file, (actions["quality_control"],))
        utils.addActions(w.menus.file, (actions["downsample_video"],))
        utils.addActions(w.menus.file, (actions["tracking_reports"],))
        utils.addActions(w.menus.file, (actions["behavior_time_budget"],))
        utils.addActions(w.menus.file, (actions["project_schema"],))
        w.menus.file.addSeparator()
        utils.addActions(w.menus.file, (actions["convert_csv"],))
        utils.addActions(w.menus.file, (actions["extract_shape_keypoints"],))
        utils.addActions(w.menus.file, (actions["convert_deeplabcut"],))
        utils.addActions(w.menus.file, (actions["convert_sleap"],))
        utils.addActions(
            w.menus.file, (actions["convert_labelme2yolo_format"],)
        )
        utils.addActions(w.menus.file, (actions["place_preference"],))
        utils.addActions(w.menus.file, (actions["add_stamps_action"],))
        utils.addActions(w.menus.file, (actions["advance_params"],))

        utils.addActions(w.menus.view, (actions["glitter2"],))
        utils.addActions(
            w.menus.view, (actions["video_depth_anything"],))
        utils.addActions(w.menus.view, (actions["depth_settings"],))
        utils.addActions(w.menus.view, (actions["visualization"],))
        utils.addActions(w.menus.view, (w.patch_similarity_action,))
        utils.addActions(w.menus.view, (w.pca_map_action,))
        utils.addActions(w.menus.view, (w.open_3d_viewer_action,))
        utils.addActions(w.menus.view, (w.realtime_control_action,))
        utils.addActions(
            w.menus.view,
            (w.patch_similarity_settings_action, w.pca_map_settings_action),
        )

        utils.addActions(w.menus.help, (actions["about_annolid"],))
