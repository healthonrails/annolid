import functools
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.theme import (
    apply_modern_theme,
    apply_light_theme,
    apply_dark_theme,
    refresh_app_styles,
)

from annolid.gui.window_base import utils
from annolid.gui.window_base import newAction, format_tool_button_text

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
        w.createPolygonSAMMode.setIcon(QtGui.QIcon(str(here / "icons/ai_polygons.svg")))

        # Configure in-tree actions from AnnolidWindowBase (do not replace them).
        # We wire draw-mode actions in AnnolidWindow after the canvas exists.
        try:
            w.actions.createAiPolygonMode.setIcon(QtGui.QIcon.fromTheme("objects"))
            w.actions.createAiPolygonMode.setToolTip(
                w.tr("Start drawing AI polygons. Ctrl+Click ends creation.")
            )
        except Exception:
            pass
        try:
            w.actions.createAiMaskMode.setIcon(QtGui.QIcon.fromTheme("objects"))
            w.actions.createAiMaskMode.setToolTip(
                w.tr("Start drawing AI masks. Ctrl+Click ends creation.")
            )
        except Exception:
            pass

        w.createGroundingSAMMode = self._action_factory(
            w.tr("Grounding SAM"),
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

        zoom_widget_action = QtWidgets.QWidgetAction(w)
        zoom_container = QtWidgets.QWidget(w)
        zoom_layout = QtWidgets.QVBoxLayout(zoom_container)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(1)
        zoom_label = QtWidgets.QLabel(w.tr("Zoom"), zoom_container)
        zoom_label.setAlignment(QtCore.Qt.AlignCenter)
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(w.zoomWidget)
        zoom_widget_action.setDefaultWidget(zoom_container)
        registry["zoom_widget"] = zoom_widget_action

        ai_model_action = QtWidgets.QWidgetAction(w)
        model_container = QtWidgets.QWidget(w)
        model_layout = QtWidgets.QVBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(1)
        model_label = QtWidgets.QLabel(w.tr("AI Model"), model_container)
        model_label.setAlignment(QtCore.Qt.AlignCenter)
        model_layout.addWidget(model_label)
        w._selectAiModelComboBox.setMinimumWidth(180)
        model_layout.addWidget(w._selectAiModelComboBox)
        ai_model_action.setDefaultWidget(model_container)
        registry["ai_model_widget"] = ai_model_action

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
                "name": "open_3d",
                "text": w.tr("Open &3D File…"),
                "slot": self._open_3d_file,
                "tip": w.tr(
                    "Open a 3D model file (STL, OBJ, PLY, CSV, XYZ) in Three.js viewer. OBJ files with MTL material files are supported."
                ),
                "icon_path": here / "icons/visualization.png",
            },
            {
                "name": "open_caption",
                "text": w.tr("Open &Caption"),
                "slot": w.openCaption,
                "tip": w.tr(
                    "Open caption widget for adding text captions to images/videos"
                ),
            },
            {
                "name": "open_florence2",
                "text": w.tr("Florence-&2 Assistant"),
                "slot": w.openFlorence2,
                "tip": w.tr(
                    "Florence-2 captioning and segmentation panel for the current project"
                ),
                "icon_path": here / "icons/florence2.svg",
            },
            {
                "name": "open_image_editing",
                "text": w.tr("Image &Editing…"),
                "slot": w.openImageEditing,
                "tip": w.tr(
                    "Generate/edit images with Diffusers or stable-diffusion.cpp (supports Qwen-Image GGUF presets)"
                ),
                "icon_path": here / "icons/image_editing.svg",
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
                "text": w.tr("&Convert SLEAP to labelme"),
                "slot": w.convert_sleap_h5_to_labelme,
                "tip": w.tr("Convert SLEAP to labelme"),
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
                "name": "toggle_agent_mode",
                "text": w.tr("&Agent Mode"),
                "slot": w.toggle_agent_mode,
                "tip": w.tr("Show or hide agent-powered workflow tools"),
                "checkable": True,
                "checked": bool(getattr(w, "_agent_mode_enabled", True)),
            },
            {
                "name": "toggle_embedding_search",
                "text": w.tr("Frame Search (&Embedding)"),
                "slot": w.toggle_embedding_search,
                "tip": w.tr("Show/hide frame search (embedding) dock"),
                "checkable": True,
                "checked": bool(getattr(w, "_show_embedding_search", False)),
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
                "icon_path": here / "icons/depth_anything.svg",
            },
            {
                "name": "sam3d_reconstruct",
                "text": w.tr("Reconstruct 3D (SAM 3D)..."),
                "slot": w.run_sam3d_reconstruction,
                "tip": w.tr("Generate a 3D Gaussian splat from video"),
                "icon_path": here / "icons/reconstruct_3d.svg",
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

        # Keep a registry entry for backwards compatibility.
        try:
            registry["create_ai_polygon_mode"] = w.actions.createAiPolygonMode
        except Exception:
            pass

        # 3D Viewer action
        w.open_3d_viewer_action = self._action_factory(
            w.tr("3D Viewer…"),
            w.open_3d_viewer,
            tip=w.tr("View and explore 3D TIFF image stacks"),
        )
        w.close_3d_viewer_action = self._action_factory(
            w.tr("Close 3D View"),
            self._close_3d_viewer,
            tip=w.tr("Close the Three.js 3D view and return to canvas"),
        )
        w.close_pdf_action = self._action_factory(
            w.tr("Close PDF View"),
            self._close_pdf_view,
            tip=w.tr("Close the PDF view and return to canvas"),
        )
        w.threejs_example_helix_action = self._action_factory(
            w.tr("Helix Point Cloud"),
            lambda: w.open_threejs_example("helix_points_csv"),
            tip=w.tr("Open a generated helix point cloud example in Three.js"),
        )
        w.threejs_example_wave_action = self._action_factory(
            w.tr("Wave Surface Mesh"),
            lambda: w.open_threejs_example("wave_surface_obj"),
            tip=w.tr("Open a generated wave mesh example in Three.js"),
        )
        w.threejs_example_sphere_action = self._action_factory(
            w.tr("Sphere Point Cloud"),
            lambda: w.open_threejs_example("sphere_points_ply"),
            tip=w.tr("Open a generated sphere point cloud example in Three.js"),
        )
        w.threejs_example_brain_viewer_action = self._action_factory(
            w.tr("Brain 3D Viewer (Web)"),
            lambda: w.open_threejs_example("brain_viewer_html"),
            tip=w.tr("Open the standalone Brain 3D point cloud viewer in a browser"),
        )
        w.threejs_examples_menu = QtWidgets.QMenu(w.tr("3D Examples"), w)
        w.threejs_examples_menu.addAction(w.threejs_example_helix_action)
        w.threejs_examples_menu.addAction(w.threejs_example_wave_action)
        w.threejs_examples_menu.addAction(w.threejs_example_sphere_action)
        w.threejs_examples_menu.addAction(w.threejs_example_brain_viewer_action)

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
        return format_tool_button_text(text)

    def _populate_tools_and_menus(self) -> None:
        w = self._window
        actions = self._actions

        tool_actions = [
            actions["frames"],
            actions["open_video"],
            actions["step_size"],
            w.actions.open,
            w.actions.openDir,
            w.actions.openPrevImg,
            w.actions.openNextImg,
            w.actions.save,
            w.actions.deleteFile,
            w.actions.createMode,
            w.actions.editMode,
            w.actions.duplicateShapes,
            w.actions.deleteShapes,
            w.actions.undo,
            w.actions.brightnessContrast,
            w.actions.fitWindow,
            w.actions.zoomOut,
            w.actions.zoomIn,
            actions["ai_model_widget"],
            w.aiRectangle.aiRectangleAction,
            actions["tracks"],
            actions["glitter2"],
            actions["coco"],
            actions["models"],
            w.createPolygonSAMMode,
            None,
            actions["colab"],
            actions["video_depth_anything"],
            actions["sam3d_reconstruct"],
            actions["visualization"],
            actions["open_florence2"],
            actions["open_image_editing"],
            w.patch_similarity_action,
            w.pca_map_action,
            w.recording_widget.record_action,
        ]
        w.actions.tool = tuple(tool_actions)

        w.tools.clear()
        w.tools.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        w.tools.setIconSize(QtCore.QSize(32, 32))
        # Use AnnolidToolButton widgets to guarantee multi-line labels (macOS styles
        # often force toolbar labels to single-line otherwise).
        for item in w.actions.tool:
            if item is None:
                w.tools.addSeparator()
                continue

            # QWidgetAction: embeds its own widget (combo boxes, prompt panel, etc).
            if isinstance(item, QtWidgets.QWidgetAction):
                w.tools.addAction(item)
                continue

            # QMenu objects (rare in the main toolbar).
            if hasattr(item, "menuAction") and not isinstance(item, QtWidgets.QAction):
                w.tools.addAction(item.menuAction())
                continue

            if isinstance(item, QtWidgets.QAction):
                formatted = self._format_tool_button_text(item.text())
                try:
                    item.setIconText(formatted)
                except Exception:
                    pass
                w.tools.add_stacked_action(
                    item,
                    formatted,
                    width=58,
                    min_height=68,
                    icon_size=QtCore.QSize(32, 32),
                )
                continue

            # Fallback: keep existing behavior.
            w.tools.addAction(item)

        # ============================================================
        # FILE MENU - Open, Save, Export operations only
        # ============================================================
        file_sections = [
            (
                actions["new_project_wizard"],
                w.actions.open,
                w.actions.openDir,
                w.actions.close,
                actions["open_video"],
                actions["open_youtube_video"],
                actions["open_3d"],
                actions["open_caption"],
            ),
            (actions["open_audio"],),
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
                actions["toggle_agent_mode"],
                actions["toggle_embedding_search"],
                w.patch_similarity_action,
                w.pca_map_action,
            ),
            (
                w.open_3d_viewer_action,
                w.close_3d_viewer_action,
                w.close_pdf_action,
                actions["sam3d_reconstruct"],
                w.threejs_examples_menu.menuAction(),
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

    def _close_3d_viewer(self) -> None:
        """Close the Three.js 3D view and return to canvas."""
        if (
            hasattr(self._window, "threejs_manager")
            and self._window.threejs_manager is not None
        ):
            self._window.threejs_manager.close_threejs()

    def _close_pdf_view(self) -> None:
        """Close the PDF view and return to canvas."""
        if (
            hasattr(self._window, "pdf_manager")
            and self._window.pdf_manager is not None
        ):
            self._window.pdf_manager.close_pdf()

    def _open_3d_file(self) -> None:
        """Open a 3D model file dialog and load it in the Three.js viewer."""
        if (
            not hasattr(self._window, "threejs_manager")
            or self._window.threejs_manager is None
        ):
            QtWidgets.QMessageBox.warning(
                self._window,
                self._window.tr("3D Viewer Not Available"),
                self._window.tr(
                    "The Three.js 3D viewer is not available in this session."
                ),
            )
            return

        start_dir = getattr(self._window, "lastOpenDir", str(Path.home()))
        filters = self._window.tr(
            "3D Models (*.stl *.obj *.ply *.csv *.xyz);;OBJ with Materials (*.obj *.mtl);;All Files (*)"
        )
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._window,
            self._window.tr("Open 3D Model"),
            start_dir,
            filters,
        )
        if not filename:
            return

        # Update last open directory
        self._window.lastOpenDir = str(Path(filename).parent)

        # Load the 3D model
        success = self._window.threejs_manager.show_model_in_viewer(filename)
        if not success:
            QtWidgets.QMessageBox.warning(
                self._window,
                self._window.tr("Load Error"),
                self._window.tr("Failed to load the 3D model: %1").replace(
                    "%1", Path(filename).name
                ),
            )
