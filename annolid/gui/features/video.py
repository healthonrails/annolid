"""Video feature setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets.video_manager import VideoManagerWidget


@dataclass(frozen=True)
class VideoFeatureState:
    video_manager_widget: VideoManagerWidget
    video_dock: QtWidgets.QDockWidget


def setup_video_feature(deps: GuiFeatureDeps) -> VideoFeatureState:
    """Create the video manager widget and dock wiring."""
    window = deps.window
    window.video_manager_widget = VideoManagerWidget()
    window.video_manager_widget.video_selected.connect(window._load_video)
    window.video_manager_widget.close_video_requested.connect(window.closeFile)
    window.video_manager_widget.output_folder_ready.connect(
        window.handle_extracted_frames
    )
    window.video_manager_widget.json_saved.connect(
        window.video_manager_widget.update_json_column
    )
    window.video_manager_widget.track_all_worker_created.connect(
        window.tracking_controller.register_track_all_worker
    )

    window.video_dock = QtWidgets.QDockWidget("Video List", window)
    window.video_dock.setObjectName("videoListDock")
    window.video_dock.setWidget(window.video_manager_widget)
    window.video_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.addDockWidget(Qt.RightDockWidgetArea, window.video_dock)
    return VideoFeatureState(
        video_manager_widget=window.video_manager_widget,
        video_dock=window.video_dock,
    )
