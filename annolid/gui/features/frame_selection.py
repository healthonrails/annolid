"""Frame Selection / Label Queue dock feature setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets.frame_selection_queue_widget import (
    FrameSelectionQueueWidget,
)


@dataclass(frozen=True)
class FrameSelectionFeatureState:
    frame_selection_queue_widget: FrameSelectionQueueWidget
    label_queue_dock: QtWidgets.QDockWidget


def setup_frame_selection_feature(
    deps: GuiFeatureDeps,
) -> FrameSelectionFeatureState:
    """Create the interactive Label Queue dock and wire it to the main window."""
    window = deps.window
    widget = FrameSelectionQueueWidget(window)
    window.frame_selection_queue_widget = widget

    if hasattr(window, "_jump_to_frame_from_log"):
        widget.jumpToFrameRequested.connect(window._jump_to_frame_from_log)
    elif hasattr(window, "set_frame_number"):
        widget.jumpToFrameRequested.connect(window.set_frame_number)

    widget.statusMessage.connect(lambda msg: deps.status_message(msg, 4000))
    if hasattr(window, "frames"):
        widget.selectFramesRequested.connect(window.frames)
    if hasattr(window, "importDirImages"):
        widget.extractedDirectoryReady.connect(window.importDirImages)
    if hasattr(window, "_sync_label_queue_slider_marks"):
        widget.queueMarksChanged.connect(window._sync_label_queue_slider_marks)

    dock = QtWidgets.QDockWidget("Label Queue", window)
    dock.setObjectName("labelQueueDock")
    dock.setWidget(widget)
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.label_queue_dock = dock
    window.addDockWidget(Qt.RightDockWidgetArea, dock)
    try:
        video_dock = getattr(window, "video_dock", None)
        if isinstance(video_dock, QtWidgets.QDockWidget):
            window.tabifyDockWidget(video_dock, dock)
            video_dock.show()
            video_dock.raise_()
    except Exception:
        pass

    return FrameSelectionFeatureState(
        frame_selection_queue_widget=widget,
        label_queue_dock=dock,
    )


__all__ = ["FrameSelectionFeatureState", "setup_frame_selection_feature"]
