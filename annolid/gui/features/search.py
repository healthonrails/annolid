"""Search feature dock setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget


@dataclass(frozen=True)
class SearchFeatureState:
    embedding_search_widget: EmbeddingSearchWidget
    embedding_search_dock: QtWidgets.QDockWidget


def setup_search_feature(deps: GuiFeatureDeps) -> SearchFeatureState:
    """Create embedding search dock and wire callbacks."""
    window = deps.window
    window.embedding_search_widget = EmbeddingSearchWidget(window)
    window.embedding_search_widget.jumpToFrame.connect(window._jump_to_frame_from_log)
    window.embedding_search_widget.statusMessage.connect(
        lambda msg: deps.status_message(msg, 4000)
    )
    window.embedding_search_widget.labelFramesRequested.connect(
        window._label_frames_from_search
    )
    window.embedding_search_widget.markFramesRequested.connect(
        window._mark_similar_frames_from_search
    )
    window.embedding_search_widget.clearMarkedFramesRequested.connect(
        window._clear_similar_frame_marks
    )

    window.embedding_search_dock = QtWidgets.QDockWidget("Embedding Search", window)
    window.embedding_search_dock.setObjectName("embeddingSearchDock")
    window.embedding_search_dock.setWidget(window.embedding_search_widget)
    window.embedding_search_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.addDockWidget(Qt.RightDockWidgetArea, window.embedding_search_dock)
    try:
        window.tabifyDockWidget(window.video_dock, window.embedding_search_dock)
        window.video_dock.show()
        window.video_dock.raise_()
    except Exception:
        pass
    return SearchFeatureState(
        embedding_search_widget=window.embedding_search_widget,
        embedding_search_dock=window.embedding_search_dock,
    )
