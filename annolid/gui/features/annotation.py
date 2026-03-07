"""Annotation-related dock setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.controllers import FlagsController
from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets import FlagTableWidget
from annolid.gui.widgets.keypoint_sequencer import KeypointSequencerWidget


@dataclass(frozen=True)
class AnnotationFeatureState:
    keypoint_sequence_widget: KeypointSequencerWidget
    keypoint_sequence_dock: QtWidgets.QDockWidget
    flag_widget: FlagTableWidget
    flags_controller: FlagsController


def setup_annotation_feature(deps: GuiFeatureDeps) -> AnnotationFeatureState:
    """Create keypoint and flag docks and wire annotation controls."""
    window = deps.window
    window.keypoint_sequence_widget = KeypointSequencerWidget(window)
    window.keypoint_sequence_dock = QtWidgets.QDockWidget(
        window.tr("Keypoint Sequencer"), window
    )
    window.keypoint_sequence_dock.setObjectName("keypointSequencerDock")
    window.keypoint_sequence_dock.setWidget(window.keypoint_sequence_widget)
    window.keypoint_sequence_widget.poseSchemaChanged.connect(
        window._on_keypoint_sequence_schema_changed
    )
    window.addDockWidget(Qt.RightDockWidgetArea, window.keypoint_sequence_dock)
    window.tabifyDockWidget(window.shape_dock, window.keypoint_sequence_dock)
    window.keypoint_sequence_dock.setVisible(False)
    window._setup_keypoint_sequence_quick_toggle()
    window._setup_keypoint_sequence_label_sync()

    window.flag_widget = FlagTableWidget()
    window.flag_dock.setWidget(window.flag_widget)
    window.flags_controller = FlagsController(
        window=window,
        widget=window.flag_widget,
        config_path=window.here.parent.resolve() / "configs" / "behaviors.yaml",
    )
    window.flags_controller.initialize()
    window.flag_dock.setVisible(True)
    window.flag_dock.raise_()
    return AnnotationFeatureState(
        keypoint_sequence_widget=window.keypoint_sequence_widget,
        keypoint_sequence_dock=window.keypoint_sequence_dock,
        flag_widget=window.flag_widget,
        flags_controller=window.flags_controller,
    )
