"""Behavior log/controls/timeline dock setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets.behavior_controls import BehaviorControlsWidget
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget
from annolid.gui.widgets.timeline_panel import TimelinePanel


@dataclass(frozen=True)
class TimelineFeatureState:
    behavior_log_widget: BehaviorEventLogWidget
    behavior_log_dock: QtWidgets.QDockWidget
    behavior_controls_widget: BehaviorControlsWidget
    behavior_controls_dock: QtWidgets.QDockWidget
    timeline_panel: TimelinePanel
    timeline_dock: QtWidgets.QDockWidget


def setup_timeline_feature(deps: GuiFeatureDeps) -> TimelineFeatureState:
    """Create behavior-related docks and timeline panel wiring."""
    window = deps.window
    window.behavior_log_widget = BehaviorEventLogWidget(
        window, color_getter=window._get_rgb_by_label
    )
    window.behavior_log_widget.jumpToFrame.connect(window._jump_to_frame_from_log)
    window.behavior_log_widget.undoRequested.connect(window.undo_last_behavior_event)
    window.behavior_log_widget.clearRequested.connect(
        window._clear_behavior_events_from_log
    )
    window.behavior_log_widget.behaviorSelected.connect(
        window._show_behavior_event_details
    )
    window.behavior_log_widget.editRequested.connect(
        window._edit_behavior_event_from_log
    )
    window.behavior_log_widget.deleteRequested.connect(
        window._delete_behavior_event_from_log
    )
    window.behavior_log_widget.confirmRequested.connect(
        window._confirm_behavior_event_from_log
    )
    window.behavior_log_widget.rejectRequested.connect(
        window._reject_behavior_event_from_log
    )

    window.behavior_log_dock = QtWidgets.QDockWidget("Behavior Log", window)
    window.behavior_log_dock.setObjectName("behaviorLogDock")
    window.behavior_log_dock.setWidget(window.behavior_log_widget)
    window.behavior_log_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.addDockWidget(Qt.RightDockWidgetArea, window.behavior_log_dock)

    window.behavior_controls_widget = BehaviorControlsWidget(window)
    window.behavior_controls_widget.subjectChanged.connect(
        window._on_active_subject_changed
    )
    window.behavior_controls_widget.modifierToggled.connect(window._on_modifier_toggled)
    window.behavior_controls_dock = QtWidgets.QDockWidget("Behavior Controls", window)
    window.behavior_controls_dock.setObjectName("behaviorControlsDock")
    window.behavior_controls_dock.setWidget(window.behavior_controls_widget)
    window.behavior_controls_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.addDockWidget(Qt.RightDockWidgetArea, window.behavior_controls_dock)
    window.tabifyDockWidget(window.behavior_log_dock, window.behavior_controls_dock)
    window.behavior_log_dock.raise_()

    window.timeline_panel = TimelinePanel(window)
    window.timeline_panel.frameSelected.connect(window._jump_to_frame_from_log)
    window.timeline_panel.set_behavior_controller(
        window.behavior_controller, color_getter=window._get_rgb_by_label
    )
    window.timeline_panel.set_timestamp_provider(window._estimate_recording_time)
    window.timeline_panel.set_behavior_catalog(
        provider=window._timeline_behavior_catalog,
        adder=window._timeline_add_behavior,
    )
    try:
        window.flag_widget.flagsSaved.connect(
            window.timeline_panel.refresh_behavior_catalog
        )
        window.flag_widget.rowSelected.connect(
            window.timeline_panel.set_active_behavior
        )
        window.flag_widget.rowSelected.connect(
            lambda _name: window.timeline_panel.refresh_behavior_catalog()
        )
        window.flag_widget.flagToggled.connect(
            lambda _name, _state: window.timeline_panel.refresh_behavior_catalog()
        )
    except Exception:
        pass

    window.timeline_dock = QtWidgets.QDockWidget("Timeline", window)
    window.timeline_dock.setObjectName("timelineDock")
    window.timeline_dock.setWidget(window.timeline_panel)
    window.timeline_dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    window.addDockWidget(Qt.BottomDockWidgetArea, window.timeline_dock)
    window._setup_timeline_view_toggle()
    window._apply_timeline_dock_visibility(video_open=False)
    window._apply_fixed_dock_sizes()
    return TimelineFeatureState(
        behavior_log_widget=window.behavior_log_widget,
        behavior_log_dock=window.behavior_log_dock,
        behavior_controls_widget=window.behavior_controls_widget,
        behavior_controls_dock=window.behavior_controls_dock,
        timeline_panel=window.timeline_panel,
        timeline_dock=window.timeline_dock,
    )
