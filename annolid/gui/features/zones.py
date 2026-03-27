"""Zone authoring dock setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
from annolid.gui.widgets.zone_dock import ZoneDockWidget


@dataclass(frozen=True)
class ZoneFeatureState:
    zone_dock_widget: ZoneDockWidget
    zone_dock: QtWidgets.QDockWidget


def setup_zone_feature(deps: GuiFeatureDeps) -> ZoneFeatureState:
    """Create the zone authoring dock and wire it to the live canvas."""
    window = deps.window
    window.zone_dock_widget = ZoneDockWidget(window)
    window.zone_dock = window.zone_dock_widget
    window._zone_authoring_panel = window.zone_dock.zone_panel
    window.addDockWidget(Qt.RightDockWidgetArea, window.zone_dock)
    try:
        window.tabifyDockWidget(window.shape_dock, window.zone_dock)
    except Exception:
        pass
    window.zone_dock.setVisible(False)
    return ZoneFeatureState(
        zone_dock_widget=window.zone_dock_widget,
        zone_dock=window.zone_dock,
    )
