from __future__ import annotations

import os

from qtpy import QtWidgets

from annolid.gui.widgets.brain_3d_dock import Brain3DSessionDockWidget


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_brain_3d_session_dock_emits_region_state_request() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_summary(region_count=2, source_page_count=3, plane_count=5)
        dock.set_current_plane(2)
        dock.set_regions(
            [
                {
                    "region_id": "r1",
                    "label": "Region 1",
                    "state": "present",
                    "source": "model",
                    "points_count": 14,
                },
                {
                    "region_id": "r2",
                    "label": "Region 2",
                    "state": "hidden",
                    "source": "state",
                    "points_count": 0,
                },
            ]
        )
        captured = []
        dock.regionStateRequested.connect(
            lambda plane, region, state: captured.append((plane, region, state))
        )
        dock.region_list.setCurrentRow(1)
        dock.state_combo.setCurrentIndex(2)  # created
        dock.apply_state_button.click()
        assert captured == [(2, "r2", "created")]
    finally:
        dock.close()
