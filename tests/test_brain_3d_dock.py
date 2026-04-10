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


def test_brain_3d_session_dock_emits_open_preview_request() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_summary(region_count=1, source_page_count=1, plane_count=1)
        captured = []
        dock.openPreviewRequested.connect(lambda: captured.append(True))
        dock.open_preview_button.click()
        assert captured == [True]
    finally:
        dock.close()


def test_brain_3d_session_dock_emits_region_selection_change() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_summary(region_count=2, source_page_count=2, plane_count=2)
        dock.set_regions(
            [
                {
                    "region_id": "r1",
                    "label": "Region 1",
                    "state": "present",
                    "source": "model",
                    "points_count": 10,
                },
                {
                    "region_id": "r2",
                    "label": "Region 2",
                    "state": "present",
                    "source": "model",
                    "points_count": 12,
                },
            ]
        )
        captured = []
        dock.regionSelectionChanged.connect(lambda value: captured.append(value))
        dock.region_list.setCurrentRow(1)
        assert captured[-1] == "r2"
    finally:
        dock.close()


def test_brain_3d_session_dock_select_region_without_emitting_signal() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_summary(region_count=2, source_page_count=2, plane_count=2)
        dock.set_regions(
            [
                {
                    "region_id": "r1",
                    "label": "Region 1",
                    "state": "present",
                    "source": "model",
                    "points_count": 10,
                },
                {
                    "region_id": "r2",
                    "label": "Region 2",
                    "state": "present",
                    "source": "model",
                    "points_count": 12,
                },
            ]
        )
        captured = []
        dock.regionSelectionChanged.connect(lambda value: captured.append(value))
        ok = dock.select_region("r2", emit_signal=False)
        assert ok is True
        assert dock.selected_region_id() == "r2"
        assert captured == []
    finally:
        dock.close()


def test_brain_3d_session_dock_emits_highlight_mode_changed() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        captured = []
        dock.highlightModeChanged.connect(lambda mode: captured.append(mode))
        dock.set_highlight_mode("label_group")
        # change via UI to trigger signal path
        dock.highlight_mode_combo.setCurrentIndex(0)
        assert dock.highlight_mode() == "region_only"
        assert captured[-1] == "region_only"
    finally:
        dock.close()


def test_brain_3d_session_dock_sets_highlight_summary_text() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_highlight_summary(
            highlighted_count=3,
            total_polygons=7,
            mode="label_group",
        )
        text = dock.highlight_summary_label.text()
        assert "3/7" in text
        assert "label group" in text
    finally:
        dock.close()


def test_brain_3d_session_dock_quick_state_buttons_emit_expected_state() -> None:
    _ensure_qapp()
    dock = Brain3DSessionDockWidget()
    try:
        dock.set_summary(region_count=1, source_page_count=1, plane_count=2)
        dock.set_current_plane(1)
        dock.set_regions(
            [
                {
                    "region_id": "r1",
                    "label": "Region 1",
                    "state": "present",
                    "source": "model",
                    "points_count": 5,
                }
            ]
        )
        captured = []
        dock.regionStateRequested.connect(
            lambda plane, region, state: captured.append((plane, region, state))
        )
        dock.create_region_button.click()
        dock.hide_region_button.click()
        dock.restore_region_button.click()
        assert captured == [
            (1, "r1", "created"),
            (1, "r1", "hidden"),
            (1, "r1", "present"),
        ]
    finally:
        dock.close()
