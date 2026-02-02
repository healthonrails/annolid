from annolid.gui.widgets.behavior_describe_widget import BehaviorSegmentDialog
import os

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy import QtCore, QtWidgets  # noqa: E402


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_behavior_segment_dialog_timeline_window_count():
    _ensure_qapp()

    dialog = BehaviorSegmentDialog(
        parent=None,
        start_time=QtCore.QTime(0, 0, 0),
        end_time=QtCore.QTime(0, 0, 5),
        notes=None,
        prompt_text="test",
        video_fps=30.0,
        total_frames=300,
    )

    # Switch to timeline mode.
    dialog.run_mode_combo.setCurrentIndex(1)
    dialog.timeline_step_spin.setValue(1)

    assert dialog.run_mode() == "timeline"
    assert dialog._timeline_window_count() == 6
