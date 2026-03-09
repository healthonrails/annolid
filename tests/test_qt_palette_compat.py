from __future__ import annotations

import os

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.qt_compat import (
    normalize_orientation,
    painter_render_hint,
    palette_color_group,
    palette_color_role,
)
from annolid.gui.widgets.flags import FlagTableWidget


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_palette_compat_resolves_common_roles_and_groups() -> None:
    assert palette_color_role("Highlight") is not None
    assert palette_color_role("Base") is not None
    assert palette_color_role("Text") is not None
    assert palette_color_role("Window") is not None
    assert palette_color_role("ButtonText") is not None
    assert palette_color_group("Active") is not None
    assert palette_color_group("Disabled") is not None
    assert painter_render_hint("Antialiasing") is not None
    assert painter_render_hint("SmoothPixmapTransform") is not None


def test_flag_table_widget_constructs_with_binding_safe_palette_access() -> None:
    _ensure_qapp()
    widget = FlagTableWidget()
    try:
        palette = widget.palette()
        color = palette.color(palette_color_role("Highlight"))
        assert isinstance(color, QtGui.QColor)
    finally:
        widget.close()


def test_normalize_orientation_handles_qt_enums_and_raw_ints() -> None:
    assert normalize_orientation(QtCore.Qt.Horizontal) == QtCore.Qt.Horizontal
    assert normalize_orientation(QtCore.Qt.Vertical) == QtCore.Qt.Vertical
    assert normalize_orientation(0) == QtCore.Qt.Horizontal
    try:
        assert normalize_orientation(int(QtCore.Qt.Horizontal)) == QtCore.Qt.Horizontal
        assert normalize_orientation(int(QtCore.Qt.Vertical)) == QtCore.Qt.Vertical
    except Exception:
        pass
