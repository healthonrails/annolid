from __future__ import annotations

from qtpy import QtCore, QtWidgets


def set_widget_busy_cursor(widget: QtWidgets.QWidget | None, busy: bool) -> None:
    """Apply or clear a wait cursor on a single widget hierarchy.

    This avoids the global QApplication override cursor stack, which is fragile
    when multiple dialogs, threads, or nested operations overlap.
    """
    if widget is None:
        return

    try:
        if busy:
            widget.setCursor(QtCore.Qt.WaitCursor)
        else:
            widget.unsetCursor()
    except Exception:
        pass

    viewport = getattr(widget, "viewport", None)
    if not callable(viewport):
        return

    try:
        vp = viewport()
        if vp is None:
            return
        if busy:
            vp.setCursor(QtCore.Qt.WaitCursor)
        else:
            vp.unsetCursor()
    except Exception:
        pass
