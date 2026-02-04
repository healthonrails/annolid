import os

from qtpy import QtCore, QtGui, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _find_action(menu: QtWidgets.QMenu, needle: str):
    needle_l = needle.lower()
    for action in menu.actions():
        if action.isSeparator():
            continue
        if needle_l in (action.text() or "").lower():
            return action
    return None


def _mk_polygon(label: str = "wood"):
    from annolid.gui.shape import Shape

    shape = Shape(label=label, shape_type="polygon")
    shape.points = [
        QtCore.QPointF(5.0, 5.0),
        QtCore.QPointF(30.0, 5.0),
        QtCore.QPointF(30.0, 30.0),
        QtCore.QPointF(5.0, 30.0),
    ]
    return shape


def test_canvas_context_menu_is_flat_with_no_nested_submenus():
    _ensure_qapp()
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(80, 60, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(90, 100, 120))
        w.image_to_canvas(img, "dummy.png", 0)

        menu = w.canvas._build_context_menu(w)
        actions = [a for a in menu.actions() if not a.isSeparator()]
        # Flat menu: every menu entry is a direct action, not a submenu.
        assert all(a.menu() is None for a in actions)

        texts = [a.text() for a in actions]
        assert "Edit Polygons" in texts
        assert "Create Polygons" in texts
        assert "AI Polygons" in texts
    finally:
        w.close()


def test_canvas_context_menu_shows_shape_actions_with_icons_when_selected():
    _ensure_qapp()
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(100, 80, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(130, 120, 110))
        w.image_to_canvas(img, "dummy.png", 0)

        shape = _mk_polygon()
        w.canvas.shapes = [shape]
        w.canvas.selectShapes([shape])

        menu = w.canvas._build_context_menu(w)

        propagate = _find_action(menu, "propagate")
        duplicate = _find_action(menu, "duplicate")
        delete = _find_action(menu, "delete")

        assert propagate is not None
        assert duplicate is not None
        assert delete is not None

        assert not propagate.icon().isNull()
        assert not duplicate.icon().isNull()
        assert not delete.icon().isNull()
    finally:
        w.close()
