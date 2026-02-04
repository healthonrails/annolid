import os

from qtpy import QtCore, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _square(label: str, dx: float = 0.0, dy: float = 0.0):
    from annolid.gui.shape import Shape

    s = Shape(label=label, shape_type="polygon")
    s.points = [
        QtCore.QPointF(0.0 + dx, 0.0 + dy),
        QtCore.QPointF(10.0 + dx, 0.0 + dy),
        QtCore.QPointF(10.0 + dx, 10.0 + dy),
        QtCore.QPointF(0.0 + dx, 10.0 + dy),
    ]
    return s


def test_rem_labels_and_selection_use_identity_not_shape_eq():
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem, AnnolidWindowBase

    w = AnnolidWindowBase(config={})
    try:
        # Two polygons that are "equal" per Shape.__eq__ (IoU high), but must
        # still be treated as distinct objects by the GUI lists.
        s1 = _square("wood", dx=0.0, dy=0.0)
        # Keep within Shape.__eq__ default epsilon (1e-1) so the objects compare equal.
        s2 = _square("wood", dx=0.05, dy=0.05)
        assert s1 == s2  # sanity: the objects compare equal

        w.labelList.addItem(AnnolidLabelListItem("wood", s1))
        w.labelList.addItem(AnnolidLabelListItem("wood", s2))

        w.shapeSelectionChanged([s1])
        assert w.labelList.item(0).isSelected() is True
        assert w.labelList.item(1).isSelected() is False

        w.remLabels([s1])
        assert w.labelList.count() == 1
        assert w.labelList.item(0).shape() is s2
    finally:
        w.close()


def test_canvas_delete_selected_clears_internal_selection_and_emits_empty():
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    c = Canvas(epsilon=2.0, double_click="close", num_backups=2, crosshair={}, sam={})
    try:
        s1 = _square("wood", dx=0.0, dy=0.0)
        c.shapes = [s1]

        emitted = []
        c.selectionChanged.connect(lambda shapes: emitted.append(list(shapes)))

        c.selectShapes([s1])
        assert len(c.selectedShapes) == 1

        deleted = c.deleteSelected()
        assert deleted and deleted[0] is s1
        assert c.selectedShapes == []
        assert c.shapes == []
        assert emitted and emitted[-1] == []
    finally:
        c.close()
