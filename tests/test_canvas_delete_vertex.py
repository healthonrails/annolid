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


def test_delete_key_removes_selected_polygon_vertex_and_keeps_polygon_valid():
    _ensure_qapp()
    from annolid.gui.app import AnnolidWindow
    from annolid.gui.shape import Shape

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(120, 80, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(90, 110, 130))
        w.image_to_canvas(img, "dummy.png", 0)

        shape = Shape(label="poly", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(5.0, 5.0),
            QtCore.QPointF(50.0, 5.0),
            QtCore.QPointF(50.0, 50.0),
            QtCore.QPointF(5.0, 50.0),
        ]
        shape.point_labels = [1, 1, 1, 1]
        w.canvas.shapes = [shape]
        w.canvas.storeShapes()
        w.canvas.selectShapes([shape])
        w.canvas.hShape = shape
        w.canvas.prevhShape = shape
        w.canvas.hVertex = 1
        w.canvas.prevhVertex = 1

        delete_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Delete,
            QtCore.Qt.NoModifier,
        )
        w.canvas.keyPressEvent(delete_event)
        assert len(shape.points) == 3

        # Must remain a valid polygon (>= 3 points), so next delete is ignored.
        second_delete_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Delete,
            QtCore.Qt.NoModifier,
        )
        w.canvas.keyPressEvent(second_delete_event)
        assert len(shape.points) == 3
    finally:
        w.close()
