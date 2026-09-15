import os

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import pytest
from qtpy import QtCore, QtGui, QtWidgets
from shapely.geometry import Polygon

from annolid.gui.shape import Shape
from annolid.gui.shape_merge import merge_shapes


def rectangle(x1=0, y1=0, x2=10, y2=10):
    shape = Shape(label="mouse", shape_type="rectangle", flags={})
    shape.points = [QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)]
    shape.point_labels = [1, 1]
    return shape


def test_merge_exact_union_preserves_metadata_and_inputs():
    a, b = rectangle(), rectangle(5, 5, 15, 15)
    for shape in (a, b):
        shape.group_id = 7
        shape.other_data = {"source": {"name": "manual"}}
    merged = merge_shapes([a, b])
    assert Polygon([(p.x(), p.y()) for p in merged.points]).area == 175
    assert merged.group_id == 7
    assert merged.shared_vertex_ids == []
    merged.other_data["source"]["name"] = "changed"
    assert a.other_data["source"]["name"] == "manual"
    assert a.shape_type == "rectangle" and len(a.points) == 2


@pytest.mark.parametrize(
    "case", ["disconnected", "hole", "metadata", "invalid", "point", "vertex"]
)
def test_reject_lossy_merge(case):
    a, b = rectangle(), rectangle(5, 0, 15, 10)
    shapes = [a, b]
    if case == "disconnected":
        shapes[1] = rectangle(20, 20, 30, 30)
    elif case == "hole":
        shapes = [
            rectangle(0, 0, 10, 2),
            rectangle(0, 8, 10, 10),
            rectangle(0, 0, 2, 10),
            rectangle(8, 0, 10, 10),
        ]
    elif case == "metadata":
        b.group_id = 8
    elif case == "invalid":
        shapes[1] = rectangle(0, 0, 0, 10)
    elif case == "point":
        b.shape_type = "point"
    else:
        b.point_labels = ["nose"]
    with pytest.raises(ValueError):
        merge_shapes(shapes)


def test_canvas_merge_action_and_undo(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        image = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor("black"))
        w.image_to_canvas(image, "merge.png", 0)
        a, b, untouched = (
            rectangle(),
            rectangle(5, 0, 15, 10),
            rectangle(40, 40, 50, 50),
        )
        w.loadShapes([a, b, untouched])
        w.canvas.setEditing(True)
        w.canvas.selectShapes([a, b])
        assert w.actions.mergeShapes.isEnabled()
        menu = w.canvas._build_context_menu(w)
        assert w.actions.mergeShapes in menu.actions()
        backups = len(w.canvas.shapesBackups)
        w.actions.mergeShapes.trigger()
        assert len(w.canvas.shapes) == w.labelList.count() == 2
        assert w.canvas.shapes[1] is untouched
        assert w.canvas.selectedShapes == [w.canvas.shapes[0]]
        assert len(w.canvas.shapesBackups) == backups + 1
        assert w.dirty
        w.undoShapeEdit()
        assert len(w.canvas.shapes) == w.labelList.count() == 3
        assert [s.shape_type for s in w.canvas.shapes] == ["rectangle"] * 3
        assert not w.canvas.selectedShapes
        w.canvas.selectShapes(w.canvas.shapes[:2])
        w.canvas.shapes[1].label = "other"
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: None)
        backups = len(w.canvas.shapesBackups)
        assert not w.mergeSelectedShapes()
        assert len(w.canvas.shapes) == 3
        assert len(w.canvas.shapesBackups) == backups
    finally:
        w.setClean()
        w.close()
    assert app is not None


@pytest.mark.parametrize("bounds,area", [((10, 0, 20, 10), 200), ((2, 2, 8, 8), 100)])
def test_merge_polygon_with_adjacent_or_contained_rectangle(bounds, area):
    a = Shape(label="mouse", shape_type="polygon", flags={})
    for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        a.addPoint(QtCore.QPointF(x, y))
    a.close()
    merged = merge_shapes([a, rectangle(*bounds)])
    assert Polygon([(p.x(), p.y()) for p in merged.points]).area == area
    assert len(merged.point_labels) == len(merged.points)
    merged.removePoint(0)
    assert len(merged.point_labels) == len(merged.points)
