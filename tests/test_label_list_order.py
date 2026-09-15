import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.shape import Shape
from annolid.gui.window_base import AnnolidLabelListWidget, AnnolidUniqLabelListWidget


@pytest.fixture
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    image = QtGui.QImage(80, 80, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor("black"))
    w.image_to_canvas(image, "order.png", 0)
    shapes = []
    for index, label in enumerate(["zebra", "ant", "mouse"]):
        shape = Shape(
            label=label,
            shape_type="rectangle",
            flags={"reviewed": True},
            group_id=index,
            visible=index != 1,
        )
        shape.points = [
            QtCore.QPointF(index * 10, 0),
            QtCore.QPointF(index * 10 + 5, 5),
        ]
        shapes.append(shape)
    w.loadShapes(shapes)
    w.setClean()
    yield w
    w.setClean()
    w.close()
    assert app is not None


def move(widget, source, destination):
    assert widget.model().moveRows(
        QtCore.QModelIndex(), source, 1, QtCore.QModelIndex(), destination
    )


def labels(widget):
    return [widget.item(i).data(QtCore.Qt.UserRole) for i in range(widget.count())]


def test_shape_order_updates_canvas_and_undo_without_losing_state(window, tmp_path):
    w = window
    original = list(w.canvas.shapes)
    original_items = list(w.labelList)
    w.canvas.selectShapes([original[0], original[2]])
    backups = len(w.canvas.shapesBackups)
    move(w.labelList, 0, 3)
    expected = [original[1], original[2], original[0]]
    assert all(a is b for a, b in zip(w.canvas.shapes, expected))
    assert list(w.labelList) == [
        original_items[1],
        original_items[2],
        original_items[0],
    ]
    assert {id(s) for s in w.canvas.selectedShapes} == {
        id(original[0]),
        id(original[2]),
    }
    assert not original[1].visible
    assert w.labelList.item(0).checkState() == QtCore.Qt.Unchecked
    assert all(s.flags == {"reviewed": True} for s in w.canvas.shapes)
    assert w.dirty and w.actions.undo.isEnabled()
    assert len(w.canvas.shapesBackups) == backups + 1
    saved = tmp_path / "ordered.json"
    assert w.saveLabels(str(saved), save_image_data=False)
    payload = json.loads(saved.read_text())
    assert [s["label"] for s in payload["shapes"]] == ["ant", "mouse", "zebra"]
    assert [s["group_id"] for s in payload["shapes"]] == [1, 2, 0]
    assert not payload["shapes"][0]["visible"]
    w.undoShapeEdit()
    assert [s.label for s in w.canvas.shapes] == ["zebra", "ant", "mouse"]
    assert [item.shape().label for item in w.labelList] == ["zebra", "ant", "mouse"]
    assert not w.canvas.shapes[1].visible


def test_unique_label_order_survives_refresh_and_file_switch(window):
    w = window
    original_shapes = list(w.canvas.shapes)
    assert labels(w.uniqLabelList) == ["ant", "mouse", "zebra"]
    w.uniqLabelList.item(2).setSelected(True)
    move(w.uniqLabelList, 2, 0)
    w._rebuild_unique_label_list()
    assert labels(w.uniqLabelList) == ["zebra", "ant", "mouse"]
    assert [i.data(QtCore.Qt.UserRole) for i in w.uniqLabelList.selectedItems()] == [
        "zebra"
    ]
    assert not w.dirty
    assert all(a is b for a, b in zip(original_shapes, w.canvas.shapes))
    w.loadShapes([original_shapes[1]])
    assert labels(w.uniqLabelList) == ["ant"]
    w.loadShapes(original_shapes)
    assert labels(w.uniqLabelList) == ["zebra", "ant", "mouse"]


@pytest.mark.parametrize(
    "widget_type", [AnnolidLabelListWidget, AnnolidUniqLabelListWidget]
)
def test_internal_drop_batches_changes_and_rejects_external(
    window, monkeypatch, widget_type
):
    widget = widget_type()
    widget.addItems(["a", "b", "c"])
    calls = []
    widget.orderChanged.connect(lambda: calls.append(True))

    class Event:
        ignored = False

        def source(self):
            return widget

        def ignore(self):
            self.ignored = True

    def drop(self, event):
        move(self, 0, 3)
        move(self, 0, 3)

    monkeypatch.setattr(QtWidgets.QListWidget, "dropEvent", drop)
    assert widget.dragDropMode() == QtWidgets.QAbstractItemView.InternalMove
    widget.dropEvent(Event())
    assert calls == [True]
    assert [widget.item(i).text() for i in range(3)] == ["c", "a", "b"]
    event = Event()
    event.source = lambda: window.labelList
    widget.dropEvent(event)
    assert event.ignored
    assert calls == [True]
    widget.close()
