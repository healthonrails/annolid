import json
import os
from types import SimpleNamespace
from pathlib import Path

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


def _shape_payload(label: str, dx: float = 0.0, dy: float = 0.0, group_id=None):
    return {
        "label": label,
        "points": [
            [0.0 + dx, 0.0 + dy],
            [10.0 + dx, 0.0 + dy],
            [10.0 + dx, 10.0 + dy],
            [0.0 + dx, 10.0 + dy],
        ],
        "group_id": group_id,
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "mask": None,
        "visible": True,
    }


class _DummyMainWindow:
    def __init__(self, frames: dict[int, Path]):
        from annolid.gui.window_base import AnnolidLabelListWidget

        self.frames = frames
        first_frame = sorted(frames.keys())[0]
        self.frame_number = first_frame
        self.filename = str(frames[first_frame])
        self.labelList = AnnolidLabelListWidget()
        self.canvas = None

    def _getLabelFile(self, filename):
        return str(Path(filename).with_suffix(".json"))

    def set_frame_number(self, frame):
        self.frame_number = frame
        self.filename = str(self.frames[frame])

    def validateLabel(self, text):
        return bool(str(text or "").strip())

    def _refresh_label_list_items_for_shapes(self, shapes):
        for shape in shapes:
            for item in self.labelList:
                if item.shape() is shape:
                    item.setText(
                        str(shape.label)
                        if shape.group_id is None
                        else f"{shape.label} ({shape.group_id})"
                    )


def _write_label_file(path: Path, shapes: list[dict]):
    from annolid.gui.label_file import LabelFile

    lf = LabelFile()
    lf.save(
        str(path),
        shapes,
        imagePath=None,
        imageHeight=100,
        imageWidth=100,
        imageData=None,
        otherData={},
        flags={},
        caption="",
    )


def _load_shapes(path: Path):
    from annolid.gui.label_file import LabelFile

    lf = LabelFile(str(path), is_video_frame=True)
    return lf.shapes


def _write_annotation_store(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


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


def test_new_shape_dialog_uses_placeholder_instead_of_default_label(monkeypatch):
    _ensure_qapp()

    from annolid.gui.mixins.shape_editing_mixin import ShapeEditingMixin
    from annolid.gui.shape import Shape

    class DummyCanvas:
        def __init__(self):
            self.createMode = "polygon"
            self.shapes = []
            self.shapesBackups = []
            self.captured = None

        def setLastLabel(self, text, flags):
            self.captured = (text, flags)
            shape = Shape(label=text, shape_type="rectangle")
            shape.points = [
                QtCore.QPointF(0.0, 0.0),
                QtCore.QPointF(10.0, 0.0),
                QtCore.QPointF(10.0, 10.0),
                QtCore.QPointF(0.0, 10.0),
            ]
            return [shape]

        def undoLastLine(self):
            self.undo_called = True

    class DummyShapeEditor(ShapeEditingMixin):
        def _active_shape_editor(self):
            return self.canvas

    dummy = DummyShapeEditor()
    dummy.canvas = DummyCanvas()
    dummy.labelList = QtWidgets.QListWidget()
    dummy.uniqLabelList = SimpleNamespace(
        selectedItems=lambda: [
            (
                lambda item: (
                    item.setData(QtCore.Qt.UserRole, "chamber_1"),
                    item,
                )[1]
            )(QtWidgets.QListWidgetItem("chamber_1"))
        ]
    )
    dummy._config = {"display_label_popup": True}
    dummy._zone_authoring_defaults = None
    dummy._try_apply_keypoint_sequence_labeling = lambda: False
    dummy.validateLabel = lambda text: True
    dummy.errorMessage = lambda *args, **kwargs: None
    dummy.setDirty = lambda: None
    dummy.addLabel = lambda *args, **kwargs: None
    dummy.actions = SimpleNamespace(
        editMode=SimpleNamespace(setEnabled=lambda *args, **kwargs: None),
        undoLastPoint=SimpleNamespace(setEnabled=lambda *args, **kwargs: None),
        undo=SimpleNamespace(setEnabled=lambda *args, **kwargs: None),
    )
    dummy.labelDialog = SimpleNamespace(
        edit=SimpleNamespace(
            text=lambda: "chamber_1",
            setText=lambda *args, **kwargs: None,
        ),
        popUp=lambda text, **kwargs: ("rover", {}, None, ""),
    )

    recorded = {}

    def _pop_up(text, **kwargs):
        recorded["text"] = text
        recorded["kwargs"] = kwargs
        return "rover", {}, None, ""

    dummy.labelDialog.popUp = _pop_up

    dummy.newShape()

    assert recorded["text"] == ""
    assert dummy.canvas.captured[0] == "rover"


def test_shape_dialog_includes_rectangle_in_shape_list(tmp_path):
    _ensure_qapp()

    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    rect = _square("box")
    rect.shape_type = "rectangle"
    point = _square("nose")
    point.shape_type = "point"

    canvas = type("Canvas", (), {})()
    canvas.shapes = [rect, point]

    window = _DummyMainWindow({0: tmp_path / "video_000000000.png"})
    window.canvas = canvas

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=10)
    labels = [
        dialog.shape_list.item(i).text() for i in range(dialog.shape_list.count())
    ]

    assert labels == ["box"]


def test_shape_dialog_default_end_frame_uses_next_manual_seed_minus_one(tmp_path):
    _ensure_qapp()

    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000010.png"
    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_1")]

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({10: current_png})
            self.canvas = canvas
            self.video_results_folder = tmp_path
            self.manual_seed_frames = {20, 30}

        def _discover_manual_seed_frames(self, folder):
            return self.manual_seed_frames

    window = Window()
    dialog = ShapePropagationDialog(canvas, window, current_frame=10, max_frame=100)

    assert dialog.frame_spin.value() == 19
    assert dialog.event_end_frame_spin.value() == 19


def test_shape_dialog_default_end_frame_falls_back_to_last_video_frame_when_no_future_seed(
    tmp_path,
):
    _ensure_qapp()

    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000010.png"
    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_1")]

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({10: current_png})
            self.canvas = canvas
            self.video_results_folder = tmp_path
            self.manual_seed_frames = {5, 8}

        def _discover_manual_seed_frames(self, folder):
            return self.manual_seed_frames

    window = Window()
    dialog = ShapePropagationDialog(canvas, window, current_frame=10, max_frame=100)

    assert dialog.frame_spin.value() == 100
    assert dialog.event_end_frame_spin.value() == 100


def test_shape_dialog_rename_and_propagate_uses_identity_and_assigns_group_id(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000000.png"
    future_png = tmp_path / "video_000000001.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")

    current_shape = _square("stim_2", dx=0.0, dy=0.0)
    future_match = _shape_payload("stim_2", dx=0.5, dy=0.5)
    future_distractor = _shape_payload("stim_2", dx=80.0, dy=80.0)
    _write_label_file(current_json, [_shape_payload("stim_2")])
    _write_label_file(future_json, [future_match, future_distractor])

    canvas = type("Canvas", (), {})()
    canvas.shapes = [current_shape]
    canvas.selectedShapes = [current_shape]
    canvas.update = lambda: None

    window = _DummyMainWindow({0: current_png, 1: future_png})
    window.canvas = canvas
    window.labelList.addItem(AnnolidLabelListItem("stim_2", current_shape))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=1)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    current_shapes = _load_shapes(current_json)
    future_shapes = _load_shapes(future_json)

    assert current_shapes[0]["label"] == "rover"
    assert current_shapes[0]["group_id"] == 0

    renamed_future = [shape for shape in future_shapes if shape["label"] == "rover"]
    assert len(renamed_future) == 2
    assert all(shape["group_id"] == 0 for shape in renamed_future)
    assert not any(shape["label"] == "stim_2" for shape in future_shapes)


def test_shape_dialog_propagate_rectangle_preserves_other_future_shapes(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000000.png"
    future_png = tmp_path / "video_000000001.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")

    current_rect = _square("box")
    current_rect.shape_type = "rectangle"
    _write_label_file(
        current_json,
        [
            _shape_payload("box", group_id=None) | {"shape_type": "rectangle"},
        ],
    )
    _write_label_file(
        future_json,
        [
            _shape_payload("mouse", dx=20.0, dy=20.0),
            _shape_payload("teaball", dx=80.0, dy=80.0),
        ],
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [current_rect]
    canvas.selectedShapes = [current_rect]
    canvas.update = lambda: None

    window = _DummyMainWindow({0: current_png, 1: future_png})
    window.canvas = canvas
    window.labelList.addItem(AnnolidLabelListItem("box", current_rect))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=1)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Propagate")

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    future_shapes = _load_shapes(future_json)
    labels = [shape["label"] for shape in future_shapes]

    assert "box" in labels
    assert labels.count("mouse") == 1
    assert labels.count("teaball") == 1


def test_shape_dialog_rename_updates_annotation_store_backed_json(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000010.png"
    future_png = tmp_path / "video_000000011.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 10,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
            {
                "frame": 11,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": future_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
        ],
    )
    current_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 10,
                "version": 1,
                "imagePath": current_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    future_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 11,
                "version": 1,
                "imagePath": future_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    window = _DummyMainWindow({10: current_png, 11: future_png})
    window.canvas = canvas
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=10, max_frame=11)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(
        shape["label"] == "rover"
        for record in store_lines
        for shape in record.get("shapes", [])
    )


def test_shape_dialog_rename_updates_store_backed_prediction_without_json_stub(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000019412.png"
    future_png = tmp_path / "video_000019413.png"
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 19412,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
            {
                "frame": 19413,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": future_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
        ],
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({19412: current_png, 19413: future_png})
            self.canvas = canvas
            self.load_predict_shapes_calls = []
            self.load_file_calls = []
            self.filename = str(current_png)
            self._prediction_store_path = store_path

        def loadPredictShapes(self, frame_number, filename):
            self.load_predict_shapes_calls.append((frame_number, filename))

        def loadFile(self, filename):
            self.load_file_calls.append(filename)

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(
        canvas, window, current_frame=19412, max_frame=19413
    )
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")
    dialog.frame_spin.setValue(19413)

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    assert not (current_png.with_suffix(".json")).exists()
    assert not (future_png.with_suffix(".json")).exists()

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(
        shape["label"] == "rover"
        for record in store_lines
        for shape in record.get("shapes", [])
    )


def test_shape_dialog_rename_updates_legacy_store_without_frame_keys(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    mouse_dir = tmp_path / "mouse"
    mouse_dir.mkdir()
    current_png = mouse_dir / "mouse_000000000.png"
    future_png = mouse_dir / "mouse_000000001.png"
    store_path = mouse_dir / "mouse_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "shapes": [
                    {
                        "label": "chamber_1",
                        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                ],
                "imageHeight": 300,
                "imageWidth": 480,
            },
            {
                "shapes": [
                    {
                        "label": "chamber_1",
                        "points": [[1, 1], [11, 1], [11, 11], [1, 11]],
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                ],
                "imageHeight": 300,
                "imageWidth": 480,
            },
        ],
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("chamber_1")]
    canvas.shapes[0].shape_type = "rectangle"
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    window = _DummyMainWindow({0: current_png, 1: future_png})
    window.canvas = canvas
    window.labelList.addItem(AnnolidLabelListItem("chamber_1", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=1)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("mouse")

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(
        shape["label"] == "mouse"
        for record in store_lines
        for shape in record.get("shapes", [])
    )


def test_shape_dialog_propagate_skips_unloadable_store_frames_instead_of_overwriting(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    mouse_dir = tmp_path / "mouse"
    mouse_dir.mkdir()
    current_png = mouse_dir / "mouse_000000000.png"
    future_png = mouse_dir / "mouse_000000001.png"
    store_path = mouse_dir / "mouse_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "shapes": [
                    {
                        "label": "chamber_1",
                        "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [[20, 20], [30, 20], [30, 30], [20, 30]],
                        "shape_type": "polygon",
                        "flags": {},
                    },
                ],
                "imageHeight": 300,
                "imageWidth": 480,
            }
        ],
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("chamber_1")]
    canvas.shapes[0].shape_type = "rectangle"
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    window = _DummyMainWindow({0: current_png, 1: future_png})
    window.canvas = canvas
    window.labelList.addItem(AnnolidLabelListItem("chamber_1", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=1)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Propagate")
    dialog.frame_spin.setValue(1)

    monkeypatch.setattr(
        dialog,
        "_resolve_annotation_target",
        lambda label_file: ("store", None, None)
        if label_file.endswith("mouse_000000001.json")
        else ("json", None, None),
    )
    monkeypatch.setattr(dialog, "_load_existing_label_file", lambda label_file: None)
    monkeypatch.setattr(
        dialog,
        "load_or_create_label_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "blank-create path should not be used for store-backed frames"
            )
        ),
    )

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [shape["label"] for shape in store_lines[0]["shapes"]] == [
        "chamber_1",
        "mouse",
    ]


def test_shape_dialog_manual_seed_json_prefers_json_over_store(
    tmp_path,
):
    _ensure_qapp()

    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000020.png"
    current_json = current_png.with_suffix(".json")
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 20,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            }
        ],
    )
    _write_label_file(current_json, [_shape_payload("stim_1")])

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_1")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({20: current_png})
            self.canvas = canvas
            self.filename = str(current_png)

    window = Window()
    dialog = ShapePropagationDialog(canvas, window, current_frame=20, max_frame=20)

    from annolid.gui.label_file import LabelFile

    lf = LabelFile(str(current_json), is_video_frame=True)
    lf.shapes[0]["label"] = "rover"
    dialog._save_shape_file(str(current_json), lf)

    updated_json = _load_shapes(current_json)
    assert updated_json[0]["label"] == "rover"

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert store_lines[0]["shapes"][0]["label"] == "stim_2"


def test_shape_dialog_rename_and_propagate_clamps_to_last_predicted_frame(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000010.png"
    future_png = tmp_path / "video_000000011.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 10,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
            {
                "frame": 11,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": future_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
        ],
    )
    current_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 10,
                "version": 1,
                "imagePath": current_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    future_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 11,
                "version": 1,
                "imagePath": future_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({10: current_png, 11: future_png})
            self.canvas = canvas
            self.load_predict_shapes_calls = []
            self.load_file_calls = []
            self.filename = str(current_png)

        def loadPredictShapes(self, frame_number, filename):
            self.load_predict_shapes_calls.append((frame_number, filename))

        def loadFile(self, filename):
            self.load_file_calls.append(filename)

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=10, max_frame=99)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")
    dialog.frame_spin.setValue(99)

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    assert window.frame_number == 11
    assert window.load_predict_shapes_calls
    assert window.load_predict_shapes_calls[-1][0] == 11
    assert not window.load_file_calls


def test_shape_dialog_rename_and_propagate_seeks_only_once(monkeypatch, tmp_path):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000010.png"
    future_png = tmp_path / "video_000000011.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 10,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
            {
                "frame": 11,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": future_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            },
        ],
    )
    current_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 10,
                "version": 1,
                "imagePath": current_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    future_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 11,
                "version": 1,
                "imagePath": future_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({10: current_png, 11: future_png})
            self.canvas = canvas
            self.load_predict_shapes_calls = []
            self.set_frame_number_calls = []
            self.filename = str(current_png)

        def set_frame_number(self, frame):
            self.set_frame_number_calls.append(frame)
            super().set_frame_number(frame)

        def loadPredictShapes(self, frame_number, filename):
            self.load_predict_shapes_calls.append((frame_number, filename))

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=10, max_frame=11)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    assert window.set_frame_number_calls == [11]
    assert window.load_predict_shapes_calls[-1][0] == 11

    store_lines = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(
        shape["label"] == "rover"
        for record in store_lines
        for shape in record.get("shapes", [])
    )


def test_shape_dialog_propagate_still_creates_future_label_files(monkeypatch, tmp_path):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000000.png"
    future_png = tmp_path / "video_000000003.png"
    current_json = current_png.with_suffix(".json")
    future_json = future_png.with_suffix(".json")

    _write_label_file(current_json, [_shape_payload("stim_2")])

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__(
                {
                    0: current_png,
                    1: tmp_path / "video_000000001.png",
                    2: tmp_path / "video_000000002.png",
                    3: future_png,
                }
            )
            self.canvas = canvas

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=3)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Propagate")
    dialog.frame_spin.setValue(3)

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )

    dialog.do_action()

    assert future_json.exists()
    future_shapes = _load_shapes(future_json)
    assert any(shape["label"] == "stim_2" for shape in future_shapes)


def test_shape_dialog_rename_and_propagate_skips_missing_future_files(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000000000.png"
    missing_future_png = tmp_path / "video_000000001.png"
    current_json = current_png.with_suffix(".json")
    missing_future_json = missing_future_png.with_suffix(".json")

    _write_label_file(current_json, [_shape_payload("stim_2")])

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({0: current_png, 1: missing_future_png})
            self.canvas = canvas

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(canvas, window, current_frame=0, max_frame=1)
    dialog.shape_list.setCurrentRow(0)
    dialog.action_combo.setCurrentText("Rename & Propagate")
    dialog.rename_line.setText("rover")
    dialog.frame_spin.setValue(1)

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: None)

    dialog.do_action()

    assert not missing_future_json.exists()
    current_shapes = _load_shapes(current_json)
    assert current_shapes[0]["label"] == "rover"


def test_shape_dialog_reload_uses_predict_shapes_for_missing_frame_png(
    monkeypatch, tmp_path
):
    _ensure_qapp()

    from annolid.gui.window_base import AnnolidLabelListItem
    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    current_png = tmp_path / "video_000019416.png"
    current_json = current_png.with_suffix(".json")
    store_path = tmp_path / f"{tmp_path.name}_annotations.ndjson"

    _write_annotation_store(
        store_path,
        [
            {
                "frame": 19416,
                "shapes": [
                    {
                        "label": "stim_2",
                        "points": [[0, 0], [10, 0], [10, 10]],
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": current_png.name,
                "imageHeight": 100,
                "imageWidth": 100,
                "annotation_store": store_path.name,
            }
        ],
    )
    current_json.write_text(
        json.dumps(
            {
                "annotation_store": store_path.name,
                "frame": 19416,
                "version": 1,
                "imagePath": current_png.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    canvas = type("Canvas", (), {})()
    canvas.shapes = [_square("stim_2")]
    canvas.selectedShapes = [canvas.shapes[0]]
    canvas.update = lambda: None

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({19416: current_png})
            self.canvas = canvas
            self.load_predict_shapes_calls = []
            self.load_file_calls = []
            self.filename = str(current_png)

        def loadPredictShapes(self, frame_number, filename):
            self.load_predict_shapes_calls.append((frame_number, filename))

        def loadFile(self, filename):
            self.load_file_calls.append(filename)

    window = Window()
    window.labelList.addItem(AnnolidLabelListItem("stim_2", canvas.shapes[0]))

    dialog = ShapePropagationDialog(
        canvas, window, current_frame=19416, max_frame=19416
    )
    dialog._reload_annotation_view(19416)

    assert window.load_predict_shapes_calls
    assert not window.load_file_calls
    assert window.load_predict_shapes_calls[-1][0] == 19416


def test_shape_dialog_reload_silently_skips_missing_png_without_predict_loader(
    tmp_path,
):
    _ensure_qapp()

    from annolid.gui.widgets.shape_dialog import ShapePropagationDialog

    missing_png = tmp_path / "video_000019416.png"

    class Window(_DummyMainWindow):
        def __init__(self):
            super().__init__({19416: missing_png})
            self.canvas = type("Canvas", (), {"update": lambda self: None})()
            self.load_file_calls = []
            self.filename = str(missing_png)

        def loadFile(self, filename):
            self.load_file_calls.append(filename)

    canvas = type(
        "Canvas", (), {"shapes": [], "selectedShapes": [], "update": lambda self: None}
    )()
    window = Window()
    dialog = ShapePropagationDialog(
        canvas, window, current_frame=19416, max_frame=19416
    )

    dialog._reload_annotation_view(19416)

    assert window.load_file_calls == []


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


def test_propagate_selected_shape_uses_identity_selection(monkeypatch):
    _ensure_qapp()

    from annolid.gui.mixins.label_panel_mixin import LabelPanelMixin

    shape_a = _square("mouse")
    shape_b = _square("mouse", dx=0.02, dy=0.02)
    assert shape_a == shape_b

    class DummyWindow(LabelPanelMixin):
        def __init__(self):
            self.canvas = type("Canvas", (), {})()
            self.canvas.selectedShapes = [shape_a]
            self.canvas.shapes = [shape_a, shape_b]
            self.canvas.update = lambda: None
            self.frame_number = 0
            self.num_frames = 1

    window = DummyWindow()
    dialog = SimpleNamespace(shape_list=QtWidgets.QListWidget())
    item_a = QtWidgets.QListWidgetItem("mouse")
    item_a.setData(QtCore.Qt.UserRole, shape_a)
    item_b = QtWidgets.QListWidgetItem("mouse")
    item_b.setData(QtCore.Qt.UserRole, shape_b)
    dialog.shape_list.addItem(item_a)
    dialog.shape_list.addItem(item_b)

    assert window._select_shape_row_in_propagation_dialog(dialog, shape_a) is True
    assert dialog.shape_list.currentRow() == 0
