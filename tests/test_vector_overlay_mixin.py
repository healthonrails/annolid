from __future__ import annotations

import os

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.vector_overlay_mixin import VectorOverlayMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.vector_overlay_dock import VectorOverlayDockWidget
from annolid.gui.window_base import AnnolidLabelListItem, AnnolidWindowBase


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _CanvasStub(QtWidgets.QWidget):
    overlayLandmarkPairSelected = QtCore.Signal(str)
    selectionChanged = QtCore.Signal(list)

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.update_calls = 0
        self.selected_pair_id = None
        self.selectedShapes = []
        self.select_shapes_calls = []
        self.last_pixmap = None
        self.last_load_clear_shapes = None
        self.editing_values = []

    def update(self):
        self.update_calls += 1
        super().update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.last_pixmap = pixmap
        self.last_load_clear_shapes = bool(clear_shapes)
        if clear_shapes:
            self.shapes = []

    def setEditing(self, value=True):
        self.editing_values.append(bool(value))

    def setSelectedOverlayLandmarkPair(self, pair_id):
        self.selected_pair_id = str(pair_id or "") or None

    def selectShapes(self, shapes):
        self.selectedShapes = list(shapes or [])
        self.select_shapes_calls.append(list(self.selectedShapes))
        self.selectionChanged.emit(self.selectedShapes)


class _LargeImageViewStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.last_shapes = None
        self.selected_pair_id = None
        self.clear_calls = 0

    def set_shapes(self, shapes):
        self.last_shapes = list(shapes or [])

    def set_selected_landmark_pair(self, pair_id):
        self.selected_pair_id = str(pair_id or "") or None

    def clear(self):
        self.clear_calls += 1


class _OverlayHost(VectorOverlayMixin, AnnolidWindowBase):
    def __init__(self):
        self._toggle_calls = []
        self.dirty_calls = 0
        self.posted_status_messages = []
        super().__init__(config={})
        self.canvas = _CanvasStub()
        self.large_image_view = _LargeImageViewStub()
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.addWidget(self.canvas)
        self._viewer_stack.addWidget(self.large_image_view)
        self._active_image_view = "canvas"

    def toggleActions(self, value):
        self._toggle_calls.append(bool(value))

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.canvas.shapes = list(shapes or [])
        else:
            self.canvas.shapes.extend(list(shapes or []))

    def setDirty(self):
        self.dirty = True
        self.dirty_calls += 1

    def errorMessage(self, title, message):
        self.last_error = (title, message)

    def post_status_message(self, message, timeout=4000):
        self.posted_status_messages.append((str(message), int(timeout)))


def _make_overlay_shape() -> Shape:
    shape = Shape("atlas", shape_type="polygon")
    shape.addPoint(QtCore.QPointF(1.0, 2.0))
    shape.addPoint(QtCore.QPointF(5.0, 2.0))
    shape.addPoint(QtCore.QPointF(5.0, 6.0))
    shape.close()
    shape.other_data = {
        "overlay_id": "overlay_a",
        "overlay_visible": True,
        "overlay_opacity": 0.5,
        "overlay_z_order": 0,
    }
    shape.visible = True
    return shape


def _make_point(label: str, x: float, y: float, *, overlay: bool) -> Shape:
    shape = Shape(label, shape_type="point")
    shape.addPoint(QtCore.QPointF(float(x), float(y)))
    shape.visible = True
    if overlay:
        shape.other_data = {
            "overlay_id": "overlay_a",
            "overlay_visible": True,
            "overlay_opacity": 0.5,
            "overlay_z_order": 0,
        }
    return shape


def test_set_vector_overlay_transform_updates_shapes_and_metadata() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        shape = _make_overlay_shape()
        window.canvas.shapes = [shape]
        item = AnnolidLabelListItem("atlas", shape)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        window.labelList.addItem(item)
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "sx": 1.0,
                        "sy": 1.0,
                        "rotation_deg": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                        "z_order": 0,
                    },
                }
            ]
        }

        changed = window.setVectorOverlayTransform(
            "overlay_a", tx=3.0, ty=4.0, opacity=0.25, visible=False, z_order=7
        )

        assert changed is True
        assert [(round(p.x(), 3), round(p.y(), 3)) for p in shape.points] == [
            (4.0, 6.0),
            (8.0, 6.0),
            (8.0, 10.0),
        ]
        assert shape.visible is False
        assert shape.other_data["overlay_visible"] is False
        assert shape.other_data["overlay_opacity"] == 0.25
        assert shape.other_data["overlay_z_order"] == 7
        assert window.otherData["svg_overlays"][0]["transform"]["tx"] == 3.0
        assert window.otherData["svg_overlays"][0]["metadata"]["transform"]["ty"] == 4.0
        assert item.checkState() == QtCore.Qt.Unchecked
        assert window.canvas.update_calls == 1
        assert window.large_image_view.last_shapes == [shape]
        assert window.dirty_calls == 1
    finally:
        window.close()


def test_import_svg_overlay_uses_post_status_message_when_status_is_missing(
    tmp_path, monkeypatch
) -> None:
    _ensure_qapp()

    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><polygon points="0,0 10,0 10,10"/></svg>',
        encoding="utf-8",
    )

    from annolid.gui.svg_overlay import SvgImportResult

    window = _OverlayHost()
    try:
        window.imagePath = str(svg_path.with_suffix(".png"))
        window.image = QtGui.QImage(16, 16, QtGui.QImage.Format_RGB32)
        window.image.fill(QtGui.QColor(10, 20, 30))
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(svg_path), "Vector files (*.svg *.pdf)"),
        )
        monkeypatch.setattr(
            "annolid.gui.mixins.vector_overlay_mixin.import_vector_shapes",
            lambda filename: SvgImportResult(
                shapes=[_make_overlay_shape()],
                metadata={"id": "overlay_a", "transform": {}},
            ),
        )

        window.importSvgOverlay()

        assert window.posted_status_messages
        assert (
            "Imported 1 vector overlay shapes" in window.posted_status_messages[-1][0]
        )
        assert window.otherData["svg_overlays"][0]["metadata"]["source_kind"] == "svg"
    finally:
        window.close()


def test_import_svg_overlay_keeps_large_image_session_in_tiled_view(
    tmp_path, monkeypatch
) -> None:
    _ensure_qapp()

    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><polyline points="0,0 10,0 10,10"/></svg>',
        encoding="utf-8",
    )

    from annolid.gui.svg_overlay import SvgImportResult

    window = _OverlayHost()
    try:
        window.imagePath = str(tmp_path / "atlas.ome.tiff")
        window.image = QtGui.QImage(128, 96, QtGui.QImage.Format_RGB32)
        window.image.fill(QtGui.QColor(10, 20, 30))
        window._active_image_view = "tiled"
        window._viewer_stack.setCurrentWidget(window.large_image_view)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(svg_path), "Vector files (*.svg *.ai *.pdf)"),
        )
        monkeypatch.setattr(
            "annolid.gui.mixins.vector_overlay_mixin.import_vector_shapes",
            lambda filename: SvgImportResult(
                shapes=[_make_overlay_shape()],
                metadata={"id": "overlay_a", "transform": {}},
            ),
        )

        window.importSvgOverlay()

        assert window._active_image_view == "tiled"
        assert window._viewer_stack.currentWidget() is window.large_image_view
        assert window.canvas.last_pixmap is None
        assert window.large_image_view.clear_calls == 0
        assert (
            "Imported 1 vector overlay shapes" in window.posted_status_messages[-1][0]
        )
    finally:
        window.close()


def test_import_svg_overlay_initially_fits_small_overlay_to_image(
    tmp_path, monkeypatch
) -> None:
    _ensure_qapp()

    svg_path = tmp_path / "atlas.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")

    from annolid.gui.svg_overlay import SvgImportResult

    window = _OverlayHost()
    try:
        window.imagePath = str(svg_path.with_suffix(".png"))
        window.image = QtGui.QImage(1000, 800, QtGui.QImage.Format_RGB32)
        window.image.fill(QtGui.QColor(10, 20, 30))
        small_shape = _make_overlay_shape()
        small_shape.points = [
            QtCore.QPointF(0.0, 0.0),
            QtCore.QPointF(100.0, 0.0),
            QtCore.QPointF(100.0, 100.0),
        ]
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(svg_path), "Vector files (*.svg *.pdf)"),
        )
        monkeypatch.setattr(
            "annolid.gui.mixins.vector_overlay_mixin.import_vector_shapes",
            lambda filename: SvgImportResult(
                shapes=[small_shape],
                metadata={
                    "id": "overlay_a",
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "sx": 1.0,
                        "sy": 1.0,
                        "rotation_deg": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                        "z_order": 0,
                    },
                },
            ),
        )

        window.importSvgOverlay()

        imported = window.canvas.shapes[-1]
        xs = [point.x() for point in imported.points]
        ys = [point.y() for point in imported.points]
        transform = window.otherData["svg_overlays"][0]["transform"]

        assert min(xs) >= 0.0
        assert max(xs) <= 1000.0
        assert min(ys) >= 0.0
        assert max(ys) <= 800.0
        assert transform["sx"] > 1.0
        assert (
            window.otherData["svg_overlays"][0]["metadata"]["initial_fit_to_image"]
            is True
        )
    finally:
        window.close()


def test_set_vector_overlay_transform_scales_around_overlay_center() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        shape = _make_overlay_shape()
        window.canvas.shapes = [shape]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "sx": 1.0,
                        "sy": 1.0,
                        "rotation_deg": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                        "z_order": 0,
                    },
                }
            ]
        }

        changed = window.setVectorOverlayTransform("overlay_a", sx=2.0, sy=2.0)

        assert changed is True
        assert [(round(p.x(), 3), round(p.y(), 3)) for p in shape.points] == [
            (-1.0, 0.0),
            (7.0, 0.0),
            (7.0, 8.0),
        ]
    finally:
        window.close()


def test_set_vector_overlay_transform_rejects_zero_scale() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {"sx": 1.0, "sy": 1.0},
                }
            ]
        }

        try:
            window.setVectorOverlayTransform("overlay_a", sx=0.0)
        except ValueError as exc:
            assert "non-zero" in str(exc)
        else:
            raise AssertionError("Expected ValueError for zero overlay scale")
    finally:
        window.close()


def test_align_vector_overlay_from_landmarks_updates_overlay_metadata() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_polygon = _make_overlay_shape()
        overlay_a = _make_point("A", 0.0, 0.0, overlay=True)
        overlay_b = _make_point("B", 10.0, 0.0, overlay=True)
        overlay_c = _make_point("C", 0.0, 10.0, overlay=True)
        image_a = _make_point("A", 5.0, 7.0, overlay=False)
        image_b = _make_point("B", 15.0, 7.0, overlay=False)
        image_c = _make_point("C", 5.0, 17.0, overlay=False)
        window.canvas.shapes = [
            overlay_polygon,
            overlay_a,
            overlay_b,
            overlay_c,
            image_a,
            image_b,
            image_c,
        ]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 2.0,
                        "ty": 3.0,
                        "sx": 1.2,
                        "sy": 0.9,
                        "rotation_deg": 10.0,
                        "opacity": 0.5,
                        "visible": True,
                        "z_order": 0,
                    },
                }
            ]
        }

        pair_count = window.alignVectorOverlayFromLandmarks("overlay_a")

        assert pair_count == 3
        assert [(round(p.x(), 3), round(p.y(), 3)) for p in overlay_polygon.points] == [
            (6.0, 9.0),
            (10.0, 9.0),
            (10.0, 13.0),
        ]
        alignment = window.otherData["svg_overlays"][0]["metadata"][
            "landmark_alignment"
        ]
        assert alignment["pair_count"] == 3
        assert alignment["keys"] == [["A", ""], ["B", ""], ["C", ""]]
        assert np.allclose(
            alignment["affine_matrix"],
            [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
        )
        assert window.otherData["svg_overlays"][0]["transform"]["tx"] == 0.0
        assert window.otherData["svg_overlays"][0]["transform"]["sx"] == 1.0
        assert window.dirty_calls == 1
    finally:
        window.close()


def test_align_vector_overlay_from_landmarks_requires_three_pairs() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        window.canvas.shapes = [
            _make_point("A", 0.0, 0.0, overlay=True),
            _make_point("B", 10.0, 0.0, overlay=True),
            _make_point("A", 5.0, 7.0, overlay=False),
            _make_point("B", 15.0, 7.0, overlay=False),
        ]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }

        try:
            window.alignVectorOverlayFromLandmarks("overlay_a")
        except ValueError as exc:
            assert "at least 3" in str(exc).lower()
        else:
            raise AssertionError("Expected landmark alignment to require 3 pairs")
    finally:
        window.close()


def test_list_vector_overlays_includes_landmark_summary() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        window.canvas.shapes = [
            _make_point("A", 0.0, 0.0, overlay=True),
            _make_point("B", 10.0, 0.0, overlay=True),
            _make_point("C", 0.0, 10.0, overlay=True),
            _make_point("A", 5.0, 7.0, overlay=False),
            _make_point("B", 15.0, 7.0, overlay=False),
            _make_point("C", 5.0, 17.0, overlay=False),
        ]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }

        overlays = window.listVectorOverlays()

        assert overlays[0]["landmark_summary"]["matched_count"] == 3
        assert overlays[0]["landmark_summary"]["labels"] == ["A", "B", "C"]
    finally:
        window.close()


def test_pair_selected_vector_overlay_landmarks_creates_explicit_pair() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        window.canvas.shapes = [overlay_point, image_point]
        window.canvas.selectedShapes = [overlay_point, image_point]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }

        pair_id = window.pairSelectedVectorOverlayLandmarks("overlay_a")
        overlays = window.listVectorOverlays()

        assert pair_id.startswith("overlay_pair_")
        assert overlay_point.other_data["overlay_landmark_pair_id"] == pair_id
        assert image_point.other_data["overlay_landmark_pair_id"] == pair_id
        assert overlays[0]["landmark_summary"]["matched_count"] == 1
        assert overlays[0]["landmark_summary"]["explicit_count"] == 1
        assert overlays[0]["landmark_summary"]["auto_count"] == 0
        assert (
            overlays[0]["landmark_summary"]["explicit_pairs"][0]["pair_id"] == pair_id
        )
        assert (
            window.otherData["svg_overlays"][0]["landmark_pairs"][0]["pair_id"]
            == pair_id
        )
        assert (
            window.otherData["svg_overlays"][0]["landmark_pairs"][0]["overlay_label"]
            == "atlas_a"
        )
    finally:
        window.close()


def test_align_vector_overlay_from_explicit_pairs_ignores_label_mismatch() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_polygon = _make_overlay_shape()
        overlay_a = _make_point("atlas_1", 0.0, 0.0, overlay=True)
        overlay_b = _make_point("atlas_2", 10.0, 0.0, overlay=True)
        overlay_c = _make_point("atlas_3", 0.0, 10.0, overlay=True)
        image_a = _make_point("target_1", 5.0, 7.0, overlay=False)
        image_b = _make_point("target_2", 15.0, 7.0, overlay=False)
        image_c = _make_point("target_3", 5.0, 17.0, overlay=False)
        window.canvas.shapes = [
            overlay_polygon,
            overlay_a,
            overlay_b,
            overlay_c,
            image_a,
            image_b,
            image_c,
        ]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        for overlay_point, image_point in (
            (overlay_a, image_a),
            (overlay_b, image_b),
            (overlay_c, image_c),
        ):
            window.canvas.selectedShapes = [overlay_point, image_point]
            window.pairSelectedVectorOverlayLandmarks("overlay_a")

        pair_count = window.alignVectorOverlayFromLandmarks("overlay_a")

        assert pair_count == 3
        assert [(round(p.x(), 3), round(p.y(), 3)) for p in overlay_polygon.points] == [
            (6.0, 9.0),
            (10.0, 9.0),
            (10.0, 13.0),
        ]
        summary = window.listVectorOverlays()[0]["landmark_summary"]
        assert summary["explicit_count"] == 3
        assert summary["auto_count"] == 0
    finally:
        window.close()


def test_remove_vector_overlay_landmark_pair_updates_summary() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        window.canvas.shapes = [overlay_point, image_point]
        window.canvas.selectedShapes = [overlay_point, image_point]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        pair_id = window.pairSelectedVectorOverlayLandmarks("overlay_a")

        removed = window.removeVectorOverlayLandmarkPair("overlay_a", pair_id)
        overlays = window.listVectorOverlays()

        assert removed is True
        assert "overlay_landmark_pair_id" not in overlay_point.other_data
        assert "overlay_landmark_pair_id" not in image_point.other_data
        assert overlays[0]["landmark_summary"]["matched_count"] == 0
        assert overlays[0]["landmark_summary"]["explicit_pairs"] == []
    finally:
        window.close()


def test_clear_vector_overlay_landmark_pairs_removes_all_pairs() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_a = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_a = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_b = _make_point("atlas_b", 10.0, 0.0, overlay=True)
        image_b = _make_point("target_b", 15.0, 7.0, overlay=False)
        window.canvas.shapes = [overlay_a, image_a, overlay_b, image_b]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        for overlay_point, image_point in ((overlay_a, image_a), (overlay_b, image_b)):
            window.canvas.selectedShapes = [overlay_point, image_point]
            window.pairSelectedVectorOverlayLandmarks("overlay_a")

        removed = window.clearVectorOverlayLandmarkPairs("overlay_a")
        overlays = window.listVectorOverlays()

        assert removed == 2
        assert overlays[0]["landmark_summary"]["matched_count"] == 0
        assert overlays[0]["landmark_summary"]["explicit_pairs"] == []
    finally:
        window.close()


def test_vector_overlay_dock_reflects_selected_overlay_state() -> None:
    _ensure_qapp()

    dock = VectorOverlayDockWidget()
    try:
        dock.set_overlays(
            [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 12.0,
                        "ty": 34.0,
                        "sx": 1.25,
                        "sy": 0.75,
                        "rotation_deg": 15.0,
                        "opacity": 0.6,
                        "visible": False,
                        "z_order": 3,
                    },
                    "landmark_summary": {
                        "matched_count": 2,
                        "explicit_count": 1,
                        "auto_count": 1,
                        "labels": ["A", "B"],
                        "explicit_pairs": [
                            {
                                "pair_id": "pair_1",
                                "overlay_label": "A",
                                "image_label": "B",
                            }
                        ],
                        "pair_candidate": {"overlay_label": "C", "image_label": "D"},
                    },
                }
            ]
        )

        assert dock.overlay_list.count() == 1
        assert dock.overlay_list.item(0).text() == "atlas.svg"
        assert dock._selected_overlay_id() == "overlay_a"
        assert dock.visible_checkbox.isChecked() is False
        assert dock.opacity_slider.value() == 60
        assert abs(dock.tx_spin.value() - 12.0) < 1e-6
        assert abs(dock.sy_spin.value() - 0.75) < 1e-6
        assert dock.z_order_spin.value() == 3
        assert (
            "Matched landmarks: 2 [manual 1, label 1]"
            in dock.landmark_status_label.text()
        )
        assert "Selected pair: A -> B" in dock.landmark_status_label.text()
        assert "Pair selected ready: C -> D" in dock.landmark_status_label.text()
        assert dock.landmark_pairs_list.count() == 1
        assert dock.landmark_pairs_list.item(0).text() == "A -> B"
        assert dock.pair_selected_button.isEnabled() is True
        assert dock.remove_pair_button.isEnabled() is True
        assert dock.clear_pairs_button.isEnabled() is True
        assert dock.align_landmarks_button.isEnabled() is False
    finally:
        dock.close()


def test_vector_overlay_dock_emits_landmark_align_signal() -> None:
    _ensure_qapp()

    dock = VectorOverlayDockWidget()
    try:
        dock.set_overlays(
            [{"id": "overlay_a", "source": "/tmp/atlas.svg", "transform": {}}]
        )
        received = []
        dock.overlayLandmarkAlignRequested.connect(received.append)

        dock._emit_landmark_align_request()

        assert received == ["overlay_a"]
    finally:
        dock.close()


def test_vector_overlay_dock_enables_align_when_three_landmarks_match() -> None:
    _ensure_qapp()

    dock = VectorOverlayDockWidget()
    try:
        dock.set_overlays(
            [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "transform": {},
                    "landmark_summary": {
                        "matched_count": 3,
                        "explicit_count": 2,
                        "auto_count": 1,
                        "labels": ["A", "B", "C"],
                        "explicit_pairs": [],
                    },
                }
            ]
        )

        assert "A, B, C" in dock.landmark_status_label.text()
        assert "manual 2, label 1" in dock.landmark_status_label.text()
        assert dock.align_landmarks_button.isEnabled() is True
        assert dock.pair_selected_button.isEnabled() is False
        assert dock.remove_pair_button.isEnabled() is False
        assert dock.clear_pairs_button.isEnabled() is False
    finally:
        dock.close()


def test_vector_overlay_dock_emits_pair_selection_changed_signal() -> None:
    _ensure_qapp()

    dock = VectorOverlayDockWidget()
    try:
        dock.set_overlays(
            [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "transform": {},
                    "landmark_summary": {
                        "matched_count": 1,
                        "explicit_count": 1,
                        "auto_count": 0,
                        "labels": ["A"],
                        "explicit_pairs": [
                            {
                                "pair_id": "pair_1",
                                "overlay_label": "A",
                                "image_label": "A1",
                            }
                        ],
                    },
                }
            ]
        )
        received = []
        dock.overlayPairSelectionChanged.connect(
            lambda overlay_id, pair_id: received.append((overlay_id, pair_id))
        )

        dock.set_selected_pair("overlay_a", "pair_1")
        dock._on_current_pair_changed(dock.landmark_pairs_list.currentItem(), None)

        assert received[-1] == ("overlay_a", "pair_1")
        assert "Selected pair: A -> A1" in dock.landmark_status_label.text()
    finally:
        dock.close()


def test_pair_selection_highlight_syncs_to_canvas_and_large_image_view() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        window.canvas.shapes = [overlay_point, image_point]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }

        window._applySelectedVectorOverlayPairHighlight("overlay_a", "pair_1")

        assert window.canvas.selected_pair_id == "pair_1"
        assert window.large_image_view.selected_pair_id == "pair_1"
        assert window.canvas.selectedShapes == [overlay_point, image_point]

        window._applySelectedVectorOverlayPairHighlight("overlay_a", "missing_pair")

        assert window.canvas.selected_pair_id is None
        assert window.large_image_view.selected_pair_id is None
    finally:
        window.close()


def test_pair_selection_syncs_label_list_selection() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        window.canvas.shapes = [overlay_point, image_point]
        overlay_item = AnnolidLabelListItem("atlas_a", overlay_point)
        image_item = AnnolidLabelListItem("target_a", image_point)
        window.labelList.addItem(overlay_item)
        window.labelList.addItem(image_item)
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }

        window._applySelectedVectorOverlayPairHighlight("overlay_a", "pair_1")

        assert overlay_item.isSelected() is True
        assert image_item.isSelected() is True
    finally:
        window.close()


def test_explicit_pair_is_inferred_from_regular_shape_selection() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        window.canvas.shapes = [overlay_point, image_point]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        window.setupVectorOverlayDock()

        window.canvas.selectShapes([overlay_point, image_point])

        assert window._selected_overlay_landmark_pair_id == "pair_1"
        assert window.canvas.selected_pair_id == "pair_1"
        assert window.large_image_view.selected_pair_id == "pair_1"
        assert window.vector_overlay_dock._selected_pair_id() == "pair_1"
    finally:
        window.close()


def test_explicit_pair_is_inferred_from_single_selected_endpoint() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        window.canvas.shapes = [overlay_point, image_point]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        window.setupVectorOverlayDock()

        window.canvas.selectShapes([overlay_point])

        assert window._selected_overlay_landmark_pair_id == "pair_1"
        assert window.vector_overlay_dock._selected_pair_id() == "pair_1"
    finally:
        window.close()


def test_explicit_pair_is_inferred_when_pair_points_are_selected_with_extra_shapes() -> (
    None
):
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        extra_shape = _make_overlay_shape()
        window.canvas.shapes = [overlay_point, image_point, extra_shape]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        window.setupVectorOverlayDock()

        window.canvas.selectShapes([overlay_point, image_point, extra_shape])

        assert window._selected_overlay_landmark_pair_id == "pair_1"
        assert window.vector_overlay_dock._selected_pair_id() == "pair_1"
    finally:
        window.close()


def test_unrelated_selection_clears_active_explicit_pair() -> None:
    _ensure_qapp()

    window = _OverlayHost()
    try:
        overlay_point = _make_point("atlas_a", 0.0, 0.0, overlay=True)
        image_point = _make_point("target_a", 5.0, 7.0, overlay=False)
        overlay_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        image_point.other_data["overlay_landmark_pair_id"] = "pair_1"
        extra_shape = _make_overlay_shape()
        window.canvas.shapes = [overlay_point, image_point, extra_shape]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "metadata": {},
                    "transform": {},
                }
            ]
        }
        window.setupVectorOverlayDock()
        window.canvas.selectShapes([overlay_point, image_point])

        window.canvas.selectShapes([extra_shape])

        assert window._selected_overlay_landmark_pair_id is None
        assert window.canvas.selected_pair_id is None
        assert window.large_image_view.selected_pair_id is None
        assert window.vector_overlay_dock._selected_pair_id() is None
    finally:
        window.close()
