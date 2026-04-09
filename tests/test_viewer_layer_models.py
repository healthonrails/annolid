from __future__ import annotations

import os
from pathlib import Path

import tifffile
import numpy as np
from qtpy import QtCore, QtGui, QtTest, QtWidgets

from annolid.gui.mixins.label_image_overlay_mixin import LabelImageOverlayMixin
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.mixins.layer_dock_mixin import LayerDockMixin
from annolid.gui.mixins.shape_editing_mixin import ShapeEditingMixin
from annolid.gui.mixins.raster_layer_mixin import RasterLayerMixin
from annolid.gui.mixins.vector_overlay_mixin import VectorOverlayMixin
from annolid.gui.shape import Shape
from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.shared_vertices import insert_shared_vertex_on_edge
from annolid.gui.shared_vertices import remove_shared_vertex_at
from annolid.gui.shared_vertices import rebuild_polygon_topology
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.layer_dock import ViewerLayerDockWidget
from annolid.gui.widgets.tiled_image_view import TileKey, TiledImageView
from annolid.gui.window_base import AnnolidWindowBase
from annolid.io.large_image.tifffile_backend import TiffFileBackend


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _send_mouse_drag(view, start: QtCore.QPoint, end: QtCore.QPoint) -> None:
    viewport = view.viewport()
    global_start = viewport.mapToGlobal(start)
    global_end = viewport.mapToGlobal(end)
    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(start),
        QtCore.QPointF(global_start),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(global_end),
        QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease,
        QtCore.QPointF(end),
        QtCore.QPointF(global_end),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)


class _CanvasStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.selectedShapes = []
        self.createMode = "polygon"

    def selectShapes(self, shapes):
        self.selectedShapes = list(shapes or [])

    def deleteSelected(self):
        deleted = list(self.selectedShapes or [])
        if deleted:
            selected_ids = {id(shape) for shape in deleted}
            self.shapes = [
                shape for shape in self.shapes if id(shape) not in selected_ids
            ]
            self.selectedShapes = []
        return deleted

    def update(self):
        return None

    def setBehaviorText(self, _value):
        return None

    def editing(self):
        return True


class _LoadingStub(AnnotationLoadingMixin):
    def __init__(self):
        self._config = {"label_flags": {}}


class _ViewerLayerWindow(
    VectorOverlayMixin,
    LabelImageOverlayMixin,
    RasterLayerMixin,
    LayerDockMixin,
    AnnolidWindowBase,
):
    def __init__(self):
        super().__init__(config={"label_flags": {}, "store_data": False})
        self.canvas = _CanvasStub()
        self.large_image_view = TiledImageView(self)
        self.large_image_view.set_host_window(self)
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.addWidget(self.canvas)
        self._viewer_stack.addWidget(self.large_image_view)
        self.fileListWidget = QtWidgets.QListWidget()
        self._active_image_view = "tiled"

    def status(self, *_args, **_kwargs):
        return None

    def loadShapes(self, shapes, replace=True):
        _ = replace
        self.canvas.shapes = list(shapes or [])
        self.large_image_view.set_shapes(self.canvas.shapes)


def _polygon(label: str, points: list[tuple[float, float]], *, overlay_id: str | None):
    shape = Shape(label=label, shape_type="polygon")
    for x, y in points:
        shape.addPoint(QtCore.QPointF(float(x), float(y)))
    shape.close()
    shape.visible = True
    if overlay_id is not None:
        shape.other_data = {
            "overlay_id": overlay_id,
            "overlay_stroke": "#00ff00",
            "overlay_fill": "#004400",
            "overlay_visible": True,
            "overlay_opacity": 0.35,
            "overlay_z_order": 4,
            "overlay_source": "atlas.svg",
            "overlay_layer": "Atlas Layer",
        }
    return shape


def test_viewer_layer_models_unify_raster_label_vector_and_annotations(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base.ome.tiff"
    label_path = tmp_path / "labels.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((32, 48), dtype=np.uint16), ome=True)
    tifffile.imwrite(label_path, np.ones((32, 48), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)

        label_backend = TiffFileBackend(label_path)
        window.large_image_view._content_size = (48, 32)
        window.large_image_view.set_label_layer(
            label_backend,
            source_path=str(label_path),
            opacity=0.4,
            visible=True,
            mapping={1: {"acronym": "CTX", "name": "Cortex"}},
        )

        overlay_shape = _polygon(
            "atlas_region",
            [(1.0, 2.0), (8.0, 2.0), (8.0, 9.0)],
            overlay_id="overlay_a",
        )
        annotation_shape = _polygon(
            "manual_region",
            [(10.0, 12.0), (14.0, 12.0), (14.0, 18.0)],
            overlay_id=None,
        )
        window.canvas.shapes = [overlay_shape, annotation_shape]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "shape_count": 1,
                    "metadata": {"layer_name": "Atlas Layer"},
                    "transform": {
                        "tx": 3.0,
                        "ty": 4.0,
                        "sx": 1.2,
                        "sy": 1.2,
                        "rotation_deg": 15.0,
                        "opacity": 0.35,
                        "visible": True,
                        "z_order": 4,
                    },
                }
            ]
        }

        layers = window.viewerLayerModels()

        assert [layer.id for layer in layers] == [
            "raster_image",
            "overlay_a",
            "overlay_a_landmarks",
            "label_image_overlay",
            "annotations",
        ]

        raster_layer = next(layer for layer in layers if layer.id == "raster_image")
        label_layer = next(
            layer for layer in layers if layer.id == "label_image_overlay"
        )
        overlay_layer = next(layer for layer in layers if layer.id == "overlay_a")
        annotation_layer = next(layer for layer in layers if layer.id == "annotations")

        assert raster_layer.name == base_path.name
        assert label_layer.source_path == str(label_path)
        assert label_layer.mapping_table[1]["acronym"] == "CTX"
        assert overlay_layer.name == "Atlas Layer"
        assert overlay_layer.shape_count == 1
        assert len(overlay_layer.shapes) == 1
        assert abs(overlay_layer.transform.tx - 3.0) < 1e-6
        assert len(annotation_layer.shapes) == 1
        assert annotation_layer.shapes[0].label == "manual_region"
    finally:
        window.close()


def test_large_image_document_centralizes_tiff_viewer_state(tmp_path: Path) -> None:
    _ensure_qapp()

    base_path = tmp_path / "stack.ome.tiff"
    label_path = tmp_path / "labels.ome.tiff"
    tifffile.imwrite(
        base_path,
        np.stack(
            [np.zeros((16, 24), dtype=np.uint16), np.ones((16, 24), dtype=np.uint16)],
            axis=0,
        ),
        metadata={"axes": "QYX"},
    )
    tifffile.imwrite(label_path, np.ones((16, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_backend.set_page(1)
        window.otherData = {
            "large_image": {
                "backend_name": "tifffile",
                "optimized_cache_path": "/tmp/cache.tif",
            },
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "shape_count": 1,
                    "metadata": {"layer_name": "Atlas Layer"},
                    "transform": {
                        "tx": 1.0,
                        "ty": 2.0,
                        "sx": 1.0,
                        "sy": 1.0,
                        "rotation_deg": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                        "z_order": 2,
                    },
                }
            ],
        }

        window.large_image_view._content_size = (24, 16)
        window.large_image_view.set_backend(window.large_image_backend)
        window.large_image_view.set_zoom_percent(250)
        window.large_image_view.centerOn(12.0, 8.0)
        window.large_image_view.setEditing(False)
        label_backend = TiffFileBackend(label_path)
        window.large_image_view.set_label_layer(
            label_backend,
            source_path=str(label_path),
            opacity=0.6,
            visible=True,
            mapping={1: {"acronym": "CTX"}},
        )

        overlay_shape = _polygon(
            "atlas_region",
            [(1.0, 2.0), (8.0, 2.0), (8.0, 9.0)],
            overlay_id="overlay_a",
        )
        window.canvas.shapes = [overlay_shape]
        window.canvas.selectedShapes = [overlay_shape]
        window._selected_overlay_landmark_pair_id = "pair_a"

        document = window.currentLargeImageDocument()

        assert document is not None
        assert document.backend is window.large_image_backend
        assert document.backend_name == "tifffile"
        assert document.current_page == 1
        assert document.page_count == 2
        assert document.surface == "tiled"
        assert document.draw_mode == "polygon"
        assert document.editing is False
        assert document.viewport.zoom_percent >= 200
        assert document.active_label_layer_id == "label_image_overlay"
        assert document.label_overlay_state["source_path"] == str(label_path)
        assert document.cache_metadata["optimized_cache_path"] == "/tmp/cache.tif"
        assert document.selection.selected_overlay_id == "overlay_a"
        assert document.selection.selected_landmark_pair_id == "pair_a"
        assert document.selection.selected_shape_count == 1
        assert any(layer.id == "overlay_a" for layer in document.active_layers)
    finally:
        window.close()


def test_viewer_layer_dock_controls_label_vector_and_annotation_layers(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_layers.ome.tiff"
    label_path = tmp_path / "labels_layers.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((32, 48), dtype=np.uint16), ome=True)
    tifffile.imwrite(label_path, np.ones((32, 48), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (48, 32)
        window.large_image_view.set_backend(window.large_image_backend)
        window.large_image_view.set_label_layer(
            TiffFileBackend(label_path),
            source_path=str(label_path),
            opacity=0.4,
            visible=True,
        )

        overlay_shape = _polygon(
            "atlas_region",
            [(1.0, 2.0), (8.0, 2.0), (8.0, 9.0)],
            overlay_id="overlay_a",
        )
        annotation_shape = _polygon(
            "manual_region",
            [(10.0, 12.0), (14.0, 12.0), (14.0, 18.0)],
            overlay_id=None,
        )
        window.canvas.shapes = [overlay_shape, annotation_shape]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "atlas.svg",
                    "shape_count": 1,
                    "metadata": {"layer_name": "Atlas Layer"},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "sx": 1.0,
                        "sy": 1.0,
                        "rotation_deg": 0.0,
                        "opacity": 0.35,
                        "visible": True,
                        "z_order": 4,
                    },
                }
            ]
        }
        overlay_point = Shape(label="A", shape_type="point")
        overlay_point.addPoint(QtCore.QPointF(2.0, 2.0))
        overlay_point.other_data = {
            "overlay_id": "overlay_a",
            "overlay_landmark_pair_id": "pair_a",
            "overlay_landmarks_visible": True,
        }
        image_point = Shape(label="A", shape_type="point")
        image_point.addPoint(QtCore.QPointF(20.0, 20.0))
        image_point.other_data = {"overlay_landmark_pair_id": "pair_a"}
        window.canvas.shapes.extend([overlay_point, image_point])

        window._syncLargeImageDocument()
        dock = window._ensureViewerLayerDock()
        window.setupVectorOverlayDock()
        entries = window._viewerLayerEntries()
        assert [entry["id"] for entry in entries] == [
            "raster_image",
            "overlay_a",
            "overlay_a_landmarks",
            "label_image_overlay",
            "annotations",
        ]

        window._onViewerLayerVisibilityChanged("label_image_overlay", False)
        assert window.large_image_view.label_layer_visible() is False

        window._onViewerLayerOpacityChanged("label_image_overlay", 0.65)
        assert (
            abs(window.large_image_view.label_overlay_state()["opacity"] - 0.65) < 1e-6
        )

        window._onViewerLayerVisibilityChanged("overlay_a", False)
        assert overlay_shape.visible is False

        window._onViewerLayerOpacityChanged("overlay_a", 0.75)
        overlay_record = window.otherData["svg_overlays"][0]
        assert abs(float(overlay_record["transform"]["opacity"]) - 0.75) < 1e-6

        window._onViewerLayerVisibilityChanged("overlay_a_landmarks", False)
        assert window.vectorOverlayLandmarkLayers()[0].visible is False
        assert overlay_point.other_data["overlay_landmarks_visible"] is False

        window._onViewerLayerSelected("overlay_a_landmarks")
        assert window.vector_overlay_dock._selected_overlay_id() == "overlay_a"

        window._onViewerLayerVisibilityChanged("annotations", False)
        assert annotation_shape.visible is False

        dock_layers = [
            dock.layer_list.item(i).data(QtCore.Qt.UserRole)
            for i in range(dock.layer_list.count())
        ]
        assert "label_image_overlay" in dock_layers
        assert "overlay_a" in dock_layers
        assert "annotations" in dock_layers
    finally:
        window.close()


def test_viewer_layer_dock_controls_raster_overlay_layer_visibility(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_raster_layers.ome.tiff"
    overlay_path = tmp_path / "overlay_raster_layers.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_test",
                    "name": "overlay layer",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 12,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()
        entries = window._viewerLayerEntries()
        assert any(entry["id"] == "raster_overlay_test" for entry in entries)
        overlay_entry = next(
            entry for entry in entries if entry["id"] == "raster_overlay_test"
        )
        assert overlay_entry["checkable"] is True
        assert overlay_entry["visible"] is True
        assert overlay_entry["supports_translate"] is True

        changed = window.setRasterImageLayerVisible("raster_overlay_test", False)
        assert changed is True
        state_after = window.large_image_view.raster_overlay_layers_state()
        assert state_after[0]["visible"] is False
        assert window.otherData["raster_image_layers"][0]["visible"] is False
    finally:
        window.close()


def test_viewer_layer_dock_move_down_keeps_raster_overlay_visible(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_move_down.ome.tiff"
    overlay_path = tmp_path / "overlay_move_down.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_move",
                    "name": "overlay move",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()
        runtime_before = window.large_image_view._raster_overlay_layers[
            "raster_overlay_move"
        ]
        backend_before = runtime_before.backend

        before_layers = {
            layer.id: layer.z_index for layer in window.viewerLayerModels()
        }
        changed = window.moveRasterImageLayer("raster_overlay_move", 1)
        assert changed is True
        after_layers = {layer.id: layer.z_index for layer in window.viewerLayerModels()}
        assert after_layers["raster_overlay_move"] > before_layers["raster_image"]
        assert (
            window.large_image_view.raster_overlay_layers_state()[0]["visible"] is True
        )
        runtime_after = window.large_image_view._raster_overlay_layers[
            "raster_overlay_move"
        ]
        assert runtime_after is runtime_before
        assert runtime_after.backend is backend_before
    finally:
        window.close()


def test_viewer_layer_dock_allows_toggling_base_and_overlay_raster_layers(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "65449_001.tif"
    overlay_path = tmp_path / "65463_001.tif"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_65463",
                    "name": "65463_001.tif",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()

        entries = {entry["id"]: entry for entry in window._viewerLayerEntries()}
        assert entries["raster_image"]["checkable"] is True
        assert entries["raster_image"]["visible"] is True
        assert entries["raster_overlay_65463"]["visible"] is True

        window._onViewerLayerVisibilityChanged("raster_image", False)
        assert window.large_image_view.base_raster_visible() is False
        assert window.currentRasterImageLayer().visible is False

        window._onViewerLayerVisibilityChanged("raster_overlay_65463", False)
        overlay_state = window.large_image_view.raster_overlay_layers_state()
        assert overlay_state[0]["visible"] is False

        window._onViewerLayerVisibilityChanged("raster_overlay_65463", True)
        window._onViewerLayerVisibilityChanged("raster_image", True)
        assert window.large_image_view.base_raster_visible() is True
        overlay_state = window.large_image_view.raster_overlay_layers_state()
        assert overlay_state[0]["visible"] is True
    finally:
        window.close()


def test_raster_overlay_transform_persists_and_updates_runtime_state(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_transform.ome.tiff"
    overlay_path = tmp_path / "overlay_transform.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_tx",
                    "name": "overlay tx",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        changed = window.setRasterImageLayerTransform(
            "raster_overlay_tx", tx=3.5, ty=-2.0, sx=1.1, sy=0.9
        )

        assert changed is True

        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        record = state["raster_overlay_tx"]
        assert abs(float(record["tx"]) - 3.5) < 1e-6
        assert abs(float(record["ty"]) - (-2.0)) < 1e-6
        assert abs(float(record["sx"]) - 1.1) < 1e-6
        assert abs(float(record["sy"]) - 0.9) < 1e-6

        stored = {
            str(item.get("id") or ""): item
            for item in list(window.otherData.get("raster_image_layers") or [])
        }
        assert abs(float(stored["raster_overlay_tx"]["tx"]) - 3.5) < 1e-6
        assert abs(float(stored["raster_overlay_tx"]["ty"]) - (-2.0)) < 1e-6
        assert abs(float(stored["raster_overlay_tx"]["sx"]) - 1.1) < 1e-6
        assert abs(float(stored["raster_overlay_tx"]["sy"]) - 0.9) < 1e-6

        model = next(
            layer
            for layer in window.viewerLayerModels()
            if str(layer.id) == "raster_overlay_tx"
        )
        assert abs(float(model.transform.tx) - 3.5) < 1e-6
        assert abs(float(model.transform.ty) - (-2.0)) < 1e-6
        assert abs(float(model.transform.sx) - 1.1) < 1e-6
        assert abs(float(model.transform.sy) - 0.9) < 1e-6
    finally:
        window.close()


def test_viewer_layer_dock_nudge_translates_raster_overlay_layer(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_nudge.ome.tiff"
    overlay_path = tmp_path / "overlay_nudge.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_nudge",
                    "name": "overlay nudge",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                    "tx": 0.0,
                    "ty": 0.0,
                    "sx": 1.0,
                    "sy": 1.0,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()
        window._onViewerLayerSelected("raster_overlay_nudge")
        window._onViewerLayerTranslateRequested("raster_overlay_nudge", 4.0, -3.0)

        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        record = state["raster_overlay_nudge"]
        assert abs(float(record["tx"]) - 4.0) < 1e-6
        assert abs(float(record["ty"]) - (-3.0)) < 1e-6

        stored = {
            str(item.get("id") or ""): item
            for item in list(window.otherData.get("raster_image_layers") or [])
        }
        assert abs(float(stored["raster_overlay_nudge"]["tx"]) - 4.0) < 1e-6
        assert abs(float(stored["raster_overlay_nudge"]["ty"]) - (-3.0)) < 1e-6
    finally:
        window.close()


def test_raster_overlay_opacity_keeps_reference_layer_visible_for_alignment(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_alignment_reference.ome.tiff"
    lower_path = tmp_path / "lower_alignment_reference.ome.tiff"
    upper_path = tmp_path / "upper_alignment_reference.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(lower_path, np.ones((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(upper_path, np.full((24, 24), 2, dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_lower",
                    "name": "lower",
                    "source_path": str(lower_path),
                    "visible": False,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                },
                {
                    "id": "raster_overlay_upper",
                    "name": "upper",
                    "source_path": str(upper_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 11,
                },
            ]
        }
        window._restoreRasterImageLayersFromState()
        changed = window.setRasterImageLayerOpacity("raster_overlay_upper", 0.5)

        assert changed is True
        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        assert state["raster_overlay_upper"]["visible"] is True
        assert state["raster_overlay_lower"]["visible"] is True
    finally:
        window.close()


def test_raster_overlay_transform_updates_existing_tile_item_in_place(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_transform_in_place.ome.tiff"
    overlay_path = tmp_path / "overlay_transform_in_place.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_in_place",
                    "name": "overlay in place",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        runtime = window.large_image_view._raster_overlay_layers[
            "raster_overlay_in_place"
        ]
        key = TileKey(level=0, tx=0, ty=0)
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(8, 8))
        runtime.tile_items[key] = item
        window.large_image_view._scene.addItem(item)

        changed = window.setRasterImageLayerTransform(
            "raster_overlay_in_place", tx=5.0, ty=3.0, sx=1.0, sy=1.0
        )

        assert changed is True
        assert runtime.tile_items[key] is item
        assert abs(float(item.pos().x()) - 5.0) < 1e-6
        assert abs(float(item.pos().y()) - 3.0) < 1e-6
    finally:
        window.close()


def test_viewer_layer_dock_reset_alignment_clears_raster_overlay_transform(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_reset.ome.tiff"
    overlay_path = tmp_path / "overlay_reset.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_reset",
                    "name": "overlay reset",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                    "tx": 7.0,
                    "ty": -5.0,
                    "sx": 1.25,
                    "sy": 0.8,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()
        window._onViewerLayerSelected("raster_overlay_reset")
        window._onViewerLayerResetTransformRequested("raster_overlay_reset")

        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        record = state["raster_overlay_reset"]
        assert abs(float(record["tx"]) - 0.0) < 1e-6
        assert abs(float(record["ty"]) - 0.0) < 1e-6
        assert abs(float(record["sx"]) - 1.0) < 1e-6
        assert abs(float(record["sy"]) - 1.0) < 1e-6

        stored = {
            str(item.get("id") or ""): item
            for item in list(window.otherData.get("raster_image_layers") or [])
        }
        assert abs(float(stored["raster_overlay_reset"]["tx"]) - 0.0) < 1e-6
        assert abs(float(stored["raster_overlay_reset"]["ty"]) - 0.0) < 1e-6
        assert abs(float(stored["raster_overlay_reset"]["sx"]) - 1.0) < 1e-6
        assert abs(float(stored["raster_overlay_reset"]["sy"]) - 1.0) < 1e-6
    finally:
        window.close()


def test_viewer_layer_dock_keyboard_shortcuts_nudge_and_reset_raster_overlay(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_shortcuts.ome.tiff"
    overlay_path = tmp_path / "overlay_shortcuts.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    dock = ViewerLayerDockWidget()
    try:
        dock.set_layers(
            [
                {
                    "id": "raster_overlay_shortcut",
                    "name": "overlay shortcut",
                    "visible": True,
                    "opacity": 1.0,
                    "supports_opacity": True,
                    "supports_translate": True,
                    "supports_reorder": True,
                    "checkable": True,
                    "details": "Raster overlay image | page 1",
                }
            ]
        )

        assert "Alt+Arrows" in dock.alignment_hint_label.text()

        translated: list[tuple[str, float, float]] = []
        resets: list[str] = []
        dock.layerTranslateRequested.connect(
            lambda layer_id, dx, dy: translated.append((layer_id, dx, dy))
        )
        dock.layerResetTransformRequested.connect(
            lambda layer_id: resets.append(layer_id)
        )

        dock.show()
        dock.layer_list.setCurrentRow(0)
        dock.layer_list.setFocus()
        QtWidgets.QApplication.processEvents()

        QtTest.QTest.keyClick(dock, QtCore.Qt.Key_Left, QtCore.Qt.AltModifier)
        QtTest.QTest.keyClick(dock, QtCore.Qt.Key_0, QtCore.Qt.AltModifier)
        QtWidgets.QApplication.processEvents()

        assert translated == [("raster_overlay_shortcut", -1.0, 0.0)]
        assert resets == ["raster_overlay_shortcut"]
    finally:
        dock.close()


def test_selected_raster_overlay_layer_can_be_dragged_in_tiled_view(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_drag.ome.tiff"
    overlay_path = tmp_path / "overlay_drag.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_drag",
                    "name": "overlay drag",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._selected_viewer_layer_id = "raster_overlay_drag"
        assert window.large_image_view._start_raster_overlay_drag(
            "raster_overlay_drag", QtCore.QPointF(30.0, 30.0)
        )
        assert window.large_image_view._apply_raster_overlay_drag(
            QtCore.QPointF(90.0, 55.0)
        )
        window.large_image_view._end_raster_overlay_drag()

        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        record = state["raster_overlay_drag"]
        assert abs(float(record["tx"])) > 0.01
        assert abs(float(record["ty"])) > 0.01
    finally:
        window.close()


def test_hidden_base_tiff_still_renders_checked_overlay_at_low_zoom(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "65449_001.tif"
    overlay_path = tmp_path / "65463_001.tif"
    tifffile.imwrite(base_path, np.zeros((1024, 1024), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((1024, 1024), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (1024, 1024)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "65463_001.tif",
                    "name": "65463_001.tif",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        window.large_image_view.set_zoom_percent(5)
        window.setBaseRasterImageVisible(False)
        window.large_image_view.refresh_visible_tiles()

        runtime = window.large_image_view._raster_overlay_layers["65463_001.tif"]
        assert runtime.visible is True
        assert len(runtime.current_visible_keys) > 0
        assert runtime.last_visible_tile_count > 0
    finally:
        window.close()


def test_viewer_layer_dock_supports_non_prefixed_raster_overlay_ids(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "65449_001.tif"
    overlay_path = tmp_path / "65463_001.tif"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(overlay_path, np.ones((24, 24), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "65463_001.tif",
                    "name": "65463_001.tif",
                    "source_path": str(overlay_path),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                }
            ]
        }
        window._restoreRasterImageLayersFromState()
        entries = {entry["id"]: entry for entry in window._viewerLayerEntries()}
        assert entries["65463_001.tif"]["supports_reorder"] is True
        assert entries["65463_001.tif"]["supports_opacity"] is True

        changed_visibility = window.setRasterImageLayerVisible("65463_001.tif", False)
        assert changed_visibility is True
        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        assert state["65463_001.tif"]["visible"] is False

        before_layers = {
            layer.id: layer.z_index for layer in window.viewerLayerModels()
        }
        changed_move = window.moveRasterImageLayer("65463_001.tif", -1)
        assert changed_move is True
        after_layers = {layer.id: layer.z_index for layer in window.viewerLayerModels()}
        assert after_layers["65463_001.tif"] > before_layers["annotations"]

        changed_opacity = window.setRasterImageLayerOpacity("65463_001.tif", 0.4)
        assert changed_opacity is True
        state = {
            str(item["id"]): item
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        assert abs(float(state["65463_001.tif"]["opacity"]) - 0.4) < 1e-6
    finally:
        window.close()


def test_viewer_layer_dock_controls_raster_overlay_opacity_and_reorder(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_reorder.ome.tiff"
    layer_a = tmp_path / "overlay_a.ome.tiff"
    layer_b = tmp_path / "overlay_b.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_a, np.ones((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_b, np.full((24, 24), 2, dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        annotation_shape = _polygon(
            "manual_region",
            [(2.0, 2.0), (7.0, 2.0), (7.0, 7.0)],
            overlay_id=None,
        )
        window.canvas.shapes = [annotation_shape]
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_a",
                    "name": "layer a",
                    "source_path": str(layer_a),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                },
                {
                    "id": "raster_overlay_b",
                    "name": "layer b",
                    "source_path": str(layer_b),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 11,
                },
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()

        changed_opacity = window._onViewerLayerOpacityChanged("raster_overlay_b", 0.33)
        _ = changed_opacity
        state = window.large_image_view.raster_overlay_layers_state()
        by_id = {str(item["id"]): item for item in state}
        assert abs(float(by_id["raster_overlay_b"]["opacity"]) - 0.33) < 1e-6

        before_layers = {
            layer.id: layer.z_index for layer in window.viewerLayerModels()
        }
        changed_move = window.moveRasterImageLayer("raster_overlay_b", -1)
        assert changed_move is True
        after_layers = {layer.id: layer.z_index for layer in window.viewerLayerModels()}
        assert after_layers["raster_overlay_b"] > before_layers["annotations"]

        changed_move_back = window.moveRasterImageLayer("raster_overlay_b", 1)
        assert changed_move_back is True
        restored_layers = {
            layer.id: layer.z_index for layer in window.viewerLayerModels()
        }
        assert restored_layers["raster_overlay_b"] < restored_layers["annotations"]
    finally:
        window.close()


def test_viewer_layer_dock_context_actions_for_raster_overlays(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_context.ome.tiff"
    layer_a = tmp_path / "ctx_overlay_a.ome.tiff"
    layer_b = tmp_path / "ctx_overlay_b.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_a, np.ones((24, 24), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_b, np.full((24, 24), 3, dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.imagePath = str(base_path)
        window.large_image_backend = TiffFileBackend(base_path)
        window.large_image_view._content_size = (24, 24)
        window.large_image_view.set_backend(window.large_image_backend)
        window.otherData = {
            "raster_image_layers": [
                {
                    "id": "raster_overlay_a",
                    "name": "layer a",
                    "source_path": str(layer_a),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10,
                },
                {
                    "id": "raster_overlay_b",
                    "name": "layer b",
                    "source_path": str(layer_b),
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 11,
                },
            ]
        }
        window._restoreRasterImageLayersFromState()
        window._syncLargeImageDocument()

        window._onViewerLayerRenameRequested("raster_overlay_a", "renamed a")
        renamed = {
            str(item["id"]): str(item["name"])
            for item in window.large_image_view.raster_overlay_layers_state()
        }
        assert renamed["raster_overlay_a"] == "renamed a"

        window._onViewerLayerMoveToTopRequested("raster_overlay_b")
        top_layers = {layer.id: layer.z_index for layer in window.viewerLayerModels()}
        assert top_layers["raster_overlay_b"] > top_layers["annotations"]

        window._onViewerLayerMoveToBottomRequested("raster_overlay_b")
        bottom_layers = {
            layer.id: layer.z_index for layer in window.viewerLayerModels()
        }
        assert bottom_layers["raster_overlay_b"] < bottom_layers["raster_overlay_a"]

        window._onViewerLayerRemoveRequested("raster_overlay_a")
        removed_state = window.large_image_view.raster_overlay_layers_state()
        assert [str(item["id"]) for item in removed_state] == ["raster_overlay_b"]
        assert all(
            str(item.get("id") or "") != "raster_overlay_a"
            for item in list(window.otherData.get("raster_image_layers") or [])
        )
    finally:
        window.close()


def test_open_file_multiselect_large_tiffs_loads_additional_layers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_multi.ome.tiff"
    layer_a = tmp_path / "layer_a.ome.tiff"
    layer_b = tmp_path / "layer_b.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((20, 20), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_a, np.ones((20, 20), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_b, np.full((20, 20), 2, dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileNames",
            staticmethod(
                lambda *args, **kwargs: (
                    [str(base_path), str(layer_a), str(layer_b)],
                    "",
                )
            ),
        )
        window.openFile()
        state = window.large_image_view.raster_overlay_layers_state()
        sources = {str(item.get("source_path") or "") for item in state}
        assert str(layer_a.resolve()) in sources
        assert str(layer_b.resolve()) in sources
        entries = window._viewerLayerEntries()
        assert any(str(entry["id"]).startswith("raster_overlay_") for entry in entries)
    finally:
        window.close()


def test_open_file_single_large_tiff_prompts_add_layer_when_tiled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    base_path = tmp_path / "base_single.ome.tiff"
    layer_path = tmp_path / "single_layer.ome.tiff"
    tifffile.imwrite(base_path, np.zeros((18, 18), dtype=np.uint16), ome=True)
    tifffile.imwrite(layer_path, np.ones((18, 18), dtype=np.uint16), ome=True)

    window = _ViewerLayerWindow()
    try:
        window.loadFile(str(base_path))
        assert window.imagePath == str(base_path)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileNames",
            staticmethod(lambda *args, **kwargs: ([str(layer_path)], "")),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            staticmethod(lambda *args, **kwargs: QtWidgets.QMessageBox.Yes),
        )
        window.openFile()
        assert window.imagePath == str(base_path)
        state = window.large_image_view.raster_overlay_layers_state()
        assert any(
            str(item.get("source_path") or "") == str(layer_path.resolve())
            for item in state
        )
    finally:
        window.close()


def test_shape_adjoining_polygon_seed_reuses_edge_coordinates() -> None:
    shape = Shape(label="brain_region_a", shape_type="polygon")
    for point in [(1.0, 2.0), (6.0, 2.0), (6.0, 7.0), (1.0, 7.0)]:
        shape.addPoint(QtCore.QPointF(*point))
    shape.close()

    seed = shape.adjoining_polygon_seed(1)

    assert seed is not None
    assert seed.shape_type == "polygon"
    assert seed.label is None
    assert not seed.isClosed()
    assert len(seed.points) == 2
    assert (seed.points[0].x(), seed.points[0].y()) == (1.0, 2.0)
    assert (seed.points[1].x(), seed.points[1].y()) == (6.0, 2.0)
    assert seed.points[0] is not shape.points[0]
    assert seed.points[1] is not shape.points[1]


def test_canvas_start_adjoining_polygon_from_selection_uses_seeded_shared_vertices() -> (
    None
):
    _ensure_qapp()

    canvas = Canvas()
    try:
        shape = Shape(shape_type="polygon")
        for point in [(2.0, 2.0), (9.0, 2.0), (9.0, 8.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        canvas.shapes = [shape]
        canvas.selectedShapes = [shape]
        canvas.hShape = shape
        canvas.hEdge = 1

        assert canvas.startAdjoiningPolygonFromSelection()
        assert canvas.current is not None
        assert len(canvas.current.points) == 2
        assert canvas.current.shared_vertex_id(0) == shape.shared_vertex_id(0)
        assert canvas.current.shared_vertex_id(1) == shape.shared_vertex_id(1)
        assert canvas.current.points[0].x() == shape.points[0].x()
        assert canvas.current.points[0].y() == shape.points[0].y()
    finally:
        canvas.close()


class _BoundaryEditorStub:
    def __init__(self, seed):
        self.seed = seed
        self.start_calls = []

    def startAdjoiningPolygonFromSelection(self, edge_index=None):
        self.start_calls.append(edge_index)
        return True


class _BoundaryHostStub(ShapeEditingMixin):
    def __init__(self, editor):
        self.editor = editor
        self.toggle_calls = []

    def _active_shape_editor(self):
        return self.editor

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        self.toggle_calls.append((edit, createMode))


class _BoundaryEditorNoAdjoiningStub:
    def canStartAdjoiningPolygon(self):
        return False

    def adjoiningPolygonSeed(self, edge_index=None):
        return None

    def startAdjoiningPolygonFromSelection(self, edge_index=None):
        _ = edge_index
        return False


class _ContextActionsStub:
    def __init__(self) -> None:
        self.editMode = QtWidgets.QAction("Edit")
        self.createMode = QtWidgets.QAction("Create Polygon")
        self.createRectangleMode = QtWidgets.QAction("Create Rectangle")
        self.createCircleMode = QtWidgets.QAction("Create Circle")
        self.createLineMode = QtWidgets.QAction("Create Line")
        self.createPointMode = QtWidgets.QAction("Create Point")
        self.createLineStripMode = QtWidgets.QAction("Create Line Strip")
        self.createAiPolygonMode = QtWidgets.QAction("AI Polygon")
        self.createAiMaskMode = QtWidgets.QAction("AI Mask")
        self.createGroundingSAMMode = QtWidgets.QAction("Grounding SAM")
        self.duplicateShapes = QtWidgets.QAction("Duplicate")
        self.startAdjoiningPolygon = QtWidgets.QAction("Start Adjoining Polygon")
        self.deleteShapes = QtWidgets.QAction("Delete")


class _ContextMenuWindowStub:
    def __init__(self, editor) -> None:
        self.actions = _ContextActionsStub()
        self._editor = editor

    def _active_shape_editor(self):
        return self._editor

    def startAdjoiningPolygonFromSelection(self, edge_index=None):
        _ = edge_index
        return None


def test_shape_editing_mixin_starts_adjoining_polygon_from_active_editor() -> None:
    seed = Shape(shape_type="polygon")
    for point in [(3.0, 4.0), (8.0, 4.0)]:
        seed.addPoint(QtCore.QPointF(*point))
    host = _BoundaryHostStub(_BoundaryEditorStub(seed))

    host.startAdjoiningPolygonFromSelection()

    assert host.toggle_calls == [(False, "polygon")]
    assert host.editor.start_calls == [None]


def test_shape_editing_mixin_falls_back_to_tiled_editor_for_adjoining_polygon() -> None:
    seed = Shape(shape_type="polygon")
    for point in [(1.0, 1.0), (2.0, 1.0)]:
        seed.addPoint(QtCore.QPointF(*point))
    tiled_editor = _BoundaryEditorStub(seed)

    class _Host(ShapeEditingMixin):
        def __init__(self):
            self.editor = _BoundaryEditorNoAdjoiningStub()
            self.large_image_view = tiled_editor
            self.toggle_calls = []

        def _active_shape_editor(self):
            return self.editor

        def toggleDrawMode(self, edit=True, createMode="polygon"):
            self.toggle_calls.append((edit, createMode))

    host = _Host()
    host.startAdjoiningPolygonFromSelection()

    assert host.toggle_calls == [(False, "polygon")]
    assert tiled_editor.start_calls == [None]


def test_shape_editing_mixin_starts_adjoining_polygon_even_with_edge_index_arg() -> (
    None
):
    seed = Shape(shape_type="polygon")
    for point in [(4.0, 4.0), (7.0, 4.0)]:
        seed.addPoint(QtCore.QPointF(*point))
    editor = _BoundaryEditorStub(seed)

    class _Host(ShapeEditingMixin):
        def __init__(self):
            self.editor = editor
            self.large_image_view = _BoundaryEditorNoAdjoiningStub()
            self.toggle_calls = []

        def _active_shape_editor(self):
            return self.editor

        def toggleDrawMode(self, edit=True, createMode="polygon"):
            self.toggle_calls.append((edit, createMode))

    host = _Host()
    host.startAdjoiningPolygonFromSelection(edge_index=2)

    assert host.toggle_calls == [(False, "polygon")]
    assert editor.start_calls == [2]


def test_shape_editing_mixin_switches_to_polygon_mode_before_seed_check() -> None:
    class _Host(ShapeEditingMixin):
        def __init__(self):
            self.editor = _BoundaryEditorNoAdjoiningStub()
            self.large_image_view = _BoundaryEditorNoAdjoiningStub()
            self.toggle_calls = []

        def _active_shape_editor(self):
            return self.editor

        def toggleDrawMode(self, edit=True, createMode="polygon"):
            self.toggle_calls.append((edit, createMode))

    host = _Host()
    host.startAdjoiningPolygonFromSelection()

    assert host.toggle_calls == [(False, "polygon")]


def test_canvas_context_menu_shows_adjoining_polygon_for_explicit_edge_only() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        shape = Shape(shape_type="polygon")
        for point in [(1.0, 1.0), (5.0, 1.0), (5.0, 4.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        canvas.shapes = [shape]
        canvas.selectedShapes = []
        canvas.hShape = shape
        canvas.hEdge = None

        window = _ContextMenuWindowStub(canvas)
        menu = canvas._build_context_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert "Start Adjoining Polygon" not in texts

        canvas.selectedShapes = [shape]
        menu = canvas._build_context_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert "Start Adjoining Polygon" in texts
        adjoining = next(
            action
            for action in menu.actions()
            if action.text() == "Start Adjoining Polygon"
        )
        assert adjoining.isEnabled()
    finally:
        canvas.close()


def test_canvas_context_menu_uses_tiled_adjoining_availability() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:

        class _TiledEditor:
            @staticmethod
            def canStartAdjoiningPolygon():
                return True

        class _Window:
            def __init__(self, editor):
                self.actions = _ContextActionsStub()
                self._editor = editor
                self.large_image_view = _TiledEditor()

            def _active_shape_editor(self):
                return self._editor

        window = _Window(_BoundaryEditorNoAdjoiningStub())
        menu = canvas._build_context_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert "Start Adjoining Polygon" in texts
        adjoining = next(
            action
            for action in menu.actions()
            if action.text() == "Start Adjoining Polygon"
        )
        assert adjoining.isEnabled()
    finally:
        canvas.close()


def test_canvas_context_menu_shows_adjoining_polygon_when_shape_selected() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        shape = Shape(shape_type="polygon")
        for point in [(1.0, 1.0), (5.0, 1.0), (5.0, 4.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        canvas.shapes = [shape]
        canvas.selectedShapes = [shape]
        canvas.hShape = shape
        canvas.hEdge = None

        window = _ContextMenuWindowStub(canvas)
        menu = canvas._build_context_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert "Start Adjoining Polygon" in texts
        adjoining = next(
            action
            for action in menu.actions()
            if action.text() == "Start Adjoining Polygon"
        )
        assert adjoining.isEnabled()
    finally:
        canvas.close()


def test_canvas_context_menu_shows_shared_boundary_reshape_for_selected_shared_polygon() -> (
    None
):
    _ensure_qapp()

    canvas = Canvas()
    try:
        left = Shape(label="left", shape_type="polygon")
        for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
            left.addPoint(QtCore.QPointF(*point))
        left.close()

        right = Shape(label="right", shape_type="polygon")
        for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
            right.addPoint(QtCore.QPointF(*point))
        right.close()

        left.set_shared_vertex_id(1, "shared-a")
        left.set_shared_vertex_id(2, "shared-b")
        right.set_shared_vertex_id(0, "shared-a")
        right.set_shared_vertex_id(3, "shared-b")
        rebuild_polygon_topology([left, right])

        canvas.shapes = [left, right]
        canvas._shared_topology_registry = SharedTopologyRegistry.from_shapes(
            canvas.shapes
        )
        canvas.selectedShapes = [left]

        window = _ContextMenuWindowStub(canvas)
        menu = canvas._build_context_menu(window)
        texts = [action.text() for action in menu.actions()]
        assert "Reshape Shared Boundary" in texts
    finally:
        canvas.close()


def test_tiled_image_view_uses_draw_cursor_for_adjoining_polygon_seed() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        seed = Shape(shape_type="polygon")
        for point in [(2.0, 2.0), (9.0, 2.0)]:
            seed.addPoint(QtCore.QPointF(*point))

        assert view.beginAdjoiningPolygonFromSeed(seed)
        assert view._cursor == QtCore.Qt.CrossCursor

        view.restoreCursor()
        assert view._cursor == QtCore.Qt.CrossCursor

        view.setEditing(True)
        assert view.mode == view.EDIT
    finally:
        view.close()


def test_tiled_image_view_shows_adjoining_polygon_for_selected_polygon() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        shape = Shape(shape_type="polygon")
        for point in [(2.0, 2.0), (9.0, 2.0), (9.0, 8.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        view.set_shapes([shape])
        view.selectedShapes = [shape]

        assert view.canStartAdjoiningPolygon()
        assert view.startAdjoiningPolygonFromSelection()
        assert view.current is not None
        assert view._adjoining_source_shape is shape
        assert view.createMode == "polygon"
        assert len(view.current.points) == 3
        assert view.current.shared_vertex_id(0) == shape.shared_vertex_id(0)
        assert view.current.shared_vertex_id(1) == shape.shared_vertex_id(1)
        assert view.current.points[0].x() == shape.points[0].x()
        assert view.current.points[0].y() == shape.points[0].y()
    finally:
        view.close()


def test_tiled_image_view_can_start_shared_boundary_reshape_from_selected_shared_polygon() -> (
    None
):
    _ensure_qapp()

    view = TiledImageView()
    try:
        left = Shape(label="left", shape_type="polygon")
        for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
            left.addPoint(QtCore.QPointF(*point))
        left.close()

        right = Shape(label="right", shape_type="polygon")
        for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
            right.addPoint(QtCore.QPointF(*point))
        right.close()

        left.set_shared_vertex_id(1, "shared-a")
        left.set_shared_vertex_id(2, "shared-b")
        right.set_shared_vertex_id(0, "shared-a")
        right.set_shared_vertex_id(3, "shared-b")
        rebuild_polygon_topology([left, right])

        view.set_shapes([left, right])
        view.selectedShapes = [left]

        assert view.canStartSharedBoundaryReshape()
        assert view.startSharedBoundaryReshape()
        assert view._shared_boundary_shape is left
        assert view._shared_boundary_edge_index is None
        assert view._shared_boundary_reshape_mode is True
    finally:
        view.close()


def test_tiled_image_view_can_start_adjoining_polygon_from_canvas_selection() -> None:
    _ensure_qapp()

    window = _ViewerLayerWindow()
    try:
        shape = Shape(shape_type="polygon")
        for point in [(12.0, 12.0), (24.0, 12.0), (24.0, 24.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        window.canvas.shapes = [shape]
        window.canvas.selectedShapes = [shape]
        window.large_image_view.set_shapes(window.canvas.shapes)
        window.large_image_view.selectedShapes = []
        window._active_image_view = "tiled"

        assert window.large_image_view.canStartAdjoiningPolygon()
        assert window.large_image_view.startAdjoiningPolygonFromSelection()
        assert window.large_image_view._adjoining_source_shape is shape
        assert window.large_image_view.createMode == "polygon"
        assert window.large_image_view.current is not None
        assert len(window.large_image_view.current.points) == 3
        assert window.large_image_view.current.shared_vertex_id(
            0
        ) == shape.shared_vertex_id(0)
    finally:
        window.close()


def test_tiled_image_view_shows_draw_cursor_before_first_polygon_click() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)
        view.setEditing(False)
        view.createMode = "polygon"

        hover_pos = QtCore.QPoint(40, 40)
        move_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(hover_pos),
            QtCore.QPointF(view.viewport().mapToGlobal(hover_pos)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(move_event)

        assert view._cursor == QtCore.Qt.CrossCursor
        assert view.toolTip() == "Click to create point"
    finally:
        view.close()


def test_tiled_image_view_hover_feedback_matches_polygon_edit_targets() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)

        shape = Shape(label="region_a", shape_type="polygon")
        for point in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]:
            shape.addPoint(QtCore.QPointF(*point))
        shape.close()
        view.set_shapes([shape])
        view.setEditing(True)

        vertex_pos = view.mapFromScene(QtCore.QPointF(20.0, 20.0))
        vertex_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(vertex_pos),
            QtCore.QPointF(view.viewport().mapToGlobal(vertex_pos)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(vertex_event)
        assert view._cursor == QtCore.Qt.PointingHandCursor
        assert view.toolTip() == "Click & drag to move point"

        edge_pos = view.mapFromScene(QtCore.QPointF(50.0, 20.0))
        edge_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(edge_pos),
            QtCore.QPointF(view.viewport().mapToGlobal(edge_pos)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(edge_event)
        assert view._cursor == QtCore.Qt.PointingHandCursor
        assert view.toolTip() == "Click to create point"

        body_pos = view.mapFromScene(QtCore.QPointF(50.0, 50.0))
        body_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(body_pos),
            QtCore.QPointF(view.viewport().mapToGlobal(body_pos)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(body_event)
        assert view._cursor == QtCore.Qt.OpenHandCursor
        assert view.toolTip() == "Click & drag to move shape 'region_a'"
    finally:
        view.close()


def test_tiled_image_view_snaps_new_polygon_point_to_existing_vertex() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)

        existing = Shape(label="region_b", shape_type="polygon")
        for point in [(40.0, 40.0), (80.0, 20.0), (120.0, 40.0), (80.0, 80.0)]:
            existing.addPoint(QtCore.QPointF(*point))
        existing.close()
        view.set_shapes([existing])

        view.setEditing(False)
        view.createMode = "polygon"

        def _left_press(scene_point: QtCore.QPointF) -> None:
            local = view.mapFromScene(scene_point)
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(local),
                QtCore.QPointF(view.viewport().mapToGlobal(local)),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
            view.mousePressEvent(event)

        _left_press(QtCore.QPointF(10.0, 10.0))
        _left_press(QtCore.QPointF(79.3, 20.4))

        assert view.current is not None
        assert len(view.current.points) >= 2
        snapped = view.current.points[1]
        assert snapped.x() == 80.0
        assert snapped.y() == 20.0
    finally:
        view.close()


def test_tiled_image_view_polygon_preview_uses_fill_brush_when_enabled() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)
        view.setEditing(False)
        view.createMode = "polygon"

        shape = Shape(shape_type="polygon")
        shape.fill = True
        shape.addPoint(QtCore.QPointF(20.0, 20.0))
        shape.addPoint(QtCore.QPointF(80.0, 20.0))
        view.current = shape

        view._update_drawing_preview(QtCore.QPointF(80.0, 80.0))

        assert view._preview_item.brush().style() != QtCore.Qt.NoBrush
    finally:
        view.close()


def test_tiled_image_view_annotation_visibility_updates_items_without_rebuild() -> None:
    _ensure_qapp()

    window = _ViewerLayerWindow()
    try:
        annotation_shape = _polygon(
            "manual_region",
            [(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)],
            overlay_id=None,
        )
        window.canvas.shapes = [annotation_shape]
        window.large_image_view.set_shapes(window.canvas.shapes)
        before_items = list(window.large_image_view._overlay_items)
        assert before_items

        window._setAnnotationLayerVisible(False)

        assert annotation_shape.visible is False
        after_items = list(window.large_image_view._overlay_items)
        assert len(after_items) == len(before_items)
        item = next(
            (
                overlay_item
                for overlay_item in after_items
                if getattr(overlay_item, "_ann_shape", None) is annotation_shape
            ),
            None,
        )
        assert item is not None
        assert item.isVisible() is False
    finally:
        window.close()


def test_tiled_image_view_delete_key_resyncs_canvas_source_of_truth() -> None:
    _ensure_qapp()

    window = _ViewerLayerWindow()
    try:
        survivor = _polygon(
            "survivor", [(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)], overlay_id=None
        )
        doomed = _polygon(
            "doomed", [(40.0, 40.0), (60.0, 40.0), (50.0, 60.0)], overlay_id=None
        )
        window.canvas.shapes = [survivor, doomed]
        window.canvas.selectedShapes = [doomed]
        window.large_image_view._content_size = (120, 80)
        window.large_image_view.set_shapes(window.canvas.shapes)
        window.large_image_view.selectedShapes = [doomed]
        window._active_image_view = "tiled"

        delete_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Delete,
            QtCore.Qt.NoModifier,
        )
        window.large_image_view.keyPressEvent(delete_event)

        assert window.canvas.shapes == [survivor]
        assert window.large_image_view._shapes == [survivor]
        assert window.large_image_view.selectedShapes == []
    finally:
        window.close()


def test_tiled_image_view_moves_shared_vertices_with_selected_shape() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))

        left = Shape(label="left", shape_type="polygon")
        for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
            left.addPoint(QtCore.QPointF(*point))
        left.close()

        right = Shape(label="right", shape_type="polygon")
        for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
            right.addPoint(QtCore.QPointF(*point))
        right.close()

        left.set_shared_vertex_id(1, "shared-a")
        left.set_shared_vertex_id(2, "shared-b")
        right.set_shared_vertex_id(0, "shared-a")
        right.set_shared_vertex_id(3, "shared-b")
        rebuild_polygon_topology([left, right])

        view.set_shapes([left, right])
        view.selectedShapes = [left]

        assert view._bounded_move_selected_shapes(QtCore.QPointF(5.0, 7.0))

        assert left.points[0].x() == 15.0
        assert left.points[0].y() == 17.0
        assert left.points[1].x() == 25.0
        assert left.points[1].y() == 17.0
        assert right.points[0].x() == 25.0
        assert right.points[0].y() == 17.0
        assert right.points[3].x() == 25.0
        assert right.points[3].y() == 27.0
        assert left.shared_vertex_id(1) == right.shared_vertex_id(0)
        assert left.shared_vertex_id(2) == right.shared_vertex_id(3)
    finally:
        view.close()


def test_tiled_image_view_adjoining_polygon_trace_starts_without_default_point() -> (
    None
):
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)

        seed = Shape(shape_type="polygon")
        seed.addPoint(QtCore.QPointF(20.0, 20.0))
        seed.addPoint(QtCore.QPointF(80.0, 20.0))
        view.set_shapes([seed])
        view.selectedShapes = [seed]
        assert view.startAdjoiningPolygonFromSelection()
        assert view.current is not None
        assert view._adjoining_source_shape is seed
        assert len(view.current.points) == 3
        assert view.current.shared_vertex_id(0) == seed.shared_vertex_id(0)
        assert view.current.shared_vertex_id(1) == seed.shared_vertex_id(1)

        def _left_press(scene_point: QtCore.QPointF) -> None:
            local = view.mapFromScene(scene_point)
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(local),
                QtCore.QPointF(view.viewport().mapToGlobal(local)),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
            view.mousePressEvent(event)

        _left_press(QtCore.QPointF(110.0, 60.0))
        assert view.current is not None
        assert len(view.current.points) == 3
        assert view.current.points[-1].x() == 110.0
        assert view.current.points[-1].y() == 60.0
    finally:
        view.close()


def test_tiled_image_view_adjoining_polygon_boundary_point_stays_shared() -> None:
    _ensure_qapp()

    view = TiledImageView()
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.setSceneRect(QtCore.QRectF(0.0, 0.0, 200.0, 200.0))
        view._content_size = (200, 200)

        source = Shape(shape_type="polygon")
        for point in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]:
            source.addPoint(QtCore.QPointF(*point))
        source.close()
        view.set_shapes([source])
        view.selectedShapes = [source]

        assert view.startAdjoiningPolygonFromSelection()
        assert view.current is not None

        def _left_press(scene_point: QtCore.QPointF) -> None:
            local = view.mapFromScene(scene_point)
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                QtCore.QPointF(local),
                QtCore.QPointF(view.viewport().mapToGlobal(local)),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                QtCore.Qt.NoModifier,
            )
            view.mousePressEvent(event)

        _left_press(QtCore.QPointF(80.0, 50.0))
        shared_vertex_id = view.current.shared_vertex_id(2)
        assert shared_vertex_id not in {None, ""}

        view.finalise()
        new_shape = view.selectedShapes[0]
        assert new_shape.shared_vertex_id(2) == shared_vertex_id
        assert source.shared_vertex_id(2) == shared_vertex_id

        new_shape.moveVertexBy(2, QtCore.QPointF(5.0, 3.0))
        view._sync_shared_vertex(new_shape, 2, new_shape.points[2])

        assert source.points[2].x() == new_shape.points[2].x()
        assert source.points[2].y() == new_shape.points[2].y()
    finally:
        view.close()


def test_annotation_loading_preserves_shared_vertex_ids() -> None:
    loader = _LoadingStub()
    shapes = [
        {
            "label": "region_a",
            "points": [(10.0, 10.0), (30.0, 10.0), (20.0, 30.0)],
            "shape_type": "polygon",
            "flags": {},
            "group_id": None,
            "description": "",
            "other_data": {},
            "mask": None,
            "visible": True,
            "shared_vertex_ids": ["shared-a", "shared-b", "shared-c"],
            "shared_edge_ids": ["edge-a", "edge-b", "edge-c"],
        }
    ]

    materialized = loader._materialize_label_shapes(shapes)

    assert len(materialized) == 1
    assert materialized[0].shared_vertex_ids == [
        "shared-a",
        "shared-b",
        "shared-c",
    ]
    assert materialized[0].shared_edge_ids == ["edge-a", "edge-b", "edge-c"]


def test_canvas_bounded_move_vertex_propagates_shared_vertices() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        canvas.pixmap = QtGui.QPixmap(200, 200)
        canvas.pixmap.fill(QtCore.Qt.white)

        left = Shape(label="left", shape_type="polygon")
        for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]:
            left.addPoint(QtCore.QPointF(*point))
        left.close()

        right = Shape(label="right", shape_type="polygon")
        for point in [(60.0, 60.0), (70.0, 60.0), (70.0, 70.0)]:
            right.addPoint(QtCore.QPointF(*point))
        right.close()

        shared_id = "vertex-shared"
        left.set_shared_vertex_id(1, shared_id)
        right.set_shared_vertex_id(0, shared_id)
        canvas.shapes = [left, right]
        canvas.hShape = left
        canvas.hVertex = 1

        canvas.boundedMoveVertex(QtCore.QPointF(26.0, 18.0))

        assert left.points[1].x() == 26.0
        assert left.points[1].y() == 18.0
        assert right.points[0].x() == 26.0
        assert right.points[0].y() == 18.0
        assert left.shared_vertex_id(1) == right.shared_vertex_id(0) == shared_id
    finally:
        canvas.close()


def test_canvas_bounded_move_vertex_merges_close_vertices() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        canvas.pixmap = QtGui.QPixmap(200, 200)
        canvas.pixmap.fill(QtCore.Qt.white)

        left = Shape(label="left", shape_type="polygon")
        for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]:
            left.addPoint(QtCore.QPointF(*point))
        left.close()

        right = Shape(label="right", shape_type="polygon")
        for point in [(40.0, 10.0), (50.0, 10.0), (50.0, 20.0)]:
            right.addPoint(QtCore.QPointF(*point))
        right.close()

        canvas.shapes = [left, right]
        canvas.hShape = left
        canvas.hVertex = 1

        canvas.boundedMoveVertex(QtCore.QPointF(40.4, 10.2))

        assert left.points[1].x() == 40.0
        assert left.points[1].y() == 10.0
        assert right.points[0].x() == 40.0
        assert right.points[0].y() == 10.0
        assert left.shared_vertex_id(1) == right.shared_vertex_id(0)
        assert left.shared_vertex_id(1) not in {None, ""}
    finally:
        canvas.close()


def test_canvas_polygon_closes_normally_after_shared_topology_update() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        canvas.mode = canvas.CREATE
        canvas.createMode = "polygon"
        current = Shape(shape_type="polygon")
        for point in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0)]:
            current.addPoint(QtCore.QPointF(*point))
        canvas.current = current
        canvas.finalise()

        assert len(canvas.shapes) == 1
        assert canvas.shapes[0].shape_type == "polygon"
        assert canvas.shapes[0].isClosed() is True
        assert len(canvas.shapes[0].points) == 3
    finally:
        canvas.close()


def test_canvas_polygon_closes_when_clicking_near_first_vertex() -> None:
    _ensure_qapp()

    canvas = Canvas()
    try:
        canvas.pixmap = QtGui.QPixmap(200, 200)
        canvas.pixmap.fill(QtCore.Qt.white)
        canvas.resize(640, 480)
        canvas.mode = canvas.CREATE
        canvas.createMode = "polygon"
        current = Shape(shape_type="polygon")
        for point in [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0)]:
            current.addPoint(QtCore.QPointF(*point))
        canvas.current = current
        canvas.line.points = [current.points[-1], current.points[-1]]
        canvas.line.point_labels = [1, 1]

        near_first = QtCore.QPointF(22.0, 21.5)
        local_pos = canvas.offsetToCenter() + near_first
        event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            local_pos,
            local_pos,
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        canvas.mousePressEvent(event)

        assert len(canvas.shapes) == 1
        assert canvas.shapes[0].isClosed() is True
        assert len(canvas.shapes[0].points) == 3
    finally:
        canvas.close()


def test_open_polygon_keeps_edge_ids_unassigned_until_commit() -> None:
    _ensure_qapp()

    polygon = Shape(shape_type="polygon")
    polygon.addPoint(QtCore.QPointF(10.0, 10.0))
    polygon.addPoint(QtCore.QPointF(20.0, 10.0))
    polygon.addPoint(QtCore.QPointF(20.0, 20.0))

    assert polygon.isClosed() is False
    assert len(polygon.shared_edge_ids) == len(polygon.points)
    assert all(not edge_id for edge_id in polygon.shared_edge_ids)

    polygon.close()
    rebuild_polygon_topology([polygon])

    assert len(polygon.shared_edge_ids) == len(polygon.points)
    assert any(edge_id for edge_id in polygon.shared_edge_ids)


def test_rebuild_polygon_topology_assigns_shared_edge_ids_for_shared_boundary() -> None:
    _ensure_qapp()

    left = Shape(label="left", shape_type="polygon")
    for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
        left.addPoint(QtCore.QPointF(*point))
    left.close()

    right = Shape(label="right", shape_type="polygon")
    for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
        right.addPoint(QtCore.QPointF(*point))
    right.close()

    shared_vertices = {
        "a": "shared-a",
        "b": "shared-b",
    }
    left.set_shared_vertex_id(1, shared_vertices["a"])
    left.set_shared_vertex_id(2, shared_vertices["b"])
    right.set_shared_vertex_id(0, shared_vertices["a"])
    right.set_shared_vertex_id(3, shared_vertices["b"])

    rebuild_polygon_topology([left, right])

    assert left.shared_edge_id(2)
    assert left.shared_edge_id(2) == right.shared_edge_id(0)


def test_shared_topology_registry_tracks_shared_memberships() -> None:
    _ensure_qapp()

    left = Shape(label="left", shape_type="polygon")
    for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
        left.addPoint(QtCore.QPointF(*point))
    left.close()

    right = Shape(label="right", shape_type="polygon")
    for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
        right.addPoint(QtCore.QPointF(*point))
    right.close()

    left.set_shared_vertex_id(1, "shared-a")
    left.set_shared_vertex_id(2, "shared-b")
    right.set_shared_vertex_id(0, "shared-a")
    right.set_shared_vertex_id(3, "shared-b")

    registry = SharedTopologyRegistry.from_shapes([left, right])

    assert registry.vertex_occurrences("shared-a")
    assert registry.vertex_occurrences("shared-b")
    assert len(registry.vertex_occurrences("shared-a")) == 2
    assert len(registry.vertex_occurrences("shared-b")) == 2
    assert left.shared_edge_id(2)
    assert left.shared_edge_id(2) == right.shared_edge_id(0)
    assert registry.edge_occurrences(left.shared_edge_id(2))


def test_shared_topology_registry_reshapes_shared_boundary_for_all_related_polygons() -> (
    None
):
    _ensure_qapp()

    left = Shape(label="left", shape_type="polygon")
    for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
        left.addPoint(QtCore.QPointF(*point))
    left.close()

    right = Shape(label="right", shape_type="polygon")
    for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
        right.addPoint(QtCore.QPointF(*point))
    right.close()

    left.set_shared_vertex_id(1, "shared-a")
    left.set_shared_vertex_id(2, "shared-b")
    right.set_shared_vertex_id(0, "shared-a")
    right.set_shared_vertex_id(3, "shared-b")
    registry = SharedTopologyRegistry.from_shapes([left, right])

    edge_id = left.shared_edge_id(2)
    result = registry.reshape_edge(left, 2, QtCore.QPointF(0.0, 5.0))

    assert result is not None
    assert left.shared_edge_id(2) == edge_id
    assert right.shared_edge_id(0) == edge_id
    assert left.points[1] == QtCore.QPointF(20.0, 15.0)
    assert left.points[2] == QtCore.QPointF(20.0, 25.0)
    assert right.points[0] == QtCore.QPointF(20.0, 15.0)
    assert right.points[3] == QtCore.QPointF(20.0, 25.0)


def test_insert_shared_vertex_on_edge_updates_all_related_polygons() -> None:
    _ensure_qapp()

    left = Shape(label="left", shape_type="polygon")
    for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
        left.addPoint(QtCore.QPointF(*point))
    left.close()

    right = Shape(label="right", shape_type="polygon")
    for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
        right.addPoint(QtCore.QPointF(*point))
    right.close()

    left.set_shared_vertex_id(1, "shared-a")
    left.set_shared_vertex_id(2, "shared-b")
    right.set_shared_vertex_id(0, "shared-a")
    right.set_shared_vertex_id(3, "shared-b")
    rebuild_polygon_topology([left, right])

    result = insert_shared_vertex_on_edge(
        left,
        2,
        QtCore.QPointF(20.0, 15.0),
        [left, right],
    )

    assert result is not None
    assert len(left.points) == 5
    assert len(right.points) == 5
    assert left.points[2] == QtCore.QPointF(20.0, 15.0)
    assert right.points[0] == QtCore.QPointF(20.0, 15.0)
    assert left.shared_vertex_id(2) == right.shared_vertex_id(0)
    assert left.shared_vertex_id(2) not in {None, ""}


def test_remove_shared_vertex_at_updates_all_related_polygons() -> None:
    _ensure_qapp()

    left = Shape(label="left", shape_type="polygon")
    for point in [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]:
        left.addPoint(QtCore.QPointF(*point))
    left.close()

    right = Shape(label="right", shape_type="polygon")
    for point in [(20.0, 10.0), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0)]:
        right.addPoint(QtCore.QPointF(*point))
    right.close()

    left.set_shared_vertex_id(1, "shared-a")
    left.set_shared_vertex_id(2, "shared-b")
    right.set_shared_vertex_id(0, "shared-a")
    right.set_shared_vertex_id(3, "shared-b")
    rebuild_polygon_topology([left, right])

    insert_shared_vertex_on_edge(
        left,
        2,
        QtCore.QPointF(20.0, 15.0),
        [left, right],
    )

    result = remove_shared_vertex_at(left, 2, [left, right])

    assert result is not None
    assert len(left.points) == 4
    assert len(right.points) == 4
    assert all(point != QtCore.QPointF(20.0, 15.0) for point in left.points)
    assert all(point != QtCore.QPointF(20.0, 15.0) for point in right.points)


def test_remove_shared_vertex_at_falls_back_to_single_polygon_vertex_delete() -> None:
    _ensure_qapp()

    shape = Shape(label="solo", shape_type="polygon")
    for point in [(5.0, 5.0), (50.0, 5.0), (50.0, 50.0), (5.0, 50.0)]:
        shape.addPoint(QtCore.QPointF(*point))
    shape.close()

    result = remove_shared_vertex_at(shape, 1, [shape])

    assert result is not None
    assert len(shape.points) == 3
