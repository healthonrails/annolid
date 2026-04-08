from __future__ import annotations

import os
from pathlib import Path

import tifffile
import numpy as np
from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.label_image_overlay_mixin import LabelImageOverlayMixin
from annolid.gui.mixins.layer_dock_mixin import LayerDockMixin
from annolid.gui.mixins.raster_layer_mixin import RasterLayerMixin
from annolid.gui.mixins.vector_overlay_mixin import VectorOverlayMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.tiled_image_view import TiledImageView
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


class _CanvasStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.selectedShapes = []
        self.createMode = "polygon"

    def update(self):
        return None

    def setBehaviorText(self, _value):
        return None

    def editing(self):
        return True


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

        changed = window.setRasterImageLayerVisible("raster_overlay_test", False)
        assert changed is True
        state_after = window.large_image_view.raster_overlay_layers_state()
        assert state_after[0]["visible"] is False
        assert window.otherData["raster_image_layers"][0]["visible"] is False
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
