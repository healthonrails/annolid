from __future__ import annotations

import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.vector_overlay_mixin import VectorOverlayMixin
from annolid.gui.shape import Shape
from annolid.gui.window_base import AnnolidWindowBase
from annolid.io.vector import (
    export_overlay_document_json,
    export_overlay_document_labelme,
    export_overlay_document_svg,
)


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

    def update(self):
        super().update()


class _LargeImageViewStub:
    def set_shapes(self, shapes):
        self.last_shapes = list(shapes or [])


class _OverlayHost(VectorOverlayMixin, AnnolidWindowBase):
    def __init__(self):
        super().__init__(config={})
        self.canvas = _CanvasStub()
        self.large_image_view = _LargeImageViewStub()
        self.last_status = None
        self.last_error = None

    def toggleActions(self, value):
        _ = value

    def setDirty(self):
        self.dirty = True

    def status(self, message):
        self.last_status = message

    def errorMessage(self, title, message):
        self.last_error = (title, message)


def _overlay_polygon() -> Shape:
    shape = Shape("region_a", shape_type="polygon")
    shape.addPoint(QtCore.QPointF(1.0, 2.0))
    shape.addPoint(QtCore.QPointF(5.0, 2.0))
    shape.addPoint(QtCore.QPointF(5.0, 6.0))
    shape.close()
    shape.other_data = {
        "overlay_id": "overlay_a",
        "overlay_source": "/tmp/atlas.svg",
        "overlay_element": "path",
        "overlay_element_id": "region_a",
        "overlay_layer": "brain",
        "overlay_stroke": "#112233",
        "overlay_fill": "#445566",
        "overlay_visible": True,
        "overlay_opacity": 0.5,
        "overlay_z_order": 0,
    }
    return shape


def _overlay_point() -> Shape:
    shape = Shape("A", shape_type="point")
    shape.addPoint(QtCore.QPointF(9.0, 10.0))
    shape.other_data = {
        "overlay_id": "overlay_a",
        "overlay_source": "/tmp/atlas.svg",
        "overlay_element": "circle",
        "overlay_element_id": "landmark_a",
        "overlay_layer": "brain",
        "overlay_stroke": "#778899",
        "overlay_fill": "#aabbcc",
        "overlay_visible": True,
        "overlay_opacity": 0.5,
        "overlay_z_order": 0,
    }
    return shape


def test_build_vector_overlay_document_collects_overlay_shapes() -> None:
    _ensure_qapp()
    window = _OverlayHost()
    try:
        window.canvas.shapes = [_overlay_polygon(), _overlay_point()]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {
                        "source_kind": "ai",
                        "source_shapes": [
                            {
                                "id": "region_a",
                                "kind": "polygon",
                                "points": [(1.0, 2.0), (5.0, 2.0), (5.0, 6.0)],
                            }
                        ],
                    },
                    "landmark_pairs": [
                        {
                            "pair_id": "pair_1",
                            "overlay_label": "A",
                            "image_label": "A_img",
                        }
                    ],
                    "transform": {"tx": 3.0, "opacity": 0.4, "visible": True},
                }
            ]
        }

        document = window.buildVectorOverlayDocument("overlay_a")

        assert document is not None
        assert document.source_path == "/tmp/atlas.svg"
        assert document.transform.tx == 3.0
        assert document.source_kind == "ai"
        assert document.source_shapes
        assert document.landmark_pairs[0]["pair_id"] == "pair_1"
        assert [shape.kind for shape in document.shapes] == ["polygon", "point"]
    finally:
        window.close()


def test_export_overlay_document_svg_and_json(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _OverlayHost()
    try:
        window.canvas.shapes = [_overlay_polygon(), _overlay_point()]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                    },
                }
            ]
        }
        document = window.buildVectorOverlayDocument("overlay_a")
        assert document is not None

        svg_path = export_overlay_document_svg(document, tmp_path / "overlay.svg")
        json_path = export_overlay_document_json(document, tmp_path / "overlay.json")

        svg_root = ET.parse(svg_path).getroot()
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        assert svg_root.tag.endswith("svg")
        ns = {"svg": "http://www.w3.org/2000/svg"}
        assert svg_root.find(".//svg:polygon", ns) is not None
        assert svg_root.find(".//svg:circle", ns) is not None
        assert payload["source_path"] == "/tmp/atlas.svg"
        assert payload["shapes"][0]["id"] == "region_a"
        assert payload["shapes"][1]["kind"] == "point"
    finally:
        window.close()


def test_export_overlay_document_labelme_json(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _OverlayHost()
    try:
        window.canvas.shapes = [_overlay_polygon(), _overlay_point()]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 2.0,
                        "ty": 3.0,
                        "opacity": 0.5,
                        "visible": True,
                    },
                }
            ]
        }
        document = window.buildVectorOverlayDocument("overlay_a")
        assert document is not None

        labelme_path = export_overlay_document_labelme(
            document, tmp_path / "overlay.labelme.json"
        )
        payload = json.loads(labelme_path.read_text(encoding="utf-8"))

        assert payload["imagePath"] == ""
        assert payload["imageHeight"] == 0
        assert payload["shapes"][0]["shape_type"] == "polygon"
        assert payload["shapes"][1]["shape_type"] == "point"
        assert payload["shapes"][0]["overlay_stroke"] == "#112233"
        assert payload["otherData"]["overlay"]["transform"]["tx"] == 2.0
    finally:
        window.close()


def test_export_vector_overlay_uses_selected_overlay_and_writes_svg(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    window = _OverlayHost()
    try:
        window.canvas.shapes = [_overlay_polygon()]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                    },
                }
            ]
        }
        window.setupVectorOverlayDock()

        out_path = tmp_path / "corrected.svg"
        exported = window.exportVectorOverlay(output_path=str(out_path))

        assert exported == str(out_path)
        assert out_path.exists()
        assert "Exported corrected overlay" in str(window.last_status)
    finally:
        window.close()


def test_export_vector_overlay_writes_labelme_json(tmp_path: Path) -> None:
    _ensure_qapp()
    window = _OverlayHost()
    try:
        window.canvas.shapes = [_overlay_polygon()]
        window.otherData = {
            "svg_overlays": [
                {
                    "id": "overlay_a",
                    "source": "/tmp/atlas.svg",
                    "metadata": {},
                    "transform": {
                        "tx": 0.0,
                        "ty": 0.0,
                        "opacity": 0.5,
                        "visible": True,
                    },
                }
            ]
        }
        window.setupVectorOverlayDock()

        out_path = tmp_path / "corrected.labelme.json"
        exported = window.exportVectorOverlay(output_path=str(out_path))
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        assert exported == str(out_path)
        assert payload["shapes"][0]["label"] == "region_a"
        assert payload["otherData"]["overlay"]["source_path"] == "/tmp/atlas.svg"
    finally:
        window.close()
