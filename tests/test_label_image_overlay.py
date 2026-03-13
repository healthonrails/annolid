from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
from qtpy import QtWidgets, QtCore

from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.mixins.file_browser_mixin import FileBrowserMixin
from annolid.gui.label_image_overlay import (
    colorize_label_image,
    label_entry_text,
    load_label_mapping_csv,
)
from annolid.gui.mixins.label_image_overlay_mixin import LabelImageOverlayMixin
from annolid.gui.window_base import AnnolidWindowBase
from annolid.gui.widgets.tiled_image_view import TiledImageView
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


class _LabelOverlayHost(LabelImageOverlayMixin):
    def __init__(self, *, base_backend=None, content_size=(0, 0)):
        self.large_image_backend = base_backend
        self.large_image_view = TiledImageView()
        self.large_image_view._content_size = tuple(content_size)

    def tr(self, text):
        return text


class _OverlayAction:
    def __init__(self):
        self.enabled = True
        self.checked = False

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setChecked(self, value):
        self.checked = bool(value)


class _LabelOverlayPageWindow(
    AnnotationLoadingMixin, FileBrowserMixin, LabelImageOverlayMixin, AnnolidWindowBase
):
    def __init__(self):
        super().__init__(config={"label_flags": {}, "store_data": False})
        self.canvas = QtWidgets.QWidget()
        self.canvas.shapes = []
        self.canvas.update = lambda: None
        self.canvas.setBehaviorText = lambda _value: None
        self.large_image_view = TiledImageView(self)
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.addWidget(self.canvas)
        self._viewer_stack.addWidget(self.large_image_view)
        self.fileListWidget = QtWidgets.QListWidget()
        self._active_image_view = "tiled"
        self.actions.toggle_label_image_overlay_visible = _OverlayAction()

    def tr(self, text):
        return text

    def status(self, *_args, **_kwargs):
        return None

    def loadShapes(self, shapes, replace=True):
        _ = replace
        self.canvas.shapes = list(shapes or [])
        self.large_image_view.set_shapes(self.canvas.shapes)


def _write_page_label_json(path: Path, *, label: str, image_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{\n"
            '  "version": "test",\n'
            '  "flags": {},\n'
            f'  "imagePath": "{image_name}",\n'
            '  "imageData": null,\n'
            '  "imageHeight": 40,\n'
            '  "imageWidth": 60,\n'
            '  "shapes": [\n'
            "    {\n"
            f'      "label": "{label}",\n'
            '      "points": [[10, 10], [20, 10], [20, 20], [10, 20]],\n'
            '      "group_id": null,\n'
            '      "shape_type": "polygon",\n'
            '      "flags": {},\n'
            '      "description": "",\n'
            '      "mask": null,\n'
            '      "visible": true\n'
            "    }\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_load_label_mapping_csv_accepts_region_columns(tmp_path: Path) -> None:
    mapping_path = tmp_path / "atlas.csv"
    mapping_path.write_text(
        "structure_id,abbreviation,structure_name\n1,CTX,Cortex\n315,VISp,Primary visual area\n",
        encoding="utf-8",
    )

    mapping = load_label_mapping_csv(mapping_path)

    assert mapping[1]["acronym"] == "CTX"
    assert mapping[315]["name"] == "Primary visual area"
    assert label_entry_text(315, mapping) == "315: VISp (Primary visual area)"


def test_colorize_label_image_dim_selected_label_context() -> None:
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint16)

    rgba = colorize_label_image(labels, opacity=0.5, selected_label=2)

    assert rgba.shape == (2, 2, 4)
    assert int(rgba[0, 0, 3]) == 0
    assert int(rgba[1, 0, 3]) >= 210
    assert int(rgba[0, 1, 3]) < int(rgba[1, 0, 3])


def test_tiled_image_view_reads_label_value_and_selection(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "image.tif"
    labels_path = tmp_path / "labels.tif"
    tifffile.imwrite(image_path, np.zeros((64, 64), dtype=np.uint8))
    label_arr = np.zeros((64, 64), dtype=np.uint16)
    label_arr[10:40, 12:50] = 315
    tifffile.imwrite(labels_path, label_arr)

    image_backend = TiffFileBackend(image_path)
    label_backend = TiffFileBackend(labels_path)
    view = TiledImageView()
    try:
        view.resize(300, 300)
        view.set_backend(image_backend)
        view.set_label_layer(
            label_backend,
            opacity=0.45,
            mapping={
                315: {"id": 315, "acronym": "VISp", "name": "Primary visual area"}
            },
            source_path=str(labels_path),
        )
        point = QtCore.QPointF(20, 20)

        label_value = view.label_value_at(point)
        view.set_selected_label_value(label_value)

        assert label_value == 315
        assert view.selected_label_value() == 315
        assert view.label_overlay_state()["source_path"] == str(labels_path)
    finally:
        view.close()


def test_label_overlay_host_picks_nonempty_page_for_stack(tmp_path: Path) -> None:
    _ensure_qapp()

    labels_path = tmp_path / "labels_stack.tif"
    tifffile.imwrite(labels_path, np.zeros((32, 40), dtype=np.uint16))
    tifffile.imwrite(labels_path, np.zeros((32, 40), dtype=np.uint16), append=True)
    page2 = np.zeros((32, 40), dtype=np.uint16)
    page2[5:20, 10:25] = 99
    tifffile.imwrite(labels_path, page2, append=True)
    backend = TiffFileBackend(labels_path)
    host = _LabelOverlayHost(content_size=(40, 32))

    page = host._resolve_initial_label_overlay_page(backend)

    assert page == 2


def test_label_overlay_host_autofits_small_label_image_to_large_view(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    labels_path = tmp_path / "labels.tif"
    labels = np.zeros((320, 456), dtype=np.uint16)
    tifffile.imwrite(labels_path, labels)
    backend = TiffFileBackend(labels_path)
    host = _LabelOverlayHost(content_size=(24253, 16892))

    transform = host._default_label_overlay_transform(backend)

    assert transform["sx"] > 1.0
    assert transform["sy"] > 1.0
    assert transform["tx"] > 0.0
    assert transform["ty"] > 0.0


def test_label_overlay_visibility_toggle_updates_state(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "image.tif"
    labels_path = tmp_path / "labels.tif"
    tifffile.imwrite(image_path, np.zeros((64, 64), dtype=np.uint8))
    label_arr = np.zeros((64, 64), dtype=np.uint16)
    label_arr[4:20, 6:24] = 7
    tifffile.imwrite(labels_path, label_arr)

    image_backend = TiffFileBackend(image_path)
    label_backend = TiffFileBackend(labels_path)
    host = _LabelOverlayHost(base_backend=image_backend, content_size=(64, 64))
    host._active_image_view = "tiled"
    host.otherData = {}
    host.actions = type(
        "_Actions", (), {"toggle_label_image_overlay_visible": _OverlayAction()}
    )()
    host.large_image_view.set_backend(image_backend)
    host.large_image_view.set_label_layer(label_backend, source_path=str(labels_path))
    host.otherData["label_image_overlay"] = host.large_image_view.label_overlay_state()

    assert host.setLabelImageOverlayVisible(False) is True
    assert host.large_image_view.label_layer_visible() is False
    assert host.otherData["label_image_overlay"]["visible"] is False
    assert host.actions.toggle_label_image_overlay_visible.checked is False

    assert host.setLabelImageOverlayVisible(True) is True
    assert host.large_image_view.label_layer_visible() is True
    assert host.otherData["label_image_overlay"]["visible"] is True
    assert host.actions.toggle_label_image_overlay_visible.checked is True


def test_large_image_page_annotation_baseline_preserves_label_overlay(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    label_path = tmp_path / "annotation_stack.tif"
    image_pages = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    label_pages = np.stack(
        [
            np.zeros((40, 60), dtype=np.uint16),
            np.full((40, 60), 11, dtype=np.uint16),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, image_pages)
    tifffile.imwrite(label_path, label_pages)
    annotation_dir = tmp_path / "multipage_stack"
    _write_page_label_json(
        annotation_dir / "multipage_stack_000000000.json",
        label="page0",
        image_name=image_path.name,
    )
    _write_page_label_json(
        annotation_dir / "multipage_stack_000000001.json",
        label="page1",
        image_name=image_path.name,
    )

    window = _LabelOverlayPageWindow()
    try:
        window.imagePath = str(image_path)
        window.loadFile(str(image_path))
        label_backend = TiffFileBackend(label_path)
        window.large_image_view.set_label_layer(
            label_backend,
            source_path=str(label_path),
            visible=True,
            page_index=0,
        )
        window.otherData["label_image_overlay"] = (
            window.large_image_view.label_overlay_state()
        )

        assert window.setLargeImagePageNumber(1) is True
        state = window.large_image_view.label_overlay_state()
        assert state["source_path"] == str(label_path)
        assert state["visible"] is True
        assert state["page_index"] == 1
        assert window.actions.toggle_label_image_overlay_visible.checked is True
    finally:
        window.close()
