from __future__ import annotations

import json
import time
from pathlib import Path

from qtpy import QtCore

import annolid.gui.mixins.annotation_loading_mixin as annotation_loading_module
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.shape import Shape
from annolid.postprocessing.zone_schema import build_zone_shape


class _CanvasStub:
    def __init__(self) -> None:
        self.shapes = []
        self.current_behavior_text = ""

    def setBehaviorText(self, value) -> None:
        self.current_behavior_text = str(value or "")


class _ZoneOverlayHost(AnnotationLoadingMixin):
    def __init__(self) -> None:
        self._config = {"label_flags": {}}
        self.canvas = _CanvasStub()
        self.zone_path = None
        self.video_file = None
        self.filename = None


class _PredictHost(AnnotationLoadingMixin):
    def __init__(self) -> None:
        self._config = {"label_flags": {}}
        self.canvas = _CanvasStub()
        self.caption_widget = None
        self.frame_number = 0
        self.video_results_folder = None
        self.annotation_dir = None
        self._pred_res_folder_suffix = "_tracking_results_labelme"
        self._df = None
        self.labelFile = None
        self.loaded_shapes = []

    def _persistent_zone_shapes_for_frame(self, _frame_path):
        return []

    def _merge_persistent_zones_into_shapes(self, shapes, _frame_path, **_kwargs):
        return list(shapes or [])

    def update_flags_from_file(self, _label_file) -> None:
        return None

    def add_highlighted_mark(self, *_args, **_kwargs) -> None:
        return None

    def _apply_timeline_caption_if_available(self, *_args, **_kwargs) -> bool:
        return False

    def loadShapes(self, shapes, replace=True):  # noqa: ARG002
        self.loaded_shapes = list(shapes or [])


def _zone_shape(label: str = "left_zone") -> Shape:
    shape = Shape(label=label, shape_type="polygon")
    for x, y in ((0, 0), (10, 0), (10, 10), (0, 10)):
        shape.addPoint(QtCore.QPointF(float(x), float(y)))
    shape.close()
    shape.flags = {"semantic_type": "zone", "zone_kind": "chamber"}
    return shape


def _animal_shape(label: str = "animal") -> Shape:
    shape = Shape(label=label, shape_type="polygon")
    for x, y in ((20, 20), (30, 20), (30, 30), (20, 30)):
        shape.addPoint(QtCore.QPointF(float(x), float(y)))
    shape.close()
    shape.flags = {"semantic_type": "instance"}
    return shape


def test_merge_persistent_zones_includes_current_canvas_zone_without_duplication() -> (
    None
):
    host = _ZoneOverlayHost()
    persistent_zone = _zone_shape("left_zone")
    host.canvas.shapes = [persistent_zone]

    merged = host._merge_persistent_zones_into_shapes(
        [_animal_shape("subject"), _zone_shape("left_zone")],
        frame_path=None,
    )

    zone_labels = [
        shape.label for shape in merged if shape.flags.get("semantic_type") == "zone"
    ]
    assert zone_labels.count("left_zone") == 1
    assert any(shape.label == "subject" for shape in merged)


def test_persistent_zone_shapes_load_from_zone_json_file(tmp_path: Path) -> None:
    host = _ZoneOverlayHost()
    zone_file = tmp_path / "session_zones.json"
    zone_file.write_text(
        json.dumps(
            {
                "shapes": [
                    build_zone_shape(
                        "stim_zone",
                        [[5, 5], [15, 5], [15, 15], [5, 15]],
                        zone_kind="chamber",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    host.zone_path = str(zone_file)

    zones = host._persistent_zone_shapes_for_frame(tmp_path / "frame_000000001.png")

    assert len(zones) == 1
    assert zones[0].label == "stim_zone"
    assert zones[0].flags["semantic_type"] == "zone"


def _fake_shape_payload(label: str = "subject") -> dict:
    return {
        "label": label,
        "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "visible": True,
        "mask": None,
    }


def test_load_predict_shapes_reuses_cached_label_file(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeLabelFile:
        calls = 0

        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            type(self).calls += 1
            self.shapes = [_fake_shape_payload()]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)

    host = _PredictHost()
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(0, str(frame_png))
    host.loadPredictShapes(0, str(frame_png))

    assert _FakeLabelFile.calls == 1
    assert len(host.loaded_shapes) == 1


def test_load_predict_shapes_cache_invalidates_when_json_changes(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeLabelFile:
        calls = 0

        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            type(self).calls += 1
            self.shapes = [_fake_shape_payload(label=f"subject_{type(self).calls}")]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)

    host = _PredictHost()
    frame_png = tmp_path / "session_000000001.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(1, str(frame_png))
    time.sleep(0.01)
    frame_json.write_text('{"updated": true}', encoding="utf-8")
    host.loadPredictShapes(1, str(frame_png))

    assert _FakeLabelFile.calls == 2
