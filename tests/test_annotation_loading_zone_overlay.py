from __future__ import annotations

import json
from pathlib import Path

from qtpy import QtCore

from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.shape import Shape
from annolid.postprocessing.zone_schema import build_zone_shape


class _CanvasStub:
    def __init__(self) -> None:
        self.shapes = []


class _ZoneOverlayHost(AnnotationLoadingMixin):
    def __init__(self) -> None:
        self._config = {"label_flags": {}}
        self.canvas = _CanvasStub()
        self.zone_path = None
        self.video_file = None
        self.filename = None


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
