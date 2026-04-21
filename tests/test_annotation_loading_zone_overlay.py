from __future__ import annotations

import json
import time
from pathlib import Path

from pandas import DataFrame
from qtpy import QtCore
import numpy as np
import pycocotools.mask as mask_util

import annolid.gui.mixins.annotation_loading_mixin as annotation_loading_module
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.shape import Shape
from annolid.infrastructure import AnnotationStore
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


def test_load_predict_shapes_preserves_loaded_shapes_when_flag_sync_fails(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeLabelFile:
        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            self.shapes = [_fake_shape_payload(label="subject")]
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

    def _fail_flags(_label_file) -> None:
        raise RuntimeError("flag sync failed")

    host.update_flags_from_file = _fail_flags  # type: ignore[method-assign]
    host.loadPredictShapes(0, str(frame_png))

    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "subject"


def test_load_predict_shapes_tracking_fallback_uses_fast_mode_while_playing(
    monkeypatch,
) -> None:
    host = _PredictHost()
    host._df = DataFrame([{"frame_number": 0, "instance_name": "animal_1"}])
    host.isPlaying = True
    host.loadPredictShapes(0, "")

    assert host.loaded_shapes == []


def test_playback_tracking_fallback_renders_bbox_when_segmentation_decode_disabled() -> (
    None
):
    host = _PredictHost()
    host._df = DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "animal_1",
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 40,
                "class_score": 1.0,
                "segmentation": "{'size': [64, 64], 'counts': 'abcd'}",
                "tracking_id": 2,
            }
        ]
    )
    host.isPlaying = True

    host.loadPredictShapes(0, "")

    assert host.loaded_shapes == []


def test_load_predict_shapes_uses_deferred_tracking_csv_source(
    monkeypatch,
) -> None:
    class _TrackingControllerStub:
        def __init__(self) -> None:
            self._tracking_csv_path = Path("/tmp/session_tracking.csv")
            self.calls: list[int] = []

        def tracking_rows_for_frame(self, frame_number: int) -> list[dict]:
            self.calls.append(int(frame_number))
            return [
                {
                    "frame_number": int(frame_number),
                    "instance_name": "animal_1",
                    "x1": 1,
                    "y1": 2,
                    "x2": 4,
                    "y2": 5,
                    "class_score": 0.95,
                    "segmentation": None,
                }
            ]

    call_flags: list[bool] = []

    def _fake_pred_dict_to_labelme(row, **kwargs):  # noqa: ANN001
        call_flags.append(bool(kwargs.get("decode_segmentation", True)))
        shape = Shape(
            label=str(row.get("instance_name", "subject")),
            shape_type="point",
        )
        shape.addPoint(QtCore.QPointF(3.0, 4.0))
        return [shape]

    monkeypatch.setattr(
        annotation_loading_module, "pred_dict_to_labelme", _fake_pred_dict_to_labelme
    )

    host = _PredictHost()
    host._df = None
    host.isPlaying = False
    controller = _TrackingControllerStub()
    host.tracking_data_controller = controller

    host.loadPredictShapes(0, "")

    assert controller.calls == []
    assert call_flags == []
    assert host.loaded_shapes == []


def test_load_predict_shapes_tracking_fallback_caches_per_frame_and_mode(
    monkeypatch,
) -> None:
    call_flags: list[bool] = []

    def _fake_pred_dict_to_labelme(row, **kwargs):  # noqa: ANN001
        call_flags.append(bool(kwargs.get("decode_segmentation", True)))
        shape = Shape(
            label=str(row.get("instance_name", "subject")),
            shape_type="point",
        )
        shape.addPoint(QtCore.QPointF(3.0, 4.0))
        return [shape]

    monkeypatch.setattr(
        annotation_loading_module, "pred_dict_to_labelme", _fake_pred_dict_to_labelme
    )

    host = _PredictHost()
    host._df = DataFrame([{"frame_number": 0, "instance_name": "animal_1"}])

    host.isPlaying = True
    host.loadPredictShapes(0, "")
    host.loadPredictShapes(0, "")
    host.isPlaying = False
    host.loadPredictShapes(0, "")
    host.loadPredictShapes(0, "")

    assert call_flags == []


def test_load_predict_shapes_clears_stale_shapes_when_frame_has_no_annotations() -> (
    None
):
    host = _PredictHost()
    host.loaded_shapes = [_animal_shape("stale")]
    host.loadPredictShapes(0, "")
    assert host.loaded_shapes == []


def test_load_predict_shapes_shows_persistent_zones_without_frame_annotations() -> None:
    host = _PredictHost()
    host._persistent_zone_shapes_for_frame = lambda _frame_path: [_zone_shape("zone_a")]
    host._merge_persistent_zones_into_shapes = (
        lambda shapes, _frame_path, **kwargs: list(shapes or [])
        + list(kwargs.get("persistent_zone_shapes") or [])
    )
    host.loadPredictShapes(0, "")
    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "zone_a"


def test_tracking_shape_cache_returns_copied_shapes_to_avoid_mutation_leak(
    monkeypatch,
) -> None:
    call_count = 0

    def _fake_pred_dict_to_labelme(row, **kwargs):  # noqa: ANN001, ARG001
        nonlocal call_count
        call_count += 1
        shape = Shape(label=str(row.get("instance_name", "animal")), shape_type="point")
        shape.addPoint(QtCore.QPointF(5.0, 6.0))
        return [shape]

    monkeypatch.setattr(
        annotation_loading_module, "pred_dict_to_labelme", _fake_pred_dict_to_labelme
    )

    host = _PredictHost()
    host._df = DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "animal_1",
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "class_score": 0.9,
                "segmentation": None,
            }
        ]
    )
    host.isPlaying = False

    host.loadPredictShapes(0, "")
    assert host.loaded_shapes == []
    assert call_count == 0


def test_tracking_segmentation_rows_render_as_polygons_for_small_masks() -> None:
    host = _PredictHost()
    host._df = None
    host.isPlaying = False

    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:7, 3:8] = 1  # area=25, below legacy threshold 512
    encoded = mask_util.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")

    class _TrackingControllerStub:
        def __init__(self) -> None:
            self._tracking_csv_path = Path("/tmp/session_tracking.csv")

        def tracking_rows_for_frame(self, frame_number: int) -> list[dict]:
            return [
                {
                    "frame_number": int(frame_number),
                    "instance_name": "animal_1",
                    "x1": 3,
                    "y1": 2,
                    "x2": 8,
                    "y2": 7,
                    "class_score": 1.0,
                    "segmentation": str(dict(encoded)),
                    "tracking_id": 0,
                }
            ]

    host.tracking_data_controller = _TrackingControllerStub()
    host.loadPredictShapes(0, "")

    assert host.loaded_shapes == []


def test_playback_tracking_fastpath_prefers_tracking_when_no_frame_labels(
    monkeypatch,
) -> None:
    class _FailLabelFile:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Label file probing should be skipped in this path.")

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FailLabelFile)

    host = _PredictHost()
    host._df = DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "animal_1",
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "class_score": 0.9,
                "segmentation": None,
            }
        ]
    )
    host.isPlaying = True
    host._iter_frame_label_candidates = lambda *_args, **_kwargs: [
        Path("/tmp/fake_000000000.json")
    ]
    host._label_candidate_cache_token = lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        annotation_loading_module,
        "pred_dict_to_labelme",
        lambda row, **kwargs: [
            Shape(label=str(row.get("instance_name")), shape_type="point")
        ],
    )
    host.loadPredictShapes(0, "")
    assert host.loaded_shapes == []


def test_playback_fastpath_uses_frame_label_when_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeLabelFile:
        calls = 0

        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            type(self).calls += 1
            self.shapes = [_fake_shape_payload(label="from_label")]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    tracking_calls: list[int] = []

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)
    monkeypatch.setattr(
        annotation_loading_module,
        "pred_dict_to_labelme",
        lambda row, **kwargs: (  # noqa: ARG005
            tracking_calls.append(1)
            or [
                Shape(
                    label=str(row.get("instance_name", "from_tracking")),
                    shape_type="point",
                )
            ]
        ),
    )

    host = _PredictHost()
    host.isPlaying = True
    host._df = DataFrame([{"frame_number": 0, "instance_name": "from_tracking"}])
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(0, str(frame_png))

    assert _FakeLabelFile.calls == 1
    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_label"
    assert tracking_calls == []


def test_playback_fastpath_uses_annotation_store_frame_when_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeLabelFile:
        calls = 0

        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            type(self).calls += 1
            self.shapes = [_fake_shape_payload(label="from_store")]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    tracking_calls: list[int] = []

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)
    monkeypatch.setattr(
        annotation_loading_module,
        "pred_dict_to_labelme",
        lambda row, **kwargs: (  # noqa: ARG005
            tracking_calls.append(1)
            or [
                Shape(
                    label=str(row.get("instance_name", "from_tracking")),
                    shape_type="point",
                )
            ]
        ),
    )

    host = _PredictHost()
    host.isPlaying = True
    host._df = DataFrame([{"frame_number": 0, "instance_name": "from_tracking"}])
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    # Intentionally do not create frame_json. Use annotation store only.

    store = AnnotationStore.for_frame_path(frame_json)
    store.append_frame(
        {
            "frame": 0,
            "version": "annolid",
            "flags": {},
            "shapes": [_fake_shape_payload(label="from_store")],
            "imagePath": "",
            "imageHeight": 1,
            "imageWidth": 1,
        }
    )

    host.loadPredictShapes(0, str(frame_png))

    assert _FakeLabelFile.calls == 0
    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_store"
    assert tracking_calls == []


def test_load_predict_shapes_prefers_annotation_store_over_existing_json(
    monkeypatch, tmp_path: Path
) -> None:
    class _FailLabelFile:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Disk JSON loader should be bypassed when store wins.")

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FailLabelFile)

    host = _PredictHost()
    host.isPlaying = False
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text(
        json.dumps(
            {
                "shapes": [_fake_shape_payload(label="from_json")],
                "flags": {},
                "imagePath": "",
                "imageHeight": 1,
                "imageWidth": 1,
            }
        ),
        encoding="utf-8",
    )

    store = AnnotationStore.for_frame_path(frame_json)
    store.append_frame(
        {
            "frame": 0,
            "version": "annolid",
            "flags": {},
            "shapes": [_fake_shape_payload(label="from_store")],
            "imagePath": "",
            "imageHeight": 1,
            "imageWidth": 1,
        }
    )

    host.loadPredictShapes(0, str(frame_png))

    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_store"


def test_load_predict_shapes_uses_latest_previous_manual_frame_when_exact_missing(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeLabelFile:
        calls: list[str] = []

        def __init__(self, path, is_video_frame=True):  # noqa: ARG002
            type(self).calls.append(str(path))
            self.shapes = [_fake_shape_payload(label="from_prev_manual")]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)

    host = _PredictHost()
    current_png = tmp_path / "session_000000500.png"
    current_png.write_bytes(b"")
    prev_png = tmp_path / "session_000000474.png"
    prev_png.write_bytes(b"")
    prev_json = prev_png.with_suffix(".json")
    prev_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(500, str(current_png))

    assert _FakeLabelFile.calls == [str(prev_json)]
    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_prev_manual"


def test_load_predict_shapes_uses_latest_previous_store_frame_when_exact_missing(
    tmp_path: Path,
) -> None:
    host = _PredictHost()
    current_png = tmp_path / "session_000000500.png"
    current_png.write_bytes(b"")

    store = AnnotationStore.for_frame_path(current_png.with_suffix(".json"))
    store.append_frame(
        {
            "frame": 474,
            "version": "annolid",
            "flags": {},
            "shapes": [_fake_shape_payload(label="from_prev_store")],
            "imagePath": "",
            "imageHeight": 1,
            "imageWidth": 1,
        }
    )

    host.loadPredictShapes(500, str(current_png))

    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_prev_store"


def test_load_predict_shapes_materializes_store_shapes_without_mask_key(
    tmp_path: Path,
) -> None:
    host = _PredictHost()
    current_png = tmp_path / "session_000000500.png"
    current_png.write_bytes(b"")

    store = AnnotationStore.for_frame_path(current_png.with_suffix(".json"))
    store.append_frame(
        {
            "frame": 500,
            "version": "annolid",
            "flags": {},
            "shapes": [
                {
                    "label": "from_store_nomask",
                    "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "description": "store shape without mask key",
                    "visible": True,
                }
            ],
            "imagePath": "",
            "imageHeight": 1,
            "imageWidth": 1,
        }
    )

    host.loadPredictShapes(500, str(current_png))

    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_store_nomask"
    assert host.loaded_shapes[0].shape_type == "polygon"


def test_paused_mode_still_uses_label_candidate_probe(
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
    host.isPlaying = False
    host._df = DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "animal_1",
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "class_score": 0.9,
                "segmentation": None,
            }
        ]
    )
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(0, str(frame_png))

    assert _FakeLabelFile.calls == 1


def test_playback_fastpath_falls_back_to_label_probe_when_tracking_empty(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeLabelFile:
        calls = 0

        def __init__(self, _path, is_video_frame=True):  # noqa: ARG002
            type(self).calls += 1
            self.shapes = [_fake_shape_payload(label="from_label")]
            self.flags = {}
            self.caption = ""

        def get_caption(self):
            return ""

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FakeLabelFile)

    host = _PredictHost()
    host.isPlaying = True
    host._df = DataFrame([{"frame_number": 999, "instance_name": "animal_1"}])
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    frame_json = frame_png.with_suffix(".json")
    frame_json.write_text("{}", encoding="utf-8")

    host.loadPredictShapes(0, str(frame_png))

    assert _FakeLabelFile.calls == 1
    assert len(host.loaded_shapes) == 1
    assert host.loaded_shapes[0].label == "from_label"


def test_tracking_fallback_applies_even_when_frame_image_exists_without_json(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        annotation_loading_module,
        "pred_dict_to_labelme",
        lambda row, **kwargs: [  # noqa: ARG005
            Shape(label=str(row.get("instance_name", "animal")), shape_type="point")
        ],
    )

    host = _PredictHost()
    host.isPlaying = False
    host._df = DataFrame([{"frame_number": 0, "instance_name": "animal_1"}])
    frame_png = tmp_path / "session_000000000.png"
    frame_png.write_bytes(b"")
    # No JSON file on purpose; CSV fallback is disabled.

    host.loadPredictShapes(0, str(frame_png))

    assert host.loaded_shapes == []


def test_annotation_store_has_frame_rechecks_stale_negative_cache(
    tmp_path: Path,
) -> None:
    host = _PredictHost()
    folder = tmp_path / "mouse"
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / "mouse_000000014.json"

    # Initial probe caches a negative before tracking writes frame 14.
    assert host._annotation_store_has_frame(str(candidate)) is False

    store = AnnotationStore.for_frame_path(candidate)
    store.append_frame(
        {
            "frame": 14,
            "version": "1.6.4",
            "imagePath": None,
            "imageHeight": 300,
            "imageWidth": 480,
            "shapes": [
                {
                    "label": "mouse_0",
                    "points": [[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "description": "",
                    "visible": True,
                    "mask": None,
                }
            ],
            "flags": {},
            "caption": None,
            "otherData": {},
        }
    )

    # Must not be stuck on the stale negative cache.
    assert host._annotation_store_has_frame(str(candidate)) is True


def test_annotation_store_has_frame_avoids_repeat_probe_until_store_changes(
    monkeypatch, tmp_path: Path
) -> None:
    host = _PredictHost()
    folder = tmp_path / "mouse"
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / "mouse_000000099.json"

    store = AnnotationStore.for_frame_path(candidate)
    store.append_frame(
        {
            "frame": 1,
            "version": "1.6.4",
            "imagePath": None,
            "imageHeight": 10,
            "imageWidth": 10,
            "shapes": [],
            "flags": {},
            "caption": None,
            "otherData": {},
        }
    )

    calls = {"count": 0}
    original_get_frame_fast = AnnotationStore.get_frame_fast

    def _counting_get_frame_fast(self, frame):  # noqa: ANN001
        calls["count"] += 1
        return original_get_frame_fast(self, frame)

    monkeypatch.setattr(
        AnnotationStore, "get_frame_fast", _counting_get_frame_fast, raising=True
    )

    assert host._annotation_store_has_frame(str(candidate)) is False
    assert host._annotation_store_has_frame(str(candidate)) is False
    assert calls["count"] == 1

    # Touch store (append different frame) to force signature change and recheck.
    store.append_frame(
        {
            "frame": 2,
            "version": "1.6.4",
            "imagePath": None,
            "imageHeight": 10,
            "imageWidth": 10,
            "shapes": [],
            "flags": {},
            "caption": None,
            "otherData": {},
        }
    )

    assert host._annotation_store_has_frame(str(candidate)) is False
    assert calls["count"] == 2
