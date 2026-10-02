from __future__ import annotations

from collections import Counter, defaultdict

import torch

from annolid.annotation.keypoints import format_shape, merge_shapes
from annolid.segmentation.yolos import InferenceProcessor


class _FakeBox:
    def __init__(self, cls_id: int) -> None:
        self.cls = int(cls_id)


class _FakeBoxes:
    def __init__(self, xywh, cls_ids, ids=None) -> None:
        self._xywh = torch.tensor(xywh, dtype=torch.float32)
        self._cls_ids = [int(c) for c in cls_ids]
        self.id = None if ids is None else torch.tensor(ids, dtype=torch.int64)

    @property
    def xywh(self):
        return self._xywh

    def __len__(self) -> int:
        return int(self._xywh.shape[0])

    def __iter__(self):
        return iter([_FakeBox(c) for c in self._cls_ids])


class _FakeKeypoints:
    def __init__(self, xy, conf) -> None:
        self.xy = torch.tensor(xy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeResult:
    def __init__(self, *, boxes, keypoints, names) -> None:
        self.boxes = boxes
        self.keypoints = keypoints
        self.names = names
        self.masks = None


def _build_processor(keypoint_names):
    proc = InferenceProcessor.__new__(InferenceProcessor)
    proc.model_type = "yolo"
    proc.keypoint_names = list(keypoint_names)
    proc.track_history = defaultdict(list)
    return proc


def test_extract_yolo_results_assigns_group_id_per_instance() -> None:
    processor = _build_processor(["nose", "tail", "paw"])
    result = _FakeResult(
        boxes=_FakeBoxes(
            xywh=[[10, 10, 5, 5], [20, 20, 6, 6]],
            cls_ids=[0, 0],
            ids=None,
        ),
        keypoints=_FakeKeypoints(
            xy=[
                [[1, 2], [3, 4], [5, 6]],
                [[7, 8], [9, 10], [11, 12]],
            ],
            conf=[
                [0.9, 0.8, 0.7],
                [0.6, 0.5, 0.4],
            ],
        ),
        names={0: "mouse"},
    )

    shapes = processor.extract_yolo_results(result)
    assert len(shapes) == 6
    assert {s.group_id for s in shapes} == {0, 1}

    label_counts = Counter(s.label for s in shapes)
    assert label_counts == {"nose": 2, "tail": 2, "paw": 2}

    for s in shapes:
        assert s.shape_type == "point"
        assert isinstance(s.group_id, int)
        assert s.other_data.get("instance_id") == s.group_id
        assert s.other_data.get("instance_label") in ("mouse_0", "mouse_1")
        assert isinstance(s.other_data.get("score"), float)

    # Ensure LabelMe merge logic will not collapse multi-instance keypoints.
    merged = merge_shapes([format_shape(s) for s in shapes], [])
    assert len(merged) == 6


def test_extract_yolo_results_uses_track_ids_as_group_id() -> None:
    processor = _build_processor(["nose", "tail"])
    result = _FakeResult(
        boxes=_FakeBoxes(
            xywh=[[10, 10, 5, 5], [20, 20, 6, 6]],
            cls_ids=[0, 0],
            ids=[42, 99],
        ),
        keypoints=_FakeKeypoints(
            xy=[
                [[1, 2], [3, 4]],
                [[7, 8], [9, 10]],
            ],
            conf=[
                [0.9, 0.8],
                [0.6, 0.5],
            ],
        ),
        names={0: "mouse"},
    )

    shapes = processor.extract_yolo_results(result)
    assert len(shapes) == 4
    assert {s.group_id for s in shapes} == {42, 99}
    assert {s.other_data.get("instance_label") for s in shapes} == {
        "mouse_42",
        "mouse_99",
    }


def test_extract_yolo_results_uses_prompt_class_names_override() -> None:
    processor = _build_processor(["nose"])
    processor.prompt_class_names = ["resident", "intruder"]
    result = _FakeResult(
        boxes=_FakeBoxes(
            xywh=[[10, 10, 5, 5], [20, 20, 6, 6]],
            cls_ids=[0, 1],
            ids=None,
        ),
        keypoints=_FakeKeypoints(
            xy=[
                [[1, 2]],
                [[7, 8]],
            ],
            conf=[
                [0.9],
                [0.6],
            ],
        ),
        names={0: "object0", 1: "object1"},
    )

    shapes = processor.extract_yolo_results(result)
    assert {s.other_data.get("instance_label") for s in shapes} == {
        "resident",
        "intruder",
    }


def test_detection_models_export_boxes_in_stream_and_frame_range(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import cv2
    import numpy as np

    result = _FakeResult(
        boxes=_FakeBoxes([[10, 12, 6, 8]], [0], ids=[42]),
        keypoints=None,
        names={0: "mouse"},
    )
    result.boxes.conf = torch.tensor([0.9])
    result.orig_shape = (32, 32)

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 2

        def set(self, prop, value):
            return True

        def read(self):
            return True, np.zeros((32, 32, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)
    source = tmp_path / "video.mp4"
    source.touch()
    for ranged in (False, True):
        processor = _build_processor([])
        processor.model_name = "yolo11n.pt"
        processor._is_coreml = False
        processor.persist_json = True
        processor.model = SimpleNamespace(
            predict=lambda source, **kwargs: iter([result])
            if kwargs.get("stream")
            else [result],
        )
        output = tmp_path / ("ranged" if ranged else "streamed")
        message = processor.run_inference(
            str(source),
            output_directory=output,
            enable_tracking=False,
            skip_existing=False,
            start_frame=1 if ranged else 0,
            end_frame=1 if ranged else None,
            save_pose_bbox=False,
        )
        assert message == "Done#1"
        frame_index = 1 if ranged else 0
        payload = json.loads(
            processor._labelme_json_path(output, frame_index=frame_index).read_text()
        )
        assert len(payload["shapes"]) == 1
        shape = payload["shapes"][0]
        assert shape["shape_type"] == "rectangle"
        assert shape["label"] == "mouse"
        assert shape["points"] == [[7.0, 8.0], [13.0, 16.0]]
        assert shape["group_id"] == 42
        assert shape["instance_label"] == "mouse"
        assert abs(shape["score"] - 0.9) < 1e-6
    assert capture.released


def test_bbox_export_policy_preserves_pose_and_segmentation():
    from types import SimpleNamespace

    pose = SimpleNamespace(keypoints=object(), masks=None)
    segmentation = SimpleNamespace(keypoints=None, masks=object())
    assert InferenceProcessor._should_save_yolo_bbox(pose, None)
    assert InferenceProcessor._should_save_yolo_bbox(pose, True)
    assert not InferenceProcessor._should_save_yolo_bbox(pose, False)
    assert not InferenceProcessor._should_save_yolo_bbox(segmentation, None)
