from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from annolid.segmentation.yolos import InferenceProcessor
from annolid.utils.annotation_store import AnnotationStore


class _TensorLike:
    def __init__(self, data):
        self._arr = np.asarray(data)

    def cpu(self):
        return self

    def int(self):
        return _TensorLike(self._arr.astype(int))

    def tolist(self):
        return self._arr.tolist()

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, idx):
        return _TensorLike(self._arr[idx])


class _FakeMask:
    def __init__(self, poly):
        self.xy = [np.asarray(poly, dtype=np.float32)]


class _FakeBoxes:
    def __init__(self, xywh, cls_ids, confs, track_ids=None):
        self.xywh = _TensorLike(xywh)
        self.cls = _TensorLike(cls_ids)
        self.conf = _TensorLike(confs)
        self.id = _TensorLike(track_ids) if track_ids is not None else None

    def __len__(self):
        return len(self.xywh)


class _FakeDetectionResult:
    def __init__(self, boxes, masks=None, names=None):
        self.boxes = boxes
        self.masks = masks
        self.names = names or {0: "mouse"}
        self.keypoints = None
        self.orig_shape = (32, 32)

    def plot(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


def _make_blank_processor() -> InferenceProcessor:
    proc = InferenceProcessor.__new__(InferenceProcessor)
    proc.model_name = "yolo11n-seg.pt"
    proc.model_type = "yolo"
    proc.class_names = None
    proc.prompt_class_names = None
    proc.keypoint_names = None
    proc.persist_json = True
    proc.frame_count = 0
    proc.detection_count = 0
    proc.track_history = defaultdict(list)
    proc.device = "cpu"
    proc.conf = 0.35
    proc.iou = 0.65
    proc.imgsz = 640
    proc.max_det = 50
    proc.half = None
    proc.model = None
    return proc


def test_build_ultralytics_infer_kwargs_merges_instance_and_call_overrides() -> None:
    proc = _make_blank_processor()
    # half=True is automatically suppressed on CPU to avoid unsupported FP16 CPU ops
    kwargs_cpu = proc._build_ultralytics_infer_kwargs(conf=0.5, half=True)
    assert kwargs_cpu == {
        "verbose": False,
        "conf": 0.5,
        "iou": 0.65,
        "imgsz": 640,
        "max_det": 50,
        "device": "cpu",
    }
    kwargs_gpu = proc._build_ultralytics_infer_kwargs(conf=0.5, device="0", half=True)
    assert kwargs_gpu["half"] is True
    assert kwargs_gpu["device"] == "0"


def test_extract_yolo_results_attaches_polygon_score_and_counts_detections() -> None:
    proc = _make_blank_processor()
    boxes = _FakeBoxes(
        xywh=[[16.0, 16.0, 10.0, 10.0], [20.0, 20.0, 8.0, 8.0]],
        cls_ids=[0, 0],
        confs=[0.91, 0.77],
        track_ids=[5, 8],
    )
    masks = [
        _FakeMask([[10, 10], [20, 10], [20, 20], [10, 20]]),
        _FakeMask([[15, 15], [25, 15], [25, 25], [15, 25]]),
    ]
    result = _FakeDetectionResult(boxes=boxes, masks=masks, names={0: "mouse"})

    shapes = proc.extract_yolo_results(result, save_bbox=False, save_track=False)

    assert len(shapes) == 2
    assert proc.detection_count == 2
    assert shapes[0].shape_type == "polygon"
    assert shapes[0].group_id == 5
    assert abs(shapes[0].other_data["score"] - 0.91) < 1e-6
    assert abs(shapes[1].other_data["score"] - 0.77) < 1e-6


def test_cv2_video_inference_batches_frames_respects_step_and_skips_store_entries(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    for idx in range(6):
        frame = np.full((32, 32, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    output_dir = tmp_path / "sample"
    output_dir.mkdir()
    # Pre-populate frame 2 inside AnnotationStore (no standalone JSON file!)
    store = AnnotationStore.for_frame_path(
        output_dir / f"{output_dir.name}_000000002.json"
    )
    store.append_frame(
        {
            "frame": 2,
            "imagePath": "",
            "imageHeight": 32,
            "imageWidth": 32,
            "shapes": [],
        }
    )

    batch_sizes_seen = []

    class _FakeModel:
        def predict(self, frames, **kwargs):
            if isinstance(frames, list):
                batch_sizes_seen.append(len(frames))
                count = len(frames)
            else:
                batch_sizes_seen.append(1)
                count = 1
            return [
                _FakeDetectionResult(
                    boxes=_FakeBoxes(
                        xywh=[[16.0, 16.0, 8.0, 8.0]],
                        cls_ids=[0],
                        confs=[0.88],
                    ),
                    masks=None,
                    names={0: "mouse"},
                )
                for _ in range(count)
            ]

    proc = _make_blank_processor()
    proc.model = _FakeModel()
    proc.model_name = "yolo11n.pt"
    proc.persist_json = False

    saved_indices = []

    def _fake_save(annotations, frame_shape, out_dir, *, frame_index=None):
        saved_indices.append(int(frame_index))

    monkeypatch.setattr(proc, "save_yolo_to_labelme", _fake_save)

    progress_events = []
    msg = proc._run_yolo_video_inference_cv2(
        source=str(video_path),
        visual_prompts=None,
        start_frame=0,
        end_frame=5,
        step=2,  # candidate frames: 0, 2 (skipped via store), 4
        skip_existing=True,
        pred_worker=None,
        stop_event=None,
        output_directory=output_dir,
        progress_callback=lambda done, total: progress_events.append((done, total)),
        enable_tracking=False,
        tracker=None,
        save_pose_bbox=None,
        conf=0.4,
        batch_size=2,
        save_video_path=None,
    )

    assert msg == "Done#3"
    assert saved_indices == [0, 4]
    assert batch_sizes_seen == [2]
    assert proc.detection_count == 2
    assert progress_events[-1] == (3, 3)


def test_streaming_inference_emits_progress_callback_and_counts_detections(
    tmp_path: Path, monkeypatch
) -> None:
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), np.zeros((32, 32, 3), dtype=np.uint8))

    class _FakeStreamModel:
        def track(self, source, **kwargs):
            assert kwargs["stream"] is True
            yield _FakeDetectionResult(
                boxes=_FakeBoxes(
                    xywh=[[16.0, 16.0, 6.0, 6.0]],
                    cls_ids=[0],
                    confs=[0.9],
                ),
                names={0: "mouse"},
            )

    proc = _make_blank_processor()
    proc.model = _FakeStreamModel()
    proc.model_name = "yolo11n.pt"
    monkeypatch.setattr(proc, "save_yolo_to_labelme", lambda *a, **kw: None)

    progress_events = []
    out = proc.run_inference(
        source=str(img_path),
        output_directory=tmp_path / "out",
        progress_callback=lambda done, total: progress_events.append((done, total)),
        enable_tracking=True,
    )

    assert out == "Done#1"
    assert proc.detection_count == 1
    assert progress_events == [(1, 1)]
