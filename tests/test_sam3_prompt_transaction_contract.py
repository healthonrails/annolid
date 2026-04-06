from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from annolid.gui.widgets.sam3_manager import Sam3Manager
from annolid.segmentation.SAM.sam3.session import Sam3SessionManager
from annolid.segmentation.SAM.sam3.sam3.model.sam3_base_predictor import (
    Sam3BasePredictor,
)
from annolid.segmentation.SAM.sam3.sam3.model.multiplex_mask_decoder import (
    MultiplexMaskDecoder,
)
from annolid.segmentation.SAM.sam3.sam3.model.video_tracking_multiplex_demo import (
    VideoTrackingMultiplexDemo,
    _safe_slice_first_dim,
)
from annolid.segmentation.SAM.sam3.sam3.model.sam3_multiplex_base import (
    _ensure_object_masks,
    _select_positive_detections,
)
from annolid.segmentation.SAM.sam_v2 import (
    load_manual_seed_annotations_from_video,
)


class _Point:
    def __init__(self, x: float, y: float) -> None:
        self._x = float(x)
        self._y = float(y)

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y


class _Shape:
    def __init__(
        self,
        *,
        shape_type: str,
        points: list[_Point],
        label: str,
        group_id: int | None = None,
        point_labels: list[int] | None = None,
    ) -> None:
        self.shape_type = shape_type
        self.points = points
        self.label = label
        self.group_id = group_id
        self.point_labels = point_labels or []


class _MismatchedTransformer(torch.nn.Module):
    def forward(self, src, pos_src, tokens):
        b, c, h, w = src.shape
        hs = src.new_zeros((tokens.shape[0], tokens.shape[1], c))
        src_out = src.new_zeros((0, c, h, w))
        return hs, src_out


class _CaptureMaskModel:
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        points=None,
        point_labels=None,
        clear_old_points=True,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        mask_inputs=None,
        mask_labels=None,
        output_prob_thresh=0.5,
    ):
        return frame_idx, {
            "inference_state": inference_state,
            "frame_idx": frame_idx,
            "text_str": text_str,
            "points": points,
            "point_labels": point_labels,
            "clear_old_points": clear_old_points,
            "boxes_xywh": boxes_xywh,
            "box_labels": box_labels,
            "clear_old_boxes": clear_old_boxes,
            "mask_inputs": mask_inputs,
            "mask_labels": mask_labels,
            "output_prob_thresh": output_prob_thresh,
        }


class _FakeSAM3VideoProcessor:
    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init_kwargs = dict(kwargs)

    def run(self, stop_event=None):  # noqa: ANN001
        return 1, 2

    def close_session(self, session_id=None):  # noqa: ANN001
        return None

    def request_stop(self):  # noqa: D401
        return None


def test_prompt_transaction_merge_keeps_all_prompt_steps() -> None:
    merged = Sam3SessionManager._merge_prompt_outputs(
        [
            {
                "out_obj_ids": np.asarray([1], dtype=np.int64),
                "out_probs": np.asarray([0.40], dtype=np.float32),
                "out_boxes_xywh": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8)], dtype=object
                ),
            },
            {
                "out_obj_ids": np.asarray([1, 2], dtype=np.int64),
                "out_probs": np.asarray([0.90, 0.70], dtype=np.float32),
                "out_boxes_xywh": np.asarray(
                    [[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.asarray(
                    [
                        np.zeros((2, 2), dtype=np.uint8),
                        np.ones((2, 2), dtype=np.uint8),
                    ],
                    dtype=object,
                ),
            },
        ]
    )

    assert merged["out_obj_ids"].tolist() == [1, 2]
    assert np.isclose(float(merged["out_probs"][0]), 0.90)
    assert np.isclose(float(merged["out_probs"][1]), 0.70)
    assert merged["out_boxes_xywh"].shape == (2, 4)
    assert merged["out_binary_masks"].dtype == object


def test_canvas_prompt_extraction_preserves_polygons_and_group_ids() -> None:
    manager = Sam3Manager.__new__(Sam3Manager)
    manager.window = SimpleNamespace(
        frame_number=7,
        canvas=SimpleNamespace(
            shapes=[
                _Shape(
                    shape_type="rectangle",
                    points=[_Point(10, 20), _Point(30, 50)],
                    label="mouse",
                    group_id=5,
                ),
                _Shape(
                    shape_type="polygon",
                    points=[_Point(1, 1), _Point(4, 1), _Point(4, 4), _Point(1, 4)],
                    label="mouse",
                    group_id=8,
                ),
                _Shape(
                    shape_type="rectangle",
                    points=[_Point(5, 5), _Point(10, 12)],
                    label="mouse",
                    group_id=9,
                ),
                _Shape(
                    shape_type="points",
                    points=[_Point(12, 14), _Point(16, 18)],
                    label="mouse",
                    group_id=8,
                    point_labels=[1, 0],
                ),
            ]
        ),
    )

    prompts = manager.extract_prompts_from_canvas()
    assert prompts["frame_idx"] == 7
    assert prompts["boxes_abs"] == [[10, 20, 20, 30], [5, 5, 5, 7]]
    assert prompts["polygons_abs"] == [[[1, 1], [4, 1], [4, 4], [1, 4]]]
    assert prompts["point_labels"] == [1, 0]

    annotations = manager._canvas_prompts_to_annotations(
        frame_idx=7,
        id_to_labels={5: "mouse", 8: "mouse"},
    )
    assert len(annotations) == 4
    assert any(a["type"] == "polygon" and a["obj_id"] == 8 for a in annotations)
    assert any(a["type"] == "box" and a["obj_id"] == 5 for a in annotations)
    assert any(a["type"] == "box" and a["obj_id"] == 9 for a in annotations)
    point_ann = next(a for a in annotations if a["type"] == "points")
    assert point_ann["obj_id"] == 8
    assert point_ann["labels"] == [1, 0]


def test_canvas_polygon_prefers_mask_annotation_when_frame_image_available() -> None:
    manager = Sam3Manager.__new__(Sam3Manager)
    manager.window = SimpleNamespace(
        frame_number=3,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
        canvas=SimpleNamespace(
            shapes=[
                _Shape(
                    shape_type="polygon",
                    points=[_Point(1, 1), _Point(10, 1), _Point(10, 8), _Point(1, 8)],
                    label="vole",
                    group_id=2,
                ),
                _Shape(
                    shape_type="rectangle",
                    points=[_Point(12, 2), _Point(20, 9)],
                    label="vole_rect",
                    group_id=3,
                ),
            ]
        ),
    )

    annotations = manager._canvas_prompts_to_annotations(
        frame_idx=3,
        id_to_labels={},
    )
    assert len(annotations) == 2
    assert {ann["obj_id"] for ann in annotations} == {2, 3}
    assert all(ann["type"] == "mask" for ann in annotations)
    assert all(ann["mask"].shape == (24, 32) for ann in annotations)
    assert all(int(np.asarray(ann["mask"]).sum()) > 0 for ann in annotations)


def test_canvas_rectangle_prefers_mask_annotation_when_frame_image_available() -> None:
    manager = Sam3Manager.__new__(Sam3Manager)
    manager.window = SimpleNamespace(
        frame_number=5,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
        canvas=SimpleNamespace(
            shapes=[
                _Shape(
                    shape_type="rectangle",
                    points=[_Point(2, 3), _Point(11, 14)],
                    label="vole",
                    group_id=4,
                ),
            ]
        ),
    )

    annotations = manager._canvas_prompts_to_annotations(
        frame_idx=5,
        id_to_labels={},
    )
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["type"] == "mask"
    assert ann["obj_id"] == 4
    assert ann["mask"].shape == (24, 32)
    assert int(np.asarray(ann["mask"]).sum()) > 0


def test_build_video_processor_merges_live_canvas_seed_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from annolid.segmentation.SAM.sam3 import adapter as sam3_adapter

    video_dir = tmp_path / "video"
    video_dir.mkdir()
    manager = Sam3Manager.__new__(Sam3Manager)
    manager.window = SimpleNamespace(
        video_file=str(video_dir),
        frame_number=42,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
        epsilon_for_polygon=0.0,
        _config={
            "sam3": {
                "use_explicit_window_reseed": False,
                "allow_private_state_mutation": True,
                "boundary_mask_match_iou_threshold": 0.33,
            }
        },
        _current_text_prompt=lambda: None,
        canvas=SimpleNamespace(
            shapes=[
                _Shape(
                    shape_type="polygon",
                    points=[_Point(1, 1), _Point(10, 1), _Point(10, 8), _Point(1, 8)],
                    label="vole_a",
                    group_id=10,
                ),
                _Shape(
                    shape_type="polygon",
                    points=[_Point(12, 1), _Point(20, 1), _Point(20, 8), _Point(12, 8)],
                    label="vole_b",
                    group_id=11,
                ),
                _Shape(
                    shape_type="polygon",
                    points=[
                        _Point(1, 10),
                        _Point(10, 10),
                        _Point(10, 18),
                        _Point(1, 18),
                    ],
                    label="vole_c",
                    group_id=12,
                ),
            ]
        ),
    )

    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam_v2.load_manual_seed_annotations_from_video",
        lambda *_args, **_kwargs: (
            [
                {"type": "box", "ann_frame_idx": 42, "box": [0, 0, 5, 5], "obj_id": 99},
                {"type": "box", "ann_frame_idx": 7, "box": [3, 3, 6, 6], "obj_id": 7},
            ],
            {99: "stale", 7: "persist"},
        ),
    )
    monkeypatch.setattr(sam3_adapter, "SAM3VideoProcessor", _FakeSAM3VideoProcessor)

    runner = manager.build_video_processor("sam3", "weights.pt", None)
    assert callable(runner)
    result = runner()
    assert result == (1, 2)

    kwargs = _FakeSAM3VideoProcessor.last_init_kwargs
    assert kwargs is not None
    annotations = kwargs["annotations"]
    assert any(int(ann["ann_frame_idx"]) == 7 for ann in annotations)
    assert any(
        int(ann["ann_frame_idx"]) == 42 and int(ann.get("obj_id", -1)) == 99
        for ann in annotations
    )
    assert sum(1 for ann in annotations if int(ann["ann_frame_idx"]) == 42) == 4
    assert kwargs["id_to_labels"][10] == "vole_a"
    assert kwargs["id_to_labels"][11] == "vole_b"
    assert kwargs["id_to_labels"][12] == "vole_c"
    assert kwargs["use_explicit_window_reseed"] is False
    assert kwargs["boundary_mask_match_iou_threshold"] == 0.33
    assert kwargs["allow_private_state_mutation"] is True


def test_merge_canvas_annotations_overrides_only_matching_prompt_key() -> None:
    saved = [
        {"type": "box", "ann_frame_idx": 42, "box": [0, 0, 10, 10], "obj_id": 1},
        {"type": "box", "ann_frame_idx": 42, "box": [5, 5, 12, 12], "obj_id": 2},
        {"type": "box", "ann_frame_idx": 7, "box": [2, 2, 6, 6], "obj_id": 3},
    ]
    canvas = [
        {"type": "box", "ann_frame_idx": 42, "box": [1, 1, 9, 9], "obj_id": 1},
    ]

    merged = Sam3Manager._merge_canvas_annotations(saved, canvas)

    assert len(merged) == 3
    assert any(
        int(ann["ann_frame_idx"]) == 42 and int(ann["obj_id"]) == 2 for ann in merged
    )
    assert any(
        int(ann["ann_frame_idx"]) == 42
        and int(ann["obj_id"]) == 1
        and ann["box"] == [1, 1, 9, 9]
        for ann in merged
    )
    assert any(
        int(ann["ann_frame_idx"]) == 7 and int(ann["obj_id"]) == 3 for ann in merged
    )


def test_dialog_defaults_keep_agent_output_dir_disabled_by_default() -> None:
    manager = Sam3Manager.__new__(Sam3Manager)
    manager.score_threshold_detection = None
    manager.new_det_thresh = None
    manager.propagation_direction = None
    manager.max_frame_num_to_track = None
    manager.device_override = None
    manager.sliding_window_size = None
    manager.sliding_window_stride = None
    manager.compile_model = None
    manager.offload_video_to_cpu = None
    manager.use_explicit_window_reseed = None
    manager.boundary_mask_match_iou_threshold = None
    manager.allow_private_state_mutation = None
    manager.max_num_objects = None
    manager.multiplex_count = None
    manager.agent_det_thresh = None
    manager.agent_window_size = None
    manager.agent_stride = None
    manager.agent_output_dir = None

    defaults = manager.dialog_defaults({})

    assert defaults["agent_output_dir"] is None
    assert defaults["boundary_mask_match_iou_threshold"] == 0.2


def test_manual_seed_loader_uses_png_json_pair_and_ignores_store(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    png_path = folder / f"{folder.name}_000000042.png"
    png_path.write_bytes(b"png")
    json_path = png_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "flags": {},
                "imagePath": png_path.name,
                "imageHeight": 24,
                "imageWidth": 32,
                "shapes": [
                    {
                        "label": "vole_a",
                        "points": [[1, 1], [10, 1], [10, 8], [1, 8]],
                        "group_id": 10,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "vole_b",
                        "points": [[12, 1], [20, 1], [20, 8], [12, 8]],
                        "group_id": 11,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "vole_c",
                        "points": [[1, 10], [10, 10], [10, 18], [1, 18]],
                        "group_id": 12,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "vole_rect",
                        "points": [[2, 12], [8, 18]],
                        "group_id": 13,
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    store_path = folder / f"{folder.name}_annotations.ndjson"
    store_path.write_text(
        json.dumps(
            {
                "frame": 42,
                "shapes": [
                    {
                        "label": "prediction",
                        "points": [[0, 0], [1, 1]],
                        "shape_type": "polygon",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    annotations, id_to_labels = load_manual_seed_annotations_from_video(folder)
    seed_annotations = [ann for ann in annotations if int(ann["ann_frame_idx"]) == 42]

    assert len(seed_annotations) == 4
    assert sorted(id_to_labels) == [10, 11, 12, 13]
    assert id_to_labels[10] == "vole_a"
    assert id_to_labels[11] == "vole_b"
    assert id_to_labels[12] == "vole_c"
    assert id_to_labels[13] == "vole_rect"
    assert {int(ann["obj_id"]) for ann in seed_annotations} == {10, 11, 12, 13}
    assert all(ann["type"] == "mask" for ann in seed_annotations)


def test_apply_seed_prompts_does_not_mix_text_into_structured_manual_seed() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.text_prompt = "black vole"
    session.id_to_labels = {7: "vole"}
    session._record_seed_frame_if_manual = lambda *args, **kwargs: None

    captured: dict[str, object] = {}

    def _fake_add_prompt(**kwargs):
        captured["text"] = kwargs.get("text")
        captured["boxes"] = kwargs.get("boxes")
        captured["mask_inputs"] = kwargs.get("mask_inputs")
        return {
            "outputs": {
                "out_obj_ids": np.asarray([7], dtype=np.int64),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8)], dtype=object
                ),
            }
        }

    session.add_prompt = _fake_add_prompt  # type: ignore[method-assign]

    mask = np.ones((4, 4), dtype=np.uint8)
    count = session._apply_seed_prompts(
        frame_idx=39,
        session_id="session-1",
        boxes=[],
        labels=[],
        mask_inputs=[mask],
        mask_labels=[1],
        points=[],
        point_labels=[],
        point_obj_ids=[],
        label_hints=["vole"],
    )

    assert captured["text"] is None
    assert captured["boxes"] is None
    assert isinstance(captured["mask_inputs"], list)
    assert count == 1


def test_window_annotation_shift_groups_by_local_frame() -> None:
    grouped = Sam3SessionManager._shift_annotations_to_window(
        [
            {"type": "box", "ann_frame_idx": 3, "box": [1, 2, 3, 4]},
            {
                "type": "polygon",
                "ann_frame_idx": 4,
                "polygon": [[1, 1], [2, 1], [2, 2]],
            },
            {"type": "points", "ann_frame_idx": 8, "points": [[5, 5]]},
        ],
        start_idx=3,
        end_idx=7,
    )

    assert sorted(grouped.keys()) == [0, 1]
    assert grouped[0][0]["ann_frame_idx"] == 0
    assert grouped[1][0]["ann_frame_idx"] == 1
    assert grouped[0][0]["box"] == [1, 2, 3, 4]


def test_polygon_seed_annotations_expand_to_dense_mask_prompt() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.frame_shape = (100, 100, 3)
    session._predictor = SimpleNamespace(
        add_prompt=lambda **kwargs: {"frame_index": kwargs["frame_idx"], "outputs": {}}
    )
    session._session_id = "session-1"
    session.id_to_labels = {7: "mouse"}

    (
        prompt_frame_idx,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        obj_ids,
        point_obj_ids,
    ) = session._prepare_prompts(
        [
            {
                "type": "polygon",
                "ann_frame_idx": 4,
                "polygon": [[10, 10], [30, 10], [30, 40], [10, 40]],
                "labels": [7],
                "obj_id": 7,
            }
        ],
        text_prompt=None,
    )

    assert prompt_frame_idx == 4
    assert boxes == []
    assert box_labels == []
    assert len(mask_inputs) == 1
    assert mask_inputs[0].shape == (100, 100)
    assert int(mask_inputs[0].sum()) > 0
    assert mask_labels == [1]
    assert obj_ids == [7]
    assert points == []
    assert point_labels == []
    assert point_obj_ids == []

    transaction = session._execute_prompt_transaction(
        session_id="session-1",
        frame_idx=prompt_frame_idx,
        text=None,
        boxes=boxes,
        box_labels=box_labels,
        mask_inputs=mask_inputs,
        mask_labels=mask_labels,
        points=points,
        point_labels=point_labels,
        obj_id=None,
    )
    assert transaction["transaction_step_kinds"] == ["semantic"]
    assert transaction["transaction_steps"][0]["prompt_kind"] == "semantic"
    assert transaction["outputs"] == {}


def test_window_telemetry_entry_defaults_boundary_skips_to_zero() -> None:
    telemetry = Sam3SessionManager._build_window_telemetry_entry(
        window_index=2,
        window_start_idx=10,
        window_end_idx=20,
        local_mask_counts={10: 2, 11: 0, 12: 1},
        latency_ms=42.0,
    )

    assert telemetry["window_index"] == 2
    assert telemetry["start"] == 10
    assert telemetry["end"] == 20
    assert telemetry["frames"] == 3
    assert telemetry["nonzero_frames"] == 2
    assert telemetry["zero_mask_frames"] == 1
    assert telemetry["boundary_empty_skips"] == 0
    assert telemetry["latency_ms"] == 42.0


def test_add_prompt_supports_explicit_window_session_id() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session._predictor = object()
    session._session_id = None
    session.frame_names = []
    session.max_frame_num_to_track = 0

    captured: dict[str, object] = {}

    def _execute_prompt_transaction(**kwargs):
        captured["session_id"] = kwargs["session_id"]
        return {"transaction_steps": [{"outputs": {}}]}

    def _handle_frame_outputs(**kwargs):
        captured["recorded"] = True
        captured["recorded_frame_idx"] = kwargs.get("frame_idx")
        return 0, False

    session._execute_prompt_transaction = _execute_prompt_transaction
    session._handle_frame_outputs = _handle_frame_outputs

    session.add_prompt(
        frame_idx=0,
        session_id="window-session-1",
        text="mouse",
        record_outputs=True,
        record_frame_idx=42,
        merge_existing_on_record=False,
        label_hints=["mouse"],
    )

    assert captured["session_id"] == "window-session-1"
    assert captured["recorded"] is True
    assert captured["recorded_frame_idx"] == 42


def test_should_accept_sam3_mask_rejects_full_frame_or_far_drift() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.frame_shape = (20, 20, 3)
    session._track_last_seen_frame = {7: 5}
    session._frame_masks = {
        5: {
            "7": np.pad(
                np.ones((4, 4), dtype=np.uint8),
                ((2, 14), (3, 13)),
                mode="constant",
            )
        }
    }

    full_frame = np.ones((20, 20), dtype=np.uint8)
    far_mask = np.pad(
        np.ones((3, 3), dtype=np.uint8),
        ((15, 2), (15, 2)),
        mode="constant",
    )
    near_mask = np.pad(
        np.ones((4, 4), dtype=np.uint8),
        ((2, 14), (3, 13)),
        mode="constant",
    )

    assert session._should_accept_sam3_mask(
        frame_idx=6,
        obj_id=7,
        mask=near_mask,
        box_xywh=np.asarray([3.0, 2.0, 4.0, 4.0], dtype=float),
    )
    assert not session._should_accept_sam3_mask(
        frame_idx=6,
        obj_id=7,
        mask=full_frame,
        box_xywh=np.asarray([0.0, 0.0, 20.0, 20.0], dtype=float),
    )
    assert not session._should_accept_sam3_mask(
        frame_idx=6,
        obj_id=7,
        mask=far_mask,
        box_xywh=np.asarray([15.0, 15.0, 3.0, 3.0], dtype=float),
    )


def test_should_accept_sam3_mask_allows_first_observation_without_history() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.frame_shape = (20, 20, 3)
    session._track_last_seen_frame = {}
    session._frame_masks = {}

    first_mask = np.pad(
        np.ones((4, 4), dtype=np.uint8),
        ((2, 14), (3, 13)),
        mode="constant",
    )

    assert session._should_accept_sam3_mask(
        frame_idx=6,
        obj_id=7,
        mask=first_mask,
        box_xywh=np.asarray([15.0, 15.0, 3.0, 3.0], dtype=float),
    )


def test_handle_frame_outputs_falls_back_to_recent_mask_when_implausible() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.frame_shape = (20, 20, 3)
    session.video_dir = "/tmp/sam3"
    session.obj_id_to_label = {}
    session.id_to_labels = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {
        5: {
            "7": np.pad(
                np.ones((4, 4), dtype=np.uint8),
                ((2, 14), (3, 13)),
                mode="constant",
            )
        }
    }
    session._frame_track_ids = {}
    session._track_last_seen_frame = {7: 5}
    session._last_mask_area_ratio = {}
    session.text_prompt = None
    session.score_threshold_detection = None
    session.get_frame_shape = lambda: (20, 20, 3)
    captured: dict[str, object] = {}

    def _save_annotations(filename, mask_dict, frame_shape, **kwargs):
        captured["mask_dict"] = mask_dict
        return None

    session._save_annotations = _save_annotations

    session._handle_frame_outputs(
        frame_idx=6,
        outputs={
            "out_obj_ids": np.asarray([7], dtype=np.int64),
            "out_probs": np.asarray([0.95], dtype=np.float32),
            "out_boxes_xywh": np.asarray([[15.0, 15.0, 3.0, 3.0]], dtype=np.float32),
            "out_binary_masks": np.asarray(
                [
                    np.pad(
                        np.ones((2, 2), dtype=np.uint8),
                        ((18, 0), (18, 0)),
                        mode="constant",
                    )
                ],
                dtype=object,
            ),
        },
        total_frames=20,
        yielded_frames=1,
        apply_score_threshold=False,
    )

    mask_dict = captured["mask_dict"]
    assert "7" in mask_dict
    assert int(np.asarray(mask_dict["7"]).sum()) == int(
        np.asarray(session._frame_masks[5]["7"]).sum()
    )


def test_ensure_prediction_json_coverage_skips_processed_frame_validation_fast_path(
    tmp_path,
) -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_dir = str(tmp_path)
    session.frame_shape = (20, 20, 3)
    session.get_frame_shape = lambda: (20, 20, 3)

    processed_path = tmp_path / "000000005.json"
    processed_path.write_text("{not-valid-json", encoding="utf-8")

    captured: list[int] = []

    def _save_annotations(filename, mask_dict, frame_shape, **kwargs):
        captured.append(int(kwargs.get("frame_idx", -1)))
        Path(filename).write_text("{}", encoding="utf-8")
        return None

    session._save_annotations = _save_annotations

    repaired, invalid = session._ensure_prediction_json_coverage(
        expected_frames=[5, 6],
        processed_frames={5},
        verify_processed_frames=False,
    )

    assert repaired == 1
    assert invalid == 0
    assert captured == [6]


def test_apply_seed_prompts_returns_materialized_mask_count() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.text_prompt = "vole"
    session.id_to_labels = {1: "vole"}

    responses = iter(
        [
            {
                "transaction_steps": [
                    {
                        "outputs": {
                            "out_obj_ids": np.asarray([1], dtype=np.int64),
                            "out_binary_masks": np.asarray(
                                [np.ones((2, 2), dtype=np.uint8)],
                                dtype=object,
                            ),
                        }
                    }
                ]
            },
            {
                "transaction_steps": [
                    {
                        "outputs": {
                            "out_obj_ids": np.asarray([], dtype=np.int64),
                            "out_binary_masks": np.asarray([], dtype=object),
                        }
                    }
                ]
            },
        ]
    )

    def _add_prompt(**kwargs):
        return next(responses)

    session.add_prompt = _add_prompt

    total_masks = session._apply_seed_prompts(
        frame_idx=0,
        session_id="window-1",
        boxes=[[0.1, 0.1, 0.4, 0.4]],
        labels=[1],
        mask_inputs=[],
        mask_labels=[],
        points=[[0.2, 0.2]],
        point_labels=[1],
        point_obj_ids=[1],
        label_hints=["vole"],
    )

    assert total_masks == 1


def test_prepare_prompts_falls_back_to_text_when_annotations_are_unusable() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.frame_shape = (64, 64, 3)
    session.video_dir = "/tmp/sam3"
    session.frame_names = []

    (
        prompt_frame_idx,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        obj_ids,
        point_obj_ids,
    ) = session._prepare_prompts(
        [
            {
                "type": "unsupported",
                "ann_frame_idx": 7,
                "labels": [1],
                "obj_id": 1,
            }
        ],
        text_prompt="vole",
    )

    assert prompt_frame_idx == 0
    assert boxes == []
    assert box_labels == []
    assert mask_inputs == []
    assert mask_labels == []
    assert points == []
    assert point_labels == []
    assert obj_ids == []
    assert point_obj_ids == []


def test_apply_seed_prompts_uses_text_only_when_no_other_prompt_formats_exist() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.text_prompt = "vole"
    session.id_to_labels = {}

    captured: dict[str, object] = {}

    def _add_prompt(**kwargs):
        captured["kwargs"] = kwargs
        return {"transaction_steps": [{"outputs": {}}]}

    session.add_prompt = _add_prompt

    total_masks = session._apply_seed_prompts(
        frame_idx=0,
        session_id="window-1",
        boxes=[],
        labels=[],
        mask_inputs=[],
        mask_labels=[],
        points=[],
        point_labels=[],
        point_obj_ids=[],
        label_hints=[],
    )

    assert total_masks == 0
    kwargs = captured["kwargs"]
    assert kwargs["text"] == "vole"
    assert kwargs["boxes"] is None
    assert kwargs["mask_inputs"] is None
    assert kwargs.get("points") is None
    assert kwargs.get("point_labels") is None


def test_base_predictor_expands_mask_batches_for_3d_manual_seed_inputs() -> None:
    predictor = Sam3BasePredictor.__new__(Sam3BasePredictor)
    predictor.model = _CaptureMaskModel()
    predictor._all_inference_states = {}
    predictor._get_session = lambda _session_id: {"state": {}}
    predictor._extend_expiration_time = lambda _session: None

    mask_inputs = [
        np.ones((6, 8), dtype=np.uint8),
        np.zeros((6, 8), dtype=np.uint8),
        np.pad(np.ones((2, 2), dtype=np.uint8), ((2, 2), (3, 3)), mode="constant"),
    ]

    result = predictor.add_prompt(
        session_id="session-1",
        frame_idx=39,
        mask_inputs=mask_inputs,
        mask_labels=[1, 1, 1],
    )

    kwargs = result["outputs"]
    assert isinstance(kwargs["mask_inputs"], torch.Tensor)
    assert tuple(kwargs["mask_inputs"].shape) == (3, 1, 6, 8)


def test_carry_forward_window_state_shifts_overlap_frames() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.allow_private_state_mutation = True
    current_state: dict[str, object] = {
        "cached_features": {},
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        },
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
        "frames_already_tracked": {},
        "user_refined_frames_per_obj": {},
        "obj_id_to_idx": {},
        "obj_idx_to_id": {},
        "obj_ids": [],
    }
    session._predictor = SimpleNamespace(
        raw=SimpleNamespace(_get_session=lambda _sid: {"state": current_state})
    )
    session._session_id = "window-1"

    previous_state = {
        "cached_features": {
            1: ("image-1", "feat-1"),
            2: ("image-2", "feat-2"),
        },
        "point_inputs_per_obj": {3: {1: "p1", 2: "p2"}},
        "mask_inputs_per_obj": {3: {1: "m1", 2: "m2"}},
        "output_dict": {
            "cond_frame_outputs": {1: "c1", 2: "c2"},
            "non_cond_frame_outputs": {1: "n1", 2: "n2"},
        },
        "output_dict_per_obj": {
            3: {
                "cond_frame_outputs": {1: "oc1", 2: "oc2"},
                "non_cond_frame_outputs": {1: "on1", 2: "on2"},
            }
        },
        "temp_output_dict_per_obj": {
            3: {
                "cond_frame_outputs": {1: "tc1", 2: "tc2"},
                "non_cond_frame_outputs": {1: "tn1", 2: "tn2"},
            }
        },
        "consolidated_frame_inds": {
            "cond_frame_outputs": {1, 2},
            "non_cond_frame_outputs": {1, 2},
        },
        "frames_already_tracked": {1: True, 2: True},
        "user_refined_frames_per_obj": {3: {1, 2}},
        "obj_id_to_idx": {7: 0},
        "obj_idx_to_id": {0: 7},
        "obj_ids": [7],
    }

    session._carry_forward_window_state(previous_state, shift=1)

    assert current_state["cached_features"] == {
        0: ("image-1", "feat-1"),
        1: ("image-2", "feat-2"),
    }
    assert current_state["output_dict"]["cond_frame_outputs"] == {0: "c1", 1: "c2"}
    assert current_state["point_inputs_per_obj"] == {}
    assert current_state["mask_inputs_per_obj"] == {}
    assert current_state["consolidated_frame_inds"]["non_cond_frame_outputs"] == {0, 1}
    assert current_state["obj_ids"] == [7]


def test_execute_prompt_transaction_allows_optional_mask_and_point_prompts() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session._predictor = SimpleNamespace(
        add_prompt=lambda **kwargs: {"outputs": {}, "kwargs": kwargs}
    )
    session._build_prompt_transaction_steps = lambda **kwargs: [
        {"kind": "semantic", "text": kwargs.get("text")}
    ]

    result = session._execute_prompt_transaction(
        session_id="window-1",
        frame_idx=0,
        text="vole",
        boxes=None,
        box_labels=None,
        points=None,
    )

    assert result["transaction_step_kinds"] == ["semantic"]
    assert result["outputs"] == {}


def test_resolve_runtime_device_does_not_mutate_global_torch_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.default_device = "cpu"

    set_default_device_calls = {"count": 0}
    set_default_dtype_calls = {"count": 0}

    def _count_set_default_device(*_args, **_kwargs):
        set_default_device_calls["count"] += 1

    def _count_set_default_dtype(*_args, **_kwargs):
        set_default_dtype_calls["count"] += 1

    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.session.select_device",
        lambda _preferred: torch.device("cpu"),
    )
    monkeypatch.setattr(torch, "set_default_device", _count_set_default_device)
    monkeypatch.setattr(torch, "set_default_dtype", _count_set_default_dtype)

    resolved = session._resolve_runtime_device(None)

    assert resolved.type == "cpu"
    assert set_default_device_calls["count"] == 0
    assert set_default_dtype_calls["count"] == 0


def test_vendored_model_builder_exposes_build_sam3_predictor() -> None:
    import importlib

    module = importlib.import_module("annolid.segmentation.SAM.sam3.sam3.model_builder")
    assert callable(getattr(module, "build_sam3_predictor", None))


def test_vendored_model_builder_hotstart_defaults_align_with_upstream() -> None:
    import importlib

    module = importlib.import_module("annolid.segmentation.SAM.sam3.sam3.model_builder")
    delay, unmatch, dup = module._resolve_hotstart_params()
    assert (delay, unmatch, dup) == (0, 0, 0)


def test_vendored_model_builder_hotstart_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module("annolid.segmentation.SAM.sam3.sam3.model_builder")
    monkeypatch.setenv("ANNOLID_SAM3_HOTSTART_DELAY", "12")
    monkeypatch.setenv("ANNOLID_SAM3_HOTSTART_UNMATCH_THRESH", "9")
    monkeypatch.setenv("ANNOLID_SAM3_HOTSTART_DUP_THRESH", "20")
    delay, unmatch, dup = module._resolve_hotstart_params()
    assert delay == 12
    # thresholds are clamped to delay for consistency
    assert unmatch == 9
    assert dup == 12


def test_reset_action_history_is_disabled_without_private_state_mutation() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.allow_private_state_mutation = False
    state = {"action_history": [1, 2, 3]}
    session._predictor = SimpleNamespace(
        raw=SimpleNamespace(_all_inference_states={"window-1": {"state": state}})
    )
    session._session_id = "window-1"

    session._reset_action_history_if_supported()

    assert state["action_history"] == [1, 2, 3]


def test_reset_action_history_is_enabled_with_private_state_mutation() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.allow_private_state_mutation = True
    state = {"action_history": [1, 2, 3]}
    session._predictor = SimpleNamespace(
        raw=SimpleNamespace(_all_inference_states={"window-1": {"state": state}})
    )
    session._session_id = "window-1"

    session._reset_action_history_if_supported()

    assert state["action_history"] == []


def test_build_boundary_reseed_boxes_prefers_expected_track_masks() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 5
    session.sliding_window_stride = 4
    session.max_num_objects = 4
    session._frame_track_ids = {10: {7}, 11: {7, 8}}
    session._frame_masks = {
        10: {
            "7": np.pad(
                np.ones((4, 4), dtype=np.uint8), ((2, 10), (3, 9)), mode="constant"
            )
        },
        11: {
            "7": np.pad(
                np.ones((4, 4), dtype=np.uint8), ((3, 9), (4, 8)), mode="constant"
            ),
            "8": np.pad(
                np.ones((3, 3), dtype=np.uint8), ((8, 5), (9, 4)), mode="constant"
            ),
        },
    }
    session._global_track_last_box = {7: np.asarray([4.0, 3.0, 4.0, 4.0], dtype=float)}

    boxes, track_ids = session._build_boundary_reseed_boxes(
        frame_idx=12,
        frame_width=16.0,
        frame_height=16.0,
        max_boxes=2,
    )

    assert len(boxes) >= 1
    assert all(len(box) == 4 for box in boxes)
    assert all(0.0 <= v <= 1.0 for box in boxes for v in box)
    assert 7 in track_ids


def test_build_boundary_reseed_prompt_bundle_uses_last_frame_masks_only() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.max_num_objects = 4
    session.id_to_labels = {7: "vole", 8: "vole", 9: "vole"}
    session._frame_track_ids = {9: {9}, 10: {8}, 11: {7}}
    session._frame_masks = {
        9: {
            "9": np.pad(
                np.ones((2, 2), dtype=np.uint8), ((11, 3), (11, 3)), mode="constant"
            ),
        },
        10: {
            "8": np.pad(
                np.ones((3, 3), dtype=np.uint8), ((6, 7), (7, 6)), mode="constant"
            ),
        },
        11: {
            "7": np.pad(
                np.ones((4, 4), dtype=np.uint8), ((2, 10), (3, 9)), mode="constant"
            ),
        },
    }

    bundle = session._build_boundary_reseed_prompt_bundle(
        frame_idx=12,
        frame_width=16.0,
        frame_height=16.0,
        max_prompts=2,
    )

    assert len(bundle.boxes) == 1
    assert len(bundle.mask_inputs) == 1
    assert bundle.track_ids == [7]
    assert bundle.label_hints == ["vole"]
    assert all(len(box) == 4 for box in bundle.boxes)
    assert all(0.0 <= v <= 1.0 for box in bundle.boxes for v in box)
    assert int(np.asarray(bundle.mask_inputs[0]).sum()) == 16


def test_build_boundary_reseed_prompt_bundle_requires_last_frame_masks() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.max_num_objects = 4
    session.id_to_labels = {7: "vole", 8: "vole"}
    session._frame_track_ids = {10: {8}, 11: {7}}
    session._frame_masks = {
        10: {
            "8": np.pad(
                np.ones((3, 3), dtype=np.uint8), ((6, 7), (7, 6)), mode="constant"
            ),
        },
        11: {},
    }

    bundle = session._build_boundary_reseed_prompt_bundle(
        frame_idx=12,
        frame_width=16.0,
        frame_height=16.0,
        max_prompts=2,
    )

    assert bundle.boxes == []
    assert bundle.mask_inputs == []
    assert bundle.track_ids == []
    assert bundle.label_hints == []


def test_boundary_mask_matching_keeps_previous_ids_and_mints_new_ones() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session._session_id = "session-1"
    session._active_global_match_session_id = None
    session._session_local_to_global_ids = {}
    session._global_track_next_id = 20
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session._global_track_obj_ptr = {}
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session._prompt_seed_frames = {12}

    ref_mask = np.zeros((8, 8), dtype=np.uint8)
    ref_mask[2:5, 2:5] = 1
    ref_mask_b = np.zeros((8, 8), dtype=np.uint8)
    ref_mask_b[0:2, 0:2] = 1
    bundle = type(
        "Bundle",
        (),
        {
            "boxes": [[2 / 8, 2 / 8, 3 / 8, 3 / 8], [0.0, 0.0, 2 / 8, 2 / 8]],
            "box_labels": [1, 1],
            "mask_inputs": [ref_mask, ref_mask_b],
            "mask_labels": [1, 1],
            "track_ids": [7, 8],
            "label_hints": ["vole", "vole"],
        },
    )()
    new_mask = np.zeros((8, 8), dtype=np.uint8)
    new_mask[2:5, 2:5] = 1
    unmatched_mask = np.zeros((8, 8), dtype=np.uint8)
    unmatched_mask[5:7, 5:7] = 1
    outputs = {
        "out_obj_ids": np.asarray([1, 2], dtype=np.int64),
        "out_boxes_xywh": np.asarray(
            [[2.0, 2.0, 3.0, 3.0], [5.0, 5.0, 2.0, 2.0]], dtype=np.float32
        ),
        "out_binary_masks": np.asarray([new_mask, unmatched_mask], dtype=object),
    }

    mapped = session._map_outputs_to_global_ids_from_boundary_bundle_at_frame(
        outputs,
        frame_idx=12,
        boundary_bundle=bundle,
        allowed_gids={7, 8},
        allow_new_ids=True,
        max_new_ids=None,
        session_id="session-1",
    )

    assert mapped["out_obj_ids"].tolist() == [7, 20]
    assert mapped["global_id_assignments"][0]["global_id"] == 7
    assert mapped["global_id_assignments"][1]["global_id"] == 20


def test_boundary_mask_matching_respects_configurable_threshold() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session._session_id = "session-1"
    session._active_global_match_session_id = None
    session._session_local_to_global_ids = {}
    session._global_track_next_id = 30
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session._global_track_obj_ptr = {}
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.boundary_mask_match_iou_threshold = 0.75
    session._prompt_seed_frames = {12}

    ref_mask = np.zeros((8, 8), dtype=np.uint8)
    ref_mask[2:5, 2:5] = 1
    bundle = type(
        "Bundle",
        (),
        {
            "boxes": [[2 / 8, 2 / 8, 3 / 8, 3 / 8]],
            "box_labels": [1],
            "mask_inputs": [ref_mask],
            "mask_labels": [1],
            "track_ids": [7],
            "label_hints": ["vole"],
        },
    )()
    shifted_mask = np.zeros((8, 8), dtype=np.uint8)
    shifted_mask[3:6, 3:6] = 1
    outputs = {
        "out_obj_ids": np.asarray([1], dtype=np.int64),
        "out_boxes_xywh": np.asarray([[3.0, 3.0, 3.0, 3.0]], dtype=np.float32),
        "out_binary_masks": np.asarray([shifted_mask], dtype=object),
    }

    mapped = session._map_outputs_to_global_ids_from_boundary_bundle_at_frame(
        outputs,
        frame_idx=12,
        boundary_bundle=bundle,
        allowed_gids={7},
        allow_new_ids=True,
        max_new_ids=None,
        session_id="session-1",
    )

    assert mapped["out_obj_ids"].tolist() == [30]
    assert mapped["global_id_assignments"][0]["global_id"] == 30


def test_multiplex_mask_decoder_returns_empty_outputs_for_empty_batch() -> None:
    decoder = MultiplexMaskDecoder(
        transformer_dim=32,
        transformer=SimpleNamespace(),
        multiplex_count=4,
        num_multimask_outputs=3,
    )

    outputs = decoder.forward(
        image_embeddings=torch.zeros(0, 32, 8, 8),
        image_pe=torch.zeros(1, 32, 8, 8),
        multimask_output=False,
    )

    assert outputs["masks"].shape == (0, 4, 1, 32, 32)
    assert outputs["iou_pred"].shape == (0, 4, 1)
    assert outputs["sam_tokens_out"].shape == (0, 4, 1, 32)
    assert outputs["object_score_logits"].shape == (0, 4, 1)


def test_multiplex_mask_decoder_handles_internal_batch_mismatch() -> None:
    decoder = MultiplexMaskDecoder(
        transformer_dim=32,
        transformer=_MismatchedTransformer(),
        multiplex_count=4,
        num_multimask_outputs=3,
    )

    outputs = decoder.forward(
        image_embeddings=torch.zeros(1, 32, 8, 8),
        image_pe=torch.zeros(1, 32, 8, 8),
        multimask_output=False,
    )

    assert outputs["masks"].shape == (1, 4, 1, 32, 32)
    assert outputs["iou_pred"].shape == (1, 4, 1)
    assert outputs["sam_tokens_out"].shape == (1, 4, 1, 32)
    assert outputs["object_score_logits"].shape == (1, 4, 1)


def test_select_positive_detections_aligns_all_detection_outputs() -> None:
    boxes = torch.randn(1, 200, 4)
    masks = torch.randn(1, 200, 288, 288)
    scores = torch.randn(1, 200)
    pos_mask = torch.tensor([[False, True] + [False] * 198], dtype=torch.bool)

    selected_boxes, selected_masks, selected_scores, selected_keep = (
        _select_positive_detections(
            pred_boxes_xyxy=boxes,
            pred_masks=masks,
            pred_probs=scores,
            pos_pred_mask=pos_mask,
        )
    )

    assert selected_boxes.shape == (1, 4)
    assert selected_masks.shape == (1, 288, 288)
    assert selected_scores.shape == (1,)
    assert selected_keep.shape == (1,)
    assert selected_keep.dtype == torch.bool


def test_select_positive_detections_rejects_rank_drift() -> None:
    boxes = torch.randn(1, 200, 4)
    masks = torch.randn(1, 200, 2, 288, 288)
    scores = torch.randn(1, 200)
    pos_mask = torch.tensor([[False, True] + [False] * 198], dtype=torch.bool)

    with pytest.raises(ValueError):
        _select_positive_detections(
            pred_boxes_xyxy=boxes,
            pred_masks=masks,
            pred_probs=scores,
            pos_pred_mask=pos_mask,
        )


def test_ensure_object_masks_accepts_singleton_channel_layout() -> None:
    masks = torch.randn(1, 1, 288, 288)
    normalized = _ensure_object_masks(masks)

    assert normalized.shape == (1, 288, 288)


def test_safe_slice_first_dim_preserves_empty_but_valid_state() -> None:
    empty = torch.zeros((0, 4, 8), dtype=torch.float32)
    sliced = _safe_slice_first_dim(empty, [0], field_name="maskmem_features")

    assert sliced.shape == (0, 4, 8)
    assert sliced.numel() == 0


def test_get_maskmem_pos_enc_handles_empty_list() -> None:
    demo = VideoTrackingMultiplexDemo.__new__(VideoTrackingMultiplexDemo)

    assert demo._get_maskmem_pos_enc({"constants": {}}, {"maskmem_pos_enc": []}) is None
