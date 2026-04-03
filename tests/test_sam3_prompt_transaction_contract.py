from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from annolid.gui.widgets.sam3_manager import Sam3Manager
from annolid.segmentation.SAM.sam3.session import Sam3SessionManager
from annolid.segmentation.SAM.sam3.sam3.model.multiplex_mask_decoder import (
    MultiplexMaskDecoder,
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
    assert prompts["boxes_abs"] == [[10, 20, 20, 30]]
    assert prompts["polygons_abs"] == [[[1, 1], [4, 1], [4, 4], [1, 4]]]
    assert prompts["point_labels"] == [1, 0]

    annotations = manager._canvas_prompts_to_annotations(
        frame_idx=7,
        id_to_labels={5: "mouse", 8: "mouse"},
    )
    assert len(annotations) == 3
    assert any(a["type"] == "polygon" and a["obj_id"] == 8 for a in annotations)
    assert any(a["type"] == "box" and a["obj_id"] == 5 for a in annotations)
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
            ]
        ),
    )

    annotations = manager._canvas_prompts_to_annotations(
        frame_idx=3,
        id_to_labels={},
    )
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["type"] == "mask"
    assert ann["obj_id"] == 2
    assert ann["mask"].shape == (24, 32)
    assert int(np.asarray(ann["mask"]).sum()) > 0


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
        return 0, False

    session._execute_prompt_transaction = _execute_prompt_transaction
    session._handle_frame_outputs = _handle_frame_outputs

    session.add_prompt(
        frame_idx=0,
        session_id="window-session-1",
        text="mouse",
        record_outputs=True,
        merge_existing_on_record=False,
        label_hints=["mouse"],
    )

    assert captured["session_id"] == "window-session-1"
    assert captured["recorded"] is True


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
    assert outputs["mask_tokens_out"].shape == (0, 4, 1, 32)
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

    assert outputs["masks"].shape[0] == 0
    assert outputs["iou_pred"].shape == (0, 4, 1)
    assert outputs["sam_tokens_out"].shape[0] == 0
    assert outputs["object_score_logits"].shape == (0, 4, 1)
