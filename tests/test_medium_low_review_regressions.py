"""Focused regression tests for Medium and Low review items (C-9, D-3, D-4, G-2, S-3, P-3)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import pycocotools.mask as mask_util
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from qtpy import QtCore, QtWidgets

from annolid.annotation.coco2labelme import _keypoint_shapes, _segmentation_polygons
from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.annotation.labelme2yolo import find_bbox_from_shape
from annolid.core.agent.tools.analytics_reporting import AnnolidBehaviorCohortTool
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.engine.run_config import select_model_mode_config
from annolid.gui.label_file import LabelFile
from annolid.gui.shape import Shape
from annolid.gui.widgets.canvas import Canvas
from annolid.main import parse_args as parse_legacy_main_args
from annolid.segmentation.videomt_onnx import VideoMTOnnxVideoProcessor
from annolid.utils.annotation_store import load_labelme_json

_APP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _APP = app
    return _APP


def test_c9_videomt_preserves_multi_instance_group_id_across_seed_and_save(
    tmp_path: Path,
) -> None:
    seed_json = tmp_path / "vid_000000000.json"
    seed_json.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "shapes": [
                    {
                        "label": "mouse",
                        "group_id": 1,
                        "shape_type": "polygon",
                        "points": [[2, 2], [10, 2], [10, 10], [2, 10]],
                    },
                    {
                        "label": "mouse",
                        "group_id": 2,
                        "shape_type": "polygon",
                        "points": [[20, 20], [28, 20], [28, 28], [20, 28]],
                    },
                ],
                "imageHeight": 32,
                "imageWidth": 32,
            }
        ),
        encoding="utf-8",
    )

    proc = VideoMTOnnxVideoProcessor.__new__(VideoMTOnnxVideoProcessor)
    proc.video_width = 32
    proc.video_height = 32
    proc.input_width = 32
    proc.input_height = 32
    proc.logit_threshold = 0.1
    proc.seed_iou_threshold = 0.01
    proc.description = "VideoMT"
    proc.results_folder = tmp_path

    seeds = proc._load_seed_masks(seed_json)
    assert len(seeds) == 2
    assert {(s.label, s.group_id) for s in seeds} == {("mouse", 1), ("mouse", 2)}

    binary_masks = np.zeros((2, 1, 32, 32), dtype=bool)
    binary_masks[0, 0] = seeds[0].mask
    binary_masks[1, 0] = seeds[1].mask
    scores = np.array([0.9, 0.85], dtype=np.float32)

    seed_map = proc._build_query_seed_map(binary_masks, scores, seeds, seed_timestep=0)
    assert set(seed_map.values()) == {("mouse", 1), ("mouse", 2)}

    proc._frame_json_path = lambda idx: tmp_path / f"vid_{int(idx):09d}.json"  # type: ignore[method-assign]
    saved_ok = proc._save_frame_masks(
        1,
        {
            ("mouse", 1): seeds[0].mask,
            ("mouse", 2): seeds[1].mask,
        },
    )
    assert saved_ok is True
    frame1 = load_labelme_json(tmp_path / "vid_000000001.json")
    saved_pairs = {(sh["label"], sh.get("group_id")) for sh in frame1["shapes"]}
    assert saved_pairs == {("mouse", 1), ("mouse", 2)}


def test_d3_coco2labelme_visibility_rle_and_labelme2yolo_circle_bbox() -> None:
    # 1. Occluded (v=1) vs visible (v=2) keypoints survive round-trip visibility lookup
    kp_shapes = _keypoint_shapes(
        [10.0, 20.0, 1, 30.0, 40.0, 2, 0.0, 0.0, 0],
        keypoint_names=["nose", "tail_base", "ear"],
        group_id=3,
    )
    assert len(kp_shapes) == 2
    assert visibility_from_labelme_shape(kp_shapes[0]) == int(
        KeypointVisibility.OCCLUDED
    )
    assert visibility_from_labelme_shape(kp_shapes[1]) == int(
        KeypointVisibility.VISIBLE
    )

    # 2. Dict RLE segmentation decodes to polygon coordinates
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:16, 6:18] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    polys = _segmentation_polygons(rle)
    assert len(polys) >= 1
    assert len(polys[0]) >= 3

    # 3. LabelMe circle shape bbox returns true center and 2*r width/height
    cx, cy, w, h = find_bbox_from_shape(
        {
            "label": "arena_dish",
            "shape_type": "circle",
            "points": [[100.0, 100.0], [130.0, 100.0]],
        }
    )
    assert (cx, cy, w, h) == pytest.approx((100.0, 100.0, 60.0, 60.0))


def test_d4_g2_label_file_load_dimensions_and_canvas_delete_hover_clear(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    # 1. LabelFile preserves relative imagePath and records checked (imageHeight, imageWidth)
    buf = io.BytesIO()
    PIL.Image.fromarray(np.zeros((18, 26, 3), dtype=np.uint8)).save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    jf = tmp_path / "sample.json"
    jf.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": "sample.png",
                "imageData": b64_img,
                "imageHeight": 999,
                "imageWidth": 999,
            }
        ),
        encoding="utf-8",
    )
    lf_img = LabelFile(str(jf), is_video_frame=None)
    assert lf_img.imagePath == "sample.png"
    assert (lf_img.imageHeight, lf_img.imageWidth) == (18, 26)

    lf_vid = LabelFile(str(jf), is_video_frame=True)
    assert lf_vid.imagePath == "sample.png"

    # 2. Canvas.deleteShape clears dangling hShape/prevhShape pointers
    canvas = Canvas()
    try:
        shp = Shape(label="mouse_1", shape_type="polygon")
        shp.addPoint(QtCore.QPointF(5.0, 5.0))
        shp.addPoint(QtCore.QPointF(15.0, 5.0))
        shp.addPoint(QtCore.QPointF(15.0, 15.0))
        shp.close()
        canvas.loadShapes([shp])
        canvas.hShape = shp
        canvas.hVertex = 0
        canvas.prevhShape = shp
        canvas.deleteShape(shp)
        assert canvas.hShape is None
        assert canvas.hVertex is None
        assert canvas.prevhShape is None
    finally:
        canvas.deleteLater()


def test_s3_cohort_manifest_sandbox_and_function_tool_schema_bounds(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "allowed_ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_ev = outside / "secret_events.csv"
    outside_ev.write_text(
        "Recording time,Subject,Behavior,Event\n1.0,m1,groom,start\n2.0,m1,groom,stop\n",
        encoding="utf-8",
    )
    manifest = ws / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {"recording_id": "r1", "events_csv": str(outside_ev), "group": "g1"}
                ]
            }
        ),
        encoding="utf-8",
    )

    cohort_tool = AnnolidBehaviorCohortTool(allowed_dir=ws)
    with pytest.raises(PermissionError):
        asyncio.run(cohort_tool.execute(manifest_path=str(manifest)))

    class _DummySchemaTool(FunctionTool):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def description(self) -> str:
            return "dummy"

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "fps": {"type": "number", "exclusiveMinimum": 0},
                    "files": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                    },
                },
            }

        async def execute(self, **kwargs: Any) -> str:
            return json.dumps(kwargs)

    dt = _DummySchemaTool()
    assert dt.validate_params({"count": True}) == ["count should be integer"]
    assert dt.validate_params({"fps": 0}) == ["fps must be > 0"]
    assert dt.validate_params({"files": ["only_one.csv"]}) == [
        "files must contain at least 2 items"
    ]
    assert (
        dt.validate_params({"count": 3, "fps": 25.0, "files": ["a.csv", "b.csv"]}) == []
    )


def test_p3_legacy_main_bool_flags_and_predict_run_config_selection() -> None:
    parsed = parse_legacy_main_args(
        ["--multiple_animals", "false", "--show_flow", "--vis", "0"]
    )
    assert parsed["multiple_animals"] is False
    assert parsed["show_flow"] is True
    assert parsed["vis"] is False

    # select_model_mode_config must not leak models.<name>.train keys into predict mode
    cfg_payload = {
        "models": {
            "yolo": {
                "train": {"epochs": 100, "batch": 16},
            }
        },
        "predict": {"conf": 0.35},
    }
    selected_pred = select_model_mode_config(
        cfg_payload, model_name="yolo", mode="predict"
    )
    assert selected_pred == {"conf": 0.35}
    selected_train = select_model_mode_config(
        cfg_payload, model_name="yolo", mode="train"
    )
    assert selected_train == {"epochs": 100, "batch": 16}
