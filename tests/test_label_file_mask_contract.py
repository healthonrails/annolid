from __future__ import annotations

import json

import numpy as np

from annolid.annotation.keypoints import format_shape
from annolid.gui.label_file import LabelFile
from annolid.gui.shape import Shape
from annolid.utils.annotation_compat import utils as labelme_utils


def test_label_file_ignores_mask_payload_for_polygon_shape(tmp_path) -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "mouse",
                "points": [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
                "shape_type": "polygon",
                "flags": {},
                "mask": labelme_utils.img_arr_to_b64(img),
            }
        ],
        "imagePath": "frame.jpg",
        "imageData": None,
        "imageHeight": 4,
        "imageWidth": 4,
    }
    path = tmp_path / "frame_000000000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    lf = LabelFile(str(path), is_video_frame=True)

    assert len(lf.shapes) == 1
    assert lf.shapes[0]["shape_type"] == "polygon"
    assert lf.shapes[0]["mask"] is None


def test_format_shape_serializes_mask_only_for_mask_shape() -> None:
    polygon = Shape(label="mouse", shape_type="polygon", flags={})
    polygon.points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    polygon.mask = np.ones((2, 2), dtype=np.uint8)
    poly_data = format_shape(polygon)

    mask_shape = Shape(label="mouse", shape_type="mask", flags={})
    mask_shape.points = []
    mask_shape.mask = np.ones((2, 2), dtype=np.uint8)
    mask_data = format_shape(mask_shape)

    assert "mask" not in poly_data
    assert "mask" in mask_data
