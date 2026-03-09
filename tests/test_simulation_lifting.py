from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from annolid.simulation import (
    lift_pose_frames_with_depth,
    load_depth_records,
    read_pose_frames,
)


def _depth_png_base64(array: np.ndarray) -> str:
    buf = io.BytesIO()
    imageio.imwrite(buf, array.astype(np.uint16), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_lift_pose_frames_with_depth_uses_camera_intrinsics(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.ndjson"
    depth_path = tmp_path / "depth.ndjson"
    pose_path.write_text(
        json.dumps(
            {
                "version": "Annolid",
                "video_name": "demo.mp4",
                "frame_index": 0,
                "imagePath": "frame_0000.png",
                "imageHeight": 2,
                "imageWidth": 2,
                "flags": {},
                "otherData": {},
                "shapes": [
                    {"label": "nose", "shape_type": "point", "points": [[1, 1]]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    depth_quantized = np.array([[0, 0], [0, 65535]], dtype=np.uint16)
    depth_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "video_name": "demo.mp4",
                "imageHeight": 2,
                "imageWidth": 2,
                "otherData": {
                    "depth_map": {
                        "image_data": _depth_png_base64(depth_quantized),
                        "scale": {"min": 0.0, "max": 2.0},
                        "dtype": "uint16",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pose_frames = read_pose_frames(pose_path)
    depth_records = load_depth_records(depth_path)
    lifted = lift_pose_frames_with_depth(
        pose_frames,
        depth_records=depth_records,
        coordinate_system={
            "camera_intrinsics": {"fx": 2.0, "fy": 4.0, "cx": 0.0, "cy": 0.0}
        },
    )

    point = lifted[0].points["nose"]
    assert point == (1.0, 0.5, 2.0)


def test_lift_pose_frames_with_depth_falls_back_to_pixel_space_without_intrinsics(
    tmp_path: Path,
) -> None:
    pose_path = tmp_path / "pose.json"
    depth_path = tmp_path / "depth.ndjson"
    pose_path.write_text(
        json.dumps(
            {
                "version": "5.0",
                "imagePath": "frame.png",
                "imageHeight": 3,
                "imageWidth": 3,
                "shapes": [
                    {"label": "tail", "shape_type": "point", "points": [[2, 1]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    depth_quantized = np.zeros((3, 3), dtype=np.uint16)
    depth_quantized[1, 2] = 32768
    depth_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "otherData": {
                    "depth_map": {
                        "image_data": _depth_png_base64(depth_quantized),
                        "scale": {"min": 10.0, "max": 20.0},
                        "dtype": "uint16",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pose_frames = read_pose_frames(pose_path, video_name="demo.mp4")
    depth_records = load_depth_records(depth_path)
    lifted = lift_pose_frames_with_depth(
        pose_frames,
        depth_records=depth_records,
        coordinate_system={},
    )

    x, y, z = lifted[0].points["tail"]
    assert x == 2.0
    assert y == 1.0
    assert 14.9 < z < 15.1
