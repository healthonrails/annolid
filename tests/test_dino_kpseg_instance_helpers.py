from __future__ import annotations

import numpy as np
from PIL import Image

from annolid.segmentation.dino_kpseg.data import (
    _crop_instance,
    _parse_yolo_pose_instance,
)


def test_parse_yolo_pose_instance():
    tokens = "0 0.5 0.6 0.4 0.3 0.1 0.2 2 0.9 0.8 2".split()
    bbox, kpts = _parse_yolo_pose_instance(tokens, kpt_count=2, dims=3)
    assert np.allclose(bbox, np.array([0.5, 0.6, 0.4, 0.3], dtype=np.float32))
    assert kpts.shape == (2, 3)
    assert np.allclose(kpts[0], np.array([0.1, 0.2, 2.0], dtype=np.float32))


def test_crop_instance_updates_keypoints_visibility():
    pil = Image.new("RGB", (100, 100), color=(10, 20, 30))
    bbox = np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32)
    keypoints = np.array(
        [
            [0.5, 0.5, 2.0],  # center
            [0.9, 0.9, 2.0],  # outside crop
        ],
        dtype=np.float32,
    )
    crop, out = _crop_instance(pil, bbox=bbox, keypoints=keypoints, bbox_scale=1.0)
    assert crop.size[0] > 0 and crop.size[1] > 0
    assert np.isclose(out[0, 0], 0.5, atol=1e-6)
    assert np.isclose(out[0, 1], 0.5, atol=1e-6)
    assert np.isclose(out[0, 2], 2.0, atol=1e-6)
    assert np.isclose(out[1, 2], 0.0, atol=1e-6)
