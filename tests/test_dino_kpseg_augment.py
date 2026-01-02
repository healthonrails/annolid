import numpy as np
from PIL import Image

from annolid.segmentation.dino_kpseg.data import (
    DinoKPSEGAugmentConfig,
    _apply_pose_augmentations,
    _invert_affine_3x3_to_pil_coeffs,
)


def test_dino_kpseg_hflip_respects_flip_idx():
    pil = Image.new("RGB", (100, 100), color=(10, 20, 30))
    keypoints_instances = [
        np.asarray(
            [
                [0.2, 0.5, 2.0],  # left
                [0.8, 0.5, 2.0],  # right
                [0.5, 0.1, 2.0],  # center
            ],
            dtype=np.float32,
        )
    ]
    cfg = DinoKPSEGAugmentConfig(enabled=True, hflip_prob=1.0)
    rng = np.random.default_rng(0)

    out_img, out_kpts = _apply_pose_augmentations(
        pil,
        keypoints_instances,
        cfg=cfg,
        flip_idx=[1, 0, 2],
        rng=rng,
    )

    assert out_img.size == pil.size
    assert len(out_kpts) == 1
    out = out_kpts[0]
    # After flip + flip_idx mapping: the first keypoint should correspond to the original "right" point.
    assert np.isclose(out[0, 0], 0.2, atol=1e-6)
    assert np.isclose(out[1, 0], 0.8, atol=1e-6)
    assert np.isclose(out[2, 0], 0.5, atol=1e-6)


def test_dino_kpseg_affine_inverse_coeffs_roundtrip():
    m = np.array(
        [
            [1.0, 0.2, 3.0],
            [-0.1, 0.9, -2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    coeffs = _invert_affine_3x3_to_pil_coeffs(m)
    inv = np.array(
        [
            [coeffs[0], coeffs[1], coeffs[2]],
            [coeffs[3], coeffs[4], coeffs[5]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ident = inv @ m.astype(np.float64)
    assert np.allclose(ident, np.eye(3), atol=1e-6)
