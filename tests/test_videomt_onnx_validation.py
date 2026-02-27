from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from annolid.segmentation.videomt_onnx import (
    _causal_window_indices,
    _choose_seed_frame,
    _SeedMask,
    _select_onnx_providers,
    _infer_videomt_input_mode,
    _preflight_validate_onnx_file,
    VideoMTOnnxVideoProcessor,
)


def test_preflight_rejects_html_payload(tmp_path: Path) -> None:
    bad = tmp_path / "bad.onnx"
    bad.write_text("<!doctype html><html><body>not a model</body></html>")
    with pytest.raises(RuntimeError, match="looks like HTML/XML"):
        _preflight_validate_onnx_file(bad)


def test_choose_seed_frame_prefers_latest_at_or_before_requested() -> None:
    assert _choose_seed_frame(1, [0]) == 0
    assert _choose_seed_frame(10, [0, 3, 9]) == 9
    assert _choose_seed_frame(9, [0, 3, 9]) == 9
    assert _choose_seed_frame(1, [5, 8]) == 5
    assert _choose_seed_frame(1, []) is None


def test_infer_videomt_input_mode_handles_clip_and_image_shapes() -> None:
    assert _infer_videomt_input_mode([1, 3, 8, 224, 224]) == "video_5d"
    assert _infer_videomt_input_mode([1, 3, 224, 224]) == "image_4d"
    assert _infer_videomt_input_mode([5, 3, 224, 224]) == "clip_4d_nchw"
    assert _infer_videomt_input_mode([5, 540, 1600, 3]) == "clip_4d_nhwc"
    assert _infer_videomt_input_mode([1, 540, 1600, 3]) == "image_4d_nhwc"


def test_select_onnx_providers_avoids_coreml_for_dynamic_temporal_inputs() -> None:
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    providers = _select_onnx_providers(
        available, input_shape=[None, 3, 224, 224], input_mode="clip_4d_nchw"
    )
    assert providers == ["CPUExecutionProvider"]

    providers_static = _select_onnx_providers(
        available, input_shape=[5, 3, 224, 224], input_mode="clip_4d_nchw"
    )
    assert providers_static == ["CPUExecutionProvider"]


def test_causal_window_indices_returns_fixed_window_ending_at_target() -> None:
    indices, local_idx = _causal_window_indices(target_frame=1, window_size=5)
    assert indices == [-3, -2, -1, 0, 1]
    assert local_idx == 4


def test_infer_square_side_from_token_mismatch_prefers_expected_patch_grid() -> None:
    side = VideoMTOnnxVideoProcessor._infer_square_side_from_token_mismatch(
        actual_tokens=540, expected_tokens=1600, current_h=300, current_w=480
    )
    # 300x480 with ~16px patches gives 540 tokens; 1600 tokens implies ~40x40 grid.
    assert side in {640, 656}


def test_iou_resizes_pred_mask_to_seed_mask_shape() -> None:
    small = np.zeros((160, 160), dtype=bool)
    small[20:80, 20:80] = True
    large = np.zeros((640, 640), dtype=bool)
    large[80:320, 80:320] = True

    iou = VideoMTOnnxVideoProcessor._iou(small, large)
    assert iou > 0.99


def test_should_skip_saving_frame_uses_seed_frame_not_requested_start() -> None:
    # Reseed fallback case: requested start 1, seed actually from frame 0.
    assert VideoMTOnnxVideoProcessor._should_skip_saving_frame(1, 1, 0) is False
    assert VideoMTOnnxVideoProcessor._should_skip_saving_frame(0, 1, 0) is True


def test_build_query_label_map_recovers_unmatched_seed_labels() -> None:
    proc = VideoMTOnnxVideoProcessor.__new__(VideoMTOnnxVideoProcessor)
    proc.logit_threshold = 0.5
    proc.seed_iou_threshold = 0.01

    # [Q=2, T=1, H=4, W=4]
    binary_masks = np.zeros((2, 1, 4, 4), dtype=bool)
    binary_masks[0, 0, 0:2, 0:2] = True  # query 0
    binary_masks[1, 0, 2:4, 2:4] = True  # query 1
    scores = np.array([0.9, -0.5], dtype=np.float32)  # query 1 below threshold

    seeds = [
        _SeedMask(label="a", mask=binary_masks[0, 0].copy()),
        _SeedMask(label="b", mask=binary_masks[1, 0].copy()),
    ]

    out = proc._build_query_label_map(binary_masks, scores, seeds, seed_timestep=0)
    assert set(out.values()) == {"a", "b"}


def test_resolve_model_path_downloads_videomt_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    proc = VideoMTOnnxVideoProcessor.__new__(VideoMTOnnxVideoProcessor)
    cached = tmp_path / "downloads" / "videomt_yt_2019_vit_small_52.8.onnx"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"onnx")

    monkeypatch.setattr(
        "annolid.segmentation.videomt_onnx.resolve_existing_model_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "annolid.segmentation.videomt_onnx.ensure_cached_model_asset",
        lambda **_kwargs: cached,
    )

    path = proc._resolve_model_path({})
    assert path == cached
