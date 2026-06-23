import numpy as np
import torch

from annolid.segmentation.cutie_vos.inference import inference_core
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    resize_frame_for_inference,
)


def test_resize_index_mask_uses_nearest_without_align_corners() -> None:
    mask = torch.tensor(
        [
            [1, 2],
            [3, 0],
        ],
        dtype=torch.int64,
    )

    resized = InferenceCore._resize_index_mask(mask, (4, 4))

    assert resized.dtype == torch.int64
    assert set(torch.unique(resized).tolist()) == {0, 1, 2, 3}
    assert resized[0, 0].item() == 1
    assert resized[0, -1].item() == 2
    assert resized[-1, 0].item() == 3
    assert resized[-1, -1].item() == 0


def test_resize_input_index_mask_preserves_object_ids() -> None:
    mask = torch.tensor(
        [
            [0, 5],
            [9, 0],
        ],
        dtype=torch.int64,
    )

    resized = InferenceCore._resize_input_mask(
        mask,
        (8, 8),
        idx_mask=True,
    )

    assert resized.shape == (8, 8)
    assert resized.dtype == torch.int64
    assert set(torch.unique(resized).tolist()) == {0, 5, 9}


def test_finalize_index_output_resizes_only_one_label_channel(monkeypatch) -> None:
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.1, 0.8], [0.2, 0.1]],
            [[0.0, 0.1], [0.7, 0.8]],
        ],
        dtype=torch.float32,
    )
    interpolate_shapes = []
    original_interpolate = inference_core.F.interpolate

    def _capture_interpolate(input_tensor, *args, **kwargs):
        interpolate_shapes.append(tuple(input_tensor.shape))
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(inference_core.F, "interpolate", _capture_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(8, 8),
        return_index_mask=True,
    )

    assert output.shape == (8, 8)
    assert output.dtype == torch.uint8
    assert set(torch.unique(output).tolist()) == {0, 1, 2}
    assert interpolate_shapes == [(1, 1, 2, 2)]


def test_finalize_probability_output_preserves_default_behavior(monkeypatch) -> None:
    probabilities = torch.rand((4, 2, 3), dtype=torch.float32)
    interpolate_shapes = []
    original_interpolate = inference_core.F.interpolate

    def _capture_interpolate(input_tensor, *args, **kwargs):
        interpolate_shapes.append(tuple(input_tensor.shape))
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(inference_core.F, "interpolate", _capture_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(8, 12),
    )

    assert output.shape == (4, 8, 12)
    assert output.dtype == probabilities.dtype
    assert interpolate_shapes == [(1, 4, 2, 3)]


def test_resize_frame_for_inference_caps_short_side_before_tensor_conversion() -> None:
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    resized = resize_frame_for_inference(frame, 480)

    assert resized.shape == (480, 853, 3)
    assert resized.dtype == np.uint8


def test_resize_frame_for_inference_reuses_small_frame() -> None:
    frame = np.zeros((230, 280, 3), dtype=np.uint8)

    resized = resize_frame_for_inference(frame, 480)

    assert resized is frame


def test_image_to_torch_returns_contiguous_float_chw_tensor() -> None:
    frame = np.full((3, 5, 3), 255, dtype=np.uint8)

    tensor = image_to_torch(frame, device="cpu")

    assert tensor.shape == (3, 3, 5)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.all(tensor == 1.0)


def test_finalize_index_output_skips_same_size_interpolation(monkeypatch) -> None:
    probabilities = torch.rand((3, 4, 5), dtype=torch.float32)

    def _unexpected_interpolate(*_args, **_kwargs):
        raise AssertionError("same-size output should not be interpolated")

    monkeypatch.setattr(inference_core.F, "interpolate", _unexpected_interpolate)

    output = InferenceCore._finalize_output(
        probabilities,
        output_size=(4, 5),
        return_index_mask=True,
    )

    assert output.shape == (4, 5)
    assert output.dtype == torch.uint8
