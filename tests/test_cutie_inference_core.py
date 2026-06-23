import torch

from annolid.segmentation.cutie_vos.inference import inference_core
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore


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
