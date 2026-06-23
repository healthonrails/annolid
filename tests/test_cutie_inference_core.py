import torch

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
