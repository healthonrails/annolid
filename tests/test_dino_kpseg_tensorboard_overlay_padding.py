import torch


def test_stack_chw_images_with_padding_preserves_content() -> None:
    from annolid.segmentation.dino_kpseg.train import _stack_chw_images_with_padding

    img0 = torch.zeros((3, 8, 12), dtype=torch.float32)
    img1 = torch.ones((3, 8, 9), dtype=torch.float32)

    out = _stack_chw_images_with_padding([img0, img1], pad_value=0.25)
    assert tuple(out.shape) == (2, 3, 8, 12)

    # Original regions preserved.
    assert torch.allclose(out[0, :, :, :12], img0)
    assert torch.allclose(out[1, :, :, :9], img1)

    # Padded region uses the pad value.
    assert torch.allclose(out[1, :, :, 9:], torch.full((3, 8, 3), 0.25))

