from __future__ import annotations

import pytest
import torch

from annolid.segmentation.SAM.sam3.sam3.sam.rope import apply_rotary_enc_real


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apply_rotary_enc_real_moves_freqs_to_query_device() -> None:
    device = torch.device("cuda:0")
    # [B, heads, tokens, dim]
    xq = torch.randn(1, 2, 4, 8, device=device)
    xk = torch.randn(1, 2, 4, 8, device=device)
    # Intentionally keep frequencies on CPU to emulate stale cache views.
    freqs_cis_real = torch.randn(4, 4, device="cpu")
    freqs_cis_imag = torch.randn(4, 4, device="cpu")

    out_q, out_k = apply_rotary_enc_real(
        xq,
        xk,
        freqs_cis_real=freqs_cis_real,
        freqs_cis_imag=freqs_cis_imag,
        repeat_freqs_k=False,
    )

    assert out_q.device.type == "cuda"
    assert out_k.device.type == "cuda"
    assert out_q.shape == xq.shape
    assert out_k.shape == xk.shape
