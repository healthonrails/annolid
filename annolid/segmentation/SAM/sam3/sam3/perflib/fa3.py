# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
import torch.nn.functional as F


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


def flash_attn_func(q, k, v):
    def _sdpa_fallback(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
        # Accept either [B, S, H, D] or [B, H, S, D] and return same layout as input.
        if q_in.ndim != 4:
            raise RuntimeError(f"Unsupported attention tensor rank: {q_in.ndim}")
        seq_first = q_in.shape[1] >= q_in.shape[2]
        if seq_first:
            q_sdpa = q_in.permute(0, 2, 1, 3)
            k_sdpa = k_in.permute(0, 2, 1, 3)
            v_sdpa = v_in.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            return out.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_in, k_in, v_in)
        return out

    # FA3 path only on CUDA; otherwise fallback.
    if q.device.type != "cuda":
        return _sdpa_fallback(q, k, v)

    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        return _sdpa_fallback(q, k, v)

    try:
        return flash_attn_func_op(q.to(dtype), k.to(dtype), v.to(dtype)).to(q.dtype)
    except Exception:
        # Runtime/backend incompatibilities (missing flash-attn, unsupported FP8, etc.)
        return _sdpa_fallback(q, k, v).to(q.dtype)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    # output needs to be bfloat16, not float8!
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
