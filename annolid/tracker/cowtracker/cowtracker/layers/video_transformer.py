# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops

    _XFORMERS_AVAILABLE = True
except ImportError:
    xops = None
    _XFORMERS_AVAILABLE = False

from timm.models.vision_transformer import Attention as TimmAttention
from cowtracker.layers.temporal_attention import TemporalSelfAttentionBlock
from cowtracker.layers.patch_embed import PatchEmbed
from cowtracker.layers.dpt_head import DPTHead

print("timm version: ", timm.__version__)


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def _get_flash_attention_ops():
    """Automatically detect GPU and return appropriate flash attention ops."""
    if not _XFORMERS_AVAILABLE or not torch.cuda.is_available():
        return None

    # Get compute capability of current device
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        try:
            return (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)
        except AttributeError:
            return (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
    else:
        return (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)


class FlashAttention3(nn.Module):
    """
    Drop-in replacement for timm.models.vision_transformer.Attention using xformers Flash Attention 3.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Get Flash Attention ops
        self.flash_ops = _get_flash_attention_ops()

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each is (B, N, num_heads, head_dim)

        if _XFORMERS_AVAILABLE and self.flash_ops is not None:
            # xformers expects [B, M, H, K] format - we already have it!
            # Use xformers memory_efficient_attention
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_mask,
                p=self.attn_drop if self.training else 0.0,
                scale=self.scale,
                op=self.flash_ops,
            )
        else:
            # Fallback to standard PyTorch implementation
            q = q.transpose(1, 2)  # [B, H, N, K]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            x = x.transpose(1, 2)  # [B, N, H, K]

        # Reshape back to [B, N, C]
        x = x.reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def replace_attention_with_flash3(model: nn.Module) -> nn.Module:
    """
    Recursively replace all timm.Attention modules with FlashAttention3.
    """
    for name, module in model.named_children():
        if isinstance(module, TimmAttention):
            # Extract parameters from timm Attention
            flash_attn = FlashAttention3(
                dim=module.qkv.in_features,
                num_heads=module.num_heads,
                qkv_bias=module.qkv.bias is not None,
                attn_drop=module.attn_drop.p if hasattr(module, "attn_drop") else 0.0,
                proj_drop=module.proj_drop.p if hasattr(module, "proj_drop") else 0.0,
            )
            # Copy weights from original attention
            flash_attn.qkv.weight.data = module.qkv.weight.data.clone()
            if module.qkv.bias is not None:
                flash_attn.qkv.bias.data = module.qkv.bias.data.clone()
            flash_attn.proj.weight.data = module.proj.weight.data.clone()
            if module.proj.bias is not None:
                flash_attn.proj.bias.data = module.proj.bias.data.clone()

            # Replace the module
            setattr(model, name, flash_attn)
            # print(
            #     f"  Replaced attention module '{name}' (dim={module.qkv.in_features}, heads={module.num_heads})"
            # )
        else:
            # Recursively apply to child modules
            replace_attention_with_flash3(module)

    return model


MODEL_CONFIGS = {
    "vitl": {
        "encoder": "vit_large_patch16_224",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitb": {
        "encoder": "vit_base_patch16_224",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vits": {
        "encoder": "vit_small_patch16_224",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitt": {
        "encoder": "vit_tiny_patch16_224",
        "features": 32,
        "out_channels": [24, 48, 96, 192],
    },
}


class VisionTransformerVideo(nn.Module):
    """
    Input: (B, T, C, H, W)
    Pipeline: per-frame ViT + interleaved Temporal Attention (across frames)
    Time pos: 1D sinusoidal encoding + linear interpolation for variable T
    """

    def __init__(
        self,
        model_name,
        input_dim,
        patch_size=16,
        temporal_interleave_stride=2,
        max_frames=256,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        drop_path=0.0,
        shared_temporal_block=False,
        num_blocks=None,
        use_flash_attention3=True,
    ):
        super().__init__()
        model = timm.create_model(
            MODEL_CONFIGS[model_name]["encoder"],
            pretrained=False,
            num_classes=0,
        )
        self.intermediate_layer_idx = {
            "vitt": [2, 5, 8, 11],
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.idx = self.intermediate_layer_idx[model_name]
        self.blks = model.blocks if num_blocks is None else model.blocks[:num_blocks]

        # Replace attention with Flash Attention 3 if enabled
        if use_flash_attention3:
            self.blks = replace_attention_with_flash3(self.blks)
            num_fa3_modules = sum(
                1 for m in self.blks.modules() if isinstance(m, FlashAttention3)
            )
            print(
                f"✓ Flash Attention 3 enabled for spatial attention: replaced {num_fa3_modules} attention modules"
            )

        self.embed_dim = model.embed_dim
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = MODEL_CONFIGS[model_name]["features"]

        # Spatial positional embedding (64 corresponds to 224/16 = 14x14)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ====== New: sinusoidal time positional embedding (buffer) ======
        self.max_frames = max_frames
        time_grid = torch.arange(max_frames, dtype=torch.float32)  # (T0,)
        time_emb = get_1d_sincos_pos_embed_from_grid(
            self.embed_dim, time_grid
        )  # (1, T0, D)
        # Follow your pattern: register_buffer + interpolate_time_embed
        self.register_buffer("time_emb", time_emb, persistent=False)

        # Patch embed and DPT head
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=input_dim,
            embed_dim=self.embed_dim,
        )
        self.dpt_head = DPTHead(
            self.embed_dim,
            MODEL_CONFIGS[model_name]["features"],
            out_channels=MODEL_CONFIGS[model_name]["out_channels"],
        )

        # Temporal block(s)
        num_heads = getattr(model.blocks[0].attn, "num_heads", 8)
        self.shared_temporal_block = shared_temporal_block

        # Insert a temporal block after every N spatial blocks
        self.temporal_interleave_stride = max(1, int(temporal_interleave_stride))

        # Calculate how many temporal blocks we need
        num_temporal_blocks = sum(
            1
            for i in range(len(self.blks))
            if (i + 1) % self.temporal_interleave_stride == 0
        )

        if shared_temporal_block:
            # Single shared temporal block for all layers
            self.temporal_block = TemporalSelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_dropout,
                drop=proj_dropout,
                drop_path=drop_path,
            )
            self.temporal_blocks = None
        else:
            # Separate temporal block for each layer
            self.temporal_block = None
            self.temporal_blocks = nn.ModuleList(
                [
                    TemporalSelfAttentionBlock(
                        dim=self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_dropout,
                        drop=proj_dropout,
                        drop_path=drop_path,
                    )
                    for _ in range(num_temporal_blocks)
                ]
            )

    # ====== New: interpolate temporal positional embedding ======
    def interpolate_time_embed(self, x_like: torch.Tensor, t: int) -> torch.Tensor:
        """
        x_like: used only to fetch dtype (e.g., fp16)
        Return: time positional embedding of shape (1, t, D)
        """
        previous_dtype = x_like.dtype
        T0 = self.time_emb.shape[1]
        if t == T0:
            return self.time_emb.to(previous_dtype)
        temb = self.time_emb.float()  # (1, T0, D)
        temb = F.interpolate(
            temb.permute(0, 2, 1), size=t, mode="linear", align_corners=False
        ).permute(0, 2, 1)  # (1, t, D)
        return temb.to(previous_dtype)

    def interpolate_pos_encoding(self, x, h, w):
        """
        Interpolate the 2D spatial positional encoding to match HxW (in patches).
        """
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False,
        )
        assert int(w0) == pos_embed.shape[-1] and int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        # Merge time into batch for per-frame spatial encoding
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, Np, D)

        x = x.view(B, T, *x.shape[1:])
        # Get time positional embedding for current T via linear interpolation: (1, T, D)
        tpos = self.interpolate_time_embed(x, T).unsqueeze(2)  # (1, T, 1, D)
        x = x + tpos  # (B, T, Np, D)
        x = x.view(B * T, *x.shape[2:])  # (B*T, Np, D)

        x = x + self.interpolate_pos_encoding(x, H, W)

        outputs = []
        temporal_block_idx = 0
        for i in range(len(self.blks)):
            # 1) Spatial self-attention (per frame)
            x = self.blks[i](x)  # (B*T, Np, D)
            # 2) Interleave temporal self-attention (across frames, same spatial patch)
            if (i + 1) % self.temporal_interleave_stride == 0:
                x = x.view(B, T, *x.shape[1:])
                if self.shared_temporal_block:
                    x = self.temporal_block(x)
                else:
                    x = self.temporal_blocks[temporal_block_idx](x)
                    temporal_block_idx += 1
                x = x.view(B * T, *x.shape[2:])
            # 3) Collect intermediate features for DPT head
            if i in self.idx:
                outputs.append([x])

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        # DPT head consumes (B*T, Np, D); here batch is B*T
        out, path_1, path_2, path_3, path_4 = self.dpt_head.forward(
            outputs, patch_h, patch_w, return_intermediate=True
        )
        # Upsample per frame
        out = F.interpolate(
            out, (H, W), mode="bilinear", align_corners=True
        )  # (B*T, Cout, H, W)

        # Restore (B, T, ...)
        def bt_to_btensor(tensor_or_none):
            if tensor_or_none is None:
                return None
            return tensor_or_none.view(B, T, *tensor_or_none.shape[1:])

        return {
            "out": out.view(B, T, *out.shape[1:]),
            "path_1": bt_to_btensor(path_1),
            "path_2": bt_to_btensor(path_2),
            "path_3": bt_to_btensor(path_3),
            "path_4": bt_to_btensor(path_4),
        }
