"""Direct structural weight compression and Conv+BN fusion for CUTIE VOS.

Provides:
1. Algebraic folding of frozen ``Conv2d + BatchNorm2d`` pairs into ``(Conv2d, Identity)``,
   eliminating 48 memory-bound BatchNorm passes per frame with zero approximation error.
2. Higher-Order SVD (HOSVD / Tucker-2) spatial-channel low-rank factorization of heavy
   per-object ``3x3`` convolutions (``CAResBlock``, ``GroupResBlock``, ``SensoryUpdater``,
   ``SensoryDeepUpdater``, ``MaskEncoder`` residual blocks) into ``1x1 -> 3x3 -> 1x1``
   bottlenecks.
3. Structured L2-importance hidden-unit pruning of ``QueryTransformerBlock.ffn``.
4. Compact mixed-precision checkpoint serialization (``float16`` for large weight matrices,
   ``float32`` for 1D biases/norms/embeddings) with self-describing metadata so
   ``CUTIE.load_weights`` seamlessly loads both legacy upstream checkpoints and compressed
   checkpoints.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

COMPACT_CHECKPOINT_FORMAT = "tucker_bn_folded_v2"

# Default Tucker-2 bottleneck ranks (rank_in, rank_out) & FFN hidden dim
DEFAULT_TUCKER_CONFIG: Dict[str, Any] = {
    "fold_bn": True,
    "pix_enc_l3_rank": [96, 96],
    "ca_res_rank": [96, 96],
    "mask_enc_l2_rank": [64, 64],
    "mask_enc_l3_rank": [96, 96],
    "dec_up16_8_c1_rank": [96, 64],
    "dec_up16_8_c2_rank": [64, 64],
    "dec_up8_4_rank": [64, 64],
    "sensory_deep_rank": [144, 192],
    "sensory_dec_rank": [128, 192],
    "ffn_dim": 1024,
}


class TuckerConv2d(nn.Module):
    """Spatial-channel Tucker-2 factorized 4D convolution ``(1x1 -> 3x3 -> 1x1)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank_in: int,
        rank_out: int,
        *,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, rank_in, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(
            rank_in,
            rank_out,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.proj_out = nn.Conv2d(rank_out, out_channels, kernel_size=1, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.spatial(self.proj_in(x)))


class TuckerGConv2d(nn.Module):
    """Group-5D ``(B, num_objects, C, H, W)`` wrapper around a Tucker-2 bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank_in: int,
        rank_out: int,
        *,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, rank_in, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(
            rank_in,
            rank_out,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.proj_out = nn.Conv2d(rank_out, out_channels, kernel_size=1, bias=has_bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects = g.shape[:2]
        x = g.flatten(start_dim=0, end_dim=1)
        x = self.proj_out(self.spatial(self.proj_in(x)))
        return x.view(batch_size, num_objects, *x.shape[1:])


def fold_conv_bn_pair(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> Tuple[nn.Conv2d, nn.Identity]:
    """Algebraically fold ``BatchNorm2d`` affine + running stats into ``Conv2d``."""
    device = conv.weight.device
    dtype = conv.weight.dtype

    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        w = conv.weight.detach().float()
        b = (
            conv.bias.detach().float()
            if conv.bias is not None
            else torch.zeros(conv.out_channels, device=device, dtype=torch.float32)
        )
        gamma = bn.weight.detach().float()
        beta = bn.bias.detach().float()
        mean = bn.running_mean.detach().float()
        var = bn.running_var.detach().float()
        scale = gamma * torch.rsqrt(var + float(bn.eps))

        fused.weight.copy_((w * scale.view(-1, 1, 1, 1)).to(dtype=dtype))
        fused.bias.copy_(((b - mean) * scale + beta).to(dtype=dtype))

    return fused, nn.Identity()


def _replace_conv_with_bias_structure(conv: nn.Conv2d) -> Tuple[nn.Conv2d, nn.Identity]:
    """Create structure-only ``(Conv2d(bias=True), Identity)`` without computing stats."""
    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    return fused, nn.Identity()


def _iter_resnet_blocks(encoder: nn.Module, layer_names: Tuple[str, ...]):
    for lname in layer_names:
        seq = getattr(encoder, lname, None)
        if isinstance(seq, nn.Sequential):
            for block in seq:
                yield block


def fold_cutie_conv_bn_in_place(
    model: nn.Module, *, structure_only: bool = False
) -> None:
    """Fold all 48 Conv+BN pairs in ``PixelEncoder`` and ``MaskEncoder`` in-place."""
    if getattr(model, "_bn_folded", False):
        return

    transform_fn = (
        (lambda c, _b: _replace_conv_with_bias_structure(c))
        if structure_only
        else fold_conv_bn_pair
    )

    # 1. PixelEncoder (ResNet-50: conv1+bn1, res2, layer2, layer3)
    pe = getattr(model, "pixel_encoder", None)
    if pe is not None and isinstance(getattr(pe, "bn1", None), nn.BatchNorm2d):
        pe.conv1, pe.bn1 = transform_fn(pe.conv1, pe.bn1)
        for blk in _iter_resnet_blocks(pe, ("res2", "layer2", "layer3")):
            if isinstance(getattr(blk, "bn1", None), nn.BatchNorm2d):
                blk.conv1, blk.bn1 = transform_fn(blk.conv1, blk.bn1)
            if isinstance(getattr(blk, "bn2", None), nn.BatchNorm2d):
                blk.conv2, blk.bn2 = transform_fn(blk.conv2, blk.bn2)
            if isinstance(getattr(blk, "bn3", None), nn.BatchNorm2d):
                blk.conv3, blk.bn3 = transform_fn(blk.conv3, blk.bn3)
            ds = getattr(blk, "downsample", None)
            if (
                isinstance(ds, nn.Sequential)
                and len(ds) >= 2
                and isinstance(ds[0], nn.Conv2d)
                and isinstance(ds[1], nn.BatchNorm2d)
            ):
                c_new, b_new = transform_fn(ds[0], ds[1])
                ds[0] = c_new
                ds[1] = b_new

    # 2. MaskEncoder (ResNet-18: conv1+bn1, layer1, layer2, layer3)
    me = getattr(model, "mask_encoder", None)
    if me is not None and isinstance(getattr(me, "bn1", None), nn.BatchNorm2d):
        me.conv1, me.bn1 = transform_fn(me.conv1, me.bn1)
        for blk in _iter_resnet_blocks(me, ("layer1", "layer2", "layer3")):
            if isinstance(getattr(blk, "bn1", None), nn.BatchNorm2d):
                blk.conv1, blk.bn1 = transform_fn(blk.conv1, blk.bn1)
            if isinstance(getattr(blk, "bn2", None), nn.BatchNorm2d):
                blk.conv2, blk.bn2 = transform_fn(blk.conv2, blk.bn2)
            ds = getattr(blk, "downsample", None)
            if (
                isinstance(ds, nn.Sequential)
                and len(ds) >= 2
                and isinstance(ds[0], nn.Conv2d)
                and isinstance(ds[1], nn.BatchNorm2d)
            ):
                c_new, b_new = transform_fn(ds[0], ds[1])
                ds[0] = c_new
                ds[1] = b_new

    model._bn_folded = True


def hosvd_decompose_conv(
    conv: nn.Conv2d,
    rank_in: int,
    rank_out: int,
    *,
    is_group_5d: bool = False,
    structure_only: bool = False,
) -> nn.Module:
    """Decompose a ``3x3`` ``Conv2d`` / ``GConv2d`` into a Tucker-2 bottleneck via HOSVD."""
    c_out, c_in, kh, kw = conv.weight.shape
    if (kh, kw) != (3, 3):
        raise ValueError(f"Tucker-2 expects 3x3 kernel, got {(kh, kw)}")
    r_in = min(int(rank_in), c_in)
    r_out = min(int(rank_out), c_out)
    has_bias = conv.bias is not None
    stride = conv.stride[0] if isinstance(conv.stride, tuple) else int(conv.stride)
    padding = conv.padding[0] if isinstance(conv.padding, tuple) else int(conv.padding)
    dilation = (
        conv.dilation[0] if isinstance(conv.dilation, tuple) else int(conv.dilation)
    )

    cls = TuckerGConv2d if is_group_5d else TuckerConv2d
    tucker = cls(
        in_channels=c_in,
        out_channels=c_out,
        rank_in=r_in,
        rank_out=r_out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    if structure_only:
        return tucker

    with torch.no_grad():
        # Run SVD on CPU float32 for deterministic, full-precision singular vectors
        w_cpu = conv.weight.detach().float().cpu()
        # Mode-0 unfolding: (C_out, C_in * 9)
        u_out_full, _, _ = torch.linalg.svd(
            w_cpu.reshape(c_out, -1), full_matrices=False
        )
        u_out = u_out_full[:, :r_out]  # (C_out, r_out)

        # Mode-1 unfolding: (C_in, C_out * 9)
        u_in_full, _, _ = torch.linalg.svd(
            w_cpu.permute(1, 0, 2, 3).reshape(c_in, -1), full_matrices=False
        )
        u_in = u_in_full[:, :r_in]  # (C_in, r_in)

        # Core 3x3 tensor: G = W x_0 U_out^T x_1 U_in^T
        core = torch.einsum("oiyx,or,is->rsyx", w_cpu, u_out, u_in)

        dev = conv.weight.device
        dt = conv.weight.dtype
        tucker.proj_in.weight.copy_(u_in.T.unsqueeze(-1).unsqueeze(-1).to(dev, dt))
        tucker.spatial.weight.copy_(core.to(dev, dt))
        tucker.proj_out.weight.copy_(u_out.unsqueeze(-1).unsqueeze(-1).to(dev, dt))
        if has_bias and conv.bias is not None:
            tucker.proj_out.bias.copy_(conv.bias.detach().to(dev, dt))

    return tucker


def _prune_ffn_in_place(
    ffn: nn.Module, target_dim: int, *, structure_only: bool
) -> None:
    lin1: nn.Linear = ffn.linear1
    lin2: nn.Linear = ffn.linear2
    cur_hidden = lin1.out_features
    if target_dim <= 0 or target_dim >= cur_hidden:
        return

    dev = lin1.weight.device
    dt = lin1.weight.dtype
    new_lin1 = nn.Linear(lin1.in_features, target_dim, bias=lin1.bias is not None).to(
        dev, dt
    )
    new_lin2 = nn.Linear(target_dim, lin2.out_features, bias=lin2.bias is not None).to(
        dev, dt
    )

    if not structure_only:
        with torch.no_grad():
            w1 = lin1.weight.detach().float()  # (H_ff, D)
            w2 = lin2.weight.detach().float()  # (D, H_ff)
            b1 = (
                lin1.bias.detach().float()
                if lin1.bias is not None
                else torch.zeros(cur_hidden, device=dev)
            )
            # Importance of hidden neuron j = (||w1_j||_2 + 0.1*|b1_j|) * ||w2_{:,j}||_2
            imp = (w1.norm(dim=1) + 0.1 * b1.abs()) * w2.norm(dim=0)
            keep = torch.topk(imp, k=target_dim, largest=True, sorted=True).indices
            new_lin1.weight.copy_(lin1.weight.detach()[keep].to(dt))
            if lin1.bias is not None:
                new_lin1.bias.copy_(lin1.bias.detach()[keep].to(dt))
            new_lin2.weight.copy_(lin2.weight.detach()[:, keep].to(dt))
            if lin2.bias is not None:
                new_lin2.bias.copy_(lin2.bias.detach().to(dt))

    ffn.linear1 = new_lin1
    ffn.linear2 = new_lin2


def compress_and_factorize_cutie_in_place(
    model: nn.Module,
    tucker_cfg: Optional[Mapping[str, Any]] = None,
    *,
    structure_only: bool = False,
) -> Dict[str, Any]:
    """Apply Conv+BN folding, HOSVD Tucker-2 factorization, and FFN pruning in-place."""
    cfg = dict(DEFAULT_TUCKER_CONFIG)
    if tucker_cfg:
        cfg.update(dict(tucker_cfg))

    if getattr(model, "_tucker_factorized", False):
        return cfg

    if cfg.get("fold_bn", True):
        fold_cutie_conv_bn_in_place(model, structure_only=structure_only)

    # 0. PixelEncoder layer3 3x3 Bottleneck conv2 blocks (stride 16)
    r_pl3 = cfg.get("pix_enc_l3_rank")
    if r_pl3 and getattr(model, "pixel_encoder", None) is not None:
        for blk in model.pixel_encoder.layer3:
            if isinstance(getattr(blk, "conv2", None), nn.Conv2d):
                blk.conv2 = hosvd_decompose_conv(
                    blk.conv2,
                    r_pl3[0],
                    r_pl3[1],
                    is_group_5d=False,
                    structure_only=structure_only,
                )

    r_ca = cfg["ca_res_rank"]
    # 1. CAResBlocks in mask_encoder.fuser, pixel_fuser.fuser, and object_transformer.blocks.*.pixel_ffn
    ca_blocks = [
        model.mask_encoder.fuser.block1,
        model.mask_encoder.fuser.block2,
        model.pixel_fuser.fuser.block1,
        model.pixel_fuser.fuser.block2,
    ]
    for blk in model.object_transformer.blocks:
        ca_blocks.append(blk.pixel_ffn.conv)

    for cab in ca_blocks:
        cab.conv1 = hosvd_decompose_conv(
            cab.conv1,
            r_ca[0],
            r_ca[1],
            is_group_5d=False,
            structure_only=structure_only,
        )
        cab.conv2 = hosvd_decompose_conv(
            cab.conv2,
            r_ca[0],
            r_ca[1],
            is_group_5d=False,
            structure_only=structure_only,
        )

    # 2. MaskEncoder layer2 & layer3 per-object 3x3 BasicBlocks
    r_ml2 = cfg["mask_enc_l2_rank"]
    for blk in model.mask_encoder.layer2:
        blk.conv1 = hosvd_decompose_conv(
            blk.conv1,
            r_ml2[0],
            r_ml2[1],
            is_group_5d=False,
            structure_only=structure_only,
        )
        blk.conv2 = hosvd_decompose_conv(
            blk.conv2,
            r_ml2[0],
            r_ml2[1],
            is_group_5d=False,
            structure_only=structure_only,
        )

    r_ml3 = cfg["mask_enc_l3_rank"]
    for blk in model.mask_encoder.layer3:
        blk.conv1 = hosvd_decompose_conv(
            blk.conv1,
            r_ml3[0],
            r_ml3[1],
            is_group_5d=False,
            structure_only=structure_only,
        )
        blk.conv2 = hosvd_decompose_conv(
            blk.conv2,
            r_ml3[0],
            r_ml3[1],
            is_group_5d=False,
            structure_only=structure_only,
        )

    # 3. MaskDecoder GroupResBlocks (stride 8 and stride 4)
    r_d16_1 = cfg["dec_up16_8_c1_rank"]
    r_d16_2 = cfg["dec_up16_8_c2_rank"]
    r_d8_4 = cfg["dec_up8_4_rank"]
    model.mask_decoder.up_16_8.out_conv.conv1 = hosvd_decompose_conv(
        model.mask_decoder.up_16_8.out_conv.conv1,
        r_d16_1[0],
        r_d16_1[1],
        is_group_5d=True,
        structure_only=structure_only,
    )
    model.mask_decoder.up_16_8.out_conv.conv2 = hosvd_decompose_conv(
        model.mask_decoder.up_16_8.out_conv.conv2,
        r_d16_2[0],
        r_d16_2[1],
        is_group_5d=True,
        structure_only=structure_only,
    )
    model.mask_decoder.up_8_4.out_conv.conv1 = hosvd_decompose_conv(
        model.mask_decoder.up_8_4.out_conv.conv1,
        r_d8_4[0],
        r_d8_4[1],
        is_group_5d=True,
        structure_only=structure_only,
    )
    model.mask_decoder.up_8_4.out_conv.conv2 = hosvd_decompose_conv(
        model.mask_decoder.up_8_4.out_conv.conv2,
        r_d8_4[0],
        r_d8_4[1],
        is_group_5d=True,
        structure_only=structure_only,
    )

    # 4. SensoryDeepUpdater & SensoryUpdater 3x3 GRU transforms
    r_sdeep = cfg["sensory_deep_rank"]
    model.mask_encoder.sensory_update.transform = hosvd_decompose_conv(
        model.mask_encoder.sensory_update.transform,
        r_sdeep[0],
        r_sdeep[1],
        is_group_5d=True,
        structure_only=structure_only,
    )
    r_sdec = cfg["sensory_dec_rank"]
    model.mask_decoder.sensory_update.transform = hosvd_decompose_conv(
        model.mask_decoder.sensory_update.transform,
        r_sdec[0],
        r_sdec[1],
        is_group_5d=True,
        structure_only=structure_only,
    )

    # 5. QueryTransformerBlock FFN structured hidden-unit pruning
    ffn_dim = int(cfg.get("ffn_dim", 0))
    if ffn_dim > 0:
        for blk in model.object_transformer.blocks:
            _prune_ffn_in_place(blk.ffn, ffn_dim, structure_only=structure_only)

    model._tucker_factorized = True
    model._tucker_config = cfg
    return cfg


def apply_cutie_tucker_structure(
    model: nn.Module, tucker_cfg: Mapping[str, Any]
) -> None:
    """Materialize empty factorized/BN-folded module structure prior to state_dict load."""
    compress_and_factorize_cutie_in_place(model, tucker_cfg, structure_only=True)


def is_compact_cutie_checkpoint(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("__annolid_cutie_format__") == COMPACT_CHECKPOINT_FORMAT
    )


def pack_compact_cutie_checkpoint(
    model: nn.Module,
    *,
    tucker_cfg: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Pack a factorized CUTIE state_dict using lossless mixed-precision (FP16 matrices, FP32 vectors)."""
    resolved_cfg = dict(
        tucker_cfg or getattr(model, "_tucker_config", DEFAULT_TUCKER_CONFIG)
    )
    compact_weights: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in model.state_dict().items():
        cpu_t = tensor.detach().cpu()
        if cpu_t.is_floating_point() and cpu_t.ndim >= 2 and cpu_t.numel() >= 256:
            compact_weights[key] = cpu_t.clamp(-65504.0, 65504.0).to(torch.float16)
        else:
            compact_weights[key] = cpu_t.clone()

    payload: Dict[str, Any] = {
        "__annolid_cutie_format__": COMPACT_CHECKPOINT_FORMAT,
        "tucker_config": resolved_cfg,
        "weights": compact_weights,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload
