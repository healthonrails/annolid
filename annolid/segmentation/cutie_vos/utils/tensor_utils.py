from typing import Iterable
import torch
import torch.nn.functional as F


# STM
def pad_divide_by(in_img: torch.Tensor, d: int) -> (torch.Tensor, Iterable[int]):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img: torch.Tensor, pad: Iterable[int]) -> torch.Tensor:
    if len(img.shape) == 4:
        if pad[2] + pad[3] > 0:
            img = img[:, :, pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, pad[0] : -pad[1]]
    elif len(img.shape) == 3:
        if pad[2] + pad[3] > 0:
            img = img[:, pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, pad[0] : -pad[1]]
    elif len(img.shape) == 2:
        if pad[2] + pad[3] > 0:
            img = img[pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, pad[0] : -pad[1]]
    elif len(img.shape) == 5:
        if pad[2] + pad[3] > 0:
            img = img[:, :, :, pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, :, pad[0] : -pad[1]]
    else:
        raise NotImplementedError
    return img


DEFAULT_MAX_BG_AGGREGATION_OBJECTS = 16
DEFAULT_BG_AGGREGATION_FG_THRESHOLD = 0.05


# @torch.jit.script
def aggregate(
    prob: torch.Tensor,
    dim: int,
    *,
    max_bg_objects: int = DEFAULT_MAX_BG_AGGREGATION_OBJECTS,
    fg_threshold: float = DEFAULT_BG_AGGREGATION_FG_THRESHOLD,
) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        prob = prob.float()
        num_objects = prob.shape[dim]
        if num_objects == 0:
            bg_shape = list(prob.shape)
            bg_shape[dim] = 1
            bg_prob = prob.new_ones(bg_shape)
        elif 0 < max_bg_objects < num_objects:
            # With hundreds of tracked instances, multiplying `(1 - p_i)` over all
            # `N` channels causes exponential background underflow from tiny residual
            # noise (`(1 - 0.03)^200 = 0.0023 < 0.03`), turning background into false
            # foreground. By order-statistic union `top_k(prob, K) U {i : p_i > tau}`:
            # - When `p_(K) <= tau`, every channel with `p_i > tau` is already inside
            #   `top_k(K)`, so sub-threshold noise is bounded to at most `K` channels.
            # - When `p_(K) > tau` (more than `K` genuinely overlapping foreground
            #   instances at the same pixel), every channel with `p_i > tau` contributes
            #   with no `K`-object overlap ceiling.
            top_prob = torch.topk(prob, k=max_bg_objects, dim=dim).values
            log_bg = torch.log1p(-top_prob.clamp(max=1.0 - 1e-7)).sum(
                dim=dim, keepdim=True
            )
            if fg_threshold > 0.0:
                kth_prob = top_prob.narrow(dim, max_bg_objects - 1, 1)
                many_overlap = kth_prob > fg_threshold
                if bool(many_overlap.any()):
                    active_prob = torch.where(
                        prob > fg_threshold, prob, torch.zeros_like(prob)
                    )
                    dense_log_bg = torch.log1p(
                        -active_prob.clamp(max=1.0 - 1e-7)
                    ).sum(dim=dim, keepdim=True)
                    log_bg = torch.where(many_overlap, dense_log_bg, log_bg)
            bg_prob = torch.exp(log_bg)
        else:
            bg_prob = torch.prod(1.0 - prob, dim=dim, keepdim=True)
        new_prob = torch.cat([bg_prob, prob], dim).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))

        return logits


# @torch.jit.script
def cls_to_one_hot(cls_gt: torch.Tensor, num_objects: int) -> torch.Tensor:
    # cls_gt: B*1*H*W
    B, _, H, W = cls_gt.shape
    one_hot = torch.zeros(B, num_objects + 1, H, W, device=cls_gt.device).scatter_(
        1, cls_gt, 1
    )
    return one_hot
