from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import json
import os
import random
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from annolid.segmentation.dino_kpseg.data import (
    DinoKPSEGAugmentConfig,
    DinoKPSEGPoseDataset,
    build_extractor,
    load_coco_pose_spec,
    load_labelme_pose_spec,
    load_yolo_pose_spec,
    materialize_coco_pose_as_yolo,
    summarize_labelme_pose_labels,
    summarize_yolo_pose_labels,
)
from annolid.segmentation.dino_kpseg.format_utils import (
    normalize_dino_kpseg_data_format,
)
from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGCheckpointMeta,
    DinoKPSEGHead,
    DinoKPSEGAttentionHead,
    DinoKPSEGAlignedHead,
    DinoKPSEGHybridHead,
    DinoKPSEGMultiTaskHead,
    checkpoint_pack,
)
from annolid.segmentation.dino_kpseg.cli_utils import normalize_device, parse_layers
from annolid.segmentation.dino_kpseg.eval import (
    DinoKPSEGEvalAccumulator,
    _collect_gt_candidates,
    _min_error_px,
)
from annolid.segmentation.dino_kpseg.keypoints import (
    infer_left_right_pairs,
    infer_orientation_anchor_indices,
    symmetric_pairs_from_flip_idx,
)
from annolid.segmentation.dino_kpseg import defaults as dino_defaults
from annolid.utils.runs import allocate_run_dir, shared_runs_root
from annolid.utils.logger import logger


_AP_IOU_THRESHOLDS: Tuple[float, ...] = tuple(0.50 + 0.05 * i for i in range(10))

_PCK_DEFAULT_THRESHOLDS: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)


def _normalize_metric_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _metric_higher_is_better(metric_name: str) -> bool:
    name = _normalize_metric_name(metric_name)
    if not name:
        return False
    if "loss" in name:
        return False
    return True


def _parse_weight_list(text: Optional[str], *, n: int) -> Tuple[float, ...]:
    raw = str(text or "").strip()
    if not raw:
        return tuple(1.0 for _ in range(int(n)))
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != int(n):
        raise ValueError(f"Expected {int(n)} comma-separated floats, got {len(parts)}")
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    return tuple(out)


def _pck_from_summary(
    pck_vals: Dict[str, object],
    *,
    thr_px: float,
) -> Optional[float]:
    key = str(float(thr_px))
    value = pck_vals.get(key)
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _weighted_pck(
    pck_vals: Dict[str, object],
    *,
    thresholds_px: Sequence[float],
    weights: Sequence[float],
) -> Optional[float]:
    if len(thresholds_px) != len(weights):
        raise ValueError("thresholds_px and weights must have the same length")
    num = 0.0
    denom = 0.0
    for thr, w in zip(thresholds_px, weights):
        w = float(w)
        if w <= 0:
            continue
        v = _pck_from_summary(pck_vals, thr_px=float(thr))
        if v is None:
            continue
        num += w * float(v)
        denom += w
    if denom <= 0:
        return None
    return float(num / denom)


def _resolve_selection_metric(
    metric_name: str,
    *,
    train_loss: float,
    val_loss: Optional[float],
    pck_vals: Dict[str, object],
    pck_weighted_weights: Sequence[float],
) -> Optional[float]:
    name = _normalize_metric_name(metric_name)
    if name in {"val_loss", "loss/val", "val"}:
        if val_loss is None:
            return None
        return float(val_loss)
    if name in {"train_loss", "loss/train", "train"}:
        return float(train_loss)
    if name in {"pck@2px", "pck2", "pck_2"}:
        return _pck_from_summary(pck_vals, thr_px=2.0)
    if name in {"pck@4px", "pck4", "pck_4"}:
        return _pck_from_summary(pck_vals, thr_px=4.0)
    if name in {"pck@8px", "pck8", "pck_8"}:
        return _pck_from_summary(pck_vals, thr_px=8.0)
    if name in {"pck@16px", "pck16", "pck_16"}:
        return _pck_from_summary(pck_vals, thr_px=16.0)
    if name in {"pck_weighted", "weighted_pck"}:
        return _weighted_pck(
            pck_vals,
            thresholds_px=_PCK_DEFAULT_THRESHOLDS,
            weights=pck_weighted_weights,
        )
    return None


def _compute_no_aug_start_epoch(*, epochs: int, no_aug_epoch: int) -> Optional[int]:
    total = max(1, int(epochs))
    tail = max(0, int(no_aug_epoch))
    if tail <= 0:
        return None
    start = total - tail + 1
    return max(1, int(start))


def _is_epoch_augment_enabled(
    *,
    epoch: int,
    epochs: int,
    augment_enabled: bool,
    aug_start_epoch: int,
    aug_stop_epoch: int,
    no_aug_epoch: int,
) -> bool:
    if not bool(augment_enabled):
        return False
    ep = max(1, int(epoch))
    start = max(1, int(aug_start_epoch))
    stop = int(aug_stop_epoch)
    if ep < start:
        return False
    if stop > 0 and ep > stop:
        return False
    no_aug_start = _compute_no_aug_start_epoch(
        epochs=int(epochs), no_aug_epoch=int(no_aug_epoch)
    )
    if no_aug_start is not None and ep >= int(no_aug_start):
        return False
    return True


def _apply_schedule_profile_defaults(args: argparse.Namespace) -> None:
    profile = (
        str(getattr(args, "schedule_profile", "baseline") or "baseline").strip().lower()
    )
    if profile == "aggressive_s":
        args.epochs = 132
        args.warmup_epochs = 4
        args.flat_epoch = 64
        args.aug_start_epoch = 4
        args.aug_stop_epoch = 120
        args.no_aug_epoch = 12
        args.change_matcher = True
        args.matcher_change_epoch = 100
        args.iou_order_alpha = 4.0


def _resolve_feature_align_dim(
    value: object,
    *,
    in_dim: int,
    hidden_dim: int,
) -> int:
    """Resolve feature alignment dim from int/str value.

    Supports:
      - 0 or negative: disable adapter
      - positive int: explicit adapter dim
      - "auto": use hidden_dim (capped by in_dim when compressing)
    """
    in_ch = max(1, int(in_dim))
    hid = max(1, int(hidden_dim))
    if value is None:
        return 0
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return 0
        if token in {"0", "off", "none", "false", "no"}:
            return 0
        if token == "auto":
            return int(min(in_ch, hid))
        try:
            parsed = int(token)
        except Exception:
            raise ValueError(
                f"Unsupported feature_align_dim value: {value!r} (expected int or 'auto')"
            )
        return max(0, int(parsed))
    try:
        parsed = int(value)
    except Exception:
        raise ValueError(
            f"Unsupported feature_align_dim value: {value!r} (expected int or 'auto')"
        )
    return max(0, int(parsed))


def _load_train_config_defaults(config_path: Path) -> Dict[str, object]:
    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training config YAML: {path}")
    out: Dict[str, object] = {}
    for key, value in payload.items():
        k = str(key).strip().replace("-", "_")
        if not k:
            continue
        out[k] = value
    return out


def _average_precision(
    scores: Sequence[float],
    is_true_positive: Sequence[bool],
    *,
    num_gt: int,
) -> float:
    """Compute COCO-style AP (area under interpolated precision-recall curve).

    This assumes one-to-one matching is already enforced (for DinoKPSEG it's
    naturally 1 prediction per keypoint per image at most).
    """
    n = len(scores)
    if num_gt <= 0:
        return 0.0
    if n == 0:
        return 0.0

    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)
    tp = [1.0 if bool(is_true_positive[i]) else 0.0 for i in order]
    fp = [0.0 if bool(is_true_positive[i]) else 1.0 for i in order]

    tp_cum: List[float] = []
    fp_cum: List[float] = []
    t, f = 0.0, 0.0
    for i in range(n):
        t += tp[i]
        f += fp[i]
        tp_cum.append(t)
        fp_cum.append(f)

    recalls = [tpi / float(num_gt) for tpi in tp_cum]
    precisions = [tpi / max(1e-12, (tpi + fpi)) for tpi, fpi in zip(tp_cum, fp_cum)]

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def _oks_from_distance(dist_px: torch.Tensor, *, sigma_px: float) -> torch.Tensor:
    """OKS-like similarity in [0,1] from pixel distance."""
    sigma = max(1e-6, float(sigma_px))
    return torch.exp(-(dist_px**2) / (2.0 * sigma * sigma))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _grid_images(
    images: torch.Tensor,
    *,
    nrow: int,
    pad: int = 2,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Create a simple grid tensor without torchvision.

    Args:
        images: Tensor of shape [N, C, H, W] on CPU.
    Returns:
        Tensor of shape [C, H_grid, W_grid].
    """
    if images.ndim != 4:
        raise ValueError("Expected images in NCHW format")
    n, c, h, w = images.shape
    if n == 0:
        raise ValueError("No images to grid")
    nrow = max(1, int(nrow))
    ncol = (n + nrow - 1) // nrow
    grid_h = ncol * h + pad * (ncol - 1)
    grid_w = nrow * w + pad * (nrow - 1)
    grid = torch.full((c, grid_h, grid_w), float(pad_value), dtype=images.dtype)
    for idx in range(n):
        r = idx // nrow
        col = idx % nrow
        y0 = r * (h + pad)
        x0 = col * (w + pad)
        grid[:, y0 : y0 + h, x0 : x0 + w] = images[idx]
    return grid


def _stack_chw_images_with_padding(
    images: Sequence[torch.Tensor],
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Stack a list of CHW tensors, padding to the max H/W.

    TensorBoard's `add_images` requires a single (N,C,H,W) tensor; in DinoKPSEG
    we sometimes log overlays for different samples with different cropped
    extents (e.g. varying resized_w), so we pad instead of failing.
    """
    if not images:
        raise ValueError("images must be non-empty")
    max_h = 0
    max_w = 0
    for img in images:
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            raise ValueError("Expected CHW torch.Tensor images")
        _, h, w = img.shape
        max_h = max(max_h, int(h))
        max_w = max(max_w, int(w))
    padded: List[torch.Tensor] = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - int(h)
        pad_w = max_w - int(w)
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Negative padding computed unexpectedly")
        if pad_h or pad_w:
            # Pad format: (left, right, top, bottom) for last two dims.
            img = F.pad(img, (0, pad_w, 0, pad_h), value=float(pad_value))
        padded.append(img)
    return torch.stack(padded, dim=0)


def _overlay_keypoints(
    image: torch.Tensor,
    *,
    pred_xy: Sequence[Tuple[float, float]],
    gt_xy: Sequence[Tuple[float, float]],
    pred_color: Tuple[int, int, int] = (230, 50, 50),
    gt_color: Tuple[int, int, int] = (50, 230, 50),
    radius: int = 3,
) -> torch.Tensor:
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("Expected image in CHW format with 3 channels")
    img = image.detach().float().cpu().clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8, copy=True)
    height, width = int(arr.shape[0]), int(arr.shape[1])

    def _draw(
        points: Sequence[Tuple[float, float]], color: Tuple[int, int, int]
    ) -> None:
        for x, y in points:
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            cx = int(round(float(x)))
            cy = int(round(float(y)))
            if cx < 0 or cy < 0 or cx >= width or cy >= height:
                continue
            for dy in range(-radius, radius + 1):
                yy = cy + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = cx + dx
                    if xx < 0 or xx >= width:
                        continue
                    arr[yy, xx, 0] = int(color[0])
                    arr[yy, xx, 1] = int(color[1])
                    arr[yy, xx, 2] = int(color[2])

    _draw(gt_xy, gt_color)
    _draw(pred_xy, pred_color)
    out = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return out


def _draw_loss_curve_image(
    csv_path: Path, *, width: int = 720, height: int = 420
) -> Optional[torch.Tensor]:
    """Draw a loss curve PNG as a CHW float tensor in [0,1] without matplotlib."""
    try:
        text = csv_path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except Exception:
        return None
    if not rows:
        return None

    xs = []
    train_ys = []
    val_ys = []
    for row in rows:
        try:
            epoch = int(float(row.get("epoch") or 0))
        except Exception:
            continue
        tr = row.get("train_loss") or ""
        va = row.get("val_loss") or ""
        try:
            tr_v = float(tr)
        except Exception:
            tr_v = None
        if tr_v is not None and not math.isfinite(float(tr_v)):
            tr_v = None
        try:
            va_v = float(va)
        except Exception:
            va_v = None
        if va_v is not None and not math.isfinite(float(va_v)):
            va_v = None
        if tr_v is None and va_v is None:
            continue
        xs.append(epoch)
        train_ys.append(tr_v)
        val_ys.append(va_v)

    if not xs:
        return None

    all_vals = [
        float(v) for v in train_ys + val_ys if v is not None and math.isfinite(float(v))
    ]
    if not all_vals:
        return None
    y_min = min(all_vals)
    y_max = max(all_vals)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except Exception:
        return None

    img = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 48
    x0, y0 = margin, margin
    x1, y1 = int(width) - margin, int(height) - margin

    draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=2)
    draw.text((x0, 10), "DinoKPSEG loss curve", fill=(20, 20, 20))

    x_min = min(xs)
    x_max = max(xs)
    if x_max == x_min:
        x_max = x_min + 1

    def to_xy(epoch: int, loss: float) -> tuple[int, int]:
        xf = (epoch - x_min) / (x_max - x_min)
        yf = (loss - y_min) / (y_max - y_min)
        px = int(x0 + xf * (x1 - x0))
        py = int(y1 - yf * (y1 - y0))
        return px, py

    def draw_series(values, color):
        pts = []
        for epoch, loss in zip(xs, values):
            if loss is None:
                pts.append(None)
                continue
            if not math.isfinite(float(loss)):
                pts.append(None)
                continue
            pts.append(to_xy(epoch, float(loss)))
        last = None
        for pt in pts:
            if pt is None:
                last = None
                continue
            if last is not None:
                draw.line([last, pt], fill=color, width=2)
            last = pt

    draw_series(train_ys, (35, 99, 229))
    if any(v is not None for v in val_ys):
        draw_series(val_ys, (229, 57, 53))

    # Legend
    draw.rectangle([x0 + 8, y0 + 8, x0 + 18, y0 + 18], fill=(35, 99, 229))
    draw.text((x0 + 24, y0 + 6), "train", fill=(20, 20, 20))
    draw.rectangle([x0 + 80, y0 + 8, x0 + 90, y0 + 18], fill=(229, 57, 53))
    draw.text((x0 + 96, y0 + 6), "val", fill=(20, 20, 20))

    arr = np.array(img, dtype=np.uint8, copy=True)
    tens = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    return tens


def _sanitize_tb_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Clamp and denoise tensorboard image tensors to avoid NaN cast warnings."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected a torch.Tensor image")
    out = image.detach().float().cpu()
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.clamp(0.0, 1.0)


def _tb_add_image(
    tb_writer: SummaryWriter, tag: str, image: torch.Tensor, step: int
) -> None:
    tb_writer.add_image(tag, _sanitize_tb_image_tensor(image), int(step))


def _tb_add_images(
    tb_writer: SummaryWriter,
    tag: str,
    images: torch.Tensor,
    step: int,
) -> None:
    tb_writer.add_images(tag, _sanitize_tb_image_tensor(images), int(step))


def _dice_loss(
    probs: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    if probs.shape != targets.shape:
        raise ValueError("Dice loss requires probs and targets with the same shape")
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _soft_argmax_coords(
    probs: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if probs.ndim != 3:
        raise ValueError("Expected probs in KHW format")
    h_p, w_p = int(probs.shape[1]), int(probs.shape[2])
    norm = probs.sum(dim=(1, 2), keepdim=False).clamp(min=1e-6)
    xs = (torch.arange(w_p, device=probs.device, dtype=probs.dtype) + 0.5) * float(
        patch_size
    )
    ys = (torch.arange(h_p, device=probs.device, dtype=probs.dtype) + 0.5) * float(
        patch_size
    )
    x_exp = (probs.sum(dim=1) * xs[None, :]).sum(dim=1) / norm
    y_exp = (probs.sum(dim=2) * ys[None, :]).sum(dim=1) / norm
    return torch.stack([x_exp, y_exp], dim=1)


def _soft_argmax_coords_batched(
    probs_bkhw: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if probs_bkhw.ndim != 4:
        raise ValueError("Expected probs in BKHW format")
    b, k, h_p, w_p = (
        int(probs_bkhw.shape[0]),
        int(probs_bkhw.shape[1]),
        int(probs_bkhw.shape[2]),
        int(probs_bkhw.shape[3]),
    )
    norm = probs_bkhw.sum(dim=(2, 3), keepdim=False).clamp(min=1e-6)  # [B,K]
    xs = (
        torch.arange(w_p, device=probs_bkhw.device, dtype=probs_bkhw.dtype) + 0.5
    ) * float(patch_size)
    ys = (
        torch.arange(h_p, device=probs_bkhw.device, dtype=probs_bkhw.dtype) + 0.5
    ) * float(patch_size)
    x_exp = (probs_bkhw.sum(dim=2) * xs[None, None, :]).sum(dim=2) / norm
    y_exp = (probs_bkhw.sum(dim=3) * ys[None, None, :]).sum(dim=2) / norm
    out = torch.stack([x_exp, y_exp], dim=2)
    if out.shape != (b, k, 2):
        raise RuntimeError("Unexpected soft-argmax output shape")
    return out


def _pad_collate(samples: list[dict], *, patch_size: int) -> dict:
    if not samples:
        raise ValueError("No samples to collate")

    feats0 = samples[0].get("feats")
    masks0 = samples[0].get("masks")
    if not isinstance(feats0, torch.Tensor) or feats0.ndim != 3:
        raise ValueError("Sample feats must be a CHW torch.Tensor")
    if not isinstance(masks0, torch.Tensor) or masks0.ndim != 3:
        raise ValueError("Sample masks must be a KHW torch.Tensor")

    b = len(samples)
    c = int(feats0.shape[0])
    k = int(masks0.shape[0])
    hs: List[int] = []
    ws: List[int] = []
    for s in samples:
        feats = s.get("feats")
        masks = s.get("masks")
        if not isinstance(feats, torch.Tensor) or feats.ndim != 3:
            raise ValueError("Sample feats must be a CHW torch.Tensor")
        if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
            raise ValueError("Sample masks must be a KHW torch.Tensor")
        if int(feats.shape[0]) != c:
            raise ValueError("Mismatched feature channels in batch")
        if int(masks.shape[0]) != k:
            raise ValueError("Mismatched keypoint count in batch")
        hs.append(int(feats.shape[1]))
        ws.append(int(feats.shape[2]))

    h_max = max(1, max(hs))
    w_max = max(1, max(ws))
    feats_bchw = torch.zeros((b, c, h_max, w_max), dtype=feats0.dtype)
    masks_bkhw = torch.zeros((b, k, h_max, w_max), dtype=masks0.dtype)
    valid_mask = torch.zeros((b, 1, h_max, w_max), dtype=torch.bool)

    coords_list: List[torch.Tensor] = []
    coord_mask_list: List[torch.Tensor] = []

    have_images = any(isinstance(s.get("image"), torch.Tensor) for s in samples)
    images_bchw = None
    if have_images:
        images_bchw = torch.zeros(
            (b, 3, h_max * int(patch_size), w_max * int(patch_size)),
            dtype=torch.float32,
        )

    have_gt = any("gt_instances" in s for s in samples)
    have_hw = any("image_hw" in s for s in samples)
    gt_instances_batch: list[list[np.ndarray]] = []
    image_hw_batch: list[tuple[int, int]] = []

    for idx, s in enumerate(samples):
        feats = s["feats"]
        masks = s["masks"]
        h_i = int(feats.shape[1])
        w_i = int(feats.shape[2])
        feats_bchw[idx, :, :h_i, :w_i] = feats
        masks_bkhw[idx, :, :h_i, :w_i] = masks
        valid_mask[idx, 0, :h_i, :w_i] = True

        coords = s.get("coords")
        coord_mask = s.get("coord_mask")
        if isinstance(coords, torch.Tensor):
            coords_list.append(coords)
        else:
            coords_list.append(torch.zeros((k, 2), dtype=torch.float32))
        if isinstance(coord_mask, torch.Tensor):
            coord_mask_list.append(coord_mask)
        else:
            coord_mask_list.append(torch.zeros((k,), dtype=torch.float32))

        if images_bchw is not None:
            img = s.get("image")
            if (
                isinstance(img, torch.Tensor)
                and img.ndim == 3
                and int(img.shape[0]) == 3
            ):
                images_bchw[idx, :, : int(img.shape[1]), : int(img.shape[2])] = img.to(
                    dtype=torch.float32
                )
        if have_gt:
            gt_instances = s.get("gt_instances")
            if isinstance(gt_instances, list):
                gt_instances_batch.append(gt_instances)
            else:
                gt_instances_batch.append([])
        if have_hw:
            image_hw = s.get("image_hw")
            parsed = None
            if isinstance(image_hw, (list, tuple)) and len(image_hw) == 2:
                parsed = (int(image_hw[0]), int(image_hw[1]))
            elif isinstance(image_hw, np.ndarray) and int(image_hw.size) == 2:
                flat = image_hw.reshape(-1).tolist()
                parsed = (int(flat[0]), int(flat[1]))
            elif isinstance(image_hw, torch.Tensor) and int(image_hw.numel()) == 2:
                flat = image_hw.reshape(-1).detach().cpu().tolist()
                parsed = (int(flat[0]), int(flat[1]))
            if parsed is None:
                image_hw_batch.append((0, 0))
            else:
                image_hw_batch.append((int(parsed[0]), int(parsed[1])))

    coords_bk2 = torch.stack(coords_list, dim=0).to(dtype=torch.float32)
    coord_mask_bk = torch.stack(coord_mask_list, dim=0).to(dtype=torch.float32)
    key_padding_mask = (~valid_mask.view(b, -1)).contiguous()

    batch = {
        "feats": feats_bchw,
        "masks": masks_bkhw,
        "coords": coords_bk2,
        "coord_mask": coord_mask_bk,
        "valid_mask": valid_mask,
        "key_padding_mask": key_padding_mask,
    }
    if images_bchw is not None:
        batch["image"] = images_bchw
    if have_gt:
        batch["gt_instances"] = gt_instances_batch
    if have_hw:
        batch["image_hw"] = image_hw_batch
    return batch


def _masked_bce_with_logits(
    logits_bkhw: torch.Tensor,
    targets_bkhw: torch.Tensor,
    valid_mask_b1hw: torch.Tensor,
    *,
    balanced: bool,
    max_pos_weight: float,
) -> torch.Tensor:
    if logits_bkhw.shape != targets_bkhw.shape:
        raise ValueError("Logits and targets must have the same shape")
    if valid_mask_b1hw.ndim != 4:
        raise ValueError("valid_mask must be B1HW")
    valid = valid_mask_b1hw.to(dtype=logits_bkhw.dtype)
    denom = (valid.sum() * float(logits_bkhw.shape[1])).clamp(min=1.0)

    pos_weight = None
    if balanced:
        with torch.no_grad():
            pos = (targets_bkhw * valid).sum(dim=(0, 2, 3))
            neg = ((1.0 - targets_bkhw) * valid).sum(dim=(0, 2, 3))
            pos_weight = neg / pos.clamp(min=1e-6)
            # If a keypoint has no positives in this batch, don't upweight it.
            pos_weight = torch.where(
                pos >= 1.0, pos_weight, torch.ones_like(pos_weight)
            )
            pos_weight = pos_weight.clamp(1.0, float(max_pos_weight))
            # Ensure broadcasting targets the channel dimension for BKHW.
            pos_weight = pos_weight.view(1, -1, 1, 1).to(
                dtype=logits_bkhw.dtype, device=logits_bkhw.device
            )

    bce = F.binary_cross_entropy_with_logits(
        logits_bkhw,
        targets_bkhw,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (bce * valid).sum() / denom


def _masked_focal_bce_with_logits(
    logits_bkhw: torch.Tensor,
    targets_bkhw: torch.Tensor,
    valid_mask_b1hw: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    if logits_bkhw.shape != targets_bkhw.shape:
        raise ValueError("Logits and targets must have the same shape")
    if valid_mask_b1hw.ndim != 4:
        raise ValueError("valid_mask must be B1HW")
    valid = valid_mask_b1hw.to(dtype=logits_bkhw.dtype)
    denom = (valid.sum() * float(logits_bkhw.shape[1])).clamp(min=1.0)
    probs = torch.sigmoid(logits_bkhw)
    bce = F.binary_cross_entropy_with_logits(
        logits_bkhw, targets_bkhw, reduction="none"
    )
    alpha_t = float(alpha) * targets_bkhw + (1.0 - float(alpha)) * (1.0 - targets_bkhw)
    p_t = probs * targets_bkhw + (1.0 - probs) * (1.0 - targets_bkhw)
    loss = alpha_t * ((1.0 - p_t) ** float(gamma)) * bce
    return (loss * valid).sum() / denom


def _dice_loss_masked(
    probs_bkhw: torch.Tensor,
    targets_bkhw: torch.Tensor,
    valid_mask_b1hw: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if probs_bkhw.shape != targets_bkhw.shape:
        raise ValueError("Dice loss requires probs and targets with the same shape")
    valid = valid_mask_b1hw.to(dtype=probs_bkhw.dtype)
    dims = (0, 2, 3)
    inter = (probs_bkhw * targets_bkhw * valid).sum(dim=dims)
    denom = (probs_bkhw * valid).sum(dim=dims) + (targets_bkhw * valid).sum(dim=dims)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return 1.0 - dice.mean()


def _instance_box_targets_from_masks(
    instance_mask_b1hw: torch.Tensor,
    *,
    valid_mask_b1hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized xyxy box targets from instance masks.

    Returns:
      boxes_b4 in [0,1], has_box_b bool
    """
    if instance_mask_b1hw.ndim != 4 or int(instance_mask_b1hw.shape[1]) != 1:
        raise ValueError("instance_mask_b1hw must be B1HW")
    b, _, h, w = instance_mask_b1hw.shape
    valid = valid_mask_b1hw.to(dtype=torch.bool)
    present = (instance_mask_b1hw > 0.5) & valid
    boxes = torch.zeros((b, 4), dtype=torch.float32, device=instance_mask_b1hw.device)
    has_box = torch.zeros((b,), dtype=torch.bool, device=instance_mask_b1hw.device)
    for i in range(int(b)):
        ys, xs = torch.where(present[i, 0])
        if int(xs.numel()) <= 0:
            continue
        x1 = float(xs.min().item()) / max(1.0, float(w - 1))
        y1 = float(ys.min().item()) / max(1.0, float(h - 1))
        x2 = float(xs.max().item()) / max(1.0, float(w - 1))
        y2 = float(ys.max().item()) / max(1.0, float(h - 1))
        boxes[i] = torch.tensor(
            [x1, y1, x2, y2], dtype=torch.float32, device=boxes.device
        )
        has_box[i] = True
    return boxes, has_box


def _ema_update_(
    ema_model: torch.nn.Module, model: torch.nn.Module, *, decay: float
) -> None:
    d = float(min(max(float(decay), 0.0), 0.999999))
    with torch.no_grad():
        msd = model.state_dict()
        for k, v_ema in ema_model.state_dict().items():
            v = msd.get(k)
            if v is None:
                continue
            if not torch.is_floating_point(v_ema):
                v_ema.copy_(v)
                continue
            v_ema.mul_(d).add_(v.detach(), alpha=(1.0 - d))


def _coord_loss(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if pred_xy.shape != target_xy.shape:
        raise ValueError("coord loss expects pred_xy and target_xy with same shape")
    mask = mask.to(dtype=pred_xy.dtype)
    if mask.ndim == 1:
        mask = mask[:, None]
    valid = mask.sum()
    if valid <= 0:
        return torch.zeros((), dtype=pred_xy.dtype, device=pred_xy.device)

    mode_norm = str(mode or "smooth_l1").strip().lower()
    if mode_norm == "l2":
        per = (pred_xy - target_xy) ** 2
        per = per.sum(dim=1, keepdim=True)
    elif mode_norm == "l1":
        per = torch.abs(pred_xy - target_xy).sum(dim=1, keepdim=True)
    else:
        per = F.smooth_l1_loss(pred_xy, target_xy, reduction="none").sum(
            dim=1, keepdim=True
        )
    return (per * mask).sum() / valid


def _compute_resize_hw(
    *, width: int, height: int, short_side: int, patch_size: int
) -> tuple[int, int]:
    if height <= width:
        scale = float(short_side) / max(1, int(height))
    else:
        scale = float(short_side) / max(1, int(width))
    new_w = max(
        int(patch_size),
        int(((width * scale) + patch_size - 1) // patch_size) * int(patch_size),
    )
    new_h = max(
        int(patch_size),
        int(((height * scale) + patch_size - 1) // patch_size) * int(patch_size),
    )
    return int(new_h), int(new_w)


def _pil_to_tensor_rgb(pil, *, out_hw: tuple[int, int]) -> Optional[torch.Tensor]:
    try:
        from PIL import ImageOps
        import numpy as np

        pil = ImageOps.exif_transpose(pil.convert("RGB"))
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        pil = pil.resize((out_w, out_h))
        arr = np.array(pil, dtype=np.uint8, copy=True)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = arr[..., :3]
        return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
    except Exception:
        return None


def _log_example_images(
    tb_writer: SummaryWriter,
    *,
    tag: str,
    image_paths: list[Path],
    short_side: int,
    patch_size: int,
    max_images: int = 8,
) -> None:
    if not image_paths:
        return
    selected = image_paths[: max(0, int(max_images))]
    tensors = []
    for p in selected:
        try:
            from PIL import Image

            pil = Image.open(p)
        except Exception:
            continue
        out_hw = _compute_resize_hw(
            width=pil.size[0],
            height=pil.size[1],
            short_side=int(short_side),
            patch_size=int(patch_size),
        )
        tens = _pil_to_tensor_rgb(pil, out_hw=out_hw)
        if tens is None:
            continue
        tensors.append(tens.clamp(0.0, 1.0))

    if not tensors:
        return

    try:
        imgs = torch.stack(tensors, dim=0)
        grid = _grid_images(imgs, nrow=min(4, int(imgs.shape[0])), pad=2, pad_value=0.0)
        _tb_add_image(tb_writer, tag, grid, 0)
    except Exception:
        for idx, img in enumerate(tensors):
            _tb_add_image(tb_writer, f"{tag}/{idx}", img, 0)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def train(
    *,
    data_yaml: Path,
    data_format: str = "auto",
    output_dir: Path,
    model_name: str,
    short_side: int,
    layers: Tuple[int, ...],
    feature_merge: str = dino_defaults.FEATURE_MERGE,
    feature_align_dim: object = dino_defaults.FEATURE_ALIGN_DIM,
    radius_px: float,
    mask_type: str = dino_defaults.MASK_TYPE,
    heatmap_sigma_px: Optional[float] = None,
    instance_mode: str = dino_defaults.INSTANCE_MODE,
    bbox_scale: float = dino_defaults.BBOX_SCALE,
    hidden_dim: int,
    lr: float,
    epochs: int,
    threshold: float,
    batch_size: int = dino_defaults.BATCH,
    accumulate: int = 1,
    grad_clip: float = 1.0,
    balanced_bce: bool = True,
    max_pos_weight: float = 50.0,
    cos_lr: bool = True,
    warmup_epochs: int = 3,
    lr_final_frac: float = 0.01,
    flat_epoch: int = dino_defaults.FLAT_EPOCH,
    device: Optional[str] = None,
    cache_features: bool = True,
    augment: Optional[DinoKPSEGAugmentConfig] = None,
    aug_start_epoch: int = dino_defaults.AUG_START_EPOCH,
    aug_stop_epoch: int = dino_defaults.AUG_STOP_EPOCH,
    no_aug_epoch: int = dino_defaults.NO_AUG_EPOCH,
    schedule_profile: str = dino_defaults.SCHEDULE_PROFILE,
    early_stop_patience: int = dino_defaults.EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = dino_defaults.EARLY_STOP_MIN_DELTA,
    early_stop_min_epochs: int = dino_defaults.EARLY_STOP_MIN_EPOCHS,
    best_metric: str = dino_defaults.BEST_METRIC,
    early_stop_metric: Optional[str] = None,
    pck_weighted_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    tb_add_graph: bool = False,
    bce_type: str = dino_defaults.BCE_TYPE,
    focal_alpha: float = dino_defaults.FOCAL_ALPHA,
    focal_gamma: float = dino_defaults.FOCAL_GAMMA,
    coord_warmup_epochs: int = dino_defaults.COORD_WARMUP_EPOCHS,
    radius_schedule: str = dino_defaults.RADIUS_SCHEDULE,
    radius_start_px: Optional[float] = dino_defaults.RADIUS_START_PX,
    radius_end_px: Optional[float] = dino_defaults.RADIUS_END_PX,
    overfit_n: int = 0,
    seed: Optional[int] = None,
    tb_projector: bool = dino_defaults.TB_PROJECTOR,
    tb_projector_split: str = dino_defaults.TB_PROJECTOR_SPLIT,
    tb_projector_max_images: int = dino_defaults.TB_PROJECTOR_MAX_IMAGES,
    tb_projector_max_patches: int = dino_defaults.TB_PROJECTOR_MAX_PATCHES,
    tb_projector_per_image_per_keypoint: int = dino_defaults.TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT,
    tb_projector_pos_threshold: float = dino_defaults.TB_PROJECTOR_POS_THRESHOLD,
    tb_projector_crop_px: int = dino_defaults.TB_PROJECTOR_CROP_PX,
    tb_projector_sprite_border_px: int = dino_defaults.TB_PROJECTOR_SPRITE_BORDER_PX,
    tb_projector_add_negatives: bool = dino_defaults.TB_PROJECTOR_ADD_NEGATIVES,
    tb_projector_neg_threshold: float = dino_defaults.TB_PROJECTOR_NEG_THRESHOLD,
    tb_projector_negatives_per_image: int = dino_defaults.TB_PROJECTOR_NEGATIVES_PER_IMAGE,
    head_type: str = dino_defaults.HEAD_TYPE,
    attn_heads: int = dino_defaults.ATTN_HEADS,
    attn_layers: int = dino_defaults.ATTN_LAYERS,
    dice_loss_weight: float = dino_defaults.DICE_LOSS_WEIGHT,
    coord_loss_weight: float = dino_defaults.COORD_LOSS_WEIGHT,
    coord_loss_type: str = dino_defaults.COORD_LOSS_TYPE,
    obj_loss_weight: float = dino_defaults.OBJ_LOSS_WEIGHT,
    box_loss_weight: float = dino_defaults.BOX_LOSS_WEIGHT,
    inst_loss_weight: float = dino_defaults.INST_LOSS_WEIGHT,
    multitask_aux_warmup_epochs: int = dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
    use_ema: bool = dino_defaults.USE_EMA,
    ema_decay: float = dino_defaults.EMA_DECAY,
    lr_pair_loss_weight: float = dino_defaults.LR_PAIR_LOSS_WEIGHT,
    lr_pair_margin_px: float = dino_defaults.LR_PAIR_MARGIN_PX,
    lr_side_loss_weight: float = dino_defaults.LR_SIDE_LOSS_WEIGHT,
    lr_side_loss_margin: float = dino_defaults.LR_SIDE_LOSS_MARGIN,
    change_matcher: bool = dino_defaults.AGGRESSIVE_S_CHANGE_MATCHER,
    matcher_change_epoch: int = dino_defaults.AGGRESSIVE_S_MATCHER_CHANGE_EPOCH,
    iou_order_alpha: float = dino_defaults.AGGRESSIVE_S_IOU_ORDER_ALPHA,
    log_every_steps: int = 100,
) -> Path:
    if seed is not None:
        _set_global_seed(int(seed))

    requested_data_format = str(data_format or "auto").strip().lower()
    if requested_data_format not in {"auto", "yolo", "labelme", "coco"}:
        raise ValueError(f"Unsupported data_format: {requested_data_format!r}")
    feature_merge = str(feature_merge or "concat").strip().lower()
    if feature_merge not in {"concat", "mean", "max"}:
        raise ValueError(
            f"Unsupported feature_merge: {feature_merge!r} (expected concat/mean/max)"
        )

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    payload = payload if isinstance(payload, dict) else {}
    data_format_norm = normalize_dino_kpseg_data_format(
        payload,
        data_format=requested_data_format,
    )

    staged_yolo_yaml: Optional[Path] = None
    label_format = "yolo"
    source_yaml = Path(data_yaml)
    if data_format_norm == "coco":
        coco_spec = load_coco_pose_spec(data_yaml)
        staged_dir = (output_dir / "dataset_coco_yolo").resolve()
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        staged_yolo_yaml = materialize_coco_pose_as_yolo(
            spec=coco_spec,
            output_dir=staged_dir,
        )
        source_yaml = staged_yolo_yaml
        label_format = "yolo"

    if data_format_norm == "labelme":
        spec_lm = load_labelme_pose_spec(source_yaml)
        train_images = list(spec_lm.train_images)
        val_images = list(spec_lm.val_images)
        train_label_paths = list(spec_lm.train_json)
        val_label_paths = list(spec_lm.val_json)
        keypoint_names = list(spec_lm.keypoint_names)
        flip_idx = spec_lm.flip_idx
        kpt_count = int(spec_lm.kpt_count)
        kpt_dims = int(spec_lm.kpt_dims)
        label_format = "labelme"
    else:
        spec = load_yolo_pose_spec(source_yaml)
        train_images = list(spec.train_images)
        val_images = list(spec.val_images)
        train_label_paths = None
        val_label_paths = None
        keypoint_names = list(spec.keypoint_names) if spec.keypoint_names else None
        flip_idx = spec.flip_idx
        kpt_count = int(spec.kpt_count)
        kpt_dims = int(spec.kpt_dims)
        label_format = "yolo"
    if not train_images:
        raise ValueError("No training images found")

    if data_format_norm == "labelme":
        summary_lm = summarize_labelme_pose_labels(
            train_images,
            label_paths=train_label_paths,
            keypoint_names=list(keypoint_names or []),
            kpt_dims=kpt_dims,
            max_issues=8,
        )
        if summary_lm.images_with_pose_instances <= 0:
            details = (
                "\n".join(summary_lm.example_issues)
                if summary_lm.example_issues
                else "(no details)"
            )
            raise ValueError(
                "No valid LabelMe pose labels found for DinoKPSEG training.\n\n"
                f"- images_total: {summary_lm.images_total}\n"
                f"- label_files_found: {summary_lm.label_files_found}\n"
                f"- images_with_pose_instances: {summary_lm.images_with_pose_instances}\n"
                f"- invalid_shapes_total: {summary_lm.invalid_shapes_total}\n\n"
                f"Examples:\n{details}\n\n"
                "Expected LabelMe JSON shapes with:\n"
                "  - point shapes: keypoints (label must match keypoint name)\n"
                "  - polygon shapes: optional instance segmentation (used for bbox crops)\n"
                "  - group_id: to associate shapes into instances\n"
            )
        if summary_lm.label_files_found < summary_lm.images_total:
            logger.warning(
                "DinoKPSEG dataset has %d/%d images missing LabelMe JSONs (first few issues: %s)",
                summary_lm.images_total - summary_lm.label_files_found,
                summary_lm.images_total,
                "; ".join(summary_lm.example_issues[:3]),
            )
    else:
        summary = summarize_yolo_pose_labels(
            train_images,
            kpt_count=kpt_count,
            kpt_dims=kpt_dims,
            max_issues=8,
        )
        if summary.images_with_pose_instances <= 0:
            details = (
                "\n".join(summary.example_issues)
                if summary.example_issues
                else "(no details)"
            )
            raise ValueError(
                "No valid YOLO pose labels found for DinoKPSEG training.\n\n"
                f"- images_total: {summary.images_total}\n"
                f"- label_files_found: {summary.label_files_found}\n"
                f"- images_with_pose_instances: {summary.images_with_pose_instances}\n"
                f"- invalid_lines_total: {summary.invalid_lines_total}\n\n"
                f"Examples:\n{details}\n\n"
                "Expected each label file to contain YOLO pose lines:\n"
                "  cls x y w h (kpt_count * kpt_dims)\n"
            )
        if summary.label_files_found < summary.images_total:
            logger.warning(
                "DinoKPSEG dataset has %d/%d images missing label files (first few issues: %s)",
                summary.images_total - summary.label_files_found,
                summary.images_total,
                "; ".join(summary.example_issues[:3]),
            )

    device_str = normalize_device(device)
    logger.info("Training DinoKPSEG on %s with device=%s", data_yaml, device_str)

    augment_cfg = augment or DinoKPSEGAugmentConfig(enabled=False)
    schedule_profile_norm = str(schedule_profile or "baseline").strip().lower()
    flat_epoch = max(0, int(flat_epoch))
    aug_start_epoch = max(1, int(aug_start_epoch))
    aug_stop_epoch = max(0, int(aug_stop_epoch))
    no_aug_epoch = max(0, int(no_aug_epoch))
    matcher_change_epoch = max(0, int(matcher_change_epoch))
    iou_order_alpha = float(iou_order_alpha)
    if iou_order_alpha <= 0:
        iou_order_alpha = float(dino_defaults.AGGRESSIVE_S_IOU_ORDER_ALPHA)
    if bool(change_matcher):
        logger.info(
            "DinoKPSEG matcher knobs are recorded as metadata only "
            "(change_matcher=%s, matcher_change_epoch=%d, iou_order_alpha=%.3f). "
            "They are currently not applied by this head-only trainer.",
            bool(change_matcher),
            int(matcher_change_epoch),
            float(iou_order_alpha),
        )
    if augment_cfg.enabled and cache_features:
        logger.info(
            "DinoKPSEG augmentations enabled: training split will skip feature "
            "cache; non-augmented splits still use cache."
        )

    overfit_n = max(0, int(overfit_n))
    if overfit_n > 0:
        rng = random.Random(int(seed or 0))
        rng.shuffle(train_images)
        train_images = train_images[:overfit_n]
        val_images = list(train_images)
        logger.info("DinoKPSEG overfit mode enabled: %d images", len(train_images))

    extractor = build_extractor(
        model_name=model_name,
        short_side=short_side,
        layers=layers,
        device=device_str,
    )

    cache_dir = None
    if cache_features:
        cache_root = Path.home() / ".cache" / "annolid" / "dinokpseg" / "features"
        cache_fingerprint = hashlib.sha1(
            f"{model_name}|{short_side}|{layers}|{feature_merge}".encode(
                "utf-8", errors="ignore"
            )
        ).hexdigest()[:12]
        cache_dir = cache_root / cache_fingerprint
        _ensure_dir(cache_dir)

    train_ds = DinoKPSEGPoseDataset(
        train_images,
        kpt_count=kpt_count,
        kpt_dims=kpt_dims,
        radius_px=radius_px,
        extractor=extractor,
        label_format=str(label_format),
        label_paths=train_label_paths,
        keypoint_names=keypoint_names,
        flip_idx=flip_idx,
        augment=augment_cfg,
        cache_dir=cache_dir,
        mask_type=str(mask_type),
        heatmap_sigma_px=heatmap_sigma_px,
        instance_mode=str(instance_mode),
        bbox_scale=float(bbox_scale),
        return_images=True,
        feature_merge=str(feature_merge),
    )
    val_ds = (
        DinoKPSEGPoseDataset(
            val_images,
            kpt_count=kpt_count,
            kpt_dims=kpt_dims,
            radius_px=radius_px,
            extractor=extractor,
            label_format=str(label_format),
            label_paths=val_label_paths,
            keypoint_names=keypoint_names,
            flip_idx=flip_idx,
            augment=DinoKPSEGAugmentConfig(enabled=False),
            cache_dir=cache_dir,
            mask_type=str(mask_type),
            heatmap_sigma_px=heatmap_sigma_px,
            instance_mode=str(instance_mode),
            bbox_scale=float(bbox_scale),
            return_images=True,
            return_keypoints=True,
            feature_merge=str(feature_merge),
        )
        if val_images
        else None
    )

    sample = train_ds[0]
    feats = sample["feats"]
    in_dim = int(feats.shape[0])
    feature_align_dim = _resolve_feature_align_dim(
        feature_align_dim,
        in_dim=int(in_dim),
        hidden_dim=int(hidden_dim),
    )

    head_type_norm = str(head_type or "conv").strip().lower()
    if head_type_norm not in ("conv", "attn", "hybrid", "multitask"):
        raise ValueError(
            f"Unsupported head_type: {head_type!r} (expected 'conv', 'attn', 'hybrid', or 'multitask')"
        )

    orientation_anchor_idx = (
        infer_orientation_anchor_indices(list(keypoint_names or []))
        if keypoint_names
        else []
    )
    if orientation_anchor_idx:
        logger.info("DinoKPSEG orientation anchors (idx): %s", orientation_anchor_idx)

    core_in_dim = int(in_dim)
    if int(feature_align_dim) > 0 and int(feature_align_dim) != int(in_dim):
        core_in_dim = int(feature_align_dim)

    if head_type_norm == "attn":
        core_head: torch.nn.Module = DinoKPSEGAttentionHead(
            in_dim=int(core_in_dim),
            hidden_dim=hidden_dim,
            num_parts=kpt_count,
            num_heads=int(attn_heads),
            num_layers=int(attn_layers),
            orientation_anchor_idx=orientation_anchor_idx,
        )
    elif head_type_norm == "hybrid":
        core_head = DinoKPSEGHybridHead(
            in_dim=int(core_in_dim),
            hidden_dim=hidden_dim,
            num_parts=kpt_count,
            num_heads=int(attn_heads),
            num_layers=int(attn_layers),
            orientation_anchor_idx=orientation_anchor_idx,
        )
    elif head_type_norm == "multitask":
        core_head = DinoKPSEGMultiTaskHead(
            in_dim=int(core_in_dim), hidden_dim=hidden_dim, num_parts=kpt_count
        )
    else:
        core_head = DinoKPSEGHead(
            in_dim=int(core_in_dim), hidden_dim=hidden_dim, num_parts=kpt_count
        )
    if int(feature_align_dim) > 0 and int(feature_align_dim) != int(in_dim):
        head = DinoKPSEGAlignedHead(
            base_head=core_head,
            in_dim=int(in_dim),
            feature_dim=int(feature_align_dim),
        ).to(device_str)
    else:
        head = core_head.to(device_str)
    ema_enabled = bool(use_ema)
    ema_decay = float(ema_decay)
    if ema_decay <= 0.0 or ema_decay >= 1.0:
        ema_decay = float(dino_defaults.EMA_DECAY)
    ema_head: Optional[torch.nn.Module] = None
    if ema_enabled:
        ema_head = copy.deepcopy(head).to(device_str).eval()
        for p in ema_head.parameters():
            p.requires_grad_(False)

    base_lr = float(lr)
    opt = torch.optim.AdamW(head.parameters(), lr=base_lr)
    scheduler = None
    warmup_epochs = max(0, int(warmup_epochs))
    lr_decay_start_epoch = max(int(warmup_epochs) + 1, int(flat_epoch) + 1)
    lr_final_frac = float(lr_final_frac)
    if lr_final_frac <= 0:
        lr_final_frac = 0.01
    if bool(cos_lr):
        try:
            from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore

            t_max = max(1, int(epochs) - int(lr_decay_start_epoch) + 1)
            scheduler = CosineAnnealingLR(
                opt,
                T_max=int(t_max),
                eta_min=float(base_lr) * float(lr_final_frac),
            )
        except Exception:
            scheduler = None

    weights_dir = output_dir / "weights"
    _ensure_dir(weights_dir)
    csv_path = output_dir / "results.csv"
    args_path = output_dir / "args.yaml"
    tb_dir = output_dir / "tensorboard"
    _ensure_dir(tb_dir)

    meta = DinoKPSEGCheckpointMeta(
        model_name=model_name,
        short_side=int(short_side),
        layers=tuple(int(x) for x in layers),
        num_parts=int(kpt_count),
        radius_px=float(radius_px),
        threshold=float(threshold),
        in_dim=in_dim,
        hidden_dim=int(hidden_dim),
        keypoint_names=list(keypoint_names) if keypoint_names else None,
        flip_idx=list(flip_idx) if flip_idx else None,
        head_type=head_type_norm,
        attn_heads=int(attn_heads),
        attn_layers=int(attn_layers),
        orientation_anchor_idx=orientation_anchor_idx
        if orientation_anchor_idx
        else None,
        feature_merge=str(feature_merge),
        feature_align_dim=int(feature_align_dim),
        multitask=bool(head_type_norm == "multitask"),
    )

    # Best-effort args.yaml compatible with existing YOLO training artifacts.
    args_text = "\n".join(
        [
            "mode: train",
            "task: dino_kpseg",
            f"data: {str(data_yaml)}",
            f"resolved_data: {str(source_yaml)}",
            f"data_format: {str(data_format_norm)}",
            f"model_name: {model_name}",
            f"short_side: {short_side}",
            f"layers: {list(layers)}",
            f"feature_merge: {str(feature_merge)}",
            f"feature_align_dim: {int(feature_align_dim)}",
            f"radius_px: {radius_px}",
            f"radius_schedule: {str(radius_schedule)}",
            f"radius_start_px: {radius_start_px}",
            f"radius_end_px: {radius_end_px}",
            f"mask_type: {str(mask_type)}",
            f"heatmap_sigma_px: {heatmap_sigma_px}",
            f"instance_mode: {str(instance_mode)}",
            f"bbox_scale: {float(bbox_scale)}",
            f"head_type: {head_type_norm}",
            f"attn_heads: {int(attn_heads)}",
            f"attn_layers: {int(attn_layers)}",
            f"hidden_dim: {hidden_dim}",
            f"bce_type: {str(bce_type)}",
            f"focal_alpha: {float(focal_alpha)}",
            f"focal_gamma: {float(focal_gamma)}",
            f"lr0: {lr}",
            f"batch: {int(batch_size)}",
            f"accumulate: {int(accumulate)}",
            f"grad_clip: {float(grad_clip)}",
            f"balanced_bce: {bool(balanced_bce)}",
            f"max_pos_weight: {float(max_pos_weight)}",
            f"cos_lr: {bool(cos_lr)}",
            f"warmup_epochs: {int(warmup_epochs)}",
            f"flat_epoch: {int(flat_epoch)}",
            f"lr_decay_start_epoch: {int(lr_decay_start_epoch)}",
            f"lr_final_frac: {float(lr_final_frac)}",
            f"epochs: {epochs}",
            f"schedule_profile: {schedule_profile_norm}",
            f"threshold: {threshold}",
            f"early_stop_patience: {int(early_stop_patience)}",
            f"early_stop_min_delta: {float(early_stop_min_delta)}",
            f"early_stop_min_epochs: {int(early_stop_min_epochs)}",
            f"best_metric: {str(best_metric)}",
            f"early_stop_metric: {str(early_stop_metric) if early_stop_metric is not None else ''}",
            f"pck_weighted_weights: {','.join(str(float(w)) for w in pck_weighted_weights)}",
            f"coord_warmup_epochs: {int(coord_warmup_epochs)}",
            f"overfit_n: {int(overfit_n)}",
            f"seed: {seed}",
            f"tb_add_graph: {bool(tb_add_graph)}",
            f"tb_projector: {bool(tb_projector)}",
            f"tb_projector_split: {str(tb_projector_split)}",
            f"tb_projector_max_images: {int(tb_projector_max_images)}",
            f"tb_projector_max_patches: {int(tb_projector_max_patches)}",
            f"tb_projector_per_image_per_keypoint: {int(tb_projector_per_image_per_keypoint)}",
            f"tb_projector_pos_threshold: {float(tb_projector_pos_threshold)}",
            f"tb_projector_crop_px: {int(tb_projector_crop_px)}",
            f"tb_projector_sprite_border_px: {int(tb_projector_sprite_border_px)}",
            f"tb_projector_add_negatives: {bool(tb_projector_add_negatives)}",
            f"tb_projector_neg_threshold: {float(tb_projector_neg_threshold)}",
            f"tb_projector_negatives_per_image: {int(tb_projector_negatives_per_image)}",
            f"dice_loss_weight: {float(dice_loss_weight)}",
            f"coord_loss_weight: {float(coord_loss_weight)}",
            f"coord_loss_type: {str(coord_loss_type)}",
            f"obj_loss_weight: {float(obj_loss_weight)}",
            f"box_loss_weight: {float(box_loss_weight)}",
            f"inst_loss_weight: {float(inst_loss_weight)}",
            f"multitask_aux_warmup_epochs: {int(multitask_aux_warmup_epochs)}",
            f"use_ema: {bool(ema_enabled)}",
            f"ema_decay: {float(ema_decay)}",
            f"lr_pair_loss_weight: {float(lr_pair_loss_weight)}",
            f"lr_pair_margin_px: {float(lr_pair_margin_px)}",
            f"lr_side_loss_weight: {float(lr_side_loss_weight)}",
            f"lr_side_loss_margin: {float(lr_side_loss_margin)}",
            f"log_every_steps: {int(log_every_steps)}",
            f"augment: {bool(augment_cfg.enabled)}",
            f"aug_start_epoch: {int(aug_start_epoch)}",
            f"aug_stop_epoch: {int(aug_stop_epoch)}",
            f"no_aug_epoch: {int(no_aug_epoch)}",
            f"change_matcher: {bool(change_matcher)}",
            f"matcher_change_epoch: {int(matcher_change_epoch)}",
            f"iou_order_alpha: {float(iou_order_alpha)}",
            f"hflip: {float(augment_cfg.hflip_prob)}",
            f"degrees: {float(augment_cfg.degrees)}",
            f"translate: {float(augment_cfg.translate)}",
            f"scale: {float(augment_cfg.scale)}",
            f"brightness: {float(augment_cfg.brightness)}",
            f"contrast: {float(augment_cfg.contrast)}",
            f"saturation: {float(augment_cfg.saturation)}",
            "",
        ]
    )
    args_path.write_text(args_text, encoding="utf-8")

    symmetric_pairs = symmetric_pairs_from_flip_idx(flip_idx) if flip_idx else []
    lr_pairs = (
        infer_left_right_pairs(list(keypoint_names or []), flip_idx=flip_idx)
        if keypoint_names and flip_idx
        else []
    )
    if symmetric_pairs:
        logger.info("DinoKPSEG symmetric keypoint pairs: %s", symmetric_pairs)
    if lr_pairs:
        logger.info("DinoKPSEG left/right keypoint pairs: %s", lr_pairs)
    pck_thresholds = (2.0, 4.0, 8.0, 16.0)

    batch_size = max(1, int(batch_size))
    accumulate = max(1, int(accumulate))
    log_interval = max(0, int(log_every_steps))
    patch_size = int(extractor.patch_size)

    def collate_fn(batch: list[dict]) -> dict:
        return _pad_collate(batch, patch_size=patch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if val_ds is not None
        else None
    )

    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    desired_best_metric = _normalize_metric_name(best_metric)
    desired_early_stop_metric = _normalize_metric_name(early_stop_metric)
    if not desired_early_stop_metric or desired_early_stop_metric in {
        "auto",
        "same",
        "same_as_best",
    }:
        desired_early_stop_metric = desired_best_metric

    best_higher_is_better = _metric_higher_is_better(desired_best_metric)
    early_stop_higher_is_better = _metric_higher_is_better(desired_early_stop_metric)

    if val_loader is None:
        if desired_best_metric not in {"train_loss", "loss/train", "train"}:
            logger.warning(
                "No validation split detected; overriding best_metric=%r -> train_loss",
                desired_best_metric,
            )
            desired_best_metric = "train_loss"
            best_higher_is_better = False
        if desired_early_stop_metric not in {"train_loss", "loss/train", "train"}:
            logger.warning(
                "No validation split detected; overriding early_stop_metric=%r -> train_loss",
                desired_early_stop_metric,
            )
            desired_early_stop_metric = "train_loss"
            early_stop_higher_is_better = False

    best_metric_value = -float("inf") if best_higher_is_better else float("inf")

    patience = max(0, int(early_stop_patience))
    min_epochs = max(0, int(early_stop_min_epochs))
    min_delta = float(early_stop_min_delta)
    if min_delta < 0:
        min_delta = 0.0
    early_stop_enabled = patience > 0
    best_metric_for_stop = (
        -float("inf") if early_stop_higher_is_better else float("inf")
    )
    bad_epochs = 0

    tb_writer = SummaryWriter(str(tb_dir))
    try:
        tb_writer.add_text("config/data_yaml", str(data_yaml), 0)
        if source_yaml != data_yaml:
            tb_writer.add_text("config/data_yaml_resolved", str(source_yaml), 0)
        tb_writer.add_text("config/model_name", str(model_name), 0)
        tb_writer.add_text("config/layers", str(list(layers)), 0)
        tb_writer.add_text("config/feature_merge", str(feature_merge), 0)
        tb_writer.add_text("config/feature_align_dim", str(int(feature_align_dim)), 0)
        tb_writer.add_text("config/device", str(device_str), 0)
        tb_writer.add_text("config/short_side", str(short_side), 0)
        tb_writer.add_text("config/patch_size", str(int(extractor.patch_size)), 0)
        tb_writer.add_text("config/mask_type", str(mask_type), 0)
        tb_writer.add_text("config/bce_type", str(bce_type), 0)
        tb_writer.add_text("config/focal_alpha", str(float(focal_alpha)), 0)
        tb_writer.add_text("config/focal_gamma", str(float(focal_gamma)), 0)
        tb_writer.add_text(
            "config/coord_warmup_epochs", str(int(coord_warmup_epochs)), 0
        )
        tb_writer.add_text("config/radius_schedule", str(radius_schedule), 0)
        tb_writer.add_text("config/radius_start_px", str(radius_start_px), 0)
        tb_writer.add_text("config/radius_end_px", str(radius_end_px), 0)
        tb_writer.add_text("config/schedule_profile", str(schedule_profile_norm), 0)
        tb_writer.add_text("config/flat_epoch", str(int(flat_epoch)), 0)
        tb_writer.add_text(
            "config/lr_decay_start_epoch", str(int(lr_decay_start_epoch)), 0
        )
        tb_writer.add_text("config/aug_start_epoch", str(int(aug_start_epoch)), 0)
        tb_writer.add_text("config/aug_stop_epoch", str(int(aug_stop_epoch)), 0)
        tb_writer.add_text("config/no_aug_epoch", str(int(no_aug_epoch)), 0)
        tb_writer.add_text("config/change_matcher", str(bool(change_matcher)), 0)
        tb_writer.add_text(
            "config/matcher_change_epoch", str(int(matcher_change_epoch)), 0
        )
        tb_writer.add_text("config/iou_order_alpha", str(float(iou_order_alpha)), 0)
        tb_writer.add_text("config/overfit_n", str(int(overfit_n)), 0)
        if seed is not None:
            tb_writer.add_text("config/seed", str(int(seed)), 0)
        tb_writer.add_text("config/heatmap_sigma_px", str(heatmap_sigma_px), 0)
        tb_writer.add_text("config/instance_mode", str(instance_mode), 0)
        tb_writer.add_text("config/bbox_scale", str(float(bbox_scale)), 0)
        tb_writer.add_text(
            "config/pck_thresholds_px",
            ",".join(str(float(x)) for x in pck_thresholds),
            0,
        )
        tb_writer.add_text("config/early_stop_patience", str(patience), 0)
        tb_writer.add_text("config/early_stop_min_delta", str(min_delta), 0)
        tb_writer.add_text("config/early_stop_min_epochs", str(min_epochs), 0)
        tb_writer.add_text("config/best_metric", str(desired_best_metric), 0)
        tb_writer.add_text(
            "config/best_metric_mode", "max" if best_higher_is_better else "min", 0
        )
        tb_writer.add_text(
            "config/pck_weighted_weights",
            ",".join(str(float(w)) for w in pck_weighted_weights),
            0,
        )
        tb_writer.add_text(
            "config/early_stop_metric", str(desired_early_stop_metric), 0
        )
        tb_writer.add_text(
            "config/early_stop_mode",
            "max" if early_stop_higher_is_better else "min",
            0,
        )
        tb_writer.add_text("config/tb_add_graph", str(bool(tb_add_graph)), 0)
        tb_writer.add_text("config/tb_projector", str(bool(tb_projector)), 0)
        tb_writer.add_text("config/tb_projector_split", str(tb_projector_split), 0)
        tb_writer.add_text("config/head_type", str(head_type_norm), 0)
        tb_writer.add_text("config/attn_heads", str(int(attn_heads)), 0)
        tb_writer.add_text("config/attn_layers", str(int(attn_layers)), 0)
        tb_writer.add_text(
            "config/orientation_anchor_idx",
            str(orientation_anchor_idx),
            0,
        )
        tb_writer.add_text("config/dice_loss_weight", str(float(dice_loss_weight)), 0)
        tb_writer.add_text("config/coord_loss_weight", str(float(coord_loss_weight)), 0)
        tb_writer.add_text("config/coord_loss_type", str(coord_loss_type), 0)
        tb_writer.add_text("config/obj_loss_weight", str(float(obj_loss_weight)), 0)
        tb_writer.add_text("config/box_loss_weight", str(float(box_loss_weight)), 0)
        tb_writer.add_text("config/inst_loss_weight", str(float(inst_loss_weight)), 0)
        tb_writer.add_text(
            "config/multitask_aux_warmup_epochs",
            str(int(multitask_aux_warmup_epochs)),
            0,
        )
        tb_writer.add_text("config/use_ema", str(bool(ema_enabled)), 0)
        tb_writer.add_text("config/ema_decay", str(float(ema_decay)), 0)
        tb_writer.add_text(
            "config/lr_pair_loss_weight", str(float(lr_pair_loss_weight)), 0
        )
        tb_writer.add_text("config/lr_pair_margin_px", str(float(lr_pair_margin_px)), 0)
        tb_writer.add_text(
            "config/lr_side_loss_weight", str(float(lr_side_loss_weight)), 0
        )
        tb_writer.add_text(
            "config/lr_side_loss_margin", str(float(lr_side_loss_margin)), 0
        )
        tb_writer.add_text("model/architecture", f"```\n{head}\n```", 0)
        try:
            total_params = int(sum(p.numel() for p in head.parameters()))
            trainable_params = int(
                sum(p.numel() for p in head.parameters() if p.requires_grad)
            )
            tb_writer.add_scalar("model/params_total", total_params, 0)
            tb_writer.add_scalar("model/params_trainable", trainable_params, 0)
        except Exception:
            pass
        # Optional: export the computation graph (can be expensive / crash-prone on some builds).
        if bool(tb_add_graph) or os.environ.get(
            "ANNOLID_TB_ADD_GRAPH", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                cpu_head = copy.deepcopy(head).cpu().eval()
                dummy = torch.zeros((1, int(in_dim), 2, 2), dtype=torch.float32)
                tb_writer.add_graph(cpu_head, dummy)
                tb_writer.add_text(
                    "model/graph", "Graph exported via SummaryWriter.add_graph()", 0
                )
            except Exception as exc:
                tb_writer.add_text(
                    "model/graph_error", f"{type(exc).__name__}: {exc}", 0
                )

        _log_example_images(
            tb_writer,
            tag="samples/train_images",
            image_paths=list(train_images),
            short_side=int(short_side),
            patch_size=int(extractor.patch_size),
        )
        _log_example_images(
            tb_writer,
            tag="samples/val_images",
            image_paths=list(val_images),
            short_side=int(short_side),
            patch_size=int(extractor.patch_size),
        )

        try:
            from annolid.segmentation.dino_kpseg.dataset_tools import (
                audit_yolo_pose_dataset,
                log_dataset_health,
            )

            if label_format != "yolo":
                raise RuntimeError(
                    "Dataset audit currently requires a YOLO pose dataset."
                )

            audit_report = audit_yolo_pose_dataset(
                source_yaml,
                split="both",
                instance_mode=str(instance_mode),
                bbox_scale=float(bbox_scale),
            )
            audit_path = output_dir / "dataset_audit.json"
            audit_path.write_text(json.dumps(audit_report, indent=2), encoding="utf-8")
            tb_writer.add_text(
                "dataset/audit_summary",
                json.dumps(audit_report.get("splits", {}), indent=2),
                0,
            )
            log_dataset_health(
                tb_writer=tb_writer,
                report=audit_report,
                split_name="train",
                image_paths=list(train_images),
                kpt_count=int(kpt_count),
                kpt_dims=int(kpt_dims),
                keypoint_names=list(keypoint_names or []),
                max_images=12,
                seed=(int(augment_cfg.seed) if augment_cfg.seed is not None else 0),
                tag_prefix="dataset",
            )
            if val_images:
                log_dataset_health(
                    tb_writer=tb_writer,
                    report=audit_report,
                    split_name="val",
                    image_paths=list(val_images),
                    kpt_count=int(kpt_count),
                    kpt_dims=int(kpt_dims),
                    keypoint_names=list(keypoint_names or []),
                    max_images=12,
                    seed=(int(augment_cfg.seed) if augment_cfg.seed is not None else 0),
                    tag_prefix="dataset",
                )
        except Exception as exc:
            tb_writer.add_text(
                "dataset/audit_error",
                f"{type(exc).__name__}: {exc}",
                0,
            )

        if bool(tb_projector):
            try:
                from annolid.segmentation.dino_kpseg.tensorboard_embeddings import (
                    add_dino_kpseg_projector_embeddings,
                )

                kpt_names = (
                    list(keypoint_names)
                    if keypoint_names
                    else [f"kpt_{i}" for i in range(int(kpt_count))]
                )
                split_pref = str(tb_projector_split or "val").strip().lower()
                splits: list[
                    tuple[str, list[Path], Optional[List[Optional[Path]]]]
                ] = []
                if split_pref == "both":
                    splits = [("train", list(train_images), train_label_paths)]
                    if val_ds is not None:
                        splits.append(("val", list(val_images), val_label_paths))
                elif split_pref == "train":
                    splits = [("train", list(train_images), train_label_paths)]
                else:
                    splits = [
                        (
                            "val",
                            list(val_images)
                            if val_ds is not None
                            else list(train_images),
                            val_label_paths
                            if val_ds is not None
                            else train_label_paths,
                        )
                    ]

                for split_name, paths, split_label_paths in splits:
                    if not paths:
                        continue
                    # Use a deterministic, non-augmented dataset for embeddings even
                    # when training uses augmentations; this keeps the projector view
                    # stable and allows feature caching when enabled.
                    ds_obj = DinoKPSEGPoseDataset(
                        list(paths),
                        kpt_count=kpt_count,
                        kpt_dims=kpt_dims,
                        radius_px=radius_px,
                        extractor=extractor,
                        label_format=str(label_format),
                        label_paths=split_label_paths,
                        keypoint_names=keypoint_names,
                        flip_idx=flip_idx,
                        augment=DinoKPSEGAugmentConfig(enabled=False),
                        cache_dir=cache_dir,
                        mask_type=str(mask_type),
                        heatmap_sigma_px=heatmap_sigma_px,
                        instance_mode=str(instance_mode),
                        bbox_scale=float(bbox_scale),
                        return_images=True,
                        feature_merge=str(feature_merge),
                    )
                    add_dino_kpseg_projector_embeddings(
                        tb_writer,
                        log_dir=tb_dir,
                        split=str(split_name),
                        ds=ds_obj,
                        keypoint_names=kpt_names,
                        max_images=int(tb_projector_max_images),
                        max_patches=int(tb_projector_max_patches),
                        per_image_per_keypoint=int(tb_projector_per_image_per_keypoint),
                        pos_threshold=float(tb_projector_pos_threshold),
                        add_negatives=bool(tb_projector_add_negatives),
                        neg_threshold=float(tb_projector_neg_threshold),
                        negatives_per_image=int(tb_projector_negatives_per_image),
                        crop_px=int(tb_projector_crop_px),
                        sprite_border_px=int(tb_projector_sprite_border_px),
                        seed=(
                            int(augment_cfg.seed) if augment_cfg.seed is not None else 0
                        ),
                        tag=f"dino_kpseg/patch_embeddings/{split_name}",
                        predict_probs_patch=None,
                        write_csv=True,
                    )
            except Exception as exc:
                tb_writer.add_text(
                    "tensorboard/projector_error",
                    f"{type(exc).__name__}: {exc}",
                    0,
                )

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "metrics/precision(P)",
                    "metrics/recall(P)",
                    "metrics/mAP50(P)",
                    "metrics/mAP50-95(P)",
                    "metrics/pck@2px",
                    "metrics/pck@4px",
                    "metrics/pck@8px",
                    "metrics/pck@16px",
                    "metrics/swap_rate",
                    "seconds",
                ],
            )
            writer.writeheader()
            prev_aug_enabled: Optional[bool] = None

            for epoch in range(1, int(epochs) + 1):
                train_steps_total = int(len(train_loader))
                val_steps_total = int(len(val_loader)) if val_loader is not None else 0
                radius_epoch = float(radius_px)
                schedule = str(radius_schedule or "none").strip().lower()
                if schedule != "none":
                    start_px = (
                        float(radius_start_px)
                        if radius_start_px is not None
                        else float(radius_px)
                    )
                    end_px = (
                        float(radius_end_px)
                        if radius_end_px is not None
                        else float(radius_px)
                    )
                    if int(epochs) > 1:
                        t = float(epoch - 1) / float(int(epochs) - 1)
                    else:
                        t = 1.0
                    if schedule == "linear":
                        radius_epoch = start_px + (end_px - start_px) * float(t)
                    else:
                        radius_epoch = float(radius_px)
                    train_ds.radius_px = float(radius_epoch)
                    if val_ds is not None:
                        val_ds.radius_px = float(radius_epoch)
                tb_writer.add_scalar("train/radius_px", float(radius_epoch), epoch)

                coord_weight = float(coord_loss_weight)
                if int(coord_warmup_epochs) > 0:
                    coord_weight = float(coord_loss_weight) * min(
                        1.0, float(epoch) / float(coord_warmup_epochs)
                    )
                tb_writer.add_scalar(
                    "train/coord_loss_weight", float(coord_weight), epoch
                )
                aux_scale = 1.0
                if int(multitask_aux_warmup_epochs) > 0:
                    aux_scale = min(
                        1.0,
                        float(epoch) / float(max(1, int(multitask_aux_warmup_epochs))),
                    )
                tb_writer.add_scalar("train/aux_loss_scale", float(aux_scale), epoch)

                epoch_aug_enabled = _is_epoch_augment_enabled(
                    epoch=int(epoch),
                    epochs=int(epochs),
                    augment_enabled=bool(augment_cfg.enabled),
                    aug_start_epoch=int(aug_start_epoch),
                    aug_stop_epoch=int(aug_stop_epoch),
                    no_aug_epoch=int(no_aug_epoch),
                )
                if prev_aug_enabled is None or bool(prev_aug_enabled) != bool(
                    epoch_aug_enabled
                ):
                    logger.info(
                        "Epoch %d/%d schedule phase: augment=%s lr_decay_started=%s",
                        epoch,
                        int(epochs),
                        "on" if epoch_aug_enabled else "off",
                        "yes" if int(epoch) >= int(lr_decay_start_epoch) else "no",
                    )
                prev_aug_enabled = bool(epoch_aug_enabled)
                if bool(train_ds.augment.enabled) != bool(epoch_aug_enabled):
                    train_ds.augment = DinoKPSEGAugmentConfig(
                        enabled=bool(epoch_aug_enabled),
                        hflip_prob=float(augment_cfg.hflip_prob),
                        degrees=float(augment_cfg.degrees),
                        translate=float(augment_cfg.translate),
                        scale=float(augment_cfg.scale),
                        brightness=float(augment_cfg.brightness),
                        contrast=float(augment_cfg.contrast),
                        saturation=float(augment_cfg.saturation),
                        seed=augment_cfg.seed,
                    )
                    if augment_cfg.seed is not None:
                        train_ds.rng = np.random.default_rng(
                            int(augment_cfg.seed) + int(epoch)
                        )
                tb_writer.add_scalar(
                    "train/augment_enabled_epoch",
                    1.0 if bool(epoch_aug_enabled) else 0.0,
                    epoch,
                )

                head.train()
                if warmup_epochs > 0 and int(epoch) <= int(warmup_epochs):
                    warm_lr = (
                        float(base_lr) * float(epoch) / float(max(1, warmup_epochs))
                    )
                    for pg in opt.param_groups:
                        pg["lr"] = float(warm_lr)
                t0 = time.time()
                train_loss = 0.0
                train_loss_total = 0.0
                n_train = 0
                batch_dice_sum = 0.0
                batch_dice_count = 0
                dice_loss_sum = 0.0
                dice_terms_count = 0
                coord_loss_sum = 0.0
                coord_terms_count = 0
                pair_overlap_sum = 0.0
                pair_margin_sum = 0.0
                pair_terms_count = 0
                side_loss_sum = 0.0
                side_terms_count = 0
                grad_norm_sum = 0.0
                grad_norm_steps = 0
                opt.zero_grad(set_to_none=True)
                for batch in train_loader:
                    feats = batch["feats"].to(device_str, non_blocking=True)  # BCHW
                    masks = batch["masks"].to(device_str, non_blocking=True)  # BKHW
                    coords = batch.get("coords")
                    coord_mask = batch.get("coord_mask")
                    valid_mask = batch.get("valid_mask")
                    key_padding_mask = batch.get("key_padding_mask")
                    if not isinstance(coords, torch.Tensor):
                        coords = torch.zeros(
                            (int(masks.shape[0]), int(masks.shape[1]), 2),
                            dtype=torch.float32,
                            device=device_str,
                        )
                    else:
                        coords = coords.to(device_str, non_blocking=True)
                    if not isinstance(coord_mask, torch.Tensor):
                        coord_mask = torch.zeros(
                            (int(masks.shape[0]), int(masks.shape[1])),
                            dtype=torch.float32,
                            device=device_str,
                        )
                    else:
                        coord_mask = coord_mask.to(device_str, non_blocking=True)
                    if not isinstance(valid_mask, torch.Tensor):
                        valid_mask = torch.ones(
                            (
                                int(masks.shape[0]),
                                1,
                                int(masks.shape[2]),
                                int(masks.shape[3]),
                            ),
                            dtype=torch.bool,
                            device=device_str,
                        )
                    else:
                        valid_mask = valid_mask.to(device_str, non_blocking=True)
                    if isinstance(key_padding_mask, torch.Tensor):
                        key_padding_mask = key_padding_mask.to(
                            device_str, non_blocking=True
                        )
                    else:
                        key_padding_mask = None

                    pred_all = None
                    if hasattr(head, "forward_all"):
                        try:
                            pred_all = head.forward_all(
                                feats, key_padding_mask=key_padding_mask
                            )
                        except TypeError:
                            pred_all = head.forward_all(feats)
                    if isinstance(pred_all, dict) and isinstance(
                        pred_all.get("kpt_logits"), torch.Tensor
                    ):
                        logits = pred_all["kpt_logits"]
                    else:
                        try:
                            logits = head(feats, key_padding_mask=key_padding_mask)
                        except TypeError:
                            logits = head(feats)
                    logits = logits.masked_fill(~valid_mask, -20.0)

                    loss_base = _masked_bce_with_logits(
                        logits,
                        masks,
                        valid_mask,
                        balanced=bool(balanced_bce),
                        max_pos_weight=float(max_pos_weight),
                    )
                    bce_mode = str(bce_type or "bce").strip().lower()
                    if bce_mode == "focal":
                        loss_base = _masked_focal_bce_with_logits(
                            logits,
                            masks,
                            valid_mask,
                            alpha=float(focal_alpha),
                            gamma=float(focal_gamma),
                        )
                    loss = loss_base
                    probs = torch.sigmoid(logits)

                    dice_loss = None
                    if float(dice_loss_weight) > 0.0:
                        dice_loss = _dice_loss_masked(probs, masks, valid_mask)
                        loss = loss + float(dice_loss_weight) * dice_loss

                    coord_loss = None
                    if (
                        float(coord_weight) > 0.0
                        and coords.numel() > 0
                        and coord_mask.numel() > 0
                    ):
                        pred_xy = _soft_argmax_coords_batched(
                            probs, patch_size=int(extractor.patch_size)
                        )
                        coord_loss = _coord_loss(
                            pred_xy.reshape(-1, 2),
                            coords.reshape(-1, 2),
                            coord_mask.reshape(-1),
                            mode=coord_loss_type,
                        )
                        loss = loss + float(coord_weight) * coord_loss

                    if isinstance(pred_all, dict):
                        inst_logits = pred_all.get("inst_logits")
                        obj_logits = pred_all.get("obj_logits")
                        box_logits = pred_all.get("box_logits")
                        instance_target = masks.max(dim=1, keepdim=True).values
                        if (
                            isinstance(inst_logits, torch.Tensor)
                            and float(inst_loss_weight) > 0.0
                        ):
                            inst_loss = _masked_bce_with_logits(
                                inst_logits,
                                instance_target,
                                valid_mask,
                                balanced=False,
                                max_pos_weight=1.0,
                            )
                            loss = (
                                loss
                                + float(inst_loss_weight) * float(aux_scale) * inst_loss
                            )
                        if (
                            isinstance(obj_logits, torch.Tensor)
                            and float(obj_loss_weight) > 0.0
                        ):
                            obj_target = (
                                (instance_target > 0.5)
                                .any(dim=(2, 3), keepdim=True)
                                .to(dtype=torch.float32)
                            )
                            obj_target_map = obj_target.expand_as(obj_logits)
                            obj_loss = _masked_bce_with_logits(
                                obj_logits,
                                obj_target_map,
                                valid_mask,
                                balanced=False,
                                max_pos_weight=1.0,
                            )
                            loss = (
                                loss
                                + float(obj_loss_weight) * float(aux_scale) * obj_loss
                            )
                        if (
                            isinstance(box_logits, torch.Tensor)
                            and float(box_loss_weight) > 0.0
                        ):
                            box_tgt, has_box = _instance_box_targets_from_masks(
                                instance_target, valid_mask_b1hw=valid_mask
                            )
                            box_pred = torch.sigmoid(box_logits).mean(dim=(2, 3))
                            if bool(has_box.any()):
                                box_l1 = torch.abs(
                                    box_pred[has_box] - box_tgt[has_box]
                                ).mean()
                                loss = (
                                    loss
                                    + float(box_loss_weight) * float(aux_scale) * box_l1
                                )

                    pair_overlap = None
                    pair_margin = None
                    side_loss = None
                    if (
                        float(lr_pair_loss_weight) > 0.0
                        and symmetric_pairs
                        and int(logits.shape[1]) == int(kpt_count)
                    ):
                        bsz, k, h_p, w_p = probs.shape
                        overlap_acc = torch.zeros(
                            (), dtype=probs.dtype, device=probs.device
                        )
                        margin_acc = torch.zeros(
                            (), dtype=probs.dtype, device=probs.device
                        )
                        patch = float(extractor.patch_size)
                        xs = (
                            torch.arange(w_p, device=probs.device, dtype=probs.dtype)
                            + 0.5
                        ) * patch
                        ys = (
                            torch.arange(h_p, device=probs.device, dtype=probs.dtype)
                            + 0.5
                        ) * patch
                        for b_i in range(int(bsz)):
                            probs_flat = probs[b_i].reshape(k, -1)
                            norm = probs_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
                            dist = probs_flat / norm  # [K, HW]
                            for i, j in symmetric_pairs:
                                overlap_acc = (
                                    overlap_acc + (dist[int(i)] * dist[int(j)]).sum()
                                )
                                if float(lr_pair_margin_px) > 0.0:
                                    pi = dist[int(i)].view(h_p, w_p)
                                    pj = dist[int(j)].view(h_p, w_p)
                                    mux_i = (pi.sum(dim=0) * xs).sum()
                                    muy_i = (pi.sum(dim=1) * ys).sum()
                                    mux_j = (pj.sum(dim=0) * xs).sum()
                                    muy_j = (pj.sum(dim=1) * ys).sum()
                                    d = torch.sqrt(
                                        (mux_i - mux_j) ** 2
                                        + (muy_i - muy_j) ** 2
                                        + 1e-8
                                    )
                                    margin_acc = margin_acc + F.relu(
                                        float(lr_pair_margin_px) - d
                                    )

                        denom_pairs = float(max(1, len(symmetric_pairs))) * float(
                            max(1, int(bsz))
                        )
                        pair_overlap = overlap_acc / denom_pairs
                        pair_margin = margin_acc / denom_pairs
                        loss = loss + float(lr_pair_loss_weight) * (
                            pair_overlap + pair_margin
                        )

                    if (
                        float(lr_side_loss_weight) > 0.0
                        and lr_pairs
                        and len(orientation_anchor_idx) >= 2
                        and int(logits.shape[1]) == int(kpt_count)
                    ):
                        bsz, k, h_p, w_p = probs.shape
                        xs = (
                            torch.arange(w_p, device=probs.device, dtype=probs.dtype)
                            + 0.5
                        )
                        ys = (
                            torch.arange(h_p, device=probs.device, dtype=probs.dtype)
                            + 0.5
                        )
                        margin = float(lr_side_loss_margin)
                        loss_acc = torch.zeros(
                            (), dtype=probs.dtype, device=probs.device
                        )
                        a0 = int(orientation_anchor_idx[0])
                        a1 = int(orientation_anchor_idx[1])
                        for b_i in range(int(bsz)):
                            probs_flat = probs[b_i].reshape(k, -1)
                            norm = probs_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
                            dist = (probs_flat / norm).view(k, h_p, w_p)
                            px = (dist.sum(dim=1) * xs[None, :]).sum(dim=1)
                            py = (dist.sum(dim=2) * ys[None, :]).sum(dim=1)
                            ax = px[a1] - px[a0]
                            ay = py[a1] - py[a0]
                            an = torch.sqrt(ax * ax + ay * ay + 1e-8)
                            for li, ri in lr_pairs:
                                li = int(li)
                                ri = int(ri)
                                vx_l = px[li] - px[a0]
                                vy_l = py[li] - py[a0]
                                vx_r = px[ri] - px[a0]
                                vy_r = py[ri] - py[a0]
                                vn_l = torch.sqrt(vx_l * vx_l + vy_l * vy_l + 1e-8)
                                vn_r = torch.sqrt(vx_r * vx_r + vy_r * vy_r + 1e-8)
                                s_l = (ax * vy_l - ay * vx_l) / (an * vn_l + 1e-8)
                                s_r = (ax * vy_r - ay * vx_r) / (an * vn_r + 1e-8)
                                loss_acc = loss_acc + F.relu(s_l * s_r + margin)
                        denom_lr = float(max(1, len(lr_pairs))) * float(
                            max(1, int(bsz))
                        )
                        side_loss = loss_acc / denom_lr
                        loss = loss + float(lr_side_loss_weight) * side_loss

                    if not torch.isfinite(loss.detach()).item():
                        logger.warning(
                            "Skipping non-finite loss at epoch=%d step=%d",
                            int(epoch),
                            int(n_train + 1),
                        )
                        opt.zero_grad(set_to_none=True)
                        continue
                    (loss / float(accumulate)).backward()
                    n_train += 1
                    if (n_train % int(accumulate)) == 0:
                        if float(grad_clip) > 0.0:
                            try:
                                gn = torch.nn.utils.clip_grad_norm_(
                                    head.parameters(), max_norm=float(grad_clip)
                                )
                                grad_norm_sum += float(gn.detach().cpu().item())
                                grad_norm_steps += 1
                            except Exception:
                                pass
                        opt.step()
                        if ema_head is not None:
                            _ema_update_(ema_head, head, decay=float(ema_decay))
                        opt.zero_grad(set_to_none=True)

                    train_loss += float(loss_base.detach().cpu().item())
                    train_loss_total += float(loss.detach().cpu().item())
                    if dice_loss is not None:
                        dice_loss_sum += float(dice_loss.detach().cpu().item())
                        dice_terms_count += 1
                    if coord_loss is not None:
                        coord_loss_sum += float(coord_loss.detach().cpu().item())
                        coord_terms_count += 1
                    if pair_overlap is not None:
                        pair_overlap_sum += float(pair_overlap.detach().cpu().item())
                        pair_margin_sum += (
                            float(pair_margin.detach().cpu().item())
                            if pair_margin is not None
                            else 0.0
                        )
                        pair_terms_count += 1
                    if side_loss is not None:
                        side_loss_sum += float(side_loss.detach().cpu().item())
                        side_terms_count += 1
                    # Flush leftover grads at epoch end.

                    # Lightweight "YOLO-like" qualitative logging: visualize GT vs prediction
                    # for the first batch of each epoch, plus basic mask metrics.
                    if n_train == 1:
                        with torch.no_grad():
                            probs0 = torch.sigmoid(logits[0]).detach()
                            masks0 = masks[0].detach()
                            preds = (probs0 >= 0.5).to(dtype=masks0.dtype)
                            inter = (preds * masks0).sum(dim=(1, 2))
                            denom = preds.sum(dim=(1, 2)) + masks0.sum(dim=(1, 2))
                            dice = (2.0 * inter) / torch.clamp(denom, min=1e-8)
                            tb_writer.add_scalar(
                                "metrics/dice_mean_train_batch0",
                                float(dice.mean().item()),
                                epoch,
                            )
                            tb_writer.add_scalar(
                                "metrics/dice_min_train_batch0",
                                float(dice.min().item()),
                                epoch,
                            )
                            tb_writer.add_scalar(
                                "metrics/dice_max_train_batch0",
                                float(dice.max().item()),
                                epoch,
                            )
                            batch_dice_sum += float(dice.mean().item())
                            batch_dice_count += 1

                            # Histograms (bounded to first batch per epoch)
                            try:
                                tb_writer.add_histogram(
                                    "debug/logits",
                                    logits[0].detach().float().cpu(),
                                    epoch,
                                )
                                tb_writer.add_histogram(
                                    "debug/probs",
                                    probs[0].detach().float().cpu(),
                                    epoch,
                                )
                            except Exception:
                                pass

                            # Images: keypoint heatmaps at patch resolution.
                            k = int(min(8, int(masks0.shape[0])))
                            try:
                                imgs = (
                                    torch.cat([masks0[:k], probs0[:k]], dim=0)
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                imgs = imgs.unsqueeze(1).clamp(0.0, 1.0)  # (2k,1,H,W)
                                grid = _grid_images(imgs, nrow=k, pad=2, pad_value=0.0)
                                _tb_add_image(
                                    tb_writer, "qual/gt_then_pred_batch0", grid, epoch
                                )
                            except Exception:
                                pass

                            # Example input image + overlays (if available from the dataset).
                            image_batch = (
                                batch.get("image") if isinstance(batch, dict) else None
                            )
                            if (
                                isinstance(image_batch, torch.Tensor)
                                and image_batch.ndim == 4
                            ):
                                img0 = (
                                    image_batch[0]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .clamp(0.0, 1.0)
                                )  # 3HW
                                _tb_add_image(
                                    tb_writer, "qual/image_batch0", img0, epoch
                                )

                                prob_mean = (
                                    probs0.detach().float().cpu().mean(dim=0)
                                )  # HW
                                pmin = float(prob_mean.min().item())
                                pmax = float(prob_mean.max().item())
                                prob_norm = (prob_mean - pmin) / max(pmax - pmin, 1e-6)

                                gt_sum = masks0.detach().float().cpu().sum(dim=0)
                                gmin = float(gt_sum.min().item())
                                gmax = float(gt_sum.max().item())
                                gt_norm = (gt_sum - gmin) / max(gmax - gmin, 1e-6)

                                _tb_add_image(
                                    tb_writer,
                                    "qual/pred_mean_batch0",
                                    prob_norm.unsqueeze(0),
                                    epoch,
                                )
                                _tb_add_image(
                                    tb_writer,
                                    "qual/gt_sum_batch0",
                                    gt_norm.unsqueeze(0),
                                    epoch,
                                )

                                # Upsample patch-grid maps to image resolution for overlays.
                                try:
                                    target_hw = (int(img0.shape[1]), int(img0.shape[2]))
                                    prob_up = F.interpolate(
                                        prob_norm.unsqueeze(0).unsqueeze(0),
                                        size=target_hw,
                                        mode="bilinear",
                                        align_corners=False,
                                    )[0, 0].clamp(0.0, 1.0)
                                    gt_up = F.interpolate(
                                        gt_norm.unsqueeze(0).unsqueeze(0),
                                        size=target_hw,
                                        mode="bilinear",
                                        align_corners=False,
                                    )[0, 0].clamp(0.0, 1.0)
                                    _tb_add_image(
                                        tb_writer,
                                        "qual/pred_mean_up_batch0",
                                        prob_up.unsqueeze(0),
                                        epoch,
                                    )
                                    _tb_add_image(
                                        tb_writer,
                                        "qual/gt_sum_up_batch0",
                                        gt_up.unsqueeze(0),
                                        epoch,
                                    )
                                except Exception:
                                    prob_up = prob_norm

                                overlay = img0.clone()
                                alpha = (0.55 * prob_up).clamp(0.0, 0.55)  # HW
                                overlay[0] = overlay[0] * (1.0 - alpha) + alpha * 1.0
                                overlay[1] = overlay[1] * (1.0 - alpha)
                                overlay[2] = overlay[2] * (1.0 - alpha)
                                _tb_add_image(
                                    tb_writer,
                                    "qual/overlay_pred_mean_batch0",
                                    overlay.clamp(0.0, 1.0),
                                    epoch,
                                )
                    if log_interval > 0 and (
                        (n_train % int(log_interval)) == 0
                        or int(n_train) == int(train_steps_total)
                    ):
                        elapsed_train = max(1e-6, float(time.time() - t0))
                        steps_per_s = float(n_train) / elapsed_train
                        logger.info(
                            "Epoch %d/%d - train step %d/%d (%.1f%%) "
                            "base_loss=%.6f total_loss=%.6f speed=%.2f steps/s",
                            epoch,
                            epochs,
                            n_train,
                            train_steps_total,
                            (
                                100.0
                                * float(n_train)
                                / max(1.0, float(train_steps_total))
                            ),
                            (train_loss / max(1, n_train)),
                            (train_loss_total / max(1, n_train)),
                            steps_per_s,
                        )

                if int(n_train) % int(accumulate) != 0:
                    if float(grad_clip) > 0.0:
                        try:
                            gn = torch.nn.utils.clip_grad_norm_(
                                head.parameters(), max_norm=float(grad_clip)
                            )
                            grad_norm_sum += float(gn.detach().cpu().item())
                            grad_norm_steps += 1
                        except Exception:
                            pass
                    opt.step()
                    if ema_head is not None:
                        _ema_update_(ema_head, head, decay=float(ema_decay))
                    opt.zero_grad(set_to_none=True)

                train_loss /= max(1, n_train)
                train_loss_total /= max(1, n_train)
                if grad_norm_steps:
                    tb_writer.add_scalar(
                        "train/grad_norm",
                        grad_norm_sum / max(1, grad_norm_steps),
                        epoch,
                    )
                try:
                    tb_writer.add_scalar(
                        "train/lr", float(opt.param_groups[0].get("lr", 0.0)), epoch
                    )
                except Exception:
                    pass

                val_loss = None
                precision_pose_mean = None
                recall_pose_mean = None
                map50 = None
                map5095 = None
                pck_summary = None
                swap_rate = None
                if val_loader is not None:
                    eval_head = ema_head if ema_head is not None else head
                    eval_head.eval()
                    losses = []
                    tp = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    fp = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    fn = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    peak_dist_sum = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    peak_dist_count = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    coord_err_sum = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    coord_err_count = torch.zeros(
                        int(kpt_count), device=device_str, dtype=torch.float32
                    )
                    det_scores: List[List[float]] = [[] for _ in range(int(kpt_count))]
                    det_oks: List[List[float]] = [[] for _ in range(int(kpt_count))]
                    gt_counts = [0 for _ in range(int(kpt_count))]
                    pck_acc = DinoKPSEGEvalAccumulator(
                        kpt_count=int(kpt_count),
                        thresholds_px=pck_thresholds,
                        keypoint_names=list(keypoint_names) if keypoint_names else None,
                    )
                    error_samples: List[Tuple[float, torch.Tensor]] = []
                    pred_overlays: List[torch.Tensor] = []
                    error_max = 4
                    pred_max = 4
                    with torch.no_grad():
                        n_val = 0
                        for batch in val_loader:
                            n_val += 1
                            feats = batch["feats"].to(device_str, non_blocking=True)
                            masks = batch["masks"].to(device_str, non_blocking=True)
                            coords = batch.get("coords")
                            coord_mask = batch.get("coord_mask")
                            valid_mask = batch.get("valid_mask")
                            key_padding_mask = batch.get("key_padding_mask")
                            if not isinstance(coords, torch.Tensor):
                                coords = torch.zeros(
                                    (int(masks.shape[0]), int(masks.shape[1]), 2),
                                    dtype=torch.float32,
                                    device=device_str,
                                )
                            else:
                                coords = coords.to(device_str, non_blocking=True)
                            if not isinstance(coord_mask, torch.Tensor):
                                coord_mask = torch.zeros(
                                    (int(masks.shape[0]), int(masks.shape[1])),
                                    dtype=torch.float32,
                                    device=device_str,
                                )
                            else:
                                coord_mask = coord_mask.to(
                                    device_str, non_blocking=True
                                )
                            if not isinstance(valid_mask, torch.Tensor):
                                valid_mask = torch.ones(
                                    (
                                        int(masks.shape[0]),
                                        1,
                                        int(masks.shape[2]),
                                        int(masks.shape[3]),
                                    ),
                                    dtype=torch.bool,
                                    device=device_str,
                                )
                            else:
                                valid_mask = valid_mask.to(
                                    device_str, non_blocking=True
                                )
                            if isinstance(key_padding_mask, torch.Tensor):
                                key_padding_mask = key_padding_mask.to(
                                    device_str, non_blocking=True
                                )
                            else:
                                key_padding_mask = None

                            pred_all = None
                            if hasattr(eval_head, "forward_all"):
                                try:
                                    pred_all = eval_head.forward_all(
                                        feats, key_padding_mask=key_padding_mask
                                    )
                                except TypeError:
                                    pred_all = eval_head.forward_all(feats)
                            if isinstance(pred_all, dict) and isinstance(
                                pred_all.get("kpt_logits"), torch.Tensor
                            ):
                                logits = pred_all["kpt_logits"]
                            else:
                                try:
                                    logits = eval_head(
                                        feats, key_padding_mask=key_padding_mask
                                    )
                                except TypeError:
                                    logits = eval_head(feats)
                            logits = logits.masked_fill(~valid_mask, -20.0)
                            probs = torch.sigmoid(logits)

                            loss_val = _masked_bce_with_logits(
                                logits,
                                masks,
                                valid_mask,
                                balanced=bool(balanced_bce),
                                max_pos_weight=float(max_pos_weight),
                            )
                            bce_mode = str(bce_type or "bce").strip().lower()
                            if bce_mode == "focal":
                                loss_val = _masked_focal_bce_with_logits(
                                    logits,
                                    masks,
                                    valid_mask,
                                    alpha=float(focal_alpha),
                                    gamma=float(focal_gamma),
                                )
                            if float(dice_loss_weight) > 0.0:
                                loss_val = loss_val + float(
                                    dice_loss_weight
                                ) * _dice_loss_masked(probs, masks, valid_mask)
                            if (
                                float(coord_weight) > 0.0
                                and coords.numel() > 0
                                and coord_mask.numel() > 0
                            ):
                                pred_xy = _soft_argmax_coords_batched(
                                    probs, patch_size=int(extractor.patch_size)
                                )
                                loss_val = loss_val + float(coord_weight) * _coord_loss(
                                    pred_xy.reshape(-1, 2),
                                    coords.reshape(-1, 2),
                                    coord_mask.reshape(-1),
                                    mode=coord_loss_type,
                                )
                            if isinstance(pred_all, dict):
                                inst_logits = pred_all.get("inst_logits")
                                obj_logits = pred_all.get("obj_logits")
                                box_logits = pred_all.get("box_logits")
                                instance_target = masks.max(dim=1, keepdim=True).values
                                if (
                                    isinstance(inst_logits, torch.Tensor)
                                    and float(inst_loss_weight) > 0.0
                                ):
                                    loss_val = loss_val + float(
                                        inst_loss_weight
                                    ) * float(aux_scale) * _masked_bce_with_logits(
                                        inst_logits,
                                        instance_target,
                                        valid_mask,
                                        balanced=False,
                                        max_pos_weight=1.0,
                                    )
                                if (
                                    isinstance(obj_logits, torch.Tensor)
                                    and float(obj_loss_weight) > 0.0
                                ):
                                    obj_target = (
                                        (instance_target > 0.5)
                                        .any(dim=(2, 3), keepdim=True)
                                        .to(dtype=torch.float32)
                                    )
                                    obj_target_map = obj_target.expand_as(obj_logits)
                                    loss_val = loss_val + float(
                                        obj_loss_weight
                                    ) * float(aux_scale) * _masked_bce_with_logits(
                                        obj_logits,
                                        obj_target_map,
                                        valid_mask,
                                        balanced=False,
                                        max_pos_weight=1.0,
                                    )
                                if (
                                    isinstance(box_logits, torch.Tensor)
                                    and float(box_loss_weight) > 0.0
                                ):
                                    box_tgt, has_box = _instance_box_targets_from_masks(
                                        instance_target, valid_mask_b1hw=valid_mask
                                    )
                                    box_pred = torch.sigmoid(box_logits).mean(
                                        dim=(2, 3)
                                    )
                                    if bool(has_box.any()):
                                        loss_val = (
                                            loss_val
                                            + float(box_loss_weight)
                                            * float(aux_scale)
                                            * torch.abs(
                                                box_pred[has_box] - box_tgt[has_box]
                                            ).mean()
                                        )
                            losses.append(float(loss_val.detach().cpu().item()))

                            gt = (masks > 0.5) & valid_mask
                            pred = (probs >= float(threshold)) & valid_mask
                            tp += (pred & gt).sum(dim=(0, 2, 3)).to(dtype=torch.float32)
                            fp += (
                                (pred & ~gt).sum(dim=(0, 2, 3)).to(dtype=torch.float32)
                            )
                            fn += (
                                (~pred & gt).sum(dim=(0, 2, 3)).to(dtype=torch.float32)
                            )

                            bsz, k, h_p, w_p = probs.shape
                            flat_gt = masks.view(bsz, k, -1)
                            gt_present = flat_gt.sum(dim=2) > 0

                            pred_idx = probs.view(bsz, k, -1).argmax(dim=2)
                            gt_idx = flat_gt.argmax(dim=2)
                            pred_y = (pred_idx // w_p).to(dtype=torch.float32)
                            pred_x = (pred_idx % w_p).to(dtype=torch.float32)
                            gt_y = (gt_idx // w_p).to(dtype=torch.float32)
                            gt_x = (gt_idx % w_p).to(dtype=torch.float32)

                            dist = torch.sqrt(
                                (pred_y - gt_y) ** 2 + (pred_x - gt_x) ** 2
                            ) * float(extractor.patch_size)
                            peak_dist_sum += (
                                dist * gt_present.to(dtype=torch.float32)
                            ).sum(dim=0)
                            peak_dist_count += gt_present.to(dtype=torch.float32).sum(
                                dim=0
                            )

                            conf = (
                                probs.view(bsz, k, -1)
                                .max(dim=2)
                                .values.to(dtype=torch.float32)
                            )
                            pred_xy = _soft_argmax_coords_batched(
                                probs, patch_size=int(extractor.patch_size)
                            )  # B,K,2
                            peak_xy = torch.stack(
                                [
                                    (gt_x + 0.5) * float(extractor.patch_size),
                                    (gt_y + 0.5) * float(extractor.patch_size),
                                ],
                                dim=2,
                            )
                            gt_xy_eval = peak_xy
                            use_coords = None
                            if coords.numel() == peak_xy.numel():
                                use_coords = (coord_mask > 0.5) & gt_present
                                gt_xy_eval = torch.where(
                                    use_coords[:, :, None],
                                    coords.to(dtype=torch.float32),
                                    peak_xy,
                                )

                            dist_xy = torch.sqrt(
                                ((pred_xy - gt_xy_eval) ** 2).sum(dim=2)
                            )
                            if isinstance(use_coords, torch.Tensor):
                                coord_err_sum += (
                                    dist_xy * use_coords.to(dtype=dist_xy.dtype)
                                ).sum(dim=0)
                                coord_err_count += use_coords.to(
                                    dtype=dist_xy.dtype
                                ).sum(dim=0)
                            oks = _oks_from_distance(
                                dist_xy, sigma_px=float(extractor.patch_size)
                            )
                            oks = oks * gt_present.to(dtype=oks.dtype)

                            for kp in range(int(kpt_count)):
                                gt_counts[kp] += int(gt_present[:, kp].sum().item())
                                det_scores[kp].extend(
                                    [
                                        float(x)
                                        for x in conf[:, kp].detach().cpu().tolist()
                                    ]
                                )
                                det_oks[kp].extend(
                                    [
                                        float(x)
                                        for x in oks[:, kp].detach().cpu().tolist()
                                    ]
                                )

                            gt_instances_batch = batch.get("gt_instances")
                            image_hw_batch = batch.get("image_hw")
                            image_batch = batch.get("image")

                            def _get_image_hw(
                                value: object, bi: int
                            ) -> Optional[Tuple[int, int]]:
                                if isinstance(value, list):
                                    if bi >= len(value):
                                        return None
                                    item = value[bi]
                                    if (
                                        isinstance(item, (list, tuple))
                                        and len(item) == 2
                                    ):
                                        return int(item[0]), int(item[1])
                                    return None
                                if (
                                    isinstance(value, tuple)
                                    and len(value) == 2
                                    and all(isinstance(v, torch.Tensor) for v in value)
                                ):
                                    h_t, w_t = value
                                    if bi >= int(h_t.shape[0]) or bi >= int(
                                        w_t.shape[0]
                                    ):
                                        return None
                                    return int(h_t[bi].item()), int(w_t[bi].item())
                                if isinstance(value, torch.Tensor):
                                    if value.ndim == 2 and int(value.shape[1]) == 2:
                                        if bi >= int(value.shape[0]):
                                            return None
                                        return int(value[bi, 0].item()), int(
                                            value[bi, 1].item()
                                        )
                                    if value.ndim == 2 and int(value.shape[0]) == 2:
                                        if bi >= int(value.shape[1]):
                                            return None
                                        return int(value[0, bi].item()), int(
                                            value[1, bi].item()
                                        )
                                return None

                            have_gt_meta = (
                                isinstance(gt_instances_batch, list)
                                and image_hw_batch is not None
                            )
                            pred_xy_cpu = pred_xy.detach().cpu()
                            valid_mask_cpu = valid_mask.detach().cpu()
                            for bi in range(int(bsz)):
                                if not have_gt_meta:
                                    break
                                if bi >= len(gt_instances_batch):
                                    continue
                                gt_instances = gt_instances_batch[bi]
                                image_hw = _get_image_hw(image_hw_batch, bi)
                                if (
                                    not isinstance(gt_instances, list)
                                    or image_hw is None
                                ):
                                    continue
                                orig_h, orig_w = int(image_hw[0]), int(image_hw[1])
                                if orig_h <= 0 or orig_w <= 0:
                                    continue
                                valid_rows = valid_mask_cpu[bi, 0].any(dim=1)
                                valid_cols = valid_mask_cpu[bi, 0].any(dim=0)
                                h_i = int(valid_rows.sum().item())
                                w_i = int(valid_cols.sum().item())
                                if h_i <= 0 or w_i <= 0:
                                    continue
                                resized_h = int(h_i) * int(extractor.patch_size)
                                resized_w = int(w_i) * int(extractor.patch_size)
                                if resized_h <= 0 or resized_w <= 0:
                                    continue

                                pred_xy_resized = [
                                    (float(x), float(y))
                                    for x, y in pred_xy_cpu[bi].tolist()
                                ]
                                pred_xy_orig = [
                                    (
                                        float(x) * (float(orig_w) / float(resized_w)),
                                        float(y) * (float(orig_h) / float(resized_h)),
                                    )
                                    for x, y in pred_xy_resized
                                ]
                                # Build comparable overlay pairs:
                                # - one GT point per keypoint index (nearest candidate)
                                # - predicted point shown only when that keypoint has GT
                                overlay_pred_resized: List[Tuple[float, float]] = []
                                overlay_gt_resized: List[Tuple[float, float]] = []
                                for kpt_idx in range(int(kpt_count)):
                                    if kpt_idx >= len(pred_xy_orig):
                                        break
                                    candidates = _collect_gt_candidates(
                                        gt_instances,
                                        kpt_idx=kpt_idx,
                                        image_hw=(orig_h, orig_w),
                                    )
                                    if not candidates:
                                        continue
                                    pred_x_o, pred_y_o = pred_xy_orig[kpt_idx]
                                    gt_x_o, gt_y_o = min(
                                        candidates,
                                        key=lambda pt: (
                                            (float(pt[0]) - float(pred_x_o)) ** 2
                                            + (float(pt[1]) - float(pred_y_o)) ** 2
                                        ),
                                    )
                                    overlay_pred_resized.append(
                                        (
                                            float(pred_x_o)
                                            * (float(resized_w) / float(orig_w)),
                                            float(pred_y_o)
                                            * (float(resized_h) / float(orig_h)),
                                        )
                                    )
                                    overlay_gt_resized.append(
                                        (
                                            float(gt_x_o)
                                            * (float(resized_w) / float(orig_w)),
                                            float(gt_y_o)
                                            * (float(resized_h) / float(orig_h)),
                                        )
                                    )
                                if (
                                    image_batch is not None
                                    and isinstance(image_batch, torch.Tensor)
                                    and len(pred_overlays) < int(pred_max)
                                ):
                                    img = (
                                        image_batch[bi, :, :resized_h, :resized_w]
                                        .detach()
                                        .cpu()
                                    )
                                    pred_overlays.append(
                                        _overlay_keypoints(
                                            img,
                                            pred_xy=overlay_pred_resized,
                                            gt_xy=overlay_gt_resized,
                                        )
                                    )
                                if gt_instances:
                                    pck_acc.update(
                                        pred_xy=pred_xy_orig,
                                        gt_instances=gt_instances,
                                        image_hw=(orig_h, orig_w),
                                        lr_pairs=lr_pairs,
                                    )

                                errors = []
                                for kpt_idx in range(int(kpt_count)):
                                    candidates = _collect_gt_candidates(
                                        gt_instances,
                                        kpt_idx=kpt_idx,
                                        image_hw=(orig_h, orig_w),
                                    )
                                    if not candidates:
                                        continue
                                    err = _min_error_px(
                                        pred_xy_orig[kpt_idx], candidates
                                    )
                                    if err is not None:
                                        errors.append(float(err))
                                if not errors:
                                    continue
                                mean_err = float(sum(errors) / len(errors))
                                if image_batch is not None and isinstance(
                                    image_batch, torch.Tensor
                                ):
                                    img = (
                                        image_batch[bi, :, :resized_h, :resized_w]
                                        .detach()
                                        .cpu()
                                    )
                                    overlay = _overlay_keypoints(
                                        img,
                                        pred_xy=overlay_pred_resized,
                                        gt_xy=overlay_gt_resized,
                                    )
                                    if len(error_samples) < int(error_max):
                                        error_samples.append((mean_err, overlay))
                                    else:
                                        error_samples.sort(key=lambda x: x[0])
                                        if mean_err > error_samples[0][0]:
                                            error_samples[0] = (mean_err, overlay)
                            if log_interval > 0 and (
                                (n_val % int(log_interval)) == 0
                                or int(n_val) == int(val_steps_total)
                            ):
                                logger.info(
                                    "Epoch %d/%d - val step %d/%d (%.1f%%)",
                                    epoch,
                                    epochs,
                                    n_val,
                                    val_steps_total,
                                    (
                                        100.0
                                        * float(n_val)
                                        / max(1.0, float(val_steps_total))
                                    ),
                                )
                    if losses:
                        val_loss = float(sum(losses) / len(losses))
                    if pck_acc is not None:
                        pck_summary = pck_acc.summary(include_per_keypoint=True)
                        swap_rate = pck_summary.get("swap_rate")
                    # Validation metrics (YOLO-like reporting) averaged over keypoints with GT present.
                    denom_dice = (2.0 * tp + fp + fn).clamp(min=1e-8)
                    dice = (2.0 * tp) / denom_dice
                    iou = tp / (tp + fp + fn).clamp(min=1e-8)
                    precision = tp / (tp + fp).clamp(min=1e-8)
                    recall = tp / (tp + fn).clamp(min=1e-8)
                    valid_kpt = (tp + fn) > 0
                    if bool(valid_kpt.any().item()):
                        dice_mean = float(dice[valid_kpt].mean().item())
                        iou_mean = float(iou[valid_kpt].mean().item())
                        precision_mean = float(precision[valid_kpt].mean().item())
                        recall_mean = float(recall[valid_kpt].mean().item())
                        peak_dist_px = (peak_dist_sum / peak_dist_count.clamp(min=1.0))[
                            valid_kpt
                        ]
                        peak_dist_px_mean = float(peak_dist_px.mean().item())
                    else:
                        dice_mean = float(dice.mean().item())
                        iou_mean = float(iou.mean().item())
                        precision_mean = float(precision.mean().item())
                        recall_mean = float(recall.mean().item())
                        peak_dist_px_mean = float(
                            (peak_dist_sum / peak_dist_count.clamp(min=1.0))
                            .mean()
                            .item()
                        )

                    # COCO-style AP over IoU thresholds (0.50:0.95) using confidence = max prob.
                    ap50_list: List[float] = []
                    ap5095_list: List[float] = []
                    prec_list: List[float] = []
                    rec_list: List[float] = []
                    conf_gate = 0.25
                    for i in range(int(kpt_count)):
                        n_gt = int(gt_counts[i])
                        if n_gt <= 0:
                            continue
                        scores_i = det_scores[i]
                        oks_i = det_oks[i]
                        ap_by_thr = []
                        for thr in _AP_IOU_THRESHOLDS:
                            tp_flags = [float(v) >= float(thr) for v in oks_i]
                            ap_by_thr.append(
                                _average_precision(scores_i, tp_flags, num_gt=n_gt)
                            )
                        ap50_list.append(float(ap_by_thr[0] if ap_by_thr else 0.0))
                        ap5095_list.append(
                            float(sum(ap_by_thr) / max(1, len(ap_by_thr)))
                        )

                        tp_i = 0
                        fp_i = 0
                        for score, oks_v in zip(scores_i, oks_i):
                            if float(score) < float(conf_gate):
                                continue
                            if float(oks_v) >= 0.50:
                                tp_i += 1
                            else:
                                fp_i += 1
                        prec_list.append(float(tp_i / max(1, (tp_i + fp_i))))
                        rec_list.append(float(tp_i / max(1, n_gt)))

                    if ap50_list:
                        map50 = float(sum(ap50_list) / len(ap50_list))
                    else:
                        map50 = 0.0
                    if ap5095_list:
                        map5095 = float(sum(ap5095_list) / len(ap5095_list))
                    else:
                        map5095 = 0.0
                    if rec_list:
                        recall_pose_mean = float(sum(rec_list) / len(rec_list))
                    else:
                        recall_pose_mean = 0.0
                    if prec_list:
                        precision_pose_mean = float(sum(prec_list) / len(prec_list))
                    else:
                        precision_pose_mean = 0.0

                elapsed = float(time.time() - t0)
                pck_vals = {}
                if isinstance(pck_summary, dict):
                    pck_vals = pck_summary.get("pck") or {}
                if not isinstance(pck_vals, dict):
                    pck_vals = {}

                def _pck_str(thr: float) -> str:
                    value = pck_vals.get(str(float(thr)))
                    if value is None:
                        return ""
                    try:
                        return f"{float(value):.6f}"
                    except Exception:
                        return ""

                row = {
                    "epoch": epoch,
                    "train_loss": f"{train_loss_total:.6f}",
                    "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                    "metrics/precision(P)": ""
                    if precision_pose_mean is None
                    else f"{precision_pose_mean:.6f}",
                    "metrics/recall(P)": ""
                    if recall_pose_mean is None
                    else f"{recall_pose_mean:.6f}",
                    "metrics/mAP50(P)": "" if map50 is None else f"{map50:.6f}",
                    "metrics/mAP50-95(P)": "" if map5095 is None else f"{map5095:.6f}",
                    "metrics/pck@2px": _pck_str(2.0),
                    "metrics/pck@4px": _pck_str(4.0),
                    "metrics/pck@8px": _pck_str(8.0),
                    "metrics/pck@16px": _pck_str(16.0),
                    "metrics/swap_rate": ""
                    if swap_rate is None
                    else f"{float(swap_rate):.6f}",
                    "seconds": f"{elapsed:.2f}",
                }
                writer.writerow(row)
                fh.flush()

                tb_writer.add_scalar("loss/train_base", train_loss, epoch)
                tb_writer.add_scalar("loss/train", train_loss_total, epoch)
                if dice_terms_count:
                    tb_writer.add_scalar(
                        "loss/dice",
                        dice_loss_sum / max(1, dice_terms_count),
                        epoch,
                    )
                if coord_terms_count:
                    tb_writer.add_scalar(
                        "loss/coord",
                        coord_loss_sum / max(1, coord_terms_count),
                        epoch,
                    )
                if pair_terms_count:
                    tb_writer.add_scalar(
                        "loss/pair_overlap",
                        pair_overlap_sum / max(1, pair_terms_count),
                        epoch,
                    )
                    tb_writer.add_scalar(
                        "loss/pair_margin",
                        pair_margin_sum / max(1, pair_terms_count),
                        epoch,
                    )
                if side_terms_count:
                    tb_writer.add_scalar(
                        "loss/lr_side",
                        side_loss_sum / max(1, side_terms_count),
                        epoch,
                    )
                if val_loss is not None:
                    tb_writer.add_scalar("loss/val", val_loss, epoch)
                    tb_writer.add_scalar("val/dice_mean", dice_mean, epoch)
                    tb_writer.add_scalar("val/iou_mean", iou_mean, epoch)
                    tb_writer.add_scalar("val/precision_mean", precision_mean, epoch)
                    tb_writer.add_scalar("val/recall_mean", recall_mean, epoch)
                    tb_writer.add_scalar(
                        "val/peak_dist_px_mean", peak_dist_px_mean, epoch
                    )
                    if map50 is not None and map5095 is not None:
                        tb_writer.add_scalar("metrics/mAP50(P)", float(map50), epoch)
                        tb_writer.add_scalar(
                            "metrics/mAP50-95(P)", float(map5095), epoch
                        )
                    if precision_pose_mean is not None and recall_pose_mean is not None:
                        tb_writer.add_scalar(
                            "metrics/precision(P)", float(precision_pose_mean), epoch
                        )
                        tb_writer.add_scalar(
                            "metrics/recall(P)", float(recall_pose_mean), epoch
                        )
                    tb_writer.add_scalar(
                        "val/keypoints_present_frac",
                        float(valid_kpt.float().mean().item()),
                        epoch,
                    )
                    if pck_summary:
                        pck_vals = pck_summary.get("pck") or {}
                        for thr in pck_thresholds:
                            key = str(float(thr))
                            if key in pck_vals and pck_vals[key] is not None:
                                tb_writer.add_scalar(
                                    f"val/pck@{int(thr)}px",
                                    float(pck_vals[key]),
                                    epoch,
                                )
                        if swap_rate is not None:
                            tb_writer.add_scalar(
                                "val/swap_rate", float(swap_rate), epoch
                            )
                        per_kpt = (
                            pck_summary.get("per_keypoint")
                            if isinstance(pck_summary, dict)
                            else None
                        )
                        if isinstance(per_kpt, dict):
                            for name, stats in per_kpt.items():
                                if not isinstance(stats, dict):
                                    continue
                                for thr in pck_thresholds:
                                    key = str(float(thr))
                                    pck_k = stats.get("pck", {}).get(key)
                                    if pck_k is None:
                                        continue
                                    tb_writer.add_scalar(
                                        f"val/pck@{int(thr)}px/{name}",
                                        float(pck_k),
                                        epoch,
                                    )
                    # Per-keypoint metrics
                    names = list(keypoint_names or [])
                    for i in range(int(kpt_count)):
                        name = names[i] if i < len(names) else f"kp_{i}"
                        tb_writer.add_scalar(
                            f"val/dice/{name}", float(dice[i].item()), epoch
                        )
                        tb_writer.add_scalar(
                            f"val/iou/{name}", float(iou[i].item()), epoch
                        )
                        if coord_err_count[i] > 0:
                            tb_writer.add_scalar(
                                f"val/coord_error_px/{name}",
                                float((coord_err_sum[i] / coord_err_count[i]).item()),
                                epoch,
                            )
                    if error_samples:
                        error_samples.sort(key=lambda x: x[0], reverse=True)
                        overlay_stack = _stack_chw_images_with_padding(
                            [sample[1] for sample in error_samples],
                            pad_value=0.0,
                        ).clamp(0.0, 1.0)
                        _tb_add_images(
                            tb_writer, "val/error_overlays", overlay_stack, epoch
                        )
                    if pred_overlays:
                        _tb_add_images(
                            tb_writer,
                            "val/pred_overlays",
                            _stack_chw_images_with_padding(
                                pred_overlays, pad_value=0.0
                            ).clamp(0.0, 1.0),
                            epoch,
                        )
                    if bool(tb_projector) and val_ds is not None:
                        interval = 5
                        if (
                            int(epoch) == 1
                            or int(epoch) % int(interval) == 0
                            or int(epoch) == int(epochs)
                        ):
                            try:
                                from annolid.segmentation.dino_kpseg.tensorboard_embeddings import (
                                    add_dino_kpseg_projector_embeddings,
                                )

                                def _predict_probs_patch(
                                    feats: torch.Tensor,
                                ) -> torch.Tensor:
                                    x = feats.unsqueeze(0).to(
                                        device_str, dtype=torch.float32
                                    )
                                    model_for_pred = (
                                        ema_head if ema_head is not None else head
                                    )
                                    try:
                                        logits = model_for_pred(x)
                                    except TypeError:
                                        logits = model_for_pred(x)
                                    logits = logits[0].to("cpu")
                                    return torch.sigmoid(logits)

                                split_tag = f"val_epoch_{int(epoch):03d}"
                                add_dino_kpseg_projector_embeddings(
                                    tb_writer,
                                    log_dir=tb_dir,
                                    split=split_tag,
                                    ds=val_ds,
                                    keypoint_names=list(keypoint_names)
                                    if keypoint_names
                                    else [f"kpt_{i}" for i in range(int(kpt_count))],
                                    max_images=min(int(tb_projector_max_images), 32),
                                    max_patches=min(
                                        int(tb_projector_max_patches), 2000
                                    ),
                                    per_image_per_keypoint=int(
                                        tb_projector_per_image_per_keypoint
                                    ),
                                    pos_threshold=float(tb_projector_pos_threshold),
                                    add_negatives=bool(tb_projector_add_negatives),
                                    neg_threshold=float(tb_projector_neg_threshold),
                                    negatives_per_image=int(
                                        tb_projector_negatives_per_image
                                    ),
                                    crop_px=int(tb_projector_crop_px),
                                    sprite_border_px=int(tb_projector_sprite_border_px),
                                    seed=(
                                        int(augment_cfg.seed)
                                        if augment_cfg.seed is not None
                                        else 0
                                    ),
                                    tag=f"dino_kpseg/patch_embeddings/{split_tag}",
                                    predict_probs_patch=_predict_probs_patch,
                                    write_csv=True,
                                    global_step=int(epoch),
                                )
                            except Exception as exc:
                                tb_writer.add_text(
                                    "tensorboard/projector_error",
                                    f"{type(exc).__name__}: {exc}",
                                    epoch,
                                )
                tb_writer.add_scalar("time/epoch_seconds", elapsed, epoch)
                if batch_dice_count:
                    tb_writer.add_scalar(
                        "metrics/dice_mean_train_epoch0",
                        batch_dice_sum / max(1, batch_dice_count),
                        epoch,
                    )

                head_for_ckpt = ema_head if ema_head is not None else head
                payload = checkpoint_pack(head=head_for_ckpt, meta=meta)
                torch.save(payload, last_path)
                selected_best = _resolve_selection_metric(
                    desired_best_metric,
                    train_loss=float(train_loss_total),
                    val_loss=(float(val_loss) if val_loss is not None else None),
                    pck_vals=pck_vals,
                    pck_weighted_weights=pck_weighted_weights,
                )
                if selected_best is None or not math.isfinite(float(selected_best)):
                    selected_best = (
                        float(val_loss)
                        if val_loss is not None
                        else float(train_loss_total)
                    )

                improved_best = (
                    float(selected_best) > float(best_metric_value)
                    if best_higher_is_better
                    else float(selected_best) < float(best_metric_value)
                )
                if improved_best:
                    best_metric_value = float(selected_best)
                    torch.save(payload, best_path)

                tb_writer.add_scalar(
                    "checkpoint/best_metric", float(best_metric_value), epoch
                )
                tb_writer.add_scalar(
                    "checkpoint/current_metric", float(selected_best), epoch
                )

                metric_for_stop = _resolve_selection_metric(
                    desired_early_stop_metric,
                    train_loss=float(train_loss_total),
                    val_loss=(float(val_loss) if val_loss is not None else None),
                    pck_vals=pck_vals,
                    pck_weighted_weights=pck_weighted_weights,
                )
                if metric_for_stop is None or not math.isfinite(float(metric_for_stop)):
                    metric_for_stop = (
                        float(val_loss)
                        if val_loss is not None
                        else float(train_loss_total)
                    )

                if early_stop_enabled and math.isfinite(float(metric_for_stop)):
                    improved = (
                        float(metric_for_stop)
                        > (float(best_metric_for_stop) + float(min_delta))
                        if early_stop_higher_is_better
                        else float(metric_for_stop)
                        < (float(best_metric_for_stop) - float(min_delta))
                    )
                    if improved:
                        best_metric_for_stop = float(metric_for_stop)
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    tb_writer.add_scalar(
                        "early_stop/current_metric", float(metric_for_stop), epoch
                    )
                    tb_writer.add_scalar(
                        "early_stop/best_metric", float(best_metric_for_stop), epoch
                    )
                    tb_writer.add_scalar(
                        "early_stop/bad_epochs", int(bad_epochs), epoch
                    )

                    if int(epoch) >= int(min_epochs) and int(bad_epochs) >= int(
                        patience
                    ):
                        reason = (
                            f"Early stopping triggered at epoch {epoch}: "
                            f"metric({desired_early_stop_metric})={float(metric_for_stop):.6f} "
                            f"best={float(best_metric_for_stop):.6f} "
                            f"min_delta={float(min_delta)} patience={int(patience)}"
                        )
                        logger.info(reason)
                        tb_writer.add_text("early_stop/stop_reason", reason, epoch)
                        tb_writer.flush()
                        break

                logger.info(
                    "Epoch %d/%d - train_loss=%.6f val_loss=%s mAP50-95(P)=%s time=%.2fs",
                    epoch,
                    epochs,
                    train_loss,
                    f"{val_loss:.6f}" if val_loss is not None else "NA",
                    f"{map5095:.4f}" if map5095 is not None else "NA",
                    elapsed,
                )
                tb_writer.flush()
                if scheduler is not None and int(epoch) >= int(lr_decay_start_epoch):
                    try:
                        scheduler.step()
                    except Exception:
                        pass

                # Best-effort: keep MPS memory stable on long runs.
                if device_str == "mps":
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass

        # Persist a "YOLO-like" loss curve image in TensorBoard (no matplotlib).
        curve = _draw_loss_curve_image(csv_path)
        if curve is not None:
            _tb_add_image(tb_writer, "plots/loss_curve", curve, int(epochs))
    finally:
        tb_writer.flush()
        tb_writer.close()

    return best_path


def main(argv: Optional[list[str]] = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    config_defaults: Dict[str, object] = {}
    if pre_args.config:
        config_defaults = _load_train_config_defaults(Path(pre_args.config))

    p = argparse.ArgumentParser(
        description="Train a DINOv3 keypoint mask segmentation head."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file; CLI flags override config values.",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML (YOLO pose data.yaml, LabelMe spec.yaml, or COCO spec.yaml)",
    )
    p.add_argument(
        "--data-format",
        choices=("auto", "yolo", "labelme", "coco"),
        default="auto",
        help="Dataset annotation format (default: auto-detect from YAML).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Run output directory (if omitted, creates a new run under ANNOLID_RUNS_ROOT/~/annolid_logs/runs).",
    )
    p.add_argument(
        "--runs-root",
        default=None,
        help="Runs root (overrides ANNOLID_RUNS_ROOT/~/annolid_logs/runs)",
    )
    p.add_argument(
        "--run-name", default=None, help="Optional run name (default: timestamp)"
    )
    p.add_argument(
        "--model-name",
        default=dino_defaults.MODEL_NAME,
        help="Hugging Face model id or dinov3 alias",
    )
    p.add_argument("--short-side", type=int, default=dino_defaults.SHORT_SIDE)
    p.add_argument(
        "--layers",
        type=str,
        default=dino_defaults.LAYERS,
        help="Comma-separated transformer block indices",
    )
    p.add_argument(
        "--feature-merge",
        choices=("concat", "mean", "max"),
        default=dino_defaults.FEATURE_MERGE,
        help="How to merge multi-layer DINO features.",
    )
    p.add_argument(
        "--feature-align-dim",
        default=str(dino_defaults.FEATURE_ALIGN_DIM),
        help="Optional trainable 1x1 feature alignment dim before the kp head (0=off, or 'auto').",
    )
    p.add_argument("--radius-px", type=float, default=dino_defaults.RADIUS_PX)
    p.add_argument(
        "--mask-type",
        choices=("disk", "gaussian"),
        default=dino_defaults.MASK_TYPE,
        help="Keypoint supervision mask type",
    )
    p.add_argument(
        "--heatmap-sigma",
        type=float,
        default=None,
        help="Gaussian sigma in pixels (original image space). Defaults to radius_px/2.",
    )
    p.add_argument(
        "--instance-mode",
        choices=("auto", "union", "per_instance"),
        default=dino_defaults.INSTANCE_MODE,
        help="How to handle multiple pose instances per image.",
    )
    p.add_argument(
        "--bbox-scale",
        type=float,
        default=dino_defaults.BBOX_SCALE,
        help="Scale factor for per-instance bounding box crops.",
    )
    p.add_argument("--hidden-dim", type=int, default=dino_defaults.HIDDEN_DIM)
    p.add_argument("--lr", type=float, default=dino_defaults.LR)
    p.add_argument("--epochs", type=int, default=dino_defaults.EPOCHS)
    p.add_argument(
        "--overfit-n",
        type=int,
        default=0,
        help="Overfit mode: train/val on N images (0=off).",
    )
    p.add_argument(
        "--batch",
        "--batch-size",
        dest="batch",
        type=int,
        default=dino_defaults.BATCH,
        help="Batch size (uses padded collation for variable feature sizes).",
    )
    p.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch * accumulate).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0=off).",
    )
    p.add_argument(
        "--log-every-steps",
        type=int,
        default=100,
        help="Emit step-level progress logs every N train/val batches (0=off).",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--balanced-bce",
        dest="balanced_bce",
        action="store_true",
        help="Enable per-batch positive class reweighting for BCE (recommended).",
    )
    group.add_argument(
        "--no-balanced-bce",
        dest="balanced_bce",
        action="store_false",
        help="Disable positive class reweighting for BCE.",
    )
    p.set_defaults(balanced_bce=True)
    p.add_argument(
        "--max-pos-weight",
        type=float,
        default=50.0,
        help="Clamp for balanced BCE pos_weight (prevents instability).",
    )
    p.add_argument(
        "--bce-type",
        choices=("bce", "focal"),
        default=dino_defaults.BCE_TYPE,
        help="Loss type for mask supervision (default: bce).",
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=dino_defaults.FOCAL_ALPHA,
        help="Alpha for focal BCE.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=dino_defaults.FOCAL_GAMMA,
        help="Gamma for focal BCE.",
    )
    sched_group = p.add_mutually_exclusive_group()
    sched_group.add_argument(
        "--cos-lr",
        dest="cos_lr",
        action="store_true",
        help="Use cosine LR decay (recommended).",
    )
    sched_group.add_argument(
        "--no-cos-lr",
        dest="cos_lr",
        action="store_false",
        help="Disable cosine LR decay.",
    )
    p.set_defaults(cos_lr=True)
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Linear LR warmup epochs (0=off).",
    )
    p.add_argument(
        "--lrf",
        "--lr-final-frac",
        dest="lr_final_frac",
        type=float,
        default=0.01,
        help="Final LR fraction for cosine schedule.",
    )
    p.add_argument(
        "--flat-epoch",
        type=int,
        default=dino_defaults.FLAT_EPOCH,
        help="Hold base LR through this epoch before cosine decay starts (0=off).",
    )
    p.add_argument(
        "--coord-warmup-epochs",
        type=int,
        default=dino_defaults.COORD_WARMUP_EPOCHS,
        help="Warm up coordinate loss over N epochs (0=off).",
    )
    p.add_argument(
        "--multitask-aux-warmup-epochs",
        type=int,
        default=dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
        help="Warm up multitask auxiliary losses over N epochs (0=off).",
    )
    p.add_argument(
        "--schedule-profile",
        choices=("baseline", "aggressive_s"),
        default=dino_defaults.SCHEDULE_PROFILE,
        help="Optional schedule preset. aggressive_s sets epoch windows to [4,64,120] with 12 no-aug tail.",
    )
    p.add_argument("--threshold", type=float, default=dino_defaults.THRESHOLD)
    p.add_argument(
        "--radius-schedule",
        choices=("none", "linear"),
        default=dino_defaults.RADIUS_SCHEDULE,
        help="Schedule radius_px across epochs.",
    )
    p.add_argument(
        "--radius-start-px",
        type=float,
        default=dino_defaults.RADIUS_START_PX,
    )
    p.add_argument(
        "--radius-end-px",
        type=float,
        default=dino_defaults.RADIUS_END_PX,
    )
    p.add_argument("--device", default=None)
    p.add_argument("--no-cache", action="store_true", help="Disable feature caching")
    ema_group = p.add_mutually_exclusive_group()
    ema_group.add_argument(
        "--ema",
        dest="use_ema",
        action="store_true",
        help="Enable EMA model for validation/checkpoints.",
    )
    ema_group.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Disable EMA model.",
    )
    p.set_defaults(use_ema=bool(dino_defaults.USE_EMA))
    p.add_argument(
        "--ema-decay",
        type=float,
        default=dino_defaults.EMA_DECAY,
        help="EMA decay factor (0..1).",
    )
    p.add_argument(
        "--head-type",
        choices=("conv", "attn", "hybrid", "multitask"),
        default=dino_defaults.HEAD_TYPE,
        help="Head architecture",
    )
    p.add_argument(
        "--attn-heads",
        type=int,
        default=dino_defaults.ATTN_HEADS,
        help="Attention heads (attn head only)",
    )
    p.add_argument(
        "--attn-layers",
        type=int,
        default=dino_defaults.ATTN_LAYERS,
        help="Attention layers (attn head only)",
    )
    p.add_argument(
        "--lr-pair-loss-weight",
        type=float,
        default=dino_defaults.LR_PAIR_LOSS_WEIGHT,
        help="Optional symmetric-pair regularizer weight (0=off).",
    )
    p.add_argument(
        "--lr-pair-margin-px",
        type=float,
        default=dino_defaults.LR_PAIR_MARGIN_PX,
        help="Optional minimum separation margin in pixels for symmetric pairs (0=off).",
    )
    p.add_argument(
        "--lr-side-loss-weight",
        type=float,
        default=dino_defaults.LR_SIDE_LOSS_WEIGHT,
        help="Optional left/right side-consistency loss weight (0=off). Uses orientation anchors when available.",
    )
    p.add_argument(
        "--lr-side-loss-margin",
        type=float,
        default=dino_defaults.LR_SIDE_LOSS_MARGIN,
        help="Margin for side-consistency in [0,1] (0=enforce opposite sign).",
    )
    p.add_argument(
        "--dice-loss-weight",
        type=float,
        default=dino_defaults.DICE_LOSS_WEIGHT,
        help="Dice loss weight (0=off).",
    )
    p.add_argument(
        "--coord-loss-weight",
        type=float,
        default=dino_defaults.COORD_LOSS_WEIGHT,
        help="Coordinate regression loss weight (0=off).",
    )
    p.add_argument(
        "--coord-loss-type",
        choices=("smooth_l1", "l1", "l2"),
        default=dino_defaults.COORD_LOSS_TYPE,
        help="Coordinate regression loss type.",
    )
    p.add_argument(
        "--obj-loss-weight",
        type=float,
        default=dino_defaults.OBJ_LOSS_WEIGHT,
        help="Auxiliary objectness loss weight (multitask head).",
    )
    p.add_argument(
        "--box-loss-weight",
        type=float,
        default=dino_defaults.BOX_LOSS_WEIGHT,
        help="Auxiliary box regression loss weight (multitask head).",
    )
    p.add_argument(
        "--inst-loss-weight",
        type=float,
        default=dino_defaults.INST_LOSS_WEIGHT,
        help="Auxiliary instance-mask loss weight (multitask head).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=dino_defaults.EARLY_STOP_PATIENCE,
        help="Early stop patience (0=off)",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=dino_defaults.EARLY_STOP_MIN_DELTA,
        help="Min metric improvement to reset patience",
    )
    p.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=dino_defaults.EARLY_STOP_MIN_EPOCHS,
        help="Do not early-stop before this epoch",
    )
    p.add_argument(
        "--best-metric",
        choices=("pck@8px", "pck_weighted", "val_loss", "train_loss"),
        default=dino_defaults.BEST_METRIC,
        help="Metric for best checkpoint selection (default: pck@8px).",
    )
    p.add_argument(
        "--early-stop-metric",
        choices=("auto", "pck@8px", "pck_weighted", "val_loss", "train_loss"),
        default=dino_defaults.EARLY_STOP_METRIC,
        help="Metric for early stopping (default: auto -> same as best-metric).",
    )
    p.add_argument(
        "--pck-weighted-weights",
        type=str,
        default=dino_defaults.PCK_WEIGHTED_WEIGHTS,
        help="Comma-separated weights for pck_weighted over thresholds [2,4,8,16] px (default: 1,1,1,1).",
    )
    p.add_argument(
        "--tb-add-graph",
        action="store_true",
        help="Export model graph to TensorBoard (can be slow)",
    )
    p.add_argument(
        "--tb-projector",
        action="store_true",
        help="Write a TensorBoard Projector embedding view for DinoKPSEG patch features.",
    )
    p.add_argument(
        "--tb-projector-split",
        choices=("train", "val", "both"),
        default=dino_defaults.TB_PROJECTOR_SPLIT,
        help="Which dataset split(s) to sample for the projector (default: val).",
    )
    p.add_argument(
        "--tb-projector-max-images",
        type=int,
        default=dino_defaults.TB_PROJECTOR_MAX_IMAGES,
    )
    p.add_argument(
        "--tb-projector-max-patches",
        type=int,
        default=dino_defaults.TB_PROJECTOR_MAX_PATCHES,
    )
    p.add_argument(
        "--tb-projector-per-image-per-keypoint",
        type=int,
        default=dino_defaults.TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT,
    )
    p.add_argument(
        "--tb-projector-pos-threshold",
        type=float,
        default=dino_defaults.TB_PROJECTOR_POS_THRESHOLD,
    )
    p.add_argument(
        "--tb-projector-crop-px",
        type=int,
        default=dino_defaults.TB_PROJECTOR_CROP_PX,
    )
    p.add_argument(
        "--tb-projector-sprite-border-px",
        type=int,
        default=dino_defaults.TB_PROJECTOR_SPRITE_BORDER_PX,
    )
    p.add_argument("--tb-projector-add-negatives", action="store_true")
    p.add_argument(
        "--tb-projector-neg-threshold",
        type=float,
        default=dino_defaults.TB_PROJECTOR_NEG_THRESHOLD,
    )
    p.add_argument(
        "--tb-projector-negatives-per-image",
        type=int,
        default=dino_defaults.TB_PROJECTOR_NEGATIVES_PER_IMAGE,
    )
    p.set_defaults(
        tb_projector_add_negatives=bool(dino_defaults.TB_PROJECTOR_ADD_NEGATIVES)
    )
    aug_group = p.add_mutually_exclusive_group()
    aug_group.add_argument(
        "--augment", dest="augment", action="store_true", help="Enable augmentations"
    )
    aug_group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable augmentations",
    )
    p.set_defaults(
        augment=bool(dino_defaults.AUGMENT_ENABLED),
        tb_projector=bool(dino_defaults.TB_PROJECTOR),
    )
    p.add_argument(
        "--hflip",
        type=float,
        default=dino_defaults.HFLIP,
        help="Horizontal flip probability",
    )
    p.add_argument(
        "--degrees",
        type=float,
        default=dino_defaults.DEGREES,
        help="Random rotation degrees (+/-)",
    )
    p.add_argument(
        "--translate",
        type=float,
        default=dino_defaults.TRANSLATE,
        help="Random translate fraction (+/-)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=dino_defaults.SCALE,
        help="Random scale fraction (+/-)",
    )
    p.add_argument(
        "--brightness",
        type=float,
        default=dino_defaults.BRIGHTNESS,
        help="Brightness jitter (+/-)",
    )
    p.add_argument(
        "--contrast",
        type=float,
        default=dino_defaults.CONTRAST,
        help="Contrast jitter (+/-)",
    )
    p.add_argument(
        "--saturation",
        type=float,
        default=dino_defaults.SATURATION,
        help="Saturation jitter (+/-)",
    )
    p.add_argument(
        "--aug-start-epoch",
        type=int,
        default=dino_defaults.AUG_START_EPOCH,
        help="First epoch where train augmentations are active.",
    )
    p.add_argument(
        "--aug-stop-epoch",
        type=int,
        default=dino_defaults.AUG_STOP_EPOCH,
        help="Last epoch where train augmentations are active (0=until training end/no-aug tail).",
    )
    p.add_argument(
        "--no-aug-epoch",
        type=int,
        default=dino_defaults.NO_AUG_EPOCH,
        help="Disable train augmentations for the final N epochs (0=off).",
    )
    p.add_argument(
        "--change-matcher",
        action="store_true",
        default=bool(dino_defaults.AGGRESSIVE_S_CHANGE_MATCHER),
        help="Record matcher-switch metadata for experiment parity (not applied by DinoKPSEG trainer).",
    )
    p.add_argument(
        "--matcher-change-epoch",
        type=int,
        default=dino_defaults.AGGRESSIVE_S_MATCHER_CHANGE_EPOCH,
        help="Epoch for matcher switch metadata (compatibility knob).",
    )
    p.add_argument(
        "--iou-order-alpha",
        type=float,
        default=dino_defaults.AGGRESSIVE_S_IOU_ORDER_ALPHA,
        help="IOU order alpha metadata for matcher switch compatibility.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (also used for augmentations)",
    )
    if config_defaults:
        valid_dests = {a.dest for a in p._actions if getattr(a, "dest", None)}
        filtered = {k: v for k, v in config_defaults.items() if k in valid_dests}
        p.set_defaults(**filtered)
    args = p.parse_args(argv)
    _apply_schedule_profile_defaults(args)

    layers = parse_layers(args.layers)

    if args.output:
        out_dir = Path(args.output).expanduser().resolve()
    else:
        runs_root = (
            Path(args.runs_root).expanduser().resolve()
            if args.runs_root
            else shared_runs_root()
        )
        out_dir = allocate_run_dir(
            task="dino_kpseg",
            model="train",
            runs_root=runs_root,
            run_name=args.run_name,
        )
    _ensure_dir(out_dir)

    best = train(
        data_yaml=Path(args.data).expanduser().resolve(),
        data_format=str(args.data_format),
        output_dir=out_dir,
        model_name=str(args.model_name),
        short_side=int(args.short_side),
        layers=layers,
        feature_merge=str(args.feature_merge),
        feature_align_dim=args.feature_align_dim,
        radius_px=float(args.radius_px),
        mask_type=str(args.mask_type),
        heatmap_sigma_px=(
            float(args.heatmap_sigma) if args.heatmap_sigma is not None else None
        ),
        instance_mode=str(args.instance_mode),
        bbox_scale=float(args.bbox_scale),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        epochs=int(args.epochs),
        threshold=float(args.threshold),
        batch_size=int(args.batch),
        accumulate=int(args.accumulate),
        grad_clip=float(args.grad_clip),
        balanced_bce=bool(args.balanced_bce),
        max_pos_weight=float(args.max_pos_weight),
        bce_type=str(args.bce_type),
        focal_alpha=float(args.focal_alpha),
        focal_gamma=float(args.focal_gamma),
        cos_lr=bool(args.cos_lr),
        warmup_epochs=int(args.warmup_epochs),
        lr_final_frac=float(args.lr_final_frac),
        flat_epoch=int(args.flat_epoch),
        coord_warmup_epochs=int(args.coord_warmup_epochs),
        multitask_aux_warmup_epochs=int(args.multitask_aux_warmup_epochs),
        schedule_profile=str(args.schedule_profile),
        device=args.device,
        cache_features=not bool(args.no_cache),
        use_ema=bool(args.use_ema),
        ema_decay=float(args.ema_decay),
        head_type=str(args.head_type),
        attn_heads=int(args.attn_heads),
        attn_layers=int(args.attn_layers),
        lr_pair_loss_weight=float(args.lr_pair_loss_weight),
        lr_pair_margin_px=float(args.lr_pair_margin_px),
        lr_side_loss_weight=float(args.lr_side_loss_weight),
        lr_side_loss_margin=float(args.lr_side_loss_margin),
        change_matcher=bool(args.change_matcher),
        matcher_change_epoch=int(args.matcher_change_epoch),
        iou_order_alpha=float(args.iou_order_alpha),
        log_every_steps=int(args.log_every_steps),
        dice_loss_weight=float(args.dice_loss_weight),
        coord_loss_weight=float(args.coord_loss_weight),
        coord_loss_type=str(args.coord_loss_type),
        obj_loss_weight=float(args.obj_loss_weight),
        box_loss_weight=float(args.box_loss_weight),
        inst_loss_weight=float(args.inst_loss_weight),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_min_epochs=int(args.early_stop_min_epochs),
        best_metric=str(args.best_metric),
        early_stop_metric=str(args.early_stop_metric),
        pck_weighted_weights=_parse_weight_list(args.pck_weighted_weights, n=4),
        tb_add_graph=bool(args.tb_add_graph),
        radius_schedule=str(args.radius_schedule),
        radius_start_px=(
            float(args.radius_start_px) if args.radius_start_px is not None else None
        ),
        radius_end_px=(
            float(args.radius_end_px) if args.radius_end_px is not None else None
        ),
        overfit_n=int(args.overfit_n),
        seed=(int(args.seed) if args.seed is not None else None),
        tb_projector=bool(args.tb_projector),
        tb_projector_split=str(args.tb_projector_split),
        tb_projector_max_images=int(args.tb_projector_max_images),
        tb_projector_max_patches=int(args.tb_projector_max_patches),
        tb_projector_per_image_per_keypoint=int(
            args.tb_projector_per_image_per_keypoint
        ),
        tb_projector_pos_threshold=float(args.tb_projector_pos_threshold),
        tb_projector_crop_px=int(args.tb_projector_crop_px),
        tb_projector_sprite_border_px=int(args.tb_projector_sprite_border_px),
        tb_projector_add_negatives=bool(args.tb_projector_add_negatives),
        tb_projector_neg_threshold=float(args.tb_projector_neg_threshold),
        tb_projector_negatives_per_image=int(args.tb_projector_negatives_per_image),
        augment=DinoKPSEGAugmentConfig(
            enabled=bool(args.augment),
            hflip_prob=float(args.hflip),
            degrees=float(args.degrees),
            translate=float(args.translate),
            scale=float(args.scale),
            brightness=float(args.brightness),
            contrast=float(args.contrast),
            saturation=float(args.saturation),
            seed=(int(args.seed) if args.seed is not None else None),
        ),
        aug_start_epoch=int(args.aug_start_epoch),
        aug_stop_epoch=int(args.aug_stop_epoch),
        no_aug_epoch=int(args.no_aug_epoch),
    )
    logger.info("Training complete. Best checkpoint: %s", best)
    logger.info("Run directory: %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
