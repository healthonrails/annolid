from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from annolid.segmentation.dino_kpseg.data import (
    DinoKPSEGAugmentConfig,
    DinoKPSEGPoseDataset,
    build_extractor,
    load_yolo_pose_spec,
    summarize_yolo_pose_labels,
)
from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGCheckpointMeta,
    DinoKPSEGHead,
    DinoKPSEGAttentionHead,
    DinoKPSEGHybridHead,
    checkpoint_pack,
)
from annolid.segmentation.dino_kpseg.cli_utils import normalize_device, parse_layers
from annolid.segmentation.dino_kpseg.keypoints import (
    infer_left_right_pairs,
    infer_orientation_anchor_indices,
    symmetric_pairs_from_flip_idx,
)
from annolid.utils.runs import new_run_dir, shared_runs_root
from annolid.utils.logger import logger


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
    grid = torch.full((c, grid_h, grid_w), float(
        pad_value), dtype=images.dtype)
    for idx in range(n):
        r = idx // nrow
        col = idx % nrow
        y0 = r * (h + pad)
        x0 = col * (w + pad)
        grid[:, y0:y0 + h, x0:x0 + w] = images[idx]
    return grid


def _draw_loss_curve_image(csv_path: Path, *, width: int = 720, height: int = 420) -> Optional[torch.Tensor]:
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
        try:
            va_v = float(va)
        except Exception:
            va_v = None
        if tr_v is None and va_v is None:
            continue
        xs.append(epoch)
        train_ys.append(tr_v)
        val_ys.append(va_v)

    if not xs:
        return None

    all_vals = [v for v in train_ys + val_ys if v is not None]
    if not all_vals:
        return None
    y_min = min(all_vals)
    y_max = max(all_vals)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    try:
        from PIL import Image, ImageDraw, ImageFont
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
    tens = torch.from_numpy(arr).permute(
        2, 0, 1).to(dtype=torch.float32) / 255.0
    return tens


def _compute_resize_hw(*, width: int, height: int, short_side: int, patch_size: int) -> tuple[int, int]:
    if height <= width:
        scale = float(short_side) / max(1, int(height))
    else:
        scale = float(short_side) / max(1, int(width))
    new_w = max(int(patch_size), int(
        ((width * scale) + patch_size - 1) // patch_size) * int(patch_size))
    new_h = max(int(patch_size), int(
        ((height * scale) + patch_size - 1) // patch_size) * int(patch_size))
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
        grid = _grid_images(imgs, nrow=min(
            4, int(imgs.shape[0])), pad=2, pad_value=0.0)
        tb_writer.add_image(tag, grid, 0)
    except Exception:
        for idx, img in enumerate(tensors):
            tb_writer.add_image(f"{tag}/{idx}", img, 0)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    *,
    data_yaml: Path,
    output_dir: Path,
    model_name: str,
    short_side: int,
    layers: Tuple[int, ...],
    radius_px: float,
    hidden_dim: int,
    lr: float,
    epochs: int,
    threshold: float,
    device: Optional[str] = None,
    cache_features: bool = True,
    augment: Optional[DinoKPSEGAugmentConfig] = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    early_stop_min_epochs: int = 0,
    tb_add_graph: bool = False,
    head_type: str = "conv",
    attn_heads: int = 4,
    attn_layers: int = 1,
    lr_pair_loss_weight: float = 0.0,
    lr_pair_margin_px: float = 0.0,
    lr_side_loss_weight: float = 0.0,
    lr_side_loss_margin: float = 0.0,
) -> Path:
    spec = load_yolo_pose_spec(data_yaml)
    if not spec.train_images:
        raise ValueError("No training images found")

    summary = summarize_yolo_pose_labels(
        spec.train_images,
        kpt_count=spec.kpt_count,
        kpt_dims=spec.kpt_dims,
        max_issues=8,
    )
    if summary.images_with_pose_instances <= 0:
        details = "\n".join(
            summary.example_issues) if summary.example_issues else "(no details)"
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
    logger.info("Training DinoKPSEG on %s with device=%s",
                data_yaml, device_str)

    augment_cfg = augment or DinoKPSEGAugmentConfig(enabled=False)
    if augment_cfg.enabled and cache_features:
        logger.warning(
            "DinoKPSEG augmentations enabled; disabling feature caching.")
        cache_features = False

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
            f"{model_name}|{short_side}|{layers}".encode(
                "utf-8", errors="ignore")
        ).hexdigest()[:12]
        cache_dir = cache_root / cache_fingerprint
        _ensure_dir(cache_dir)

    train_ds = DinoKPSEGPoseDataset(
        spec.train_images,
        kpt_count=spec.kpt_count,
        kpt_dims=spec.kpt_dims,
        radius_px=radius_px,
        extractor=extractor,
        flip_idx=spec.flip_idx,
        augment=augment_cfg,
        cache_dir=cache_dir,
        return_images=True,
    )
    val_ds = (
        DinoKPSEGPoseDataset(
            spec.val_images,
            kpt_count=spec.kpt_count,
            kpt_dims=spec.kpt_dims,
            radius_px=radius_px,
            extractor=extractor,
            flip_idx=spec.flip_idx,
            augment=DinoKPSEGAugmentConfig(enabled=False),
            cache_dir=cache_dir,
            return_images=True,
        )
        if spec.val_images
        else None
    )

    sample = train_ds[0]
    feats = sample["feats"]
    in_dim = int(feats.shape[0])

    head_type_norm = str(head_type or "conv").strip().lower()
    if head_type_norm not in ("conv", "attn", "hybrid"):
        raise ValueError(
            f"Unsupported head_type: {head_type!r} (expected 'conv', 'attn', or 'hybrid')"
        )

    orientation_anchor_idx = (
        infer_orientation_anchor_indices(spec.keypoint_names or [])
        if spec.keypoint_names
        else []
    )
    if orientation_anchor_idx:
        logger.info("DinoKPSEG orientation anchors (idx): %s",
                    orientation_anchor_idx)

    if head_type_norm == "attn":
        head = DinoKPSEGAttentionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_parts=spec.kpt_count,
            num_heads=int(attn_heads),
            num_layers=int(attn_layers),
            orientation_anchor_idx=orientation_anchor_idx,
        ).to(device_str)
    elif head_type_norm == "hybrid":
        head = DinoKPSEGHybridHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_parts=spec.kpt_count,
            num_heads=int(attn_heads),
            num_layers=int(attn_layers),
            orientation_anchor_idx=orientation_anchor_idx,
        ).to(device_str)
    else:
        head = DinoKPSEGHead(in_dim=in_dim, hidden_dim=hidden_dim,
                             num_parts=spec.kpt_count).to(device_str)

    opt = torch.optim.AdamW(head.parameters(), lr=float(lr))
    loss_fn = nn.BCEWithLogitsLoss()

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
        num_parts=int(spec.kpt_count),
        radius_px=float(radius_px),
        threshold=float(threshold),
        in_dim=in_dim,
        hidden_dim=int(hidden_dim),
        keypoint_names=spec.keypoint_names,
        flip_idx=spec.flip_idx,
        head_type=head_type_norm,
        attn_heads=int(attn_heads),
        attn_layers=int(attn_layers),
        orientation_anchor_idx=orientation_anchor_idx if orientation_anchor_idx else None,
    )

    # Best-effort args.yaml compatible with existing YOLO training artifacts.
    args_text = "\n".join(
        [
            f"mode: train",
            f"task: dino_kpseg",
            f"data: {str(data_yaml)}",
            f"model_name: {model_name}",
            f"short_side: {short_side}",
            f"layers: {list(layers)}",
            f"radius_px: {radius_px}",
            f"head_type: {head_type_norm}",
            f"attn_heads: {int(attn_heads)}",
            f"attn_layers: {int(attn_layers)}",
            f"hidden_dim: {hidden_dim}",
            f"lr0: {lr}",
            f"epochs: {epochs}",
            f"threshold: {threshold}",
            f"early_stop_patience: {int(early_stop_patience)}",
            f"early_stop_min_delta: {float(early_stop_min_delta)}",
            f"early_stop_min_epochs: {int(early_stop_min_epochs)}",
            f"tb_add_graph: {bool(tb_add_graph)}",
            f"lr_pair_loss_weight: {float(lr_pair_loss_weight)}",
            f"lr_pair_margin_px: {float(lr_pair_margin_px)}",
            f"lr_side_loss_weight: {float(lr_side_loss_weight)}",
            f"lr_side_loss_margin: {float(lr_side_loss_margin)}",
            f"augment: {bool(augment_cfg.enabled)}",
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

    symmetric_pairs = (
        symmetric_pairs_from_flip_idx(spec.flip_idx) if spec.flip_idx else []
    )
    lr_pairs = (
        infer_left_right_pairs(spec.keypoint_names or [],
                               flip_idx=spec.flip_idx)
        if spec.keypoint_names and spec.flip_idx
        else []
    )
    if symmetric_pairs:
        logger.info("DinoKPSEG symmetric keypoint pairs: %s", symmetric_pairs)
    if lr_pairs:
        logger.info("DinoKPSEG left/right keypoint pairs: %s", lr_pairs)

    # NOTE: variable feature grid sizes mean batching > 1 requires padding; keep it simple.
    train_loader = DataLoader(train_ds, batch_size=1,
                              shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_ds, batch_size=1, shuffle=False,
                   num_workers=0) if val_ds is not None else None
    )

    best_loss = float("inf")
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    patience = max(0, int(early_stop_patience))
    min_epochs = max(0, int(early_stop_min_epochs))
    min_delta = float(early_stop_min_delta)
    if min_delta < 0:
        min_delta = 0.0
    early_stop_enabled = patience > 0
    best_metric_for_stop = float("inf")
    bad_epochs = 0

    tb_writer = SummaryWriter(str(tb_dir))
    try:
        tb_writer.add_text("config/data_yaml", str(data_yaml), 0)
        tb_writer.add_text("config/model_name", str(model_name), 0)
        tb_writer.add_text("config/layers", str(list(layers)), 0)
        tb_writer.add_text("config/device", str(device_str), 0)
        tb_writer.add_text("config/short_side", str(short_side), 0)
        tb_writer.add_text("config/patch_size",
                           str(int(extractor.patch_size)), 0)
        tb_writer.add_text("config/early_stop_patience", str(patience), 0)
        tb_writer.add_text("config/early_stop_min_delta", str(min_delta), 0)
        tb_writer.add_text("config/early_stop_min_epochs", str(min_epochs), 0)
        tb_writer.add_text("config/tb_add_graph", str(bool(tb_add_graph)), 0)
        tb_writer.add_text("config/head_type", str(head_type_norm), 0)
        tb_writer.add_text("config/attn_heads", str(int(attn_heads)), 0)
        tb_writer.add_text("config/attn_layers", str(int(attn_layers)), 0)
        tb_writer.add_text(
            "config/orientation_anchor_idx",
            str(orientation_anchor_idx),
            0,
        )
        tb_writer.add_text("config/lr_pair_loss_weight",
                           str(float(lr_pair_loss_weight)), 0)
        tb_writer.add_text("config/lr_pair_margin_px",
                           str(float(lr_pair_margin_px)), 0)
        tb_writer.add_text("config/lr_side_loss_weight",
                           str(float(lr_side_loss_weight)), 0)
        tb_writer.add_text("config/lr_side_loss_margin",
                           str(float(lr_side_loss_margin)), 0)
        tb_writer.add_text("model/architecture", f"```\n{head}\n```", 0)
        try:
            total_params = int(sum(p.numel() for p in head.parameters()))
            trainable_params = int(sum(p.numel()
                                   for p in head.parameters() if p.requires_grad))
            tb_writer.add_scalar("model/params_total", total_params, 0)
            tb_writer.add_scalar("model/params_trainable", trainable_params, 0)
        except Exception:
            pass
        # Optional: export the computation graph (can be expensive / crash-prone on some builds).
        if bool(tb_add_graph) or os.environ.get("ANNOLID_TB_ADD_GRAPH", "").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                if head_type_norm == "attn":
                    cpu_head = DinoKPSEGAttentionHead(
                        in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        num_parts=spec.kpt_count,
                        num_heads=int(attn_heads),
                        num_layers=int(attn_layers),
                        orientation_anchor_idx=orientation_anchor_idx,
                    ).cpu().eval()
                elif head_type_norm == "hybrid":
                    cpu_head = DinoKPSEGHybridHead(
                        in_dim=in_dim,
                        hidden_dim=hidden_dim,
                        num_parts=spec.kpt_count,
                        num_heads=int(attn_heads),
                        num_layers=int(attn_layers),
                        orientation_anchor_idx=orientation_anchor_idx,
                    ).cpu().eval()
                else:
                    cpu_head = DinoKPSEGHead(
                        in_dim=in_dim, hidden_dim=hidden_dim, num_parts=spec.kpt_count).cpu().eval()
                state = {k: v.detach().cpu()
                         for k, v in head.state_dict().items()}
                cpu_head.load_state_dict(state, strict=True)
                dummy = torch.zeros((1, int(in_dim), 2, 2),
                                    dtype=torch.float32)
                tb_writer.add_graph(cpu_head, dummy)
                tb_writer.add_text(
                    "model/graph", "Graph exported via SummaryWriter.add_graph()", 0)
            except Exception as exc:
                tb_writer.add_text("model/graph_error",
                                   f"{type(exc).__name__}: {exc}", 0)

        _log_example_images(
            tb_writer,
            tag="samples/train_images",
            image_paths=list(spec.train_images),
            short_side=int(short_side),
            patch_size=int(extractor.patch_size),
        )
        _log_example_images(
            tb_writer,
            tag="samples/val_images",
            image_paths=list(spec.val_images),
            short_side=int(short_side),
            patch_size=int(extractor.patch_size),
        )

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["epoch", "train_loss", "val_loss", "seconds"],
            )
            writer.writeheader()

            for epoch in range(1, int(epochs) + 1):
                head.train()
                t0 = time.time()
                train_loss = 0.0
                train_loss_total = 0.0
                n_train = 0
                batch_dice_sum = 0.0
                batch_dice_count = 0
                pair_overlap_sum = 0.0
                pair_margin_sum = 0.0
                pair_terms_count = 0
                side_loss_sum = 0.0
                side_terms_count = 0
                for batch in train_loader:
                    feats = batch["feats"][0].to(
                        device_str, non_blocking=True)  # CHW
                    masks = batch["masks"][0].to(
                        device_str, non_blocking=True)  # KHW
                    logits = head(feats.unsqueeze(0))[0]
                    loss_base = loss_fn(logits, masks)
                    loss = loss_base

                    pair_overlap = None
                    pair_margin = None
                    side_loss = None
                    if (
                        float(lr_pair_loss_weight) > 0.0
                        and symmetric_pairs
                        and int(logits.shape[0]) == int(spec.kpt_count)
                    ):
                        k, h_p, w_p = logits.shape
                        probs = torch.sigmoid(logits).reshape(k, -1)
                        norm = probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        dist = probs / norm  # [K, HW]

                        overlap_acc = 0.0
                        margin_acc = 0.0
                        for i, j in symmetric_pairs:
                            overlap_acc = overlap_acc + \
                                (dist[int(i)] * dist[int(j)]).sum()
                            if float(lr_pair_margin_px) > 0.0:
                                patch = float(extractor.patch_size)
                                xs = (torch.arange(w_p, device=logits.device,
                                      dtype=logits.dtype) + 0.5) * patch
                                ys = (torch.arange(h_p, device=logits.device,
                                      dtype=logits.dtype) + 0.5) * patch

                                pi = dist[int(i)].view(h_p, w_p)
                                pj = dist[int(j)].view(h_p, w_p)
                                mux_i = (pi.sum(dim=0) * xs).sum()
                                muy_i = (pi.sum(dim=1) * ys).sum()
                                mux_j = (pj.sum(dim=0) * xs).sum()
                                muy_j = (pj.sum(dim=1) * ys).sum()
                                d = torch.sqrt((mux_i - mux_j) **
                                               2 + (muy_i - muy_j) ** 2 + 1e-8)
                                margin_acc = margin_acc + \
                                    F.relu(float(lr_pair_margin_px) - d)

                        denom_pairs = float(max(1, len(symmetric_pairs)))
                        pair_overlap = overlap_acc / denom_pairs
                        pair_margin = margin_acc / denom_pairs
                        loss = loss + float(lr_pair_loss_weight) * \
                            (pair_overlap + pair_margin)

                    if (
                        float(lr_side_loss_weight) > 0.0
                        and lr_pairs
                        and len(orientation_anchor_idx) >= 2
                        and int(logits.shape[0]) == int(spec.kpt_count)
                    ):
                        k, h_p, w_p = logits.shape
                        probs = torch.sigmoid(logits).reshape(k, -1)
                        norm = probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
                        dist = (probs / norm).view(k, h_p, w_p)

                        xs = torch.arange(
                            w_p, device=logits.device, dtype=logits.dtype) + 0.5
                        ys = torch.arange(
                            h_p, device=logits.device, dtype=logits.dtype) + 0.5
                        px = (dist.sum(dim=1) * xs[None, :]).sum(dim=1)  # [K]
                        py = (dist.sum(dim=2) * ys[None, :]).sum(dim=1)  # [K]

                        a0 = int(orientation_anchor_idx[0])
                        a1 = int(orientation_anchor_idx[1])
                        ax = px[a1] - px[a0]
                        ay = py[a1] - py[a0]
                        an = torch.sqrt(ax * ax + ay * ay + 1e-8)

                        margin = float(lr_side_loss_margin)
                        loss_acc = 0.0
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
                            # Encourage left/right to fall on opposite sides of the axis:
                            # s_l * s_r should be <= -margin (margin in [0,1]).
                            loss_acc = loss_acc + F.relu(s_l * s_r + margin)
                        denom_lr = float(max(1, len(lr_pairs)))
                        side_loss = loss_acc / denom_lr
                        loss = loss + float(lr_side_loss_weight) * side_loss

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    train_loss += float(loss_base.detach().cpu().item())
                    train_loss_total += float(loss.detach().cpu().item())
                    if pair_overlap is not None:
                        pair_overlap_sum += float(
                            pair_overlap.detach().cpu().item())
                        pair_margin_sum += float(pair_margin.detach().cpu().item()
                                                 ) if pair_margin is not None else 0.0
                        pair_terms_count += 1
                    if side_loss is not None:
                        side_loss_sum += float(side_loss.detach().cpu().item())
                        side_terms_count += 1
                    n_train += 1

                    # Lightweight "YOLO-like" qualitative logging: visualize GT vs prediction
                    # for the first batch of each epoch, plus basic mask metrics.
                    if n_train == 1:
                        with torch.no_grad():
                            probs = torch.sigmoid(logits).detach()
                            preds = (probs >= 0.5).to(dtype=masks.dtype)
                            inter = (preds * masks).sum(dim=(1, 2))
                            denom = preds.sum(dim=(1, 2)) + \
                                masks.sum(dim=(1, 2))
                            dice = (2.0 * inter) / torch.clamp(denom, min=1e-8)
                            tb_writer.add_scalar(
                                "metrics/dice_mean_train_batch0", float(dice.mean().item()), epoch)
                            tb_writer.add_scalar(
                                "metrics/dice_min_train_batch0", float(dice.min().item()), epoch)
                            tb_writer.add_scalar(
                                "metrics/dice_max_train_batch0", float(dice.max().item()), epoch)
                            batch_dice_sum += float(dice.mean().item())
                            batch_dice_count += 1

                            # Histograms (bounded to first batch per epoch)
                            try:
                                tb_writer.add_histogram(
                                    "debug/logits", logits.detach().float().cpu(), epoch)
                                tb_writer.add_histogram(
                                    "debug/probs", probs.detach().float().cpu(), epoch)
                            except Exception:
                                pass

                            # Images: keypoint heatmaps at patch resolution.
                            k = int(min(8, int(masks.shape[0])))
                            try:
                                imgs = torch.cat(
                                    [masks[:k], probs[:k]], dim=0).detach().float().cpu()
                                imgs = imgs.unsqueeze(1).clamp(
                                    0.0, 1.0)  # (2k,1,H,W)
                                grid = _grid_images(
                                    imgs, nrow=k, pad=2, pad_value=0.0)
                                tb_writer.add_image(
                                    "qual/gt_then_pred_batch0", grid, epoch)
                            except Exception:
                                pass

                            # Example input image + overlays (if available from the dataset).
                            image_batch = batch.get("image") if isinstance(
                                batch, dict) else None
                            if isinstance(image_batch, torch.Tensor) and image_batch.ndim == 4:
                                img0 = image_batch[0].detach().float(
                                ).cpu().clamp(0.0, 1.0)  # 3HW
                                tb_writer.add_image(
                                    "qual/image_batch0", img0, epoch)

                                prob_mean = probs.detach().float().cpu().mean(dim=0)  # HW
                                pmin = float(prob_mean.min().item())
                                pmax = float(prob_mean.max().item())
                                prob_norm = (prob_mean - pmin) / \
                                    max(pmax - pmin, 1e-6)

                                gt_sum = masks.detach().float().cpu().sum(dim=0)
                                gmin = float(gt_sum.min().item())
                                gmax = float(gt_sum.max().item())
                                gt_norm = (gt_sum - gmin) / \
                                    max(gmax - gmin, 1e-6)

                                tb_writer.add_image(
                                    "qual/pred_mean_batch0", prob_norm.unsqueeze(0), epoch)
                                tb_writer.add_image(
                                    "qual/gt_sum_batch0", gt_norm.unsqueeze(0), epoch)

                                # Upsample patch-grid maps to image resolution for overlays.
                                try:
                                    target_hw = (
                                        int(img0.shape[1]), int(img0.shape[2]))
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
                                    tb_writer.add_image(
                                        "qual/pred_mean_up_batch0", prob_up.unsqueeze(0), epoch)
                                    tb_writer.add_image(
                                        "qual/gt_sum_up_batch0", gt_up.unsqueeze(0), epoch)
                                except Exception:
                                    prob_up = prob_norm

                                overlay = img0.clone()
                                alpha = (0.55 * prob_up).clamp(0.0, 0.55)  # HW
                                overlay[0] = overlay[0] * \
                                    (1.0 - alpha) + alpha * 1.0
                                overlay[1] = overlay[1] * (1.0 - alpha)
                                overlay[2] = overlay[2] * (1.0 - alpha)
                                tb_writer.add_image(
                                    "qual/overlay_pred_mean_batch0", overlay.clamp(0.0, 1.0), epoch)

                train_loss /= max(1, n_train)
                train_loss_total /= max(1, n_train)

                val_loss = None
                if val_loader is not None:
                    head.eval()
                    losses = []
                    tp = torch.zeros(int(spec.kpt_count),
                                     device=device_str, dtype=torch.float32)
                    fp = torch.zeros(int(spec.kpt_count),
                                     device=device_str, dtype=torch.float32)
                    fn = torch.zeros(int(spec.kpt_count),
                                     device=device_str, dtype=torch.float32)
                    peak_dist_sum = torch.zeros(
                        int(spec.kpt_count), device=device_str, dtype=torch.float32)
                    peak_dist_count = torch.zeros(
                        int(spec.kpt_count), device=device_str, dtype=torch.float32)
                    with torch.no_grad():
                        for batch in val_loader:
                            feats = batch["feats"][0].to(
                                device_str, non_blocking=True)
                            masks = batch["masks"][0].to(
                                device_str, non_blocking=True)
                            logits = head(feats.unsqueeze(0))[0]
                            losses.append(
                                float(loss_fn(logits, masks).cpu().item()))

                            probs = torch.sigmoid(logits)
                            gt = masks > 0.5
                            pred = probs >= float(threshold)
                            tp += (pred & gt).sum(dim=(1, 2)
                                                  ).to(dtype=torch.float32)
                            fp += (pred & ~gt).sum(dim=(1, 2)
                                                   ).to(dtype=torch.float32)
                            fn += (~pred & gt).sum(dim=(1, 2)
                                                   ).to(dtype=torch.float32)

                            # Peak localization error (patch-grid -> pixels via patch_size)
                            k, h_p, w_p = probs.shape
                            flat_gt = masks.view(k, -1)
                            valid = (flat_gt.sum(dim=1) > 0).to(
                                dtype=torch.float32)
                            pred_idx = probs.view(k, -1).argmax(dim=1)
                            gt_idx = flat_gt.argmax(dim=1)
                            pred_y = (pred_idx // w_p).to(dtype=torch.float32)
                            pred_x = (pred_idx % w_p).to(dtype=torch.float32)
                            gt_y = (gt_idx // w_p).to(dtype=torch.float32)
                            gt_x = (gt_idx % w_p).to(dtype=torch.float32)
                            dist = (
                                torch.sqrt((pred_y - gt_y) ** 2 +
                                           (pred_x - gt_x) ** 2)
                                * float(extractor.patch_size)
                            )
                            peak_dist_sum += dist * valid
                            peak_dist_count += valid
                    if losses:
                        val_loss = float(sum(losses) / len(losses))
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
                        precision_mean = float(
                            precision[valid_kpt].mean().item())
                        recall_mean = float(recall[valid_kpt].mean().item())
                        peak_dist_px = (
                            peak_dist_sum / peak_dist_count.clamp(min=1.0))[valid_kpt]
                        peak_dist_px_mean = float(peak_dist_px.mean().item())
                    else:
                        dice_mean = float(dice.mean().item())
                        iou_mean = float(iou.mean().item())
                        precision_mean = float(precision.mean().item())
                        recall_mean = float(recall.mean().item())
                        peak_dist_px_mean = float(
                            (peak_dist_sum / peak_dist_count.clamp(min=1.0)).mean().item()
                        )

                elapsed = float(time.time() - t0)
                row = {
                    "epoch": epoch,
                    "train_loss": f"{train_loss_total:.6f}",
                    "val_loss": f"{val_loss:.6f}" if val_loss is not None else "",
                    "seconds": f"{elapsed:.2f}",
                }
                writer.writerow(row)
                fh.flush()

                tb_writer.add_scalar("loss/train_base", train_loss, epoch)
                tb_writer.add_scalar("loss/train", train_loss_total, epoch)
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
                    tb_writer.add_scalar(
                        "val/precision_mean", precision_mean, epoch)
                    tb_writer.add_scalar("val/recall_mean", recall_mean, epoch)
                    tb_writer.add_scalar(
                        "val/peak_dist_px_mean", peak_dist_px_mean, epoch)
                    tb_writer.add_scalar(
                        "val/keypoints_present_frac",
                        float(valid_kpt.float().mean().item()),
                        epoch,
                    )
                    # Per-keypoint metrics
                    names = spec.keypoint_names or []
                    for i in range(int(spec.kpt_count)):
                        name = names[i] if i < len(names) else f"kp_{i}"
                        tb_writer.add_scalar(
                            f"val/dice/{name}", float(dice[i].item()), epoch)
                        tb_writer.add_scalar(
                            f"val/iou/{name}", float(iou[i].item()), epoch)
                tb_writer.add_scalar("time/epoch_seconds", elapsed, epoch)
                if batch_dice_count:
                    tb_writer.add_scalar(
                        "metrics/dice_mean_train_epoch0",
                        batch_dice_sum / max(1, batch_dice_count),
                        epoch,
                    )

                payload = checkpoint_pack(head=head, meta=meta)
                torch.save(payload, last_path)
                metric = val_loss if val_loss is not None else train_loss_total
                if metric < best_loss:
                    best_loss = metric
                    torch.save(payload, best_path)

                tb_writer.add_scalar(
                    "checkpoint/best_metric", float(best_loss), epoch)

                if early_stop_enabled and math.isfinite(float(metric)):
                    improved = float(metric) < (
                        float(best_metric_for_stop) - float(min_delta))
                    if improved:
                        best_metric_for_stop = float(metric)
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                    tb_writer.add_scalar(
                        "early_stop/current_metric", float(metric), epoch)
                    tb_writer.add_scalar(
                        "early_stop/best_metric", float(best_metric_for_stop), epoch)
                    tb_writer.add_scalar(
                        "early_stop/bad_epochs", int(bad_epochs), epoch)

                    if int(epoch) >= int(min_epochs) and int(bad_epochs) >= int(patience):
                        reason = (
                            f"Early stopping triggered at epoch {epoch}: "
                            f"metric={float(metric):.6f} best={float(best_metric_for_stop):.6f} "
                            f"min_delta={float(min_delta)} patience={int(patience)}"
                        )
                        logger.info(reason)
                        tb_writer.add_text(
                            "early_stop/stop_reason", reason, epoch)
                        tb_writer.flush()
                        break

                logger.info(
                    "Epoch %d/%d - train_loss=%.6f val_loss=%s time=%.2fs",
                    epoch,
                    epochs,
                    train_loss,
                    f"{val_loss:.6f}" if val_loss is not None else "NA",
                    elapsed,
                )
                tb_writer.flush()

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
            tb_writer.add_image("plots/loss_curve", curve, int(epochs))
    finally:
        tb_writer.flush()
        tb_writer.close()

    return best_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Train a DINOv3 keypoint mask segmentation head.")
    p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    p.add_argument(
        "--output",
        default=None,
        help="Run output directory (if omitted, creates a new run under ANNOLID_RUNS_ROOT/~/annolid_logs/runs).",
    )
    p.add_argument("--runs-root", default=None,
                   help="Runs root (overrides ANNOLID_RUNS_ROOT/~/annolid_logs/runs)")
    p.add_argument("--run-name", default=None,
                   help="Optional run name (default: timestamp)")
    p.add_argument(
        "--model-name",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id or dinov3 alias",
    )
    p.add_argument("--short-side", type=int, default=768)
    p.add_argument("--layers", type=str, default="-1",
                   help="Comma-separated transformer block indices")
    p.add_argument("--radius-px", type=float, default=6.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--device", default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable feature caching")
    p.add_argument("--head-type", choices=("conv", "attn", "hybrid"),
                   default="conv", help="Head architecture")
    p.add_argument("--attn-heads", type=int, default=4,
                   help="Attention heads (attn head only)")
    p.add_argument("--attn-layers", type=int, default=1,
                   help="Attention layers (attn head only)")
    p.add_argument(
        "--lr-pair-loss-weight",
        type=float,
        default=0.0,
        help="Optional symmetric-pair regularizer weight (0=off).",
    )
    p.add_argument(
        "--lr-pair-margin-px",
        type=float,
        default=0.0,
        help="Optional minimum separation margin in pixels for symmetric pairs (0=off).",
    )
    p.add_argument(
        "--lr-side-loss-weight",
        type=float,
        default=0.0,
        help="Optional left/right side-consistency loss weight (0=off). Uses orientation anchors when available.",
    )
    p.add_argument(
        "--lr-side-loss-margin",
        type=float,
        default=0.0,
        help="Margin for side-consistency in [0,1] (0=enforce opposite sign).",
    )
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Early stop patience (0=off)")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="Min metric improvement to reset patience")
    p.add_argument("--early-stop-min-epochs", type=int, default=0,
                   help="Do not early-stop before this epoch")
    p.add_argument("--tb-add-graph", action="store_true",
                   help="Export model graph to TensorBoard (can be slow)")
    p.add_argument("--augment", action="store_true",
                   help="Enable YOLO-like pose augmentations")
    p.add_argument("--hflip", type=float, default=0.5,
                   help="Horizontal flip probability")
    p.add_argument("--degrees", type=float, default=0.0,
                   help="Random rotation degrees (+/-)")
    p.add_argument("--translate", type=float, default=0.0,
                   help="Random translate fraction (+/-)")
    p.add_argument("--scale", type=float, default=0.0,
                   help="Random scale fraction (+/-)")
    p.add_argument("--brightness", type=float, default=0.0,
                   help="Brightness jitter (+/-)")
    p.add_argument("--contrast", type=float, default=0.0,
                   help="Contrast jitter (+/-)")
    p.add_argument("--saturation", type=float, default=0.0,
                   help="Saturation jitter (+/-)")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional augmentation RNG seed")
    args = p.parse_args(argv)

    layers = parse_layers(args.layers)

    if args.output:
        out_dir = Path(args.output).expanduser().resolve()
    else:
        runs_root = (
            Path(args.runs_root).expanduser().resolve()
            if args.runs_root
            else shared_runs_root()
        )
        out_dir = new_run_dir(task="dino_kpseg", model="train",
                              runs_root=runs_root, run_name=args.run_name)
    _ensure_dir(out_dir)

    best = train(
        data_yaml=Path(args.data).expanduser().resolve(),
        output_dir=out_dir,
        model_name=str(args.model_name),
        short_side=int(args.short_side),
        layers=layers,
        radius_px=float(args.radius_px),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        epochs=int(args.epochs),
        threshold=float(args.threshold),
        device=args.device,
        cache_features=not bool(args.no_cache),
        head_type=str(args.head_type),
        attn_heads=int(args.attn_heads),
        attn_layers=int(args.attn_layers),
        lr_pair_loss_weight=float(args.lr_pair_loss_weight),
        lr_pair_margin_px=float(args.lr_pair_margin_px),
        lr_side_loss_weight=float(args.lr_side_loss_weight),
        lr_side_loss_margin=float(args.lr_side_loss_margin),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_min_epochs=int(args.early_stop_min_epochs),
        tb_add_graph=bool(args.tb_add_graph),
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
    )
    logger.info("Training complete. Best checkpoint: %s", best)
    logger.info("Run directory: %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
