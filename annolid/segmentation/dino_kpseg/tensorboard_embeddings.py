from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from annolid.segmentation.dino_kpseg.cli_utils import (
    normalize_device,
    parse_layers,
)
from annolid.segmentation.dino_kpseg.data import (
    DinoKPSEGAugmentConfig,
    DinoKPSEGPoseDataset,
    build_extractor,
    load_yolo_pose_spec,
)
from annolid.segmentation.dino_kpseg.model import checkpoint_unpack
from annolid.utils.runs import allocate_run_dir, shared_runs_root

# Type alias for callables that operate on a single torch.Tensor
TorchTensorCallable = Callable[
    [torch.Tensor], torch.Tensor
]


@dataclass(frozen=True)
class DinoKPSEGTensorBoardEmbeddingConfig:
    data_yaml: Path
    log_dir: Path
    split: str = "val"
    weights: Optional[Path] = None
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    short_side: int = 768
    layers: Tuple[int, ...] = (-1,)
    device: Optional[str] = None
    radius_px: float = 6.0
    mask_type: str = "gaussian"
    heatmap_sigma_px: Optional[float] = None
    instance_mode: str = "auto"
    bbox_scale: float = 1.25
    cache_features: bool = True
    max_images: int = 64
    max_patches: int = 4000
    per_image_per_keypoint: int = 3
    pos_threshold: float = 0.35
    add_negatives: bool = False
    neg_threshold: float = 0.02
    negatives_per_image: int = 6
    crop_px: int = 96
    sprite_border_px: int = 3
    seed: int = 0


def add_dino_kpseg_projector_embeddings(
    tb_writer: SummaryWriter,
    *,
    log_dir: Path,
    split: str,
    ds: DinoKPSEGPoseDataset,
    keypoint_names: Sequence[str],
    max_images: int = 64,
    max_patches: int = 4000,
    per_image_per_keypoint: int = 3,
    pos_threshold: float = 0.35,
    add_negatives: bool = False,
    neg_threshold: float = 0.02,
    negatives_per_image: int = 6,
    crop_px: int = 96,
    sprite_border_px: int = 3,
    seed: int = 0,
    tag: str = "dino_kpseg/patch_embeddings",
    predict_probs_patch: Optional[TorchTensorCallable] = None,
    write_csv: bool = True,
    global_step: int = 0,
) -> dict:
    """Write TensorBoard embedding projector data for a DinoKPSEGPoseDataset.

    Notes
    -----
        - Uses DINO patch features (CHW) and patch-grid masks (KHW).
        - Uses dataset-provided resized images to generate sprite thumbnails.
        - If `predict_probs_patch` is provided, adds an overlay image where
            red=pred union.
    """
    rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed))
    split_norm = str(split or "val").strip().lower()

    _ext = getattr(ds, "extractor", None)
    patch_size = int(getattr(_ext, "patch_size", 16))
    crop_px = max(int(patch_size), int(crop_px))
    border_px = max(0, int(sprite_border_px))
    max_images = max(1, int(max_images))
    max_patches = max(1, int(max_patches))
    per_img_per_kpt = max(0, int(per_image_per_keypoint))

    img_indices = list(range(len(ds)))
    rng.shuffle(img_indices)
    img_indices = img_indices[:max_images]

    embed_rows: List[torch.Tensor] = []
    thumbs: List[torch.Tensor] = []
    metadata_rows: List[List[str]] = []
    overlay_images: List[torch.Tensor] = []

    for ds_idx in img_indices:
        if len(embed_rows) >= max_patches:
            break
        sample = ds[int(ds_idx)]
        feats = sample.get("feats")
        masks = sample.get("masks")
        img = sample.get("image")
        if (
            not isinstance(feats, torch.Tensor)
            or not isinstance(masks, torch.Tensor)
            or not isinstance(img, torch.Tensor)
        ):
            continue
        if feats.ndim != 3 or masks.ndim != 3 or img.ndim != 3:
            continue
        if (
            int(img.shape[1]) != int(masks.shape[1] * patch_size)
            or int(img.shape[2]) != int(masks.shape[2] * patch_size)
        ):
            continue

        probs_patch = None
        if predict_probs_patch is not None:
            try:
                probs_patch = predict_probs_patch(feats)
                if probs_patch.ndim != 3:
                    probs_patch = None
                else:
                    probs_patch = probs_patch.detach().to("cpu")
            except Exception:
                probs_patch = None

        union_gt = masks.max(dim=0).values.clamp(0.0, 1.0)
        union_gt_px = _upsample_patch_map(
            union_gt.to("cpu"), patch_size=patch_size
        )
        overlay = _overlay_heatmap(
            img.to("cpu"),
            union_gt_px,
            color=(0.1, 1.0, 0.1),
            alpha=0.35,
        )
        if probs_patch is not None:
            union_pred = probs_patch.max(dim=0).values.clamp(0.0, 1.0)
            union_pred_px = _upsample_patch_map(
                union_pred.to("cpu"), patch_size=patch_size
            )
            overlay = _overlay_heatmap(
                overlay,
                union_pred_px,
                color=(1.0, 0.1, 0.1),
                alpha=0.35,
            )
        overlay_images.append(overlay)

        feats_cpu = feats.to("cpu", dtype=torch.float32)
        masks_cpu = masks.to("cpu", dtype=torch.float32)
        coords = sample.get("coords")
        coord_mask = sample.get("coord_mask")
        coords_cpu = (
            coords.to("cpu", dtype=torch.float32)
            if isinstance(coords, torch.Tensor)
            else None
        )
        coord_mask_cpu = (
            coord_mask.to("cpu", dtype=torch.float32)
            if isinstance(coord_mask, torch.Tensor)
            else None
        )

        kpt_count = int(masks_cpu.shape[0])
        for k in range(kpt_count):
            if len(embed_rows) >= max_patches:
                break
            mask_k = masks_cpu[k]
            pos = mask_k.numpy()
            pos_idxs = np.argwhere(pos >= float(pos_threshold))
            if pos_idxs.size == 0:
                continue
            scores = pos[pos_idxs[:, 0], pos_idxs[:, 1]]
            chosen = _select_indices(
                scores, max_count=per_img_per_kpt, rng=rng
            )
            color = _hsv_color(k, total=kpt_count)
            kpt_name = str(keypoint_names[k]) if k < len(
                keypoint_names) else f"kpt_{k}"

            for j in chosen:
                if len(embed_rows) >= max_patches:
                    break
                rr = int(pos_idxs[j, 0])
                cc = int(pos_idxs[j, 1])
                score = float(scores[j])

                vec = feats_cpu[:, rr, cc].contiguous()
                embed_rows.append(vec)

                cx = (float(cc) + 0.5) * float(patch_size)
                cy = (float(rr) + 0.5) * float(patch_size)
                x0 = int(math.floor(cx - float(crop_px) / 2.0))
                y0 = int(math.floor(cy - float(crop_px) / 2.0))
                thumb = _pad_crop(
                    img.to("cpu"), y0=y0, x0=x0, h=crop_px, w=crop_px
                )
                thumb = _draw_border(thumb, color=color, px=border_px)
                thumbs.append(thumb)

                image_path = ds.image_paths[int(ds_idx)]
                coord_x = ""
                coord_y = ""
                if coords_cpu is not None and coord_mask_cpu is not None:
                    if int(coord_mask_cpu[k].item()) > 0:
                        coord_x = f"{float(coords_cpu[k, 0].item()):.2f}"
                        coord_y = f"{float(coords_cpu[k, 1].item()):.2f}"

                pred_value = ""
                pred_hit = ""
                if (
                    probs_patch is not None
                    and int(k) < int(probs_patch.shape[0])
                ):
                    pred_v = float(probs_patch[int(k), rr, cc].item())
                    pred_value = f"{pred_v:.4f}"
                    pred_hit = "1" if pred_v >= float(pos_threshold) else "0"

                row = [
                    split_norm,
                    str(image_path),
                    str(int(k)),
                    str(kpt_name),
                    f"{float(score):.4f}",
                    str(int(rr)),
                    str(int(cc)),
                    f"{float(cx):.2f}",
                    f"{float(cy):.2f}",
                    coord_x,
                    coord_y,
                ]
                if predict_probs_patch is not None:
                    row.extend([pred_value, pred_hit])
                metadata_rows.append(row)

    if add_negatives and len(embed_rows) < max_patches:
        union = masks_cpu.max(dim=0).values.numpy()
        neg_idxs = np.argwhere(union <= float(neg_threshold))
        if neg_idxs.size > 0:
            pick_n = min(
                int(negatives_per_image), int(neg_idxs.shape[0]),
                max_patches - len(embed_rows)
            )
            chosen_rows = np_rng.choice(
                int(neg_idxs.shape[0]), size=pick_n, replace=False
            ).tolist()
            for j in chosen_rows:
                if len(embed_rows) >= max_patches:
                    break
                rr = int(neg_idxs[j, 0])
                cc = int(neg_idxs[j, 1])
                vec = feats_cpu[:, rr, cc].contiguous()
                embed_rows.append(vec)
                cx = (float(cc) + 0.5) * float(patch_size)
                cy = (float(rr) + 0.5) * float(patch_size)
                x0 = int(math.floor(cx - float(crop_px) / 2.0))
                y0 = int(math.floor(cy - float(crop_px) / 2.0))
                thumb = _pad_crop(
                    img.to("cpu"), y0=y0, x0=x0, h=crop_px, w=crop_px
                )
                thumbs.append(
                    _draw_border(thumb, color=(0.4, 0.4, 0.4), px=border_px)
                )
                image_path = ds.image_paths[int(ds_idx)]
                pred_value = ""
                pred_hit = ""
                if probs_patch is not None:
                    union_pred = probs_patch.max(dim=0).values
                    pred_v = float(union_pred[int(rr), int(cc)].item())
                    pred_value = f"{pred_v:.4f}"
                    pred_hit = "1" if pred_v >= float(pos_threshold) else "0"
                row = [
                    split_norm,
                    str(image_path),
                    "-1",
                    "background",
                    "0.0",
                    str(int(rr)),
                    str(int(cc)),
                    f"{float(cx):.2f}",
                    f"{float(cy):.2f}",
                    "",
                    "",
                ]
                if predict_probs_patch is not None:
                    row.extend([pred_value, pred_hit])
                metadata_rows.append(row)

    if not embed_rows:
        raise RuntimeError(
            "No patch embeddings collected. Try lowering "
            "pos_threshold or increasing max_images."
        )

    embeddings = torch.stack(embed_rows, dim=0).to(
        "cpu", dtype=torch.float32
    )
    label_img = torch.stack(thumbs, dim=0).to(
        "cpu", dtype=torch.float32
    ).clamp(0.0, 1.0)

    md_header = [
        "split",
        "image_path",
        "kpt_idx",
        "kpt_name",
        "mask_value",
        "patch_row",
        "patch_col",
        "x_resized",
        "y_resized",
        "coord_x_resized",
        "coord_y_resized",
    ]
    if predict_probs_patch is not None:
        md_header.extend(["pred_value", "pred_hit"])
    tb_writer.add_embedding(
        embeddings,
        metadata=metadata_rows,
        metadata_header=md_header,
        label_img=label_img,
        global_step=int(global_step),
        tag=str(tag),
    )

    if overlay_images:
        grid = torch.stack(
            overlay_images[: min(16, len(overlay_images))], dim=0
        )
        tb_writer.add_images(
            f"dino_kpseg/overlays/{split_norm}/gt_green_pred_red",
            grid,
            int(global_step),
        )
        if predict_probs_patch is None:
            tb_writer.add_text(
                f"dino_kpseg/overlays/{split_norm}/info",
                "Green=GT union mask (pred overlay disabled).",
                int(global_step),
            )
        else:
            tb_writer.add_text(
                f"dino_kpseg/overlays/{split_norm}/info",
                "Green=GT union mask, Red=pred union mask.",
                int(global_step),
            )

    if write_csv:
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = (
            Path(log_dir)
            / f"dino_kpseg_patch_embeddings_metadata_{split_norm}.csv"
        )
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(md_header)
            w.writerows(metadata_rows)

    return {
        "split": split_norm,
        "num_points": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "patch_size": int(patch_size),
    }


def _hsv_color(index: int, *, total: int) -> Tuple[float, float, float]:
    if total <= 0:
        total = 1
    h = (float(index) / float(total)) % 1.0
    s = 0.85
    v = 0.95
    i = int(h * 6.0)
    f = (h * 6.0) - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def _pad_crop(
    img: torch.Tensor,
    *,
    y0: int,
    x0: int,
    h: int,
    w: int,
) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError("Expected CHW image tensor")
    _, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
    x1 = x0 + int(w)
    y1 = y0 + int(h)

    src_x0 = max(0, int(x0))
    src_y0 = max(0, int(y0))
    src_x1 = min(int(W), int(x1))
    src_y1 = min(int(H), int(y1))

    crop = img[:, src_y0:src_y1, src_x0:src_x1]
    pad_left = max(0, -int(x0))
    pad_top = max(0, -int(y0))
    pad_right = max(0, int(x1) - int(W))
    pad_bottom = max(0, int(y1) - int(H))

    if pad_left or pad_right or pad_top or pad_bottom:
        crop = F.pad(
            crop,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0.0,
        )
    if int(crop.shape[1]) != int(h) or int(crop.shape[2]) != int(w):
        crop = crop[:, : int(h), : int(w)]
    return crop


def _draw_border(
    img: torch.Tensor,
    *,
    color: Tuple[float, float, float],
    px: int,
) -> torch.Tensor:
    if px <= 0:
        return img
    if img.ndim != 3 or int(img.shape[0]) != 3:
        return img
    px = min(int(px), int(img.shape[1]) // 2, int(img.shape[2]) // 2)
    if px <= 0:
        return img
    out = img.clone()
    r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    out[0, :px, :] = r
    out[1, :px, :] = g
    out[2, :px, :] = b
    out[0, -px:, :] = r
    out[1, -px:, :] = g
    out[2, -px:, :] = b
    out[0, :, :px] = r
    out[1, :, :px] = g
    out[2, :, :px] = b
    out[0, :, -px:] = r
    out[1, :, -px:] = g
    out[2, :, -px:] = b
    return out


def _overlay_heatmap(
    img: torch.Tensor,
    heat: torch.Tensor,
    *,
    color: Tuple[float, float, float] = (1.0, 0.1, 0.1),
    alpha: float = 0.45,
) -> torch.Tensor:
    if img.ndim != 3 or int(img.shape[0]) != 3:
        raise ValueError("Expected CHW RGB image in [0,1]")
    if heat.ndim != 2:
        raise ValueError("Expected HW heatmap")
    heat = heat.clamp(0.0, 1.0)
    if (
        int(heat.shape[0]) != int(img.shape[1])
        or int(heat.shape[1]) != int(img.shape[2])
    ):
        heat = F.interpolate(
            heat[None, None, :, :],
            size=(int(img.shape[1]), int(img.shape[2])),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heat = heat.clamp(0.0, 1.0)
    col = torch.tensor(
        color, dtype=img.dtype, device=img.device
    ).view(3, 1, 1)
    alpha_heat = alpha * heat[None, :, :]
    blended = img * (1.0 - alpha_heat) + col * alpha_heat
    return blended.clamp(0.0, 1.0)


def _upsample_patch_map(
    patch_map: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if patch_map.ndim != 2:
        raise ValueError("Expected patch HW map")
    h_p, w_p = int(patch_map.shape[0]), int(patch_map.shape[1])
    out = patch_map.repeat_interleave(int(patch_size), dim=0)
    out = out.repeat_interleave(int(patch_size), dim=1)
    if (
        int(out.shape[0]) != int(h_p * patch_size)
        or int(out.shape[1]) != int(w_p * patch_size)
    ):
        out = out[: int(h_p * patch_size), : int(w_p * patch_size)]
    return out


def _select_indices(
    scores: np.ndarray,
    *,
    max_count: int,
    rng: random.Random,
) -> List[int]:
    if scores.size == 0 or max_count <= 0:
        return []
    idxs = np.arange(scores.size)
    # Favor strong positives but keep diversity.
    # Use a weighted sample without replacement.
    weights = scores.astype(np.float64, copy=False)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        shuffled = idxs.tolist()
        rng.shuffle(shuffled)
        return shuffled[:max_count]
    weights = weights / float(weights.sum())
    chosen: List[int] = []
    pool = idxs.tolist()
    w = weights.tolist()
    while pool and len(chosen) < int(max_count):
        pick = rng.choices(pool, weights=w, k=1)[0]
        j = pool.index(pick)
        chosen.append(int(pick))
        pool.pop(j)
        w.pop(j)
        s = float(sum(w))
        if s > 0:
            w = [float(x) / s for x in w]
        else:
            break
    return chosen


def write_dino_kpseg_tensorboard_embeddings(
    cfg: DinoKPSEGTensorBoardEmbeddingConfig,
) -> Path:
    spec = load_yolo_pose_spec(Path(cfg.data_yaml))
    split = str(cfg.split or "val").strip().lower()
    if split not in {"train", "val"}:
        raise ValueError("--split must be 'train' or 'val'")
    image_paths = spec.train_images if split == "train" else spec.val_images
    if not image_paths:
        raise ValueError(f"No images found for split={split!r}")

    device_str = normalize_device(cfg.device)

    head = None
    meta = None
    model_name = str(cfg.model_name)
    short_side = int(cfg.short_side)
    layers = tuple(int(x) for x in cfg.layers)
    if cfg.weights is not None:
        payload = torch.load(
            Path(cfg.weights).expanduser().resolve(),
            map_location="cpu",
        )
        head, meta = checkpoint_unpack(payload)
        model_name = str(meta.model_name)
        short_side = int(meta.short_side)
        layers = tuple(int(x) for x in meta.layers)

    extractor = build_extractor(
        model_name=model_name,
        short_side=short_side,
        layers=layers,
        device=device_str,
    )

    cache_dir = None
    if bool(cfg.cache_features):
        cache_root = (
            Path.home()
            / ".cache"
            / "annolid"
            / "dinokpseg"
            / "features"
        )
        cache_fingerprint = f"{model_name}|{short_side}|{layers}"
        digest = hashlib.sha1(
            cache_fingerprint.encode("utf-8")
        ).hexdigest()[:12]
        cache_dir = cache_root / digest
        cache_dir.mkdir(parents=True, exist_ok=True)

    ds = DinoKPSEGPoseDataset(
        list(image_paths),
        kpt_count=spec.kpt_count if meta is None else int(meta.num_parts),
        kpt_dims=spec.kpt_dims,
        radius_px=float(cfg.radius_px),
        extractor=extractor,
        flip_idx=spec.flip_idx if meta is None else (
            meta.flip_idx or spec.flip_idx),
        augment=DinoKPSEGAugmentConfig(enabled=False),
        cache_dir=cache_dir,
        mask_type=str(cfg.mask_type),
        heatmap_sigma_px=cfg.heatmap_sigma_px,
        instance_mode=str(cfg.instance_mode),
        bbox_scale=float(cfg.bbox_scale),
        return_images=True,
    )

    keypoint_names: List[str]
    if meta is not None and meta.keypoint_names:
        keypoint_names = [str(x) for x in meta.keypoint_names]
    elif spec.keypoint_names:
        keypoint_names = [str(x) for x in spec.keypoint_names]
    else:
        keypoint_names = [f"kpt_{i}" for i in range(int(ds.kpt_count))]

    if head is not None:
        head = head.to(device_str)
        head.eval()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(cfg.log_dir))
    try:
        patch_size = int(getattr(extractor, "patch_size", 16))
        writer.add_text("dino_kpseg/data_yaml", str(cfg.data_yaml), 0)
        writer.add_text("dino_kpseg/split", str(split), 0)
        writer.add_text("dino_kpseg/model_name", str(model_name), 0)
        writer.add_text("dino_kpseg/short_side", str(int(short_side)), 0)
        writer.add_text("dino_kpseg/layers", str(list(layers)), 0)
        writer.add_text("dino_kpseg/patch_size", str(int(patch_size)), 0)
        if meta is not None:
            writer.add_text("dino_kpseg/checkpoint", str(cfg.weights), 0)
        predict_fn = None
        if head is not None:
            def _predict_probs(feats: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    x = feats.unsqueeze(0).to(
                        device_str, dtype=torch.float32
                    )
                    logits = head(x)[0].detach().to("cpu")
                    return torch.sigmoid(logits).clamp(0.0, 1.0)

            predict_fn = _predict_probs

        summary = add_dino_kpseg_projector_embeddings(
            writer,
            log_dir=Path(cfg.log_dir),
            split=split,
            ds=ds,
            keypoint_names=keypoint_names,
            max_images=int(cfg.max_images),
            max_patches=int(cfg.max_patches),
            per_image_per_keypoint=int(cfg.per_image_per_keypoint),
            pos_threshold=float(cfg.pos_threshold),
            add_negatives=bool(cfg.add_negatives),
            neg_threshold=float(cfg.neg_threshold),
            negatives_per_image=int(cfg.negatives_per_image),
            crop_px=int(cfg.crop_px),
            sprite_border_px=int(cfg.sprite_border_px),
            seed=int(cfg.seed),
            tag="dino_kpseg/patch_embeddings",
            predict_probs_patch=predict_fn,
            write_csv=True,
        )
        writer.add_scalar("embeddings/num_points",
                          int(summary["num_points"]), 0)
        writer.add_scalar("embeddings/dim", int(summary["dim"]), 0)
    finally:
        writer.flush()
        writer.close()

    return cfg.log_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "TensorBoard projector visualization for DinoKPSEG "
            "(DINOv3 patch embeddings)."
        ),
    )
    p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional DinoKPSEG checkpoint (.pt); enables pred overlays "
            "and keypoint names."
        ),
    )
    p.add_argument(
        "--model-name",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help=(
            "Hugging Face model id or dinov3 alias "
            "(ignored when --weights is set)."
        ),
    )
    p.add_argument(
        "--short-side",
        type=int,
        default=768,
        help=(
            "Resize short side before patch snapping "
            "(ignored when --weights is set)."
        ),
    )
    p.add_argument(
        "--layers",
        type=str,
        default="-1",
        help=(
            "Comma-separated transformer block indices "
            "(ignored when --weights is set)."
        ),
    )
    p.add_argument("--device", default=None,
                   help="cuda|mps|cpu (default: auto)")
    p.add_argument("--radius-px", type=float, default=6.0)
    p.add_argument("--mask-type", choices=("disk",
                   "gaussian"), default="gaussian")
    p.add_argument("--heatmap-sigma", type=float, default=None)
    p.add_argument("--instance-mode", choices=("auto", "union",
                   "per_instance"), default="auto")
    p.add_argument("--bbox-scale", type=float, default=1.25)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable feature caching")
    p.add_argument("--max-images", type=int, default=64)
    p.add_argument("--max-patches", type=int, default=4000)
    p.add_argument("--per-image-per-keypoint", type=int, default=3)
    p.add_argument("--pos-threshold", type=float, default=0.35)
    p.add_argument("--add-negatives", action="store_true",
                   help="Sample some background patches too.")
    p.add_argument("--neg-threshold", type=float, default=0.02)
    p.add_argument("--negatives-per-image", type=int, default=6)
    p.add_argument("--crop-px", type=int, default=96,
                   help="Thumbnail crop size in pixels (resized image space).")
    p.add_argument("--sprite-border-px", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--output", default=None,
                   help="Run output directory (optional)")
    p.add_argument("--runs-root", default=None, help="Runs root (optional)")
    p.add_argument("--run-name", default=None,
                   help="Optional run name (default: timestamp)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.output:
        out_dir = Path(args.output).expanduser().resolve()
    else:
        runs_root = Path(args.runs_root).expanduser(
        ).resolve() if args.runs_root else shared_runs_root()
        out_dir = allocate_run_dir(
            task="dino_kpseg",
            model="embeddings",
            runs_root=runs_root,
            run_name=args.run_name,
        )
    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    log_dir = write_dino_kpseg_tensorboard_embeddings(
        DinoKPSEGTensorBoardEmbeddingConfig(
            data_yaml=Path(args.data).expanduser().resolve(),
            log_dir=tb_dir,
            split=str(args.split),
            weights=(Path(args.weights).expanduser().resolve()
                     if args.weights else None),
            model_name=str(args.model_name),
            short_side=int(args.short_side),
            layers=layers,
            device=(str(args.device).strip() if args.device else None),
            radius_px=float(args.radius_px),
            mask_type=str(args.mask_type),
            heatmap_sigma_px=(float(args.heatmap_sigma)
                              if args.heatmap_sigma is not None else None),
            instance_mode=str(args.instance_mode),
            bbox_scale=float(args.bbox_scale),
            cache_features=not bool(args.no_cache),
            max_images=int(args.max_images),
            max_patches=int(args.max_patches),
            per_image_per_keypoint=int(args.per_image_per_keypoint),
            pos_threshold=float(args.pos_threshold),
            add_negatives=bool(args.add_negatives),
            neg_threshold=float(args.neg_threshold),
            negatives_per_image=int(args.negatives_per_image),
            crop_px=int(args.crop_px),
            sprite_border_px=int(args.sprite_border_px),
            seed=int(args.seed),
        )
    )
    print(str(log_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
