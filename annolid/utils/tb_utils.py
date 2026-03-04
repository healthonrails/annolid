"""
Shared TensorBoard utilities for Annolid.

This module consolidates helpers used by multiple models (Mask R-CNN,
DINO keypoint segmentation, YOLO) so that the same code powers every
model's TensorBoard integration.

Public API
----------
- ``_try_create_summary_writer(log_dir)``
- ``_sanitize_tb_image_tensor(img)``
- ``_draw_loss_curve_image(csv_path, *, title, width, height)``
- ``_read_image_rgb(path, *, max_edge_px)``
- ``sanitize_tensorboard_tag(tag)``
- ``ScalarWriter`` (Protocol)
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Optional, Protocol


# ---------------------------------------------------------------------------
# Protocol: matches SummaryWriter's relevant surface
# ---------------------------------------------------------------------------


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...
    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict, global_step: int
    ) -> None: ...
    def add_text(self, tag: str, text_string: str, global_step: int) -> None: ...
    def add_image(
        self, tag: str, img_tensor: Any, global_step: int, dataformats: str = "CHW"
    ) -> None: ...
    def add_images(self, tag: str, img_tensor: Any, global_step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Tag sanitisation (previously only in yolo/tensorboard_logging.py)
# ---------------------------------------------------------------------------

_TAG_SAFE = re.compile(r"[^a-zA-Z0-9._/-]+")


def sanitize_tensorboard_tag(tag: str) -> str:
    """Return a TensorBoard-safe tag string."""
    tag = str(tag or "").strip()
    tag = tag.replace(" ", "_")
    for ch in "()[]{} ":
        tag = tag.replace(ch, "_")
    tag = _TAG_SAFE.sub("_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag or "metric"


# ---------------------------------------------------------------------------
# SummaryWriter construction (graceful degradation when tensorboard absent)
# ---------------------------------------------------------------------------


def _try_create_summary_writer(log_dir: Path) -> Optional[ScalarWriter]:
    """Return a ``SummaryWriter`` or ``None`` when torch.utils.tensorboard is absent."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image helpers (safe GPU→CPU conversion)
# ---------------------------------------------------------------------------


def _sanitize_tb_image_tensor(image: Any) -> Any:
    """Return a CPU float32 CHW tensor clamped to [0, 1].

    Handles NaN / Inf values from model outputs before they reach
    TensorBoard's uint8 conversion and trigger cast warnings.
    """
    import torch  # local import keeps this module import-time light

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(image)}")
    out = image.detach().float().cpu()
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.clamp(0.0, 1.0)


def _read_image_rgb(path: Path, *, max_edge_px: int = 1280) -> Optional[Any]:
    """Read an image from disk into an RGB uint8 HWC numpy array.

    Tries OpenCV first, then Pillow.  Optionally down-scales so the
    longest edge is at most *max_edge_px* pixels.
    """
    try:
        import numpy as np
    except Exception:
        return None

    arr: Any = None
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        arr = None

    if arr is None:
        try:
            from PIL import Image

            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
        except Exception:
            return None

    try:
        h, w = int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return None

    edge = max(h, w)
    if max_edge_px and edge > int(max_edge_px):
        scale = float(max_edge_px) / float(edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2  # type: ignore

            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            try:
                from PIL import Image

                im = Image.fromarray(arr).resize((new_w, new_h))
                arr = np.asarray(im)
            except Exception:
                pass

    try:
        import numpy as np_mod

        if arr.dtype != np_mod.dtype("uint8"):
            arr = arr.astype("uint8", copy=False)
    except Exception:
        return None
    return arr


# ---------------------------------------------------------------------------
# Loss-curve renderer (PIL-based, no matplotlib)
# ---------------------------------------------------------------------------


def _draw_loss_curve_image(
    csv_path: Path,
    *,
    title: str = "Loss curve",
    width: int = 720,
    height: int = 420,
    train_key: str = "train_loss",
    val_key: str = "val_loss",
    step_key: str = "iteration",
) -> Optional[Any]:
    """Render a loss curve from a CSV file and return a CHW float tensor.

    The CSV must contain at minimum a step/epoch column and at least one
    of ``train_key`` / ``val_key`` columns.  Returns ``None`` if PIL or
    the CSV is unavailable.
    """
    try:
        text = Path(csv_path).read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except Exception:
        return None
    if not rows:
        return None

    xs, train_ys, val_ys = [], [], []
    for row in rows:
        try:
            x = int(float(row.get(step_key) or row.get("epoch") or 0))
        except Exception:
            continue
        tr = _safe_float(row.get(train_key))
        va = _safe_float(row.get(val_key))
        if tr is None and va is None:
            continue
        xs.append(x)
        train_ys.append(tr)
        val_ys.append(va)

    if not xs:
        return None

    all_vals = [v for v in train_ys + val_ys if v is not None and math.isfinite(v)]
    if not all_vals:
        return None
    y_min, y_max = min(all_vals), max(all_vals)
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    try:
        from PIL import Image, ImageDraw
        import numpy as np
        import torch
    except Exception:
        return None

    img = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 48
    x0, y0 = margin, margin
    x1, y1 = int(width) - margin, int(height) - margin
    draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=2)
    draw.text((x0, 10), str(title), fill=(20, 20, 20))

    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1

    def to_px(xv: int, yv: float) -> tuple:
        xf = (xv - x_min) / (x_max - x_min)
        yf = (float(yv) - y_min) / (y_max - y_min)
        return int(x0 + xf * (x1 - x0)), int(y1 - yf * (y1 - y0))

    def draw_series(values, color):
        pts = []
        for xv, yv in zip(xs, values):
            if yv is None or not math.isfinite(yv):
                pts.append(None)
            else:
                pts.append(to_px(xv, yv))
        last = None
        for pt in pts:
            if pt is None:
                last = None
                continue
            if last is not None:
                draw.line([last, pt], fill=color, width=2)
            last = pt

    draw_series(train_ys, (35, 99, 229))  # blue
    if any(v is not None for v in val_ys):
        draw_series(val_ys, (229, 57, 53))  # red

    # Legend
    draw.rectangle([x0 + 8, y0 + 8, x0 + 18, y0 + 18], fill=(35, 99, 229))
    draw.text((x0 + 24, y0 + 6), "train", fill=(20, 20, 20))
    if any(v is not None for v in val_ys):
        draw.rectangle([x0 + 80, y0 + 8, x0 + 90, y0 + 18], fill=(229, 57, 53))
        draw.text((x0 + 96, y0 + 6), "val", fill=(20, 20, 20))

    arr = np.array(img, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None
