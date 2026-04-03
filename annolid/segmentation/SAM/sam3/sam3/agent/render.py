from __future__ import annotations

from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

try:  # pragma: no cover - exercised indirectly when optional deps are present
    from .viz import visualize as _native_visualize
except Exception:  # pragma: no cover - fallback path
    _native_visualize = None


_PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)


def _color_for_index(idx: int) -> np.ndarray:
    return _PALETTE[idx % len(_PALETTE)]


def _coerce_mask(mask_repr: Any, height: int, width: int) -> np.ndarray:
    if isinstance(mask_repr, np.ndarray):
        mask = mask_repr
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8)

    if isinstance(mask_repr, (list, tuple)):
        mask = np.asarray(mask_repr)
        if mask.ndim != 2:
            raise ValueError("Mask array must be 2D.")
        return (mask > 0).astype(np.uint8)

    if isinstance(mask_repr, dict):
        rle = mask_repr
    else:
        rle = {"counts": mask_repr, "size": [height, width]}

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def _box_xywh_to_xyxy(box: Iterable[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in box]
    if max(x, y, w, h) <= 1.5:
        x *= width
        w *= width
        y *= height
        h *= height
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    return x1, y1, x2, y2


def _draw_overlay(image_rgb: np.ndarray, masks: list[np.ndarray], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    canvas = image_rgb.copy().astype(np.float32)
    for idx, mask in enumerate(masks):
        color = _color_for_index(idx).astype(np.float32)
        mask_idx = mask.astype(bool)
        if mask_idx.any():
            canvas[mask_idx] = 0.65 * canvas[mask_idx] + 0.35 * color
    output = canvas.astype(np.uint8)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), _color_for_index(idx).tolist(), 2)
        cv2.putText(
            output,
            str(idx + 1),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _color_for_index(idx).tolist(),
            1,
            cv2.LINE_AA,
        )
    return output


def _fallback_visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json["original_image_path"]
    pred_masks = list(input_json.get("pred_masks", []))
    pred_boxes = list(input_json.get("pred_boxes", []))

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask_arrays = [_coerce_mask(mask_repr, orig_h, orig_w) for mask_repr in pred_masks]
    boxes_xyxy = [
        _box_xywh_to_xyxy(box, orig_w, orig_h) for box in pred_boxes
    ] if pred_boxes else []

    if zoom_in_index is None:
        rendered = _draw_overlay(img_rgb, mask_arrays, boxes_xyxy)
        return Image.fromarray(rendered)

    idx = int(zoom_in_index)
    num_masks = len(mask_arrays)
    if idx < 0 or idx >= num_masks:
        raise ValueError(f"zoom_in_index {idx} is out of range (0..{num_masks - 1}).")

    mask = mask_arrays[idx].astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        x1, y1, x2, y2 = boxes_xyxy[idx] if idx < len(boxes_xyxy) else (0, 0, orig_w, orig_h)
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

    pad_x = max(8, int((x2 - x1) * 0.2))
    pad_y = max(8, int((y2 - y1) * 0.2))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(orig_w, x2 + pad_x)
    crop_y2 = min(orig_h, y2 + pad_y)

    selected_overlay = _draw_overlay(
        img_rgb,
        [mask],
        [boxes_xyxy[idx] if idx < len(boxes_xyxy) else (crop_x1, crop_y1, crop_x2, crop_y2)],
    )
    zoom_overlay = selected_overlay[crop_y1:crop_y2, crop_x1:crop_x2]
    full_overlay = _draw_overlay(img_rgb, mask_arrays, boxes_xyxy)

    return Image.fromarray(full_overlay), Image.fromarray(zoom_overlay)


def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    if _native_visualize is not None:
        return _native_visualize(
            input_json,
            zoom_in_index=zoom_in_index,
            mask_alpha=mask_alpha,
            label_mode=label_mode,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
    return _fallback_visualize(
        input_json,
        zoom_in_index=zoom_in_index,
        mask_alpha=mask_alpha,
        label_mode=label_mode,
        font_size_multiplier=font_size_multiplier,
        boarder_width_multiplier=boarder_width_multiplier,
    )
