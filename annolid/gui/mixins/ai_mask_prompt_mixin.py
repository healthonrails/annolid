from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from qtpy import QtGui

from annolid.gui.window_base import utils
from annolid.utils.logger import logger
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image


class AiMaskPromptMixin:
    """AI mask render export and visual prompt extraction helpers."""

    def _saveImageFile(self, filename):
        image_filename = filename.replace(".json", ".png")
        if self.imageData is None:
            return image_filename
        try:
            if not self.imageData.save(image_filename):
                logger.warning(f"Failed to save seed image: {image_filename}")
        except Exception as exc:
            logger.warning(f"Exception while saving seed image {image_filename}: {exc}")
        return image_filename

    def _save_ai_mask_renders(self, image_filename: str) -> None:
        if not image_filename or self.labelList is None:
            return

        def _qimage_to_np(qimage_obj):
            try:
                return convert_qt_image_to_rgb_cv_image(qimage_obj).copy()
            except Exception:
                return None

        base_image = None
        try:
            if isinstance(self.imageData, QtGui.QImage):
                base_image = _qimage_to_np(self.imageData)
            elif isinstance(self.imageData, (bytes, bytearray)):
                base_image = utils.img_data_to_arr(self.imageData).copy()
            elif isinstance(self.imageData, np.ndarray):
                base_image = np.asarray(self.imageData).copy()
        except Exception as exc:
            logger.warning(f"Unable to convert image for AI mask export: {exc}")

        if base_image is None:
            canvas_pixmap = getattr(self.canvas, "pixmap", None)
            if canvas_pixmap is not None and not canvas_pixmap.isNull():
                base_image = _qimage_to_np(canvas_pixmap.toImage())

        if base_image is None:
            logger.debug("Skipping AI mask render save: unsupported image data.")
            return

        mask_shapes = []

        def _maybe_add_shape(shape):
            if (
                shape is not None
                and getattr(shape, "shape_type", None) == "mask"
                and getattr(shape, "mask", None) is not None
                and len(getattr(shape, "points", [])) >= 1
            ):
                mask_shapes.append(shape)

        try:
            if self.labelList:
                for item in self.labelList:
                    shape_obj = item.shape() if item is not None else None
                    _maybe_add_shape(shape_obj)
        except Exception as exc:
            logger.warning(f"Failed to collect AI mask shapes from label list: {exc}")

        if not mask_shapes and getattr(self.canvas, "shapes", None):
            for shape in self.canvas.shapes:
                _maybe_add_shape(shape)

        if not mask_shapes:
            return

        def paste_mask(mask_arr, top_left_point, canvas):
            mask_arr = np.asarray(mask_arr).astype(np.uint8)
            if mask_arr.size == 0:
                return
            if mask_arr.ndim > 2:
                mask_arr = mask_arr[..., 0]
            x1 = max(int(round(top_left_point.x())), 0)
            y1 = max(int(round(top_left_point.y())), 0)
            if x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
                return
            h, w = mask_arr.shape[:2]
            x2 = min(x1 + w, canvas.shape[1])
            y2 = min(y1 + h, canvas.shape[0])
            if x2 <= x1 or y2 <= y1:
                return
            crop_w = x2 - x1
            crop_h = y2 - y1
            canvas[y1:y2, x1:x2] = np.maximum(
                canvas[y1:y2, x1:x2],
                mask_arr[:crop_h, :crop_w].astype(np.uint8),
            )

        combined_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        for shape in mask_shapes:
            paste_mask(shape.mask, shape.points[0], combined_mask)

        if not combined_mask.any():
            return

        stem = Path(image_filename).stem
        base_dir = Path(image_filename).parent

        def save_masked_image(mask_array, suffix):
            masked_image = np.zeros_like(base_image)
            mask_bool = mask_array.astype(bool)
            masked_image[mask_bool] = base_image[mask_bool]
            if suffix:
                out_name = f"{stem}_{suffix}_mask.png"
            else:
                out_name = f"{stem}_mask.png"
            out_path = base_dir / out_name
            try:
                Image.fromarray(masked_image).save(str(out_path))
            except Exception as exc:
                logger.warning(f"Failed to save AI mask render {out_path}: {exc}")

        save_masked_image(combined_mask, "")

        for idx, shape in enumerate(mask_shapes):
            per_mask = np.zeros_like(combined_mask)
            paste_mask(shape.mask, shape.points[0], per_mask)
            if not per_mask.any():
                continue
            safe_label = re.sub(r"[^0-9A-Za-z_-]", "_", shape.label or "")
            if not safe_label:
                safe_label = f"mask_{idx + 1}"
            save_masked_image(per_mask, f"{safe_label}")

    def extract_visual_prompts_from_canvas(self) -> dict:
        bboxes = []
        cls_list = []
        labels = {
            shape.label
            for shape in self.canvas.shapes
            if shape.shape_type == "rectangle" and shape.label
        }
        if labels:
            self.class_mapping = {
                label: idx for idx, label in enumerate(sorted(labels))
            }
        else:
            self.class_mapping = {}

        for shape in self.canvas.shapes:
            if shape.shape_type != "rectangle":
                continue
            if not shape.points or len(shape.points) < 2:
                continue

            xs = [pt.x() if hasattr(pt, "x") else pt[0] for pt in shape.points]
            ys = [pt.y() if hasattr(pt, "y") else pt[1] for pt in shape.points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            bboxes.append([x1, y1, x2, y2])

            cls_idx = self.class_mapping.get(shape.label, 0)
            cls_list.append(cls_idx)

        if not bboxes:
            logger.info("No rectangle shapes found on canvas for visual prompts.")
            return {}

        return {"bboxes": bboxes, "cls": cls_list}
