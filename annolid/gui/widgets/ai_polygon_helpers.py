"""Pure geometry and prediction helpers for AI polygon/mask modes.

These functions were extracted from :class:`Canvas` so they can be tested and
maintained independently.  Each function receives explicit parameters instead
of relying on Canvas instance state.
"""

from __future__ import annotations

import cv2
import numpy as np
from qtpy import QtCore, QtGui

from annolid.annotation.masks import mask_to_polygons
from annolid.utils.logger import logger


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def mask_bbox(mask) -> tuple[int, int, int, int] | None:
    """Return *(y1, x1, y2, x2)* of the bounding box of *mask*, or ``None``."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or not np.any(mask_array):
        return None
    rows = np.where(mask_array.any(axis=1))[0]
    cols = np.where(mask_array.any(axis=0))[0]
    y1 = int(rows[0])
    y2 = int(rows[-1]) + 1
    x1 = int(cols[0])
    x2 = int(cols[-1]) + 1
    return y1, x1, y2, x2


def _pixmap_bounds(
    pixmap: QtGui.QPixmap | None,
) -> tuple[float, float] | None:
    """Return *(max_x, max_y)* from *pixmap*, or ``None``."""
    if pixmap is not None and not pixmap.isNull():
        return (
            max(0.0, float(pixmap.width() - 1)),
            max(0.0, float(pixmap.height() - 1)),
        )
    return None


def normalize_ai_polygon_points(
    points,
    pixmap: QtGui.QPixmap | None = None,
) -> list[QtCore.QPointF]:
    """Validate, clamp, and de-duplicate polygon *points*.

    Returns a list of :class:`QPointF` with at least 3 vertices, or an empty
    list when the input is invalid/degenerate.
    """
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return []
    if arr.shape[0] < 3:
        return []
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 3:
        return []

    bounds = _pixmap_bounds(pixmap)
    if bounds is not None:
        max_x, max_y = bounds
        # Guard against "clip-to-frame" artifacts: large off-frame polygons
        # should be rejected, not forcibly projected into a giant border shape.
        overflow_tolerance = 2.0
        if (
            float(np.min(arr[:, 0])) < -overflow_tolerance
            or float(np.max(arr[:, 0])) > (max_x + overflow_tolerance)
            or float(np.min(arr[:, 1])) < -overflow_tolerance
            or float(np.max(arr[:, 1])) > (max_y + overflow_tolerance)
        ):
            return []
        arr[:, 0] = np.clip(arr[:, 0], 0.0, max_x)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, max_y)

    dedup: list[np.ndarray] = []
    for point in arr:
        if dedup:
            delta = point - dedup[-1]
            if float(np.hypot(float(delta[0]), float(delta[1]))) < 0.5:
                continue
        dedup.append(point)
    if len(dedup) >= 2:
        closing = dedup[0] - dedup[-1]
        if float(np.hypot(float(closing[0]), float(closing[1]))) < 0.5:
            dedup = dedup[:-1]
    if len(dedup) < 3:
        return []

    poly = np.asarray(dedup, dtype=np.float32)
    area = abs(float(cv2.contourArea(poly.reshape(-1, 1, 2))))
    if area < 1.0:
        return []
    return [QtCore.QPointF(float(x), float(y)) for x, y in poly]


def simplify_ai_polygon_points(
    points: list[QtCore.QPointF] | np.ndarray,
    pixmap: QtGui.QPixmap | None = None,
) -> list[QtCore.QPointF]:
    """Simplify a polygon while preserving its overall shape."""
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return []
    if arr.shape[0] < 3:
        return []
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 3:
        return []

    contour = arr.reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    if not np.isfinite(perimeter) or perimeter <= 0:
        return normalize_ai_polygon_points(arr, pixmap)

    epsilon = max(1.0, min(6.0, perimeter * 0.01))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if simplified is None or simplified.shape[0] < 3:
        simplified = contour

    simplified_arr = simplified.reshape(-1, 2).astype(np.float32)
    if simplified_arr.shape[0] > 256:
        step = int(np.ceil(float(simplified_arr.shape[0]) / 256.0))
        simplified_arr = simplified_arr[:: max(step, 1)].copy()
        if simplified_arr.shape[0] < 3:
            simplified_arr = arr

    normalized = normalize_ai_polygon_points(simplified_arr, pixmap)
    if len(normalized) < 3:
        return normalize_ai_polygon_points(arr, pixmap)

    simplified_area = abs(float(cv2.contourArea(simplified_arr.reshape(-1, 1, 2))))
    original_area = abs(float(cv2.contourArea(arr.reshape(-1, 1, 2))))
    if original_area > 0 and simplified_area / original_area < 0.5:
        return normalize_ai_polygon_points(arr, pixmap)
    return normalized


# ---------------------------------------------------------------------------
# Mask → polygon conversion
# ---------------------------------------------------------------------------


def polygon_from_refined_mask(
    mask,
    *,
    pixmap: QtGui.QPixmap | None = None,
    prompt_points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
) -> list[QtCore.QPointF]:
    """Convert the current refined mask into a stable polygon."""
    try:
        mask_array = np.asarray(mask)
    except Exception:
        return []
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return []
    mask_u8 = (mask_array > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return []

    # Smooth small raster artifacts while preserving the refined mask footprint.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if not np.any(mask_u8):
        return []

    def _remove_thin_full_span_bands(mask_binary: np.ndarray) -> np.ndarray:
        cleaned = mask_binary.copy()
        height, width = cleaned.shape
        max_row_band = max(3, int(height * 0.08))
        max_col_band = max(3, int(width * 0.08))

        def _zero_bands(indices: np.ndarray, axis: int, max_band: int) -> None:
            if indices.size == 0:
                return
            start = int(indices[0])
            prev = int(indices[0])
            for idx in indices[1:]:
                cur = int(idx)
                if cur == prev + 1:
                    prev = cur
                    continue
                if (prev - start + 1) <= max_band:
                    if axis == 0:
                        cleaned[start : prev + 1, :] = 0
                    else:
                        cleaned[:, start : prev + 1] = 0
                start = cur
                prev = cur
            if (prev - start + 1) <= max_band:
                if axis == 0:
                    cleaned[start : prev + 1, :] = 0
                else:
                    cleaned[:, start : prev + 1] = 0

        row_full = np.where(np.sum(cleaned > 0, axis=1) >= int(width * 0.9))[0]
        col_full = np.where(np.sum(cleaned > 0, axis=0) >= int(height * 0.9))[0]
        _zero_bands(row_full, axis=0, max_band=max_row_band)
        _zero_bands(col_full, axis=1, max_band=max_col_band)
        return cleaned

    mask_u8 = _remove_thin_full_span_bands(mask_u8)
    if not np.any(mask_u8):
        return []

    def _is_strip_like_component(
        x: int, y: int, width: int, height: int, full_width: int, full_height: int
    ) -> bool:
        if width <= 0 or height <= 0:
            return True
        near_full_width = width >= int(max(1, full_width * 0.9))
        near_full_height = height >= int(max(1, full_height * 0.9))
        very_thin_h = height <= max(3, int(full_height * 0.08))
        very_thin_w = width <= max(3, int(full_width * 0.08))
        touches_left = x <= 1
        touches_right = (x + width) >= (full_width - 2)
        touches_top = y <= 1
        touches_bottom = (y + height) >= (full_height - 2)
        if near_full_width and very_thin_h and (touches_left or touches_right):
            return True
        if near_full_height and very_thin_w and (touches_top or touches_bottom):
            return True
        return False

    # Prefer the component supported by positive prompts and avoid large strip-like
    # artifacts that can span the whole frame due to transient mask glitches.
    selected_mask = mask_u8
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        if num_labels > 2:
            points_arr = np.asarray(prompt_points or [], dtype=np.float32).reshape(
                -1, 2
            )
            labels_arr = np.asarray(point_labels or [], dtype=np.int32).reshape(-1)
            has_prompts = points_arr.size > 0 and labels_arr.size == points_arr.shape[0]
            full_h, full_w = mask_u8.shape
            best_label = None
            best_score = None
            for comp_idx in range(1, int(num_labels)):
                x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
                y = int(stats[comp_idx, cv2.CC_STAT_TOP])
                width = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
                height = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
                area = int(stats[comp_idx, cv2.CC_STAT_AREA])
                if area <= 1:
                    continue
                strip_like = _is_strip_like_component(
                    x, y, width, height, full_w, full_h
                )

                positive_hits = 0
                negative_hits = 0
                if has_prompts:
                    for idx in range(points_arr.shape[0]):
                        px, py = points_arr[idx]
                        qx = int(np.clip(round(float(px)), 0, full_w - 1))
                        qy = int(np.clip(round(float(py)), 0, full_h - 1))
                        if int(labels[qy, qx]) != comp_idx:
                            continue
                        if int(labels_arr[idx]) > 0:
                            positive_hits += 1
                        else:
                            negative_hits += 1
                    if positive_hits == 0:
                        continue

                score = (positive_hits, -negative_hits, -int(strip_like), area)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = comp_idx

            if best_label is None:
                # Fallback: choose the largest non-strip component.
                for comp_idx in range(1, int(num_labels)):
                    x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
                    y = int(stats[comp_idx, cv2.CC_STAT_TOP])
                    width = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
                    height = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
                    area = int(stats[comp_idx, cv2.CC_STAT_AREA])
                    if area <= 1:
                        continue
                    strip_like = _is_strip_like_component(
                        x, y, width, height, full_w, full_h
                    )
                    if strip_like:
                        continue
                    score = area
                    if best_score is None or score > best_score:
                        best_score = score
                        best_label = comp_idx

            if best_label is not None:
                selected_mask = (labels == int(best_label)).astype(np.uint8)
    except Exception:
        selected_mask = mask_u8

    polygons, _ = mask_to_polygons(selected_mask, simplify=False)
    if not polygons:
        return []

    best_polygon = None
    best_area = None
    for polygon in polygons:
        arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        area = abs(float(cv2.contourArea(arr.reshape(-1, 1, 2))))
        if area <= 1.0:
            continue
        if best_area is None or area > best_area:
            best_polygon = arr
            best_area = area

    if best_polygon is None:
        return []

    contour = best_polygon.reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(contour, True))
    epsilon = max(1.0, min(4.0, perimeter * 0.005))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx is None or approx.shape[0] < 3:
        approx = contour

    if approx.shape[0] > 300:
        step = int(np.ceil(float(approx.shape[0]) / 300.0))
        approx = approx[:: max(step, 1)].copy()

    arr_points = approx.reshape(-1, 2).astype(np.float32)
    return normalize_ai_polygon_points(arr_points, pixmap)


# ---------------------------------------------------------------------------
# AI model prediction orchestration
# ---------------------------------------------------------------------------


def predict_ai_polygon_points(
    *,
    ai_model,
    pixmap: QtGui.QPixmap | None = None,
    prompt_points: list[list[float]],
    point_labels: list[int],
) -> list[QtCore.QPointF]:
    """Predict AI polygon points from the refined mask.

    If a mask predictor exists, it is authoritative. We avoid falling back to
    direct polygon prediction on mask errors because that can produce unstable
    oversized polygons during interactive refinement.
    """
    if ai_model is None:
        return []

    mask_predictor = getattr(ai_model, "predict_mask_from_points", None)
    if callable(mask_predictor):
        try:
            mask = mask_predictor(points=prompt_points, point_labels=point_labels)
        except Exception as exc:
            logger.debug(
                "AI polygon mask prediction failed; trying polygon fallback. Error: %s",
                exc,
                exc_info=True,
            )
            return []
        mask_points = polygon_from_refined_mask(
            mask,
            pixmap=pixmap,
            prompt_points=prompt_points,
            point_labels=point_labels,
        )
        if len(mask_points) >= 3:
            simplified_points = simplify_ai_polygon_points(mask_points, pixmap)
            if len(simplified_points) >= 3:
                return simplified_points
            return mask_points
        return []

    polygon_predictor = getattr(ai_model, "predict_polygon_from_points", None)
    if not callable(polygon_predictor):
        return []

    try:
        polygon = polygon_predictor(
            points=prompt_points,
            point_labels=point_labels,
        )
    except Exception as exc:
        logger.debug(
            "AI polygon fallback predictor failed. Error: %s",
            exc,
            exc_info=True,
        )
        return []

    normalized_points = normalize_ai_polygon_points(polygon, pixmap)
    if len(normalized_points) < 3:
        return []
    simplified_points = simplify_ai_polygon_points(normalized_points, pixmap)
    if len(simplified_points) >= 3:
        return simplified_points
    return normalized_points
