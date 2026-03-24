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

    polygons, _ = mask_to_polygons(mask_u8, simplify=False)
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
