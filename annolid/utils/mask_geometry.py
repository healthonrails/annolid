"""Fast, deterministic bounding-box-sliced geometry primitives for 2D masks.

Multi-instance tracking and post-processing (`N = 10..300+` instances) cannot
afford `O(N * H * W)` full-frame scans for centroids, polygon extraction,
overlap checks, or optical-flow summaries. These helpers extract tight
bounding boxes in a single `O(H * W)` pass and run downstream moments,
intersections, and scalar/flow aggregations strictly within each instance's
tight sub-window `O(bbox_area)`.
"""

from __future__ import annotations

from collections.abc import Mapping
import cv2
import numpy as np


BBoxXYXY = tuple[int, int, int, int]
_MAX_DIRECT_LUT_ID = 262_144
_MAX_DIRECT_FIND_OBJECTS_ID = 65_536


def _as_uint8_binary_view(mask: np.ndarray) -> np.ndarray:
    """Return a 2D C-contiguous `uint8` binary view (`0`/`1` or `0..255`) without unnecessary copies."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape!r}.")
    if arr.dtype == np.bool_:
        return np.ascontiguousarray(arr).view(np.uint8)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr > 0).view(np.uint8)


def tight_mask_bbox(mask: np.ndarray) -> BBoxXYXY | None:
    """Return tight half-open `(x0, y0, x1, y1)` bounds of nonzero pixels, or `None` if empty."""
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {arr.shape!r}.")
        return None
    u8 = _as_uint8_binary_view(arr)
    x, y, w, h = cv2.boundingRect(u8)
    if w <= 0 or h <= 0:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def _extract_index_mask_bboxes_numpy(
    arr: np.ndarray,
    max_id: int,
) -> dict[int, BBoxXYXY]:
    """Pure-NumPy single-pass fallback when `scipy.ndimage` is unavailable."""
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return {}
    vals = arr[ys, xs]
    height, width = arr.shape

    if max_id <= _MAX_DIRECT_FIND_OBJECTS_ID:
        table_len = max_id + 1
        xmin = np.full(table_len, width, dtype=np.intp)
        ymin = np.full(table_len, height, dtype=np.intp)
        xmax = np.full(table_len, -1, dtype=np.intp)
        ymax = np.full(table_len, -1, dtype=np.intp)
        np.minimum.at(xmin, vals, xs)
        np.minimum.at(ymin, vals, ys)
        np.maximum.at(xmax, vals, xs)
        np.maximum.at(ymax, vals, ys)
        present = np.flatnonzero(xmax >= xmin)
        return {
            int(obj_id): (
                int(xmin[obj_id]),
                int(ymin[obj_id]),
                int(xmax[obj_id]) + 1,
                int(ymax[obj_id]) + 1,
            )
            for obj_id in present
        }

    unique_ids, inverse = np.unique(vals, return_inverse=True)
    count = int(unique_ids.shape[0])
    xmin = np.full(count, width, dtype=np.intp)
    ymin = np.full(count, height, dtype=np.intp)
    xmax = np.full(count, -1, dtype=np.intp)
    ymax = np.full(count, -1, dtype=np.intp)
    np.minimum.at(xmin, inverse, xs)
    np.minimum.at(ymin, inverse, ys)
    np.maximum.at(xmax, inverse, xs)
    np.maximum.at(ymax, inverse, ys)
    return {
        int(unique_ids[idx]): (
            int(xmin[idx]),
            int(ymin[idx]),
            int(xmax[idx]) + 1,
            int(ymax[idx]) + 1,
        )
        for idx in range(count)
    }


def extract_index_mask_bboxes(index_mask: np.ndarray) -> dict[int, BBoxXYXY]:
    """Extract tight `(x0, y0, x1, y1)` bounding boxes for all positive IDs in one `O(H * W)` pass."""
    arr = np.asarray(index_mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D index mask, got shape {arr.shape!r}.")
    if arr.size == 0:
        return {}
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)

    max_id = int(arr.max())
    if max_id <= 0:
        return {}

    min_id = int(arr.min())
    if min_id < 0:
        arr = np.where(arr > 0, arr, 0)

    try:
        from scipy import ndimage
    except ModuleNotFoundError:
        return _extract_index_mask_bboxes_numpy(arr, max_id)

    if max_id <= _MAX_DIRECT_FIND_OBJECTS_ID:
        slices = ndimage.find_objects(arr, max_label=max_id)
        bboxes: dict[int, BBoxXYXY] = {}
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys, xs = slc
            y0, y1 = int(ys.start), int(ys.stop)
            x0, x1 = int(xs.start), int(xs.stop)
            if x1 > x0 and y1 > y0:
                bboxes[idx] = (x0, y0, x1, y1)
        return bboxes

    # Sparse large object IDs (> 65536): compress positive IDs to 1..K first so
    # `ndimage.find_objects` never allocates an oversized slice table.
    unique_ids, inverse = np.unique(arr, return_inverse=True)
    pos_mask = unique_ids > 0
    if not np.any(pos_mask):
        return {}
    dense_map = np.zeros(unique_ids.shape[0], dtype=np.int32)
    positive_ids = [int(v) for v in unique_ids[pos_mask]]
    dense_map[pos_mask] = np.arange(1, len(positive_ids) + 1, dtype=np.int32)
    dense_arr = dense_map[inverse].reshape(arr.shape)
    slices = ndimage.find_objects(dense_arr, max_label=len(positive_ids))
    bboxes = {}
    for dense_idx, slc in enumerate(slices):
        if slc is None:
            continue
        ys, xs = slc
        y0, y1 = int(ys.start), int(ys.stop)
        x0, x1 = int(xs.start), int(xs.stop)
        if x1 > x0 and y1 > y0:
            bboxes[positive_ids[dense_idx]] = (x0, y0, x1, y1)
    return bboxes


def compute_mask_centroid_and_area(
    mask: np.ndarray,
    bbox: BBoxXYXY | None = None,
) -> tuple[float, float, int] | None:
    """Return `(cx, cy, area_px)` matching OpenCV binary moments in `O(bbox_area)`."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape!r}.")
    if arr.size == 0:
        return None

    resolved_bbox = bbox if bbox is not None else tight_mask_bbox(arr)
    if resolved_bbox is None:
        return None
    x0, y0, x1, y1 = resolved_bbox
    if x1 <= x0 or y1 <= y0:
        return None

    sub = arr[y0:y1, x0:x1]
    if sub.dtype == np.bool_:
        sub_u8 = np.ascontiguousarray(sub).view(np.uint8)
    else:
        sub_u8 = np.ascontiguousarray(sub > 0).view(np.uint8)

    moments = cv2.moments(sub_u8, binaryImage=True)
    m00 = float(moments["m00"])
    if m00 <= 0.0:
        return None
    cx = float(x0) + (float(moments["m10"]) / m00)
    cy = float(y0) + (float(moments["m01"]) / m00)
    return cx, cy, int(round(m00))


def compute_mask_mean_scalar(
    mask: np.ndarray,
    scalar_map: np.ndarray,
    bbox: BBoxXYXY | None = None,
) -> float | None:
    """Compute the mean of `scalar_map` over foreground pixels of `mask` inside `bbox`."""
    arr = np.asarray(mask)
    scalars = np.asarray(scalar_map)
    if arr.ndim != 2 or scalars.ndim != 2 or arr.shape != scalars.shape:
        raise ValueError(
            f"Mask and scalar_map must share one 2D shape; got {arr.shape!r} and {scalars.shape!r}."
        )
    resolved_bbox = bbox if bbox is not None else tight_mask_bbox(arr)
    if resolved_bbox is None:
        return None
    x0, y0, x1, y1 = resolved_bbox
    if x1 <= x0 or y1 <= y0:
        return None
    sub_mask = arr[y0:y1, x0:x1]
    if sub_mask.dtype != np.bool_:
        sub_mask = sub_mask > 0
    selected = scalars[y0:y1, x0:x1][sub_mask]
    if selected.size == 0:
        return None
    return float(np.mean(selected, dtype=np.float64))


def compute_mask_mean_flow_magnitude(
    mask: np.ndarray,
    dense_flow_xy: np.ndarray,
    bbox: BBoxXYXY | None = None,
) -> float | None:
    """Compute mean optical-flow magnitude `hypot(fx, fy)` strictly over foreground pixels."""
    arr = np.asarray(mask)
    flow = np.asarray(dense_flow_xy)
    if (
        arr.ndim != 2
        or flow.ndim != 3
        or flow.shape[:2] != arr.shape
        or flow.shape[2] < 2
    ):
        raise ValueError(
            f"Incompatible mask {arr.shape!r} and dense_flow_xy {flow.shape!r}."
        )
    resolved_bbox = bbox if bbox is not None else tight_mask_bbox(arr)
    if resolved_bbox is None:
        return None
    x0, y0, x1, y1 = resolved_bbox
    if x1 <= x0 or y1 <= y0:
        return None
    sub_mask = arr[y0:y1, x0:x1]
    if sub_mask.dtype != np.bool_:
        sub_mask = sub_mask > 0
    vectors = flow[y0:y1, x0:x1][sub_mask]
    if vectors.size == 0:
        return None
    mags = np.hypot(
        vectors[:, 0].astype(np.float64, copy=False),
        vectors[:, 1].astype(np.float64, copy=False),
    )
    return float(np.mean(mags, dtype=np.float64))


def remap_index_mask_numpy(
    mask: np.ndarray,
    id_mapping: Mapping[int, int],
    *,
    dtype: np.dtype | type = np.int32,
) -> np.ndarray:
    """Remap integer mask IDs in `O(H * W)` using a 1D LUT when IDs are non-negative and bounded."""
    arr = np.asarray(mask)
    out_dtype = np.dtype(dtype)
    if arr.size == 0 or not id_mapping:
        return np.zeros(arr.shape, dtype=out_dtype)

    normalized_pairs = [(int(src), int(dst)) for src, dst in id_mapping.items()]
    min_src = min(src for src, _ in normalized_pairs)
    max_src = max(src for src, _ in normalized_pairs)

    if (
        np.issubdtype(arr.dtype, np.integer)
        and min_src >= 0
        and max_src <= _MAX_DIRECT_LUT_ID
    ):
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        if arr_min >= 0 and arr_max <= _MAX_DIRECT_LUT_ID:
            lut_size = max(max_src, arr_max) + 1
            lut = np.zeros(lut_size, dtype=out_dtype)
            for src, dst in normalized_pairs:
                lut[src] = dst
            return lut[arr]

    remapped = np.zeros(arr.shape, dtype=out_dtype)
    for src, dst in normalized_pairs:
        if dst == 0:
            continue
        remapped[arr == src] = dst
    return remapped
