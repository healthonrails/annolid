"""modified from https://github.com/wkentaro/labelme/blob/main/labelme/ai/models/segment_anything.py"""

import collections
import hashlib
import inspect
import threading

import imgviz
import numpy as np
import skimage.measure
import cv2
from annolid.utils.cancellation import (
    InferenceCancelledError,
    ObservableStopEvent,
    onnx_cancellable_run,
)
from annolid.utils.logger import logger


def _require_onnxruntime():
    try:
        import onnxruntime  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optional dependency 'onnxruntime' is required for ONNX SAM models. "
            "Install it with: pip install onnxruntime"
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'onnxruntime'. This usually indicates a broken install "
            "or missing system DLLs. Reinstall onnxruntime (or onnxruntime-gpu) and "
            "ensure the Microsoft Visual C++ Redistributable is installed on Windows. "
            f"Original error: {exc}"
        ) from exc
    return onnxruntime


class SegmentAnythingModel:
    def __init__(self, name, encoder_path, decoder_path, *, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise InferenceCancelledError("SAM ONNX initialization cancelled.")
        self.name = name

        self._image_size = 1024
        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

        self._thread = None
        self._encode_stop_event: ObservableStopEvent | None = None
        self._image_generation = 0
        self._image_embedding = None
        self._image = None
        self._encoder_session = None
        self._decoder_session = None

        ort = _require_onnxruntime()
        try:
            self._encoder_session = ort.InferenceSession(encoder_path)
            if stop_event is not None and stop_event.is_set():
                raise InferenceCancelledError("SAM ONNX initialization cancelled.")
            self._decoder_session = ort.InferenceSession(decoder_path)
            if stop_event is not None and stop_event.is_set():
                raise InferenceCancelledError("SAM ONNX initialization cancelled.")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _compute_image_key(image: np.ndarray) -> tuple:
        arr = np.ascontiguousarray(image)
        digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
        return (arr.shape, str(arr.dtype), digest)

    def close(self) -> None:
        """Cancel any active image encoding thread and release ONNX sessions/caches."""
        with self._lock:
            prev_stop = getattr(self, "_encode_stop_event", None)
            if prev_stop is not None:
                prev_stop.set()
            thread = getattr(self, "_thread", None)
            self._thread = None
            self._encode_stop_event = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            cache = getattr(self, "_image_embedding_cache", None)
            if cache is not None:
                cache.clear()
            self._image_embedding = None
            self._image = None
            self._encoder_session = None
            self._decoder_session = None

    def set_image(self, image: np.ndarray, *, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise InferenceCancelledError("SAM image setup cancelled.")
        image = np.asarray(image).copy()
        image_key = self._compute_image_key(image)
        with self._lock:
            prev_stop = getattr(self, "_encode_stop_event", None)
            if prev_stop is not None:
                prev_stop.set()
            self._image_generation = getattr(self, "_image_generation", 0) + 1
            generation = int(self._image_generation)
            self._image = image
            self._image_embedding = self._image_embedding_cache.get(image_key)
            if self._image_embedding is not None:
                self._encode_stop_event = None
                self._thread = None
                return
            encode_stop = ObservableStopEvent()
            self._encode_stop_event = encode_stop

        self._thread = threading.Thread(
            target=self._compute_and_cache_image_embedding,
            args=(image, image_key, generation, encode_stop),
            daemon=True,
        )
        self._thread.start()

    def _compute_and_cache_image_embedding(
        self,
        image,
        image_key,
        generation,
        encode_stop=None,
    ):
        logger.debug("Computing image embedding...")
        accepts_stop = False
        try:
            accepts_stop = (
                "stop_event" in inspect.signature(_compute_image_embedding).parameters
            )
        except (TypeError, ValueError):
            accepts_stop = False

        try:
            if accepts_stop:
                image_embedding = _compute_image_embedding(
                    image_size=self._image_size,
                    encoder_session=self._encoder_session,
                    image=image,
                    stop_event=encode_stop,
                )
            else:
                image_embedding = _compute_image_embedding(
                    image_size=self._image_size,
                    encoder_session=self._encoder_session,
                    image=image,
                )
        except InferenceCancelledError:
            logger.debug("Cancelled superseded or stopped SAM image embedding.")
            return

        with self._lock:
            if len(self._image_embedding_cache) > 10:
                oldest_key = next(iter(self._image_embedding_cache))
                self._image_embedding_cache.pop(oldest_key, None)
            self._image_embedding_cache[image_key] = image_embedding
            if int(generation) == int(self._image_generation):
                self._image_embedding = image_embedding
            logger.debug("Done computing image embedding.")

    def _get_image_embedding(self, *, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise InferenceCancelledError("SAM image embedding cancelled.")

        thread = getattr(self, "_thread", None)
        if thread is not None:
            unsub = None
            encode_stop = getattr(self, "_encode_stop_event", None)
            if stop_event is not None and encode_stop is not None:
                add_listener = getattr(stop_event, "add_listener", None)
                if callable(add_listener):
                    unsub = add_listener(encode_stop.set)
                elif stop_event.is_set():
                    encode_stop.set()
            try:
                while thread.is_alive():
                    thread.join(timeout=0.02)
                    if stop_event is not None and stop_event.is_set():
                        if encode_stop is not None:
                            encode_stop.set()
                        thread.join(timeout=0.5)
                        raise InferenceCancelledError("SAM image embedding cancelled.")
            finally:
                if callable(unsub):
                    unsub()
            self._thread = None

        with self._lock:
            emb = self._image_embedding
        if stop_event is not None and stop_event.is_set():
            raise InferenceCancelledError("SAM image embedding cancelled.")
        return emb

    def predict_polygon_from_points(self, points, point_labels, *, stop_event=None):
        image_embedding = self._get_image_embedding(stop_event=stop_event)
        polygon = _compute_polygon_from_points(
            image_size=self._image_size,
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=image_embedding,
            points=points,
            point_labels=point_labels,
            stop_event=stop_event,
        )
        return polygon

    def predict_mask_from_points(self, points, point_labels, *, stop_event=None):
        image_embedding = self._get_image_embedding(stop_event=stop_event)
        mask = _compute_mask_from_points(
            image_size=self._image_size,
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=image_embedding,
            points=points,
            point_labels=point_labels,
            stop_event=stop_event,
        )
        return mask


def _compute_scale_to_resize_image(image_size, image):
    height, width = image.shape[:2]
    if width > height:
        scale = image_size / width
        new_height = int(round(height * scale))
        new_width = image_size
    else:
        scale = image_size / height
        new_height = image_size
        new_width = int(round(width * scale))
    return scale, new_height, new_width


def _resize_image(image_size, image):
    scale, new_height, new_width = _compute_scale_to_resize_image(
        image_size=image_size, image=image
    )
    scaled_image = imgviz.resize(
        image,
        height=new_height,
        width=new_width,
        backend="pillow",
    ).astype(np.float32)
    return scale, scaled_image


def postprocess_masks(mask, img_size, input_size, original_size):
    mask = mask.squeeze(0).transpose(1, 2, 0)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask = mask[: input_size[0], : input_size[1], :]
    mask = cv2.resize(
        mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR
    )
    mask = mask.transpose(2, 0, 1)[None, :, :, :]
    return mask


def _compute_image_embedding(image_size, encoder_session, image, *, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        raise InferenceCancelledError("SAM image embedding cancelled.")
    image = imgviz.asrgb(image)

    scale, x = _resize_image(image_size, image)
    x = (x - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
        [58.395, 57.12, 57.375], dtype=np.float32
    )
    x = np.pad(
        x,
        (
            (0, image_size - x.shape[0]),
            (0, image_size - x.shape[1]),
            (0, 0),
        ),
    )
    x = x.transpose(2, 0, 1)[None, :, :, :]
    input_names = [input.name for input in encoder_session.get_inputs()]
    feed = {"image": x} if input_names[0] == "image" else {"x": x}
    output = onnx_cancellable_run(
        encoder_session,
        None,
        feed,
        stop_event=stop_event,
    )
    image_embedding = output[0]

    return image_embedding


def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


def _compute_mask_from_points(
    image_size,
    decoder_session,
    image,
    image_embedding,
    points,
    point_labels,
    *,
    stop_event=None,
):
    if stop_event is not None and stop_event.is_set():
        raise InferenceCancelledError("SAM mask decoding cancelled.")
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array(point_labels, dtype=np.int32)

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[
        None, :, :
    ]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(
        np.float32
    )

    scale, new_height, new_width = _compute_scale_to_resize_image(
        image_size=image_size, image=image
    )
    onnx_coord = (
        onnx_coord.astype(float)
        * (new_width / image.shape[1], new_height / image.shape[0])
    ).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.array([-1], dtype=np.float32)

    input_names = [input.name for input in decoder_session.get_inputs()]
    score_candidates = None
    if len(input_names) <= 3:
        outputs = onnx_cancellable_run(
            decoder_session,
            None,
            {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
            },
            stop_event=stop_event,
        )
        scores, masks = outputs
        score_candidates = scores
        masks = postprocess_masks(
            masks, image_size, (new_height, new_width), np.array(image.shape[:2])
        )

    else:
        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
        }

        outputs = onnx_cancellable_run(
            decoder_session,
            None,
            decoder_inputs,
            stop_event=stop_event,
        )
        masks = outputs[0]
        if len(outputs) > 1:
            score_candidates = outputs[1]
    mask_index = _select_best_mask_index(
        masks=masks,
        scores=score_candidates,
        points=input_point,
        point_labels=input_label,
    )
    mask = masks[0, mask_index]  # (1, N, H, W) -> (H, W)
    mask = mask > 0.0

    # Apply morphological operations to clean up mask edges
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask = mask_u8.astype(bool)

    MIN_SIZE_RATIO = 0.02
    min_size = int(mask.sum() * MIN_SIZE_RATIO)
    if min_size > 0:
        remove_small_objects_params = inspect.signature(
            skimage.morphology.remove_small_objects
        ).parameters
        if "max_size" in remove_small_objects_params:
            # New API removes components <= max_size.
            skimage.morphology.remove_small_objects(
                mask, max_size=max(min_size - 1, 0), out=mask
            )
        else:
            skimage.morphology.remove_small_objects(mask, min_size=min_size, out=mask)

    if 0:
        imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
    return mask


def _select_best_mask_index(
    *,
    masks: np.ndarray,
    scores: np.ndarray | None,
    points: np.ndarray,
    point_labels: np.ndarray,
) -> int:
    arr = np.asarray(masks)
    if arr.ndim != 4 or arr.shape[0] < 1 or arr.shape[1] < 1:
        return 0

    num_candidates = int(arr.shape[1])
    candidate_scores = None
    if scores is not None:
        s = np.asarray(scores)
        if s.ndim >= 2:
            s = s[0]
        s = s.reshape(-1)
        if s.size >= num_candidates:
            candidate_scores = s[:num_candidates]

    h = int(arr.shape[2])
    w = int(arr.shape[3])
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    if labels.size != pts.shape[0]:
        labels = np.ones((pts.shape[0],), dtype=np.int32)

    pos_idx = np.where(labels == 1)[0].tolist()
    neg_idx = np.where(labels == 0)[0].tolist()
    tl_idx = np.where(labels == 2)[0].tolist()
    br_idx = np.where(labels == 3)[0].tolist()

    box_bounds: tuple[int, int, int, int] | None = None
    if tl_idx and br_idx:
        bx0 = max(
            0,
            min(
                w,
                int(round(min(float(pts[tl_idx[0], 0]), float(pts[br_idx[0], 0])))),
            ),
        )
        by0 = max(
            0,
            min(
                h,
                int(round(min(float(pts[tl_idx[0], 1]), float(pts[br_idx[0], 1])))),
            ),
        )
        bx1 = max(
            0,
            min(
                w,
                int(round(max(float(pts[tl_idx[0], 0]), float(pts[br_idx[0], 0])))),
            ),
        )
        by1 = max(
            0,
            min(
                h,
                int(round(max(float(pts[tl_idx[0], 1]), float(pts[br_idx[0], 1])))),
            ),
        )
        if bx1 > bx0 and by1 > by0:
            box_bounds = (bx0, by0, bx1, by1)

    best_index = 0
    best_score = None
    total_pixels = max(1.0, float(h * w))
    for idx in range(num_candidates):
        mask_bool = arr[0, idx] > 0.0
        positive_hits = 0
        negative_hits = 0
        for pidx in pos_idx:
            x = int(round(float(pts[pidx, 0])))
            y = int(round(float(pts[pidx, 1])))
            if 0 <= x < w and 0 <= y < h and bool(mask_bool[y, x]):
                positive_hits += 1
        for nidx in neg_idx:
            x = int(round(float(pts[nidx, 0])))
            y = int(round(float(pts[nidx, 1])))
            if 0 <= x < w and 0 <= y < h and bool(mask_bool[y, x]):
                negative_hits += 1
        mask_pixels = float(mask_bool.sum())
        coverage = mask_pixels / total_pixels
        model_score = (
            float(candidate_scores[idx]) if candidate_scores is not None else 0.0
        )
        if box_bounds is not None:
            bx0, by0, bx1, by1 = box_bounds
            box_area = max(1.0, float((bx1 - bx0) * (by1 - by0)))
            inside_pixels = float(mask_bool[by0:by1, bx0:bx1].sum())
            union_pixels = max(1.0, mask_pixels + box_area - inside_pixels)
            box_iou = inside_pixels / union_pixels
            inside_precision = inside_pixels / max(1.0, mask_pixels)
            well_contained = int(inside_pixels > 0.0 and inside_precision >= 0.70)
            rank = (
                positive_hits,
                -negative_hits,
                well_contained,
                box_iou * 0.7 + model_score * 0.3,
            )
        else:
            # Prefer masks that satisfy positive clicks, avoid negative clicks,
            # avoid frame-wide masks, then use model quality as final tie-breaker.
            rank = (positive_hits, -negative_hits, -coverage, model_score)
        if best_score is None or rank > best_score:
            best_index = idx
            best_score = rank
    return int(best_index)


def _compute_polygon_from_points(
    image_size,
    decoder_session,
    image,
    image_embedding,
    points,
    point_labels,
    *,
    stop_event=None,
):
    from annolid.annotation.masks import mask_to_polygons

    mask = _compute_mask_from_points(
        image_size=image_size,
        decoder_session=decoder_session,
        image=image,
        image_embedding=image_embedding,
        points=points,
        point_labels=point_labels,
        stop_event=stop_event,
    )
    polygons, has_holes = mask_to_polygons(mask)
    if len(polygons) == 0:
        logger.warning("No polygon found, returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)
    polys = _select_best_polygon(polygons, points=points, point_labels=point_labels)
    if polys is None:
        logger.warning("No valid polygon contour found, returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)
    all_points = np.array(list(zip(polys[0::2], polys[1::2])), dtype=np.float32)
    return all_points


def _polygon_area_from_flat(flat_polygon) -> float:
    arr = np.asarray(flat_polygon, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0.0
    contour = arr.reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(contour)))


def _polygon_hit_counts(flat_polygon, points, point_labels) -> tuple[int, int]:
    arr = np.asarray(flat_polygon, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0, 0
    contour = arr.reshape(-1, 1, 2)
    positive_hits = 0
    negative_hits = 0
    iter_points = [] if points is None else points
    iter_labels = [] if point_labels is None else point_labels
    for point, label in zip(iter_points, iter_labels):
        if point is None or len(point) < 2:
            continue
        px = float(point[0])
        py = float(point[1])
        inside_or_edge = cv2.pointPolygonTest(contour, (px, py), False) >= 0
        if not inside_or_edge:
            continue
        if int(label) == 1:
            positive_hits += 1
        elif int(label) == 0:
            negative_hits += 1
    return positive_hits, negative_hits


def _select_best_polygon(polygons, *, points, point_labels):
    best_polygon = None
    best_score = None
    labels_arr = (
        np.asarray(point_labels, dtype=np.int32).reshape(-1)
        if point_labels is not None
        else np.empty((0,), dtype=np.int32)
    )
    has_box_prompt = bool(np.any(labels_arr == 2) and np.any(labels_arr == 3))
    has_pos_clicks = bool(np.any(labels_arr == 1))
    for candidate in list(polygons or []):
        flat = np.asarray(candidate, dtype=np.float32).flatten()
        if flat.size < 6:
            continue
        area = _polygon_area_from_flat(flat)
        if area <= 0.0:
            continue
        positive_hits, negative_hits = _polygon_hit_counts(flat, points, point_labels)
        area_term = area if (has_box_prompt and not has_pos_clicks) else -area
        score = (positive_hits, -negative_hits, area_term)
        if best_score is None or score > best_score:
            best_polygon = flat
            best_score = score
    return best_polygon
