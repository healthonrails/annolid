"""modified from https://github.com/wkentaro/labelme/blob/main/labelme/ai/models/segment_anything.py"""

import collections
import inspect
import threading

import imgviz
import numpy as np
import skimage.measure
import cv2
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
    def __init__(self, name, encoder_path, decoder_path):
        self.name = name

        self._image_size = 1024

        ort = _require_onnxruntime()
        self._encoder_session = ort.InferenceSession(encoder_path)
        self._decoder_session = ort.InferenceSession(decoder_path)

        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

        self._thread = None

    def set_image(self, image: np.ndarray):
        with self._lock:
            self._image = image
            self._image_embedding = self._image_embedding_cache.get(
                self._image.tobytes()
            )

        if self._image_embedding is None:
            self._thread = threading.Thread(
                target=self._compute_and_cache_image_embedding
            )
            self._thread.start()

    def _compute_and_cache_image_embedding(self):
        with self._lock:
            logger.debug("Computing image embedding...")
            self._image_embedding = _compute_image_embedding(
                image_size=self._image_size,
                encoder_session=self._encoder_session,
                image=self._image,
            )
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[self._image.tobytes()] = self._image_embedding
            logger.debug("Done computing image embedding.")

    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def predict_polygon_from_points(self, points, point_labels):
        image_embedding = self._get_image_embedding()
        polygon = _compute_polygon_from_points(
            image_size=self._image_size,
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=image_embedding,
            points=points,
            point_labels=point_labels,
        )
        return polygon

    def predict_mask_from_points(self, points, point_labels):
        image_embedding = self._get_image_embedding()
        mask = _compute_mask_from_points(
            image_size=self._image_size,
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=image_embedding,
            points=points,
            point_labels=point_labels,
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


def _compute_image_embedding(image_size, encoder_session, image):
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
    if input_names[0] == "image":
        output = encoder_session.run(output_names=None, input_feed={"image": x})
    else:
        output = encoder_session.run(output_names=None, input_feed={"x": x})
    image_embedding = output[0]

    return image_embedding


def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


def _compute_mask_from_points(
    image_size, decoder_session, image, image_embedding, points, point_labels
):
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
        outputs = decoder_session.run(
            None,
            {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
            },
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

        outputs = decoder_session.run(None, decoder_inputs)
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

    MIN_SIZE_RATIO = 0.05
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

    pos_idx = np.where(labels > 0)[0].tolist()
    neg_idx = np.where(labels <= 0)[0].tolist()

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
        coverage = float(mask_bool.sum()) / total_pixels
        model_score = (
            float(candidate_scores[idx]) if candidate_scores is not None else 0.0
        )
        # Prefer masks that satisfy positive clicks, avoid negative clicks,
        # avoid frame-wide masks, then use model quality as final tie-breaker.
        rank = (positive_hits, -negative_hits, -coverage, model_score)
        if best_score is None or rank > best_score:
            best_index = idx
            best_score = rank
    return int(best_index)


def _compute_polygon_from_points(
    image_size, decoder_session, image, image_embedding, points, point_labels
):
    from annolid.annotation.masks import mask_to_polygons

    mask = _compute_mask_from_points(
        image_size=image_size,
        decoder_session=decoder_session,
        image=image,
        image_embedding=image_embedding,
        points=points,
        point_labels=point_labels,
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
        if int(label) > 0:
            positive_hits += 1
        else:
            negative_hits += 1
    return positive_hits, negative_hits


def _select_best_polygon(polygons, *, points, point_labels):
    best_polygon = None
    best_score = None
    for candidate in list(polygons or []):
        flat = np.asarray(candidate, dtype=np.float32).flatten()
        if flat.size < 6:
            continue
        area = _polygon_area_from_flat(flat)
        if area <= 0.0:
            continue
        positive_hits, negative_hits = _polygon_hit_counts(flat, points, point_labels)
        # Prefer polygons that satisfy positive prompts, avoid negative prompts,
        # then prefer smaller valid areas as a tie-breaker to avoid capturing bounding boxes.
        score = (positive_hits, -negative_hits, -area)
        if best_score is None or score > best_score:
            best_polygon = flat
            best_score = score
    return best_polygon
