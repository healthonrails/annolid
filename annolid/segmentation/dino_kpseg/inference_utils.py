from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from annolid.segmentation.dino_kpseg.predictor import (
    DinoKPSEGPrediction,
    DinoKPSEGPredictor,
)
from annolid.segmentation.dino_kpseg.keypoints import LRStabilizeConfig


@dataclass(frozen=True)
class DinoKPSEGInstanceCrop:
    instance_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    crop_bgr: np.ndarray
    crop_mask: Optional[np.ndarray]
    offset_xy: Tuple[int, int]


def mask_bbox(
    mask: np.ndarray,
    *,
    pad_px: int,
    image_hw: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if not np.any(mask):
        return None

    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    pad = max(0, int(pad_px))
    height, width = int(image_hw[0]), int(image_hw[1])
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(int(width), x2 + pad)
    y2 = min(int(height), y2 + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def crop_mask(
    mask: Optional[np.ndarray], bbox_xyxy: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    return mask[y1:y2, x1:x2]


def build_instance_crops(
    frame_bgr: np.ndarray,
    instance_masks: Sequence[Tuple[int, np.ndarray]],
    *,
    pad_px: int = 8,
    use_mask_gate: bool = True,
) -> List[DinoKPSEGInstanceCrop]:
    if frame_bgr.ndim != 3:
        raise ValueError("Expected BGR frame with shape HxWx3")
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])

    crops: List[DinoKPSEGInstanceCrop] = []
    for instance_id, mask in list(instance_masks):
        if mask is None:
            continue
        if mask.shape[:2] != (height, width):
            raise ValueError("Instance mask must match frame size")
        bbox = mask_bbox(mask.astype(bool), pad_px=pad_px, image_hw=(height, width))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_gate = crop_mask(mask, bbox) if use_mask_gate else None
        crops.append(
            DinoKPSEGInstanceCrop(
                instance_id=int(instance_id),
                bbox_xyxy=bbox,
                crop_bgr=crop_bgr,
                crop_mask=crop_gate,
                offset_xy=(int(x1), int(y1)),
            )
        )
    return crops


def predict_on_instance_crops(
    predictor: DinoKPSEGPredictor,
    crops: Iterable[DinoKPSEGInstanceCrop],
    *,
    threshold: Optional[float] = None,
    return_patch_masks: bool = False,
    stabilize_lr: bool = False,
    stabilize_cfg: Optional[LRStabilizeConfig] = None,
    tta_hflip: bool = False,
    tta_merge: str = "mean",
) -> List[Tuple[int, DinoKPSEGPrediction]]:
    results: List[Tuple[int, DinoKPSEGPrediction]] = []
    for crop in crops:
        feats = predictor.extract_features(crop.crop_bgr)
        kwargs = dict(
            frame_shape=(int(crop.crop_bgr.shape[0]), int(crop.crop_bgr.shape[1])),
            mask=crop.crop_mask,
            threshold=threshold,
            return_patch_masks=return_patch_masks,
            stabilize_lr=stabilize_lr,
            stabilize_cfg=stabilize_cfg,
            instance_id=int(crop.instance_id),
            tta_hflip=bool(tta_hflip),
            tta_merge=str(tta_merge),
        )
        try:
            pred = predictor.predict_from_features(feats, **kwargs)
        except TypeError:
            kwargs.pop("tta_hflip", None)
            kwargs.pop("tta_merge", None)
            pred = predictor.predict_from_features(feats, **kwargs)
        shifted_xy = [
            (float(x) + float(crop.offset_xy[0]), float(y) + float(crop.offset_xy[1]))
            for x, y in pred.keypoints_xy
        ]
        results.append(
            (
                int(crop.instance_id),
                DinoKPSEGPrediction(
                    keypoints_xy=shifted_xy,
                    keypoint_scores=pred.keypoint_scores,
                    masks_patch=pred.masks_patch,
                    resized_hw=pred.resized_hw,
                    patch_size=pred.patch_size,
                ),
            )
        )
    return results


def filter_keypoints_by_score(
    pred: DinoKPSEGPrediction,
    *,
    min_score: float = 0.0,
    return_indices: bool = False,
) -> DinoKPSEGPrediction | Tuple[DinoKPSEGPrediction, List[int]]:
    """Return a prediction with low-confidence keypoints dropped."""
    thr = float(min_score)
    if not math.isfinite(thr) or thr <= 0:
        if return_indices:
            return pred, list(range(len(pred.keypoint_scores)))
        return pred
    keep_xy: List[Tuple[float, float]] = []
    keep_scores: List[float] = []
    keep_idx: List[int] = []
    for idx, ((x, y), s) in enumerate(zip(pred.keypoints_xy, pred.keypoint_scores)):
        score = float(s)
        if score < thr:
            continue
        keep_xy.append((float(x), float(y)))
        keep_scores.append(score)
        keep_idx.append(int(idx))
    filtered = DinoKPSEGPrediction(
        keypoints_xy=keep_xy,
        keypoint_scores=keep_scores,
        masks_patch=pred.masks_patch,
        resized_hw=pred.resized_hw,
        patch_size=pred.patch_size,
    )
    if return_indices:
        return filtered, keep_idx
    return filtered


def build_instance_crops_for_tracking(
    frame_bgr: np.ndarray,
    instance_masks: Sequence[Tuple[int, np.ndarray]],
    pad_px: int = 8,
    min_crop_size: int = 32,
) -> List[DinoKPSEGInstanceCrop]:
    """
    Build crops for each instance mask with padding for tracking.

    Args:
        frame_bgr: Input frame in BGR format (HxWx3).
        instance_masks: Sequence of (instance_id, mask) tuples.
        pad_px: Padding in pixels around bounding box.
        min_crop_size: Minimum crop size; skip smaller instances.

    Returns:
        List of DinoKPSEGInstanceCrop objects.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR frame with shape HxWx3")

    H, W = frame_bgr.shape[:2]
    crops = []

    for instance_id, mask in instance_masks:
        # Find bounding box from mask
        mask_binary = (mask > 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
        rows, cols = np.where(mask_binary)

        if len(rows) < 10:  # Skip tiny masks
            continue

        y_min, y_max = int(rows.min()), int(rows.max())
        x_min, x_max = int(cols.min()), int(cols.max())

        # Add padding
        x1 = max(0, x_min - pad_px)
        y1 = max(0, y_min - pad_px)
        x2 = min(W, x_max + pad_px)
        y2 = min(H, y_max + pad_px)

        # Skip if too small
        if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
            continue

        # Extract crop
        crop_bgr = frame_bgr[y1:y2, x1:x2].copy()

        # Extract mask crop for reference
        crop_mask = mask[y1:y2, x1:x2].copy()

        crops.append(
            DinoKPSEGInstanceCrop(
                instance_id=int(instance_id),
                bbox_xyxy=(x1, y1, x2, y2),
                crop_bgr=crop_bgr,
                crop_mask=crop_mask,
                offset_xy=(x1, y1),
            )
        )

    return crops
