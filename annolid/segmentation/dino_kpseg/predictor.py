from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from annolid.features import Dinov3FeatureExtractor
from annolid.segmentation.dino_kpseg.data import build_extractor, merge_feature_layers
from annolid.segmentation.dino_kpseg.cli_utils import normalize_device
from annolid.segmentation.dino_kpseg.keypoints import (
    LRStabilizeConfig,
    infer_flip_idx_from_names,
    stabilize_symmetric_keypoints_xy,
    symmetric_pairs_from_flip_idx,
)
from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGCheckpointMeta,
    checkpoint_unpack,
)


@dataclass(frozen=True)
class DinoKPSEGPrediction:
    keypoints_xy: List[Tuple[float, float]]
    keypoint_scores: List[float]
    masks_patch: Optional[np.ndarray]
    resized_hw: Tuple[int, int]
    patch_size: int


class DinoKPSEGPredictor:
    """Run keypoint mask segmentation using frozen DINOv3 features."""

    def __init__(
        self, weight_path: str | Path, *, device: Optional[str] = None
    ) -> None:
        weight_path = self._resolve_checkpoint_path(weight_path)
        payload = torch.load(weight_path, map_location="cpu")
        head, meta = checkpoint_unpack(payload)

        self.meta: DinoKPSEGCheckpointMeta = meta
        self.keypoint_names = meta.keypoint_names

        self.flip_idx: Optional[List[int]] = None
        if isinstance(getattr(meta, "flip_idx", None), list):
            try:
                candidate = [int(v) for v in list(meta.flip_idx or [])]
            except Exception:
                candidate = None
            if candidate and len(candidate) == int(meta.num_parts):
                self.flip_idx = candidate
        if self.flip_idx is None and meta.keypoint_names:
            self.flip_idx = infer_flip_idx_from_names(
                meta.keypoint_names, kpt_count=int(meta.num_parts)
            )
        self._symmetric_pairs = (
            symmetric_pairs_from_flip_idx(self.flip_idx) if self.flip_idx else []
        )

        self._prev_keypoints_xy: Optional[List[Tuple[float, float]]] = None
        self._prev_keypoint_scores: Optional[List[float]] = None
        self._prev_by_instance: dict[
            int, Tuple[List[Tuple[float, float]], List[float]]
        ] = {}

        device_norm = normalize_device(device)
        self.device = torch.device(device_norm)
        self.extractor: Dinov3FeatureExtractor = build_extractor(
            model_name=meta.model_name,
            short_side=meta.short_side,
            layers=meta.layers,
            device=str(self.device),
        )

        # head can be conv or attention; both expose `.in_dim` and return [B,K,H,W] logits.
        self.head = head.to(self.device)
        self.head.eval()

    @staticmethod
    def _sanitize_threshold(threshold: float) -> float:
        thr = float(threshold)
        if not math.isfinite(thr):
            return 0.0
        return thr

    @staticmethod
    def _normalize_instance_id(value: object, *, fallback: int) -> int:
        if value is None:
            return int(fallback)
        if isinstance(value, bool):  # bool is also int
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and math.isfinite(float(value)):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw:
                sign = 1
                if raw[0] in ("+", "-"):
                    sign = -1 if raw[0] == "-" else 1
                    raw = raw[1:]
                if raw.isdigit():
                    return int(sign * int(raw))
        return int(fallback)

    @classmethod
    def _resolve_instance_ids(
        cls,
        *,
        count: int,
        instance_ids: Optional[Sequence[object]] = None,
    ) -> List[int]:
        n = max(0, int(count))
        if n == 0:
            return []

        raw_ids = list(instance_ids) if instance_ids is not None else []
        resolved: List[int] = []
        used: set[int] = set()
        for idx in range(n):
            raw = raw_ids[idx] if idx < len(raw_ids) else None
            candidate = cls._normalize_instance_id(raw, fallback=idx)
            unique = int(candidate)
            while unique in used:
                unique += 1
            used.add(unique)
            resolved.append(int(unique))
        return resolved

    @staticmethod
    def _clip_xy_to_frame(
        x: float,
        y: float,
        *,
        frame_h: int,
        frame_w: int,
    ) -> Tuple[float, float]:
        if frame_w <= 0 or frame_h <= 0:
            return (0.0, 0.0)
        x_clip = float(min(max(float(x), 0.0), float(frame_w - 1)))
        y_clip = float(min(max(float(y), 0.0), float(frame_h - 1)))
        return (x_clip, y_clip)

    @staticmethod
    def _resized_to_original_xy(
        x_resized: float,
        y_resized: float,
        *,
        resized_h: int,
        resized_w: int,
        frame_h: int,
        frame_w: int,
    ) -> Tuple[float, float]:
        if resized_h <= 0 or resized_w <= 0:
            return (0.0, 0.0)
        x_orig = float(x_resized) * (float(frame_w) / float(resized_w))
        y_orig = float(y_resized) * (float(frame_h) / float(resized_h))
        return DinoKPSEGPredictor._clip_xy_to_frame(
            x_orig, y_orig, frame_h=frame_h, frame_w=frame_w
        )

    @staticmethod
    def _mask_to_uint8(mask: np.ndarray, *, frame_shape: Tuple[int, int]) -> np.ndarray:
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:2] != tuple(frame_shape):
            raise ValueError("mask must have the same HxW as the input frame/crop")
        if mask_arr.dtype == bool:
            return mask_arr.astype(np.uint8) * 255
        if np.issubdtype(mask_arr.dtype, np.floating):
            clipped = np.clip(mask_arr, 0.0, 1.0)
            return (clipped * 255.0).astype(np.uint8)
        if mask_arr.dtype != np.uint8:
            return np.clip(mask_arr, 0, 255).astype(np.uint8)
        return mask_arr

    @staticmethod
    def _apply_mask_gate(
        probs_raw: torch.Tensor, mask_patch: torch.Tensor
    ) -> torch.Tensor:
        gated = probs_raw * mask_patch.unsqueeze(0)
        gated_mass = gated.sum(dim=(1, 2))
        fallback = gated_mass <= 1e-6
        if bool(torch.any(fallback)):
            probs = probs_raw.clone()
            probs[~fallback] = gated[~fallback]
            return probs
        return gated

    @staticmethod
    def _soft_argmax_coords(
        probs: torch.Tensor,
        *,
        patch_size: int,
    ) -> List[Tuple[float, float]]:
        if probs.ndim != 3:
            raise ValueError("Expected probs in KHW format")
        h_p, w_p = int(probs.shape[1]), int(probs.shape[2])
        norm = probs.sum(dim=(1, 2), keepdim=False).clamp(min=1e-6)
        xs = (torch.arange(w_p, device=probs.device, dtype=probs.dtype) + 0.5) * float(
            patch_size
        )
        ys = (torch.arange(h_p, device=probs.device, dtype=probs.dtype) + 0.5) * float(
            patch_size
        )
        x_exp = (probs.sum(dim=1) * xs[None, :]).sum(dim=1) / norm
        y_exp = (probs.sum(dim=2) * ys[None, :]).sum(dim=1) / norm
        return [(float(x), float(y)) for x, y in zip(x_exp.tolist(), y_exp.tolist())]

    @staticmethod
    def _resolve_checkpoint_path(weight_path: str | Path) -> Path:
        p = Path(weight_path).expanduser()
        if p.is_dir():
            candidates = [
                p / "weights" / "best.pt",
                p / "weights" / "last.pt",
                p / "best.pt",
                p / "last.pt",
            ]
            for c in candidates:
                if c.is_file():
                    return c.resolve()
            raise FileNotFoundError(f"No DinoKPSEG checkpoint found under: {p}")
        resolved = p.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"DinoKPSEG checkpoint not found: {resolved}")
        return resolved

    def reset_state(self) -> None:
        """Reset any temporal stabilization state."""
        self._prev_keypoints_xy = None
        self._prev_keypoint_scores = None
        self._prev_by_instance = {}

    @staticmethod
    def _local_peaks_2d(
        heatmap: torch.Tensor,
        *,
        topk: int,
        threshold: float,
        nms_radius: int,
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Return top-K local maxima (x_idx, y_idx) and their scores from a 2D heatmap."""
        if heatmap.ndim != 2:
            raise ValueError("Expected heatmap in HW format")
        if topk <= 0:
            return [], []
        thr = float(threshold)
        if not math.isfinite(thr):
            thr = 0.0
        r = max(0, int(nms_radius))
        k = 2 * r + 1
        if k <= 1:
            pooled = heatmap
        else:
            pooled = F.max_pool2d(
                heatmap[None, None, ...],
                kernel_size=int(k),
                stride=1,
                padding=int(r),
            )[0, 0]
        peaks = (heatmap >= thr) & (heatmap >= pooled)
        ys, xs = torch.nonzero(peaks, as_tuple=True)
        if xs.numel() == 0:
            # Always return at least the global maximum, even if below threshold,
            # so downstream pipelines can continue deterministically.
            flat_idx = int(torch.argmax(heatmap).item())
            w = int(heatmap.shape[1])
            y = flat_idx // max(1, w)
            x = flat_idx % max(1, w)
            return [(int(x), int(y))], [float(heatmap[int(y), int(x)].item())]

        scores = heatmap[ys, xs]
        k_eff = min(int(topk), int(scores.numel()))
        top_scores, top_idx = torch.topk(scores, k=k_eff, largest=True)
        coords = [(int(xs[i].item()), int(ys[i].item())) for i in top_idx]
        return coords, [float(v.item()) for v in top_scores]

    def _compute_probs(
        self,
        feats: torch.Tensor,
        *,
        frame_shape: Tuple[int, int],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int], int]:
        """Compute keypoint probability maps in patch-grid resolution (K, H_p, W_p)."""
        feats = self._validate_features(feats)
        _, h_p, w_p = feats.shape
        patch_size = int(self.extractor.patch_size)
        resized_h, resized_w = int(h_p) * patch_size, int(w_p) * patch_size

        x = feats.unsqueeze(0).to(self.device, dtype=torch.float32)
        logits = self.head(x)[0]  # [K, H_p, W_p]
        probs_raw = torch.sigmoid(logits).to("cpu")
        probs = probs_raw

        if mask is not None:
            mask_arr = self._mask_to_uint8(mask, frame_shape=frame_shape)

            mask_patch = self.extractor.quantize_mask(mask_arr).to(
                dtype=probs_raw.dtype, device=probs_raw.device
            )
            probs = self._apply_mask_gate(probs_raw, mask_patch)

        return probs, (int(resized_h), int(resized_w)), int(patch_size)

    def predict_multi_peaks_from_features(
        self,
        feats: torch.Tensor,
        *,
        frame_shape: Tuple[int, int],
        mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        topk: int = 5,
        nms_radius_px: float = 12.0,
        keypoint_indices: Optional[Iterable[int]] = None,
    ) -> List[List[Tuple[float, float, float]]]:
        """Return multiple peaks per keypoint channel as [(x, y, score), ...]."""
        probs, (resized_h, resized_w), patch_size = self._compute_probs(
            feats, frame_shape=frame_shape, mask=mask
        )
        thr = self._sanitize_threshold(
            float(threshold) if threshold is not None else float(self.meta.threshold)
        )
        nms_patch = max(
            0, int(round(float(nms_radius_px) / max(1.0, float(patch_size))))
        )

        if keypoint_indices is None:
            indices = list(range(int(probs.shape[0])))
        else:
            indices = [int(i) for i in keypoint_indices if i is not None]

        out: List[List[Tuple[float, float, float]]] = [
            [] for _ in range(int(probs.shape[0]))
        ]
        h0, w0 = int(frame_shape[0]), int(frame_shape[1])

        for k in indices:
            if k < 0 or k >= int(probs.shape[0]):
                continue
            coords_patch, scores = self._local_peaks_2d(
                probs[int(k)].to(dtype=torch.float32),
                topk=int(topk),
                threshold=float(thr),
                nms_radius=int(nms_patch),
            )
            peaks: List[Tuple[float, float, float]] = []
            for (x_idx, y_idx), score in zip(coords_patch, scores):
                x_res = (float(x_idx) + 0.5) * float(patch_size)
                y_res = (float(y_idx) + 0.5) * float(patch_size)
                x_orig, y_orig = self._resized_to_original_xy(
                    x_res,
                    y_res,
                    resized_h=resized_h,
                    resized_w=resized_w,
                    frame_h=h0,
                    frame_w=w0,
                )
                peaks.append((float(x_orig), float(y_orig), float(score)))
            out[int(k)] = peaks

        return out

    def predict_multi_peaks(
        self,
        frame_bgr: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        topk: int = 5,
        nms_radius_px: float = 12.0,
        keypoint_indices: Optional[Iterable[int]] = None,
    ) -> List[List[Tuple[float, float, float]]]:
        feats = self.extract_features(frame_bgr)
        return self.predict_multi_peaks_from_features(
            feats,
            frame_shape=(int(frame_bgr.shape[0]), int(frame_bgr.shape[1])),
            mask=mask,
            threshold=threshold,
            topk=topk,
            nms_radius_px=nms_radius_px,
            keypoint_indices=keypoint_indices,
        )

    def extract_features(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Extract frozen DINO features (CHW) for a frame or crop."""
        feats = self.extractor.extract(
            frame_bgr, color_space="BGR", return_type="torch"
        )
        feats = merge_feature_layers(feats)
        if feats.ndim != 3:
            raise ValueError("Expected DINO features as CHW")
        return feats

    def _validate_features(self, feats: torch.Tensor) -> torch.Tensor:
        if not isinstance(feats, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if feats.ndim != 3:
            raise ValueError("Expected DINO features as CHW")
        if int(feats.shape[0]) != int(self.head.in_dim):
            raise RuntimeError(
                "DinoKPSEG checkpoint/backbone mismatch: "
                f"checkpoint expects {self.head.in_dim} channels but extractor produced {int(feats.shape[0])}. "
                "This often happens when training reused stale cached features from a different DINO backbone or layer set. "
                "Fix by retraining with cache disabled (--no-cache) or clearing "
                "~/.cache/annolid/dinokpseg/features, and ensure the checkpoint matches the backbone."
            )
        return feats

    def seed_instance_state(
        self,
        instance_id: int,
        *,
        keypoints_xy: Sequence[Sequence[float]],
        keypoint_scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Seed per-instance stabilization state from external keypoints (e.g., manual labels)."""
        instance_id_int = int(instance_id)
        coords = [(float(x), float(y)) for x, y in keypoints_xy]
        if keypoint_scores is None:
            scores = [1.0 for _ in coords]
        else:
            scores = [float(s) for s in keypoint_scores]
        if len(scores) != len(coords):
            raise ValueError("keypoint_scores must match keypoints_xy length")
        self._prev_by_instance[instance_id_int] = (coords, scores)

    @torch.inference_mode()
    def predict_from_features(
        self,
        feats: torch.Tensor,
        *,
        frame_shape: Tuple[int, int],
        mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        return_patch_masks: bool = False,
        stabilize_lr: bool = False,
        stabilize_cfg: Optional[LRStabilizeConfig] = None,
        instance_id: Optional[int] = None,
    ) -> DinoKPSEGPrediction:
        prev_xy = self._prev_keypoints_xy
        prev_scores = self._prev_keypoint_scores
        if instance_id is not None:
            cached = self._prev_by_instance.get(int(instance_id))
            if cached is not None:
                prev_xy, prev_scores = cached
            else:
                prev_xy, prev_scores = None, None

        probs, (resized_h, resized_w), patch_size = self._compute_probs(
            feats, frame_shape=frame_shape, mask=mask
        )

        thr = self._sanitize_threshold(
            float(threshold) if threshold is not None else float(self.meta.threshold)
        )
        masks: Optional[np.ndarray] = None
        masks_t = None
        if return_patch_masks:
            masks_t = probs >= thr
            masks = masks_t.numpy().astype(np.uint8, copy=False)

        # Soft-argmax per keypoint channel for sub-patch localization.
        coords_resized = self._soft_argmax_coords(
            probs.to(dtype=torch.float32), patch_size=patch_size
        )

        # Scores from peak probability for consistency with prior outputs.
        flat = probs.view(probs.shape[0], -1)
        scores = flat.max(dim=1).values.tolist()

        keypoints_xy: List[Tuple[float, float]] = []
        h0, w0 = int(frame_shape[0]), int(frame_shape[1])
        for x_res, y_res in coords_resized:
            x_orig, y_orig = self._resized_to_original_xy(
                x_res,
                y_res,
                resized_h=resized_h,
                resized_w=resized_w,
                frame_h=h0,
                frame_w=w0,
            )
            keypoints_xy.append((float(x_orig), float(y_orig)))

        if stabilize_lr and self._symmetric_pairs and prev_xy is not None:
            keypoints_xy = stabilize_symmetric_keypoints_xy(
                prev_xy,
                list(keypoints_xy),
                pairs=self._symmetric_pairs,
                prev_scores=prev_scores,
                curr_scores=[float(s) for s in scores],
                cfg=stabilize_cfg,
            )

        if instance_id is not None:
            self._prev_by_instance[int(instance_id)] = (
                list(keypoints_xy),
                [float(s) for s in scores],
            )
        else:
            self._prev_keypoints_xy = list(keypoints_xy)
            self._prev_keypoint_scores = [float(s) for s in scores]

        return DinoKPSEGPrediction(
            keypoints_xy=keypoints_xy,
            keypoint_scores=[float(s) for s in scores],
            masks_patch=masks,
            resized_hw=(resized_h, resized_w),
            patch_size=patch_size,
        )

    @torch.inference_mode()
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        return_patch_masks: bool = False,
        stabilize_lr: bool = False,
        stabilize_cfg: Optional[LRStabilizeConfig] = None,
        instance_id: Optional[int] = None,
    ) -> DinoKPSEGPrediction:
        feats = self.extract_features(frame_bgr)
        return self.predict_from_features(
            feats,
            frame_shape=(int(frame_bgr.shape[0]), int(frame_bgr.shape[1])),
            mask=mask,
            threshold=threshold,
            return_patch_masks=return_patch_masks,
            stabilize_lr=stabilize_lr,
            stabilize_cfg=stabilize_cfg,
            instance_id=instance_id,
        )

    def predict_instances(
        self,
        frame_bgr: np.ndarray,
        *,
        bboxes_xyxy: Sequence[Sequence[float]],
        instance_ids: Optional[Sequence[object]] = None,
        threshold: Optional[float] = None,
        return_patch_masks: bool = False,
        stabilize_lr: bool = False,
        stabilize_cfg: Optional[LRStabilizeConfig] = None,
        bbox_scale: float = 1.0,
        normalized: bool = False,
    ) -> List[Tuple[int, DinoKPSEGPrediction]]:
        if frame_bgr.ndim != 3:
            raise ValueError("Expected BGR frame with shape HxWx3")
        frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if frame_h <= 1 or frame_w <= 1:
            return []

        bboxes = list(bboxes_xyxy)
        resolved_ids = self._resolve_instance_ids(
            count=len(bboxes),
            instance_ids=instance_ids,
        )
        results: List[Tuple[int, DinoKPSEGPrediction]] = []
        scale = float(bbox_scale) if bbox_scale is not None else 1.0
        for bbox, instance_id in zip(bboxes, resolved_ids):
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            if normalized:
                x1 *= float(frame_w)
                x2 *= float(frame_w)
                y1 *= float(frame_h)
                y2 *= float(frame_h)

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = max(1.0, float(x2 - x1)) * scale
            bh = max(1.0, float(y2 - y1)) * scale
            rx1 = max(0, int(math.floor(cx - bw / 2.0)))
            ry1 = max(0, int(math.floor(cy - bh / 2.0)))
            rx2 = min(frame_w, int(math.ceil(cx + bw / 2.0)))
            ry2 = min(frame_h, int(math.ceil(cy + bh / 2.0)))
            if rx2 - rx1 < 2 or ry2 - ry1 < 2:
                continue

            crop = frame_bgr[ry1:ry2, rx1:rx2]
            feats = self.extract_features(crop)
            pred = self.predict_from_features(
                feats,
                frame_shape=(int(crop.shape[0]), int(crop.shape[1])),
                threshold=threshold,
                return_patch_masks=return_patch_masks,
                stabilize_lr=stabilize_lr,
                stabilize_cfg=stabilize_cfg,
                instance_id=int(instance_id),
            )
            shifted_xy = [
                (float(x) + float(rx1), float(y) + float(ry1))
                for x, y in pred.keypoints_xy
            ]
            results.append(
                (
                    int(instance_id),
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
