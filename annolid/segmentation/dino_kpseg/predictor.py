from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from annolid.features import Dinov3FeatureExtractor
from annolid.segmentation.dino_kpseg.data import build_extractor
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

    def __init__(self, weight_path: str | Path, *, device: Optional[str] = None) -> None:
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
                meta.keypoint_names, kpt_count=int(meta.num_parts))
        self._symmetric_pairs = symmetric_pairs_from_flip_idx(
            self.flip_idx) if self.flip_idx else []

        self._prev_keypoints_xy: Optional[List[Tuple[float, float]]] = None
        self._prev_keypoint_scores: Optional[List[float]] = None
        self._prev_by_instance: dict[int,
                                     Tuple[List[Tuple[float, float]], List[float]]] = {}

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
    def _soft_argmax_coords(
        probs: torch.Tensor,
        *,
        patch_size: int,
    ) -> List[Tuple[float, float]]:
        if probs.ndim != 3:
            raise ValueError("Expected probs in KHW format")
        k, h_p, w_p = int(probs.shape[0]), int(
            probs.shape[1]), int(probs.shape[2])
        norm = probs.sum(dim=(1, 2), keepdim=False).clamp(min=1e-6)
        xs = (torch.arange(w_p, device=probs.device,
              dtype=probs.dtype) + 0.5) * float(patch_size)
        ys = (torch.arange(h_p, device=probs.device,
              dtype=probs.dtype) + 0.5) * float(patch_size)
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
            raise FileNotFoundError(
                f"No DinoKPSEG checkpoint found under: {p}")
        resolved = p.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"DinoKPSEG checkpoint not found: {resolved}")
        return resolved

    def reset_state(self) -> None:
        """Reset any temporal stabilization state."""
        self._prev_keypoints_xy = None
        self._prev_keypoint_scores = None
        self._prev_by_instance = {}

    @torch.inference_mode()
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
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

        feats = self.extractor.extract(
            frame_bgr, color_space="BGR", return_type="torch")
        if feats.ndim != 3:
            raise ValueError("Expected DINO features as CHW")
        if int(feats.shape[0]) != int(self.head.in_dim):
            raise RuntimeError(
                "DinoKPSEG checkpoint/backbone mismatch: "
                f"checkpoint expects {self.head.in_dim} channels but extractor produced {int(feats.shape[0])}. "
                "This often happens when training reused stale cached features from a different DINO backbone. "
                "Fix by retraining with cache disabled (--no-cache) or clearing "
                "~/.cache/annolid/dinokpseg/features, and ensure the checkpoint matches the backbone."
            )
        c, h_p, w_p = feats.shape
        patch_size = int(self.extractor.patch_size)
        resized_h, resized_w = int(h_p) * patch_size, int(w_p) * patch_size

        x = feats.unsqueeze(0).to(self.device, dtype=torch.float32)
        logits = self.head(x)[0]  # [K, H_p, W_p]
        probs = torch.sigmoid(logits).to("cpu")

        thr = float(threshold) if threshold is not None else float(
            self.meta.threshold)
        masks: Optional[np.ndarray] = None
        masks_t = None
        if return_patch_masks:
            masks_t = probs >= thr
            masks = masks_t.numpy().astype(np.uint8, copy=False)

        # Soft-argmax per keypoint channel for sub-patch localization.
        coords_resized = self._soft_argmax_coords(
            probs.to(dtype=torch.float32), patch_size=patch_size)

        # Scores from peak probability for consistency with prior outputs.
        flat = probs.view(probs.shape[0], -1)
        best_idx = torch.argmax(flat, dim=1)
        scores = torch.gather(flat, 1, best_idx[:, None]).squeeze(1).tolist()

        keypoints_xy: List[Tuple[float, float]] = []
        h0, w0 = frame_bgr.shape[0], frame_bgr.shape[1]
        for x_res, y_res in coords_resized:
            x_orig = float(x_res) * (float(w0) / float(resized_w))
            y_orig = float(y_res) * (float(h0) / float(resized_h))
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

    def predict_instances(
        self,
        frame_bgr: np.ndarray,
        *,
        bboxes_xyxy: Sequence[Sequence[float]],
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

        results: List[Tuple[int, DinoKPSEGPrediction]] = []
        scale = float(bbox_scale) if bbox_scale is not None else 1.0
        for idx, bbox in enumerate(list(bboxes_xyxy)):
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
            pred = self.predict(
                crop,
                threshold=threshold,
                return_patch_masks=return_patch_masks,
                stabilize_lr=stabilize_lr,
                stabilize_cfg=stabilize_cfg,
                instance_id=int(idx),
            )
            shifted_xy = [
                (float(x) + float(rx1), float(y) + float(ry1))
                for x, y in pred.keypoints_xy
            ]
            results.append(
                (
                    int(idx),
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
