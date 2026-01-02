from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    DinoKPSEGHead,
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

        device_norm = normalize_device(device)
        self.device = torch.device(device_norm)
        self.extractor: Dinov3FeatureExtractor = build_extractor(
            model_name=meta.model_name,
            short_side=meta.short_side,
            layers=meta.layers,
            device=str(self.device),
        )

        self.head: DinoKPSEGHead = head.to(self.device)
        self.head.eval()

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

    @torch.inference_mode()
    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        threshold: Optional[float] = None,
        return_patch_masks: bool = False,
        stabilize_lr: bool = False,
        stabilize_cfg: Optional[LRStabilizeConfig] = None,
    ) -> DinoKPSEGPrediction:
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

        # Argmax per keypoint channel
        flat = probs.view(probs.shape[0], -1)
        best_idx = torch.argmax(flat, dim=1)
        scores = torch.gather(flat, 1, best_idx[:, None]).squeeze(1).tolist()
        best_idx = best_idx.tolist()

        keypoints_xy: List[Tuple[float, float]] = []
        for kpt_id, idx in enumerate(best_idx):
            x_res = None
            y_res = None
            if masks_t is not None:
                try:
                    mask_k = masks_t[int(kpt_id)]
                    if bool(mask_k.any()):
                        rows, cols = torch.nonzero(mask_k, as_tuple=True)
                        if rows.numel() > 0:
                            x_res = (cols.to(dtype=torch.float32) +
                                     0.5).mean().item() * float(patch_size)
                            y_res = (rows.to(dtype=torch.float32) +
                                     0.5).mean().item() * float(patch_size)
                except Exception:
                    x_res = None
                    y_res = None
            if x_res is None or y_res is None:
                r = int(idx // w_p)
                c0 = int(idx % w_p)
                x_res = (float(c0) + 0.5) * patch_size
                y_res = (float(r) + 0.5) * patch_size

            h0, w0 = frame_bgr.shape[0], frame_bgr.shape[1]
            x_orig = x_res * (float(w0) / float(resized_w))
            y_orig = y_res * (float(h0) / float(resized_h))
            keypoints_xy.append((float(x_orig), float(y_orig)))

        if stabilize_lr and self._symmetric_pairs and self._prev_keypoints_xy is not None:
            keypoints_xy = stabilize_symmetric_keypoints_xy(
                self._prev_keypoints_xy,
                list(keypoints_xy),
                pairs=self._symmetric_pairs,
                prev_scores=self._prev_keypoint_scores,
                curr_scores=[float(s) for s in scores],
                cfg=stabilize_cfg,
            )

        self._prev_keypoints_xy = list(keypoints_xy)
        self._prev_keypoint_scores = [float(s) for s in scores]

        return DinoKPSEGPrediction(
            keypoints_xy=keypoints_xy,
            keypoint_scores=[float(s) for s in scores],
            masks_patch=masks,
            resized_hw=(resized_h, resized_w),
            patch_size=patch_size,
        )
