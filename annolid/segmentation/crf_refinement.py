"""Optional CRF-style boundary refinement for binary masks."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from annolid.utils.logger import logger


@dataclass(frozen=True)
class CrfRefinementConfig:
    """Runtime settings for mask boundary refinement."""

    enabled: bool = False
    backend: str = "auto"
    band_px: int = 10
    p_core: float = 0.95
    iterations: int = 10
    alpha: float = 12.0
    beta: float = 0.03
    gamma: float = 4.0
    spatial_weight: float = 3.0
    bilateral_weight: float = 20.0
    compatibility: float = 1.0


class CrfMaskRefiner:
    """Refine binary segmentation masks with optional dense CRF support."""

    def __init__(
        self,
        config: Optional[CrfRefinementConfig] = None,
        *,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or CrfRefinementConfig()
        self.device = device
        self._crf_module: Optional[torch.nn.Module] = None
        self._crf_key: Optional[Tuple[int, int, str]] = None
        self._dense_crf_unavailable = False

    def refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a refined boolean mask with the same shape as ``mask``."""
        mask_bool = np.asarray(mask, dtype=bool)
        if not self.config.enabled or not bool(mask_bool.any()):
            return mask_bool.copy()

        backend = str(self.config.backend or "auto").lower()
        if backend in {"none", "off", "disabled"}:
            return mask_bool.copy()

        image_arr = _rgb_array(image)
        if image_arr.shape[:2] != mask_bool.shape:
            logger.warning(
                "CRF refinement skipped: image shape %s does not match mask shape %s.",
                image_arr.shape[:2],
                mask_bool.shape,
            )
            return mask_bool.copy()

        if backend in {"auto", "crf", "dense_crf", "dense"}:
            refined = self._try_dense_crf(image_arr, mask_bool)
            if refined is not None:
                return refined
            if backend != "auto":
                return mask_bool.copy()

        if backend in {"auto", "opencv", "cv2"}:
            return self._refine_with_opencv(image_arr, mask_bool)

        logger.warning(
            "Unknown CRF refinement backend '%s'; mask left unchanged.", backend
        )
        return mask_bool.copy()

    def _try_dense_crf(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._dense_crf_unavailable:
            return None
        try:
            device = self._select_device()
            crf = self._dense_crf_module(mask.shape, device)
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            logits, band, inside_core, outside = self._binary_logits(mask, device)
            with torch.no_grad():
                refined_logits = crf(image_tensor, logits)
            fg = (refined_logits[0, 1] > refined_logits[0, 0]).detach().cpu().numpy()
            refined = mask.copy()
            refined[band] = fg[band]
            refined[inside_core] = True
            refined[outside] = False
            return refined.astype(bool)
        except Exception as exc:  # pragma: no cover - optional backend specific
            self._dense_crf_unavailable = True
            logger.warning(
                "Dense CRF refinement unavailable; falling back when possible: %s",
                exc,
            )
            return None

    def _dense_crf_module(
        self,
        mask_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.nn.Module:
        height, width = int(mask_shape[0]), int(mask_shape[1])
        key = (height, width, str(device))
        if self._crf_module is not None and self._crf_key == key:
            return self._crf_module

        import CRF as crf_lib  # type: ignore[import-not-found]

        scale = float(max(height, width)) / 512.0
        params = crf_lib.FrankWolfeParams(
            scheme="fixed",
            stepsize=1.0,
            regularizer="l2",
            lambda_=1.0,
            lambda_learnable=False,
            x0_weight=0.0,
            x0_weight_learnable=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            module = crf_lib.DenseGaussianCRF(
                classes=2,
                alpha=float(self.config.alpha) * scale,
                beta=float(self.config.beta),
                gamma=float(self.config.gamma) * scale,
                spatial_weight=float(self.config.spatial_weight),
                bilateral_weight=float(self.config.bilateral_weight),
                compatibility=float(self.config.compatibility),
                init="potts",
                solver="fw",
                iterations=max(1, int(self.config.iterations)),
                params=params,
            ).to(device)
        module.eval()
        self._crf_module = module
        self._crf_key = key
        return module

    def _select_device(self) -> torch.device:
        if self.device and str(self.device).startswith("cuda"):
            return torch.device(str(self.device))
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _binary_logits(
        self,
        mask: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        band, inside_core, outside = _boundary_regions(mask, int(self.config.band_px))
        p_core = float(np.clip(self.config.p_core, 0.5, 0.999))
        fg_prob = np.full(mask.shape, 0.5, dtype=np.float32)
        fg_prob[inside_core] = p_core
        fg_prob[outside] = 1.0 - p_core
        if not bool(band.any()):
            fg_prob[mask] = p_core
            fg_prob[~mask] = 1.0 - p_core
        fg = torch.from_numpy(fg_prob).to(device=device, dtype=torch.float32)
        fg_logit = torch.logit(fg.clamp(1e-4, 1.0 - 1e-4))
        logits = torch.stack((-fg_logit, fg_logit), dim=0).unsqueeze(0)
        return logits, band, inside_core, outside

    def _refine_with_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        band, inside_core, outside = _boundary_regions(mask, int(self.config.band_px))
        if not bool(band.any()):
            return mask.copy()

        radius = max(1, min(3, int(self.config.band_px) // 2 or 1))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        score = _edge_aware_mask_score(image, mask, int(self.config.band_px))
        candidate_u8 = (score >= 0.5).astype(np.uint8) * 255
        opened = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        cleaned = _fill_holes(closed > 0)
        cleaned = _keep_components_overlapping(cleaned, mask)
        if not bool(cleaned.any()):
            return mask.copy()

        refined = mask.copy()
        refined[band] = cleaned[band]
        refined[inside_core] = True
        refined[outside] = False
        return refined.astype(bool)


def _rgb_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got {arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}.")
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _boundary_regions(
    mask: np.ndarray,
    band_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    radius = max(1, int(band_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    mask_u8 = mask_bool.astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel) > 0
    eroded = cv2.erode(mask_u8, kernel) > 0
    band = dilated & ~eroded
    outside = ~dilated
    return band, eroded, outside


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    work = np.pad(mask_bool.astype(np.uint8), 1, mode="constant", constant_values=0)
    flood_mask = np.zeros((work.shape[0] + 2, work.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(work, flood_mask, (0, 0), 2)
    background = work[1:-1, 1:-1] == 2
    holes = ~background & ~mask_bool
    return mask_bool | holes


def _edge_aware_mask_score(
    image: np.ndarray, mask: np.ndarray, band_px: int
) -> np.ndarray:
    source = np.asarray(mask, dtype=np.float32)
    radius = max(1, int(band_px))
    guide = np.asarray(image, dtype=np.float32) / 255.0
    try:
        ximgproc = getattr(cv2, "ximgproc")
        guided_filter = getattr(ximgproc, "guidedFilter")
        score = guided_filter(
            guide=guide,
            src=source,
            radius=radius,
            eps=1e-3,
        )
        return np.asarray(score, dtype=np.float32)
    except Exception:
        diameter = max(3, radius * 2 + 1)
        score = cv2.bilateralFilter(
            source,
            d=diameter,
            sigmaColor=0.25,
            sigmaSpace=float(radius),
        )
        return np.asarray(score, dtype=np.float32)


def _keep_components_overlapping(
    candidate: np.ndarray, original: np.ndarray
) -> np.ndarray:
    candidate_u8 = np.asarray(candidate, dtype=np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate_u8, connectivity=8)
    if num_labels <= 1:
        return candidate.astype(bool)
    keep = np.zeros_like(candidate_u8, dtype=bool)
    original_bool = np.asarray(original, dtype=bool)
    for label_idx in range(1, num_labels):
        component = labels == label_idx
        if bool(np.logical_and(component, original_bool).any()):
            keep |= component
    return keep
