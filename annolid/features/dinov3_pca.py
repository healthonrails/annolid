from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, List

import numpy as np
import torch
from PIL import Image
import colorsys

from annolid.features.dinov3_extractor import Dinov3Config, Dinov3FeatureExtractor


@dataclass
class PCAMapResult:
    """Container for DINOv3 PCA visualizations."""

    feature_rgb: np.ndarray  # (H_p, W_p, 3) floats in [0,1]
    output_rgb: np.ndarray  # (H_out, W_out, 3) floats in [0,1]
    feature_hw: Tuple[int, int]
    resized_hw: Tuple[int, int]
    output_mode: Literal["feature", "resized", "input", "custom"]
    patch_size: int
    input_size: Tuple[int, int]
    cluster_rgb: Optional[np.ndarray] = None
    cluster_labels: Optional[List[str]] = None

    def as_pil(self) -> Image.Image:
        """Return the output visualization as a PIL image."""
        arr = np.clip(self.output_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def save(self, path: Union[str, Path]) -> None:
        """Persist the PCA map to disk."""
        self.as_pil().save(path)


def features_to_pca_rgb(
    features: Union[torch.Tensor, np.ndarray],
    *,
    num_components: int = 3,
    clip_percentile: Optional[float] = 1.0,
    eps: float = 1e-6,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project dense features to a 3-channel PCA embedding.

    Parameters
    ----------
    features:
        Tensor shaped [D, H, W] on CPU.
    num_components:
        Number of principal components to output (<=3 recommended for RGB).
    clip_percentile:
        Optional symmetric percentile clipping applied per channel to reduce
        outliers before min/max normalization. Pass ``None`` to disable.
    eps:
        Numerical stability term for normalization.
    mask:
        Optional boolean mask on the feature grid (H_p, W_p). When provided,
        PCA components are estimated using only the masked spatial positions.
    """
    if isinstance(features, np.ndarray):
        feats = torch.from_numpy(features)
    elif isinstance(features, torch.Tensor):
        feats = features.detach().cpu()
    else:
        raise TypeError("features must be a torch.Tensor or np.ndarray")

    if feats.dim() != 3:
        raise ValueError("Expected features with shape [D, H, W]")

    D, H, W = feats.shape
    flattened = feats.reshape(D, -1).T  # (N, D)
    if flattened.shape[0] < 2:
        raise ValueError("PCA requires at least two spatial positions")

    mask_flat: Optional[torch.Tensor] = None
    mean_vec = flattened.mean(dim=0, keepdim=True)
    if mask is not None:
        if mask.shape != (H, W):
            raise ValueError("mask shape must match feature grid")
        mask_flat = torch.from_numpy(mask.astype(bool).reshape(-1))
        valid = int(mask_flat.sum().item())
        if valid < 2:
            raise ValueError("PCA mask must cover at least two spatial positions")
        mean_vec = flattened[mask_flat].mean(dim=0, keepdim=True)

    centered = flattened - mean_vec
    q = min(num_components, centered.shape[0], centered.shape[1])
    if q <= 0:
        raise ValueError("Cannot compute PCA with zero components")

    if mask_flat is not None:
        pca_source = centered[mask_flat]
    else:
        pca_source = centered

    # Low-rank PCA for efficiency on large feature grids
    U, S, V = torch.pca_lowrank(pca_source, q=q, center=False)
    projected = centered @ V[:, :q]  # (N, q)

    arr = projected.reshape(H, W, q).cpu().numpy()
    if q < num_components:
        pad = np.zeros((H, W, num_components - q), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=2)

    arr = arr[:, :, :num_components]
    arr_flat = arr.reshape(-1, num_components)

    if mask_flat is not None:
        reference = arr_flat[mask_flat.numpy()]
    else:
        reference = arr_flat

    if clip_percentile is not None:
        if not (0.0 <= clip_percentile < 50.0):
            raise ValueError("clip_percentile must be in [0, 50)")
        lower = np.percentile(reference, clip_percentile, axis=0)
        upper = np.percentile(reference, 100.0 - clip_percentile, axis=0)
    else:
        lower = reference.min(axis=0)
        upper = reference.max(axis=0)

    scale = np.maximum(upper - lower, eps)
    arr_flat = np.clip(arr_flat, lower, upper)
    arr_norm = (arr_flat - lower) / scale
    return arr_norm.reshape(H, W, num_components).astype(np.float32)


class Dinov3PCAMapper:
    """Utility to produce PCA feature visualizations for DINOv3."""

    def __init__(
        self,
        extractor: Optional[Dinov3FeatureExtractor] = None,
        *,
        num_components: int = 3,
        clip_percentile: Optional[float] = 1.0,
        feature_resample: int = Image.BILINEAR,
    ) -> None:
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        self.extractor = extractor or Dinov3FeatureExtractor(Dinov3Config())
        self.num_components = num_components
        self.clip_percentile = clip_percentile
        self.feature_resample = feature_resample

    @torch.inference_mode()
    def map_image(
        self,
        image: Union[Image.Image, np.ndarray],
        *,
        color_space: Literal["RGB", "BGR"] = "RGB",
        output_size: Literal["feature", "resized", "input"] = "input",
        custom_size: Optional[Tuple[int, int]] = None,
        return_type: Literal["pil", "array"] = "pil",
        normalize_features: bool = True,
        mask: Optional[np.ndarray] = None,
        cluster_k: Optional[int] = None,
    ) -> PCAMapResult:
        """Generate a PCA map for an input image.

        Parameters
        ----------
        image:
            Input PIL image or numpy RGB/BGR array.
        color_space:
            Interpretation for numpy arrays (ignored for PIL input).
        output_size:
            Desired output resolution: ``"feature"`` keeps the patch grid,
            ``"resized"`` upsamples to the model's processed size, and
            ``"input"`` resizes back to the original input size.
        custom_size:
            Explicit (width, height) override. Only used when ``output_size`` is
            ``"input"`` and you want a custom target.
        return_type:
            ``"pil"`` returns the final map as a PIL.Image via `PCAMapResult`.
            ``"array"`` leaves data as np.ndarray inside the result (PIL always
            available via ``.as_pil()`` regardless).
        normalize_features:
            Whether to L2-normalize per-location features before PCA. Defaults
            to True for consistency with typical DINO usage.
        mask:
            Optional boolean or binary array matching the original image size.
            When provided, PCA components are fitted using only masked pixels
            and the returned visualization is derived from that sub-region.
        cluster_k:
            When greater than 1, unsupervised clustering is performed over the
            PCA feature grid (respecting the optional mask). The resulting
            cluster overlay replaces the standard PCA coloring.
        """
        pil = self.extractor._to_pil(image, color_space=color_space)
        mask_bool: Optional[np.ndarray] = None
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.shape != (pil.height, pil.width):
                raise ValueError("mask shape must match the input image size")
            mask_bool = mask_arr.astype(bool)

        feats = self.extractor.extract(
            pil,
            color_space=color_space,
            return_type="torch",
            return_layer="last",
            normalize=normalize_features,
        )

        mask_patch = None
        if mask_bool is not None:
            quantized = self.extractor.quantize_mask(mask_bool.astype(np.uint8) * 255)
            mask_patch = quantized.numpy() >= 0.5

        pca_feat = features_to_pca_rgb(
            feats,
            num_components=self.num_components,
            clip_percentile=self.clip_percentile,
            mask=mask_patch,
        )

        h_p, w_p = pca_feat.shape[:2]
        patch_sz = self.extractor.patch_size
        resized_hw = (h_p * patch_sz, w_p * patch_sz)

        pca_output_arr = self._resize_feature_image(
            feature=pca_feat,
            output_size=output_size,
            custom_size=custom_size,
            pil_size=pil.size,
            resized_hw=resized_hw,
        )

        cluster_rgb = None
        cluster_output_arr = None
        cluster_labels: List[str] = []
        if cluster_k is not None and cluster_k > 1:
            cluster_rgb, cluster_output_arr, cluster_labels = (
                self._cluster_pca_features(
                    pca_feat=pca_feat,
                    mask_patch=mask_patch,
                    cluster_k=cluster_k,
                    output_size=output_size,
                    custom_size=custom_size,
                    pil_size=pil.size,
                    resized_hw=resized_hw,
                )
            )

        output_arr = (
            cluster_output_arr if cluster_output_arr is not None else pca_output_arr
        )

        if return_type not in {"array", "pil"}:
            raise ValueError(f"Unsupported return_type '{return_type}'")

        return PCAMapResult(
            feature_rgb=pca_feat,
            output_rgb=output_arr,
            feature_hw=(h_p, w_p),
            resized_hw=resized_hw,
            output_mode=output_size if custom_size is None else "custom",
            patch_size=patch_sz,
            input_size=pil.size,
            cluster_rgb=cluster_rgb,
            cluster_labels=cluster_labels if cluster_labels else None,
        )

    def _resize_feature_image(
        self,
        *,
        feature: np.ndarray,
        output_size: Literal["feature", "resized", "input"],
        custom_size: Optional[Tuple[int, int]],
        pil_size: Tuple[int, int],
        resized_hw: Tuple[int, int],
    ) -> np.ndarray:
        feature_img = Image.fromarray(
            np.clip(feature * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )
        if output_size == "feature":
            return feature.astype(np.float32)
        if output_size == "resized":
            output_img = feature_img.resize(
                (resized_hw[1], resized_hw[0]), resample=self.feature_resample
            )
        elif output_size == "input":
            target_size = custom_size or pil_size
            output_img = feature_img.resize(target_size, resample=self.feature_resample)
        else:
            raise ValueError(f"Unsupported output_size '{output_size}'")
        return np.asarray(output_img).astype(np.float32) / 255.0

    def _cluster_pca_features(
        self,
        *,
        pca_feat: np.ndarray,
        mask_patch: Optional[np.ndarray],
        cluster_k: int,
        output_size: Literal["feature", "resized", "input"],
        custom_size: Optional[Tuple[int, int]],
        pil_size: Tuple[int, int],
        resized_hw: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        H, W, C = pca_feat.shape
        flat = pca_feat.reshape(-1, C)

        if mask_patch is not None:
            mask_flat = mask_patch.reshape(-1).astype(bool)
            candidate_idx = np.flatnonzero(mask_flat)
        else:
            candidate_idx = np.arange(flat.shape[0])

        if candidate_idx.size == 0:
            return None, None, []

        k = int(max(2, cluster_k))
        if candidate_idx.size < k:
            k = int(candidate_idx.size)
        if k < 2:
            return None, None, []

        subset = flat[candidate_idx]
        labels_subset = self._kmeans(subset, k)

        full_labels = np.full(flat.shape[0], fill_value=-1, dtype=np.int32)
        full_labels[candidate_idx] = labels_subset
        label_image = full_labels.reshape(H, W)

        palette = self._generate_palette(k)
        cluster_rgb = np.zeros((H, W, 3), dtype=np.float32)
        for idx in range(k):
            cluster_rgb[label_image == idx] = palette[idx]

        cluster_output_arr = self._resize_feature_image(
            feature=cluster_rgb,
            output_size=output_size,
            custom_size=custom_size,
            pil_size=pil_size,
            resized_hw=resized_hw,
        )

        labels = [f"cluster_{i}" for i in range(k)]
        return cluster_rgb, cluster_output_arr, labels

    @staticmethod
    def _kmeans(features: np.ndarray, k: int, iterations: int = 40) -> np.ndarray:
        rng = np.random.default_rng()
        n_samples = features.shape[0]
        if k > n_samples:
            raise ValueError("k cannot exceed number of samples")

        centroids = features[rng.choice(n_samples, size=k, replace=False)]
        labels = np.zeros(n_samples, dtype=np.int32)

        for _ in range(iterations):
            distances = np.linalg.norm(
                features[:, None, :] - centroids[None, :, :], axis=2
            )
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centroids[idx] = features[mask].mean(axis=0)
                else:
                    centroids[idx] = features[rng.integers(0, n_samples)]

        return labels

    @staticmethod
    def _generate_palette(k: int) -> np.ndarray:
        colors = []
        for idx in range(k):
            hue = idx / float(k)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
            colors.append((r, g, b))
        return np.array(colors, dtype=np.float32)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate DINOv3 PCA feature maps")
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument(
        "--model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id or legacy alias",
    )
    ap.add_argument(
        "--short-side",
        type=int,
        default=768,
        help="Target short side before patch snapping (higher = higher resolution)",
    )
    ap.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], default=None, help="Compute device"
    )
    ap.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of principal components to visualize (<=3)",
    )
    ap.add_argument(
        "--clip-percentile",
        type=float,
        default=1.0,
        help="Percentile clipping to stabilize colors (set to 0 to disable)",
    )
    ap.add_argument(
        "--output-size",
        choices=["feature", "resized", "input"],
        default="input",
        help="Resolution of the PCA visualization",
    )
    ap.add_argument(
        "--output",
        default="pca_map.png",
        help="Path to the output visualization (PNG)",
    )
    ap.add_argument(
        "--save-npy",
        help="Optional path to store the raw PCA feature grid as .npy",
    )
    ap.add_argument(
        "--cluster-k",
        type=int,
        default=0,
        help="Optional number of clusters to run on PCA features (>1)",
    )
    return ap


def main() -> None:
    logging_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_fmt)
    torch.set_grad_enabled(False)
    parser = _build_argparser()
    args = parser.parse_args()

    cfg = Dinov3Config(
        model_name=args.model,
        short_side=args.short_side,
        device=args.device,
    )
    mapper = Dinov3PCAMapper(
        Dinov3FeatureExtractor(cfg),
        num_components=args.components,
        clip_percentile=None if args.clip_percentile <= 0 else args.clip_percentile,
    )

    img = Image.open(args.image).convert("RGB")
    logging.info("Generating PCA map for %s", args.image)
    cluster_k = args.cluster_k if args.cluster_k and args.cluster_k > 1 else None
    result = mapper.map_image(img, output_size=args.output_size, cluster_k=cluster_k)
    result.save(args.output)
    logging.info("Saved PCA visualization to %s", args.output)

    if args.save_npy:
        np.save(args.save_npy, result.feature_rgb)
        logging.info("Saved PCA feature grid to %s", args.save_npy)


if __name__ == "__main__":  # pragma: no cover
    main()
