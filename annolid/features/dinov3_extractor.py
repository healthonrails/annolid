from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Union
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
try:  # transformers is an optional dependency
    from transformers import AutoImageProcessor, AutoModel
except ImportError as exc:  # pragma: no cover - informative path
    AutoImageProcessor = None
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# ImageNet normalization used by DINOv3 ViTs
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Dinov3Config:
    """Configuration for DINOv3 feature extraction.

    Attributes
    ----------
    model_name: Hugging Face model id or legacy torch.hub alias.
        Recommended checkpoints:
            - "facebook/dinov3-vits16-pretrain-lvd1689m"
            - "facebook/dinov3-vits16plus-pretrain-lvd1689m"
            - "facebook/dinov3-vitb16-pretrain-lvd1426"
            - "facebook/dinov3-vitl16-pretrain-lvd1689m"
            - "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        Legacy aliases such as "dinov3_vitl16" are translated automatically.
    short_side: int
        Target short side before snapping to multiples of patch size.
    patch_size: int
        Expected patch size (will be validated vs model.patch_size if present).
    cache_dir: Optional[str]
        Optional cache directory for Hugging Face `from_pretrained` calls.
        If None, uses the environment variable DINOV3_LOCATION when defined.
    device: Optional[str]
        "cuda", "mps", or "cpu"; if None select automatically.
    use_amp: bool
        Enable autocast on CUDA to reduce VRAM.
    return_layer: Literal["last", "all"]
        Whether `extract()` returns only the last layer or all requested layers.
    layers: Optional[Iterable[int]]
        Which intermediate transformer blocks to retrieve (0-indexed). Negative
        indices are supported (e.g., -1 for the last block). If None, uses the
        full depth reported by the checkpoint configuration.
    """

    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    short_side: int = 768
    patch_size: int = 16
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    use_amp: bool = True
    return_layer: Literal["last", "all"] = "last"
    layers: Optional[Iterable[int]] = None


LEGACY_ALIAS_TO_HF_ID = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1426",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


class Dinov3FeatureExtractor:
    """Dense feature extractor for DINOv3 ViT backbones via Hugging Face Transformers.

    Public API:
      - extract(image, return_type) -> [D, H, W] or [L, D, H, W]
      - quantize_mask(mask_image) -> [H_patches, W_patches]
      - map between patches and image pixels
    """

    def __init__(self, config: Optional[Dinov3Config] = None) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ImportError(
                "Dinov3FeatureExtractor requires the optional dependency 'transformers'. "
                "Install it with: pip install 'transformers>=4.39'"
            ) from _TRANSFORMERS_IMPORT_ERROR
        self.cfg = config or Dinov3Config()

        # Resolve model identifier and cache directory
        self.model_id = LEGACY_ALIAS_TO_HF_ID.get(
            self.cfg.model_name, self.cfg.model_name)
        cache_env = os.getenv("DINOV3_LOCATION")
        self.cache_dir = self.cfg.cache_dir or cache_env or None

        # Select device
        self.device = self._select_device(self.cfg.device)
        logger.info("Using device: %s", self.device)

        # Load processor + model
        self.processor, self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Determine feature geometry from model config
        self.patch_size = int(
            getattr(self.model.config, "patch_size", self.cfg.patch_size))
        if self.patch_size != self.cfg.patch_size:
            logger.info(
                "Using patch_size=%s from model config (override config value %s)",
                self.patch_size,
                self.cfg.patch_size,
            )

        cls_tokens = 1 if getattr(
            self.model.config, "use_cls_token", True) else 0
        register_tokens = int(
            getattr(self.model.config, "num_register_tokens", 0))
        self._num_special_tokens = cls_tokens + register_tokens

        self.num_hidden_layers = int(
            getattr(self.model.config, "num_hidden_layers", 0))
        if self.num_hidden_layers <= 0:
            raise RuntimeError("DINOv3 model did not report num_hidden_layers")

        self._layers = tuple(self.cfg.layers) if self.cfg.layers is not None else tuple(
            range(self.num_hidden_layers))

        # Store normalization from processor if available
        proc_mean = getattr(self.processor, "image_mean", None)
        proc_std = getattr(self.processor, "image_std", None)
        self._mean = tuple(proc_mean) if proc_mean else IMAGENET_MEAN
        self._std = tuple(proc_std) if proc_std else IMAGENET_STD

        # Avg-pool kernel for mask quantization into patch grid
        self._patch_quant_filter = torch.nn.Conv2d(
            1, 1, self.patch_size, stride=self.patch_size, bias=False)
        with torch.no_grad():
            self._patch_quant_filter.weight.data.fill_(
                1.0 / (self.patch_size * self.patch_size))

    # --------------------------
    # Device & loading helpers
    # --------------------------
    @staticmethod
    def _select_device(preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        try:
            logger.info("Loading DINOv3 model '%s' via Hugging Face Transformers",
                        self.model_id)
            processor = AutoImageProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir)
            model = AutoModel.from_pretrained(
                self.model_id, cache_dir=self.cache_dir)
            return processor, model
        except Exception as exc:  # pragma: no cover - informative path
            hint = (
                "Failed to load DINOv3 via transformers. Ensure 'transformers' is installed "
                "and the requested checkpoint is available (network or local cache)."
            )
            raise RuntimeError(f"DINOv3 load error: {exc}. {hint}")

    # --------------------------
    # Preprocessing utilities
    # --------------------------
    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray], color_space: Literal["RGB", "BGR"] = "RGB") -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("ndarray image must be HxWx3")
            if color_space.upper() == "BGR":
                image = image[:, :, ::-1]
            img = Image.fromarray(image)
        else:
            raise TypeError("image must be PIL.Image or np.ndarray")
        # Normalize EXIF (camera rotations)
        return ImageOps.exif_transpose(img.convert("RGB"))

    def _compute_resized_hw(self, w: int, h: int) -> Tuple[int, int]:
        """Aspect-preserving resize with both sides snapped to patch multiple.
        Returns (new_h, new_w).
        """
        short = int(self.cfg.short_side)

        # scale so short side == `short`
        if h <= w:
            scale = short / h
        else:
            scale = short / w
        new_w = max(self.patch_size, int(
            math.ceil((w * scale) / self.patch_size) * self.patch_size))
        new_h = max(self.patch_size, int(
            math.ceil((h * scale) / self.patch_size) * self.patch_size))
        return new_h, new_w

    def _preprocess(self, pil: Image.Image) -> torch.Tensor:
        new_h, new_w = self._compute_resized_hw(*pil.size)
        x = TF.to_tensor(pil).unsqueeze(0)
        x = TF.resize(x, size=[new_h, new_w], antialias=True)
        x = TF.normalize(x, mean=self._mean, std=self._std)
        return x.to(self.device, non_blocking=True)

    # --------------------------
    # Public API
    # --------------------------
    @torch.inference_mode()
    def extract(
        self,
        image: Union[Image.Image, np.ndarray],
        *,
        color_space: Literal["RGB", "BGR"] = "RGB",
        return_type: Literal["torch", "numpy"] = "torch",
        return_layer: Optional[Literal["last", "all"]] = None,
        normalize: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract dense features.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Input image. If ndarray, defaults to RGB ordering unless color_space="BGR".
        color_space : {"RGB", "BGR"}
            Only used when `image` is a numpy array.
        return_type : {"torch", "numpy"}
            Output array type.
        return_layer : {"last", "all"} or None
            Override constructor setting. "last" returns [D, H, W]. "all" returns [L, D, H, W].
        normalize : bool
            If True, returns L2-normalized channel features (recommended).
        """
        pil = self._to_pil(image, color_space=color_space)
        x = self._preprocess(pil)

        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

        ret_mode = return_layer or self.cfg.return_layer

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = self.model(
                pixel_values=x,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  # (num_layers + 1) tensors
        selected_grids = []
        num_layers = len(hidden_states) - 1
        for layer_idx in self._layers:
            resolved_idx = layer_idx
            if layer_idx < 0:
                resolved_idx = num_layers + layer_idx
            if resolved_idx < 0 or resolved_idx >= num_layers:
                raise IndexError(
                    f"Requested layer {layer_idx} outside available range -{num_layers}..{num_layers - 1}")
            tokens = hidden_states[resolved_idx + 1]
            grid = self._tokens_to_grid(
                tokens,
                spatial_hw=(x.shape[-2], x.shape[-1]),
                detach=True,
                normalize=normalize,
            )
            selected_grids.append(grid)

        if not selected_grids:
            raise RuntimeError("No layers selected for feature extraction")

        if ret_mode == "last":
            f = selected_grids[-1]
        else:
            f = torch.stack(selected_grids, dim=0)

        if return_type == "numpy":
            return f.numpy()
        return f

    def _tokens_to_grid(
        self,
        tokens: torch.Tensor,
        spatial_hw: Tuple[int, int],
        *,
        detach: bool,
        normalize: bool,
    ) -> torch.Tensor:
        if detach:
            tokens = tokens.detach()
        B, seq_len, dim = tokens.shape
        if B != 1:
            raise ValueError(
                "Only batch size 1 is supported for feature extraction")
        H, W = spatial_hw
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        expected = grid_h * grid_w
        special = self._num_special_tokens
        if seq_len == expected + special:
            tokens = tokens[:, special:, :]
        elif seq_len == expected:
            pass  # no special tokens to drop
        else:
            raise ValueError(
                f"Unexpected sequence length {seq_len}; expected {expected} or {expected}+{special} (special tokens)")
        grid = tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w).to("cpu")
        grid = grid.squeeze(0)
        if normalize:
            flat = grid.view(dim, -1)
            flat = F.normalize(flat, dim=0)
            grid = flat.view(dim, grid_h, grid_w)
        return grid

    @torch.inference_mode()
    def quantize_mask(self, mask_image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Quantize an RGBA/RGB/gray mask onto the patch grid by uniform box averaging.

        Returns
        -------
        torch.Tensor
            [H_patches, W_patches] with values in [0,1] on CPU.
        """
        if isinstance(mask_image, Image.Image):
            pil = ImageOps.exif_transpose(mask_image)
            arr = np.array(pil)
        elif isinstance(mask_image, np.ndarray):
            arr = mask_image
        else:
            raise TypeError("mask_image must be PIL.Image or np.ndarray")

        # Extract a single-channel mask in [0,1]
        if arr.ndim == 3 and arr.shape[2] == 4:
            mask = arr[:, :, 3]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mask = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :,
                    1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
        else:
            mask = arr.squeeze()

        tens = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0

        # Resize to match feature grid snaps
        h, w = mask.shape[:2]
        new_h, new_w = self._compute_resized_hw(w, h)
        tens = TF.resize(tens, size=[new_h, new_w], antialias=True)

        # Average into patch cells
        q = self._patch_quant_filter.to(tens.device)(tens)
        return q.squeeze(0).squeeze(0).cpu()

    # --------------------------
    # Coordinate mapping helpers
    # --------------------------
    def grid_shape_for_image(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Given an input image size (W, H), return (H_patches, W_patches) after preprocessing."""
        new_h, new_w = self._compute_resized_hw(*image_size)
        return new_h // self.patch_size, new_w // self.patch_size

    @staticmethod
    def to_patch_coords(
        pts_xy: np.ndarray,
        feats_hw: Tuple[int, int],
        patch_size: int,
        image_size_px: Tuple[int, int],
    ) -> np.ndarray:
        """Map image pixel coordinates → patch indices (row, col).

        Parameters
        ----------
        pts_xy : (N,2)
            Pixel coordinates in the **resized** image space used for the model.
        feats_hw : (H_p, W_p)
            Feature grid shape.
        patch_size : int
            Patch size.
        image_size_px : (W, H)
            The **resized** image size (after preprocessing).
        """
        h_p, w_p = feats_hw
        W, H = image_size_px
        px_x = W / w_p
        px_y = H / h_p
        cols = np.clip((pts_xy[:, 0] / px_x).astype(int), 0, w_p - 1)
        rows = np.clip((pts_xy[:, 1] / px_y).astype(int), 0, h_p - 1)
        return np.stack([rows, cols], axis=1)

    @staticmethod
    def to_image_coords(patch_rc: np.ndarray, patch_size: int) -> np.ndarray:
        """Map patch indices (row, col) → pixel centers (x, y) in the **resized** image space."""
        rows = patch_rc[:, 0].astype(float)
        cols = patch_rc[:, 1].astype(float)
        xs = (cols + 0.5) * patch_size
        ys = (rows + 0.5) * patch_size
        return np.stack([xs, ys], axis=1)
