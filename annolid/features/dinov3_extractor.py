import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Dinov3Config:
    model_name: str = "dinov3_vitl16"
    image_size: int = 768
    patch_size: int = 16
    repo_or_dir: Optional[str] = None
    device: Optional[str] = None
    use_amp: bool = True
    return_layer: Literal["last", "all"] = "last"


class Dinov3FeatureExtractor:
    """DINOv3 dense feature extractor.

    - Loads a DINOv3 ViT via torch.hub using either a local repo path provided by
      environment variable `DINOV3_LOCATION` or the public GitHub repo
      `facebookresearch/dinov3`.
    - Preprocesses images to RGB, resizes so that both dimensions are multiples of
      `patch_size`, normalizes by ImageNet stats, and runs a forward pass to
      return dense features with shape [D, H, W].
    - Provides utilities to quantize a mask onto the patch grid and convert
      between image coordinates and patch indices.
    """

    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }

    def __init__(self, config: Optional[Dinov3Config] = None):
        self.cfg = config or Dinov3Config()
        self.patch_size = int(self.cfg.patch_size)

        repo_env = os.getenv("DINOV3_LOCATION")
        self.repo_or_dir = (
            self.cfg.repo_or_dir
            if self.cfg.repo_or_dir is not None
            else (repo_env if repo_env is not None else "facebookresearch/dinov3")
        )

        self.device = self._select_device(self.cfg.device)
        self.n_layers = self.MODEL_TO_NUM_LAYERS.get(self.cfg.model_name, 24)
        self.model = self._load_model()

        self._patch_quant_filter = torch.nn.Conv2d(
            1, 1, self.patch_size, stride=self.patch_size, bias=False
        )
        with torch.no_grad():
            self._patch_quant_filter.weight.data.fill_(
                1.0 / (self.patch_size * self.patch_size)
            )

    def _select_device(self, preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # macOS MPS fallback (best-effort); if unsupported at runtime, torch will raise
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        try:
            source = "local" if self.repo_or_dir != "facebookresearch/dinov3" else "github"
            model = torch.hub.load(
                repo_or_dir=self.repo_or_dir,
                model=self.cfg.model_name,
                source=source,
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            hint = (
                "Failed to load DINOv3 model. Set DINOV3_LOCATION to a local clone "
                "or install via: pip install git+https://github.com/facebookresearch/dinov3"
            )
            raise RuntimeError(f"DINOv3 load error: {e}. {hint}")

    @staticmethod
    def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr[:, :, ::-1]

    def _resize_keep_aspect_to_multiple(self, w: int, h: int) -> Tuple[int, int]:
        target_h = int(self.cfg.image_size)
        target_w = int(round(w * (target_h / h)))
        # snap to multiples of patch_size
        target_h = (target_h // self.patch_size) * self.patch_size
        target_w = (target_w // self.patch_size) * self.patch_size
        target_h = max(self.patch_size, target_h)
        target_w = max(self.patch_size, target_w)
        return target_w, target_h

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 BGR array")
        rgb = self._bgr_to_rgb(image_bgr)
        h, w = rgb.shape[:2]
        tw, th = self._resize_keep_aspect_to_multiple(w, h)
        img = TF.to_tensor(rgb)
        img = TF.resize(img, size=[th, tw])
        img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img.unsqueeze(0).to(self.device, non_blocking=True)

    @torch.inference_mode()
    def extract(self, image_bgr: np.ndarray, return_type: Literal["torch", "numpy"] = "torch"):
        """Extract dense features for an image.

        Returns a tensor/array of shape [D, H, W] on CPU.
        """
        x = self._preprocess(image_bgr)
        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            feats = self.model.get_intermediate_layers(
                x, n=range(self.n_layers), reshape=True, norm=True
            )
            f = feats[-1].squeeze(0).detach()  # [D, H, W]
        f = f.to("cpu")
        if return_type == "numpy":
            return f.numpy()
        return f

    @torch.inference_mode()
    def quantize_mask(self, mask_image: np.ndarray) -> torch.Tensor:
        """Quantize a mask to the patch grid using a uniform box blur.

        Accepts an RGBA or RGB/Binary mask image as HxWx{1,3,4} uint8.
        Returns a tensor [H_p, W_p] with mask values in [0,1].
        """
        if mask_image.ndim == 3 and mask_image.shape[2] == 4:
            # alpha channel as mask
            mask = mask_image[:, :, 3]
        elif mask_image.ndim == 3 and mask_image.shape[2] == 3:
            # convert to grayscale heuristic
            mask = (0.299 * mask_image[:, :, 0] + 0.587 * mask_image[:, :, 1] + 0.114 * mask_image[:, :, 2]).astype(
                np.uint8
            )
        else:
            mask = mask_image.squeeze()

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        tens = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
        h, w = mask.shape[:2]
        tw, th = self._resize_keep_aspect_to_multiple(w, h)
        tens = TF.resize(tens, size=[th, tw])
        q = self._patch_quant_filter.to(tens.device)(tens)
        return q.squeeze(0).squeeze(0).cpu()

    def to_patch_coords(self, pts_xy: np.ndarray, feats_hw: Tuple[int, int]) -> np.ndarray:
        """Map image pixel coordinates to patch indices (row, col).

        pts_xy: Nx2 in image pixel space (x, y)
        feats_hw: (H_patches, W_patches)
        """
        h_p, w_p = feats_hw
        scale_x = (w_p * self.patch_size)
        scale_y = (h_p * self.patch_size)
        # integer index of the patch containing the point
        cols = np.clip((pts_xy[:, 0] / self.patch_size).astype(int), 0, w_p - 1)
        rows = np.clip((pts_xy[:, 1] / self.patch_size).astype(int), 0, h_p - 1)
        return np.stack([rows, cols], axis=1)

    def to_image_coords(self, patch_rc: np.ndarray) -> np.ndarray:
        """Map patch indices (row, col) to image pixel centers (x, y)."""
        rows = patch_rc[:, 0].astype(float)
        cols = patch_rc[:, 1].astype(float)
        xs = (cols + 0.5) * self.patch_size
        ys = (rows + 0.5) * self.patch_size
        return np.stack([xs, ys], axis=1)

    @staticmethod
    @lru_cache(maxsize=1)
    def hub_location_info() -> str:
        loc = os.getenv("DINOV3_LOCATION") or "facebookresearch/dinov3"
        src = "local" if os.getenv("DINOV3_LOCATION") else "github"
        return f"{loc} (source={src})"
