"""Dense and crop-level feature extractors for Google and on-device vision models.

Supports:
- ``google/embeddinggemma-2`` (Gemma 4 / EmbeddingGemma 2 multimodal architecture:
  selective vision-tower loading with ``audio_config=None``, strict ``bfloat16`` /
  ``float32`` execution avoiding ``float16`` ``NaN`` overflow, 2D
  ``pixel_position_ids`` support, unpooled/pooled spatial grid alignment, and
  Matryoshka Representation Learning truncation ``{128, 256, 512, 768}``).
- ``google/siglip2-*`` and ``google/siglip-*`` (vision-tower extraction from
  two-tower checkpoints, ``use_cls_token=False``, and ``interpolate_pos_encoding=True``
  for aspect-preserving video frames).
- On-device CNN/hybrid pyramids via ``timm/mobilenetv4_*`` and
  ``torchvision/mobilenet_v3_*`` (multi-scale stride-8 + stride-16 feature fusion
  for fast CPU/MPS/edge keypoint tracking).
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from annolid.features.dino_models import resolve_dino_model_id
from annolid.features.dinov3_extractor import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Dinov3Config,
    Dinov3FeatureExtractor,
)

logger = logging.getLogger(__name__)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def apply_matryoshka_truncation(
    features: torch.Tensor,
    matryoshka_dim: Optional[int],
    *,
    channel_dim: int = 0,
    normalize: bool = True,
) -> torch.Tensor:
    """Apply Matryoshka prefix channel truncation and optional L2 renormalization."""
    if matryoshka_dim is not None:
        target_dim = int(matryoshka_dim)
        if target_dim <= 0:
            raise ValueError(f"matryoshka_dim must be positive, got {matryoshka_dim!r}")
        current_dim = int(features.shape[channel_dim])
        if target_dim < current_dim:
            indices = torch.arange(target_dim, device=features.device)
            features = torch.index_select(features, dim=channel_dim, index=indices)
    if normalize:
        features = F.normalize(features.float(), dim=channel_dim, eps=1e-12)
    return features


class _TorchvisionMobileNetPyramid(nn.Module):
    """Lightweight multi-scale feature wrapper over ``torchvision`` MobileNetV3."""

    def __init__(self, features: nn.Sequential, stage_indices: Tuple[int, int]) -> None:
        super().__init__()
        self.features = features
        self.stage_indices = stage_indices

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        target_set = set(self.stage_indices)
        max_idx = max(self.stage_indices)
        outputs: dict[int, torch.Tensor] = {}
        cur = x
        for idx, layer in enumerate(self.features):
            cur = layer(cur)
            if idx in target_set:
                outputs[idx] = cur
            if idx >= max_idx:
                break
        return [outputs[idx] for idx in self.stage_indices if idx in outputs]


class GoogleVisionFeatureExtractor(Dinov3FeatureExtractor):
    """Unified extractor for EmbeddingGemma 2, SigLIP 2, and MobileNetV4/V3."""

    def __init__(self, config: Optional[Dinov3Config] = None) -> None:
        self.cfg = config or Dinov3Config()
        self.model_id = resolve_dino_model_id(self.cfg.model_name)
        cache_env = os.getenv("DINOV3_LOCATION")
        self.cache_dir = self.cfg.cache_dir or cache_env or None

        self.device = self._select_device(self.cfg.device)
        self.backend_kind = self._detect_backend_kind(self.model_id)
        self._model_dtype = self._select_inference_dtype(self.device, self.backend_kind)
        self._pooling_kernel_size = 1
        self._snap_multiple = max(1, int(self.cfg.patch_size))
        self._accepts_pixel_position_ids = False
        self._accepts_interpolate_pos_encoding = False
        self._accepts_output_hidden_states = False
        self._full_model: Optional[nn.Module] = None

        logger.info(
            "Initializing GoogleVisionFeatureExtractor (%s) for '%s' on %s (dtype=%s)",
            self.backend_kind,
            self.model_id,
            self.device,
            self._model_dtype,
        )

        self.processor, self.model = self._load_model()
        self.model.to(device=self.device, dtype=self._model_dtype)
        self.model.eval()
        if self._full_model is not None and self._full_model is not self.model:
            self._full_model.to(device=self.device, dtype=self._model_dtype)
            self._full_model.eval()

        self._resolve_model_properties()

        default_mean = (
            SIGLIP_MEAN
            if self.backend_kind in ("embeddinggemma", "siglip")
            else IMAGENET_MEAN
        )
        default_std = (
            SIGLIP_STD
            if self.backend_kind in ("embeddinggemma", "siglip")
            else IMAGENET_STD
        )
        proc_mean = getattr(self.processor, "image_mean", None)
        proc_std = getattr(self.processor, "image_std", None)
        self._mean = tuple(float(v) for v in proc_mean) if proc_mean else default_mean
        self._std = tuple(float(v) for v in proc_std) if proc_std else default_std

        self._patch_quant_filter = nn.Conv2d(
            1, 1, self.patch_size, stride=self.patch_size, bias=False
        )
        with torch.no_grad():
            self._patch_quant_filter.weight.data.fill_(
                1.0 / float(self.patch_size * self.patch_size)
            )

    @staticmethod
    def _detect_backend_kind(
        model_id: str,
    ) -> Literal["embeddinggemma", "siglip", "timm", "torchvision"]:
        lowered = str(model_id or "").strip().lower()
        if "embeddinggemma" in lowered or "embedding_gemma" in lowered:
            return "embeddinggemma"
        if "siglip" in lowered:
            return "siglip"
        if lowered.startswith("torchvision/") or "mobilenet_v3" in lowered:
            return "torchvision"
        if lowered.startswith("timm/") or "mobilenet" in lowered:
            return "timm"
        return "siglip"

    @staticmethod
    def _select_inference_dtype(
        device: torch.device,
        backend_kind: str,
    ) -> torch.dtype:
        """Choose safe execution precision.

        ``EmbeddingGemma 2`` overflows in ``float16`` and requires ``bfloat16`` or
        ``float32`` precision.
        """
        if backend_kind == "embeddinggemma":
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float32
        return torch.float32

    def _load_model(self) -> Tuple[Any, nn.Module]:
        if self.backend_kind == "embeddinggemma":
            return self._load_embeddinggemma_model()
        if self.backend_kind == "siglip":
            return self._load_siglip_model()
        if self.backend_kind == "timm":
            return self._load_timm_pyramid_model()
        return self._load_torchvision_pyramid_model()

    @staticmethod
    def _extract_image_processor(processor_obj: Any) -> Any:
        if processor_obj is None:
            return None
        inner = getattr(processor_obj, "image_processor", None)
        return inner if inner is not None else processor_obj

    def _load_hf_processor(self, *, trust_remote_code: bool = False) -> Any:
        from transformers import AutoImageProcessor, AutoProcessor

        for loader in (AutoImageProcessor, AutoProcessor):
            try:
                proc = loader.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=trust_remote_code,
                )
                return self._extract_image_processor(proc)
            except Exception:
                continue
        return None

    def _load_embeddinggemma_model(self) -> Tuple[Any, nn.Module]:
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Google EmbeddingGemma 2 requires 'transformers>=5.19'. "
                "Install it with: pip install -U 'transformers>=5.19'"
            ) from exc

        config = None
        last_exc: Optional[Exception] = None
        used_remote_code = False
        for remote_code in (False, True):
            try:
                config = AutoConfig.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=remote_code,
                )
                used_remote_code = remote_code
                break
            except Exception as exc:
                last_exc = exc

        if config is None:
            msg = str(last_exc or "")
            if any(
                token in msg.lower()
                for token in (
                    "embedding_gemma",
                    "embeddinggemma",
                    "gemma4",
                    "model type",
                    "does not recognize",
                )
            ):
                raise RuntimeError(
                    f"Model '{self.model_id}' requires Hugging Face Transformers >= 5.19 "
                    f"for EmbeddingGemma 2 architecture support (current load error: {last_exc}). "
                    "Upgrade with: pip install -U 'transformers>=5.19'"
                ) from last_exc
            raise RuntimeError(
                f"Failed to load EmbeddingGemma 2 config for '{self.model_id}': {last_exc}"
            ) from last_exc

        # Selective modality loading: disable the ~300M audio tower for vision tracking.
        if hasattr(config, "audio_config"):
            config.audio_config = None

        processor = self._load_hf_processor(trust_remote_code=used_remote_code)

        model = None
        for kwargs in (
            {
                "config": config,
                "cache_dir": self.cache_dir,
                "dtype": self._model_dtype,
                "trust_remote_code": used_remote_code,
            },
            {
                "config": config,
                "cache_dir": self.cache_dir,
                "torch_dtype": self._model_dtype,
                "trust_remote_code": used_remote_code,
            },
            {
                "config": config,
                "cache_dir": self.cache_dir,
                "trust_remote_code": used_remote_code,
            },
        ):
            try:
                model = AutoModel.from_pretrained(self.model_id, **kwargs)
                break
            except TypeError:
                continue
            except Exception as exc:
                last_exc = exc
                break

        if model is None:
            raise RuntimeError(
                f"Failed to load EmbeddingGemma 2 checkpoint '{self.model_id}': {last_exc}. "
                "Ensure 'transformers>=5.19' is installed and the checkpoint is cached or reachable."
            ) from last_exc

        self._full_model = model
        vision_tower = (
            getattr(model, "vision_tower", None)
            or getattr(model, "vision_model", None)
            or model
        )
        return processor, vision_tower

    def _load_siglip_model(self) -> Tuple[Any, nn.Module]:
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "SigLIP/SigLIP 2 requires 'transformers'. "
                "Install it with: pip install 'transformers>=4.49'"
            ) from exc

        processor = self._load_hf_processor(trust_remote_code=False)
        try:
            full_model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SigLIP model '{self.model_id}': {exc}"
            ) from exc

        self._full_model = full_model
        vision_model = getattr(full_model, "vision_model", None) or full_model
        return processor, vision_model

    def _load_timm_pyramid_model(self) -> Tuple[Any, nn.Module]:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "MobileNetV4 on-device backbone requires 'timm'. "
                "Install it with: pip install 'timm>=1.0.0'"
            ) from exc

        arch = self.model_id
        if arch.lower().startswith("timm/"):
            arch = arch.split("/", 1)[1]
        pretrained = not bool(getattr(self.cfg, "_disable_pretrained", False))
        create_kwargs: dict[str, Any] = {
            "features_only": True,
            "out_indices": (2, 3),
            "pretrained": pretrained,
        }
        if self.cache_dir:
            sig = inspect.signature(timm.create_model)
            if "cache_dir" in sig.parameters:
                create_kwargs["cache_dir"] = str(self.cache_dir)
        try:
            model = timm.create_model(arch, **create_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load timm MobileNet model '{self.model_id}': {exc}"
            ) from exc

        data_cfg = getattr(model, "pretrained_cfg", {}) or {}
        proc = type(
            "_TimmNorm",
            (),
            {
                "image_mean": data_cfg.get("mean", IMAGENET_MEAN),
                "image_std": data_cfg.get("std", IMAGENET_STD),
            },
        )()
        return proc, model

    def _load_torchvision_pyramid_model(self) -> Tuple[Any, nn.Module]:
        try:
            from torchvision import models as tv_models
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Torchvision MobileNet backbone requires 'torchvision'."
            ) from exc

        arch = self.model_id.lower()
        if "/" in arch:
            arch = arch.split("/", 1)[1].strip()
        pretrained = not bool(getattr(self.cfg, "_disable_pretrained", False))

        prev_hub_dir = torch.hub.get_dir()
        try:
            if self.cache_dir:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                torch.hub.set_dir(str(self.cache_dir))
            if arch == "mobilenet_v3_large":
                weights = (
                    tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                )
                backbone = tv_models.mobilenet_v3_large(weights=weights)
                stage_indices = (6, 12)
            else:
                weights = (
                    tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                )
                backbone = tv_models.mobilenet_v3_small(weights=weights)
                stage_indices = (3, 8)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load torchvision MobileNet '{self.model_id}': {exc}"
            ) from exc
        finally:
            torch.hub.set_dir(prev_hub_dir)

        pyramid = _TorchvisionMobileNetPyramid(backbone.features, stage_indices)
        proc = type(
            "_TorchvisionNorm",
            (),
            {"image_mean": IMAGENET_MEAN, "image_std": IMAGENET_STD},
        )()
        return proc, pyramid

    def _resolve_model_properties(self) -> None:
        if self.backend_kind in ("timm", "torchvision"):
            # Stride-8 multi-scale CNN pyramid (2x finer spatial grid than 16px ViTs).
            self.patch_size = 8
            self._pooling_kernel_size = 1
            self._snap_multiple = 32
            self._num_special_tokens = 0
            self.num_hidden_layers = 2
            self._layers = (-1,)
            return

        root_cfg = getattr(self._full_model, "config", None)
        vision_cfg = (
            getattr(self.model, "config", None)
            or getattr(root_cfg, "vision_config", None)
            or root_cfg
        )
        if (
            hasattr(vision_cfg, "vision_config")
            and getattr(vision_cfg, "vision_config", None) is not None
        ):
            vision_cfg = vision_cfg.vision_config

        self.patch_size = int(getattr(vision_cfg, "patch_size", self.cfg.patch_size))
        self._pooling_kernel_size = max(
            1, int(getattr(vision_cfg, "pooling_kernel_size", 1) or 1)
        )
        self._snap_multiple = self.patch_size * self._pooling_kernel_size

        use_cls = bool(getattr(vision_cfg, "use_cls_token", False))
        reg_tokens = int(getattr(vision_cfg, "num_register_tokens", 0) or 0)
        self._num_special_tokens = (1 if use_cls else 0) + reg_tokens

        num_layers = int(getattr(vision_cfg, "num_hidden_layers", 0) or 0)
        if num_layers <= 0:
            encoder = getattr(self.model, "encoder", None)
            layers_seq = getattr(encoder, "layers", None) or getattr(
                self.model, "layers", None
            )
            if layers_seq is not None:
                num_layers = len(layers_seq)
        self.num_hidden_layers = max(1, num_layers)
        self._layers = (
            tuple(self.cfg.layers)
            if self.cfg.layers is not None
            else (self.num_hidden_layers - 1,)
        )

        try:
            sig = inspect.signature(self.model.forward)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            self._accepts_pixel_position_ids = "pixel_position_ids" in sig.parameters
            self._accepts_interpolate_pos_encoding = (
                "interpolate_pos_encoding" in sig.parameters
                or (has_var_kw and self.backend_kind == "siglip")
            )
            self._accepts_output_hidden_states = (
                "output_hidden_states" in sig.parameters or has_var_kw
            )
        except (TypeError, ValueError):
            self._accepts_pixel_position_ids = False
            self._accepts_interpolate_pos_encoding = False
            self._accepts_output_hidden_states = True

    def _compute_resized_hw(self, w: int, h: int) -> Tuple[int, int]:
        """Aspect-preserving resize snapped to ``_snap_multiple``."""
        short = int(self.cfg.short_side)
        scale = (short / h) if h <= w else (short / w)
        multiple = max(self.patch_size, int(self._snap_multiple))
        new_w = max(multiple, int(math.ceil((w * scale) / multiple) * multiple))
        new_h = max(multiple, int(math.ceil((h * scale) / multiple) * multiple))
        return new_h, new_w

    @staticmethod
    def _build_pixel_position_ids(
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct ``[1, H_g * W_g, 2]`` ``(col, row)`` patch coordinates."""
        rows = torch.arange(grid_h, device=device, dtype=torch.long)
        cols = torch.arange(grid_w, device=device, dtype=torch.long)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_c.reshape(-1), grid_r.reshape(-1)], dim=-1)
        return coords.unsqueeze(0)

    def _tokens_or_map_to_grid(
        self,
        tensor: torch.Tensor,
        spatial_hw: Tuple[int, int],
        *,
        normalize: bool,
    ) -> torch.Tensor:
        """Convert 3D token sequence or 4D feature map to ``[D, H_g, W_g]`` on CPU."""
        tensor = tensor.detach().float()
        h_px, w_px = spatial_hw
        target_h = max(1, h_px // self.patch_size)
        target_w = max(1, w_px // self.patch_size)

        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1 for dense grid extraction, got {tensor.shape[0]}"
                )
            # Detect NHWC vs NCHW
            if (
                tensor.shape[1]
                in (target_h, max(1, target_h // self._pooling_kernel_size))
                and tensor.shape[2]
                in (target_w, max(1, target_w // self._pooling_kernel_size))
                and tensor.shape[3] > tensor.shape[2]
            ):
                grid = tensor.permute(0, 3, 1, 2)
            else:
                grid = tensor
            if grid.shape[-2:] != (target_h, target_w):
                grid = F.interpolate(
                    grid,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            grid_3d = grid.squeeze(0).to("cpu")
            return apply_matryoshka_truncation(
                grid_3d,
                self.cfg.matryoshka_dim,
                channel_dim=0,
                normalize=normalize,
            )

        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(
                f"Expected [1, N, D] token tensor or [1, C, H, W] map, got {tuple(tensor.shape)}"
            )

        seq_len = int(tensor.shape[1])
        dim = int(tensor.shape[2])
        expected = target_h * target_w
        pooled_h = max(1, target_h // self._pooling_kernel_size)
        pooled_w = max(1, target_w // self._pooling_kernel_size)
        expected_pooled = pooled_h * pooled_w

        if (
            seq_len == expected + self._num_special_tokens
            and self._num_special_tokens > 0
        ):
            tokens = tensor[:, self._num_special_tokens :, :]
            grid_hw = (target_h, target_w)
        elif seq_len == expected:
            tokens = tensor
            grid_hw = (target_h, target_w)
        elif seq_len == expected + 1:
            tokens = tensor[:, 1:, :]
            grid_hw = (target_h, target_w)
        elif self._pooling_kernel_size > 1 and seq_len == expected_pooled:
            tokens = tensor
            grid_hw = (pooled_h, pooled_w)
        elif (
            self._pooling_kernel_size > 1
            and seq_len == expected_pooled + self._num_special_tokens
            and self._num_special_tokens > 0
        ):
            tokens = tensor[:, self._num_special_tokens :, :]
            grid_hw = (pooled_h, pooled_w)
        else:
            raise ValueError(
                f"Unexpected token sequence length {seq_len} for spatial_hw={spatial_hw} "
                f"(expected {expected} or pooled {expected_pooled})"
            )

        grid = tokens.transpose(1, 2).reshape(1, dim, grid_hw[0], grid_hw[1])
        if grid_hw != (target_h, target_w):
            grid = F.interpolate(
                grid,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        grid_3d = grid.squeeze(0).to("cpu")
        return apply_matryoshka_truncation(
            grid_3d,
            self.cfg.matryoshka_dim,
            channel_dim=0,
            normalize=normalize,
        )

    def _extract_cnn_pyramid(
        self,
        x: torch.Tensor,
        *,
        return_layer: str,
        normalize: bool,
    ) -> torch.Tensor:
        target_h = max(1, int(x.shape[-2]) // self.patch_size)
        target_w = max(1, int(x.shape[-1]) // self.patch_size)
        stages = self.model(x.to(dtype=self._model_dtype))
        if isinstance(stages, torch.Tensor):
            stages = [stages]
        fused_parts: list[torch.Tensor] = []
        for stage in stages:
            stage_f = stage.detach().float()
            if stage_f.shape[-2:] != (target_h, target_w):
                stage_f = F.interpolate(
                    stage_f,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            stage_f = F.normalize(stage_f, dim=1, eps=1e-12)
            fused_parts.append(stage_f.squeeze(0).to("cpu"))
        fused = torch.cat(fused_parts, dim=0)
        fused = apply_matryoshka_truncation(
            fused,
            self.cfg.matryoshka_dim,
            channel_dim=0,
            normalize=normalize,
        )
        if return_layer == "all":
            return fused.unsqueeze(0)
        return fused

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
        pil = self._to_pil(image, color_space=color_space)
        x = self._preprocess(pil)
        ret_mode = return_layer or self.cfg.return_layer

        if self.backend_kind in ("timm", "torchvision"):
            out = self._extract_cnn_pyramid(
                x, return_layer=ret_mode, normalize=normalize
            )
            return out.numpy() if return_type == "numpy" else out

        x_input = x.to(dtype=self._model_dtype)
        spatial_hw = (int(x_input.shape[-2]), int(x_input.shape[-1]))
        grid_h = max(1, spatial_hw[0] // self.patch_size)
        grid_w = max(1, spatial_hw[1] // self.patch_size)

        need_hidden_states = bool(self._accepts_output_hidden_states) and (
            ret_mode == "all"
            or self._pooling_kernel_size > 1
            or any(
                (idx if idx >= 0 else self.num_hidden_layers + idx)
                != (self.num_hidden_layers - 1)
                for idx in self._layers
            )
        )

        forward_kwargs: dict[str, Any] = {"pixel_values": x_input}
        if self._accepts_output_hidden_states:
            forward_kwargs["output_hidden_states"] = bool(need_hidden_states)
        if self._accepts_interpolate_pos_encoding:
            forward_kwargs["interpolate_pos_encoding"] = True
        if self._accepts_pixel_position_ids:
            forward_kwargs["pixel_position_ids"] = self._build_pixel_position_ids(
                grid_h, grid_w, x_input.device
            )

        # Never enable float16 autocast for EmbeddingGemma 2 (overflows to NaN).
        if self.backend_kind == "embeddinggemma":
            outputs = self.model(**forward_kwargs)
        else:
            use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
            autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = self.model(**forward_kwargs)

        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            last_hidden = outputs[0]

        selected_grids: list[torch.Tensor] = []
        if hidden_states is not None and len(hidden_states) >= 2:
            num_layers = len(hidden_states) - 1
            layer_indices = self._layers if ret_mode == "all" else (self._layers[-1],)
            for layer_idx in layer_indices:
                resolved_idx = (
                    layer_idx if layer_idx >= 0 else num_layers + int(layer_idx)
                )
                resolved_idx = max(0, min(num_layers - 1, int(resolved_idx)))
                layer_tokens = hidden_states[resolved_idx + 1]
                grid = self._tokens_or_map_to_grid(
                    layer_tokens,
                    spatial_hw=spatial_hw,
                    normalize=normalize,
                )
                # When EmbeddingGemma 2 pools last_hidden_state spatially (e.g. 3x3)
                # at the same channel width, blend the fine unpooled token grid with
                # the upsampled pooled semantic representation.
                if (
                    self.backend_kind == "embeddinggemma"
                    and self._pooling_kernel_size > 1
                    and last_hidden is not None
                    and resolved_idx == num_layers - 1
                    and last_hidden.shape[-1] == layer_tokens.shape[-1]
                ):
                    try:
                        pooled_grid = self._tokens_or_map_to_grid(
                            last_hidden,
                            spatial_hw=spatial_hw,
                            normalize=True,
                        )
                        if pooled_grid.shape == grid.shape:
                            blended = (
                                0.65 * F.normalize(grid, dim=0, eps=1e-12)
                                + 0.35 * pooled_grid
                            )
                            grid = (
                                F.normalize(blended, dim=0, eps=1e-12)
                                if normalize
                                else blended
                            )
                    except ValueError:
                        pass
                selected_grids.append(grid)
        elif last_hidden is not None:
            selected_grids.append(
                self._tokens_or_map_to_grid(
                    last_hidden,
                    spatial_hw=spatial_hw,
                    normalize=normalize,
                )
            )
        else:
            raise RuntimeError(
                f"Model '{self.model_id}' did not return last_hidden_state or hidden_states."
            )

        result = (
            selected_grids[-1]
            if ret_mode == "last"
            else torch.stack(selected_grids, dim=0)
        )
        if return_type == "numpy":
            return result.numpy()
        return result

    @torch.inference_mode()
    def embed_keypoint_regions(
        self,
        image: Union[Image.Image, np.ndarray],
        centers_xy: Sequence[Tuple[float, float]],
        *,
        crop_size_px: int = 64,
        gaussian_sigma_ratio: float = 0.32,
        color_space: Literal["RGB", "BGR"] = "RGB",
        matryoshka_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Embed Gaussian-vignetted local image regions centered on keypoints.

        Each crop is sampled continuously via ``F.grid_sample`` around ``(x, y)``
        and softly vignetted toward its Gaussian-weighted mean color so outer
        background clutter is suppressed without introducing artificial black
        aperture edges.

        Returns
        -------
        torch.Tensor
            ``[N, D]`` L2-normalized ``float32`` region embeddings on CPU.
        """
        effective_dim = (
            matryoshka_dim if matryoshka_dim is not None else self.cfg.matryoshka_dim
        )
        if not centers_xy:
            out_dim = int(effective_dim or 768)
            return torch.zeros((0, out_dim), dtype=torch.float32)

        pil = self._to_pil(image, color_space=color_space)
        arr = np.array(pil, dtype=np.uint8, copy=True)[..., :3]
        img_h, img_w = arr.shape[:2]
        img_tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        ).unsqueeze(0)

        multiple = max(self.patch_size, int(self._snap_multiple))
        target_crop = max(
            multiple, int(math.ceil(max(16, int(crop_size_px)) / multiple) * multiple)
        )
        raw_crop = max(8.0, float(crop_size_px))
        half_span = 0.5 * raw_crop

        axis = torch.linspace(
            -half_span, half_span, steps=target_crop, dtype=torch.float32
        )
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")

        num_pts = len(centers_xy)
        cx = torch.tensor(
            [float(pt[0]) for pt in centers_xy], dtype=torch.float32
        ).view(num_pts, 1, 1)
        cy = torch.tensor(
            [float(pt[1]) for pt in centers_xy], dtype=torch.float32
        ).view(num_pts, 1, 1)
        sample_x = cx + grid_x.unsqueeze(0)
        sample_y = cy + grid_y.unsqueeze(0)
        norm_x = (2.0 * (sample_x + 0.5) / max(1.0, float(img_w))) - 1.0
        norm_y = (2.0 * (sample_y + 0.5) / max(1.0, float(img_h))) - 1.0
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)

        expanded_img = img_tensor.expand(num_pts, -1, -1, -1)
        crops = F.grid_sample(
            expanded_img,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # Smooth 2D Gaussian window blending peripheral background toward local mean.
        sigma_norm = max(0.08, float(gaussian_sigma_ratio))
        coord_1d = torch.linspace(-0.5, 0.5, steps=target_crop, dtype=torch.float32)
        gy, gx = torch.meshgrid(coord_1d, coord_1d, indexing="ij")
        gauss_window = torch.exp(-0.5 * (gx * gx + gy * gy) / (sigma_norm * sigma_norm))
        gauss_window = gauss_window / gauss_window.max().clamp_min(1e-6)
        w4d = gauss_window.view(1, 1, target_crop, target_crop)
        w_sum = w4d.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        weighted_mean = (crops * w4d).sum(dim=(-2, -1), keepdim=True) / w_sum
        vignetted = w4d * crops + (1.0 - w4d) * weighted_mean

        mean = torch.tensor(self._mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._std, dtype=torch.float32).view(1, 3, 1, 1)
        batch = ((vignetted - mean) / std).to(
            device=self.device, dtype=self._model_dtype
        )

        # Prefer dedicated multimodal/vision pooled embedders when exposed.
        full_model = self._full_model
        get_image_features = (
            getattr(full_model, "get_image_features", None)
            if full_model is not None
            else None
        )
        if callable(get_image_features):
            feat_out = get_image_features(pixel_values=batch)
            if isinstance(feat_out, torch.Tensor):
                pooled = feat_out
            else:
                pooled = (
                    getattr(feat_out, "image_embeds", None)
                    or getattr(feat_out, "pooler_output", None)
                    or feat_out[0]
                )
            if pooled.ndim == 3:
                pooled = pooled.mean(dim=1)
            pooled_cpu = pooled.detach().float().to("cpu")
            return apply_matryoshka_truncation(
                pooled_cpu,
                effective_dim,
                channel_dim=1,
                normalize=True,
            )

        if self.backend_kind in ("timm", "torchvision"):
            stages = self.model(batch)
            if isinstance(stages, torch.Tensor):
                stages = [stages]
            stage_vecs: list[torch.Tensor] = []
            for stage in stages:
                sf = stage.detach().float()
                sh, sw = sf.shape[-2:]
                sy = torch.linspace(-0.5, 0.5, steps=sh, device=sf.device)
                sx = torch.linspace(-0.5, 0.5, steps=sw, device=sf.device)
                wy, wx = torch.meshgrid(sy, sx, indexing="ij")
                sw_mask = torch.exp(-0.5 * (wx * wx + wy * wy) / (sigma_norm**2))
                sw_mask = (sw_mask / sw_mask.sum().clamp_min(1e-6)).view(1, 1, sh, sw)
                vec = (sf * sw_mask).sum(dim=(-2, -1))
                stage_vecs.append(F.normalize(vec, dim=1, eps=1e-12).to("cpu"))
            combined = torch.cat(stage_vecs, dim=1)
            return apply_matryoshka_truncation(
                combined,
                effective_dim,
                channel_dim=1,
                normalize=True,
            )

        # Fallback: run vision tower per-crop or batched and Gaussian-pool spatial tokens.
        forward_kwargs: dict[str, Any] = {"pixel_values": batch}
        if self._accepts_interpolate_pos_encoding:
            forward_kwargs["interpolate_pos_encoding"] = True
        if self._accepts_pixel_position_ids:
            gh = max(1, target_crop // self.patch_size)
            gw = max(1, target_crop // self.patch_size)
            pos = self._build_pixel_position_ids(gh, gw, batch.device).expand(
                num_pts, -1, -1
            )
            forward_kwargs["pixel_position_ids"] = pos
        outputs = self.model(**forward_kwargs)
        pooler = getattr(outputs, "pooler_output", None)
        if isinstance(pooler, torch.Tensor) and pooler.ndim == 2:
            return apply_matryoshka_truncation(
                pooler.detach().float().to("cpu"),
                effective_dim,
                channel_dim=1,
                normalize=True,
            )
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None and isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        if tokens is None:
            raise RuntimeError(
                f"Model '{self.model_id}' did not produce embeddings for keypoint crops."
            )
        tokens_f = tokens.detach().float()
        if tokens_f.shape[1] > 1 and self._num_special_tokens > 0:
            tokens_f = tokens_f[:, self._num_special_tokens :, :]
        n_tok = int(tokens_f.shape[1])
        side = int(round(math.sqrt(n_tok)))
        if side * side == n_tok and side > 1:
            ty = torch.linspace(-0.5, 0.5, steps=side, device=tokens_f.device)
            tx = torch.linspace(-0.5, 0.5, steps=side, device=tokens_f.device)
            wy, wx = torch.meshgrid(ty, tx, indexing="ij")
            tw = torch.exp(-0.5 * (wx * wx + wy * wy) / (sigma_norm**2)).reshape(
                1, n_tok, 1
            )
            tw = tw / tw.sum(dim=1, keepdim=True).clamp_min(1e-6)
            vecs = (tokens_f * tw).sum(dim=1).to("cpu")
        else:
            vecs = tokens_f.mean(dim=1).to("cpu")
        return apply_matryoshka_truncation(
            vecs,
            effective_dim,
            channel_dim=1,
            normalize=True,
        )
