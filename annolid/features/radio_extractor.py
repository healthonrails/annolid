from __future__ import annotations
from typing import Optional, Union, Literal
import logging
import torch
import numpy as np
from PIL import Image
import inspect

from .dinov3_extractor import Dinov3FeatureExtractor

logger = logging.getLogger(__name__)


class RadioFeatureExtractor(Dinov3FeatureExtractor):
    """
    Feature extractor specialized for NVIDIA RADIO models (e.g. nvidia/C-RADIOv4-SO400M).
    Inherits from Dinov3FeatureExtractor for shared preprocessing and grid logic.
    """

    def _load_model(self):
        # Override to ensure trust_remote_code=True for RADIO
        from transformers import AutoImageProcessor, AutoModel

        try:
            logger.info(
                "Loading RADIO model '%s' via Hugging Face Transformers", self.model_id
            )
            processor = AutoImageProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                self.model_id, cache_dir=self.cache_dir, trust_remote_code=True
            )
            return processor, model
        except ImportError as exc:
            if "open_clip" in str(exc):
                raise RuntimeError(
                    "RADIO load error: missing optional dependency 'open-clip-torch' "
                    "(import name: open_clip). Install it with: pip install open-clip-torch"
                ) from exc
            hint = (
                "Failed to load RADIO model via transformers. Ensure optional dependencies "
                "required by the checkpoint are installed."
            )
            raise RuntimeError(f"RADIO load error: {exc}. {hint}") from exc
        except Exception as exc:
            hint = (
                "Failed to load RADIO model via transformers. Ensure 'transformers' is installed "
                "and the requested checkpoint is available."
            )
            if "open_clip" in str(exc):
                hint = "Install RADIO optional dependency with: pip install open-clip-torch"
            raise RuntimeError(f"RADIO load error: {exc}. {hint}") from exc

    def _resolve_model_properties(self) -> None:
        """Resolve model properties specific to RADIO models."""
        cfg = self.model.config

        # 1. Patch Size
        attr_patch = getattr(cfg, "patch_size", None)
        if attr_patch is None:
            attr_patch = getattr(self.model, "patch_size", None)

        if attr_patch is not None:
            self.patch_size = int(attr_patch)
        else:
            self.patch_size = self.cfg.patch_size

        if self.patch_size != self.cfg.patch_size:
            logger.info("Using patch_size=%s from model config", self.patch_size)

        # 2. Number of Hidden Layers
        # Check for RADIO/timm style structures
        layers = None
        backbone = getattr(self.model, "radio_model", None)
        if backbone is None:
            backbone = getattr(self.model, "model", None)

        if backbone:
            if hasattr(backbone, "blocks"):
                layers = len(backbone.blocks)
            elif hasattr(backbone, "layers"):
                layers = len(backbone.layers)

        # Fallback to direct blocks/encoder check
        if layers is None:
            if hasattr(self.model, "blocks"):
                layers = len(self.model.blocks)
            elif hasattr(self.model, "encoder") and hasattr(
                self.model.encoder, "layers"
            ):
                layers = len(self.model.encoder.layers)

        if layers is None or layers <= 0:
            # Try config as last resort
            layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "depth", None))

        if layers is None or layers <= 0:
            raise RuntimeError(
                f"Could not determine num_hidden_layers for RADIO model {self.model_id}"
            )

        self.num_hidden_layers = int(layers)
        self._layers = (
            tuple(self.cfg.layers)
            if self.cfg.layers is not None
            else tuple(range(self.num_hidden_layers))
        )

        # 3. Special Tokens
        # RADIO typically tracks tokens via num_cls_tokens + num_summary_tokens
        cls = 1 if getattr(cfg, "use_cls_token", True) else 0
        regs = int(getattr(cfg, "num_register_tokens", 0))
        self._num_special_tokens = cls + regs

        if backbone is not None:
            n_cls = getattr(backbone, "num_cls_tokens", 0)
            n_sum = getattr(backbone, "num_summary_tokens", 0)
            if hasattr(backbone, "num_cls_tokens"):
                self._num_special_tokens = n_cls + n_sum
                logger.info(
                    "Detected RADIO model. Special tokens: CLS=%d, SUM=%d (Total=%d)",
                    n_cls,
                    n_sum,
                    self._num_special_tokens,
                )

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
        """Extract features specialized for RADIO models."""
        pil = self._to_pil(image, color_space=color_space)
        x = self._preprocess(pil)

        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

        ret_mode = return_layer or self.cfg.return_layer

        need_hidden_states = True
        if ret_mode == "last":
            resolved = []
            for layer_idx in self._layers:
                resolved_idx = layer_idx
                if layer_idx < 0:
                    resolved_idx = self.num_hidden_layers + layer_idx
                resolved.append(int(resolved_idx))
            if resolved and set(resolved) == {int(self.num_hidden_layers - 1)}:
                need_hidden_states = False

        # Prepare arguments based on model signature (RADIO often takes 'x')
        sig = inspect.signature(self.model.forward)
        call_kwargs = {}
        if "pixel_values" in sig.parameters:
            call_kwargs["pixel_values"] = x
        elif "x" in sig.parameters:
            call_kwargs["x"] = x
        else:
            call_kwargs["pixel_values"] = x

        # Only request hidden states if supported
        supports_hidden = "output_hidden_states" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if supports_hidden and need_hidden_states:
            call_kwargs["output_hidden_states"] = True

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = self.model(**call_kwargs)

        selected_grids = []

        # Helper to extract last_hidden_state from various output types
        last_hidden_state = None
        all_hidden_states = None

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = getattr(outputs, "hidden_states", None)
        elif isinstance(outputs, (tuple, list)):
            # Heuristic for RADIO: (summary, features)
            if len(outputs) >= 2:
                last_hidden_state = outputs[1]
            else:
                last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs

        # Fallback logic identical to refactored Dinov3FeatureExtractor
        if not need_hidden_states or all_hidden_states is None:
            if last_hidden_state is None:
                raise RuntimeError(
                    f"Could not parse output from model type {type(outputs)}"
                )

            tokens = last_hidden_state
            grid = self._tokens_to_grid(
                tokens,
                spatial_hw=(x.shape[-2], x.shape[-1]),
                detach=True,
                normalize=normalize,
            )
            selected_grids.append(grid)
        else:
            hidden_states = all_hidden_states
            selected_grids = []
            num_layers = len(hidden_states) - 1
            for layer_idx in self._layers:
                resolved_idx = layer_idx
                if layer_idx < 0:
                    resolved_idx = num_layers + layer_idx
                if resolved_idx < 0 or resolved_idx >= num_layers:
                    raise IndexError(
                        f"Requested layer {layer_idx} outside available range -{num_layers}..{num_layers - 1}"
                    )
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
