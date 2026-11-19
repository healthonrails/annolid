"""Legacy CLIP and ResNet-18 feature extractors exposed lazily via ``annolid.features``."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class CLIPModel:
    """Wrapper class for OpenAI CLIP image and text feature extraction."""

    def __init__(self, clip_version: str = "ViT-B/32") -> None:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required for CLIPModel. "
                "Install it with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self._clip = clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(
            clip_version, device=self.device
        )

    def encode_image(self, image) -> np.ndarray:
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image_tensor)
        return image_features.cpu().detach().numpy()

    def encode_text(self, text) -> np.ndarray:
        tokens = self._clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        return text_features.cpu().detach().numpy()


class Embedding:
    """Extract 512-D ResNet-18 avgpool embeddings from RGB image crops."""

    def __init__(
        self, layer_output_size: int = 512, *, pretrained: bool = True
    ) -> None:
        if hasattr(models, "ResNet18_Weights"):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.net = models.resnet18(weights=weights)
        else:  # pragma: no cover - legacy torchvision fallback
            self.net = models.resnet18(pretrained=pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer = self.net._modules.get("avgpool")
        self.net.eval()
        self.net.to(self.device)
        self.size = (224, 224)
        self.layer_output_size = layer_output_size

        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img.astype(np.float32) / 255.0, self.size)

    def __call__(self, imgs) -> np.ndarray:
        if not imgs:
            return np.zeros((0, self.layer_output_size), dtype=np.float32)
        img_batch = torch.cat(
            [self.norm(self.resize(im)).unsqueeze(0) for im in imgs], dim=0
        ).float()
        img_batch = img_batch.to(self.device)
        embedding = torch.zeros(
            len(imgs), self.layer_output_size, 1, 1, device=self.device
        )

        def copy_layer(_m, _i, output) -> None:
            embedding.copy_(output.detach())

        hook = self.layer.register_forward_hook(copy_layer)
        try:
            with torch.no_grad():
                self.net(img_batch)
        finally:
            hook.remove()
        return embedding.cpu().numpy()[:, :, 0, 0]
