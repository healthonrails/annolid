from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..base import ModelCapabilities, ModelRequest, ModelResponse, RuntimeModel


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    label_id: int
    score: float

    def to_dict(self) -> Dict[str, object]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return {
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "label_id": int(self.label_id),
            "score": float(self.score),
        }


class TorchvisionMaskRCNNAdapter(RuntimeModel):
    """Local CV adapter wrapping Annolid's torchvision Mask R-CNN utilities."""

    def __init__(
        self,
        *,
        pretrained: bool = False,
        score_threshold: float = 0.5,
        device: Optional[str] = None,
        model_factory: Optional[Callable[[], Any]] = None,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._pretrained = bool(pretrained)
        self._score_threshold = float(score_threshold)
        self._device_override = device
        self._model_factory = model_factory
        self._label_names = list(label_names) if label_names else None

        self._torch: Any = None
        self._device: Any = None
        self._model: Any = None

    @property
    def model_id(self) -> str:
        base = "maskrcnn_torchvision"
        suffix = "pretrained" if self._pretrained else "random"
        return f"{base}:{suffix}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("detect", "caption"),
            input_modalities=("image",),
            output_modalities=("detections", "text"),
            streaming=False,
        )

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TorchvisionMaskRCNNAdapter requires torch. Install torch/torchvision to use this adapter."
            ) from exc

        self._torch = torch
        if self._device_override:
            self._device = torch.device(self._device_override)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._model_factory is not None:
            model = self._model_factory()
        else:
            from annolid.segmentation.maskrcnn.model import get_maskrcnn_model

            model = get_maskrcnn_model(finetuning=False, pretrained=self._pretrained)

        try:
            model.to(self._device)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass

        self._model = model

    def predict(self, request: ModelRequest) -> ModelResponse:
        self.load()
        task = str(request.task).strip().lower()
        if task not in {"detect", "caption"}:
            raise ValueError(
                f"Unsupported task for TorchvisionMaskRCNNAdapter: {request.task!r}"
            )

        image_rgb = self._load_image_rgb(request)
        tensor = self._to_tensor(image_rgb).to(self._device)

        with self._torch.no_grad():
            outputs = self._model([tensor])

        detections = self._parse_detections(outputs)
        if task == "detect":
            payload = [det.to_dict() for det in detections]
            return ModelResponse(
                task=request.task,
                output={"detections": payload},
                meta={"num_detections": len(payload)},
                raw=outputs,
            )

        caption = self._caption_from_detections(detections)
        return ModelResponse(
            task=request.task,
            output={
                "detections": [det.to_dict() for det in detections],
                "text": caption,
            },
            text=caption,
            meta={"num_detections": len(detections)},
            raw=outputs,
        )

    def close(self) -> None:
        torch_mod = self._torch
        device = self._device
        self._model = None
        self._device = None
        self._torch = None
        gc.collect()
        if torch_mod is None:
            return

        device_type = str(getattr(device, "type", device or "")).lower()
        if device_type.startswith("cuda"):
            try:
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass
        elif device_type.startswith("mps"):
            try:
                mps = getattr(torch_mod, "mps", None)
                mps_available = bool(
                    getattr(getattr(torch_mod, "backends", None), "mps", None)
                    and torch_mod.backends.mps.is_available()
                )
                if mps_available and mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
            except Exception:
                pass

    def _load_image_rgb(self, request: ModelRequest):
        if request.image is not None:
            return request.image
        if request.image_path:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OpenCV (cv2) is required to load image_path for TorchvisionMaskRCNNAdapter."
                ) from exc
            path = Path(request.image_path)
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(
            "TorchvisionMaskRCNNAdapter requires request.image or request.image_path."
        )

    def _to_tensor(self, image_rgb):
        arr = image_rgb
        if hasattr(arr, "dtype") and getattr(arr, "ndim", None) == 3:
            return self._torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        raise ValueError(
            "Unsupported image type for TorchvisionMaskRCNNAdapter; expected HxWxC numpy array."
        )

    def _parse_detections(self, outputs) -> List[Detection]:
        if not outputs:
            return []
        out0 = outputs[0]
        boxes = out0.get("boxes")
        labels = out0.get("labels")
        scores = out0.get("scores")
        if boxes is None or labels is None or scores is None:
            return []

        dets: List[Detection] = []
        for box, label, score in zip(boxes, labels, scores):
            s = float(score.detach().cpu().item())
            if s < self._score_threshold:
                continue
            xyxy = tuple(float(v) for v in box.detach().cpu().tolist())
            if len(xyxy) != 4:
                continue
            dets.append(
                Detection(
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    label_id=int(label.detach().cpu().item()),
                    score=s,
                )
            )
        return dets

    def _caption_from_detections(self, detections: Sequence[Detection]) -> str:
        if not detections:
            return "No objects detected."
        lines = []
        for det in detections[:10]:
            name = self._label_name(det.label_id)
            lines.append(f"{name} ({det.score:.2f})")
        suffix = "" if len(detections) <= 10 else f" (+{len(detections) - 10} more)"
        return "Detected: " + ", ".join(lines) + suffix

    def _label_name(self, label_id: int) -> str:
        if self._label_names and 0 <= label_id < len(self._label_names):
            name = str(self._label_names[label_id]).strip()
            if name:
                return name
        return f"label_{label_id}"
