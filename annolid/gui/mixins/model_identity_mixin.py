from __future__ import annotations

from pathlib import Path
from typing import Optional

from annolid.utils.runs import find_latest_checkpoint


class ModelIdentityMixin:
    """Helpers for resolving current model identity and type checks."""

    def _get_current_model_config(self):
        """Return the ModelConfig for the currently selected model, if any."""
        return self.ai_model_manager.get_current_model()

    def get_current_model_weight_file(self) -> str:
        """
        Returns the weight file associated with the currently selected model.
        If no matching model is found, returns a default fallback weight file.
        """
        return self.ai_model_manager.get_current_weight()

    def _resolve_model_identity(self):
        model_config = self._get_current_model_config()
        identifier = model_config.identifier if model_config else None
        weight = model_config.weight_file if model_config else None
        if identifier is None and weight is None:
            fallback = self.get_current_model_weight_file()
            identifier = fallback
            weight = fallback
        return model_config, identifier or "", weight or ""

    def _resolve_dino_kpseg_weight(self, model_weight: str) -> Optional[str]:
        raw = str(model_weight or "").strip()
        if raw:
            try:
                p = Path(raw).expanduser()
                if p.is_absolute() and p.exists():
                    return str(p.resolve())
                if not p.is_absolute():
                    resolved = p.resolve()
                    if resolved.exists():
                        return str(resolved)
            except Exception:
                pass

        try:
            saved = self.settings.value("ai/dino_kpseg_last_best", "", type=str)
        except Exception:
            saved = ""
        if saved:
            try:
                p = Path(saved).expanduser().resolve()
                if p.exists():
                    return str(p)
            except Exception:
                pass

        try:
            latest = find_latest_checkpoint(task="dino_kpseg", model="train")
            if latest is not None:
                return str(latest)
        except Exception:
            pass

        return None

    @staticmethod
    def _is_cotracker_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return identifier == "cotracker" or weight == "cotracker.pt"

    @staticmethod
    def _is_cowtracker_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return (
            identifier == "cowtracker"
            or weight == "cowtracker.pt"
            or "cowtracker" in weight
        )

    @staticmethod
    def _is_dino_keypoint_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        return (
            identifier == "dinov3_keypoint_tracker" or weight == "dino_keypoint_tracker"
        )

    @staticmethod
    def _is_dino_kpseg_tracker_model(identifier: str, weight: str) -> bool:
        key = f"{identifier or ''} {weight or ''}".lower()
        return (
            identifier.lower() == "dino_kpseg_tracker"
            or "dino_kpseg_tracker" in key
            or "dinokpseg_tracker" in key
        )

    @staticmethod
    def _is_dino_kpseg_model(identifier: str, weight: str) -> bool:
        identifier = (identifier or "").lower()
        weight = (weight or "").lower()
        key = f"{identifier} {weight}"
        if "dino_kpseg_tracker" in key or "dinokpseg_tracker" in key:
            return False
        return (
            identifier == "dino_kpseg"
            or "dino_kpseg" in key
            or "dinokpseg" in key
            or "kpseg" in key
        )

    @staticmethod
    def _is_yolo_model(identifier: str, weight: str) -> bool:
        identifier = identifier.lower()
        weight = weight.lower()
        if "yolo" in identifier or "yolo" in weight:
            return True

        non_yolo_keywords = (
            "sam",
            "dinov",
            "dino",
            "cotracker",
            "cutie",
            "efficientvit",
            "mediapipe",
            "maskrcnn",
            "videomt",
        )
        if any(keyword in identifier for keyword in non_yolo_keywords):
            return False
        if any(keyword in weight for keyword in non_yolo_keywords):
            return False

        yolo_extensions = (".pt", ".pth", ".onnx", ".engine", ".mlpackage")
        if weight.endswith(yolo_extensions):
            return True

        return False

    @staticmethod
    def _is_efficienttam_model(identifier: str, weight: str) -> bool:
        """Detect EfficientTAM models based on identifier/weight strings."""
        key = f"{identifier or ''} {weight or ''}".lower()
        return "efficienttam" in key

    @staticmethod
    def _is_mediapipe_model(identifier: str, weight: str) -> bool:
        """Detect MediaPipe models based on identifier/weight strings."""
        key = f"{identifier or ''} {weight or ''}".lower()
        return "mediapipe" in key
