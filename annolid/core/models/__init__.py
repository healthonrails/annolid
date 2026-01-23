"""Runtime model interfaces and adapters (GUI-free)."""

from .base import ModelCapabilities, ModelRequest, ModelResponse, RuntimeModel
from .pipeline import run_caption, run_model

__all__ = [
    "ModelCapabilities",
    "ModelRequest",
    "ModelResponse",
    "RuntimeModel",
    "run_model",
    "run_caption",
]
