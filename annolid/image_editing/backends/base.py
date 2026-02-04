from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict

from annolid.image_editing.types import ImageEditRequest, ImageEditResult


class ImageEditingBackend(ABC):
    """Backends implement a single entrypoint to generate/edit images."""

    name: str

    @abstractmethod
    def run(self, request: ImageEditRequest) -> ImageEditResult:
        raise NotImplementedError


def filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only kwargs accepted by callable_obj (best-effort)."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    accepted = set(sig.parameters.keys())
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}
