from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .base import ModelRequest, ModelResponse, RuntimeModel


def run_model(adapter: RuntimeModel, request: ModelRequest) -> ModelResponse:
    """Small demo pipeline that can swap adapters without changing calling code."""

    with adapter:
        return adapter.predict(request)


def run_caption(
    adapter: RuntimeModel,
    image_path: Union[str, Path],
    *,
    prompt: Optional[str] = None,
) -> ModelResponse:
    request = ModelRequest(
        task="caption",
        text=prompt,
        image_path=str(image_path),
    )
    return run_model(adapter, request)
