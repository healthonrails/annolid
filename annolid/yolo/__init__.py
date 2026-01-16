"""Shared YOLO utilities for Annolid.

This package centralizes Ultralytics-specific configuration (cache/weights),
weight resolution, and model loading so GUI, realtime, and batch pipelines
behave consistently.
"""

from .runtime import (  # noqa: F401
    YOLOModelSpec,
    configure_ultralytics_cache,
    ensure_ultralytics_asset_cached,
    get_cache_root,
    get_ultralytics_weights_cache_dir,
    load_yolo_model,
    resolve_weight_path,
    select_backend,
)

__all__ = [
    "YOLOModelSpec",
    "configure_ultralytics_cache",
    "ensure_ultralytics_asset_cached",
    "get_cache_root",
    "get_ultralytics_weights_cache_dir",
    "load_yolo_model",
    "resolve_weight_path",
    "select_backend",
]
