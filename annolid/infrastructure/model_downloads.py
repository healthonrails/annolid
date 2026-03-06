"""Model download/cache adapters behind the infrastructure layer."""

from annolid.yolo.runtime import (
    configure_ultralytics_cache,
    ensure_ultralytics_asset_cached,
    get_cache_root,
    get_ultralytics_weights_cache_dir,
    load_yolo_model,
    resolve_weight_path,
    select_backend,
)

__all__ = [
    "configure_ultralytics_cache",
    "ensure_ultralytics_asset_cached",
    "get_cache_root",
    "get_ultralytics_weights_cache_dir",
    "load_yolo_model",
    "resolve_weight_path",
    "select_backend",
]
