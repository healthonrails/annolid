from __future__ import annotations

from .base import LargeImageBackend, LargeImageLoadResult, LargeImageMetadata
from .cache import (
    DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES,
    DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES,
    LargeImageCacheEntry,
    clear_all_large_image_caches,
    format_large_image_cache_size,
    large_image_cache_size_bytes,
    large_image_cache_root,
    list_large_image_cache_entries,
    optimize_large_tiff_for_viewing,
    optimized_large_image_cache_path,
    prune_large_image_caches,
    pyvips_optimization_available,
    remove_large_image_cache_file,
    resolve_fresh_optimized_large_image_path,
)
from .openslide_backend import OpenSlideBackend
from .registry import (
    TIFF_SUFFIXES,
    available_large_image_backends,
    is_large_tiff_path,
    load_image_with_backends,
    open_large_image,
    probe_large_image,
    sniff_large_image,
)
from .tifffile_backend import (
    TiffFileBackend,
    load_tiff_with_tifffile,
    probe_tiff_metadata,
)
from .vips_backend import VipsBackend

__all__ = [
    "DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES",
    "DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES",
    "LargeImageBackend",
    "LargeImageCacheEntry",
    "LargeImageLoadResult",
    "LargeImageMetadata",
    "OpenSlideBackend",
    "TIFF_SUFFIXES",
    "TiffFileBackend",
    "VipsBackend",
    "available_large_image_backends",
    "is_large_tiff_path",
    "clear_all_large_image_caches",
    "format_large_image_cache_size",
    "large_image_cache_size_bytes",
    "large_image_cache_root",
    "list_large_image_cache_entries",
    "load_image_with_backends",
    "load_tiff_with_tifffile",
    "optimize_large_tiff_for_viewing",
    "optimized_large_image_cache_path",
    "open_large_image",
    "prune_large_image_caches",
    "pyvips_optimization_available",
    "remove_large_image_cache_file",
    "probe_large_image",
    "probe_tiff_metadata",
    "resolve_fresh_optimized_large_image_path",
    "sniff_large_image",
]
