"""Lazy feature exports.

Importing model metadata such as ``annolid.features.dino_models`` must not pull
optional ML runtimes into GUI startup or frozen desktop bundles.
"""

from __future__ import annotations

from typing import Any


_EXPORTS = {
    "Dinov3FeatureExtractor": (
        "annolid.features.dinov3_extractor",
        "Dinov3FeatureExtractor",
    ),
    "Dinov3Config": ("annolid.features.dinov3_extractor", "Dinov3Config"),
    "Dinov3PCAMapper": ("annolid.features.dinov3_pca", "Dinov3PCAMapper"),
    "PCAMapResult": ("annolid.features.dinov3_pca", "PCAMapResult"),
    "features_to_pca_rgb": ("annolid.features.dinov3_pca", "features_to_pca_rgb"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'annolid.features' has no attribute {name!r}"
        ) from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
