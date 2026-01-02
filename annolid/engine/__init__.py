"""Unified training/inference entrypoints for Annolid models.

This package provides a small plugin system so new models can be added with:
1) a lightweight wrapper implementing `ModelPluginBase`
2) registering via `@register_model`
3) (optionally) exposing CLI via `annolid-run`
"""

from annolid.engine.registry import ModelPluginBase, get_model, list_models, register_model

__all__ = [
    "ModelPluginBase",
    "get_model",
    "list_models",
    "register_model",
]

