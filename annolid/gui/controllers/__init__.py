"""Lazy controller exports for Annolid GUI."""

from __future__ import annotations

from typing import Any


_EXPORTS = {
    "AnnotationController": (
        "annolid.gui.controllers.annotation_controller",
        "AnnotationController",
    ),
    "InferenceController": (
        "annolid.gui.controllers.inference_controller",
        "InferenceController",
    ),
    "ProjectController": (
        "annolid.gui.controllers.project_controller",
        "ProjectController",
    ),
    "VideoController": ("annolid.gui.controllers.video_controller", "VideoController"),
    "TrackingController": ("annolid.gui.controllers.tracking", "TrackingController"),
    "MenuController": ("annolid.gui.controllers.menu", "MenuController"),
    "FlagsController": ("annolid.gui.controllers.flags", "FlagsController"),
    "DinoController": ("annolid.gui.controllers.dino", "DinoController"),
    "TrackingDataController": (
        "annolid.gui.controllers.tracking_data",
        "TrackingDataController",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
