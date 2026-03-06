"""Background-service interface adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FlexibleWorker": "annolid.interfaces.background.services",
    "TrackAllWorker": "annolid.interfaces.background.services",
    "TrackingWorker": "annolid.interfaces.background.services",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
