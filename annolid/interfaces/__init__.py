"""Interface-layer package namespace."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "background",
    "bot",
    "cli",
    "gui",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module(f"annolid.interfaces.{name}")
