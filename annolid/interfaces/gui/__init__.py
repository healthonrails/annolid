"""GUI interface adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AnnolidWindow": "annolid.interfaces.gui.app",
    "build_parser": "annolid.interfaces.gui.app",
    "create_qapp": "annolid.interfaces.gui.app",
    "main": "annolid.interfaces.gui.app",
    "parse_cli": "annolid.interfaces.gui.app",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
