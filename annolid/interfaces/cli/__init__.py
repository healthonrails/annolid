"""CLI interface adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_cli_args_from_config": "annolid.interfaces.cli.engine",
    "build_parser": "annolid.interfaces.cli.gui",
    "expand_argv_with_run_config": "annolid.interfaces.cli.engine",
    "main": "annolid.interfaces.cli.engine",
    "parse_cli": "annolid.interfaces.cli.gui",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
