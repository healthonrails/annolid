"""CLI interface entry points."""

from annolid.engine.cli import main
from annolid.engine.run_config import (
    build_cli_args_from_config,
    expand_argv_with_run_config,
)

__all__ = [
    "build_cli_args_from_config",
    "expand_argv_with_run_config",
    "main",
]
