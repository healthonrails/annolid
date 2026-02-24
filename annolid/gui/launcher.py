"""
Lightweight entrypoint that parses CLI args before importing heavy GUI deps.

This keeps `annolid --help` and `annolid --version` fast by deferring the
heavy Qt/torch/OpenCV imports until the GUI actually launches.
"""

import importlib
from importlib import metadata
import os
import sys
from typing import Any, Optional, Sequence

from annolid.gui.cli import parse_cli
from annolid.gui.qt_env import sanitize_qt_plugin_env
from annolid.utils.logger import configure_logging


def _print_version() -> None:
    """Print the installed Annolid version without importing the full GUI."""
    try:
        version = metadata.version("annolid")
    except metadata.PackageNotFoundError:
        version = "unknown"
    print(version)


def main(argv: Optional[Sequence[str]] = None) -> Any:
    config, _, version_requested = parse_cli(argv)
    if version_requested:
        _print_version()
        return 0

    # Windows: avoid OpenMP runtime conflicts between PyTorch and ONNX Runtime
    # (labelme's AI helpers import onnxruntime; annolid imports torch).
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sanitize_qt_plugin_env(os.environ)

    configure_logging()

    # Import heavy dependencies only when actually launching the GUI.
    try:
        gui_app = importlib.import_module("annolid.gui.app")
    except Exception as exc:
        if exc.__class__.__name__ == "QtBindingsNotFoundError":
            print(
                (
                    "No Qt binding detected. Install GUI dependencies with:\n"
                    '  pip install -e ".[gui]"\n'
                    "or install a Qt binding directly (for example `PyQt5` or `PySide6`)."
                ),
                file=sys.stderr,
            )
            return 1
        raise

    return gui_app.main(argv=argv, config=config)


if __name__ == "__main__":
    raise SystemExit(main())
