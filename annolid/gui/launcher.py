"""
Lightweight entrypoint that parses CLI args before importing heavy GUI deps.

This keeps `annolid --help` and `annolid --version` fast by deferring the
heavy Qt/torch/OpenCV imports until the GUI actually launches.
"""

from importlib import metadata
from typing import Any, Optional, Sequence

from annolid.gui.cli import parse_cli


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

    # Import heavy dependencies only when actually launching the GUI.
    from annolid.gui import app as gui_app

    return gui_app.main(argv=argv, config=config)


if __name__ == "__main__":
    raise SystemExit(main())
