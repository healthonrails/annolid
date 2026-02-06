"""
Lightweight entrypoint that parses CLI args before importing heavy GUI deps.

This keeps `annolid --help` and `annolid --version` fast by deferring the
heavy Qt/torch/OpenCV imports until the GUI actually launches.
"""

from importlib import metadata
import os
import sys
from typing import Any, MutableMapping, Optional, Sequence

from annolid.gui.cli import parse_cli
from annolid.utils.logger import configure_logging


def _print_version() -> None:
    """Print the installed Annolid version without importing the full GUI."""
    try:
        version = metadata.version("annolid")
    except metadata.PackageNotFoundError:
        version = "unknown"
    print(version)


def _is_cv2_qt_plugin_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "cv2/qt/plugins" in normalized


def _sanitize_qt_plugin_env(
    env: MutableMapping[str, str],
    *,
    is_linux: Optional[bool] = None,
) -> None:
    """Avoid OpenCV's Qt plugin path overriding the GUI Qt runtime on Linux."""
    if is_linux is None:
        is_linux = sys.platform.startswith("linux")
    if not is_linux:
        return

    for var_name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = env.get(var_name)
        if value and _is_cv2_qt_plugin_path(value):
            env.pop(var_name, None)

    plugin_path = env.get("QT_PLUGIN_PATH")
    if not plugin_path:
        return

    path_entries = [entry for entry in plugin_path.split(os.pathsep) if entry]
    kept_entries = [
        entry for entry in path_entries if not _is_cv2_qt_plugin_path(entry)
    ]
    if len(kept_entries) == len(path_entries):
        return
    if kept_entries:
        env["QT_PLUGIN_PATH"] = os.pathsep.join(kept_entries)
    else:
        env.pop("QT_PLUGIN_PATH", None)


def main(argv: Optional[Sequence[str]] = None) -> Any:
    config, _, version_requested = parse_cli(argv)
    if version_requested:
        _print_version()
        return 0

    # Windows: avoid OpenMP runtime conflicts between PyTorch and ONNX Runtime
    # (labelme's AI helpers import onnxruntime; annolid imports torch).
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _sanitize_qt_plugin_env(os.environ)

    configure_logging()

    # Import heavy dependencies only when actually launching the GUI.
    from annolid.gui import app as gui_app

    return gui_app.main(argv=argv, config=config)


if __name__ == "__main__":
    raise SystemExit(main())
