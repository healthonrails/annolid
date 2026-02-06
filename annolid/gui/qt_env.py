import os
import sys
from typing import MutableMapping, Optional


def _is_cv2_qt_plugin_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "cv2/qt/plugins" in normalized


def sanitize_qt_plugin_env(
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
