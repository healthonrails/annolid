"""Runtime patching/environment adapters behind the infrastructure layer."""

from annolid.gui.application import create_qapp
from annolid.gui.qt_env import sanitize_qt_plugin_env
from annolid.utils.macos_fixes import apply_macos_webengine_sandbox_patch

__all__ = [
    "apply_macos_webengine_sandbox_patch",
    "create_qapp",
    "sanitize_qt_plugin_env",
]
