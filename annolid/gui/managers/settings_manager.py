"""
Settings Manager for Annolid GUI Application.

Handles configuration persistence, validation, and access to application settings.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from qtpy import QtCore

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Centralized settings management for the Annolid application.

    Provides a clean interface for accessing and persisting application settings,
    with validation and type safety.
    """

    def __init__(self, organization: str = "Annolid", application: str = "Annolid"):
        """
        Initialize the settings manager.

        Args:
            organization: Organization name for QSettings
            application: Application name for QSettings
        """
        self._qt_settings = QtCore.QSettings(organization, application)
        self._defaults: Dict[str, Any] = {
            "recentFiles": [],
            "window/position": QtCore.QPoint(0, 0),
            "window/size": QtCore.QSize(1600, 900),
            "ui/agent_mode": True,
            "ui/show_embedding_search": False,
            "pose/show_edges": True,
            "pose/show_bbox": True,
            "pose/save_bbox": True,
            "timeline/show_dock": False,
        }

    @property
    def qt_settings(self) -> QtCore.QSettings:
        """Expose underlying QSettings for legacy integrations."""
        return self._qt_settings

    def get_setting(
        self, key: str, default: Any = None, value_type: type = None
    ) -> Any:
        """
        Get a setting value with optional type conversion.

        Args:
            key: Setting key
            default: Default value if setting doesn't exist
            value_type: Expected type for the value

        Returns:
            The setting value, or default if not found
        """
        try:
            if default is None and key in self._defaults:
                default = self._defaults[key]

            if value_type is not None:
                return self._qt_settings.value(key, default, type=value_type)
            else:
                return self._qt_settings.value(key, default)
        except Exception as e:
            logger.warning(f"Failed to get setting '{key}': {e}")
            return default

    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a setting value.

        Args:
            key: Setting key
            value: Value to set

        Returns:
            True if successful, False otherwise
        """
        try:
            self._qt_settings.setValue(key, value)
            self._qt_settings.sync()
            return True
        except Exception as e:
            logger.error(f"Failed to set setting '{key}': {e}")
            return False

    def has_setting(self, key: str) -> bool:
        """
        Check if a setting exists.

        Args:
            key: Setting key

        Returns:
            True if the setting exists
        """
        return self._qt_settings.contains(key)

    def remove_setting(self, key: str) -> bool:
        """
        Remove a setting.

        Args:
            key: Setting key

        Returns:
            True if successful, False otherwise
        """
        try:
            self._qt_settings.remove(key)
            self._qt_settings.sync()
            return True
        except Exception as e:
            logger.error(f"Failed to remove setting '{key}': {e}")
            return False

    def get_recent_files(self) -> list[str]:
        """Get the list of recent files."""
        files = self.get_setting("recentFiles", [], list)
        return [f for f in files if f] if files else []

    def add_recent_file(self, file_path: str) -> None:
        """Add a file to the recent files list."""
        if not file_path:
            return

        recent_files = self.get_recent_files()
        if file_path in recent_files:
            recent_files.remove(file_path)

        recent_files.insert(0, file_path)
        # Keep only the last 10 files
        recent_files = recent_files[:10]

        self.set_setting("recentFiles", recent_files)

    def get_window_geometry(self) -> Dict[str, Any]:
        """Get window position and size."""
        return {
            "position": self.get_setting("window/position", QtCore.QPoint(0, 0)),
            "size": self.get_setting("window/size", QtCore.QSize(1600, 900)),
        }

    def set_window_geometry(self, position: QtCore.QPoint, size: QtCore.QSize) -> None:
        """Set window position and size."""
        self.set_setting("window/position", position)
        self.set_setting("window/size", size)

    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-related settings."""
        return {
            "agent_mode": self.get_setting("ui/agent_mode", True, bool),
            "show_embedding_search": self.get_setting(
                "ui/show_embedding_search", False, bool
            ),
        }

    def get_pose_settings(self) -> Dict[str, Any]:
        """Get pose-related settings."""
        return {
            "show_edges": self.get_setting("pose/show_edges", True, bool),
            "show_bbox": self.get_setting("pose/show_bbox", True, bool),
            "save_bbox": self.get_setting("pose/save_bbox", True, bool),
        }

    def get_timeline_settings(self) -> Dict[str, Any]:
        """Get timeline-related settings."""
        return {
            "show_dock": self.get_setting("timeline/show_dock", False, bool),
        }

    def export_settings(self, file_path: str) -> bool:
        """
        Export settings to a JSON file.

        Args:
            file_path: Path to export settings to

        Returns:
            True if successful, False otherwise
        """
        try:
            settings_dict = {}
            for key in self._qt_settings.allKeys():
                settings_dict[key] = self._qt_settings.value(key)

            with open(file_path, "w") as f:
                json.dump(settings_dict, f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"Failed to export settings to '{file_path}': {e}")
            return False

    def import_settings(self, file_path: str) -> bool:
        """
        Import settings from a JSON file.

        Args:
            file_path: Path to import settings from

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "r") as f:
                settings_dict = json.load(f)

            for key, value in settings_dict.items():
                self.set_setting(key, value)

            return True
        except Exception as e:
            logger.error(f"Failed to import settings from '{file_path}': {e}")
            return False

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        try:
            # Clear all settings
            self._qt_settings.clear()

            # Set defaults
            for key, value in self._defaults.items():
                self.set_setting(key, value)

            self._qt_settings.sync()
        except Exception as e:
            logger.error(f"Failed to reset settings to defaults: {e}")
