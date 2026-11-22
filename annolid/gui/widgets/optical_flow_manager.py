from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy import QtCore

from annolid.gui.widgets.optical_flow_tool import OpticalFlowTool
from annolid.motion.optical_flow import (
    OpticalFlowPreferences,
    normalize_optical_flow_backend,
)
from annolid.utils.logger import logger

# Backward-compatible alias for internal references.
_FlowDefaults = OpticalFlowPreferences


class OpticalFlowManager(QtCore.QObject):
    """Encapsulates optical flow settings, tool, and overlays."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.compute_optical_flow: bool = True
        self._preferences: OpticalFlowPreferences = OpticalFlowPreferences()
        self._preferences.apply_to_target(self)

        self.optical_flow_tool = OpticalFlowTool(window, manager=self)
        self._load_settings()

    # ------------------------------------------------------------------ public API
    def run_tool(self) -> None:
        self.optical_flow_tool.run()

    def configure_tool(self) -> None:
        self.optical_flow_tool.configure()

    def stop_tool(self) -> None:
        self.optical_flow_tool.stop()

    def clear(self) -> None:
        self.optical_flow_tool.clear()

    def load_records(self, video_file: Optional[str] = None) -> None:
        self.optical_flow_tool.load_records(video_file)

    def update_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        self.optical_flow_tool.update_overlay_for_frame(frame_number, frame_rgb)

    def has_overlay_for_frame(self, frame_number: int) -> bool:
        return self.optical_flow_tool.has_overlay_for_frame(frame_number)

    def is_live_running(self) -> bool:
        return self.optical_flow_tool.is_live_running()

    # ------------------------------------------------------------------ settings
    def preferences(self) -> OpticalFlowPreferences:
        """Return a snapshot of the current optical-flow preferences."""
        return OpticalFlowPreferences.from_source(self)

    def apply_preferences(
        self,
        prefs: OpticalFlowPreferences,
        *,
        persist: bool = True,
        refresh_overlay: bool = False,
    ) -> None:
        """Update manager, window mirror, and optional QSettings atomically."""
        normalized = OpticalFlowPreferences.from_source(prefs)
        self._preferences = normalized
        normalized.apply_to_target(self)
        self._apply_to_window()
        if persist:
            self._persist_preferences(normalized)
        if refresh_overlay:
            self.optical_flow_tool.refresh_current_frame_overlay()

    def _apply_to_window(self) -> None:
        """Mirror current preferences onto the window for compatibility."""
        w = self.window
        setattr(w, "compute_optical_flow", self.compute_optical_flow)
        OpticalFlowPreferences.from_source(self).apply_to_target(w)

    def _persist_preferences(self, prefs: OpticalFlowPreferences) -> None:
        settings = getattr(self.window, "settings", None)
        if not isinstance(settings, QtCore.QSettings):
            return
        for key, value in prefs.qsettings_items():
            settings.setValue(key, value)

    def _load_settings(self) -> None:
        """Load persisted optical-flow preferences and sync to manager + window."""
        settings = getattr(self.window, "settings", None)
        if isinstance(settings, QtCore.QSettings):
            try:
                raw: dict[str, object] = {}
                for key, default_val in self._preferences.qsettings_items():
                    raw[key] = settings.value(key, default_val)
                loaded = OpticalFlowPreferences.from_source(
                    raw, defaults=self._preferences
                )
                self._preferences = loaded
                loaded.apply_to_target(self)
            except Exception:
                logger.warning(
                    "Failed to load persisted optical-flow settings; using defaults.",
                    exc_info=True,
                )
        self._apply_to_window()

    def set_backend(self, backend: str) -> None:
        self.optical_flow_backend = normalize_optical_flow_backend(backend)
        self._preferences.backend = self.optical_flow_backend
        self._apply_to_window()

    def set_compute_optical_flow(self, enabled: bool) -> None:
        self.compute_optical_flow = bool(enabled)
        self._apply_to_window()
