from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qtpy import QtCore

from annolid.gui.widgets.optical_flow_tool import OpticalFlowTool


@dataclass
class _FlowDefaults:
    backend: str = "farneback"
    raft_model: str = "small"
    visualization: str = "hsv"
    opacity: int = 70
    quiver_step: int = 16
    quiver_gain: float = 1.0
    stable_hsv: bool = True
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 1
    farneback_winsize: int = 1
    farneback_iterations: int = 3
    farneback_poly_n: int = 3
    farneback_poly_sigma: float = 1.1


class OpticalFlowManager(QtCore.QObject):
    """Encapsulates optical flow settings, tool, and overlays."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.compute_optical_flow: bool = True
        self.optical_flow_backend: str = _FlowDefaults.backend
        self.optical_flow_raft_model: str = _FlowDefaults.raft_model
        self.flow_visualization: str = _FlowDefaults.visualization
        self.flow_opacity: int = _FlowDefaults.opacity
        self.flow_quiver_step: int = _FlowDefaults.quiver_step
        self.flow_quiver_gain: float = _FlowDefaults.quiver_gain
        self.flow_stable_hsv: bool = _FlowDefaults.stable_hsv
        self.flow_farneback_pyr_scale: float = _FlowDefaults.farneback_pyr_scale
        self.flow_farneback_levels: int = _FlowDefaults.farneback_levels
        self.flow_farneback_winsize: int = _FlowDefaults.farneback_winsize
        self.flow_farneback_iterations: int = _FlowDefaults.farneback_iterations
        self.flow_farneback_poly_n: int = _FlowDefaults.farneback_poly_n
        self.flow_farneback_poly_sigma: float = _FlowDefaults.farneback_poly_sigma

        self.optical_flow_tool = OpticalFlowTool(window)
        self._load_settings()

    # ------------------------------------------------------------------ public API
    def run_tool(self) -> None:
        self.optical_flow_tool.run()

    def configure_tool(self) -> None:
        self.optical_flow_tool.configure()

    def clear(self) -> None:
        self.optical_flow_tool.clear()

    def load_records(self, video_file: Optional[str] = None) -> None:
        self.optical_flow_tool.load_records(video_file)

    def update_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional["np.ndarray"] = None
    ) -> None:
        self.optical_flow_tool.update_overlay_for_frame(
            frame_number, frame_rgb)

    # ------------------------------------------------------------------ settings
    def _apply_to_window(self) -> None:
        """Mirror current preferences onto the window for compatibility."""
        w = self.window
        setattr(w, "compute_optical_flow", self.compute_optical_flow)
        setattr(w, "optical_flow_backend", self.optical_flow_backend)
        setattr(w, "optical_flow_raft_model", self.optical_flow_raft_model)
        setattr(w, "flow_visualization", self.flow_visualization)
        setattr(w, "flow_opacity", self.flow_opacity)
        setattr(w, "flow_quiver_step", self.flow_quiver_step)
        setattr(w, "flow_quiver_gain", self.flow_quiver_gain)
        setattr(w, "flow_stable_hsv", self.flow_stable_hsv)
        setattr(w, "flow_farneback_pyr_scale", self.flow_farneback_pyr_scale)
        setattr(w, "flow_farneback_levels", self.flow_farneback_levels)
        setattr(w, "flow_farneback_winsize", self.flow_farneback_winsize)
        setattr(w, "flow_farneback_iterations", self.flow_farneback_iterations)
        setattr(w, "flow_farneback_poly_n", self.flow_farneback_poly_n)
        setattr(w, "flow_farneback_poly_sigma", self.flow_farneback_poly_sigma)

    def _load_settings(self) -> None:
        """Load persisted optical-flow preferences and sync to window."""
        settings = getattr(self.window, "settings", None)
        if isinstance(settings, QtCore.QSettings):
            try:
                self.optical_flow_backend = str(
                    settings.value("optical_flow/backend",
                                   self.optical_flow_backend)
                )
                self.optical_flow_raft_model = str(
                    settings.value("optical_flow/raft_model",
                                   self.optical_flow_raft_model)
                )
                self.flow_visualization = str(
                    settings.value("optical_flow/visualization",
                                   self.flow_visualization)
                )
                self.flow_opacity = int(
                    settings.value("optical_flow/opacity", self.flow_opacity)
                )
                self.flow_quiver_step = int(
                    settings.value("optical_flow/quiver_step",
                                   self.flow_quiver_step)
                )
                self.flow_quiver_gain = float(
                    settings.value("optical_flow/quiver_gain",
                                   self.flow_quiver_gain)
                )
                self.flow_stable_hsv = bool(
                    settings.value("optical_flow/stable_hsv",
                                   self.flow_stable_hsv)
                )
                self.flow_farneback_pyr_scale = float(
                    settings.value(
                        "optical_flow/farneback_pyr_scale", self.flow_farneback_pyr_scale
                    )
                )
                self.flow_farneback_levels = int(
                    settings.value("optical_flow/farneback_levels",
                                   self.flow_farneback_levels)
                )
                self.flow_farneback_winsize = int(
                    settings.value("optical_flow/farneback_winsize",
                                   self.flow_farneback_winsize)
                )
                self.flow_farneback_iterations = int(
                    settings.value(
                        "optical_flow/farneback_iterations", self.flow_farneback_iterations
                    )
                )
                self.flow_farneback_poly_n = int(
                    settings.value("optical_flow/farneback_poly_n",
                                   self.flow_farneback_poly_n)
                )
                self.flow_farneback_poly_sigma = float(
                    settings.value("optical_flow/farneback_poly_sigma",
                                   self.flow_farneback_poly_sigma)
                )
            except Exception:
                pass
        self._apply_to_window()

    def set_backend(self, backend: str) -> None:
        self.optical_flow_backend = backend
        self._apply_to_window()

    def set_compute_optical_flow(self, enabled: bool) -> None:
        self.compute_optical_flow = bool(enabled)
        self._apply_to_window()
