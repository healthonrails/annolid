from __future__ import annotations

from typing import Callable


class VTKRenderLifecycleController:
    """Owns VTK init/render scheduling while delegating concrete operations to host."""

    def __init__(
        self,
        host: object,
        *,
        single_shot: Callable[[int, Callable[[], None]], None],
        max_retries: int,
        retry_base_ms: int,
    ) -> None:
        self._host = host
        self._single_shot = single_shot
        self._max_retries = max_retries
        self._retry_base_ms = retry_base_ms

    def _h(self):
        return self._host

    def on_show(self) -> None:
        self.schedule_initialization(0)

    def schedule_initialization(self, delay_ms: int) -> None:
        h = self._h()
        if getattr(h, "_vtk_initialized", False) or getattr(
            h, "_vtk_init_scheduled", False
        ):
            return
        h._vtk_init_scheduled = True
        self._single_shot(max(0, int(delay_ms)), h._start_vtk_initialization)

    def start_initialization(self) -> None:
        h = self._h()
        h._vtk_init_scheduled = False
        if not h._ensure_vtk_initialized():
            h._set_load_busy(False)
            h._vtk_init_failures = int(getattr(h, "_vtk_init_failures", 0)) + 1
            if h._vtk_init_failures < self._max_retries and h.isVisible():
                self.schedule_initialization(self._retry_base_ms * h._vtk_init_failures)
            h._refresh_status_summary()
            return
        h._vtk_init_failures = 0
        h._vtk_init_last_error = ""
        if getattr(h, "_initial_source_load_pending", False) and not getattr(
            h, "_initial_source_load_scheduled", False
        ):
            h._initial_source_load_scheduled = True
            self._single_shot(0, h._load_initial_source)

    def request_render(self, *, reset_camera: bool = False) -> None:
        h = self._h()
        if reset_camera:
            h._render_reset_camera_pending = True
        if getattr(h, "_render_pending", False):
            return
        if not getattr(h, "_vtk_initialized", False):
            self.schedule_initialization(0)
            return
        h._render_pending = True

        def _do_render() -> None:
            if not h.isVisible():
                h._render_pending = False
                return
            h._render_pending = False
            try:
                if getattr(h, "_render_reset_camera_pending", False):
                    h.renderer.ResetCamera()
            except Exception:
                pass
            finally:
                h._render_reset_camera_pending = False
            try:
                render_window = h._get_render_window()
                if render_window is not None:
                    render_window.Render()
            except Exception:
                pass

        self._single_shot(0, _do_render)
