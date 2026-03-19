from __future__ import annotations

import time
from typing import Callable

from annolid.utils.logger import logger


class RenderLifecycleController:
    """Owns render init/render scheduling while delegating concrete operations to host."""

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
        logger.info(
            "3D viewer lifecycle: show event received, scheduling render initialization."
        )
        try:
            self._h()._log_cursor_state("lifecycle.on_show")
        except Exception:
            pass
        self.schedule_initialization(0)

    def schedule_initialization(self, delay_ms: int) -> None:
        h = self._h()
        if getattr(h, "_scene_initialized", False) or getattr(
            h, "_scene_init_scheduled", False
        ):
            logger.info(
                "3D viewer lifecycle: initialization already ready/scheduled (initialized=%s scheduled=%s).",
                getattr(h, "_scene_initialized", False),
                getattr(h, "_scene_init_scheduled", False),
            )
            return
        h._scene_init_scheduled = True
        starter = h._start_scene_initialization
        logger.info(
            "3D viewer lifecycle: scheduling render initialization in %d ms.",
            max(0, int(delay_ms)),
        )
        self._single_shot(max(0, int(delay_ms)), starter)

    def start_initialization(self) -> None:
        h = self._h()
        h._scene_init_scheduled = False
        ensure_fn = h._ensure_scene_initialized
        if not ensure_fn():
            logger.warning(
                "3D viewer lifecycle: render initialization failed (attempt=%s, last_error=%s).",
                int(getattr(h, "_scene_init_failures", 0)) + 1,
                getattr(h, "_scene_init_last_error", ""),
            )
            h._set_load_busy(False)
            failures = int(getattr(h, "_scene_init_failures", 0)) + 1
            h._scene_init_failures = failures
            if failures < self._max_retries and h.isVisible():
                self.schedule_initialization(self._retry_base_ms * failures)
            h._refresh_status_summary()
            try:
                h._log_cursor_state("lifecycle.start_initialization.failed")
            except Exception:
                pass
            return
        logger.info("3D viewer lifecycle: render initialization succeeded.")
        h._scene_init_failures = 0
        h._scene_init_last_error = ""
        if getattr(h, "_initial_source_load_pending", False) and not getattr(
            h, "_initial_source_load_scheduled", False
        ):
            h._initial_source_load_scheduled = True
            logger.info("3D viewer lifecycle: scheduling initial source load.")
            self._single_shot(0, h._load_initial_scene_source)
        try:
            h._log_cursor_state("lifecycle.start_initialization.succeeded")
        except Exception:
            pass

    def request_render(self, *, reset_camera: bool = False) -> None:
        h = self._h()
        if reset_camera:
            h._scene_reset_camera_pending = True
        if getattr(h, "_scene_render_pending", False):
            h._scene_refresh_requested = True
            logger.info(
                "3D viewer lifecycle: render already pending, marking refresh requested."
            )
            return
        h._scene_refresh_requested = False
        if not getattr(h, "_scene_initialized", False):
            logger.info(
                "3D viewer lifecycle: render not initialized, scheduling init instead of render."
            )
            self.schedule_initialization(0)
            return
        h._scene_render_pending = True
        h._scene_render_pending_since = time.monotonic()
        h._scene_hidden_retry_count = 0
        logger.info(
            "3D viewer lifecycle: render queued reset_camera=%s pending_since=%.6f",
            reset_camera,
            float(getattr(h, "_scene_render_pending_since", 0.0) or 0.0),
        )

        def _do_render() -> None:
            logger.info("3D viewer lifecycle: render callback started.")
            if not h.isVisible():
                retries = int(getattr(h, "_scene_hidden_retry_count", 0)) + 1
                h._scene_hidden_retry_count = retries
                logger.info(
                    "3D viewer lifecycle: render deferred because viewer is hidden (retry=%d).",
                    retries,
                )
                if retries <= 20:
                    self._single_shot(100, _do_render)
                    return
                logger.warning(
                    "3D viewer lifecycle: abandoning deferred render after %d hidden retries.",
                    retries,
                )
                h._scene_render_pending = False
                h._scene_render_pending_since = 0.0
                h._scene_hidden_retry_count = 0
                try:
                    h._log_cursor_state("lifecycle.request_render.hidden_abandon")
                except Exception:
                    pass
                return
            refresh_requested = bool(getattr(h, "_scene_refresh_requested", False))
            h._scene_refresh_requested = False
            h._scene_render_pending = False
            h._scene_render_pending_since = 0.0
            h._scene_hidden_retry_count = 0
            try:
                if getattr(h, "_scene_reset_camera_pending", False):
                    reset_fn = getattr(h, "_reset_scene_camera", None)
                    if callable(reset_fn):
                        reset_fn()
                    else:
                        h.renderer.ResetCamera()
            except Exception:
                pass
            finally:
                h._scene_reset_camera_pending = False
            try:
                render_fn = getattr(h, "_present_scene", None)
                if callable(render_fn):
                    render_fn()
                else:
                    render_window = h._get_render_window()
                    if render_window is not None:
                        render_window.Render()
            except Exception:
                pass
            try:
                h._set_load_busy(False)
            except Exception:
                pass
            try:
                h._log_cursor_state("lifecycle.request_render.rendered")
            except Exception:
                pass
            if refresh_requested:
                logger.info(
                    "3D viewer lifecycle: scheduling follow-up render for refreshed scene."
                )
                self.request_render()

        logger.info("3D viewer lifecycle: render callback scheduled in 0 ms.")
        scheduler = getattr(h, "_schedule_scene_render_callback", None)
        if callable(scheduler):
            scheduler(_do_render, 0)
            return
        self._single_shot(0, _do_render)
