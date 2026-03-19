from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from qtpy import QtWidgets


class RenderQtBackend:
    """Adapter for Qt render widgets that expose a render window."""

    name = "qvtk"

    def __init__(self, widget_factory: Callable[..., QtWidgets.QWidget]) -> None:
        self._widget_factory = widget_factory

    def create_widget(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        return self._widget_factory(parent)

    def get_render_window(self, widget: Optional[QtWidgets.QWidget]):
        if widget is None:
            return None
        getter = getattr(widget, "GetRenderWindow", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def get_interactor(
        self,
        widget: Optional[QtWidgets.QWidget],
        render_window: Any,
    ):
        if render_window is None:
            return None
        getter = getattr(render_window, "GetInteractor", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def reset_camera(self, renderer: Any, widget: Optional[QtWidgets.QWidget]) -> None:
        if renderer is None:
            return
        try:
            renderer.ResetCamera()
        except Exception:
            return

    def render(self, widget: Optional[QtWidgets.QWidget], render_window: Any) -> None:
        if render_window is None:
            return
        try:
            render_window.Render()
        except Exception:
            return

    def set_background(
        self,
        renderer: Any,
        widget: Optional[QtWidgets.QWidget],
        color: tuple[float, float, float],
    ) -> None:
        if renderer is None:
            return
        try:
            renderer.SetBackground(float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            return

    def add_actor(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], actor: Any
    ) -> None:
        if renderer is None:
            return
        renderer.AddActor(actor)

    def remove_actor(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], actor: Any
    ) -> None:
        if renderer is None:
            return
        renderer.RemoveActor(actor)

    def add_volume(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], volume: Any
    ) -> None:
        if renderer is None:
            return
        renderer.AddVolume(volume)

    def remove_volume(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], volume: Any
    ) -> None:
        if renderer is None:
            return
        renderer.RemoveVolume(volume)

    def add_light(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], light: Any
    ) -> None:
        if renderer is None:
            return
        renderer.AddLight(light)

    def get_active_camera(self, renderer: Any, widget: Optional[QtWidgets.QWidget]):
        if renderer is None:
            return None
        getter = getattr(renderer, "GetActiveCamera", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def reset_camera_clipping_range(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget]
    ) -> None:
        if renderer is None:
            return
        try:
            renderer.ResetCameraClippingRange()
        except Exception:
            return

    def has_view_prop(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], prop: Any
    ) -> bool:
        if renderer is None:
            return False
        checker = getattr(renderer, "HasViewProp", None)
        if checker is None:
            return False
        try:
            return bool(checker(prop))
        except Exception:
            return False


class PyVistaQtBackend(RenderQtBackend):
    """PyVista-backed render widget adapter."""

    name = "pyvistaqt"

    def get_render_window(self, widget: Optional[QtWidgets.QWidget]):
        if widget is None:
            return None
        # pyvistaqt.QtInteractor usually proxies GetRenderWindow, but keep
        # compatible fallbacks to avoid hard coupling to one internal name.
        render_window = super().get_render_window(widget)
        if render_window is not None:
            return render_window
        for attr in ("ren_win", "render_window"):
            candidate = getattr(widget, attr, None)
            if candidate is not None:
                return candidate
        return None

    def get_interactor(
        self,
        widget: Optional[QtWidgets.QWidget],
        render_window: Any,
    ):
        interactor = getattr(widget, "interactor", None) if widget is not None else None
        if interactor is not None:
            return interactor
        return super().get_interactor(widget, render_window)

    def reset_camera(self, renderer: Any, widget: Optional[QtWidgets.QWidget]) -> None:
        if widget is not None:
            fn = getattr(widget, "reset_camera", None)
            if callable(fn):
                try:
                    fn(render=False)
                    return
                except Exception:
                    pass
        super().reset_camera(renderer, widget)

    def render(self, widget: Optional[QtWidgets.QWidget], render_window: Any) -> None:
        if widget is not None:
            fn = getattr(widget, "render", None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass
        super().render(widget, render_window)

    def set_background(
        self,
        renderer: Any,
        widget: Optional[QtWidgets.QWidget],
        color: tuple[float, float, float],
    ) -> None:
        if widget is not None:
            fn = getattr(widget, "set_background", None)
            if callable(fn):
                try:
                    fn(color=color, render=False)
                    return
                except Exception:
                    pass
        super().set_background(renderer, widget, color)

    def add_actor(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], actor: Any
    ) -> None:
        if widget is not None:
            fn = getattr(widget, "add_actor", None)
            if callable(fn):
                try:
                    fn(actor, render=False, reset_camera=False)
                    return
                except Exception:
                    pass
        super().add_actor(renderer, widget, actor)

    def remove_actor(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], actor: Any
    ) -> None:
        if widget is not None:
            fn = getattr(widget, "remove_actor", None)
            if callable(fn):
                try:
                    fn(actor, render=False)
                    return
                except Exception:
                    pass
        super().remove_actor(renderer, widget, actor)

    def add_volume(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], volume: Any
    ) -> None:
        # Many code paths construct vtkVolume directly; PyVista's add_actor
        # handles vtkProp input while preserving this contract.
        if widget is not None:
            fn = getattr(widget, "add_actor", None)
            if callable(fn):
                try:
                    fn(volume, render=False, reset_camera=False)
                    return
                except Exception:
                    pass
        super().add_volume(renderer, widget, volume)

    def remove_volume(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], volume: Any
    ) -> None:
        if widget is not None:
            fn = getattr(widget, "remove_actor", None)
            if callable(fn):
                try:
                    fn(volume, render=False)
                    return
                except Exception:
                    pass
        super().remove_volume(renderer, widget, volume)

    def add_light(
        self, renderer: Any, widget: Optional[QtWidgets.QWidget], light: Any
    ) -> None:
        if widget is not None:
            fn = getattr(widget, "add_light", None)
            if callable(fn):
                try:
                    fn(light)
                    return
                except Exception:
                    pass
        super().add_light(renderer, widget, light)


@dataclass(frozen=True)
class RenderQtBackendSelection:
    backend: RenderQtBackend
    pyvista_error: Optional[str] = None


def select_render_qt_backend(
    *,
    widget_cls: Optional[Callable[..., QtWidgets.QWidget]],
    prefer_pyvista: bool = True,
) -> RenderQtBackendSelection:
    """Select a Qt render backend, preferring PyVista when available."""
    pyvista_error: Optional[str] = None
    if prefer_pyvista:
        try:
            from pyvistaqt import QtInteractor

            return RenderQtBackendSelection(backend=PyVistaQtBackend(QtInteractor))
        except Exception as exc:  # pragma: no cover - optional dependency path
            pyvista_error = str(exc) or exc.__class__.__name__
    if widget_cls is not None:
        return RenderQtBackendSelection(
            backend=RenderQtBackend(widget_cls),
            pyvista_error=pyvista_error,
        )
    detail = f" (PyVista unavailable: {pyvista_error})" if pyvista_error else ""
    raise ModuleNotFoundError(
        f"No Qt render backend is available{detail}. Install pyvistaqt."
    )
