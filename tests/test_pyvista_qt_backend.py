from __future__ import annotations

import sys
import types

import pytest

from annolid.gui.widgets.pyvista_qt_backend import (
    PyVistaQtBackend,
    RenderQtBackend,
    select_render_qt_backend,
)


def test_select_render_qt_backend_returns_qvtk_when_pyvista_disabled() -> None:
    class _QVTKWidget:
        def __init__(self, parent):
            self.parent = parent

    selection = select_render_qt_backend(
        widget_cls=_QVTKWidget,
        prefer_pyvista=False,
    )

    assert isinstance(selection.backend, RenderQtBackend)
    assert selection.backend.name == "qvtk"


def test_select_render_qt_backend_prefers_pyvista_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.ModuleType("pyvistaqt")

    class _QtInteractor:
        def __init__(self, parent):
            self.parent = parent

    fake_module.QtInteractor = _QtInteractor
    monkeypatch.setitem(sys.modules, "pyvistaqt", fake_module)

    class _QVTKWidget:
        def __init__(self, parent):
            self.parent = parent

    selection = select_render_qt_backend(
        widget_cls=_QVTKWidget,
        prefer_pyvista=True,
    )

    assert isinstance(selection.backend, PyVistaQtBackend)
    assert selection.backend.name == "pyvistaqt"


def test_pyvista_backend_uses_ren_win_fallback() -> None:
    class _QtInteractor:
        def __init__(self, parent):
            self.parent = parent

    backend = PyVistaQtBackend(_QtInteractor)

    class _Widget:
        ren_win = "render-window"

    assert backend.get_render_window(_Widget()) == "render-window"


def test_select_render_qt_backend_raises_without_any_backend() -> None:
    with pytest.raises(ModuleNotFoundError):
        select_render_qt_backend(widget_cls=None, prefer_pyvista=False)


def test_render_backend_render_and_reset_camera_paths() -> None:
    class _QVTKWidget:
        def __init__(self, parent):
            self.parent = parent

    backend = RenderQtBackend(_QVTKWidget)

    class _Renderer:
        def __init__(self):
            self.reset_calls = 0

        def ResetCamera(self):
            self.reset_calls += 1

    class _RenderWindow:
        def __init__(self):
            self.render_calls = 0

        def Render(self):
            self.render_calls += 1

    renderer = _Renderer()
    rw = _RenderWindow()
    backend.reset_camera(renderer, None)
    backend.render(None, rw)

    assert renderer.reset_calls == 1
    assert rw.render_calls == 1


def test_pyvista_backend_render_and_reset_camera_paths() -> None:
    class _QtInteractor:
        def __init__(self, parent):
            self.parent = parent

    backend = PyVistaQtBackend(_QtInteractor)

    class _Widget:
        def __init__(self):
            self.reset_calls = []
            self.render_calls = 0

        def reset_camera(self, **kwargs):
            self.reset_calls.append(kwargs)

        def render(self):
            self.render_calls += 1

    widget = _Widget()
    backend.reset_camera(None, widget)
    backend.render(widget, None)

    assert widget.reset_calls == [{"render": False}]
    assert widget.render_calls == 1


def test_render_backend_scene_mutation_paths() -> None:
    class _QVTKWidget:
        def __init__(self, parent):
            self.parent = parent

    backend = RenderQtBackend(_QVTKWidget)

    class _Renderer:
        def __init__(self):
            self.calls = []

        def SetBackground(self, r, g, b):
            self.calls.append(("bg", r, g, b))

        def AddActor(self, actor):
            self.calls.append(("add_actor", actor))

        def RemoveActor(self, actor):
            self.calls.append(("remove_actor", actor))

        def AddVolume(self, volume):
            self.calls.append(("add_volume", volume))

        def RemoveVolume(self, volume):
            self.calls.append(("remove_volume", volume))

    renderer = _Renderer()
    backend.set_background(renderer, None, (0.1, 0.2, 0.3))
    backend.add_actor(renderer, None, "a")
    backend.remove_actor(renderer, None, "a")
    backend.add_volume(renderer, None, "v")
    backend.remove_volume(renderer, None, "v")

    assert renderer.calls == [
        ("bg", 0.1, 0.2, 0.3),
        ("add_actor", "a"),
        ("remove_actor", "a"),
        ("add_volume", "v"),
        ("remove_volume", "v"),
    ]


def test_pyvista_backend_scene_mutation_paths() -> None:
    class _QtInteractor:
        def __init__(self, parent):
            self.parent = parent

    backend = PyVistaQtBackend(_QtInteractor)

    class _Widget:
        def __init__(self):
            self.calls = []

        def set_background(self, **kwargs):
            self.calls.append(("bg", kwargs))

        def add_actor(self, actor, **kwargs):
            self.calls.append(("add_actor", actor, kwargs))

        def remove_actor(self, actor, **kwargs):
            self.calls.append(("remove_actor", actor, kwargs))

    widget = _Widget()
    backend.set_background(None, widget, (0.1, 0.2, 0.3))
    backend.add_actor(None, widget, "a")
    backend.remove_actor(None, widget, "a")
    backend.add_volume(None, widget, "v")
    backend.remove_volume(None, widget, "v")

    assert widget.calls == [
        ("bg", {"color": (0.1, 0.2, 0.3), "render": False}),
        ("add_actor", "a", {"render": False, "reset_camera": False}),
        ("remove_actor", "a", {"render": False}),
        ("add_actor", "v", {"render": False, "reset_camera": False}),
        ("remove_actor", "v", {"render": False}),
    ]
