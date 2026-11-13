import contextlib
import gc
import os
import sys

# Select PySide6 as the default Qt backend for tests to avoid collection errors
# when PyQt5 is missing or mis-detected.
os.environ.setdefault("QT_API", "pyside6")
# Keep headless tests from implicitly selecting QtAgg once Qt modules are loaded.
os.environ.setdefault("MPLBACKEND", "Agg")

import importlib.util  # noqa: E402

import pytest  # noqa: E402

_QAPP_REF: object = None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _simulation_skip_reason() -> str | None:
    if os.environ.get("ANNOLID_RUN_FLYBODY_RUNTIME") != "1":
        return (
            "set ANNOLID_RUN_FLYBODY_RUNTIME=1 to run optional simulation runtime tests"
        )

    required_modules = ("flybody", "dm_control", "mujoco")
    missing = [name for name in required_modules if not _has_module(name)]
    if missing:
        missing_text = ", ".join(missing)
        return (
            f"optional FlyBody runtime dependencies are not installed: {missing_text}"
        )

    return None


def _active_provider_skip_reason() -> str | None:
    if os.environ.get("ANNOLID_RUN_ACTIVE_PROVIDER_TESTS") == "1":
        return None

    return (
        "active-provider tests are opt-in; set "
        "ANNOLID_RUN_ACTIVE_PROVIDER_TESTS=1 after provider runtime setup"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    reason = _simulation_skip_reason()
    active_provider_reason = _active_provider_skip_reason()
    skip_marker = pytest.mark.skip(reason=reason) if reason else None
    active_provider_skip = (
        pytest.mark.skip(reason=active_provider_reason)
        if active_provider_reason
        else None
    )
    for item in items:
        if skip_marker and item.get_closest_marker("simulation") is not None:
            item.add_marker(skip_marker)
        if (
            active_provider_skip
            and item.get_closest_marker("active_provider") is not None
        ):
            item.add_marker(active_provider_skip)


def _loaded_qt_modules():
    for prefix in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
        qt_core = sys.modules.get(f"{prefix}.QtCore")
        qt_widgets = sys.modules.get(f"{prefix}.QtWidgets")
        if qt_core is not None and qt_widgets is not None:
            return qt_core, qt_widgets
    return None, None


def _widget_key(widget: object) -> int:
    shiboken = sys.modules.get("shiboken6")
    if shiboken is not None:
        with contextlib.suppress(Exception):
            if not shiboken.isValid(widget):
                return -1
            ptrs = shiboken.getCppPointer(widget)
            if ptrs:
                return int(ptrs[0])
    return id(widget)


def _is_python_owned_top_level_widget(widget: object, qt_core: object) -> bool:
    shiboken = sys.modules.get("shiboken6")
    if shiboken is not None:
        try:
            return bool(shiboken.isValid(widget) and shiboken.ownedByPython(widget))
        except Exception:
            return False
    # Fallback for non-Shiboken bindings: skip C++-managed transient popups/tooltips.
    with contextlib.suppress(Exception):
        win_type = widget.windowType()  # type: ignore[attr-defined]
        qt_ns = getattr(qt_core, "Qt", None)
        if qt_ns is not None:
            popup_types = {
                getattr(qt_ns, "Popup", None),
                getattr(qt_ns, "ToolTip", None),
            }
            window_type_enum = getattr(qt_ns, "WindowType", None)
            if window_type_enum is not None:
                popup_types.add(getattr(window_type_enum, "Popup", None))
                popup_types.add(getattr(window_type_enum, "ToolTip", None))
            if win_type in popup_types:
                return False
    return True


def _snapshot_top_level_widget_keys() -> set[int]:
    global _QAPP_REF
    _, qt_widgets = _loaded_qt_modules()
    if qt_widgets is None:
        return set()
    app = qt_widgets.QApplication.instance()
    if app is None:
        return set()
    _QAPP_REF = app
    keys: set[int] = set()
    for widget in list(app.topLevelWidgets()):
        key = _widget_key(widget)
        if key != -1:
            keys.add(key)
    return keys


def _drain_qt_widgets(
    *, skip_keys: set[int] | None = None, force_gc: bool = False
) -> None:
    global _QAPP_REF
    qt_core, qt_widgets = _loaded_qt_modules()
    if qt_core is None or qt_widgets is None:
        return
    app = qt_widgets.QApplication.instance()
    if app is None:
        return
    _QAPP_REF = app

    deferred_delete = getattr(
        getattr(qt_core.QEvent, "Type", qt_core.QEvent),
        "DeferredDelete",
        qt_core.QEvent.DeferredDelete,
    )

    targets = []
    for widget in list(app.topLevelWidgets()):
        key = _widget_key(widget)
        if key == -1:
            continue
        if skip_keys is not None and key in skip_keys:
            continue
        if not _is_python_owned_top_level_widget(widget, qt_core):
            continue
        targets.append(widget)

    for widget in targets:
        if _widget_key(widget) == -1:
            continue
        if getattr(widget, "dirty", False) and hasattr(widget, "setClean"):
            with contextlib.suppress(Exception):
                widget.setClean()
        clean_up = getattr(widget, "clean_up", None)
        if callable(clean_up):
            with contextlib.suppress(Exception):
                clean_up()
        with contextlib.suppress(Exception):
            widget.close()

    if not targets and not force_gc:
        with contextlib.suppress(Exception):
            app.processEvents()
            app.sendPostedEvents(None, deferred_delete)
        return

    # Let pending 0ms single-shot callbacks run before scheduling C++ widget deletion.
    with contextlib.suppress(Exception):
        app.processEvents()

    for widget in targets:
        if not _is_python_owned_top_level_widget(widget, qt_core):
            continue
        with contextlib.suppress(Exception):
            widget.deleteLater()

    for _ in range(3):
        with contextlib.suppress(Exception):
            app.sendPostedEvents(None, deferred_delete)
            app.processEvents()
    gc.collect()
    with contextlib.suppress(Exception):
        app.sendPostedEvents(None, deferred_delete)
        app.processEvents()


@pytest.fixture(autouse=True)
def _cleanup_qt_test_widgets():
    before_keys = _snapshot_top_level_widget_keys()
    yield
    _drain_qt_widgets(skip_keys=before_keys)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        with contextlib.suppress(Exception):
            pyplot.close("all")

    qt_core, qt_widgets = _loaded_qt_modules()
    if qt_core is None or qt_widgets is None:
        return
    app = qt_widgets.QApplication.instance()
    if app is None:
        return

    with contextlib.suppress(Exception):
        pool = qt_core.QThreadPool.globalInstance()
        if pool is not None:
            pool.waitForDone(2000)

    _drain_qt_widgets(skip_keys=None, force_gc=True)
    _drain_qt_widgets(skip_keys=None, force_gc=True)
