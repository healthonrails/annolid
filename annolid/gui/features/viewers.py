"""Viewer stack and related manager setup for main window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps

if TYPE_CHECKING:
    from annolid.gui.widgets.depth_manager import DepthManager
    from annolid.gui.widgets.tiled_image_view import TiledImageView
    from annolid.gui.widgets.optical_flow_manager import OpticalFlowManager
    from annolid.gui.widgets.pdf_manager import PdfManager
    from annolid.gui.widgets.realtime_manager import RealtimeManager
    from annolid.gui.widgets.sam2_manager import Sam2Manager
    from annolid.gui.widgets.sam3_manager import Sam3Manager
    from annolid.gui.widgets.sam3d_manager import Sam3DManager
    from annolid.gui.widgets.threejs_manager import ThreeJsManager
    from annolid.gui.widgets.web_manager import WebManager


@dataclass(frozen=True)
class ViewersFeatureState:
    viewer_stack: QtWidgets.QStackedWidget
    main_scroll_area: QtWidgets.QScrollArea
    large_image_view: TiledImageView
    pdf_manager: Optional[PdfManager]
    web_manager: Optional[WebManager]
    threejs_manager: Optional[ThreeJsManager]
    depth_manager: Optional[DepthManager]
    optical_flow_manager: OpticalFlowManager
    sam3d_manager: Optional[Sam3DManager]
    sam2_manager: Sam2Manager
    sam3_manager: Sam3Manager
    realtime_manager: Optional[RealtimeManager]
    ai_chat_manager: object


class _LazyAIChatManager:
    """Stable ai_chat_manager facade that instantiates the heavy manager on demand."""

    def __init__(self, window) -> None:
        self._window = window
        self._manager = None

    def _resolve(self):
        if self._manager is None:
            from annolid.gui.widgets.ai_chat_manager import AIChatManager

            self._manager = AIChatManager(self._window)
        return self._manager

    def initialize_annolid_bot(self, *args, **kwargs):
        return self._resolve().initialize_annolid_bot(*args, **kwargs)

    def show_chat_dock(self, *args, **kwargs):
        return self._resolve().show_chat_dock(*args, **kwargs)

    def cleanup(self):
        if self._manager is None:
            return None
        return self._manager.cleanup()

    @property
    def ai_chat_widget(self):
        if self._manager is None:
            return None
        return getattr(self._manager, "ai_chat_widget", None)

    @property
    def ai_chat_dock(self):
        if self._manager is None:
            return None
        return getattr(self._manager, "ai_chat_dock", None)


def ensure_pdf_manager(window):
    manager = getattr(window, "pdf_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.pdf_manager import PdfManager

    manager = PdfManager(window, window._viewer_stack)
    window.pdf_manager = manager
    return manager


def ensure_web_manager(window):
    manager = getattr(window, "web_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.web_manager import WebManager

    manager = WebManager(window, window._viewer_stack)
    window.web_manager = manager
    return manager


def ensure_threejs_manager(window):
    manager = getattr(window, "threejs_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.threejs_manager import ThreeJsManager

    manager = ThreeJsManager(window, window._viewer_stack)
    window.threejs_manager = manager
    return manager


def ensure_depth_manager(window):
    manager = getattr(window, "depth_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.depth_manager import DepthManager

    manager = DepthManager(window)
    window.depth_manager = manager
    return manager


def ensure_sam3d_manager(window):
    manager = getattr(window, "sam3d_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.sam3d_manager import Sam3DManager

    manager = Sam3DManager(window)
    window.sam3d_manager = manager
    return manager


def ensure_realtime_manager(window):
    manager = getattr(window, "realtime_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.realtime_manager import RealtimeManager

    manager = RealtimeManager(window)
    window.realtime_manager = manager
    return manager


def ensure_ai_chat_manager(window):
    manager = getattr(window, "ai_chat_manager", None)
    if manager is not None:
        return manager
    from annolid.gui.widgets.ai_chat_manager import AIChatManager

    manager = AIChatManager(window)
    window.ai_chat_manager = manager
    return manager


def setup_viewers_feature(deps: GuiFeatureDeps) -> ViewersFeatureState:
    """Create the canvas viewer stack and viewer managers."""
    from annolid.gui.widgets.tiled_image_view import TiledImageView
    from annolid.gui.widgets.optical_flow_manager import OpticalFlowManager
    from annolid.gui.widgets.sam2_manager import Sam2Manager
    from annolid.gui.widgets.sam3_manager import Sam3Manager

    window = deps.window
    window._viewer_stack = QtWidgets.QStackedWidget()
    window._viewer_stack.setContentsMargins(0, 0, 0, 0)
    window._viewer_stack.addWidget(window.canvas)
    window.large_image_view = TiledImageView(window)
    window.large_image_view.set_host_window(window)
    window._viewer_stack.addWidget(window.large_image_view)

    # Keep heavy managers lazy so app startup remains responsive.
    window.pdf_manager = None
    window.web_manager = None
    window.threejs_manager = None
    window.depth_manager = None
    window.optical_flow_manager = OpticalFlowManager(window)
    window.sam3d_manager = None
    window.sam2_manager = Sam2Manager(window)
    window.sam3_manager = Sam3Manager(window)
    window.realtime_manager = None
    window.ai_chat_manager = _LazyAIChatManager(window)

    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidget(window._viewer_stack)
    scroll_area.setWidgetResizable(True)
    scroll_area.setAlignment(Qt.AlignCenter)
    window.scrollBars = {
        Qt.Vertical: scroll_area.verticalScrollBar(),
        Qt.Horizontal: scroll_area.horizontalScrollBar(),
    }
    window._main_scroll_area = scroll_area
    return ViewersFeatureState(
        viewer_stack=window._viewer_stack,
        main_scroll_area=window._main_scroll_area,
        large_image_view=window.large_image_view,
        pdf_manager=window.pdf_manager,
        web_manager=window.web_manager,
        threejs_manager=window.threejs_manager,
        depth_manager=window.depth_manager,
        optical_flow_manager=window.optical_flow_manager,
        sam3d_manager=window.sam3d_manager,
        sam2_manager=window.sam2_manager,
        sam3_manager=window.sam3_manager,
        realtime_manager=window.realtime_manager,
        ai_chat_manager=window.ai_chat_manager,
    )
