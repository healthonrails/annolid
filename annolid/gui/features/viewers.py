"""Viewer stack and related manager setup for main window."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from annolid.gui.features.container import GuiFeatureDeps
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
from annolid.gui.widgets.ai_chat_manager import AIChatManager


@dataclass(frozen=True)
class ViewersFeatureState:
    viewer_stack: QtWidgets.QStackedWidget
    main_scroll_area: QtWidgets.QScrollArea
    large_image_view: TiledImageView
    pdf_manager: PdfManager
    web_manager: WebManager
    threejs_manager: ThreeJsManager
    depth_manager: DepthManager
    optical_flow_manager: OpticalFlowManager
    sam3d_manager: Sam3DManager
    sam2_manager: Sam2Manager
    sam3_manager: Sam3Manager
    realtime_manager: RealtimeManager
    ai_chat_manager: AIChatManager


def setup_viewers_feature(deps: GuiFeatureDeps) -> ViewersFeatureState:
    """Create the canvas viewer stack and viewer managers."""
    window = deps.window
    window._viewer_stack = QtWidgets.QStackedWidget()
    window._viewer_stack.setContentsMargins(0, 0, 0, 0)
    window._viewer_stack.addWidget(window.canvas)
    window.large_image_view = TiledImageView(window)
    window.large_image_view.set_host_window(window)
    window._viewer_stack.addWidget(window.large_image_view)

    window.pdf_manager = PdfManager(window, window._viewer_stack)
    window.web_manager = WebManager(window, window._viewer_stack)
    window.threejs_manager = ThreeJsManager(window, window._viewer_stack)
    window.depth_manager = DepthManager(window)
    window.optical_flow_manager = OpticalFlowManager(window)
    window.sam3d_manager = Sam3DManager(window)
    window.sam2_manager = Sam2Manager(window)
    window.sam3_manager = Sam3Manager(window)
    window.realtime_manager = RealtimeManager(window)
    window.ai_chat_manager = AIChatManager(window)

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
