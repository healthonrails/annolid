from __future__ import annotations

import asyncio
import threading
from annolid.utils.logger import logger
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.core.agent.bus import MessageBus
from annolid.core.agent.bus.service import AgentBusService
from annolid.core.agent.channels.manager import ChannelManager
from annolid.core.agent.config import load_config
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.tools import (
    FunctionToolRegistry,
    register_nanobot_style_tools,
)
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.gui.widgets.ai_chat_widget import AIChatWidget


class AIChatManager(QtCore.QObject):
    """Manage the dedicated right-side Annolid Bot dock."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.ai_chat_dock: Optional[QtWidgets.QDockWidget] = None
        self.ai_chat_widget: Optional[AIChatWidget] = None
        self._background_bus: Optional[MessageBus] = None
        self._channel_manager: Optional[ChannelManager] = None
        self._bus_service: Optional[AgentBusService] = None
        self._background_thread: Optional[threading.Thread] = None
        self._background_loop: Optional[asyncio.AbstractEventLoop] = None

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self.window.set_unrelated_docks_visible(True)

    def _ensure_dock(self) -> tuple[QtWidgets.QDockWidget, AIChatWidget]:
        if self.ai_chat_dock is not None and self.ai_chat_widget is not None:
            return self.ai_chat_dock, self.ai_chat_widget

        dock = QtWidgets.QDockWidget(self.window.tr("Annolid Bot"), self.window)
        dock.setObjectName("aiChatDock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        widget = AIChatWidget(dock)
        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        widget.set_default_visual_share_mode(attach_canvas=False, attach_window=False)

        dock.setWidget(widget)
        dock.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.ai_chat_dock = dock
        self.ai_chat_widget = widget
        return dock, widget

    def show_chat_dock(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        hide_other_docks: bool = True,
    ) -> None:
        dock, widget = self._ensure_dock()
        self.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        image_path = getattr(self.window, "filename", None)
        if image_path:
            suffix = str(image_path).lower()
            if suffix.endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
            ):
                widget.set_image_path(image_path)
        if provider:
            widget.set_provider_and_model(provider, model or "")
        elif model:
            widget.set_provider_and_model(widget.selected_provider, model)

        if not dock.isVisible():
            if hide_other_docks:
                self.window.set_unrelated_docks_visible(False, exclude=[dock])
            dock.show()

        dock.raise_()
        widget.prompt_text_edit.setFocus()

    def initialize_annolid_bot(self, *, start_visible: bool = True) -> None:
        """Start bot session resources when the app launches."""
        dock, widget = self._ensure_dock()
        self.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        widget.set_canvas(getattr(self.window, "canvas", None))
        widget.set_host_window(self.window)
        if start_visible:
            if not dock.isVisible():
                self.show_chat_dock(hide_other_docks=False)
        else:
            # Some Qt setups briefly show dock widgets after addDockWidget.
            # Hide immediately and once again on the next event-loop tick.
            dock.hide()
            QtCore.QTimer.singleShot(0, dock.hide)

        # Start background automation services (Email polling, etc.)
        self._start_background_services()

    def _start_background_services(self) -> None:
        """Initialize and start background messaging services in a separate thread."""

        def _run_background_loop():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._background_loop = loop

            async def _async_start():
                try:
                    config = load_config()
                    if not config.tools.email.enabled:
                        return

                    self._background_bus = MessageBus()

                    # Setup Agent Loop for background replies
                    tools = FunctionToolRegistry()
                    await register_nanobot_style_tools(
                        tools,
                        allowed_dir=get_agent_workspace_path(),
                        email_cfg=config.tools.email,
                    )

                    # In background mode, we use a standard loop.
                    workspace = get_agent_workspace_path()
                    loop_instance = AgentLoop(tools=tools, workspace=workspace)
                    logger.info("AgentLoop initialized with workspace: %s", workspace)

                    self._bus_service = AgentBusService.from_agent_config(
                        bus=self._background_bus,
                        loop=loop_instance,
                        agent_config=config,
                    )

                    self._channel_manager = ChannelManager(
                        bus=self._background_bus,
                        channels_config={"email": config.tools.email.to_dict()},
                    )

                    await self._bus_service.start()
                    await self._channel_manager.start_all()

                    logger.info(
                        "Annolid Bot background services started (Email monitor enabled)"
                    )
                except Exception as exc:
                    logger.exception("Failed to start background services: %s", exc)

            # Schedule the start coroutine
            loop.create_task(_async_start())

            # Run the loop forever until stopped
            try:
                loop.run_forever()
            finally:
                loop.close()
                self._background_loop = None

        self._background_thread = threading.Thread(
            target=_run_background_loop, name="AnnolidBotBackground", daemon=True
        )
        self._background_thread.start()

    def cleanup(self) -> None:
        """Stop background services and the event loop."""
        if self._background_loop is not None:
            # Thread-safe way to stop the loop from another thread (main Qt thread)
            self._background_loop.call_soon_threadsafe(self._background_loop.stop)

        if self._background_thread is not None:
            # We use daemon=True, but joining ensures it stops gracefully if possible.
            # Don't block for too long during GUI shutdown.
            self._background_thread.join(timeout=1.0)
            self._background_thread = None
