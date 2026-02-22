from __future__ import annotations

import asyncio
import threading
from annolid.utils.logger import logger
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.core.agent.bus import MessageBus
from annolid.core.agent.bus.events import InboundMessage
from annolid.core.agent.bus.service import AgentBusService
from annolid.core.agent.channels.manager import ChannelManager
from annolid.core.agent.channels.whatsapp import WhatsAppChannel
from annolid.core.agent.channels.whatsapp_python_bridge import WhatsAppPythonBridge
from annolid.core.agent.channels.whatsapp_webhook_server import WhatsAppWebhookServer
from annolid.core.agent.config import load_config
from annolid.core.agent.cron import CronJob, CronService
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
        self._whatsapp_webhook_server: Optional[WhatsAppWebhookServer] = None
        self._whatsapp_python_bridge: Optional[WhatsAppPythonBridge] = None
        self._channel_start_task: Optional[asyncio.Task[None]] = None
        self._cron_service: Optional[CronService] = None

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
                    whatsapp_enabled = bool(config.tools.whatsapp.enabled)
                    whatsapp_auto_start = bool(config.tools.whatsapp.auto_start)
                    whatsapp_start_runtime = whatsapp_enabled and whatsapp_auto_start
                    logger.info(
                        "Background services config: email_enabled=%s whatsapp_enabled=%s whatsapp_auto_start=%s bridge_mode=%s webhook_enabled=%s max_parallel_sessions=%s max_pending_messages=%s collapse_superseded_pending=%s transient_retry_attempts=%s",
                        bool(config.tools.email.enabled),
                        whatsapp_enabled,
                        whatsapp_auto_start,
                        str(config.tools.whatsapp.bridge_mode or "python"),
                        bool(config.tools.whatsapp.webhook_enabled),
                        int(
                            getattr(config.agents.defaults, "max_parallel_sessions", 1)
                        ),
                        int(
                            getattr(
                                config.agents.defaults, "max_pending_messages", 2048
                            )
                        ),
                        bool(
                            getattr(
                                config.agents.defaults,
                                "collapse_superseded_pending",
                                True,
                            )
                        ),
                        int(
                            getattr(
                                config.agents.defaults, "transient_retry_attempts", 2
                            )
                        ),
                    )
                    if not (config.tools.email.enabled or whatsapp_start_runtime):
                        logger.info("Background services disabled by config")
                        return

                    self._background_bus = MessageBus()
                    whatsapp_cfg = config.tools.whatsapp.to_dict()
                    whatsapp_cfg["enabled"] = bool(whatsapp_start_runtime)
                    bridge_mode = (
                        str(config.tools.whatsapp.bridge_mode or "python")
                        .strip()
                        .lower()
                    )
                    if whatsapp_start_runtime and bridge_mode == "python":
                        bridge = WhatsAppPythonBridge(
                            host=config.tools.whatsapp.bridge_host,
                            port=int(config.tools.whatsapp.bridge_port),
                            token=config.tools.whatsapp.bridge_token,
                            session_dir=config.tools.whatsapp.bridge_session_dir,
                            headless=bool(config.tools.whatsapp.bridge_headless),
                        )
                        await bridge.start()
                        self._whatsapp_python_bridge = bridge
                        whatsapp_cfg["bridge_url"] = bridge.bridge_url
                        logger.info(
                            "Embedded WhatsApp Python bridge started at %s",
                            bridge.bridge_url,
                        )

                    # Setup Agent Loop for background replies
                    tools = FunctionToolRegistry()
                    calendar_cfg = getattr(config.tools, "calendar", None)
                    await register_nanobot_style_tools(
                        tools,
                        allowed_dir=get_agent_workspace_path(),
                        email_cfg=config.tools.email,
                        calendar_cfg=calendar_cfg,
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
                        channels_config={
                            "email": config.tools.email.to_dict(),
                            "whatsapp": whatsapp_cfg,
                        },
                    )

                    async def _on_cron_job(job: CronJob) -> Optional[str]:
                        if not self._background_bus:
                            return "Error: background bus unavailable"
                        channel = str(job.payload.channel or "cli")
                        chat_id = str(job.payload.to or "direct")
                        msg = str(job.payload.message or "").strip()
                        if msg:
                            await self._background_bus.publish_inbound(
                                InboundMessage(
                                    channel=channel,
                                    chat_id=chat_id,
                                    content=msg,
                                )
                            )
                        return "Inbound generated"

                    cron_store_path = workspace / "cron" / "jobs.json"
                    self._cron_service = CronService(
                        store_path=cron_store_path,
                        on_job=_on_cron_job,
                        logger=logger,
                    )

                    await self._bus_service.start()
                    await self._cron_service.start()
                    if (
                        config.tools.whatsapp.webhook_enabled
                        and whatsapp_start_runtime
                        and bridge_mode != "python"
                        and not str(whatsapp_cfg.get("bridge_url", "")).strip()
                    ):
                        channel = self._channel_manager.get_channel("whatsapp")
                        if isinstance(channel, WhatsAppChannel):
                            self._whatsapp_webhook_server = WhatsAppWebhookServer(
                                channel=channel,
                                host=config.tools.whatsapp.webhook_host,
                                port=int(config.tools.whatsapp.webhook_port),
                                webhook_path=config.tools.whatsapp.webhook_path,
                                ingest_loop=loop,
                            )
                            webhook_url = self._whatsapp_webhook_server.start()
                            logger.info(
                                "WhatsApp webhook server enabled at %s", webhook_url
                            )
                            host = (
                                str(config.tools.whatsapp.webhook_host or "")
                                .strip()
                                .lower()
                            )
                            if host in {"127.0.0.1", "localhost", "0.0.0.0"}:
                                logger.warning(
                                    "WhatsApp webhook server is bound to %s. Meta cannot reach localhost directly. Use a public HTTPS URL/tunnel that forwards to %s",
                                    host or "localhost",
                                    webhook_url,
                                )
                        else:
                            logger.warning(
                                "WhatsApp webhook requested but whatsapp channel is not initialized"
                            )
                    elif whatsapp_start_runtime and bridge_mode != "python":
                        logger.info(
                            "WhatsApp webhook server not started (webhook_enabled=%s bridge_url=%s)",
                            bool(config.tools.whatsapp.webhook_enabled),
                            str(whatsapp_cfg.get("bridge_url", "")),
                        )
                    elif whatsapp_enabled and not whatsapp_auto_start:
                        logger.info(
                            "WhatsApp is configured but auto_start=false; skipping WhatsApp startup"
                        )

                    self._channel_start_task = asyncio.create_task(
                        self._channel_manager.start_all()
                    )

                    logger.info(
                        "Annolid Bot background services started (%s)",
                        ", ".join(self._channel_manager.enabled_channels) or "none",
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
        start_task = self._channel_start_task
        self._channel_start_task = None
        if start_task is not None:
            start_task.cancel()
        server = self._whatsapp_webhook_server
        self._whatsapp_webhook_server = None
        if server is not None:
            try:
                server.stop()
            except Exception:
                logger.exception("Failed stopping WhatsApp webhook server")
        bridge = self._whatsapp_python_bridge
        self._whatsapp_python_bridge = None
        if bridge is not None and self._background_loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    bridge.stop(), self._background_loop
                )
                fut.result(timeout=2.0)
            except Exception:
                logger.exception("Failed stopping WhatsApp Python bridge")
        if self._channel_manager is not None and self._background_loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._channel_manager.stop_all(), self._background_loop
                )
                fut.result(timeout=2.0)
            except Exception:
                logger.exception("Failed stopping channel manager")
            self._channel_manager = None
        if self._cron_service is not None and self._background_loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._cron_service.stop(), self._background_loop
                )
                fut.result(timeout=2.0)
            except Exception:
                logger.exception("Failed stopping cron service")
            self._cron_service = None
        if self._bus_service is not None and self._background_loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._bus_service.stop(), self._background_loop
                )
                fut.result(timeout=2.0)
            except Exception:
                logger.exception("Failed stopping bus service")
            self._bus_service = None

        if self._background_loop is not None:
            # Thread-safe way to stop the loop from another thread (main Qt thread)
            self._background_loop.call_soon_threadsafe(self._background_loop.stop)

        if self._background_thread is not None:
            # We use daemon=True, but joining ensures it stops gracefully if possible.
            # Don't block for too long during GUI shutdown.
            self._background_thread.join(timeout=1.0)
            self._background_thread = None
