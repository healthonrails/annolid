"""Service wrappers for chat manager background runtime primitives."""

from __future__ import annotations

from annolid.core.agent.bus.service import AgentBusService
from annolid.core.agent.channels.manager import ChannelManager
from annolid.core.agent.channels.whatsapp import WhatsAppChannel
from annolid.core.agent.channels.whatsapp_python_bridge import WhatsAppPythonBridge
from annolid.core.agent.channels.whatsapp_webhook_server import WhatsAppWebhookServer
from annolid.core.agent.cron import (
    CronJob,
    CronPayload,
    CronSchedule,
    CronService,
    default_cron_store_path,
)
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.scheduler import ScheduledTask, TaskScheduler
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.tools import FunctionToolRegistry, register_nanobot_style_tools
from annolid.core.agent.tools.policy import (
    build_tool_permission_context,
    resolve_allowed_tools,
)

__all__ = [
    "AgentBusService",
    "AgentLoop",
    "AgentSessionManager",
    "ChannelManager",
    "CronJob",
    "CronPayload",
    "CronSchedule",
    "CronService",
    "FunctionToolRegistry",
    "PersistentSessionStore",
    "ScheduledTask",
    "TaskScheduler",
    "WhatsAppChannel",
    "WhatsAppPythonBridge",
    "WhatsAppWebhookServer",
    "build_tool_permission_context",
    "default_cron_store_path",
    "register_nanobot_style_tools",
    "resolve_allowed_tools",
]
