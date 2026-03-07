"""Service wrappers for chat agent runtime, tools, and provider state."""

from __future__ import annotations

from annolid.core.agent.bus import InboundMessage as BusInboundMessage
from annolid.core.agent.memory import AgentMemoryStore
from annolid.core.agent.providers import (
    ollama_mark_plain_mode,
    ollama_plain_mode_decrement,
    ollama_plain_mode_remaining,
    recover_with_plain_ollama_reply,
)
from annolid.core.agent.providers.background_chat import (
    OLLAMA_PLAIN_MODE_COOLDOWN_TURNS as PROVIDER_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS,
)
from annolid.core.agent.providers.background_chat import (
    _OLLAMA_FORCE_PLAIN_CACHE as PROVIDER_OLLAMA_FORCE_PLAIN_CACHE,
)
from annolid.core.agent.providers.background_chat import (
    _OLLAMA_TOOL_SUPPORT_CACHE as PROVIDER_OLLAMA_TOOL_SUPPORT_CACHE,
)
from annolid.core.agent.tools import FunctionToolRegistry
from annolid.core.agent.tools.automation_scheduler import AutomationSchedulerTool
from annolid.core.agent.tools.camera import (
    _annotate_snapshot_frame,
    build_camera_mission_status,
)
from annolid.core.agent.tools.clawhub import (
    clawhub_install_skill,
    clawhub_search_skills,
)
from annolid.core.agent.tools.cron import CronTool
from annolid.core.agent.tools.email import EmailTool
from annolid.core.agent.tools.filesystem import RenameFileTool
from annolid.core.agent.tools.pdf import DownloadPdfTool
from annolid.core.agent.tools.policy import resolve_allowed_tools

__all__ = [
    "AgentMemoryStore",
    "AutomationSchedulerTool",
    "BusInboundMessage",
    "CronTool",
    "DownloadPdfTool",
    "EmailTool",
    "FunctionToolRegistry",
    "PROVIDER_OLLAMA_FORCE_PLAIN_CACHE",
    "PROVIDER_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS",
    "PROVIDER_OLLAMA_TOOL_SUPPORT_CACHE",
    "RenameFileTool",
    "_annotate_snapshot_frame",
    "build_camera_mission_status",
    "clawhub_install_skill",
    "clawhub_search_skills",
    "ollama_mark_plain_mode",
    "ollama_plain_mode_decrement",
    "ollama_plain_mode_remaining",
    "recover_with_plain_ollama_reply",
    "resolve_allowed_tools",
]
