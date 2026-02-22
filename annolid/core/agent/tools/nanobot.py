from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from annolid.utils.logger import logger

from .citation import (
    BibtexListEntriesTool,
    BibtexRemoveEntryTool,
    BibtexUpsertEntryTool,
)
from .code import CodeExplainTool, CodeSearchTool
from .clawhub import ClawHubInstallSkillTool, ClawHubSearchSkillsTool
from .mcp import connect_mcp_servers
from .cron import CronTool
from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    RenameFileTool,
    WriteFileTool,
)
from .function_registry import FunctionToolRegistry
from .function_video import (
    VideoInfoTool,
    VideoProcessSegmentsTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from .git import (
    GitDiffTool,
    GitHubPrChecksTool,
    GitHubPrStatusTool,
    GitLogTool,
    GitStatusTool,
)
from .memory import MemoryGetTool, MemorySearchTool, MemorySetTool
from .messaging import MessageTool, SpawnTool, ListTasksTool, CancelTaskTool
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .sandboxed_shell import SandboxedExecTool
from .email import EmailTool, ListEmailsTool, ReadEmailTool
from .calendar import GoogleCalendarTool
from .camera import CameraSnapshotTool
from .automation_scheduler import AutomationSchedulerTool
from .web import DownloadUrlTool, WebFetchTool, WebSearchTool
from .swarm_tool import SwarmTool

if TYPE_CHECKING:
    from annolid.core.agent.config.schema import CalendarToolConfig, EmailChannelConfig
    from annolid.core.agent.scheduler import TaskScheduler


async def register_nanobot_style_tools(
    registry: FunctionToolRegistry,
    *,
    allowed_dir: Path | None = None,
    allowed_read_roots: Sequence[str | Path] | None = None,
    send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
    spawn_callback: Callable[[str, str | None], Awaitable[str] | str] | None = None,
    mcp_servers: dict | None = None,
    stack: Any | None = None,
    email_cfg: EmailChannelConfig | None = None,
    calendar_cfg: CalendarToolConfig | None = None,
    task_scheduler: "TaskScheduler | None" = None,
    ignored_tools: Sequence[str] = (),
) -> None:
    """Register a Nanobot-like default tool set."""

    registry.register(
        ReadFileTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        ExtractPdfTextTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        OpenPdfTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        ExtractPdfImagesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(WriteFileTool(allowed_dir=allowed_dir))
    registry.register(EditFileTool(allowed_dir=allowed_dir))
    registry.register(RenameFileTool(allowed_dir=allowed_dir))
    registry.register(
        ListDirTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        CodeSearchTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        CodeExplainTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(MemorySearchTool(workspace=allowed_dir))
    registry.register(MemoryGetTool(workspace=allowed_dir))
    registry.register(MemorySetTool(workspace=allowed_dir))
    registry.register(
        GitStatusTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        GitDiffTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        GitLogTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        GitHubPrStatusTool(
            allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots
        )
    )
    registry.register(
        GitHubPrChecksTool(
            allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots
        )
    )
    registry.register(SandboxedExecTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(DownloadUrlTool(allowed_dir=allowed_dir))
    registry.register(DownloadPdfTool(allowed_dir=allowed_dir))
    registry.register(
        BibtexListEntriesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(BibtexUpsertEntryTool(allowed_dir=allowed_dir))
    registry.register(BibtexRemoveEntryTool(allowed_dir=allowed_dir))
    registry.register(ClawHubSearchSkillsTool(workspace=allowed_dir))
    registry.register(ClawHubInstallSkillTool(workspace=allowed_dir))
    registry.register(
        VideoInfoTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        VideoSampleFramesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        VideoSegmentTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        VideoProcessSegmentsTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(CameraSnapshotTool(allowed_dir=allowed_dir))
    registry.register(AutomationSchedulerTool(scheduler=task_scheduler))
    if "message" not in ignored_tools:
        registry.register(MessageTool(send_callback=send_callback))
    if "spawn" not in ignored_tools:
        registry.register(SpawnTool(spawn_callback=spawn_callback))
    if "list_tasks" not in ignored_tools:
        registry.register(ListTasksTool())
    if "cancel_task" not in ignored_tools:
        registry.register(CancelTaskTool())
    if "run_swarm" not in ignored_tools:
        registry.register(SwarmTool())
    registry.register(CronTool(send_callback=send_callback))

    if email_cfg and email_cfg.enabled:
        attachment_roots: list[str | Path] = []
        if allowed_dir is not None:
            attachment_roots.append(allowed_dir)
        for root in allowed_read_roots or []:
            if str(root).strip():
                attachment_roots.append(root)
        registry.register(
            EmailTool(
                smtp_host=email_cfg.smtp_host,
                smtp_port=email_cfg.smtp_port,
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
                allowed_attachment_roots=attachment_roots,
            )
        )
        registry.register(
            ListEmailsTool(
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
            )
        )
        registry.register(
            ReadEmailTool(
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
            )
        )

    if calendar_cfg and calendar_cfg.enabled:
        provider_name = str(calendar_cfg.provider or "google").strip().lower()
        if provider_name == "google":
            if GoogleCalendarTool.is_available():
                registry.register(
                    GoogleCalendarTool(
                        credentials_file=calendar_cfg.credentials_file,
                        token_file=calendar_cfg.token_file,
                        calendar_id=calendar_cfg.calendar_id,
                        timezone_name=calendar_cfg.timezone,
                        default_event_duration_minutes=calendar_cfg.default_event_duration_minutes,
                    )
                )
            else:
                logger.warning(
                    "Calendar tool is enabled but Google dependencies are missing. "
                    'Install optional extras with `pip install "annolid[google_calendar]"`.'
                )
        else:
            logger.warning(
                "Calendar tool provider %r is not supported. Supported providers: google",
                provider_name,
            )

    if mcp_servers and stack:
        await connect_mcp_servers(mcp_servers, registry, stack)


__all__ = ["register_nanobot_style_tools"]
