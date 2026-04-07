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
from .research import DraftPaperSwarmTool, LiteratureSearchTool
from .function_ffmpeg import VideoFFmpegProcessTool
from .function_video import (
    VideoInfoTool,
    VideoListInferenceModelsTool,
    VideoProcessSegmentsTool,
    VideoRunModelInferenceTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from .function_sam3 import Sam3AgentVideoTrackTool
from .git import (
    GitCliTool,
    GitDiffTool,
    GitHubCliTool,
    GitHubPrChecksTool,
    GitHubPrStatusTool,
    GitLogTool,
    GitStatusTool,
)
from .memory import MemoryGetTool, MemorySearchTool, MemorySetTool
from .messaging import MessageTool, SpawnTool, ListTasksTool, CancelTaskTool
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .sandboxed_shell import SandboxedExecTool
from .shell_sessions import ExecProcessTool, ExecStartTool
from .email import EmailTool, ListEmailsTool, ReadEmailTool
from .calendar import GoogleCalendarTool
from .box import BoxTool
from .workspace import GoogleWorkspaceTool
from .gws_setup import GWSSetupTool
from .camera import CameraSnapshotTool
from .coding_harness import (
    CodingSessionCloseTool,
    CodingSessionListTool,
    CodingSessionPollTool,
    CodingSessionSendTool,
    CodingSessionStartTool,
)
from .automation_scheduler import AutomationSchedulerTool
from .annolid_run import AnnolidRunTool
from .dataset import AnnolidDatasetInspectTool, AnnolidDatasetPrepareTool
from .eval_reporting import AnnolidEvalReportTool
from .eval_start import AnnolidEvalStartTool
from .novelty import AnnolidNoveltyCheckTool
from .paper_reporting import AnnolidPaperRunReportTool
from .training import (
    AnnolidTrainHelpTool,
    AnnolidTrainModelsTool,
    AnnolidTrainStartTool,
)
from .function_admin import (
    AdminEvalRunTool,
    AdminMemoryFlushTool,
    AdminSkillsRefreshTool,
    AdminUpdateRunTool,
)
from .web import DownloadUrlTool, WebFetchTool, WebSearchTool
from .swarm_tool import SwarmTool

if TYPE_CHECKING:
    from annolid.core.agent.config.schema import (
        CalendarToolConfig,
        BoxToolConfig,
        EmailChannelConfig,
        GWSToolConfig,
    )
    from annolid.core.agent.scheduler import TaskScheduler


async def register_nanobot_style_tools(
    registry: FunctionToolRegistry,
    *,
    allowed_dir: Path | None = None,
    allowed_read_roots: Sequence[str | Path] | None = None,
    cron_store_path: Path | None = None,
    send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
    spawn_callback: Callable[[str, str | None], Awaitable[str] | str] | None = None,
    mcp_servers: dict | None = None,
    stack: Any | None = None,
    email_cfg: EmailChannelConfig | None = None,
    calendar_cfg: CalendarToolConfig | None = None,
    box_cfg: BoxToolConfig | None = None,
    gws_cfg: "GWSToolConfig | None" = None,
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
        GitCliTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
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
        GitHubCliTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    registry.register(
        GitHubPrChecksTool(
            allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots
        )
    )
    registry.register(SandboxedExecTool())
    registry.register(ExecStartTool())
    registry.register(ExecProcessTool())
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
    registry.register(
        VideoListInferenceModelsTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        VideoRunModelInferenceTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        Sam3AgentVideoTrackTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        VideoFFmpegProcessTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(CameraSnapshotTool(allowed_dir=allowed_dir))
    registry.register(CodingSessionStartTool(workspace=allowed_dir))
    registry.register(CodingSessionSendTool())
    registry.register(CodingSessionPollTool())
    registry.register(CodingSessionListTool())
    registry.register(CodingSessionCloseTool())
    registry.register(
        AnnolidRunTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidDatasetInspectTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidDatasetPrepareTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidEvalReportTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidEvalStartTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidNoveltyCheckTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidPaperRunReportTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(AnnolidTrainModelsTool())
    registry.register(
        AnnolidTrainHelpTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(
        AnnolidTrainStartTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    registry.register(AutomationSchedulerTool(scheduler=task_scheduler))
    registry.register(AdminSkillsRefreshTool())
    registry.register(AdminMemoryFlushTool())
    registry.register(AdminEvalRunTool())
    registry.register(AdminUpdateRunTool())

    registry.register(LiteratureSearchTool())
    registry.register(DraftPaperSwarmTool())

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
    registry.register(CronTool(store_path=cron_store_path, send_callback=send_callback))

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
            try:
                calendar_available = GoogleCalendarTool.is_available()
            except Exception as exc:  # pragma: no cover - defensive guard
                calendar_available = False
                logger.warning(
                    "Calendar tool availability check failed: %s. "
                    "Continuing without calendar tool.",
                    exc,
                )
            if calendar_available:
                calendar_ready, calendar_reason = GoogleCalendarTool.preflight(
                    credentials_file=calendar_cfg.credentials_file,
                    token_file=calendar_cfg.token_file,
                    allow_interactive_auth=bool(calendar_cfg.allow_interactive_auth),
                )
                if not calendar_ready:
                    logger.warning(
                        "Calendar tool is enabled but not ready: %s "
                        "Set a valid token, add credentials, or enable "
                        "`allow_interactive_auth` for first-run authorization.",
                        calendar_reason,
                    )
                    calendar_available = False
            if calendar_available:
                registry.register(
                    GoogleCalendarTool(
                        credentials_file=calendar_cfg.credentials_file,
                        token_file=calendar_cfg.token_file,
                        allow_interactive_auth=bool(
                            calendar_cfg.allow_interactive_auth
                        ),
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
        elif provider_name == "gws":
            if GoogleWorkspaceTool.is_available():
                registry.register(GoogleWorkspaceTool(allowed_services=["calendar"]))
                logger.info("Calendar tool using gws CLI backend.")
            else:
                logger.warning(
                    "Calendar tool provider is gws but gws is not on PATH. "
                    "Install with: npm install -g @googleworkspace/cli"
                )
        else:
            logger.warning(
                "Calendar tool provider %r is not supported. Supported providers: google, gws",
                provider_name,
            )

    if box_cfg and box_cfg.enabled:
        registry.register(
            BoxTool(
                access_token=box_cfg.access_token,
                client_id=box_cfg.client_id,
                client_secret=box_cfg.client_secret,
                refresh_token=box_cfg.refresh_token,
                token_url=box_cfg.token_url,
                api_base=box_cfg.api_base,
                upload_api_base=box_cfg.upload_api_base,
                enterprise_id=box_cfg.enterprise_id,
                auto_refresh=bool(box_cfg.auto_refresh),
                allowed_dir=allowed_dir,
                allowed_read_roots=allowed_read_roots,
            )
        )

    # -- Google Workspace CLI tools --
    if gws_cfg and gws_cfg.enabled:
        if GoogleWorkspaceTool.is_available():
            registry.register(
                GoogleWorkspaceTool(
                    allowed_services=gws_cfg.services or None,
                )
            )
            registry.register(GWSSetupTool())
            logger.info("Google Workspace CLI tools registered.")
        else:
            if gws_cfg.auto_install:
                logger.info(
                    "gws CLI not found but auto_install is enabled. "
                    "The gws_setup tool can install it at runtime."
                )
                registry.register(GWSSetupTool())
            else:
                logger.warning(
                    "Google Workspace CLI tool is enabled but gws is not on PATH. "
                    "Install with: npm install -g @googleworkspace/cli"
                )

    if mcp_servers and stack:
        await connect_mcp_servers(mcp_servers, registry, stack)


__all__ = ["register_nanobot_style_tools"]
