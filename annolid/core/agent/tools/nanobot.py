from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from annolid.utils.logger import logger

from .citation import (
    BibtexListEntriesTool,
    BibtexRemoveEntryTool,
    BibtexUpsertEntryTool,
)
from .code import CodeExplainTool, CodeSearchTool
from .clawhub import ClawHubInstallSkillTool, ClawHubSearchSkillsTool
from .cron import CronTool
from .function_base import FunctionTool
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
    VideoSegmentFrameGridTool,
)
from .mcp import connect_mcp_servers
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
from .messaging import (
    CancelTaskTool,
    ListTasksTool,
    MessageTool,
    SpawnBehaviorSubagentTool,
    SpawnTool,
)
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .sandboxed_shell import SandboxedExecTool
from .shell_sessions import ExecProcessTool, ExecStartTool
from .email import EmailTool, ListEmailsTool, ReadEmailTool
from .calendar import GoogleCalendarTool
from .google_drive import GoogleDriveTool
from .box import BoxTool
from .camera import CameraSnapshotTool
from .coding_harness import (
    CodingSessionAbortTool,
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
        GoogleAuthConfig,
    )
    from annolid.core.agent.scheduler import TaskScheduler


_INFO_LOG_ONCE_KEYS: set[str] = set()
_INFO_LOG_ONCE_LOCK = Lock()


def _log_info_once(key: str, message: str) -> None:
    with _INFO_LOG_ONCE_LOCK:
        if key in _INFO_LOG_ONCE_KEYS:
            return
        _INFO_LOG_ONCE_KEYS.add(key)
    logger.info(message)


def _normalize_ignored_tool_names(ignored_tools: Sequence[str]) -> set[str]:
    return {str(name).strip().lower() for name in ignored_tools if str(name).strip()}


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
    google_auth_cfg: "GoogleAuthConfig | None" = None,
    google_drive_enabled: bool = False,
    box_cfg: BoxToolConfig | None = None,
    task_scheduler: "TaskScheduler | None" = None,
    ignored_tools: Sequence[str] = (),
) -> None:
    """Register a Nanobot-like default tool set.

    ``ignored_tools`` contains exact tool names to omit from this registration pass.
    """

    ignored_tool_names = _normalize_ignored_tool_names(ignored_tools)

    def should_register_tool(name: str) -> bool:
        return str(name).strip().lower() not in ignored_tool_names

    def register_tool(tool: FunctionTool) -> bool:
        if not should_register_tool(tool.name):
            logger.debug("Skipping ignored Annolid Bot tool: %s", tool.name)
            return False
        registry.register(tool)
        return True

    register_tool(
        ReadFileTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        ExtractPdfTextTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        OpenPdfTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        ExtractPdfImagesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(WriteFileTool(allowed_dir=allowed_dir))
    register_tool(EditFileTool(allowed_dir=allowed_dir))
    register_tool(RenameFileTool(allowed_dir=allowed_dir))
    register_tool(
        ListDirTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        CodeSearchTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        CodeExplainTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(MemorySearchTool(workspace=allowed_dir))
    register_tool(MemoryGetTool(workspace=allowed_dir))
    register_tool(MemorySetTool(workspace=allowed_dir))
    register_tool(
        GitStatusTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        GitCliTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        GitDiffTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        GitLogTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        GitHubPrStatusTool(
            allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots
        )
    )
    register_tool(
        GitHubCliTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        GitHubPrChecksTool(
            allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots
        )
    )
    register_tool(SandboxedExecTool())
    register_tool(ExecStartTool())
    register_tool(ExecProcessTool())
    register_tool(WebSearchTool())
    register_tool(WebFetchTool())
    register_tool(DownloadUrlTool(allowed_dir=allowed_dir))
    register_tool(DownloadPdfTool(allowed_dir=allowed_dir))
    register_tool(
        BibtexListEntriesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(BibtexUpsertEntryTool(allowed_dir=allowed_dir))
    register_tool(BibtexRemoveEntryTool(allowed_dir=allowed_dir))
    register_tool(ClawHubSearchSkillsTool(workspace=allowed_dir))
    register_tool(ClawHubInstallSkillTool(workspace=allowed_dir))
    register_tool(
        VideoInfoTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        VideoSampleFramesTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        VideoSegmentTool(allowed_dir=allowed_dir, allowed_read_roots=allowed_read_roots)
    )
    register_tool(
        VideoProcessSegmentsTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        VideoSegmentFrameGridTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        VideoListInferenceModelsTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        VideoRunModelInferenceTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        Sam3AgentVideoTrackTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        VideoFFmpegProcessTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(CameraSnapshotTool(allowed_dir=allowed_dir))
    register_tool(CodingSessionStartTool(workspace=allowed_dir))
    register_tool(CodingSessionSendTool())
    register_tool(CodingSessionPollTool())
    register_tool(CodingSessionListTool())
    register_tool(CodingSessionAbortTool())
    register_tool(CodingSessionCloseTool())
    register_tool(
        AnnolidRunTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidDatasetInspectTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidDatasetPrepareTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidEvalReportTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidEvalStartTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidNoveltyCheckTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidPaperRunReportTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(AnnolidTrainModelsTool())
    register_tool(
        AnnolidTrainHelpTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(
        AnnolidTrainStartTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    )
    register_tool(AutomationSchedulerTool(scheduler=task_scheduler))
    register_tool(AdminSkillsRefreshTool())
    register_tool(AdminMemoryFlushTool())
    register_tool(AdminEvalRunTool())
    register_tool(AdminUpdateRunTool())

    register_tool(LiteratureSearchTool())
    register_tool(DraftPaperSwarmTool())

    register_tool(MessageTool(send_callback=send_callback))
    register_tool(SpawnTool(spawn_callback=spawn_callback))
    register_tool(SpawnBehaviorSubagentTool(spawn_callback=spawn_callback))
    register_tool(ListTasksTool())
    register_tool(CancelTaskTool())
    register_tool(SwarmTool())
    register_tool(CronTool(store_path=cron_store_path, send_callback=send_callback))

    email_tools_enabled = any(
        should_register_tool(name) for name in ("email", "list_emails", "read_email")
    )
    if email_cfg and email_cfg.enabled and email_tools_enabled:
        attachment_roots: list[str | Path] = []
        if allowed_dir is not None:
            attachment_roots.append(allowed_dir)
        for root in allowed_read_roots or []:
            if str(root).strip():
                attachment_roots.append(root)
        register_tool(
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
        register_tool(
            ListEmailsTool(
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
            )
        )
        register_tool(
            ReadEmailTool(
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
            )
        )

    if (
        calendar_cfg
        and calendar_cfg.enabled
        and should_register_tool("google_calendar")
    ):
        credentials_file = (
            str(getattr(google_auth_cfg, "credentials_file", "") or "").strip()
            or calendar_cfg.credentials_file
        )
        token_file = (
            str(getattr(google_auth_cfg, "token_file", "") or "").strip()
            or calendar_cfg.token_file
        )
        allow_interactive_auth = bool(
            getattr(google_auth_cfg, "allow_interactive_auth", False)
            or calendar_cfg.allow_interactive_auth
        )
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
                credentials_file=credentials_file,
                token_file=token_file,
                allow_interactive_auth=allow_interactive_auth,
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
            register_tool(
                GoogleCalendarTool(
                    credentials_file=credentials_file,
                    token_file=token_file,
                    allow_interactive_auth=allow_interactive_auth,
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

    if box_cfg and box_cfg.enabled and should_register_tool("box"):
        register_tool(
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

    if getattr(calendar_cfg, "enabled", False) and registry.has("google_calendar"):
        _log_info_once(
            "google_calendar_registered_oauth",
            "Google Calendar tool registered with Google OAuth backend.",
        )

    if google_drive_enabled and should_register_tool("google_drive"):
        credentials_file = str(
            getattr(google_auth_cfg, "credentials_file", "")
            or "~/.annolid/agent/google_oauth_credentials.json"
        ).strip()
        token_file = str(
            getattr(google_auth_cfg, "token_file", "")
            or "~/.annolid/agent/google_oauth_token.json"
        ).strip()
        allow_interactive_auth = bool(
            getattr(google_auth_cfg, "allow_interactive_auth", False)
        )
        try:
            drive_available = GoogleDriveTool.is_available()
        except Exception as exc:  # pragma: no cover
            drive_available = False
            logger.warning("Google Drive availability check failed: %s", exc)
        if drive_available:
            drive_ready, drive_reason = GoogleDriveTool.preflight(
                credentials_file=credentials_file,
                token_file=token_file,
                allow_interactive_auth=allow_interactive_auth,
            )
            if not drive_ready:
                logger.warning(
                    "Google Drive tool is enabled but not ready: %s", drive_reason
                )
                drive_available = False
        if drive_available:
            upload_roots: list[str | Path] = []
            if allowed_dir is not None:
                upload_roots.append(allowed_dir)
            for root in allowed_read_roots or []:
                if str(root).strip():
                    upload_roots.append(root)
            register_tool(
                GoogleDriveTool(
                    credentials_file=credentials_file,
                    token_file=token_file,
                    allow_interactive_auth=allow_interactive_auth,
                    allowed_local_roots=upload_roots,
                )
            )
        else:
            logger.warning(
                "Google Drive tool is enabled but unavailable. "
                'Install optional extras with `pip install "annolid[google_calendar]"` '
                "and configure shared Google OAuth files."
            )

    if mcp_servers and stack:
        before_mcp_tool_names = set(registry.tool_names)
        await connect_mcp_servers(mcp_servers, registry, stack)
        for tool_name in set(registry.tool_names) - before_mcp_tool_names:
            if not should_register_tool(tool_name):
                logger.debug("Unregistering ignored MCP tool: %s", tool_name)
                registry.unregister(tool_name)


__all__ = ["register_nanobot_style_tools"]
