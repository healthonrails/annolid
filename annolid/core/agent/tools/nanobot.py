from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from .code import CodeExplainTool, CodeSearchTool
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
from .messaging import MessageTool, SpawnTool
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .shell import ExecTool
from .email import EmailTool, ListEmailsTool, ReadEmailTool
from .web import DownloadUrlTool, WebFetchTool, WebSearchTool

if TYPE_CHECKING:
    from annolid.core.agent.config.schema import EmailChannelConfig


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
    registry.register(ExecTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(DownloadUrlTool(allowed_dir=allowed_dir))
    registry.register(DownloadPdfTool(allowed_dir=allowed_dir))
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
    registry.register(MessageTool(send_callback=send_callback))
    registry.register(SpawnTool(spawn_callback=spawn_callback))
    registry.register(CronTool(send_callback=send_callback))

    if email_cfg and email_cfg.enabled:
        registry.register(
            EmailTool(
                smtp_host=email_cfg.smtp_host,
                smtp_port=email_cfg.smtp_port,
                imap_host=email_cfg.imap_host,
                imap_port=email_cfg.imap_port,
                user=email_cfg.user,
                password=email_cfg.password,
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

    if mcp_servers and stack:
        await connect_mcp_servers(mcp_servers, registry, stack)


__all__ = ["register_nanobot_style_tools"]
