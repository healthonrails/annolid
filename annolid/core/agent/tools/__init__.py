"""Pluggable agent tools.

This package defines the core tool contracts (context + IO containers) that
Phase 4 orchestration will compose into pipelines.
"""

from .base import (
    ArtifactStore,
    CancellationToken,
    FrameBatch,
    FrameData,
    Instance,
    Instances,
    Tool,
    ToolContext,
    ToolError,
)
from .artifacts import FileArtifactStore, content_hash
from .detection import DetectionResult, DetectionTool
from .embedding import EmbeddingResult, EmbeddingTool
from .email import EmailTool
from .function_base import FunctionTool
from .cron import CronTool
from .calendar import GoogleCalendarTool
from .clawhub import (
    ClawHubInstallSkillTool,
    ClawHubSearchSkillsTool,
    clawhub_install_skill,
    clawhub_search_skills,
    run_clawhub_command,
)
from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    RenameFileTool,
    WriteFileTool,
)
from .memory import MemoryGetTool, MemorySetTool, MemorySearchTool
from .messaging import MessageTool, SpawnTool
from .nanobot import register_nanobot_style_tools
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .shell import ExecTool
from .web import DownloadUrlTool, WebFetchTool, WebSearchTool
from .function_gui import (
    GuiContextTool,
    GuiLabelBehaviorSegmentsTool,
    GuiOpenInBrowserTool,
    GuiOpenPdfTool,
    GuiOpenUrlTool,
    GuiOpenVideoTool,
    GuiRunAiTextSegmentationTool,
    GuiSegmentTrackVideoTool,
    GuiSelectAnnotationModelTool,
    GuiSendPromptTool,
    GuiStartRealtimeStreamTool,
    GuiStopRealtimeStreamTool,
    GuiSetAiTextPromptTool,
    GuiSetChatModelTool,
    GuiSetFrameTool,
    GuiSetPromptTool,
    GuiSharedImagePathTool,
    GuiTrackNextFramesTool,
    GuiWebClickTool,
    GuiWebFindFormsTool,
    GuiWebGetDomTextTool,
    GuiWebRunStepsTool,
    GuiWebScrollTool,
    GuiWebTypeTool,
    register_annolid_gui_tools,
)
from .function_video import (
    VideoInfoTool,
    VideoProcessSegmentsTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from .policy import ResolvedToolPolicy, resolve_allowed_tools
from .function_registry import FunctionToolRegistry
from .llm import CaptionResult, CaptionTool
from .sampling import FPSampler, MotionSampler, RandomSampler, UniformSampler
from .registry import ToolRegistry, build_pipeline
from .tracking import SimpleTrackTool, TrackingResult
from .utility import (
    CalculatorResult,
    CalculatorTool,
    DateTimeResult,
    DateTimeTool,
    TextStatsResult,
    TextStatsTool,
    register_builtin_utility_tools,
)
from .vector_index import NumpyEmbeddingIndex, SearchResult
from .mcp_browser import (
    McpBrowserClickTool,
    McpBrowserCloseTool,
    McpBrowserNavigateTool,
    McpBrowserScreenshotTool,
    McpBrowserScrollTool,
    McpBrowserSnapshotTool,
    McpBrowserTypeTool,
    McpBrowserWaitTool,
    register_mcp_browser_tools,
)

__all__ = [
    "ArtifactStore",
    "CancellationToken",
    "CaptionResult",
    "CaptionTool",
    "DetectionResult",
    "DetectionTool",
    "EmbeddingResult",
    "EmbeddingTool",
    "EmailTool",
    "FunctionTool",
    "FunctionToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "RenameFileTool",
    "ListDirTool",
    "MemorySearchTool",
    "MemoryGetTool",
    "MemorySetTool",
    "ExecTool",
    "ClawHubSearchSkillsTool",
    "ClawHubInstallSkillTool",
    "clawhub_search_skills",
    "clawhub_install_skill",
    "run_clawhub_command",
    "WebSearchTool",
    "WebFetchTool",
    "DownloadUrlTool",
    "DownloadPdfTool",
    "ExtractPdfTextTool",
    "OpenPdfTool",
    "ExtractPdfImagesTool",
    "VideoInfoTool",
    "VideoSampleFramesTool",
    "VideoSegmentTool",
    "VideoProcessSegmentsTool",
    "ResolvedToolPolicy",
    "resolve_allowed_tools",
    "GuiContextTool",
    "GuiOpenUrlTool",
    "GuiOpenInBrowserTool",
    "GuiWebGetDomTextTool",
    "GuiWebClickTool",
    "GuiWebTypeTool",
    "GuiWebScrollTool",
    "GuiWebFindFormsTool",
    "GuiWebRunStepsTool",
    "GuiOpenPdfTool",
    "GuiOpenVideoTool",
    "GuiSetFrameTool",
    "GuiSetPromptTool",
    "GuiSendPromptTool",
    "GuiSetChatModelTool",
    "GuiSelectAnnotationModelTool",
    "GuiTrackNextFramesTool",
    "GuiSetAiTextPromptTool",
    "GuiRunAiTextSegmentationTool",
    "GuiSegmentTrackVideoTool",
    "GuiLabelBehaviorSegmentsTool",
    "GuiStartRealtimeStreamTool",
    "GuiStopRealtimeStreamTool",
    "GuiSharedImagePathTool",
    "MessageTool",
    "SpawnTool",
    "CronTool",
    "GoogleCalendarTool",
    "register_nanobot_style_tools",
    "register_annolid_gui_tools",
    "FileArtifactStore",
    "FrameBatch",
    "FrameData",
    "Instance",
    "Instances",
    "FPSampler",
    "MotionSampler",
    "RandomSampler",
    "UniformSampler",
    "SimpleTrackTool",
    "TrackingResult",
    "CalculatorResult",
    "CalculatorTool",
    "DateTimeResult",
    "DateTimeTool",
    "TextStatsResult",
    "TextStatsTool",
    "register_builtin_utility_tools",
    "NumpyEmbeddingIndex",
    "SearchResult",
    "ToolRegistry",
    "build_pipeline",
    "Tool",
    "ToolContext",
    "ToolError",
    "content_hash",
    "McpBrowserNavigateTool",
    "McpBrowserClickTool",
    "McpBrowserTypeTool",
    "McpBrowserSnapshotTool",
    "McpBrowserScreenshotTool",
    "McpBrowserScrollTool",
    "McpBrowserCloseTool",
    "McpBrowserWaitTool",
    "register_mcp_browser_tools",
]
