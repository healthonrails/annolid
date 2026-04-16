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
from .function_admin import (
    AdminEvalRunTool,
    AdminMemoryFlushTool,
    AdminSkillsRefreshTool,
    AdminUpdateRunTool,
)
from .citation import (
    BibtexListEntriesTool,
    BibtexRemoveEntryTool,
    BibtexUpsertEntryTool,
)
from .cron import CronTool
from .automation_scheduler import AutomationSchedulerTool
from .annolid_run import AnnolidRunTool
from .dataset import AnnolidDatasetInspectTool, AnnolidDatasetPrepareTool
from .eval_reporting import (
    AnnolidEvalReportTool,
    build_model_eval_report,
    write_model_eval_report_files,
)
from .eval_start import AnnolidEvalStartTool
from .novelty import AnnolidNoveltyCheckTool
from .paper_reporting import AnnolidPaperRunReportTool
from .training import (
    AnnolidTrainHelpTool,
    AnnolidTrainModelsTool,
    AnnolidTrainStartTool,
)
from .camera import CameraSnapshotTool
from .box import BoxTool
from .coding_harness import (
    CodingSessionCloseTool,
    CodingSessionListTool,
    CodingSessionPollTool,
    CodingSessionSendTool,
    CodingSessionStartTool,
)
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
from .messaging import MessageTool, SpawnTool, ListTasksTool, CancelTaskTool
from .swarm_tool import SwarmTool
from .nanobot import register_nanobot_style_tools
from .pdf import DownloadPdfTool, ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool
from .shell import ExecTool
from .sandboxed_shell import SandboxedExecTool
from .shell_sessions import ExecProcessTool, ExecStartTool
from .web import DownloadUrlTool, WebFetchTool, WebSearchTool
from .function_gui_core import (
    GuiContextTool,
    GuiSelfUpdateTool,
    GuiSaveCitationTool,
    GuiVerifyCitationsTool,
    GuiSelectAnnotationModelTool,
    GuiSendPromptTool,
    GuiSetChatModelTool,
    GuiSetPromptTool,
    GuiSharedImagePathTool,
)
from .function_gui_web import (
    GuiOpenInBrowserTool,
    GuiOpenUrlTool,
    GuiWebCaptureScreenshotTool,
    GuiWebClickTool,
    GuiWebDescribeViewTool,
    GuiWebExtractStructuredTool,
    GuiWebFindFormsTool,
    GuiWebGetDomTextTool,
    GuiWebRunStepsTool,
    GuiWebScrollTool,
    GuiWebTypeTool,
)
from .function_gui_threejs import (
    GuiOpenThreeJsExampleTool,
    GuiOpenThreeJsTool,
)
from .function_gui_video import (
    GuiAnalyzeTrackingStatsTool,
    GuiCheckStreamSourceTool,
    GuiGetRealtimeStatusTool,
    GuiLabelBehaviorSegmentsTool,
    GuiBehaviorCatalogTool,
    GuiListRealtimeLogsTool,
    GuiListRealtimeModelsTool,
    GuiOpenVideoTool,
    GuiProcessVideoBehaviorsTool,
    GuiRunAiTextSegmentationTool,
    GuiSegmentTrackVideoTool,
    GuiSetAiTextPromptTool,
    GuiSetFrameTool,
    GuiStartRealtimeStreamTool,
    GuiStopRealtimeStreamTool,
    GuiTrackNextFramesTool,
)
from .function_gui_pdf import GuiOpenPdfTool
from .function_gui_pdf import (
    GuiArxivSearchTool,
    GuiListPdfsTool,
    GuiPdfFindSectionsTool,
    GuiPdfGetStateTool,
    GuiPdfGetTextTool,
    GuiPdfSummarizeTool,
)
from .function_gui_registry import register_annolid_gui_tools
from .function_video import (
    VideoInfoTool,
    VideoListInferenceModelsTool,
    VideoProcessSegmentsTool,
    VideoRunModelInferenceTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from .function_sam3 import Sam3AgentVideoTrackTool
from .policy import (
    ResolvedToolPolicy,
    ToolPermissionContext,
    build_tool_permission_context,
    resolve_allowed_tools,
)
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
    McpBrowserTool,
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
    "AdminSkillsRefreshTool",
    "AdminMemoryFlushTool",
    "AdminEvalRunTool",
    "AdminUpdateRunTool",
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
    "SandboxedExecTool",
    "ExecStartTool",
    "ExecProcessTool",
    "AnnolidDatasetInspectTool",
    "AnnolidDatasetPrepareTool",
    "AnnolidEvalReportTool",
    "AnnolidEvalStartTool",
    "AnnolidNoveltyCheckTool",
    "AnnolidPaperRunReportTool",
    "build_model_eval_report",
    "write_model_eval_report_files",
    "AnnolidTrainModelsTool",
    "AnnolidTrainHelpTool",
    "AnnolidTrainStartTool",
    "ClawHubSearchSkillsTool",
    "ClawHubInstallSkillTool",
    "clawhub_search_skills",
    "clawhub_install_skill",
    "run_clawhub_command",
    "WebSearchTool",
    "WebFetchTool",
    "DownloadUrlTool",
    "DownloadPdfTool",
    "BibtexListEntriesTool",
    "BibtexUpsertEntryTool",
    "BibtexRemoveEntryTool",
    "ExtractPdfTextTool",
    "OpenPdfTool",
    "ExtractPdfImagesTool",
    "VideoInfoTool",
    "VideoListInferenceModelsTool",
    "VideoSampleFramesTool",
    "VideoSegmentTool",
    "VideoProcessSegmentsTool",
    "VideoRunModelInferenceTool",
    "Sam3AgentVideoTrackTool",
    "ResolvedToolPolicy",
    "ToolPermissionContext",
    "build_tool_permission_context",
    "resolve_allowed_tools",
    "GuiContextTool",
    "GuiSelfUpdateTool",
    "GuiOpenUrlTool",
    "GuiOpenInBrowserTool",
    "GuiOpenThreeJsTool",
    "GuiOpenThreeJsExampleTool",
    "GuiWebGetDomTextTool",
    "GuiWebCaptureScreenshotTool",
    "GuiWebDescribeViewTool",
    "GuiWebExtractStructuredTool",
    "GuiWebClickTool",
    "GuiWebTypeTool",
    "GuiWebScrollTool",
    "GuiWebFindFormsTool",
    "GuiWebRunStepsTool",
    "GuiOpenPdfTool",
    "GuiPdfGetStateTool",
    "GuiPdfGetTextTool",
    "GuiPdfSummarizeTool",
    "GuiPdfFindSectionsTool",
    "GuiArxivSearchTool",
    "GuiListPdfsTool",
    "GuiOpenVideoTool",
    "GuiSetFrameTool",
    "GuiSetPromptTool",
    "GuiSendPromptTool",
    "GuiSetChatModelTool",
    "GuiSelectAnnotationModelTool",
    "GuiTrackNextFramesTool",
    "GuiSetAiTextPromptTool",
    "GuiRunAiTextSegmentationTool",
    "GuiAnalyzeTrackingStatsTool",
    "GuiSaveCitationTool",
    "GuiVerifyCitationsTool",
    "GuiSegmentTrackVideoTool",
    "GuiLabelBehaviorSegmentsTool",
    "GuiProcessVideoBehaviorsTool",
    "GuiBehaviorCatalogTool",
    "GuiStartRealtimeStreamTool",
    "GuiStopRealtimeStreamTool",
    "GuiGetRealtimeStatusTool",
    "GuiListRealtimeModelsTool",
    "GuiListRealtimeLogsTool",
    "GuiCheckStreamSourceTool",
    "GuiSharedImagePathTool",
    "MessageTool",
    "SpawnTool",
    "ListTasksTool",
    "CancelTaskTool",
    "SwarmTool",
    "CronTool",
    "AutomationSchedulerTool",
    "AnnolidRunTool",
    "CameraSnapshotTool",
    "CodingSessionStartTool",
    "CodingSessionSendTool",
    "CodingSessionPollTool",
    "CodingSessionListTool",
    "CodingSessionCloseTool",
    "BoxTool",
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
    "McpBrowserTool",
    "McpBrowserClickTool",
    "McpBrowserTypeTool",
    "McpBrowserSnapshotTool",
    "McpBrowserScreenshotTool",
    "McpBrowserScrollTool",
    "McpBrowserCloseTool",
    "McpBrowserWaitTool",
    "register_mcp_browser_tools",
]
