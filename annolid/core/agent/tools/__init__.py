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
from .function_base import FunctionTool
from .function_builtin import (
    CronTool,
    EditFileTool,
    ExecTool,
    ListDirTool,
    MemoryGetTool,
    MemorySetTool,
    MemorySearchTool,
    MessageTool,
    ReadFileTool,
    SpawnTool,
    WebFetchTool,
    WebSearchTool,
    WriteFileTool,
    register_nanobot_style_tools,
)
from .function_gui import (
    GuiContextTool,
    GuiOpenVideoTool,
    GuiSelectAnnotationModelTool,
    GuiSendPromptTool,
    GuiSetChatModelTool,
    GuiSetFrameTool,
    GuiSetPromptTool,
    GuiSharedImagePathTool,
    GuiTrackNextFramesTool,
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

__all__ = [
    "ArtifactStore",
    "CancellationToken",
    "CaptionResult",
    "CaptionTool",
    "DetectionResult",
    "DetectionTool",
    "EmbeddingResult",
    "EmbeddingTool",
    "FunctionTool",
    "FunctionToolRegistry",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    "MemorySearchTool",
    "MemoryGetTool",
    "MemorySetTool",
    "ExecTool",
    "WebSearchTool",
    "WebFetchTool",
    "VideoInfoTool",
    "VideoSampleFramesTool",
    "VideoSegmentTool",
    "VideoProcessSegmentsTool",
    "ResolvedToolPolicy",
    "resolve_allowed_tools",
    "GuiContextTool",
    "GuiOpenVideoTool",
    "GuiSetFrameTool",
    "GuiSetPromptTool",
    "GuiSendPromptTool",
    "GuiSetChatModelTool",
    "GuiSelectAnnotationModelTool",
    "GuiTrackNextFramesTool",
    "GuiSharedImagePathTool",
    "MessageTool",
    "SpawnTool",
    "CronTool",
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
]
