from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiOpenVideoTool(FunctionTool):
    def __init__(self, open_video_callback: Optional[ActionCallback] = None):
        self._open_video_callback = open_video_callback

    @property
    def name(self) -> str:
        return "gui_open_video"

    @property
    def description(self) -> str:
        return (
            "Open a video in Annolid GUI using an absolute or workspace-relative path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_video_callback, **kwargs)


class GuiSetFrameTool(FunctionTool):
    def __init__(self, set_frame_callback: Optional[ActionCallback] = None):
        self._set_frame_callback = set_frame_callback

    @property
    def name(self) -> str:
        return "gui_set_frame"

    @property
    def description(self) -> str:
        return "Set the current frame index in the active Annolid video session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"frame_index": {"type": "integer", "minimum": 0}},
            "required": ["frame_index"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_frame_callback, **kwargs)


class GuiTrackNextFramesTool(FunctionTool):
    def __init__(self, track_callback: Optional[ActionCallback] = None):
        self._track_callback = track_callback

    @property
    def name(self) -> str:
        return "gui_track_next_frames"

    @property
    def description(self) -> str:
        return "Run Annolid tracking/prediction from current frame to target frame."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"to_frame": {"type": "integer", "minimum": 1}},
            "required": ["to_frame"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._track_callback, **kwargs)


class GuiSetAiTextPromptTool(FunctionTool):
    def __init__(self, set_ai_text_prompt_callback: Optional[ActionCallback] = None):
        self._set_ai_text_prompt_callback = set_ai_text_prompt_callback

    @property
    def name(self) -> str:
        return "gui_set_ai_text_prompt"

    @property
    def description(self) -> str:
        return "Set the GUI AI text prompt used by GroundingDINO + SAM segmentation."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "use_countgd": {"type": "boolean"},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_ai_text_prompt_callback, **kwargs)


class GuiRunAiTextSegmentationTool(FunctionTool):
    def __init__(
        self, run_ai_text_segmentation_callback: Optional[ActionCallback] = None
    ):
        self._run_ai_text_segmentation_callback = run_ai_text_segmentation_callback

    @property
    def name(self) -> str:
        return "gui_run_ai_text_segmentation"

    @property
    def description(self) -> str:
        return (
            "Run GUI GroundingDINO + SAM segmentation using the current AI text prompt."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._run_ai_text_segmentation_callback)


class GuiSegmentTrackVideoTool(FunctionTool):
    def __init__(self, segment_track_video_callback: Optional[ActionCallback] = None):
        self._segment_track_video_callback = segment_track_video_callback

    @property
    def name(self) -> str:
        return "gui_segment_track_video"

    @property
    def description(self) -> str:
        return (
            "Open a video, run text-prompt GroundingDINO+SAM segmentation, save, and "
            "optionally track to a target frame."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "text_prompt": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["segment", "track"]},
                "use_countgd": {"type": "boolean"},
                "model_name": {"type": "string"},
                "to_frame": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "text_prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._segment_track_video_callback, **kwargs)


class GuiLabelBehaviorSegmentsTool(FunctionTool):
    def __init__(
        self, label_behavior_segments_callback: Optional[ActionCallback] = None
    ):
        self._label_behavior_segments_callback = label_behavior_segments_callback

    @property
    def name(self) -> str:
        return "gui_label_behavior_segments"

    @property
    def description(self) -> str:
        return (
            "Auto-label behavior intervals from video segments using an LLM model "
            "and write labels into the Annolid behavior timeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "behavior_labels": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "segment_mode": {"type": "string", "enum": ["timeline", "uniform"]},
                "segment_frames": {"type": "integer", "minimum": 1},
                "max_segments": {"type": "integer", "minimum": 1},
                "subject": {"type": "string"},
                "overwrite_existing": {"type": "boolean"},
                "llm_profile": {"type": "string"},
                "llm_provider": {"type": "string"},
                "llm_model": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._label_behavior_segments_callback, **kwargs)


class GuiStartRealtimeStreamTool(FunctionTool):
    def __init__(self, start_realtime_stream_callback: Optional[ActionCallback] = None):
        self._start_realtime_stream_callback = start_realtime_stream_callback

    @property
    def name(self) -> str:
        return "gui_start_realtime_stream"

    @property
    def description(self) -> str:
        return (
            "Start realtime inference stream in Annolid and optionally enable "
            "MediaPipe face blink classification, bot reporting, and RTSP transport."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera_source": {"type": "string"},
                "model_name": {"type": "string"},
                "target_behaviors": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "viewer_type": {"type": "string", "enum": ["pyqt", "threejs"]},
                "rtsp_transport": {"type": "string", "enum": ["auto", "tcp", "udp"]},
                "classify_eye_blinks": {"type": "boolean"},
                "blink_ear_threshold": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.6,
                },
                "blink_min_consecutive_frames": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                },
                "bot_report_enabled": {"type": "boolean"},
                "bot_report_interval_sec": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 3600.0,
                },
                "bot_watch_labels": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "bot_email_report": {"type": "boolean"},
                "bot_email_to": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._start_realtime_stream_callback, **kwargs)


class GuiStopRealtimeStreamTool(FunctionTool):
    def __init__(self, stop_realtime_stream_callback: Optional[ActionCallback] = None):
        self._stop_realtime_stream_callback = stop_realtime_stream_callback

    @property
    def name(self) -> str:
        return "gui_stop_realtime_stream"

    @property
    def description(self) -> str:
        return "Stop the current realtime inference stream in Annolid."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._stop_realtime_stream_callback)


class GuiGetRealtimeStatusTool(FunctionTool):
    def __init__(self, get_realtime_status_callback: Optional[ActionCallback] = None):
        self._get_realtime_status_callback = get_realtime_status_callback

    @property
    def name(self) -> str:
        return "gui_get_realtime_status"

    @property
    def description(self) -> str:
        return "Get realtime stream status, source, model, and active viewer details."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._get_realtime_status_callback)


class GuiListRealtimeModelsTool(FunctionTool):
    def __init__(self, list_realtime_models_callback: Optional[ActionCallback] = None):
        self._list_realtime_models_callback = list_realtime_models_callback

    @property
    def name(self) -> str:
        return "gui_list_realtime_models"

    @property
    def description(self) -> str:
        return "List available realtime model presets and their weight identifiers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._list_realtime_models_callback)


class GuiListRealtimeLogsTool(FunctionTool):
    def __init__(self, list_realtime_logs_callback: Optional[ActionCallback] = None):
        self._list_realtime_logs_callback = list_realtime_logs_callback

    @property
    def name(self) -> str:
        return "gui_list_realtime_logs"

    @property
    def description(self) -> str:
        return "List current realtime detection and bot-event log file paths."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._list_realtime_logs_callback)


class GuiCheckStreamSourceTool(FunctionTool):
    def __init__(self, check_stream_source_callback: Optional[ActionCallback] = None):
        self._check_stream_source_callback = check_stream_source_callback

    @property
    def name(self) -> str:
        return "gui_check_stream_source"

    @property
    def description(self) -> str:
        return (
            "Probe a camera/video stream source for connectivity and frame availability "
            "before starting realtime inference."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera_source": {"type": "string"},
                "rtsp_transport": {"type": "string", "enum": ["auto", "tcp", "udp"]},
                "timeout_sec": {"type": "number", "minimum": 0.5, "maximum": 30.0},
                "probe_frames": {"type": "integer", "minimum": 1, "maximum": 60},
                "save_snapshot": {"type": "boolean"},
                "email_to": {"type": "string"},
                "email_subject": {"type": "string"},
                "email_content": {"type": "string"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._check_stream_source_callback, **kwargs)
