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
            "MediaPipe face blink classification."
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
