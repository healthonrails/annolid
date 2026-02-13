from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional

from .function_base import FunctionTool
from .function_registry import FunctionToolRegistry


ContextCallback = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]
PathCallback = Callable[[], str | Awaitable[str]]
ActionCallback = Callable[..., Any | Awaitable[Any]]


async def _run_callback(callback: Optional[ActionCallback], **kwargs: Any) -> str:
    if callback is None:
        return json.dumps({"error": "GUI action callback is not configured."})
    try:
        payload = callback(**kwargs)
        if asyncio.iscoroutine(payload):
            payload = await payload
        if payload is None:
            payload = {"ok": True}
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return json.dumps(payload)
        return json.dumps({"ok": True, "result": str(payload)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


class GuiContextTool(FunctionTool):
    def __init__(self, context_callback: Optional[ContextCallback] = None):
        self._context_callback = context_callback

    @property
    def name(self) -> str:
        return "gui_context"

    @property
    def description(self) -> str:
        return (
            "Get current Annolid GUI state (session, model, media, frame, file paths)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        if self._context_callback is None:
            return json.dumps({"error": "GUI context callback is not configured."})
        try:
            payload = self._context_callback()
            if asyncio.iscoroutine(payload):
                payload = await payload
            if not isinstance(payload, dict):
                return json.dumps(
                    {"error": "GUI context callback returned invalid payload."}
                )
            return json.dumps(payload)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


class GuiSharedImagePathTool(FunctionTool):
    def __init__(self, image_path_callback: Optional[PathCallback] = None):
        self._image_path_callback = image_path_callback

    @property
    def name(self) -> str:
        return "gui_shared_image_path"

    @property
    def description(self) -> str:
        return "Return currently shared image path in Annolid Bot UI."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        if self._image_path_callback is None:
            return json.dumps({"error": "GUI image-path callback is not configured."})
        try:
            value = self._image_path_callback()
            if asyncio.iscoroutine(value):
                value = await value
            path = str(value or "").strip()
            return json.dumps({"image_path": path, "has_image": bool(path)})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


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


class GuiOpenPdfTool(FunctionTool):
    def __init__(self, open_pdf_callback: Optional[ActionCallback] = None):
        self._open_pdf_callback = open_pdf_callback

    @property
    def name(self) -> str:
        return "gui_open_pdf"

    @property
    def description(self) -> str:
        return (
            "Open a PDF in Annolid using the same workflow as File > Open PDF... "
            "If path is provided, open that file directly without prompting."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._open_pdf_callback)


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


class GuiSetPromptTool(FunctionTool):
    def __init__(self, set_prompt_callback: Optional[ActionCallback] = None):
        self._set_prompt_callback = set_prompt_callback

    @property
    def name(self) -> str:
        return "gui_set_chat_prompt"

    @property
    def description(self) -> str:
        return "Set Annolid Bot prompt text in the chat input box."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_prompt_callback, **kwargs)


class GuiSendPromptTool(FunctionTool):
    def __init__(self, send_prompt_callback: Optional[ActionCallback] = None):
        self._send_prompt_callback = send_prompt_callback

    @property
    def name(self) -> str:
        return "gui_send_chat_prompt"

    @property
    def description(self) -> str:
        return "Send the current Annolid Bot chat prompt."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._send_prompt_callback)


class GuiSetChatModelTool(FunctionTool):
    def __init__(self, set_model_callback: Optional[ActionCallback] = None):
        self._set_model_callback = set_model_callback

    @property
    def name(self) -> str:
        return "gui_set_chat_model"

    @property
    def description(self) -> str:
        return "Set Annolid Bot provider and model, then persist the selection."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["ollama", "openai", "openrouter", "gemini"],
                },
                "model": {"type": "string", "minLength": 1},
            },
            "required": ["provider", "model"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_model_callback, **kwargs)


class GuiSelectAnnotationModelTool(FunctionTool):
    def __init__(self, select_model_callback: Optional[ActionCallback] = None):
        self._select_model_callback = select_model_callback

    @property
    def name(self) -> str:
        return "gui_select_annotation_model"

    @property
    def description(self) -> str:
        return (
            "Select the Annolid annotation/tracking model in the main model dropdown."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"model_name": {"type": "string", "minLength": 1}},
            "required": ["model_name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._select_model_callback, **kwargs)


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


def register_annolid_gui_tools(
    registry: FunctionToolRegistry,
    *,
    context_callback: Optional[ContextCallback] = None,
    image_path_callback: Optional[PathCallback] = None,
    open_video_callback: Optional[ActionCallback] = None,
    open_pdf_callback: Optional[ActionCallback] = None,
    set_frame_callback: Optional[ActionCallback] = None,
    set_prompt_callback: Optional[ActionCallback] = None,
    send_prompt_callback: Optional[ActionCallback] = None,
    set_chat_model_callback: Optional[ActionCallback] = None,
    select_annotation_model_callback: Optional[ActionCallback] = None,
    track_next_frames_callback: Optional[ActionCallback] = None,
    set_ai_text_prompt_callback: Optional[ActionCallback] = None,
    run_ai_text_segmentation_callback: Optional[ActionCallback] = None,
    segment_track_video_callback: Optional[ActionCallback] = None,
    label_behavior_segments_callback: Optional[ActionCallback] = None,
    start_realtime_stream_callback: Optional[ActionCallback] = None,
    stop_realtime_stream_callback: Optional[ActionCallback] = None,
) -> None:
    """Register GUI-only tools for Annolid Bot sessions."""
    registry.register(GuiContextTool(context_callback=context_callback))
    registry.register(GuiSharedImagePathTool(image_path_callback=image_path_callback))
    registry.register(GuiOpenVideoTool(open_video_callback=open_video_callback))
    registry.register(GuiOpenPdfTool(open_pdf_callback=open_pdf_callback))
    registry.register(GuiSetFrameTool(set_frame_callback=set_frame_callback))
    registry.register(GuiSetPromptTool(set_prompt_callback=set_prompt_callback))
    registry.register(GuiSendPromptTool(send_prompt_callback=send_prompt_callback))
    registry.register(GuiSetChatModelTool(set_model_callback=set_chat_model_callback))
    registry.register(
        GuiSelectAnnotationModelTool(
            select_model_callback=select_annotation_model_callback
        )
    )
    registry.register(GuiTrackNextFramesTool(track_callback=track_next_frames_callback))
    registry.register(
        GuiSetAiTextPromptTool(set_ai_text_prompt_callback=set_ai_text_prompt_callback)
    )
    registry.register(
        GuiRunAiTextSegmentationTool(
            run_ai_text_segmentation_callback=run_ai_text_segmentation_callback
        )
    )
    registry.register(
        GuiSegmentTrackVideoTool(
            segment_track_video_callback=segment_track_video_callback
        )
    )
    registry.register(
        GuiLabelBehaviorSegmentsTool(
            label_behavior_segments_callback=label_behavior_segments_callback
        )
    )
    registry.register(
        GuiStartRealtimeStreamTool(
            start_realtime_stream_callback=start_realtime_stream_callback
        )
    )
    registry.register(
        GuiStopRealtimeStreamTool(
            stop_realtime_stream_callback=stop_realtime_stream_callback
        )
    )
