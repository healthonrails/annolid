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


def register_annolid_gui_tools(
    registry: FunctionToolRegistry,
    *,
    context_callback: Optional[ContextCallback] = None,
    image_path_callback: Optional[PathCallback] = None,
    open_video_callback: Optional[ActionCallback] = None,
    set_frame_callback: Optional[ActionCallback] = None,
    set_prompt_callback: Optional[ActionCallback] = None,
    send_prompt_callback: Optional[ActionCallback] = None,
    set_chat_model_callback: Optional[ActionCallback] = None,
    select_annotation_model_callback: Optional[ActionCallback] = None,
    track_next_frames_callback: Optional[ActionCallback] = None,
) -> None:
    """Register GUI-only tools for Annolid Bot sessions."""
    registry.register(GuiContextTool(context_callback=context_callback))
    registry.register(GuiSharedImagePathTool(image_path_callback=image_path_callback))
    registry.register(GuiOpenVideoTool(open_video_callback=open_video_callback))
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
