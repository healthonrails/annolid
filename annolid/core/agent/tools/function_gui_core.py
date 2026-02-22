from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from .function_gui_base import (
    ActionCallback,
    ContextCallback,
    PathCallback,
    _run_callback,
)
from .function_base import FunctionTool


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


class GuiSaveCitationTool(FunctionTool):
    def __init__(self, save_citation_callback: Optional[ActionCallback] = None):
        self._save_citation_callback = save_citation_callback

    @property
    def name(self) -> str:
        return "gui_save_citation"

    @property
    def description(self) -> str:
        return (
            "Save a citation to a BibTeX file from the currently open PDF/web page in "
            "Annolid."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "bib_file": {"type": "string"},
                "source": {"type": "string", "enum": ["auto", "pdf", "web"]},
                "entry_type": {"type": "string"},
                "validate_before_save": {"type": "boolean", "default": True},
                "strict_validation": {"type": "boolean", "default": False},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._save_citation_callback, **kwargs)


class GuiGenerateAnnolidTutorialTool(FunctionTool):
    def __init__(
        self, generate_tutorial_callback: Optional[ActionCallback] = None
    ) -> None:
        self._generate_tutorial_callback = generate_tutorial_callback

    @property
    def name(self) -> str:
        return "gui_generate_annolid_tutorial"

    @property
    def description(self) -> str:
        return (
            "Generate an Annolid tutorial from local code/docs, optionally save it, "
            "and return a structured response."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "minLength": 1},
                "level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                },
                "save_to_file": {"type": "boolean", "default": False},
                "include_code_refs": {"type": "boolean", "default": True},
                "open_in_web_viewer": {"type": "boolean", "default": True},
            },
            "required": ["topic"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._generate_tutorial_callback, **kwargs)
