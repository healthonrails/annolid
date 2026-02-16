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


class GuiOpenUrlTool(FunctionTool):
    def __init__(self, open_url_callback: Optional[ActionCallback] = None):
        self._open_url_callback = open_url_callback

    @property
    def name(self) -> str:
        return "gui_open_url"

    @property
    def description(self) -> str:
        return (
            "Open a web URL in Annolid's embedded canvas web viewer. "
            "Accepts full URLs or domains like google.com."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_url_callback, **kwargs)


class GuiOpenInBrowserTool(FunctionTool):
    def __init__(self, open_in_browser_callback: Optional[ActionCallback] = None):
        self._open_in_browser_callback = open_in_browser_callback

    @property
    def name(self) -> str:
        return "gui_open_in_browser"

    @property
    def description(self) -> str:
        return (
            "Open a web URL in the system browser instead of the embedded Annolid "
            "web viewer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_in_browser_callback, **kwargs)


class GuiWebGetDomTextTool(FunctionTool):
    def __init__(self, web_get_dom_text_callback: Optional[ActionCallback] = None):
        self._web_get_dom_text_callback = web_get_dom_text_callback

    @property
    def name(self) -> str:
        return "gui_web_get_dom_text"

    @property
    def description(self) -> str:
        return "Read visible text content from the active embedded web page."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"max_chars": {"type": "integer", "minimum": 200}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_get_dom_text_callback, **kwargs)


class GuiWebClickTool(FunctionTool):
    def __init__(self, web_click_callback: Optional[ActionCallback] = None):
        self._web_click_callback = web_click_callback

    @property
    def name(self) -> str:
        return "gui_web_click"

    @property
    def description(self) -> str:
        return "Click an element in the embedded web page by CSS selector."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"selector": {"type": "string", "minLength": 1}},
            "required": ["selector"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_click_callback, **kwargs)


class GuiWebTypeTool(FunctionTool):
    def __init__(self, web_type_callback: Optional[ActionCallback] = None):
        self._web_type_callback = web_type_callback

    @property
    def name(self) -> str:
        return "gui_web_type"

    @property
    def description(self) -> str:
        return (
            "Type text into an input-like element in embedded web page by CSS selector."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "minLength": 1},
                "text": {"type": "string"},
                "submit": {"type": "boolean"},
            },
            "required": ["selector", "text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_type_callback, **kwargs)


class GuiWebScrollTool(FunctionTool):
    def __init__(self, web_scroll_callback: Optional[ActionCallback] = None):
        self._web_scroll_callback = web_scroll_callback

    @property
    def name(self) -> str:
        return "gui_web_scroll"

    @property
    def description(self) -> str:
        return "Scroll the active embedded web page by delta pixels."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"delta_y": {"type": "integer"}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_scroll_callback, **kwargs)


class GuiWebFindFormsTool(FunctionTool):
    def __init__(self, web_find_forms_callback: Optional[ActionCallback] = None):
        self._web_find_forms_callback = web_find_forms_callback

    @property
    def name(self) -> str:
        return "gui_web_find_forms"

    @property
    def description(self) -> str:
        return (
            "List forms and input fields on the active embedded web page for "
            "automation planning."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._web_find_forms_callback)


class GuiWebRunStepsTool(FunctionTool):
    def __init__(self, web_run_steps_callback: Optional[ActionCallback] = None):
        self._web_run_steps_callback = web_run_steps_callback

    @property
    def name(self) -> str:
        return "gui_web_run_steps"

    @property
    def description(self) -> str:
        return (
            "Run a sequence of embedded web actions for browser automation. "
            "Supported actions: open_url, open_in_browser, get_text, click, type, "
            "scroll, find_forms, wait."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "minLength": 1},
                            "url": {"type": "string"},
                            "selector": {"type": "string"},
                            "text": {"type": "string"},
                            "submit": {"type": "boolean"},
                            "delta_y": {"type": "integer"},
                            "max_chars": {"type": "integer", "minimum": 200},
                            "wait_ms": {"type": "integer", "minimum": 0},
                        },
                        "required": ["action"],
                    },
                    "minItems": 1,
                },
                "stop_on_error": {"type": "boolean"},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["steps"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_run_steps_callback, **kwargs)


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
        return await _run_callback(self._open_pdf_callback, **kwargs)


class GuiPdfGetStateTool(FunctionTool):
    def __init__(self, pdf_get_state_callback: Optional[ActionCallback] = None):
        self._pdf_get_state_callback = pdf_get_state_callback

    @property
    def name(self) -> str:
        return "gui_pdf_get_state"

    @property
    def description(self) -> str:
        return "Get state of the currently opened PDF in Annolid."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._pdf_get_state_callback)


class GuiPdfGetTextTool(FunctionTool):
    def __init__(self, pdf_get_text_callback: Optional[ActionCallback] = None):
        self._pdf_get_text_callback = pdf_get_text_callback

    @property
    def name(self) -> str:
        return "gui_pdf_get_text"

    @property
    def description(self) -> str:
        return (
            "Read text from the currently opened PDF in Annolid, starting from the "
            "current page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_chars": {"type": "integer", "minimum": 200},
                "pages": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._pdf_get_text_callback, **kwargs)


class GuiPdfFindSectionsTool(FunctionTool):
    def __init__(self, pdf_find_sections_callback: Optional[ActionCallback] = None):
        self._pdf_find_sections_callback = pdf_find_sections_callback

    @property
    def name(self) -> str:
        return "gui_pdf_find_sections"

    @property
    def description(self) -> str:
        return (
            "Detect likely section headings in the currently opened PDF and return "
            "their page numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_sections": {"type": "integer", "minimum": 1, "maximum": 200},
                "max_pages": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._pdf_find_sections_callback, **kwargs)


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


class GuiArxivSearchTool(FunctionTool):
    def __init__(self, arxiv_search_callback: Optional[ActionCallback] = None):
        self._arxiv_search_callback = arxiv_search_callback

    @property
    def name(self) -> str:
        return "gui_arxiv_search"

    @property
    def description(self) -> str:
        return "Search arXiv for papers, download the best match, and open it in Annolid's PDF viewer."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._arxiv_search_callback, **kwargs)


class GuiListPdfsTool(FunctionTool):
    def __init__(self, list_pdfs_callback: Optional[ActionCallback] = None):
        self._list_pdfs_callback = list_pdfs_callback

    @property
    def name(self) -> str:
        return "gui_list_pdfs"

    @property
    def description(self) -> str:
        return (
            "List all local PDF files available in the workspace downloads or "
            "other accessible directories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search query to filter PDF files by name.",
                }
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._list_pdfs_callback, **kwargs)


def register_annolid_gui_tools(
    registry: FunctionToolRegistry,
    *,
    context_callback: Optional[ContextCallback] = None,
    image_path_callback: Optional[PathCallback] = None,
    open_video_callback: Optional[ActionCallback] = None,
    open_url_callback: Optional[ActionCallback] = None,
    open_in_browser_callback: Optional[ActionCallback] = None,
    web_get_dom_text_callback: Optional[ActionCallback] = None,
    web_click_callback: Optional[ActionCallback] = None,
    web_type_callback: Optional[ActionCallback] = None,
    web_scroll_callback: Optional[ActionCallback] = None,
    web_find_forms_callback: Optional[ActionCallback] = None,
    web_run_steps_callback: Optional[ActionCallback] = None,
    open_pdf_callback: Optional[ActionCallback] = None,
    pdf_get_state_callback: Optional[ActionCallback] = None,
    pdf_get_text_callback: Optional[ActionCallback] = None,
    pdf_find_sections_callback: Optional[ActionCallback] = None,
    arxiv_search_callback: Optional[ActionCallback] = None,
    list_pdfs_callback: Optional[ActionCallback] = None,
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
    registry.register(GuiOpenUrlTool(open_url_callback=open_url_callback))
    registry.register(
        GuiOpenInBrowserTool(open_in_browser_callback=open_in_browser_callback)
    )
    registry.register(
        GuiWebGetDomTextTool(web_get_dom_text_callback=web_get_dom_text_callback)
    )
    registry.register(GuiWebClickTool(web_click_callback=web_click_callback))
    registry.register(GuiWebTypeTool(web_type_callback=web_type_callback))
    registry.register(GuiWebScrollTool(web_scroll_callback=web_scroll_callback))
    registry.register(
        GuiWebFindFormsTool(web_find_forms_callback=web_find_forms_callback)
    )
    registry.register(GuiWebRunStepsTool(web_run_steps_callback=web_run_steps_callback))
    registry.register(GuiOpenPdfTool(open_pdf_callback=open_pdf_callback))
    registry.register(GuiPdfGetStateTool(pdf_get_state_callback=pdf_get_state_callback))
    registry.register(GuiPdfGetTextTool(pdf_get_text_callback=pdf_get_text_callback))
    registry.register(
        GuiPdfFindSectionsTool(pdf_find_sections_callback=pdf_find_sections_callback)
    )
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
    registry.register(GuiArxivSearchTool(arxiv_search_callback=arxiv_search_callback))
    registry.register(GuiListPdfsTool(list_pdfs_callback=list_pdfs_callback))
