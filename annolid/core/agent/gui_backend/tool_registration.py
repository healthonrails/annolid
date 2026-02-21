from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from annolid.core.agent.tools import FunctionToolRegistry, register_annolid_gui_tools


_WRAPPED_GUI_TOOL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("open_video_callback", "open_video"),
    ("open_url_callback", "open_url"),
    ("open_in_browser_callback", "open_in_browser"),
    ("open_threejs_callback", "open_threejs"),
    ("open_threejs_example_callback", "open_threejs_example"),
    ("web_get_dom_text_callback", "web_get_dom_text"),
    ("web_click_callback", "web_click"),
    ("web_type_callback", "web_type"),
    ("web_scroll_callback", "web_scroll"),
    ("web_find_forms_callback", "web_find_forms"),
    ("web_run_steps_callback", "web_run_steps"),
    ("open_pdf_callback", "open_pdf"),
    ("pdf_get_state_callback", "pdf_get_state"),
    ("pdf_get_text_callback", "pdf_get_text"),
    ("pdf_find_sections_callback", "pdf_find_sections"),
    ("set_frame_callback", "set_frame"),
    ("set_prompt_callback", "set_prompt"),
    ("send_prompt_callback", "send_prompt"),
    ("set_chat_model_callback", "set_chat_model"),
    ("select_annotation_model_callback", "select_annotation_model"),
    ("track_next_frames_callback", "track_next_frames"),
    ("set_ai_text_prompt_callback", "set_ai_text_prompt"),
    ("run_ai_text_segmentation_callback", "run_ai_text_segmentation"),
    ("segment_track_video_callback", "segment_track_video"),
    ("label_behavior_segments_callback", "label_behavior_segments"),
    ("start_realtime_stream_callback", "start_realtime_stream"),
    ("stop_realtime_stream_callback", "stop_realtime_stream"),
    ("get_realtime_status_callback", "get_realtime_status"),
    ("list_realtime_models_callback", "list_realtime_models"),
    ("list_realtime_logs_callback", "list_realtime_logs"),
    ("check_stream_source_callback", "check_stream_source"),
    ("arxiv_search_callback", "arxiv_search"),
    ("list_pdfs_callback", "list_pdfs"),
    ("save_citation_callback", "save_citation"),
)


def register_chat_gui_tools(
    tools: FunctionToolRegistry,
    *,
    context_callback: Callable[..., Any],
    image_path_callback: Callable[..., Any],
    wrap_tool_callback: Callable[[str, Callable[..., Any]], Callable[..., Any]],
    handlers: Dict[str, Callable[..., Any]],
) -> None:
    kwargs: Dict[str, Any] = {
        "context_callback": context_callback,
        "image_path_callback": image_path_callback,
    }
    for register_kwarg, handler_key in _WRAPPED_GUI_TOOL_SPECS:
        callback = handlers.get(handler_key)
        if callback is None:
            raise KeyError(f"Missing GUI tool handler: {handler_key}")
        kwargs[register_kwarg] = wrap_tool_callback(handler_key, callback)
    register_annolid_gui_tools(tools, **kwargs)
