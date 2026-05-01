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
    ("web_capture_screenshot_callback", "web_capture_screenshot"),
    ("web_describe_view_callback", "web_describe_view"),
    ("web_extract_structured_callback", "web_extract_structured"),
    ("web_click_callback", "web_click"),
    ("web_type_callback", "web_type"),
    ("web_scroll_callback", "web_scroll"),
    ("web_find_forms_callback", "web_find_forms"),
    ("web_save_current_callback", "web_save_current"),
    ("web_run_steps_callback", "web_run_steps"),
    ("open_pdf_callback", "open_pdf"),
    ("pdf_get_state_callback", "pdf_get_state"),
    ("pdf_get_text_callback", "pdf_get_text"),
    ("pdf_summarize_callback", "pdf_summarize"),
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
    ("process_video_behaviors_callback", "process_video_behaviors"),
    ("score_aggression_bouts_callback", "score_aggression_bouts"),
    ("behavior_catalog_callback", "behavior_catalog"),
    ("analyze_tracking_stats_callback", "analyze_tracking_stats"),
    ("correct_tracking_ndjson_callback", "correct_tracking_ndjson"),
    ("start_realtime_stream_callback", "start_realtime_stream"),
    ("stop_realtime_stream_callback", "stop_realtime_stream"),
    ("get_realtime_status_callback", "get_realtime_status"),
    ("list_realtime_models_callback", "list_realtime_models"),
    ("list_realtime_logs_callback", "list_realtime_logs"),
    ("list_logs_callback", "list_logs"),
    ("open_log_folder_callback", "open_log_folder"),
    ("remove_log_folder_callback", "remove_log_folder"),
    ("list_log_files_callback", "list_log_files"),
    ("read_log_file_callback", "read_log_file"),
    ("search_logs_callback", "search_logs"),
    ("check_stream_source_callback", "check_stream_source"),
    ("list_shapes_callback", "list_shapes"),
    ("select_shapes_callback", "select_shapes"),
    ("set_selected_shape_label_callback", "set_selected_shape_label"),
    ("delete_selected_shapes_callback", "delete_selected_shapes"),
    ("list_shapes_in_annotation_callback", "list_shapes_in_annotation"),
    ("relabel_shapes_in_annotation_callback", "relabel_shapes_in_annotation"),
    ("delete_shapes_in_annotation_callback", "delete_shapes_in_annotation"),
    ("arxiv_search_callback", "arxiv_search"),
    ("list_pdfs_callback", "list_pdfs"),
    ("save_citation_callback", "save_citation"),
    ("verify_citations_callback", "verify_citations"),
    ("generate_annolid_tutorial_callback", "generate_annolid_tutorial"),
    ("self_update_callback", "self_update"),
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
