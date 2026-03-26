"""Service wrappers for GUI chat backend helper functions and constants."""

from __future__ import annotations

from annolid.core.agent.gui_backend.commands import (
    looks_like_local_access_refusal,
    parse_direct_gui_command,
    prompt_may_need_tools,
)
from annolid.core.agent.gui_backend.context_blocks import (
    build_live_pdf_context_prompt_block,
    build_live_web_context_prompt_block,
)
from annolid.core.agent.gui_backend.fallbacks import (
    candidate_web_urls_for_prompt,
    extract_page_text_from_web_steps,
    try_browser_search_fallback,
    try_open_page_content_fallback,
    try_open_pdf_content_fallback,
    try_web_fetch_fallback,
    try_web_search_fallback,
)
from annolid.core.agent.gui_backend.heuristics import (
    build_extractive_summary,
    contains_hint,
    extract_web_urls,
    looks_like_pdf_phrase_miss_response,
    looks_like_pdf_summary_request,
    looks_like_knowledge_gap_response,
    looks_like_pdf_read_promise,
    looks_like_open_pdf_suggestion,
    looks_like_open_url_suggestion,
    looks_like_url_request,
    looks_like_web_access_refusal,
    prompt_may_need_mcp,
    should_attach_live_pdf_context,
    should_attach_live_web_context,
    should_attach_tracking_stats_context,
    topic_tokens,
)
from annolid.core.agent.gui_backend.ollama_adapter import (
    build_gui_ollama_llm_callable,
    collect_gui_ollama_stream,
    extract_gui_ollama_text,
    format_gui_tool_trace,
    normalize_gui_messages_for_ollama,
    parse_gui_ollama_tool_calls,
)
from annolid.core.agent.gui_backend.paths import (
    build_workspace_roots,
    extract_pdf_path_candidates,
    extract_video_path_candidates,
    find_video_by_basename_in_roots,
    list_available_pdfs_in_roots,
    resolve_video_path_for_roots,
)
from annolid.core.agent.gui_backend.prompt_builder import (
    PromptBuildInputs,
    build_compact_system_prompt as build_gui_compact_system_prompt,
)
from annolid.core.agent.gui_backend.response_finalize import (
    apply_direct_gui_fallback as gui_apply_direct_gui_fallback,
    apply_empty_ollama_recovery as gui_apply_empty_ollama_recovery,
    apply_pdf_response_fallback as gui_apply_pdf_response_fallback,
    apply_web_response_fallbacks as gui_apply_web_response_fallbacks,
    ensure_non_empty_final_text as gui_ensure_non_empty_final_text,
    sanitize_final_response_text as gui_sanitize_final_response_text,
    should_apply_web_refusal_fallback as gui_should_apply_web_refusal_fallback,
)
from annolid.core.agent.gui_backend.pdf_summary import (
    looks_like_raw_pdf_extract_response as gui_looks_like_raw_pdf_extract_response,
    summarize_active_pdf_with_cache as gui_summarize_active_pdf_with_cache,
)
from annolid.core.agent.gui_backend.runtime_flow import (
    emit_agent_loop_result,
    maybe_handle_ollama_plain_mode,
)
from annolid.core.agent.gui_backend.telemetry import (
    log_agent_result as gui_log_agent_result,
    wrap_tool_callback as gui_wrap_tool_callback,
)
from annolid.core.agent.gui_backend.tutorials import (
    build_tutorial_fallback_markdown,
    build_tutorial_model_prompts,
    collect_tutorial_evidence,
    select_annolid_reference_paths,
)
from annolid.core.agent.gui_backend.turn_state import (
    ERROR_TYPE_CANCELLED,
    ERROR_TYPE_POLICY,
    ERROR_TYPE_USER,
    TURN_STATUS_CANCELLING,
    TURN_STATUS_CANCELLED,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_FAILED,
    TURN_STATUS_QUEUED,
    TURN_STATUS_RUNNING,
)

__all__ = [
    "ERROR_TYPE_CANCELLED",
    "ERROR_TYPE_POLICY",
    "ERROR_TYPE_USER",
    "PromptBuildInputs",
    "TURN_STATUS_CANCELLING",
    "TURN_STATUS_CANCELLED",
    "TURN_STATUS_COMPLETED",
    "TURN_STATUS_FAILED",
    "TURN_STATUS_QUEUED",
    "TURN_STATUS_RUNNING",
    "build_extractive_summary",
    "build_gui_compact_system_prompt",
    "build_live_pdf_context_prompt_block",
    "build_live_web_context_prompt_block",
    "build_tutorial_fallback_markdown",
    "build_tutorial_model_prompts",
    "build_workspace_roots",
    "candidate_web_urls_for_prompt",
    "collect_gui_ollama_stream",
    "collect_tutorial_evidence",
    "contains_hint",
    "emit_agent_loop_result",
    "extract_gui_ollama_text",
    "extract_page_text_from_web_steps",
    "extract_pdf_path_candidates",
    "extract_video_path_candidates",
    "extract_web_urls",
    "find_video_by_basename_in_roots",
    "format_gui_tool_trace",
    "gui_apply_direct_gui_fallback",
    "gui_apply_empty_ollama_recovery",
    "gui_apply_pdf_response_fallback",
    "gui_apply_web_response_fallbacks",
    "gui_ensure_non_empty_final_text",
    "gui_log_agent_result",
    "gui_sanitize_final_response_text",
    "gui_should_apply_web_refusal_fallback",
    "gui_looks_like_raw_pdf_extract_response",
    "gui_summarize_active_pdf_with_cache",
    "gui_wrap_tool_callback",
    "list_available_pdfs_in_roots",
    "looks_like_knowledge_gap_response",
    "looks_like_pdf_phrase_miss_response",
    "looks_like_pdf_summary_request",
    "looks_like_pdf_read_promise",
    "looks_like_local_access_refusal",
    "looks_like_open_pdf_suggestion",
    "looks_like_open_url_suggestion",
    "looks_like_url_request",
    "looks_like_web_access_refusal",
    "maybe_handle_ollama_plain_mode",
    "normalize_gui_messages_for_ollama",
    "parse_direct_gui_command",
    "parse_gui_ollama_tool_calls",
    "prompt_may_need_mcp",
    "prompt_may_need_tools",
    "resolve_video_path_for_roots",
    "select_annolid_reference_paths",
    "should_attach_live_pdf_context",
    "should_attach_live_web_context",
    "should_attach_tracking_stats_context",
    "topic_tokens",
    "try_browser_search_fallback",
    "try_open_page_content_fallback",
    "try_open_pdf_content_fallback",
    "try_web_fetch_fallback",
    "try_web_search_fallback",
    "build_gui_ollama_llm_callable",
]
