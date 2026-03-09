"""Infrastructure-layer package namespace."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "get_agent_config_path": "annolid.infrastructure.agent_config",
    "get_agent_workspace_path": "annolid.infrastructure.agent_workspace",
    "AnnotationStore": "annolid.infrastructure.persistence",
    "AnnotationStoreError": "annolid.infrastructure.persistence",
    "LLMChatAdapter": "annolid.infrastructure.external_apis",
    "Qwen3EmbeddingAdapter": "annolid.infrastructure.external_apis",
    "configure_ultralytics_cache": "annolid.infrastructure.model_downloads",
    "construct_filename": "annolid.infrastructure.filesystem",
    "create_qapp": "annolid.infrastructure.runtime",
    "ensure_provider_env": "annolid.infrastructure.external_apis",
    "ensure_ultralytics_asset_cached": "annolid.infrastructure.model_downloads",
    "find_manual_labeled_json_files": "annolid.infrastructure.filesystem",
    "find_most_recent_file": "annolid.infrastructure.filesystem",
    "get_cache_root": "annolid.infrastructure.model_downloads",
    "load_agent_config": "annolid.infrastructure.agent_config",
    "get_frame_number_from_json": "annolid.infrastructure.filesystem",
    "load_labelme_json": "annolid.infrastructure.persistence",
    "save_agent_config": "annolid.infrastructure.agent_config",
    "resolve_llm_config": "annolid.infrastructure.external_apis",
    "resolve_weight_path": "annolid.infrastructure.model_downloads",
    "sanitize_qt_plugin_env": "annolid.infrastructure.runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
