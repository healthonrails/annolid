"""Application services layer for Annolid."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AgentPipelineRequest": "annolid.services.agent",
    "BehaviorTimeBudgetReport": "annolid.services.time_budget",
    "EmbeddingSearchMatch": "annolid.services.embedding_search",
    "build_tracking_video_processor": "annolid.services.tracking",
    "build_yolo_dataset_from_index": "annolid.services.export",
    "compute_behavior_time_budget_report": "annolid.services.time_budget",
    "create_cutie_engine": "annolid.services.tracking",
    "create_segmented_cutie_executor": "annolid.services.tracking",
    "export_behavior_time_budget": "annolid.services.export",
    "export_labelme_json_to_csv": "annolid.services.export",
    "has_tracking_completion_artifacts": "annolid.services.tracking",
    "import_deeplabcut_dataset": "annolid.services.export",
    "is_cutie_model_name": "annolid.services.tracking",
    "predict_behavior": "annolid.services.inference",
    "resolve_tracking_video_processor_class": "annolid.services.tracking",
    "run_agent_pipeline": "annolid.services.agent",
    "run_behavior_inference_cli": "annolid.services.inference",
    "run_behavior_training_cli": "annolid.services.training",
    "run_behavior_video_agent": "annolid.services.inference",
    "run_embedding_search": "annolid.services.embedding_search",
    "run_polygon_frame_training_cli": "annolid.services.training",
    "run_tracking_video_frames": "annolid.services.tracking",
    "run_video_processor_frames": "annolid.services.tracking",
    "search_indexed_frames": "annolid.services.search",
    "search_video_frames": "annolid.services.search",
    "train_behavior_model": "annolid.services.training",
    "validate_behavior_model": "annolid.services.training",
    "write_behavior_time_budget_report_csv": "annolid.services.time_budget",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
