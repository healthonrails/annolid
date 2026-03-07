"""Stable domain-layer entry points for Annolid business concepts."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_SCHEMA_FILENAME": "annolid.domain.project_schema",
    "BehaviorDefinition": "annolid.domain.project_schema",
    "BehaviorEvent": "annolid.domain.behavior_events",
    "BehaviorInterval": "annolid.domain.timelines",
    "BehaviorSpan": "annolid.domain.behavior_events",
    "BehaviorSpec": "annolid.domain.project_schema",
    "BinnedTimeBudgetRow": "annolid.domain.timelines",
    "CategoryDefinition": "annolid.domain.project_schema",
    "CollectedPair": "annolid.domain.datasets",
    "Config": "annolid.domain.datasets",
    "DeepLabCutTrainingImportConfig": "annolid.domain.datasets",
    "FilePaths": "annolid.domain.datasets",
    "FrameRef": "annolid.domain.timelines",
    "InstanceRegistry": "annolid.domain.keypoints",
    "InstanceState": "annolid.domain.keypoints",
    "KeypointState": "annolid.domain.keypoints",
    "ModifierDefinition": "annolid.domain.project_schema",
    "ProcessingConfig": "annolid.domain.datasets",
    "ProjectSchema": "annolid.domain.project_schema",
    "SubjectDefinition": "annolid.domain.project_schema",
    "TimeBudgetRow": "annolid.domain.timelines",
    "Track": "annolid.domain.tracks",
    "TrackObservation": "annolid.domain.tracks",
    "VideoFrameDataset": "annolid.domain.datasets",
    "combine_labels": "annolid.domain.keypoints",
    "default_behavior_spec": "annolid.domain.project_schema",
    "load_behavior_spec": "annolid.domain.project_schema",
    "normalize_event_label": "annolid.domain.behavior_events",
    "save_behavior_spec": "annolid.domain.project_schema",
    "validate_behavior_spec": "annolid.domain.project_schema",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
