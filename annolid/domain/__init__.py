"""Stable domain-layer entry points for Annolid business concepts."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AnalysisRun": "annolid.domain.behavior_agent",
    "AggressionSubEventType": "annolid.domain.behavior_agent",
    "DEFAULT_SCHEMA_FILENAME": "annolid.domain.project_schema",
    "BehaviorDefinition": "annolid.domain.project_schema",
    "BehaviorEvent": "annolid.domain.behavior_events",
    "BehaviorIRRRow": "annolid.domain.timelines",
    "BehaviorInterval": "annolid.domain.timelines",
    "BehaviorSegment": "annolid.domain.behavior_agent",
    "BehaviorSpan": "annolid.domain.behavior_events",
    "BehaviorSpec": "annolid.domain.project_schema",
    "BehaviorSubEvent": "annolid.domain.behavior_agent",
    "BinnedTimeBudgetRow": "annolid.domain.timelines",
    "BoutDefinition": "annolid.domain.timelines",
    "BoutSummaryRow": "annolid.domain.timelines",
    "CategoryDefinition": "annolid.domain.project_schema",
    "CohortBehaviorReport": "annolid.domain.timelines",
    "CohortGroupSummaryRow": "annolid.domain.timelines",
    "CohortRecordingSpec": "annolid.domain.timelines",
    "CohortSubjectRow": "annolid.domain.timelines",
    "CollectedPair": "annolid.domain.datasets",
    "Config": "annolid.domain.datasets",
    "DeepLabCutTrainingImportConfig": "annolid.domain.datasets",
    "Episode": "annolid.domain.behavior_agent",
    "EthogramIssue": "annolid.domain.timelines",
    "EthogramQCReport": "annolid.domain.timelines",
    "FilePaths": "annolid.domain.datasets",
    "FrameRef": "annolid.domain.timelines",
    "IRRReport": "annolid.domain.timelines",
    "InstanceRegistry": "annolid.domain.keypoints",
    "InstanceState": "annolid.domain.keypoints",
    "InstanceTrackSummary": "annolid.domain.tracks",
    "KeypointState": "annolid.domain.keypoints",
    "MemoryRecord": "annolid.domain.behavior_agent",
    "ModifierDefinition": "annolid.domain.project_schema",
    "PointBehaviorEvent": "annolid.domain.timelines",
    "ProcessingConfig": "annolid.domain.datasets",
    "ProjectSchema": "annolid.domain.project_schema",
    "ReviewBout": "annolid.domain.tracks",
    "SCHEMA_VERSION": "annolid.domain.behavior_agent",
    "SubjectDefinition": "annolid.domain.project_schema",
    "TaskPlan": "annolid.domain.behavior_agent",
    "TimeBudgetRow": "annolid.domain.timelines",
    "Track": "annolid.domain.tracks",
    "TrackArtifact": "annolid.domain.behavior_agent",
    "TrackObservation": "annolid.domain.tracks",
    "TrackQCIssue": "annolid.domain.tracks",
    "TrackQCReport": "annolid.domain.tracks",
    "TransitionEdge": "annolid.domain.timelines",
    "VideoRef": "annolid.domain.behavior_agent",
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
