# Behavior-Agent Architecture

This document consolidates the behavior-agent architecture into one stable
reference, replacing the previous phase-numbered notes.

## Goal

Provide a typed, replayable behavior-analysis surface that remains additive to
existing Annolid GUI/CLI labeling and export contracts.

Core flow:

`task inference -> perception -> behavior segmentation -> memory -> analysis/report -> immutable artifacts`

## Typed contracts

Defined in `annolid/domain/behavior_agent.py`:

- `Episode`
- `TaskPlan`
- `TrackArtifact`
- `BehaviorSubEvent`
- `BehaviorSegment`
- `AnalysisRun`
- `MemoryRecord`

## Service interfaces

Defined in `annolid/services/behavior_agent/interfaces.py`:

- `TaskInferencer`
- `PerceptionAdapter`
- `BehaviorSegmenter`
- `MemoryStore`
- `AnalysisRunner`

## Reusable runtime wiring

To avoid duplicated endpoint-specific setup logic, CLI and GUI now reuse shared
runtime helpers in `annolid/services/behavior_agent/runtime.py`:

- `resolve_behavior_results_root`
- `build_default_behavior_agent_pipeline`
- `run_default_behavior_agent_pipeline`

This is the default wiring used by:

- `annolid-run agent-behavior`
- `gui_score_aggression_bouts` backend handler

## Default pipeline components

Default implementations live in `annolid/services/behavior_agent/defaults.py`:

- `KeywordTaskInferencer`
- `PassThroughPerceptionAdapter`
- `NDJSONPerceptionAdapter`
- `AggressionSubEventSegmenter`
- `InMemoryMemoryStore`
- `DeterministicAnalysisRunner`

## Aggression bout scoring

Deterministic aggregation and validation:

- `annolid/services/behavior_agent/bout_scoring.py`

Canonical sub-events:

- `slap_face`
- `run_away`
- `fight_initiation`

## Immutable artifact store

`BehaviorAgentArtifactStore` writes immutable run outputs:

`<results-root>/analysis_runs/<run_id>/`

Key outputs:

- `manifest.json`
- `artifacts/tracks.ndjson`
- `artifacts/behaviors.ndjson`
- `artifacts/memory.ndjson`
- `artifacts/metrics.ndjson` (baseline)
- `artifacts/metrics.parquet` (optional)
- optional report/code/evidence artifacts

## Specialized behavior agents

Specialized components in `annolid/agents/*`:

- assay inference
- feature planning
- perception routing
- segmentation
- analysis coding
- reporting

Orchestrated path:

- `annolid/services/behavior_agent/specialized_pipeline.py`

## Bot and subagent integration

GUI tool path for bout scoring:

- function tool: `gui_score_aggression_bouts`
- backend handler: `_tool_gui_score_aggression_bouts`
- service wrapper: `score_chat_aggression_bouts_tool`
- workflow handler: `score_aggression_bouts_tool`

Behavior-aware delegation profiles for runtime subagents:

- `behavior_assay_inference`
- `behavior_feature_planning`
- `behavior_perception_routing`
- `behavior_segmentation`
- `behavior_analysis_coding`
- `behavior_reporting`

These profiles are defined in `annolid/services/behavior_agent/subagents.py` and
invoked through `spawn_behavior_subagent`.

## Compatibility

- Existing labeling/export workflows are unchanged.
- Existing `segment_track_video`, `label_behavior_segments`, and
  `process_video_behaviors` flows are unchanged.
- Behavior-agent capabilities remain additive and opt-in.
