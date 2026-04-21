"""Behavior-agent service layer (additive, typed orchestration surface)."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BehaviorAgentArtifactStore": "annolid.services.behavior_agent.artifact_store",
    "AGGRESSION_SUBEVENT_TYPES": "annolid.services.behavior_agent.bout_scoring",
    "AggressionBoutCount": "annolid.services.behavior_agent.bout_scoring",
    "aggregate_aggression_bout_counts": "annolid.services.behavior_agent.bout_scoring",
    "normalize_aggression_sub_event_type": "annolid.services.behavior_agent.bout_scoring",
    "validate_aggression_bout_counts": "annolid.services.behavior_agent.bout_scoring",
    "TaskInferencer": "annolid.services.behavior_agent.interfaces",
    "PerceptionAdapter": "annolid.services.behavior_agent.interfaces",
    "BehaviorSegmenter": "annolid.services.behavior_agent.interfaces",
    "MemoryStore": "annolid.services.behavior_agent.interfaces",
    "AnalysisRunner": "annolid.services.behavior_agent.interfaces",
    "AggressionSubEventSegmenter": "annolid.services.behavior_agent.defaults",
    "DeterministicAnalysisRunner": "annolid.services.behavior_agent.defaults",
    "InMemoryMemoryStore": "annolid.services.behavior_agent.defaults",
    "KeywordTaskInferencer": "annolid.services.behavior_agent.defaults",
    "NDJSONPerceptionAdapter": "annolid.services.behavior_agent.defaults",
    "PassThroughPerceptionAdapter": "annolid.services.behavior_agent.defaults",
    "BehaviorAgentPipeline": "annolid.services.behavior_agent.pipeline",
    "BehaviorAgentPipelineResult": "annolid.services.behavior_agent.pipeline",
    "BehaviorAgentOrchestrator": "annolid.services.behavior_agent.orchestrator",
    "OrchestrationResult": "annolid.services.behavior_agent.orchestrator",
    "BehaviorModelPolicy": "annolid.services.behavior_agent.model_policy",
    "DEFAULT_POLICY": "annolid.services.behavior_agent.model_policy",
    "PRIVACY_POLICY": "annolid.services.behavior_agent.model_policy",
    "resolve_behavior_model_policy": "annolid.services.behavior_agent.model_policy",
    "SpecializedBehaviorAgentPipeline": "annolid.services.behavior_agent.specialized_pipeline",
    "SpecializedBehaviorPipelineResult": "annolid.services.behavior_agent.specialized_pipeline",
    "BehaviorAgentBenchmarkRow": "annolid.services.behavior_agent.evaluation",
    "BehaviorAgentBenchmarkSummary": "annolid.services.behavior_agent.evaluation",
    "summarize_behavior_agent_benchmarks": "annolid.services.behavior_agent.evaluation",
    "build_default_behavior_agent_pipeline": "annolid.services.behavior_agent.runtime",
    "resolve_behavior_results_root": "annolid.services.behavior_agent.runtime",
    "run_default_behavior_agent_pipeline": "annolid.services.behavior_agent.runtime",
    "BehaviorSubagentProfile": "annolid.services.behavior_agent.subagents",
    "list_behavior_subagent_profiles": "annolid.services.behavior_agent.subagents",
    "resolve_behavior_subagent_profile": "annolid.services.behavior_agent.subagents",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
