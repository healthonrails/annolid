"""Behavior-agent service layer (additive, typed orchestration surface)."""

from annolid.services.behavior_agent.artifact_store import BehaviorAgentArtifactStore
from annolid.services.behavior_agent.bout_scoring import (
    AGGRESSION_SUBEVENT_TYPES,
    AggressionBoutCount,
    aggregate_aggression_bout_counts,
    normalize_aggression_sub_event_type,
    validate_aggression_bout_counts,
)
from annolid.services.behavior_agent.interfaces import (
    AnalysisRunner,
    BehaviorSegmenter,
    MemoryStore,
    PerceptionAdapter,
    TaskInferencer,
)
from annolid.services.behavior_agent.defaults import (
    AggressionSubEventSegmenter,
    DeterministicAnalysisRunner,
    InMemoryMemoryStore,
    KeywordTaskInferencer,
    NDJSONPerceptionAdapter,
    PassThroughPerceptionAdapter,
)
from annolid.services.behavior_agent.pipeline import (
    BehaviorAgentPipeline,
    BehaviorAgentPipelineResult,
)
from annolid.services.behavior_agent.orchestrator import (
    BehaviorAgentOrchestrator,
    OrchestrationResult,
)
from annolid.services.behavior_agent.model_policy import (
    BehaviorModelPolicy,
    DEFAULT_POLICY,
    PRIVACY_POLICY,
    resolve_behavior_model_policy,
)
from annolid.services.behavior_agent.specialized_pipeline import (
    SpecializedBehaviorAgentPipeline,
    SpecializedBehaviorPipelineResult,
)
from annolid.services.behavior_agent.evaluation import (
    BehaviorAgentBenchmarkRow,
    BehaviorAgentBenchmarkSummary,
    summarize_behavior_agent_benchmarks,
)
from annolid.services.behavior_agent.runtime import (
    build_default_behavior_agent_pipeline,
    resolve_behavior_results_root,
    run_default_behavior_agent_pipeline,
)
from annolid.services.behavior_agent.subagents import (
    BehaviorSubagentProfile,
    list_behavior_subagent_profiles,
    resolve_behavior_subagent_profile,
)

__all__ = [
    "BehaviorAgentArtifactStore",
    "AGGRESSION_SUBEVENT_TYPES",
    "AggressionBoutCount",
    "aggregate_aggression_bout_counts",
    "normalize_aggression_sub_event_type",
    "validate_aggression_bout_counts",
    "TaskInferencer",
    "PerceptionAdapter",
    "BehaviorSegmenter",
    "MemoryStore",
    "AnalysisRunner",
    "AggressionSubEventSegmenter",
    "DeterministicAnalysisRunner",
    "InMemoryMemoryStore",
    "KeywordTaskInferencer",
    "NDJSONPerceptionAdapter",
    "PassThroughPerceptionAdapter",
    "BehaviorAgentPipeline",
    "BehaviorAgentPipelineResult",
    "BehaviorAgentOrchestrator",
    "OrchestrationResult",
    "BehaviorModelPolicy",
    "DEFAULT_POLICY",
    "PRIVACY_POLICY",
    "resolve_behavior_model_policy",
    "SpecializedBehaviorAgentPipeline",
    "SpecializedBehaviorPipelineResult",
    "BehaviorAgentBenchmarkRow",
    "BehaviorAgentBenchmarkSummary",
    "summarize_behavior_agent_benchmarks",
    "build_default_behavior_agent_pipeline",
    "resolve_behavior_results_root",
    "run_default_behavior_agent_pipeline",
    "BehaviorSubagentProfile",
    "list_behavior_subagent_profiles",
    "resolve_behavior_subagent_profile",
]
