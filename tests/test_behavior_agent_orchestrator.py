from __future__ import annotations

from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact
from annolid.services.behavior_agent.orchestrator import BehaviorAgentOrchestrator
from annolid.services.behavior_agent.interfaces import (
    AnalysisRunner,
    BehaviorSegmenter,
    MemoryStore,
    PerceptionAdapter,
    TaskInferencer,
)


class _Inferencer(TaskInferencer):
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def infer(self, video, context=None) -> TaskPlan:
        self.calls.append((video, context))
        return TaskPlan(assay_type="aggression", confidence=0.9)


class _Perception(PerceptionAdapter):
    def __init__(self) -> None:
        self.calls: list[tuple[object, TaskPlan]] = []

    def run(self, video, plan: TaskPlan) -> list[TrackArtifact]:
        self.calls.append((video, plan))
        return [TrackArtifact(artifact_id="a1", frame_index=1, label="slap_face")]


class _Segmenter(BehaviorSegmenter):
    def __init__(self) -> None:
        self.calls: list[tuple[TaskPlan, list[TrackArtifact]]] = []

    def segment(
        self, plan: TaskPlan, artifacts: list[TrackArtifact]
    ) -> list[BehaviorSegment]:
        self.calls.append((plan, artifacts))
        return [
            BehaviorSegment(
                segment_id="s1",
                label="aggression_bout",
                start_frame=1,
                end_frame=1,
            )
        ]


class _Memory(MemoryStore):
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[dict]]] = []

    def upsert(self, namespace: str, records: list[dict]) -> None:
        self.upserts.append((namespace, list(records)))

    def search(self, namespace: str, query: str, top_k: int = 5) -> list[dict]:
        _ = (namespace, query, top_k)
        return []


class _Analysis(AnalysisRunner):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate_code(self, plan: TaskPlan, inputs: dict) -> str:
        _ = plan
        return "def run(inputs):\n    return inputs\n"

    def execute(self, code: str, inputs: dict) -> dict:
        self.calls.append((code, dict(inputs)))
        return {"ok": True, "inputs": dict(inputs)}


def test_orchestrator_runs_services_without_domain_business_logic() -> None:
    inferencer = _Inferencer()
    perception = _Perception()
    segmenter = _Segmenter()
    memory = _Memory()
    analysis = _Analysis()

    orchestrator = BehaviorAgentOrchestrator(
        task_inferencer=inferencer,
        perception_adapter=perception,
        behavior_segmenter=segmenter,
        memory_store=memory,
        analysis_runner=analysis,
    )

    result = orchestrator.run(
        video="/tmp/video.mp4",
        context={"prompt": "score aggression"},
        memory_namespace="behavior_agent",
        memory_records=[{"key": "assay_type", "value": "aggression"}],
        analysis_inputs={"artifact_count": 1},
    )

    assert result.task_plan.assay_type == "aggression"
    assert len(result.artifacts) == 1
    assert len(result.segments) == 1
    assert result.analysis_code
    assert result.analysis_result == {"ok": True, "inputs": {"artifact_count": 1}}

    assert len(inferencer.calls) == 1
    assert len(perception.calls) == 1
    assert len(segmenter.calls) == 1
    assert memory.upserts == [
        ("behavior_agent", [{"key": "assay_type", "value": "aggression"}])
    ]
    assert len(analysis.calls) == 1
