"""Analysis service built on coding agent."""

from __future__ import annotations

from annolid.agents.coding_agent import AnalysisCodingAgent
from annolid.domain.behavior_agent import TaskPlan
from annolid.services.behavior_agent.interfaces import AnalysisRunner


class CodingAnalysisService(AnalysisRunner):
    def __init__(self, *, coding_agent: AnalysisCodingAgent | None = None) -> None:
        self._coding_agent = coding_agent or AnalysisCodingAgent()

    def generate_code(self, plan: TaskPlan, inputs: dict) -> str:
        # Keep this method deterministic and tiny; execution uses the paired service call.
        _ = inputs
        return self._coding_agent.generate_code(plan)

    def execute(self, code: str, inputs: dict) -> dict:
        return self._coding_agent.execute_code(str(code), dict(inputs))


__all__ = ["CodingAnalysisService"]
