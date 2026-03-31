"""SAM3 inference wrapper package housed under annolid.segmentation.SAM.

Keep package imports lightweight so test collection can import submodules
without forcing optional SAM3 runtime dependencies up front.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AgentConfig": "annolid.segmentation.SAM.sam3.agent_video_orchestrator",
    "TrackingConfig": "annolid.segmentation.SAM.sam3.agent_video_orchestrator",
    "run_agent_seeded_sam3_video": "annolid.segmentation.SAM.sam3.agent_video_orchestrator",
    "process_video_with_agent": "annolid.segmentation.SAM.sam3.adapter",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
