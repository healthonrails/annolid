from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class AgentPipelineRequest:
    """Normalized request for the shared agent orchestration workflow."""

    video_path: str
    behavior_spec_path: Optional[str] = None
    results_dir: Optional[str] = None
    out_ndjson_name: str = "agent.ndjson"
    max_frames: Optional[int] = None
    stride: int = 1
    include_llm_summary: bool = False
    llm_summary_prompt: str = "Summarize the behaviors defined in this behavior spec."
    vision_adapter: str = "none"
    vision_pretrained: bool = False
    vision_score_threshold: float = 0.5
    vision_device: Optional[str] = None
    vision_weights: Optional[str] = None
    llm_adapter: str = "none"
    llm_profile: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_persist: bool = False
    reuse_cache: bool = True


def _build_vision_model(request: AgentPipelineRequest):
    adapter = str(request.vision_adapter or "").strip().lower()
    if adapter in {"", "none"}:
        return None
    if adapter == "maskrcnn":
        from annolid.core.models.adapters.maskrcnn_torchvision import (
            TorchvisionMaskRCNNAdapter,
        )

        return TorchvisionMaskRCNNAdapter(
            pretrained=bool(request.vision_pretrained),
            score_threshold=float(request.vision_score_threshold),
            device=str(request.vision_device) if request.vision_device else None,
        )
    if adapter == "dino_kpseg":
        from annolid.core.models.adapters.dino_kpseg_adapter import DinoKPSEGAdapter

        if not request.vision_weights:
            raise ValueError("DinoKPSEG adapter requires a weights path.")
        weight_path = str(Path(request.vision_weights).expanduser())
        return DinoKPSEGAdapter(
            weight_path=weight_path,
            device=str(request.vision_device) if request.vision_device else None,
            score_threshold=float(request.vision_score_threshold),
        )
    raise ValueError(f"Unsupported vision adapter: {request.vision_adapter!r}")


def _build_llm_model(request: AgentPipelineRequest):
    adapter = str(request.llm_adapter or "").strip().lower()
    if adapter in {"", "none"}:
        return None
    if adapter == "llm_chat":
        from annolid.core.models.adapters.llm_chat import LLMChatAdapter

        return LLMChatAdapter(
            profile=str(request.llm_profile) if request.llm_profile else None,
            provider=str(request.llm_provider) if request.llm_provider else None,
            model=str(request.llm_model) if request.llm_model else None,
            persist=bool(request.llm_persist),
        )
    raise ValueError(f"Unsupported LLM adapter: {request.llm_adapter!r}")


def run_agent_pipeline(
    request: AgentPipelineRequest,
    *,
    progress_callback: Optional[Callable[[int, int, Optional[int]], None]] = None,
    stop_event: Optional[object] = None,
):
    """Run agent inference through a single service used by GUI/CLI/bot layers."""
    from annolid.core.agent.runner import AgentRunConfig
    from annolid.core.agent.service import run_agent_to_results

    config = AgentRunConfig(
        max_frames=request.max_frames,
        stride=int(request.stride),
        include_llm_summary=bool(request.include_llm_summary),
        llm_summary_prompt=str(request.llm_summary_prompt),
    )
    return run_agent_to_results(
        video_path=request.video_path,
        behavior_spec_path=request.behavior_spec_path,
        results_dir=request.results_dir,
        out_ndjson_name=str(request.out_ndjson_name),
        vision_model=_build_vision_model(request),
        llm_model=_build_llm_model(request),
        config=config,
        progress_callback=progress_callback,
        stop_event=stop_event,
        reuse_cache=bool(request.reuse_cache),
    )
