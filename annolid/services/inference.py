"""Service-layer entry points for inference workflows."""

from __future__ import annotations

from typing import Any


def predict_behavior(*args: Any, **kwargs: Any):
    from annolid.behavior.inference import predict

    return predict(*args, **kwargs)


def run_behavior_inference_cli() -> None:
    from annolid.behavior.inference import main

    main()


def run_behavior_video_agent(video_path: str, user_prompt: str, agent=None):
    from annolid.agents.behavior_agent import (
        initialize_agent,
        process_video_with_agent,
    )

    resolved_agent = agent if agent is not None else initialize_agent()
    return process_video_with_agent(video_path, user_prompt, resolved_agent)


__all__ = [
    "predict_behavior",
    "run_behavior_inference_cli",
    "run_behavior_video_agent",
]
