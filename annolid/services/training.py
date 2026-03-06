"""Service-layer entry points for training workflows."""

from __future__ import annotations

from typing import Any


def train_behavior_model(*args: Any, **kwargs: Any):
    from annolid.behavior.training.train import train_model

    return train_model(*args, **kwargs)


def validate_behavior_model(*args: Any, **kwargs: Any):
    from annolid.behavior.training.train import validate_model

    return validate_model(*args, **kwargs)


def run_behavior_training_cli() -> None:
    from annolid.behavior.training.train import main

    main()


def run_polygon_frame_training_cli() -> None:
    from annolid.behavior.training.polygon_frame_training import main

    main()


__all__ = [
    "run_behavior_training_cli",
    "run_polygon_frame_training_cli",
    "train_behavior_model",
    "validate_behavior_model",
]
