"""Dataset-oriented domain types and import configs."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CollectedPair": "annolid.datasets.labelme_collection",
    "Config": "annolid.datasets.video_clips",
    "DeepLabCutTrainingImportConfig": (
        "annolid.datasets.importers.deeplabcut_training_data"
    ),
    "FilePaths": "annolid.datasets.video_clips",
    "ProcessingConfig": "annolid.datasets.video_clips",
    "VideoFrameDataset": "annolid.datasets.video_dataset",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
