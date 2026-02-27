from __future__ import annotations

import uuid

from annolid.segmentation.cutie_vos.runtime import (
    is_cutie_model_name,
    register_tracking_backend,
    resolve_tracking_video_processor_class,
)
from annolid.tracker.processor_registry import (
    TrackingBackendSpec,
    TrackingProcessorRegistry,
    VIDEOMT_BACKEND,
    model_name_contains,
)


class _ProcessorA:
    pass


class _ProcessorB:
    pass


class _FallbackProcessor:
    pass


def test_registry_resolves_in_registration_order() -> None:
    registry = TrackingProcessorRegistry()
    registry.register(
        TrackingBackendSpec(
            name="a",
            predicate=model_name_contains("tracker"),
            loader=lambda: _ProcessorA,
        )
    )
    registry.register(
        TrackingBackendSpec(
            name="b",
            predicate=model_name_contains("tracker"),
            loader=lambda: _ProcessorB,
        )
    )
    registry.register(
        TrackingBackendSpec(
            name="default",
            predicate=lambda _: True,
            loader=lambda: _FallbackProcessor,
        )
    )

    resolved = registry.resolve_processor_class("my-tracker")
    assert resolved is _ProcessorA


def test_runtime_cutie_detection_uses_registry() -> None:
    assert is_cutie_model_name("Cutie")
    assert is_cutie_model_name("my-cutie-model")
    assert not is_cutie_model_name("cowtracker")


def test_runtime_backend_registration_supports_extensions() -> None:
    class _CustomProcessor:
        pass

    token = f"customtracker-{uuid.uuid4().hex}"
    register_tracking_backend(
        name=f"test-{token}",
        predicate=model_name_contains(token),
        loader=lambda: _CustomProcessor,
    )

    resolved = resolve_tracking_video_processor_class(f"prefix-{token}-suffix")
    assert resolved is _CustomProcessor


def test_runtime_videomt_backend_is_registered() -> None:
    resolved = resolve_tracking_video_processor_class("videomt")
    assert resolved.__name__ == "VideoMTOnnxVideoProcessor"
    assert VIDEOMT_BACKEND == "videomt"
