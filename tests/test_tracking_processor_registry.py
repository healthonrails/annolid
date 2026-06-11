from __future__ import annotations

import sys
import subprocess
import uuid

from annolid.segmentation.cutie_vos.runtime import (
    is_cutie_model_name,
    register_tracking_backend,
    resolve_tracking_video_processor_class,
)
from annolid.tracker.processor_registry import (
    INSID3_BACKEND,
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


def test_default_backend_resolution_does_not_import_cutie_runtime() -> None:
    sys.modules.pop("annolid.segmentation.cutie_vos.predict", None)
    sys.modules.pop("annolid.segmentation.cutie_vos.dependencies", None)

    resolved = resolve_tracking_video_processor_class("default")

    assert resolved.__name__ == "VideoProcessor"
    assert "annolid.segmentation.cutie_vos.predict" not in sys.modules
    assert "annolid.segmentation.cutie_vos.dependencies" not in sys.modules


def test_cutie_backend_auto_installs_missing_runtime_before_import(
    monkeypatch,
) -> None:
    for module_name in [
        "annolid.segmentation.cutie_vos.video_processor",
        "annolid.segmentation.cutie_vos.predict",
    ]:
        sys.modules.pop(module_name, None)

    from annolid.segmentation.cutie_vos import dependencies

    installed = set()
    calls = []
    real_find_spec = dependencies.importlib.util.find_spec

    def _find_spec(name: str):
        if name == "torch" and name not in installed:
            return None
        return real_find_spec(name) or object()

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        installed.add("torch")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dependencies.importlib.util, "find_spec", _find_spec)
    monkeypatch.setattr(dependencies.subprocess, "run", _run)

    resolved = resolve_tracking_video_processor_class("Cutie")

    assert resolved.__name__ == "CutieVideoProcessor"
    assert calls
    assert any(str(part).startswith("torch>=") for part in calls[0][0])
    assert "annolid.segmentation.cutie_vos.predict" in sys.modules


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


def test_runtime_insid3_backend_is_registered() -> None:
    resolved = resolve_tracking_video_processor_class("insid3_video")
    assert resolved.__name__ == "Insid3VideoProcessor"
    assert INSID3_BACKEND == "insid3"
