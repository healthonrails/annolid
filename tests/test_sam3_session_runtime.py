from __future__ import annotations

from types import SimpleNamespace

import torch

from annolid.segmentation.SAM.sam3.session_runtime import Sam3SessionRuntime


class _FakePredictor:
    def __init__(self) -> None:
        self.start_calls = 0
        self.reset_calls = 0
        self.close_calls = 0
        self.cancel_calls = 0

    def start_session(
        self, *, resource_path: str, offload_video_to_cpu: bool, session_id=None
    ):
        self.start_calls += 1
        return {"session_id": session_id or f"sess-{self.start_calls}"}

    def reset_session(self, _session_id: str):
        self.reset_calls += 1
        return {}

    def close_session(self, _session_id: str):
        self.close_calls += 1
        return {}

    def cancel_propagation(self, *, session_id: str):
        self.cancel_calls += 1
        return {"session_id": session_id}


def test_session_runtime_start_close_and_reset_lifecycle() -> None:
    created = {"count": 0}
    predictor = _FakePredictor()
    activated: list[str | None] = []

    def _init(_device: torch.device):
        created["count"] += 1
        return predictor

    runtime = Sam3SessionRuntime(
        default_device="cpu",
        offload_video_to_cpu=True,
        initialize_predictor=_init,
        activate_global_match_session=lambda sid: activated.append(sid),
    )

    session_id = runtime.start_session(
        target_device="cpu",
        resource_path="/tmp/video",
        session_id="window-1",
    )
    assert session_id == "window-1"
    assert created["count"] == 1
    assert activated[-1] == "window-1"

    runtime.reset_session_state()
    assert predictor.reset_calls == 1

    runtime.cancel_propagation()
    assert predictor.cancel_calls == 1

    runtime.close_session()
    assert predictor.close_calls == 1
    assert runtime.session_id is None
    assert activated[-1] is None


def test_session_runtime_reuses_predictor_for_same_device() -> None:
    created = {"count": 0}
    predictor = _FakePredictor()

    def _init(_device: torch.device):
        created["count"] += 1
        return predictor

    runtime = Sam3SessionRuntime(
        default_device="cpu",
        offload_video_to_cpu=True,
        initialize_predictor=_init,
        activate_global_match_session=lambda _sid: None,
    )
    runtime.start_session(target_device="cpu", resource_path="/tmp/a")
    runtime.start_session(target_device="cpu", resource_path="/tmp/b")
    assert created["count"] == 1


def test_session_runtime_resolves_mps_to_cpu(monkeypatch) -> None:
    runtime = Sam3SessionRuntime(
        default_device="mps",
        offload_video_to_cpu=True,
        initialize_predictor=lambda _device: SimpleNamespace(),
        activate_global_match_session=lambda _sid: None,
    )
    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.session_runtime.select_device",
        lambda _target: torch.device("mps"),
    )
    resolved = runtime.resolve_runtime_device("mps")
    assert resolved.type == "cpu"
