from __future__ import annotations

from typing import Any, Dict, Optional

from annolid.gui.controllers.inference_controller import InferenceController


class _FakeThread:
    def __init__(self, running: bool = True) -> None:
        self._running = running
        self.cancel_called = False
        self.interruption_called = False
        self.quit_called = False
        self.wait_args: list[int] = []

    def isRunning(self) -> bool:
        return self._running

    def cancel(self) -> None:
        self.cancel_called = True

    def requestInterruption(self) -> None:
        self.interruption_called = True

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, timeout_ms: int) -> bool:
        self.wait_args.append(int(timeout_ms))
        self._running = False
        return True


class _RaisingInferenceService:
    def validate_model_config(self, _model_config: Dict[str, Any]):
        raise RuntimeError("service exploded")

    def prepare_inference_input(
        self, _model_type: str, input_data: Any, _model_config: Dict[str, Any]
    ) -> Any:
        return input_data

    def process_inference_results(
        self,
        _model_type: str,
        raw_results: Any,
        _model_config: Dict[str, Any],
        _postprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {"model_type": "x", "results": raw_results}


def test_cancel_inference_forwards_to_running_thread() -> None:
    controller = InferenceController()
    fake_thread = _FakeThread(running=True)
    controller._inference_thread = fake_thread

    controller.cancel_inference()

    assert fake_thread.cancel_called is True
    assert fake_thread.interruption_called is True


def test_shutdown_cleans_running_thread_and_resets_reference() -> None:
    controller = InferenceController()
    fake_thread = _FakeThread(running=True)
    controller._inference_thread = fake_thread

    controller.shutdown(timeout_ms=321)

    assert fake_thread.cancel_called is True
    assert fake_thread.interruption_called is True
    assert fake_thread.quit_called is True
    assert fake_thread.wait_args == [321]
    assert controller._inference_thread is None


def test_shutdown_handles_non_running_thread() -> None:
    controller = InferenceController()
    fake_thread = _FakeThread(running=False)
    controller._inference_thread = fake_thread

    controller.shutdown(timeout_ms=250)

    assert fake_thread.cancel_called is False
    assert fake_thread.interruption_called is False
    assert fake_thread.quit_called is False
    assert fake_thread.wait_args == []
    assert controller._inference_thread is None


def test_validate_model_config_propagates_service_exception_to_signal() -> None:
    controller = InferenceController(inference_service=_RaisingInferenceService())
    errors: list[str] = []
    controller.inference_error.connect(errors.append)

    is_valid, details = controller.validate_model_config({"identifier": "x"})

    assert is_valid is False
    assert details
    assert "Failed to validate model config: service exploded" in details[0]
    assert errors
    assert "Failed to validate model config: service exploded" in errors[-1]
