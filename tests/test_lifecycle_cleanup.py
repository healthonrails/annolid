from __future__ import annotations

from annolid.gui.mixins.lifecycle_mixin import LifecycleMixin


class _FakeThread:
    def __init__(self) -> None:
        self.quit_calls = 0
        self.wait_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1

    def wait(self) -> None:
        self.wait_calls += 1


class _FakeRealtimeManager:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop_realtime_inference(self) -> None:
        self.stop_calls += 1


class _FakeTrainingManager:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _FakeAiChatManager:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _FakeTrackingController:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _LifecycleProbe(LifecycleMixin):
    def __init__(self) -> None:
        self._cleanup_done = False
        self.frame_worker = _FakeThread()
        self.seg_train_thread = _FakeThread()
        self.seg_pred_thread = _FakeThread()
        self.realtime_manager = _FakeRealtimeManager()
        self.yolo_training_manager = _FakeTrainingManager()
        self.dino_kpseg_training_manager = _FakeTrainingManager()
        self.ai_chat_manager = _FakeAiChatManager()
        self.tracking_data_controller = _FakeTrackingController()
        self._training_dashboard_dialog = None
        self.stop_frame_loader_calls = 0
        self.stop_csv_worker_calls = 0
        self.release_audio_loader_calls = 0

    def _stop_frame_loader(self) -> None:
        self.stop_frame_loader_calls += 1

    def _stop_csv_worker(self) -> None:
        self.stop_csv_worker_calls += 1

    def _release_audio_loader(self, *, clear_pending: bool = True) -> None:
        _ = clear_pending
        self.release_audio_loader_calls += 1


def test_cleanup_stops_tracking_and_audio_workers() -> None:
    probe = _LifecycleProbe()
    probe.clean_up()

    assert probe.tracking_data_controller.shutdown_calls == 1
    assert probe.release_audio_loader_calls == 1
    assert probe.stop_frame_loader_calls == 1
    assert probe.stop_csv_worker_calls == 1
    assert probe.realtime_manager.stop_calls == 1
    assert probe.yolo_training_manager.cleanup_calls == 1
    assert probe.dino_kpseg_training_manager.cleanup_calls == 1
    assert probe.ai_chat_manager.cleanup_calls == 1
    assert probe.frame_worker.quit_calls == 1
    assert probe.frame_worker.wait_calls == 1


def test_cleanup_is_idempotent_for_tracking_shutdown() -> None:
    probe = _LifecycleProbe()
    probe.clean_up()
    probe.clean_up()
    assert probe.tracking_data_controller.shutdown_calls == 1
    assert probe.release_audio_loader_calls == 1
