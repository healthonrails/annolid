from __future__ import annotations

from pathlib import Path

import annolid.gui.widgets.video_rescale_worker as worker_mod


def test_video_rescale_worker_emits_progress_and_summary(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _fake_compress(*args, **kwargs):
        _ = args
        _ = kwargs
        return {"clip.mp4": "ffmpeg -y -i clip.mp4"}

    save_calls: list[dict] = []

    def _fake_save(*args, **kwargs):
        _ = args
        save_calls.append(dict(kwargs))

    monkeypatch.setattr(worker_mod, "compress_and_rescale_video", _fake_compress)
    monkeypatch.setattr(worker_mod, "save_processing_summary", _fake_save)

    job = worker_mod.VideoRescaleJob(
        selected_videos=[str(video)],
        input_mode="single video",
        input_source=str(video),
        input_folder=str(tmp_path),
        output_folder=str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        collect_only=False,
        rescale=True,
        apply_denoise=False,
        auto_contrast=False,
        auto_contrast_strength=1.0,
        crop_params=None,
    )
    worker = worker_mod.VideoRescaleWorker(job)
    progress_events: list[tuple[int, str]] = []
    finished_results: list[dict] = []
    worker.progress.connect(
        lambda value, message: progress_events.append((value, message))
    )
    worker.finished.connect(lambda result: finished_results.append(result))

    worker.run()

    assert progress_events[0][1] == "Starting"
    assert progress_events[-1][1] == "Complete"
    assert finished_results
    assert "Downsample Parameters" in finished_results[0]["summary"]
    assert finished_results[0]["success_count"] == 1
    assert len(save_calls) == 1
    assert save_calls[0]["video_paths"] == [str(output_dir / "clip.mp4")]


def test_video_rescale_worker_cancel_request_emits_canceled(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    def _fake_compress(*args, **kwargs):
        _ = args
        assert kwargs["cancel_callback"]()
        raise RuntimeError("Processing cancelled by user.")

    monkeypatch.setattr(worker_mod, "compress_and_rescale_video", _fake_compress)
    monkeypatch.setattr(worker_mod, "save_processing_summary", lambda *a, **k: None)

    job = worker_mod.VideoRescaleJob(
        selected_videos=[str(video)],
        input_mode="single video",
        input_source=str(video),
        input_folder=str(tmp_path),
        output_folder=str(tmp_path / "output"),
        scale_factor=0.5,
        fps=15.0,
        collect_only=False,
        rescale=True,
        apply_denoise=False,
        auto_contrast=False,
        auto_contrast_strength=1.0,
        crop_params=None,
    )
    worker = worker_mod.VideoRescaleWorker(job)
    canceled: list[bool] = []
    failed: list[str] = []
    worker.canceled.connect(lambda: canceled.append(True))
    worker.failed.connect(lambda err: failed.append(err))
    worker.cancel()
    worker.run()

    assert canceled == [True]
    assert failed == []
