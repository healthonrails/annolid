from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from annolid.tracker.processor_registry import resolve_tracking_processor_class
from annolid.tracker.point_tracking_processor import BasePointTrackingProcessor
from annolid.tracker.tapnext_onnx.track import (
    TAPNEXT_ONNX_FILE_NAME,
    TAPNEXT_ONNX_RELATIVE_PATH,
    TAPNEXT_ONNX_SHA256,
    TAPNEXT_ONNX_URL,
    TapNextOnnxProcessor,
    _select_tapnext_providers,
)


def _processor() -> TapNextOnnxProcessor:
    processor = TapNextOnnxProcessor.__new__(TapNextOnnxProcessor)
    processor.video_width = 640
    processor.video_height = 480
    processor._input_width = 256
    processor._input_height = 256
    processor.visibility_threshold = 0.0
    return processor


def test_tapnext_routes_from_model_name() -> None:
    assert resolve_tracking_processor_class("TAPNext") is TapNextOnnxProcessor
    assert (
        resolve_tracking_processor_class("downloads/tapnext.onnx")
        is TapNextOnnxProcessor
    )


def test_tapnext_uses_shared_point_tracking_base() -> None:
    assert issubclass(TapNextOnnxProcessor, BasePointTrackingProcessor)
    assert TapNextOnnxProcessor.supports_online is True


def test_format_queries_scales_to_model_resolution() -> None:
    processor = _processor()
    processor._query_input = type("Spec", (), {"shape": [1, 3, 3], "rank": 3})()
    queries = torch.tensor([[12.0, 320.0, 120.0]], dtype=torch.float32)

    formatted = processor._format_queries(
        queries,
        chunk_start_frame=10,
        clip_len=8,
        query_capacity=3,
    )

    assert formatted.shape == (1, 3, 3)
    # TAPNext expects [t, y, x], with Annolid [x, y] scaled to model space.
    assert formatted.tolist() == [
        [
            [2.0, 64.0, 128.0],
            [2.0, 64.0, 128.0],
            [2.0, 64.0, 128.0],
        ]
    ]


def test_extract_clip_tracks_scales_back_to_video_resolution() -> None:
    processor = _processor()
    processor._track_output = type("Spec", (), {"name": "tracks"})()
    processor._visibility_output = type("Spec", (), {"name": "visible_logits"})()

    outputs = {
        "tracks": np.array(
            [[[[128.0, 128.0], [64.0, 32.0]], [[96.0, 64.0], [32.0, 16.0]]]],
            dtype=np.float32,
        ),
        "visible_logits": np.array([[[1.0], [-1.0]], [[2.0], [3.0]]], dtype=np.float32)[
            None
        ],
    }

    tracks, visibility = processor._extract_clip_tracks(
        outputs,
        point_count=2,
        frame_count=2,
    )

    assert tracks.tolist() == [
        [[320.0, 240.0], [80.0, 120.0]],
        [[160.0, 180.0], [40.0, 60.0]],
    ]
    assert visibility.tolist() == [[True, False], [True, True]]


def test_query_batches_split_fixed_query_capacity() -> None:
    processor = _processor()
    processor._query_capacity = 2
    queries = torch.zeros((5, 3), dtype=torch.float32)

    batches = processor._query_batches(queries)

    assert [batch.shape[0] for batch in batches] == [2, 2, 1]


def test_provider_selection_keeps_coreml_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    assert _select_tapnext_providers(available) == ["CPUExecutionProvider"]

    monkeypatch.setenv("ANNOLID_TAPNEXT_USE_COREML", "1")
    assert _select_tapnext_providers(available) == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_online_processing_saves_incrementally_and_reports_progress() -> None:
    class FakeVideoLoader:
        def total_frames(self) -> int:
            return 4

        def load_frame(self, frame_number: int) -> np.ndarray:
            return np.full((4, 4, 3), frame_number, dtype=np.uint8)

    class FakeWorker:
        def __init__(self) -> None:
            self.progress: list[int] = []

        def is_stopped(self) -> bool:
            return False

        def report_progress(self, value: int) -> None:
            self.progress.append(int(value))

    processor = _processor()
    processor.video_loader = FakeVideoLoader()
    processor.start_frame = 0
    processor.end_frame = 3
    processor.device = torch.device("cpu")
    processor.queries = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    processor._temporal_window = 2
    processor.pred_worker = FakeWorker()
    processor._stop_triggered = False
    processor._should_stop_callback = lambda: False
    processor._ensure_model = lambda: object()
    chunk_starts: list[int] = []
    query_x: list[float] = []

    def _run_query_batches(_session, frames, *, queries, chunk_start_frame):
        chunk_starts.append(int(chunk_start_frame))
        query_x.append(float(queries[0, 1].item()))
        tracks = np.zeros((len(frames), 1, 2), dtype=np.float32)
        tracks[:, 0, 0] = np.arange(len(frames), dtype=np.float32) + chunk_start_frame
        tracks[:, 0, 1] = 2.0
        return tracks, np.ones((len(frames), 1), dtype=bool)

    processor._run_query_batches = _run_query_batches
    saved_frames: list[int] = []
    processor.save_frame_json = (
        lambda frame_number, _points, _description="": saved_frames.append(frame_number)
    )

    message = processor._process_video_online(
        grid_size=10,
        grid_query_frame=0,
        need_visualize=False,
    )

    assert saved_frames == [0, 1, 2, 3]
    assert chunk_starts == [0, 1, 2]
    assert query_x == [1.0, 1.0, 2.0]
    assert processor.pred_worker.progress == [50, 75, 100]
    assert message == "Completed. Saved frames 0-3"


def test_resolve_model_path_accepts_existing_onnx(tmp_path: Path) -> None:
    model = tmp_path / "custom_tapnext.onnx"
    model.write_bytes(b"onnx")

    processor = TapNextOnnxProcessor.__new__(TapNextOnnxProcessor)

    assert processor._resolve_model_path(str(model)) == model


def test_resolve_model_path_downloads_official_tapnext_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = TapNextOnnxProcessor.__new__(TapNextOnnxProcessor)
    cached = tmp_path / "downloads" / "tapnext.onnx"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"onnx")
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        "annolid.tracker.tapnext_onnx.track.resolve_existing_model_path",
        lambda *_args, **_kwargs: None,
    )

    def _fake_ensure_cached_model_asset(**kwargs):
        calls.append(kwargs)
        return cached

    monkeypatch.setattr(
        "annolid.tracker.tapnext_onnx.track.ensure_cached_model_asset",
        _fake_ensure_cached_model_asset,
    )

    assert processor._resolve_model_path("TAPNext") == cached
    assert calls == [
        {
            "file_name": TAPNEXT_ONNX_FILE_NAME,
            "url": TAPNEXT_ONNX_URL,
            "expected_sha256": TAPNEXT_ONNX_SHA256,
        }
    ]


def test_resolve_model_path_verifies_default_cache_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = TapNextOnnxProcessor.__new__(TapNextOnnxProcessor)
    stale_cached = tmp_path / ".annolid" / "workspace" / TAPNEXT_ONNX_RELATIVE_PATH
    official_cached = tmp_path / "downloads" / "tapnext.onnx"
    stale_cached.parent.mkdir(parents=True, exist_ok=True)
    official_cached.parent.mkdir(parents=True, exist_ok=True)
    stale_cached.write_bytes(b"stale")
    official_cached.write_bytes(b"official")
    calls: list[dict[str, str]] = []

    monkeypatch.setattr(
        "annolid.tracker.tapnext_onnx.track.resolve_existing_model_path",
        lambda *_args, **_kwargs: stale_cached,
    )

    def _fake_ensure_cached_model_asset(**kwargs):
        calls.append(kwargs)
        return official_cached

    monkeypatch.setattr(
        "annolid.tracker.tapnext_onnx.track.ensure_cached_model_asset",
        _fake_ensure_cached_model_asset,
    )

    assert processor._resolve_model_path(TAPNEXT_ONNX_RELATIVE_PATH) == official_cached
    assert calls == [
        {
            "file_name": TAPNEXT_ONNX_FILE_NAME,
            "url": TAPNEXT_ONNX_URL,
            "expected_sha256": TAPNEXT_ONNX_SHA256,
        }
    ]
