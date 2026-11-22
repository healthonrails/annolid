from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import PIL.Image
import pytest
from qtpy import QtCore, QtGui, QtWidgets

import annolid.gui.mixins.annotation_loading_mixin as annotation_loading_module
from annolid.core.media.video import CV2Video
from annolid.data.videos import TiffStackVideo, extract_frames_from_video
from annolid.gui.label_file import LabelFile
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.mixins.behavior_interaction_mixin import BehaviorInteractionMixin
from annolid.gui.mixins.frame_playback_mixin import FramePlaybackMixin
from annolid.gui.mixins.video_workflow_mixin import VideoWorkflowMixin
from annolid.gui.services.video_service import VideoService
from annolid.gui.widgets.boris_sync_dock import SecondaryVideoTile
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget
from annolid.gui.widgets.video_slider import VideoSlider
from annolid.gui.workers import LoadFrameThread, _array_to_qimage
from annolid.infrastructure import AnnotationStore
from annolid.postprocessing.zone_schema import build_zone_shape
from annolid.utils.qt2cv import convert_cv_image_to_qt_image, numpy_to_qimage


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _write_indexed_video(path: Path, *, fps: float = 10.0, frames: int = 24) -> None:
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(int(frames)):
            # Encode distinct constant RGB values per frame (written as BGR).
            frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            frame_bgr[..., 2] = min(255, idx * 10)  # Red channel in RGB
            frame_bgr[..., 1] = min(255, 250 - idx * 10)  # Green channel in RGB
            frame_bgr[..., 0] = min(255, (idx * 7) % 250)  # Blue channel in RGB
            writer.write(frame_bgr)
    finally:
        writer.release()


def _write_multipage_tiff(path: Path, *, pages: int = 5) -> None:
    images = []
    for idx in range(pages):
        arr = np.full((16, 20, 3), idx * 40, dtype=np.uint8)
        images.append(PIL.Image.fromarray(arr, mode="RGB"))
    images[0].save(str(path), save_all=True, append_images=images[1:])


# ---------------------------------------------------------------------------
# CV2Video & TiffStackVideo tests
# ---------------------------------------------------------------------------


def test_cv2video_forward_grab_skip_matches_sequential_and_avoids_seeks(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "indexed.avi"
    _write_indexed_video(video_path, fps=10.0, frames=20)

    with CV2Video(video_path) as ref:
        reference_frames = {i: ref.load_frame(i) for i in range(20)}

    with CV2Video(video_path) as video:
        real_cap = video.cap
        set_calls: list[tuple[int, object]] = []
        grab_calls = {"count": 0}

        class _TrackingCap:
            def __getattr__(self, name):
                return getattr(real_cap, name)

            def set(self, prop, value):
                set_calls.append((prop, value))
                return real_cap.set(prop, value)

            def grab(self):
                grab_calls["count"] += 1
                return real_cap.grab()

        video.cap = _TrackingCap()  # type: ignore[assignment]

        # Frame 0 after open should not call cap.set()
        f0 = video.load_frame(0)
        assert np.array_equal(f0, reference_frames[0])
        assert set_calls == []

        # Stride +5 forward should advance via grab() with zero seeks
        f5 = video.load_frame(5)
        f10 = video.load_frame(10)
        assert np.array_equal(f5, reference_frames[5])
        assert np.array_equal(f10, reference_frames[10])
        assert set_calls == []
        assert grab_calls["count"] == 8  # 4 grabs (1..4) + 4 grabs (6..9)

        # Backward jump must seek
        f2 = video.load_frame(2)
        assert np.array_equal(f2, reference_frames[2])
        assert len(set_calls) == 1


def test_cv2video_thread_safe_concurrent_reads_and_release_guard(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "concurrent.avi"
    _write_indexed_video(video_path, fps=10.0, frames=15)

    video = CV2Video(video_path)
    errors: list[Exception] = []

    def _worker(indices: list[int]) -> None:
        try:
            for idx in indices:
                frame = video.load_frame(idx)
                expected_red = idx * 10
                actual_red = int(np.median(frame[..., 0]))
                assert abs(actual_red - expected_red) <= 5
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=([0, 3, 6, 9, 12],)),
        threading.Thread(target=_worker, args=([1, 4, 7, 10, 13],)),
        threading.Thread(target=_worker, args=([2, 5, 8, 11, 14],)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    video.release()
    with pytest.raises(RuntimeError, match="released"):
        video.load_frame(0)


def test_cv2video_invalidates_position_on_read_failure_and_synthesizes_timestamp(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "ts.avi"
    _write_indexed_video(video_path, fps=20.0, frames=8)

    with CV2Video(video_path) as video:
        real_cap = video.cap

        class _ZeroMsecCap:
            def __init__(self) -> None:
                self.fail_reads = 0
                self.set_calls: list[int] = []

            def __getattr__(self, name):
                return getattr(real_cap, name)

            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_MSEC:
                    return 0.0
                return real_cap.get(prop)

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.set_calls.append(int(value))
                return real_cap.set(prop, value)

            def read(self):
                if self.fail_reads > 0:
                    self.fail_reads -= 1
                    return False, None
                return real_cap.read()

        wrapper = _ZeroMsecCap()
        video.cap = wrapper  # type: ignore[assignment]

        video.load_frame(0)
        video.load_frame(4)
        # 4 / 20fps * 1000ms = 200ms synthesized when POS_MSEC returns 0.0
        assert video.get_time_stamp() == pytest.approx(200.0)

        # Persistent read failure (both sequential read and explicit seek retry fail)
        # raises KeyError and invalidates stream position.
        wrapper.fail_reads = 2
        with pytest.raises(KeyError):
            video.load_frame(5)
        assert video._next_frame_index is None

        # Next load must seek explicitly because stream position was invalidated
        wrapper.set_calls.clear()
        video.load_frame(5)
        assert 5 in wrapper.set_calls


def test_tiff_stack_video_frame0_reuse_empty_slice_and_release(
    tmp_path: Path,
) -> None:
    tiff_path = tmp_path / "stack.tif"
    _write_multipage_tiff(tiff_path, pages=4)

    with TiffStackVideo(tiff_path, fps=2.0) as tiff:
        f0 = tiff.load_frame(0)
        f0[...] = 123
        # Cached first_frame must not be mutated by caller
        assert int(tiff.load_frame(0)[0, 0, 0]) == 0
        assert int(tiff.load_frame(2)[0, 0, 0]) == 80
        assert tiff.get_time_stamp() == pytest.approx(1000.0)

        empty = tiff.get_frames_between(10, 12)
        assert empty.shape[0] == 0

    with pytest.raises(RuntimeError, match="released"):
        tiff.load_frame(0)


def test_extract_frames_from_video_and_video_service_range_extraction(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "extract.avi"
    _write_indexed_video(video_path, fps=10.0, frames=12)

    out_dir = tmp_path / "frames"
    paths = extract_frames_from_video(
        str(video_path),
        str(out_dir),
        frame_indices=[0, 2, 5, 11],
    )
    assert len(paths) == 4
    assert all(Path(p).exists() for p in paths)

    svc = VideoService()
    frames = svc.extract_frames_range(video_path, start_frame=1, end_frame=9, step=2)
    assert len(frames) == 5
    for idx, frame in zip([1, 3, 5, 7, 9], frames):
        assert abs(int(np.median(frame[..., 0])) - idx * 10) <= 5


# ---------------------------------------------------------------------------
# Fast QImage conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [5, 7, 13, 64])
def test_numpy_to_qimage_handles_odd_widths_formats_and_detaches(width: int) -> None:
    _ensure_qapp()
    height = 7

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 33
    rgb[:, :, 1] = 144
    rgb[:, :, 2] = 211
    qimg_rgb = numpy_to_qimage(rgb)
    rgb[...] = 0
    color = qimg_rgb.pixelColor(width - 1, height - 1)
    assert (color.red(), color.green(), color.blue()) == (33, 144, 211)

    gray = np.full((height, width), 99, dtype=np.uint8)
    qimg_gray = convert_cv_image_to_qt_image(gray)
    gray[...] = 0
    assert qimg_gray.pixelColor(width - 1, height - 1).red() == 99

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :] = (10, 20, 30, 128)
    qimg_rgba = _array_to_qimage(rgba)
    rgba[...] = 0
    c4 = qimg_rgba.pixelColor(0, 0)
    assert (c4.red(), c4.green(), c4.blue(), c4.alpha()) == (10, 20, 30, 128)


# ---------------------------------------------------------------------------
# LoadFrameThread tests: FIFO ordering, cache hit queue clear, byte budget,
# video switch generation guard, idle prefetch, error reporting
# ---------------------------------------------------------------------------


def test_frame_loader_sequential_playback_is_fifo_not_lifo() -> None:
    _ensure_qapp()

    class _CountingLoader:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def load_frame(self, frame_number: int):
            self.calls.append(int(frame_number))
            return np.full((8, 8, 3), int(frame_number), dtype=np.uint8)

    worker = LoadFrameThread(prefetch_enabled=False)
    loader = _CountingLoader()
    worker.video_loader = loader

    emitted: list[int] = []
    worker.res_frame.connect(lambda idx, _img: emitted.append(int(idx)))

    # Simulate worker currently decoding frame 10 while playback ticks 11 & 12 arrive.
    with worker.working_lock:
        worker._is_processing = True
        worker._active_frame_number = 10

    worker.request(11)
    worker.request(12)

    with worker.working_lock:
        worker._is_processing = False
        worker._active_frame_number = None

    worker.load()
    worker.load()

    assert loader.calls == [11, 12]
    assert emitted == [11, 12]


def test_frame_loader_cache_hit_clears_stale_pending_queue() -> None:
    _ensure_qapp()

    class _CountingLoader:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def load_frame(self, frame_number: int):
            self.calls.append(int(frame_number))
            return np.zeros((8, 8, 3), dtype=np.uint8)

    worker = LoadFrameThread(prefetch_enabled=False)
    worker.video_loader = _CountingLoader()

    cached = QtGui.QImage(4, 4, QtGui.QImage.Format_RGB888)
    cached.fill(QtGui.QColor(1, 2, 3))
    worker._update_cache(5, cached, timestamp_ms=500.0)

    emitted: list[int] = []
    worker.res_frame.connect(lambda idx, _img: emitted.append(int(idx)))

    worker.request(20)  # queued uncached
    worker.request(5)  # cache hit must clear stale 20 from queue
    worker.load()  # should now be a no-op

    assert emitted == [5]
    assert worker.video_loader.calls == []
    assert worker.get_cached_timestamp(5) == pytest.approx(500.0)


def test_frame_loader_enforces_byte_budget_and_invalidates_on_video_switch() -> None:
    _ensure_qapp()

    class _DummyLoader:
        def __init__(self, fill_value: int) -> None:
            self.fill_value = fill_value
            self.released = False

        def load_frame(self, _frame_number: int):
            return np.full((32, 32, 3), self.fill_value, dtype=np.uint8)

        def get_time_stamp(self):
            return 42.0

        def release(self) -> None:
            self.released = True

    # Each 32x32 RGB888 image is ~3 KB; with a 10 KB budget only 3 frames fit,
    # strictly respecting max_cache_bytes without overshooting.
    worker = LoadFrameThread(
        cache_size=20,
        max_cache_bytes=10 * 1024,
        prefetch_enabled=False,
    )
    loader_a = _DummyLoader(11)
    loader_b = _DummyLoader(99)

    worker.set_video_loader(loader_a)
    for idx in range(8):
        worker.request(idx)
        worker.load()

    assert worker._cache_bytes <= 10 * 1024
    assert len(worker._frame_cache) == 3
    assert list(worker._frame_cache.keys()) == [5, 6, 7]
    assert set(worker._timestamp_cache.keys()) == {5, 6, 7}

    # Switching video loader must clear stale cache and release previous loader.
    emitted_reds: list[int] = []
    worker.res_frame.connect(
        lambda _idx, img: emitted_reds.append(img.pixelColor(0, 0).red())
    )
    worker.set_video_loader(loader_b, release_previous=True)
    assert loader_a.released is True
    assert len(worker._frame_cache) == 0

    worker.request(7)
    worker.load()
    assert emitted_reds == [99]


def test_cv2video_detects_vfr_grab_drift_falls_back_and_disables_grab_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "vfr_sim.avi"
    _write_indexed_video(video_path, fps=10.0, frames=20)

    with CV2Video(video_path) as ref:
        reference_frames = {i: ref.load_frame(i) for i in range(20)}

    with CV2Video(video_path) as video:
        real_cap = video.cap
        grab_calls = {"count": 0}
        set_calls: list[int] = []

        class _DriftingVfrCap:
            def __getattr__(self, name):
                return getattr(real_cap, name)

            def grab(self):
                grab_calls["count"] += 1
                # Simulate VFR drop-frame jump: grab advances by 2 frames instead of 1
                real_cap.grab()
                return real_cap.grab()

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    set_calls.append(int(value))
                return real_cap.set(prop, value)

        video.cap = _DriftingVfrCap()  # type: ignore[assignment]

        f0 = video.load_frame(0)
        assert np.array_equal(f0, reference_frames[0])

        # Stride +5 detects drift via POS_FRAMES, falls back to explicit seek(5),
        # returns exact frame 5, and disables grab-skip permanently for this reader.
        f5 = video.load_frame(5)
        assert np.array_equal(f5, reference_frames[5])
        assert video._grab_skip_disabled is True
        assert 5 in set_calls

        grabs_after_disable = grab_calls["count"]
        f10 = video.load_frame(10)
        assert np.array_equal(f10, reference_frames[10])
        assert grab_calls["count"] == grabs_after_disable  # zero wasted grabs
        assert 10 in set_calls

    # Environment variable override disables grab-skip by default when set to 0.
    monkeypatch.setenv("ANNOLID_MAX_FORWARD_GRAB_SKIP", "0")
    with CV2Video(video_path) as env_video:
        assert env_video._max_forward_grab_skip == 0
        assert env_video._grab_skip_disabled is True


def test_frame_loader_env_budget_trim_and_oversized_prefetch_guard(
    monkeypatch,
) -> None:
    app = _ensure_qapp()

    monkeypatch.setenv("ANNOLID_FRAME_CACHE_MB", "32")
    env_worker = LoadFrameThread(prefetch_enabled=False)
    assert env_worker._max_cache_bytes == 32 * 1024 * 1024

    class _LargeFrameLoader:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def total_frames(self) -> int:
            return 10

        def load_frame(self, frame_number: int):
            self.calls.append(int(frame_number))
            return np.full((64, 64, 3), int(frame_number), dtype=np.uint8)  # 12 KB

    # Budget (15 KB) fits only 1 frame (12 KB), NOT 2 frames (24 KB):
    # must keep at most 1 frame and skip idle prefetch so active frame is not evicted.
    worker = LoadFrameThread(
        cache_size=16,
        max_cache_bytes=15 * 1024,
        prefetch_enabled=True,
        prefetch_ahead=3,
    )
    loader = _LargeFrameLoader()
    worker.video_loader = loader

    worker.request(0)
    worker.load()
    app.processEvents()
    app.processEvents()
    assert loader.calls == [0]  # Prefetch skipped!
    assert list(worker._frame_cache.keys()) == [0]

    worker.request(1)
    worker.load()
    assert list(worker._frame_cache.keys()) == [1]

    # Larger budget allows caching multiple frames; trim_cache reclaims down to 1.
    worker.set_cache_limits(max_bytes=100 * 1024)
    for idx in (2, 3, 4):
        worker.request(idx)
        worker.load()
    assert len(worker._frame_cache) >= 3

    worker.trim_cache(keep_entries=1)
    assert list(worker._frame_cache.keys()) == [4]
    assert worker._cache_bytes == worker._qimage_byte_size(worker._frame_cache[4])


def test_secondary_video_tile_enforces_byte_budget(tmp_path: Path) -> None:
    _ensure_qapp()
    video_path = tmp_path / "cam_budget.avi"
    _write_indexed_video(video_path, fps=10.0, frames=10)

    tile = SecondaryVideoTile(video_path, player_id="2", offset_s=0.0)
    try:
        # 64x48 RGB888 frame is ~9 KB; cap at 12 KB so at most 1 frame stays cached.
        tile._frame_cache_max_bytes = 12 * 1024
        for sec in (0.1, 0.2, 0.3, 0.4):
            tile.sync_to_time(sec, immediate=True)
        assert len(tile._frame_cache) == 1
        assert tile._frame_cache_bytes <= 12 * 1024
    finally:
        tile.close_reader()
        tile.close()


def test_frame_loader_idle_prefetch_warms_upcoming_frames_without_emitting() -> None:
    app = _ensure_qapp()

    class _PrefetchLoader:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def total_frames(self) -> int:
            return 30

        def load_frame(self, frame_number: int):
            self.calls.append(int(frame_number))
            return np.full((8, 8, 3), int(frame_number), dtype=np.uint8)

    worker = LoadFrameThread(prefetch_enabled=True, prefetch_ahead=2)
    loader = _PrefetchLoader()
    worker.video_loader = loader

    emitted: list[int] = []
    worker.res_frame.connect(lambda idx, _img: emitted.append(int(idx)))

    # Step by +5 so worker infers stride +5
    worker.request(0)
    worker.load()
    worker.request(5)
    worker.load()
    app.processEvents()
    app.processEvents()

    assert emitted == [0, 5]
    # Prefetch warmed frames 10 and 15 in cache without emitting res_frame
    assert worker.get_cached_frame(10) is not None
    assert worker.get_cached_frame(15) is not None

    # Now requesting frame 10 is an instant cache hit (no extra load_frame(10) call)
    calls_before = list(loader.calls)
    worker.request(10)
    assert emitted == [0, 5, 10]
    assert loader.calls == calls_before


def test_frame_loader_emits_frame_error_and_fallback_without_double_process() -> None:
    _ensure_qapp()

    class _FlakyLoader:
        def load_frame(self, frame_number: int):
            if frame_number == 1:
                raise KeyError("corrupt frame 1")
            return np.full((4, 4, 3), 77, dtype=np.uint8)

    worker = LoadFrameThread(prefetch_enabled=False)
    worker.video_loader = _FlakyLoader()

    errors: list[tuple[int, str]] = []
    frames: list[int] = []
    worker.frame_error.connect(lambda idx, msg: errors.append((int(idx), str(msg))))
    worker.res_frame.connect(lambda idx, _img: frames.append(int(idx)))

    worker.request(0)
    worker.load()
    worker.request(1)
    worker.load()

    assert frames == [0, 1]
    assert len(errors) == 1
    assert errors[0][0] == 1
    assert "corrupt frame 1" in errors[0][1]


# ---------------------------------------------------------------------------
# LabelFile, Canvas, AnnotationLoadingMixin, Playback & Slider tests
# ---------------------------------------------------------------------------


def test_label_file_video_frame_skips_sibling_png_reload_but_keeps_embedded_data(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "vid_000000000.png"
    PIL.Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8)).save(png_path)

    # Case 1: imageData is None -> video frame must NOT read/re-encode sibling PNG
    json_null = tmp_path / "vid_000000000.json"
    json_null.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": png_path.name,
                "imageData": None,
                "imageHeight": 16,
                "imageWidth": 24,
            }
        ),
        encoding="utf-8",
    )
    with patch.object(
        LabelFile,
        "load_image_file",
        side_effect=AssertionError("Should not reload PNG"),
    ):
        lf = LabelFile(str(json_null), is_video_frame=True)
    assert lf.imageData is None
    assert lf.imagePath == png_path.name

    # Case 2: embedded base64 imageData -> preserved intact
    buf = io.BytesIO()
    PIL.Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    json_embedded = tmp_path / "vid_000000001.json"
    json_embedded.write_text(
        json.dumps(
            {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": png_path.name,
                "imageData": base64.b64encode(raw_bytes).decode("ascii"),
                "imageHeight": 8,
                "imageWidth": 8,
            }
        ),
        encoding="utf-8",
    )
    lf_embedded = LabelFile(str(json_embedded), is_video_frame=True)
    assert lf_embedded.imageData == raw_bytes


def test_canvas_load_pixmap_skips_signature_hashing_when_no_ai_prompt_active(
    monkeypatch,
) -> None:
    _ensure_qapp()
    canvas = Canvas()
    try:
        sig_calls = {"count": 0}
        original = canvas._ai_model_image_signature_value

        def _counting_sig():
            sig_calls["count"] += 1
            return original()

        monkeypatch.setattr(canvas, "_ai_model_image_signature_value", _counting_sig)

        img1 = QtGui.QImage(32, 24, QtGui.QImage.Format_RGB888)
        img1.fill(QtGui.QColor(10, 20, 30))
        img2 = QtGui.QImage(32, 24, QtGui.QImage.Format_RGB888)
        img2.fill(QtGui.QColor(40, 50, 60))

        canvas.loadPixmap(QtGui.QPixmap.fromImage(img1))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(img2))
        assert sig_calls["count"] == 0

        # When a text prompt is running, signature check MUST run to cancel stale prompt
        monkeypatch.setattr(canvas, "is_text_prompt_running", lambda: True)
        cancelled: list[str] = []
        monkeypatch.setattr(
            canvas,
            "cancel_async_text_prompt",
            lambda **kwargs: cancelled.append(str(kwargs.get("reason"))),
        )
        canvas.loadPixmap(QtGui.QPixmap.fromImage(img1))
        assert sig_calls["count"] >= 2
        assert cancelled == ["frame_changed"]
    finally:
        canvas.close()


def test_annotation_loading_multi_file_zone_cache_and_unlabeled_caption_clear(
    monkeypatch, tmp_path: Path
) -> None:
    class _CaptionStub:
        def __init__(self) -> None:
            self.captions: list[str] = []
            self.paths: list[str] = []

        def set_image_path(self, path) -> None:
            self.paths.append(str(path))

        def set_caption(self, text: str) -> None:
            self.captions.append(str(text))

    class _Host(AnnotationLoadingMixin):
        def __init__(self) -> None:
            self._config = {"label_flags": {}}
            self.canvas = type(
                "C",
                (),
                {
                    "shapes": [],
                    "current_behavior_text": "",
                    "setBehaviorText": lambda _s, _v: None,
                },
            )()
            self.zone_path = None
            self.video_file = None
            self.filename = None
            self.video_results_folder = tmp_path
            self.annotation_dir = None
            self._pred_res_folder_suffix = "_tracking_results_labelme"
            self._display_zones_on_all_frames = True
            self.caption_widget = _CaptionStub()
            self.loaded_shapes = []

        def _apply_timeline_caption_if_available(self, *_args, **_kwargs) -> bool:
            return False

        def loadShapes(self, shapes, replace=True):  # noqa: ARG002
            self.loaded_shapes = list(shapes or [])

    # Create 4 manual seed pairs (one with a zone)
    for idx in range(4):
        (tmp_path / f"s_{idx:09d}.png").write_bytes(b"")
        shapes = (
            [
                build_zone_shape(
                    "arena_zone",
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    zone_kind="chamber",
                )
            ]
            if idx == 2
            else []
        )
        (tmp_path / f"s_{idx:09d}.json").write_text(
            json.dumps({"shapes": shapes}), encoding="utf-8"
        )

    host = _Host()
    read_calls = {"count": 0}
    orig_read_text = Path.read_text

    def _counting_read_text(self, *args, **kwargs):
        read_calls["count"] += 1
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counting_read_text)

    frame_path = tmp_path / "s_000000010.png"
    zones1 = host._persistent_zone_shapes_for_frame(frame_path)
    assert [z.label for z in zones1] == ["arena_zone"]
    assert read_calls["count"] == 4

    # Subsequent calls hit the multi-file cache (zero JSON re-reads!)
    zones2 = host._persistent_zone_shapes_for_frame(frame_path)
    assert [z.label for z in zones2] == ["arena_zone"]
    assert read_calls["count"] == 4

    # Modifying a zone file invalidates that file's entry
    time.sleep(0.01)
    (tmp_path / "s_000000002.json").write_text(
        json.dumps(
            {
                "shapes": [
                    build_zone_shape(
                        "updated_zone",
                        [[0, 0], [10, 0], [10, 10], [0, 10]],
                        zone_kind="chamber",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    zones3 = host._persistent_zone_shapes_for_frame(frame_path)
    assert [z.label for z in zones3] == ["updated_zone"]
    assert read_calls["count"] == 5

    # Unlabeled frame must clear stale caption (previously unreachable dead code)
    assert host.loadPredictShapes(10, str(frame_path)) is False
    assert host.caption_widget.captions == [""]


def test_prefetch_label_for_frame_respects_store_preferred(
    monkeypatch, tmp_path: Path
) -> None:
    class _FailLabelFile:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Prefetch should read annotation store, not disk JSON")

    monkeypatch.setattr(annotation_loading_module, "LabelFile", _FailLabelFile)

    class _Host(AnnotationLoadingMixin):
        def __init__(self) -> None:
            self.video_results_folder = tmp_path
            self.annotation_dir = None
            self._pred_res_folder_suffix = "_tracking_results_labelme"
            self.frame_number = 0

        def _frame_image_path(self, frame_idx: int) -> Path:
            return tmp_path / f"{tmp_path.name}_{frame_idx:09d}.png"

    host = _Host()
    candidate = tmp_path / f"{tmp_path.name}_000000003.json"
    store = AnnotationStore.for_frame_path(candidate)
    store.append_frame(
        {
            "frame": 3,
            "version": "1.7.0",
            "flags": {},
            "shapes": [],
            "imagePath": "",
            "imageHeight": 10,
            "imageWidth": 10,
        }
    )

    host._prefetch_label_for_frame(3, host._frame_image_path(0))
    token = host._label_candidate_cache_token(candidate)
    assert token is not None and token[0] == "store_preferred"
    assert host._label_file_cache_get(candidate, token) is not None


def test_jump_to_frame_from_log_invokes_set_frame_number_with_video_slider() -> None:
    _ensure_qapp()

    class _JumpHost(BehaviorInteractionMixin, FramePlaybackMixin):
        def __init__(self) -> None:
            self.seekbar = VideoSlider(min=0, max=100, val=0)
            self.num_frames = 101
            self.frame_number = 0
            self.fps = 25.0
            self.video_loader = None
            self.frame_loader = None
            self.timeline_panel = None
            self.caption_widget = None
            self.video_results_folder = None
            self.filename = ""
            self.isPlaying = False
            self._suppress_audio_seek = False
            self.requested: list[int] = []

        def _update_audio_playhead(self, _frame_number: int) -> None:
            return

        def _active_audio_loader(self, ensure_ready: bool = False):  # noqa: ARG002
            return None

        def _update_embedding_query_frame(self) -> None:
            return

    host = _JumpHost()
    try:
        host._jump_to_frame_from_log(42)
        assert host.frame_number == 42
        assert host.seekbar.value() == 42
        assert host.current_frame_time_stamp == pytest.approx((42 / 25.0) * 1000.0)

        # Typing into VideoSlider input_value must also emit valueChanged
        emitted: list[int] = []
        host.seekbar.valueChanged.connect(lambda v: emitted.append(int(v)))
        host.seekbar.input_value.setText("67")
        host.seekbar.updateValueFromInput()
        assert host.seekbar.value() == 67
        assert emitted == [67]
    finally:
        host.seekbar.close()


def test_embedding_search_widget_caches_resolved_paths(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    widget = EmbeddingSearchWidget()
    try:
        resolve_calls = {"count": 0}
        orig_resolve = Path.resolve

        def _counting_resolve(self, *args, **kwargs):
            resolve_calls["count"] += 1
            return orig_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _counting_resolve)

        vid = tmp_path / "demo.mp4"
        widget.set_video_path(vid)
        widget.set_video_path(vid)
        widget.set_video_path(vid)
        assert resolve_calls["count"] == 1
    finally:
        widget.close()


def test_secondary_video_tile_reuses_frame_cache_on_revisit(tmp_path: Path) -> None:
    _ensure_qapp()
    video_path = tmp_path / "cam2.avi"
    _write_indexed_video(video_path, fps=10.0, frames=10)

    tile = SecondaryVideoTile(video_path, player_id="2", offset_s=0.0)
    try:
        load_calls: list[int] = []
        orig_load = tile._reader.load_frame

        def _counting_load(idx: int):
            load_calls.append(int(idx))
            return orig_load(idx)

        tile._reader.load_frame = _counting_load  # type: ignore[method-assign]

        tile.sync_to_time(0.3, immediate=True)  # frame 3
        tile.sync_to_time(0.5, immediate=True)  # frame 5
        tile.sync_to_time(0.3, immediate=True)  # frame 3 -> cached!

        assert load_calls == [3, 5]
        assert tile.displayed_frame == 3
    finally:
        tile.close_reader()
        tile.close()


def test_open_video_programmatic_switch_releases_old_loader_and_connects_once(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    class _FakeLoader:
        def __init__(self, fill_value: int) -> None:
            self.fill_value = fill_value
            self.released = False

        def get_fps(self) -> float:
            return 10.0

        def total_frames(self) -> int:
            return 5

        def load_frame(self, _idx: int):
            return np.full((8, 8, 3), self.fill_value, dtype=np.uint8)

        def get_time_stamp(self):
            return 0.0

        def release(self) -> None:
            self.released = True

    class _FakeStatusBar:
        def removeWidget(self, _w) -> None:
            return

        def addPermanentWidget(self, _w, stretch: int = 0) -> None:  # noqa: ARG002
            return

    class _FakeWorkerThread(QtCore.QThread):
        def __init__(self) -> None:
            super().__init__()
            self.started_priorities: list = []

        def isRunning(self) -> bool:  # noqa: N802
            return False

        def start(self, priority=QtCore.QThread.InheritPriority) -> None:  # noqa: N802
            self.started_priorities.append(priority)

    class _OpenHost(VideoWorkflowMixin, FramePlaybackMixin, QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.dirty = False
            self.video_loader = None
            self.video_file = None
            self.video_results_folder = None
            self.annotation_dir = None
            self.filename = None
            self.fps = None
            self.num_frames = None
            self.frame_number = 0
            self.isPlaying = False
            self._suppress_audio_seek = False
            self.timeline_panel = None
            self.caption_widget = None
            self.seekbar = None
            self.playButton = None
            self.frameJumpLabel = None
            self.frameJumpInput = None
            self.saveButton = None
            self.stepSizeWidget = QtWidgets.QWidget()
            self.behavior_controller = type(
                "BC",
                (),
                {
                    "attach_annotation_store_for_video": lambda _s, _p: None,
                    "attach_slider": lambda _s, _sl: None,
                    "events_count": 0,
                    "highlighted_mark": None,
                },
            )()
            self.behavior_log_widget = type(
                "BL", (), {"set_fps": lambda _s, _f: None}
            )()
            self.actions = type(
                "Act",
                (),
                {
                    "openNextImg": type("A", (), {"setEnabled": lambda _s, _v: None})(),
                    "openPrevImg": type("A", (), {"setEnabled": lambda _s, _v: None})(),
                },
            )()
            self.frame_worker = _FakeWorkerThread()
            self.frame_loader = LoadFrameThread(prefetch_enabled=False)
            self._status_bar = _FakeStatusBar()
            self.loaded_events: list[tuple[int, int]] = []
            self.cleared_caches = 0
            self._loaders = [_FakeLoader(10), _FakeLoader(200)]
            self._loader_idx = 0

        def statusBar(self):  # noqa: N802
            return self._status_bar

        def _cleanup_audio_ui(self) -> None:
            return

        def _refresh_embedding_file_list(self) -> None:
            return

        def _create_video_loader(self, _path: str):
            loader = self._loaders[self._loader_idx]
            self._loader_idx += 1
            return loader

        def _clear_frame_annotation_caches(self) -> None:
            self.cleared_caches += 1

        def _configure_project_schema_for_video(self, _path: str) -> None:
            return

        def _apply_timeline_dock_visibility(self, *, video_open: bool) -> None:
            _ = video_open

        def _configure_audio_for_video(self, *_args, **_kwargs) -> None:
            return

        def _create_frame_jump_input(self, *, minimum: int, maximum: int, value: int):
            spin = QtWidgets.QSpinBox(self)
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            return spin

        def tooltip_callable(self, _val) -> str:
            return ""

        def saveTimestampList(self) -> None:  # noqa: N802
            return

        def _apply_timeline_caption_if_available(self, *_args, **_kwargs) -> bool:
            return False

        def _schedule_video_sidecar_preload(self, _path: str) -> None:
            return

        def _schedule_video_open_background_tasks(self, **_kwargs) -> None:
            return

        def _update_audio_playhead(self, _frame: int) -> None:
            return

        def _active_audio_loader(self, ensure_ready: bool = False):  # noqa: ARG002
            return None

        def _update_embedding_query_frame(self) -> None:
            return

        def image_to_canvas(self, qimage, _path, frame_number: int) -> None:
            self.loaded_events.append(
                (int(frame_number), qimage.pixelColor(0, 0).red())
            )

    host = _OpenHost()
    try:
        # Keep frame_loader on current thread for deterministic synchronous testing
        host.frame_loader.moveToThread = lambda _t: None  # type: ignore[method-assign]

        vid_a = tmp_path / "a.mp4"
        vid_b = tmp_path / "b.mp4"
        vid_a.write_bytes(b"")
        vid_b.write_bytes(b"")

        host.openVideo(
            from_video_list=True, video_path=str(vid_a), programmatic_call=True
        )
        host.frame_loader.load()
        assert host.loaded_events == [(0, 10)]
        assert host.frame_worker.started_priorities[-1] == QtCore.QThread.NormalPriority

        # Programmatically switch to vid_b without closeFile()
        host.frame_number = 4
        host.openVideo(
            from_video_list=True, video_path=str(vid_b), programmatic_call=True
        )
        host.frame_loader.load()

        # Old loader released, stale frame-0 cache invalidated, frame_number reset,
        # and res_frame fired EXACTLY ONCE (no duplicate connection!)
        assert host._loaders[0].released is True
        assert host.cleared_caches == 2
        assert host.frame_number == 0
        assert host.loaded_events == [(0, 10), (0, 200)]
    finally:
        host.frame_loader.shutdown()
        host.close()
