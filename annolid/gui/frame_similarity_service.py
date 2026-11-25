"""Qt worker service for asynchronous video frame and behavior embedding search."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from qtpy import QtCore

from annolid.gui.qt_threads import configure_worker_thread_stack
from annolid.infrastructure import AnnotationStore  # noqa: F401
from annolid.infrastructure import load_labelme_json  # noqa: F401
from annolid.services.embedding_search import (
    DEFAULT_SEARCH_BACKEND,
    FrameEmbeddingBackend,
    VideoFrameEmbeddingIndex,
    backend_cache_key,
    build_embedding_backend,
    default_video_index_cache_path,
    load_shapes_for_frame,
    normalize_backend_name,
    overlay_shapes_on_frame_rgb,
    search_video_behavior,
)
from annolid.core.models.adapters.embeddinggemma2_onnx import l2_normalize_np

# Backward-compatible internal helper aliases retained for existing callers/tests.
_l2_normalize = l2_normalize_np
_overlay_shapes_on_frame_rgb = overlay_shapes_on_frame_rgb
_load_shapes_for_frame = load_shapes_for_frame
_MAX_CACHED_VIDEO_INDICES = 4


@dataclass(frozen=True)
class FrameSimilarityMatch:
    frame_index: int
    similarity: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame_index": int(self.frame_index),
            "similarity": float(self.similarity),
        }
        if self.start_frame is not None:
            payload["start_frame"] = int(self.start_frame)
        if self.end_frame is not None:
            payload["end_frame"] = int(self.end_frame)
        if self.label is not None:
            payload["label"] = str(self.label)
        return payload


@dataclass(frozen=True)
class FrameSimilaritySearchRequest:
    video_path: Path
    query_frame_index: Optional[int] = None
    annotation_dir: Optional[Path] = None
    overlay_shapes: bool = True
    stride: int = 5
    max_frames: Optional[int] = None
    top_k: int = 50
    threshold: Optional[float] = None
    backend: str = DEFAULT_SEARCH_BACKEND
    backend_params: Dict[str, Any] = None  # type: ignore[assignment]
    query_text: Optional[str] = None
    query_mode: str = "auto"
    clip_frames: int = 1
    batch_size: int = 4
    group_bouts: bool = True
    use_disk_cache: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(
            self,
            "query_frame_index",
            int(self.query_frame_index) if self.query_frame_index is not None else None,
        )
        object.__setattr__(
            self,
            "annotation_dir",
            Path(self.annotation_dir) if self.annotation_dir else None,
        )
        object.__setattr__(self, "overlay_shapes", bool(self.overlay_shapes))
        object.__setattr__(self, "stride", max(1, int(self.stride)))
        object.__setattr__(self, "top_k", max(1, int(self.top_k)))
        object.__setattr__(
            self,
            "threshold",
            float(self.threshold) if self.threshold is not None else None,
        )
        object.__setattr__(
            self,
            "backend",
            normalize_backend_name(self.backend or DEFAULT_SEARCH_BACKEND),
        )
        object.__setattr__(self, "backend_params", dict(self.backend_params or {}))
        clean_text = str(self.query_text).strip() if self.query_text else None
        object.__setattr__(self, "query_text", clean_text or None)
        object.__setattr__(
            self, "query_mode", str(self.query_mode or "auto").strip().lower()
        )
        object.__setattr__(self, "clip_frames", max(1, int(self.clip_frames)))
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))
        object.__setattr__(self, "group_bouts", bool(self.group_bouts))
        object.__setattr__(self, "use_disk_cache", bool(self.use_disk_cache))


def _build_backend(request: FrameSimilaritySearchRequest) -> FrameEmbeddingBackend:
    return build_embedding_backend(request.backend, request.backend_params)


def _backend_key_for_request(
    request: FrameSimilaritySearchRequest,
) -> Tuple[str, ...]:
    return backend_cache_key(request.backend, request.backend_params)


def _index_cache_key(
    *,
    backend_key: Tuple[str, ...],
    video_path: Path,
    stride: int,
    max_frames: Optional[int],
    clip_frames: int,
    overlay_shapes: bool,
    annotation_dir: Optional[Path],
) -> Tuple[Any, ...]:
    try:
        stat = video_path.stat()
        mtime_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    except OSError:
        mtime_ns = 0
        file_size = 0
    return (
        tuple(backend_key),
        str(video_path),
        mtime_ns,
        file_size,
        max(1, int(stride)),
        int(max_frames) if max_frames is not None else None,
        max(1, int(clip_frames)),
        bool(overlay_shapes),
        str(annotation_dir) if overlay_shapes and annotation_dir is not None else "",
    )


class _FrameSimilarityWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    boutsFound = QtCore.Signal(list)
    scoreSummary = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    matchFound = QtCore.Signal(int, float)

    def __init__(self, stop_event: Optional[threading.Event] = None) -> None:
        super().__init__()
        self._stop_event = stop_event or threading.Event()
        self._backend: Optional[FrameEmbeddingBackend] = None
        self._backend_key: Optional[Tuple[str, ...]] = None
        self._index_cache: "OrderedDict[Tuple[Any, ...], VideoFrameEmbeddingIndex]" = (
            OrderedDict()
        )
        self._session_disk_cache_paths: set[Path] = set()

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop_event.set()

    @QtCore.Slot()
    def clear_cache(self) -> None:
        self._index_cache.clear()
        for path in list(self._session_disk_cache_paths):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        self._session_disk_cache_paths.clear()

    @QtCore.Slot()
    def close_backend(self) -> None:
        self._index_cache.clear()
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None
            self._backend_key = None

    @QtCore.Slot(object)
    # type: ignore[override]
    def run(self, request: FrameSimilaritySearchRequest) -> None:
        try:
            if self._stop_event.is_set():
                self.scoreSummary.emit({})
                self.boutsFound.emit([])
                self.finished.emit([])
                return

            video_path = Path(request.video_path).expanduser().resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"Video does not exist: {video_path}")

            annotation_dir = (
                Path(request.annotation_dir).expanduser().resolve()
                if request.annotation_dir
                else video_path.with_suffix("")
            )
            backend = self._ensure_backend(request)

            cache_key = _index_cache_key(
                backend_key=backend.key,
                video_path=video_path,
                stride=request.stride,
                max_frames=request.max_frames,
                clip_frames=request.clip_frames,
                overlay_shapes=request.overlay_shapes,
                annotation_dir=annotation_dir,
            )
            cached_index = self._index_cache.get(cache_key)
            if cached_index is not None:
                self._index_cache.move_to_end(cache_key)

            use_auto_disk_cache = bool(
                cached_index is None
                and request.use_disk_cache
                and not request.overlay_shapes
            )
            disk_cache_path: Optional[Path] = None
            if use_auto_disk_cache:
                disk_cache_path = default_video_index_cache_path(
                    video_path=video_path,
                    backend_key=backend.key,
                    stride=request.stride,
                    max_frames=request.max_frames,
                    clip_frames=request.clip_frames,
                    overlay_shapes=False,
                )

            result = search_video_behavior(
                video_path,
                query_text=request.query_text,
                query_frame_index=request.query_frame_index,
                query_mode=request.query_mode,
                backend=backend,
                stride=request.stride,
                max_frames=request.max_frames,
                clip_frames=request.clip_frames,
                batch_size=request.batch_size,
                threshold=request.threshold,
                top_k=request.top_k,
                overlay_shapes=request.overlay_shapes,
                annotation_dir=annotation_dir,
                group_bouts=request.group_bouts,
                index=cached_index,
                auto_disk_cache=use_auto_disk_cache,
                is_stopped=self._stop_event.is_set,
                on_match=lambda m: self.matchFound.emit(
                    int(m.frame_index), float(m.similarity)
                ),
                on_progress=lambda step, total: self.progress.emit(
                    int(step), int(total)
                ),
            )

            if disk_cache_path is not None and disk_cache_path.is_file():
                self._session_disk_cache_paths.add(disk_cache_path)

            if result.completed and len(result.index) > 0 and cached_index is None:
                self._index_cache[cache_key] = result.index
                self._index_cache.move_to_end(cache_key)
                while len(self._index_cache) > _MAX_CACHED_VIDEO_INDICES:
                    self._index_cache.popitem(last=False)

            payload = [m.to_dict() for m in result.matches]
            bouts_payload = [b.to_dict() for b in result.bouts]
            summary_payload = (
                result.score_summary.to_dict()
                if result.score_summary is not None
                else {}
            )
            self.scoreSummary.emit(summary_payload)
            self.boutsFound.emit(bouts_payload)
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))

    def _ensure_backend(
        self, request: FrameSimilaritySearchRequest
    ) -> FrameEmbeddingBackend:
        key = _backend_key_for_request(request)
        if self._backend is None or self._backend_key != key:
            if self._backend is not None:
                try:
                    self._backend.close()
                except Exception:
                    pass
            backend = _build_backend(request)
            self._backend = backend
            self._backend_key = tuple(backend.key)
        return self._backend


class FrameSimilarityService(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal(list)
    boutsFound = QtCore.Signal(list)
    scoreSummary = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    matchFound = QtCore.Signal(int, float)
    _request_run = QtCore.Signal(object)
    _request_clear_cache = QtCore.Signal()
    _request_close_backend = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        # Shared thread-safe stop event so stop() immediately interrupts an active
        # run() loop even while the worker thread's Qt event loop is busy.
        self._stop_event = threading.Event()
        self._thread = configure_worker_thread_stack(QtCore.QThread(self))
        self._worker = _FrameSimilarityWorker(stop_event=self._stop_event)
        self._worker.moveToThread(self._thread)
        self._request_run.connect(self._worker.run)
        self._request_clear_cache.connect(self._worker.clear_cache)
        self._request_close_backend.connect(self._worker.close_backend)

        self._worker.finished.connect(self._handle_finished)
        self._worker.boutsFound.connect(self.boutsFound)
        self._worker.scoreSummary.connect(self.scoreSummary)
        self._worker.error.connect(self._handle_error)
        self._worker.progress.connect(self.progress)
        self._worker.matchFound.connect(self.matchFound)
        self._thread.start()
        self._busy = False

    def is_busy(self) -> bool:
        return bool(self._busy)

    def request(self, request: FrameSimilaritySearchRequest) -> bool:
        if self._busy:
            return False
        self._stop_event.clear()
        self._busy = True
        self.started.emit()
        self._request_run.emit(request)
        return True

    def stop(self) -> None:
        if not self._busy:
            return
        self._stop_event.set()

    def clear_cache(self) -> None:
        self._request_clear_cache.emit()

    @QtCore.Slot(list)
    def _handle_finished(self, results: list) -> None:
        self._busy = False
        self.finished.emit(results)

    @QtCore.Slot(str)
    def _handle_error(self, message: str) -> None:
        self._busy = False
        self.error.emit(message)

    def close(self) -> None:
        try:
            self._stop_event.set()
            self._request_close_backend.emit()
        except Exception:
            pass
        try:
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass


__all__ = [
    "FrameSimilarityMatch",
    "FrameSimilaritySearchRequest",
    "FrameSimilarityService",
    "_backend_key_for_request",
    "_build_backend",
    "_l2_normalize",
    "_load_shapes_for_frame",
    "_overlay_shapes_on_frame_rgb",
]
