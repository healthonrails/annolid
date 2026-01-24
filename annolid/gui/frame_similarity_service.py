from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from qtpy import QtCore

from annolid.core.media.video import CV2Video


@dataclass(frozen=True)
class FrameSimilarityMatch:
    frame_index: int
    similarity: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "similarity": float(self.similarity),
        }


@dataclass(frozen=True)
class FrameSimilaritySearchRequest:
    video_path: Path
    query_frame_index: int
    annotation_dir: Optional[Path] = None
    overlay_shapes: bool = True
    stride: int = 5
    max_frames: Optional[int] = None
    top_k: int = 50
    threshold: float = 0.35
    backend: str = "dinov3"
    backend_params: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(self, "query_frame_index", int(self.query_frame_index))
        object.__setattr__(
            self,
            "annotation_dir",
            Path(self.annotation_dir) if self.annotation_dir else None,
        )
        object.__setattr__(self, "overlay_shapes", bool(self.overlay_shapes))
        object.__setattr__(self, "stride", max(1, int(self.stride)))
        object.__setattr__(self, "top_k", max(1, int(self.top_k)))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self, "backend", str(self.backend or "dinov3").strip().lower()
        )
        object.__setattr__(self, "backend_params", dict(self.backend_params or {}))


def _color_for_label(label: str) -> Tuple[int, int, int]:
    # Deterministic, reasonably bright RGB color from a label.
    value = abs(hash(label or "shape"))
    r = 64 + (value & 0x7F)
    g = 64 + ((value >> 7) & 0x7F)
    b = 64 + ((value >> 14) & 0x7F)
    return int(r), int(g), int(b)


def _clip_point(pt: Tuple[int, int], *, width: int, height: int) -> Tuple[int, int]:
    x, y = int(pt[0]), int(pt[1])
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y


def _overlay_shapes_on_frame_rgb(
    frame_rgb: np.ndarray,
    shapes: List[Dict[str, Any]],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    if not shapes:
        return frame_rgb
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return frame_rgb

    height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    base = frame_rgb.astype(np.uint8, copy=True)
    overlay = base.copy()

    # Draw filled regions on overlay, then blend; outlines go on the blended image.
    outlines: List[Tuple[str, Tuple[int, int, int], Any]] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "")
        shape_type = str(shape.get("shape_type") or "").lower()
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            continue

        color = _color_for_label(label)
        try:
            pts = np.asarray(points, dtype=np.float32)
        except Exception:
            continue
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        pts_i = np.round(pts[:, :2]).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, width - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, height - 1)

        if shape_type in {"polygon", "linestrip"} and len(pts_i) >= 2:
            closed = shape_type == "polygon" and len(pts_i) >= 3
            if closed:
                try:
                    cv2.fillPoly(overlay, [pts_i], color)
                except Exception:
                    pass
            outlines.append(("poly", color, (pts_i, closed)))
            continue

        if shape_type == "rectangle" and len(pts_i) >= 2:
            (x1, y1) = _clip_point(
                (int(pts_i[0, 0]), int(pts_i[0, 1])), width=width, height=height
            )
            (x2, y2) = _clip_point(
                (int(pts_i[1, 0]), int(pts_i[1, 1])), width=width, height=height
            )
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            outlines.append(("rect", color, (x1, y1, x2, y2)))
            continue

        if shape_type == "circle" and len(pts_i) >= 2:
            c = _clip_point(
                (int(pts_i[0, 0]), int(pts_i[0, 1])), width=width, height=height
            )
            p = _clip_point(
                (int(pts_i[1, 0]), int(pts_i[1, 1])), width=width, height=height
            )
            radius = int(
                max(1, round(float(np.linalg.norm(np.asarray(c) - np.asarray(p)))))
            )
            cv2.circle(overlay, c, radius, color, thickness=-1)
            outlines.append(("circle", color, (c, radius)))
            continue

        if shape_type == "line" and len(pts_i) >= 2:
            p0 = _clip_point(
                (int(pts_i[0, 0]), int(pts_i[0, 1])), width=width, height=height
            )
            p1 = _clip_point(
                (int(pts_i[1, 0]), int(pts_i[1, 1])), width=width, height=height
            )
            outlines.append(("line", color, (p0, p1)))
            continue

        # Default to point marker.
        p0 = _clip_point(
            (int(pts_i[0, 0]), int(pts_i[0, 1])), width=width, height=height
        )
        cv2.circle(overlay, p0, 6, color, thickness=-1)
        outlines.append(("point", color, p0))

    blended = cv2.addWeighted(overlay, float(alpha), base, 1.0 - float(alpha), 0)

    for kind, color, payload in outlines:
        if kind == "poly":
            pts_i, closed = payload
            cv2.polylines(blended, [pts_i], bool(closed), color, thickness=2)
        elif kind == "rect":
            x1, y1, x2, y2 = payload
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=2)
        elif kind == "circle":
            c, radius = payload
            cv2.circle(blended, c, int(radius), color, thickness=2)
        elif kind == "line":
            p0, p1 = payload
            cv2.line(blended, p0, p1, color, thickness=2)
        elif kind == "point":
            p0 = payload
            cv2.circle(blended, p0, 6, color, thickness=2)

    return blended


def _load_shapes_for_frame(
    *,
    annotation_dir: Path,
    frame_index: int,
    store: Optional[object],
) -> List[Dict[str, Any]]:
    if not annotation_dir.exists():
        return []

    try:
        idx = int(frame_index)
    except Exception:
        return []

    primary = annotation_dir / f"{annotation_dir.name}_{idx:09d}.json"
    legacy = annotation_dir / f"{idx:09d}.json"

    # Prefer discrete JSON (when it exists), otherwise fall back to AnnotationStore.
    for candidate in (primary, legacy):
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                from annolid.utils.annotation_store import load_labelme_json

                data = load_labelme_json(candidate)
                shapes = data.get("shapes") if isinstance(data, dict) else None
                if isinstance(shapes, list):
                    return [s for s in shapes if isinstance(s, dict)]
                return []
        except Exception:
            continue

    if store is not None:
        try:
            record = store.get_frame(idx)  # type: ignore[attr-defined]
        except Exception:
            record = None
        if isinstance(record, dict):
            shapes = record.get("shapes")
            if isinstance(shapes, list):
                return [s for s in shapes if isinstance(s, dict)]

    return []


class _StopToken:
    def __init__(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def stopped(self) -> bool:
        return bool(self._stopped)


class _EmbeddingBackend:
    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def key(self) -> Tuple[str, ...]:
        raise NotImplementedError


class _Dinov3Backend(_EmbeddingBackend):
    def __init__(
        self, *, model_name: str, short_side: int, device: Optional[str]
    ) -> None:
        from annolid.features import Dinov3Config, Dinov3FeatureExtractor

        cfg = Dinov3Config(
            model_name=model_name, short_side=int(short_side), device=device
        )
        self._extractor = Dinov3FeatureExtractor(cfg)
        self._key = (
            "dinov3",
            str(model_name),
            str(short_side),
            str(device) if device else "",
        )

    @property
    def key(self) -> Tuple[str, ...]:
        return self._key

    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        feats = self._extractor.extract(
            frame_rgb, return_type="numpy", return_layer="last", normalize=True
        )
        vec = feats.mean(axis=(1, 2)).astype(np.float32, copy=False)
        return _l2_normalize(vec)


class _Qwen3VLBackend(_EmbeddingBackend):
    def __init__(
        self,
        *,
        model_id: str,
        torch_dtype: Optional[str],
        attn_implementation: Optional[str],
    ) -> None:
        from annolid.core.models.adapters.qwen3_embedding import Qwen3EmbeddingAdapter
        from annolid.core.models.base import ModelRequest

        self._ModelRequest = ModelRequest
        self._adapter = Qwen3EmbeddingAdapter(
            model_id=model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self._adapter.load()
        self._key = (
            "qwen3vl",
            str(model_id),
            str(torch_dtype or ""),
            str(attn_implementation or ""),
        )

    @property
    def key(self) -> Tuple[str, ...]:
        return self._key

    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        resp = self._adapter.predict(self._ModelRequest(task="embed", image=frame_rgb))
        vec = (resp.output or {}).get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("Qwen3VL embedding adapter returned no embedding.")
        arr = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(arr)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    denom = float(np.linalg.norm(vec))
    if denom > 1e-12:
        return vec / denom
    return vec


def _build_backend(request: FrameSimilaritySearchRequest) -> _EmbeddingBackend:
    backend = request.backend
    params = dict(request.backend_params or {})
    if backend in {"dinov3", "dino", "dino_v3"}:
        model_name = str(
            params.get("model_name") or "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        short_side = int(params.get("short_side") or 384)
        device = params.get("device")
        device = str(device).strip() if device else None
        return _Dinov3Backend(
            model_name=model_name, short_side=short_side, device=device
        )
    if backend in {"qwen3vl", "qwen3_vl", "qwen3"}:
        model_id = str(params.get("model_id") or "Qwen/Qwen3-VL-Embedding-8B")
        torch_dtype = params.get("torch_dtype")
        torch_dtype = str(torch_dtype).strip() if torch_dtype else None
        attn_impl = params.get("attn_implementation")
        attn_impl = str(attn_impl).strip() if attn_impl else None
        return _Qwen3VLBackend(
            model_id=model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    raise ValueError(f"Unknown embedding backend: {backend!r}")


def _backend_key_for_request(request: FrameSimilaritySearchRequest) -> Tuple[str, ...]:
    backend = request.backend
    params = dict(request.backend_params or {})
    if backend in {"dinov3", "dino", "dino_v3"}:
        model_name = str(
            params.get("model_name") or "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        short_side = int(params.get("short_side") or 384)
        device = params.get("device")
        device_str = str(device).strip() if device else ""
        return ("dinov3", model_name, str(short_side), device_str)
    if backend in {"qwen3vl", "qwen3_vl", "qwen3"}:
        model_id = str(params.get("model_id") or "Qwen/Qwen3-VL-Embedding-8B")
        torch_dtype = params.get("torch_dtype")
        torch_dtype_str = str(torch_dtype).strip() if torch_dtype else ""
        attn_impl = params.get("attn_implementation")
        attn_impl_str = str(attn_impl).strip() if attn_impl else ""
        return ("qwen3vl", model_id, torch_dtype_str, attn_impl_str)
    return (backend,)


class _FrameSimilarityWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    matchFound = QtCore.Signal(int, float)

    def __init__(self) -> None:
        super().__init__()
        self._stop = _StopToken()
        self._stop_requested = False
        self._backend: Optional[_EmbeddingBackend] = None
        self._backend_key: Optional[Tuple[str, ...]] = None
        self._query_cache: Dict[Tuple[Tuple[str, ...], str, int], np.ndarray] = {}

    @QtCore.Slot()
    def stop(self) -> None:
        self._stop_requested = True
        self._stop.stop()

    @QtCore.Slot(object)
    # type: ignore[override]
    def run(self, request: FrameSimilaritySearchRequest) -> None:
        try:
            self._stop = _StopToken()
            if self._stop_requested:
                # Ensure queued stop requests (before the run starts) are honored.
                self._stop.stop()
                self._stop_requested = False
            if self._stop.stopped():
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
            overlay_shapes = bool(request.overlay_shapes)

            store = None
            if annotation_dir.exists():
                try:
                    from annolid.utils.annotation_store import AnnotationStore

                    stub = annotation_dir / f"{annotation_dir.name}_000000000.json"
                    store = AnnotationStore.for_frame_path(stub)
                    if not store.store_path.exists():
                        store = None
                except Exception:
                    store = None

            backend = self._ensure_backend(request)

            if overlay_shapes and annotation_dir is not None:
                query_video = CV2Video(video_path)
                try:
                    query_frame_rgb = query_video.load_frame(
                        int(request.query_frame_index)
                    )
                finally:
                    query_video.release()
                try:
                    shapes = _load_shapes_for_frame(
                        annotation_dir=annotation_dir,
                        frame_index=int(request.query_frame_index),
                        store=store,
                    )
                    if shapes:
                        query_frame_rgb = _overlay_shapes_on_frame_rgb(
                            query_frame_rgb, shapes
                        )
                except Exception:
                    pass
                query_vec = backend.embed_frame_rgb(query_frame_rgb)
            else:
                query_vec = self._query_embedding(
                    backend=backend,
                    video_path=video_path,
                    query_frame_index=int(request.query_frame_index),
                )

            threshold = float(request.threshold)
            top_k = int(request.top_k)
            stride = max(1, int(request.stride))
            max_frames = request.max_frames

            matches: List[FrameSimilarityMatch] = []
            video = CV2Video(video_path)
            try:
                total_frames = int(video.total_frames())
                if total_frames <= 0:
                    raise ValueError("No frames selected for search.")
                max_steps = (total_frames + stride - 1) // stride
                if max_frames is not None:
                    max_steps = min(max_steps, max(1, int(max_frames)))
                total = int(max_steps)
                if total <= 0:
                    raise ValueError("No frames selected for search.")

                step = 0
                for frame_idx in range(0, total_frames, stride):
                    if self._stop.stopped():
                        break
                    if step >= total:
                        break
                    step += 1
                    if int(frame_idx) == int(request.query_frame_index):
                        continue
                    frame_rgb = video.load_frame(int(frame_idx))
                    if overlay_shapes and annotation_dir is not None:
                        try:
                            shapes = _load_shapes_for_frame(
                                annotation_dir=annotation_dir,
                                frame_index=int(frame_idx),
                                store=store,
                            )
                            if shapes:
                                frame_rgb = _overlay_shapes_on_frame_rgb(
                                    frame_rgb, shapes
                                )
                        except Exception:
                            pass
                    vec = backend.embed_frame_rgb(frame_rgb)
                    sim = float(vec @ query_vec)
                    if sim >= threshold:
                        match = FrameSimilarityMatch(
                            frame_index=int(frame_idx), similarity=sim
                        )
                        matches.append(match)
                        self.matchFound.emit(int(frame_idx), float(sim))
                    if step % 10 == 0 or step == total:
                        self.progress.emit(step, total)
            finally:
                video.release()

            matches.sort(key=lambda m: m.similarity, reverse=True)
            payload = [
                {
                    "frame_index": int(m.frame_index),
                    "similarity": float(m.similarity),
                }
                for m in matches[:top_k]
            ]
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))

    def _ensure_backend(
        self, request: FrameSimilaritySearchRequest
    ) -> _EmbeddingBackend:
        key = _backend_key_for_request(request)
        if self._backend is None or self._backend_key != key:
            backend = _build_backend(request)
            self._backend = backend
            self._backend_key = tuple(backend.key)
        return self._backend

    def _query_embedding(
        self,
        *,
        backend: _EmbeddingBackend,
        video_path: Path,
        query_frame_index: int,
    ) -> np.ndarray:
        # Cache is per backend + per video + per query frame index.
        key = (tuple(backend.key), str(Path(video_path)), int(query_frame_index))
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        video = CV2Video(video_path)
        try:
            frame_rgb = video.load_frame(int(query_frame_index))
        finally:
            video.release()
        vec = backend.embed_frame_rgb(frame_rgb)
        self._query_cache[key] = vec
        return vec


class FrameSimilarityService(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal(list)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    matchFound = QtCore.Signal(int, float)
    _request_run = QtCore.Signal(object)
    _request_stop = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._thread = QtCore.QThread(self)
        self._worker = _FrameSimilarityWorker()
        self._worker.moveToThread(self._thread)
        self._request_run.connect(self._worker.run)
        self._request_stop.connect(self._worker.stop)

        self._worker.finished.connect(self._handle_finished)
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
        self._busy = True
        self.started.emit()
        self._request_run.emit(request)
        return True

    def stop(self) -> None:
        if not self._busy:
            return
        self._request_stop.emit()

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
            self.stop()
        except Exception:
            pass
        try:
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass
