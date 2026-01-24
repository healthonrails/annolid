from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from qtpy import QtCore

from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.utils.annotation_store import AnnotationStore


@dataclass(frozen=True)
class DinoFrameSearchRequest:
    query_path: Path
    frame_paths: Sequence[Path]
    top_k: int = 25
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    short_side: int = 384
    device: Optional[str] = None


class _DinoFrameSearchWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self._extractor: Optional[Dinov3FeatureExtractor] = None
        self._index_key: Optional[Tuple[str, int, Optional[str]]] = None
        self._frame_paths: list[Path] = []
        self._frame_indices: list[int] = []
        self._embeddings: Optional[np.ndarray] = None

    @QtCore.Slot(object)
    # type: ignore[override]
    def search(self, request: DinoFrameSearchRequest) -> None:
        try:
            query_path = Path(request.query_path)
            frame_paths = [Path(p) for p in (request.frame_paths or [])]
            if not query_path.exists():
                raise FileNotFoundError(f"Query image does not exist: {query_path}")
            if not frame_paths:
                raise ValueError("No frame paths provided to DINO frame search.")

            top_k = max(1, int(request.top_k))
            self._ensure_index(
                frame_paths=frame_paths,
                model_name=str(request.model_name),
                short_side=int(request.short_side),
                device=str(request.device) if request.device else None,
            )
            if self._embeddings is None:
                raise RuntimeError("Frame embedding index is not available.")

            query_vec = self._embed_image(query_path)
            sims = self._embeddings @ query_vec
            order = np.argsort(-sims)

            results: list[dict] = []
            used = 0
            query_frame = AnnotationStore.frame_number_from_path(query_path)
            for idx in order.tolist():
                if idx < 0 or idx >= len(self._frame_paths):
                    continue
                frame_idx = self._frame_indices[idx]
                if query_frame is not None and frame_idx == int(query_frame):
                    continue
                score = float(sims[idx])
                results.append(
                    {
                        "image_uri": str(self._frame_paths[idx]),
                        "frame_index": int(frame_idx),
                        "similarity": score,
                    }
                )
                used += 1
                if used >= top_k:
                    break

            self.finished.emit(results)
        except Exception as exc:  # pragma: no cover - GUI surface
            self.error.emit(str(exc))

    def _ensure_index(
        self,
        *,
        frame_paths: Sequence[Path],
        model_name: str,
        short_side: int,
        device: Optional[str],
    ) -> None:
        key = (str(model_name), int(short_side), str(device) if device else None)
        if self._index_key != key:
            cfg = Dinov3Config(
                model_name=model_name, short_side=int(short_side), device=device
            )
            self._extractor = Dinov3FeatureExtractor(cfg)
            self._index_key = key
            self._frame_paths = []
            self._frame_indices = []
            self._embeddings = None

        resolved_paths: list[Path] = [Path(p) for p in frame_paths]
        if self._embeddings is not None and resolved_paths == self._frame_paths:
            return

        frame_indices: list[int] = []
        kept_paths: list[Path] = []
        for path in resolved_paths:
            idx = AnnotationStore.frame_number_from_path(path)
            if idx is None:
                continue
            if not path.exists():
                continue
            frame_indices.append(int(idx))
            kept_paths.append(path)

        if not kept_paths:
            raise ValueError("No usable frame paths for DINO frame search.")

        embeddings = np.zeros((len(kept_paths), 1), dtype=np.float32)
        sample = self._embed_image(kept_paths[0])
        dim = int(sample.shape[0])
        embeddings = np.zeros((len(kept_paths), dim), dtype=np.float32)
        embeddings[0] = sample

        total = len(kept_paths)
        self.progress.emit(1, total)
        for i in range(1, total):
            embeddings[i] = self._embed_image(kept_paths[i])
            if (i + 1) % 5 == 0 or i == total - 1:
                self.progress.emit(i + 1, total)

        self._frame_paths = kept_paths
        self._frame_indices = frame_indices
        self._embeddings = embeddings

    def _embed_image(self, path: Path) -> np.ndarray:
        if self._extractor is None:
            raise RuntimeError("DINO extractor not initialized.")
        img = Image.open(path).convert("RGB")
        feats = self._extractor.extract(
            img, return_type="numpy", return_layer="last", normalize=True
        )
        vec = feats.mean(axis=(1, 2)).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return vec


class DinoFrameSearchService(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal(list)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    _request_search = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._settings = getattr(parent, "settings", None)
        self._thread = QtCore.QThread(self)
        self._worker = _DinoFrameSearchWorker()
        self._worker.moveToThread(self._thread)
        self._request_search.connect(self._worker.search)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)
        self._worker.progress.connect(self.progress)
        self._thread.start()
        self._busy = False

    def is_busy(self) -> bool:
        return bool(self._busy)

    def request(self, request: DinoFrameSearchRequest) -> bool:
        if self._busy:
            return False
        settings = self._settings
        if settings is not None:
            try:
                default_model = "facebook/dinov3-vits16-pretrain-lvd1689m"
                model_name = str(
                    settings.value(
                        "patch_similarity/model",
                        default_model,
                    )
                )
                updates = {}
                if str(getattr(request, "model_name", "") or "") == default_model:
                    updates["model_name"] = model_name
                if updates:
                    request = replace(request, **updates)
            except Exception:
                pass

        self._busy = True
        self.started.emit()
        self._request_search.emit(request)
        return True

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
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass
