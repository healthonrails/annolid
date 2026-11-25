"""Frame and video behavior search using ONNX EmbeddingGemma 2.

Supports:
- Multimodal text-to-frame and image-to-frame queries over a portable ``.npz``
  ``FolderFrameIndex`` or an optional LanceDB table.
- Direct video behavior search (by text description or reference frame index)
  delegating to ``annolid.services.embedding_search.search_video_behavior``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import cv2
import numpy as np
from PIL import Image

from annolid.agents.frame_embedder import Config
from annolid.core.media.video import CV2Video
from annolid.services.embedding_search import (
    DEFAULT_SEARCH_BACKEND,
    FolderFrameIndex,
    FrameEmbeddingBackend,
    build_embedding_backend,
    resolve_query_vector,
    search_video_behavior,
)

logger = logging.getLogger(__name__)

# Lazy default search engine instance (created on first use, never at import time).
indexer: Optional["FrameSearchEngine"] = None


def cv2_to_pil(frame: np.ndarray, *, input_color: str = "rgb") -> Image.Image:
    """Convert a NumPy frame array (RGB or BGR) to a PIL Image in RGB mode."""
    if frame is None:
        raise ValueError("The input frame is None.")

    color_mode = str(input_color or "").strip().lower()
    rgb_frame = frame
    if color_mode == "bgr":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def _is_image_path(value: str) -> bool:
    try:
        p = Path(value).expanduser()
        return p.is_file()
    except OSError:
        return False


class FrameSearchEngine:
    """Unified frame index searcher supporting ONNX ``.npz`` indices and LanceDB."""

    def __init__(
        self,
        db_path: Union[str, Path] = Config.INDEX_PATH,
        table_name: str = Config.TABLE_NAME,
        *,
        backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
        backend_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.table_name = table_name
        self._owns_backend = not isinstance(backend, FrameEmbeddingBackend)
        self._backend_spec = backend
        self._backend_params = dict(backend_params or {})
        self._backend: Optional[FrameEmbeddingBackend] = (
            backend if isinstance(backend, FrameEmbeddingBackend) else None
        )
        self._npz_index: Optional[FolderFrameIndex] = None
        self.db = None
        self.table = None
        self.setup_table()

    def _get_backend(self) -> FrameEmbeddingBackend:
        if self._backend is None:
            self._backend = build_embedding_backend(
                str(self._backend_spec), self._backend_params
            )
        return self._backend

    def setup_table(self) -> None:
        """Attach to an existing ``.npz`` index or optional LanceDB table."""
        if self.db_path.is_file() and self.db_path.suffix.lower() == ".npz":
            self._npz_index = FolderFrameIndex.load_npz(self.db_path)
            logger.info(
                "Loaded ONNX frame index '%s' (%d frames).",
                self.db_path,
                len(self._npz_index),
            )
            return

        if self.db_path.exists():
            try:
                import lancedb  # type: ignore

                self.db = lancedb.connect(str(self.db_path))
                if self.table_name in self.db:
                    self.table = self.db[self.table_name]
                    logger.info(
                        "Connected to table '%s' in LanceDB (%s).",
                        self.table_name,
                        self.db_path,
                    )
                    return
            except Exception as exc:
                logger.warning("Unable to open LanceDB at %s: %s", self.db_path, exc)

        logger.warning(
            "No frame embedding index found at '%s' (table=%s).",
            self.db_path,
            self.table_name,
        )

    def _query_to_vector(
        self, query: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        backend = self._get_backend()
        if isinstance(query, Path) or (
            isinstance(query, str) and _is_image_path(query)
        ):
            qvec, _ = resolve_query_vector(
                backend, query_image=Path(query), query_mode="frame"
            )
            return qvec
        if isinstance(query, str):
            qvec, _ = resolve_query_vector(backend, query_text=query, query_mode="text")
            return qvec
        qvec, _ = resolve_query_vector(backend, query_image=query, query_mode="frame")
        return qvec

    def search_similar_images(
        self,
        query: Union[str, Path, Image.Image, np.ndarray],
        limit: int = 3,
        *,
        threshold: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """Search indexed frames by image, image path, or natural-language text."""
        if self._npz_index is not None:
            qvec = self._query_to_vector(query)
            matches = self._npz_index.search(qvec, top_k=limit, threshold=threshold)
            return [
                {
                    "image_uri": m.image_uri or "",
                    "flags": list(m.flags or ()),
                    "caption": m.caption or "",
                    "similarity": float(m.similarity),
                    "frame_index": int(m.frame_index),
                }
                for m in matches
            ]

        if self.table is None:
            logger.error("No frame index or LanceDB table connected.")
            return []

        try:
            qvec = self._query_to_vector(query)
            rows = self.table.search(qvec.tolist()).limit(int(limit)).to_list()
            return [
                {
                    "image_uri": str(r.get("image_uri") or ""),
                    "flags": list(r.get("flags") or []),
                    "caption": str(r.get("caption") or ""),
                    "similarity": float(1.0 - float(r["_distance"]))
                    if "_distance" in r
                    else None,
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("Error searching indexed frames: %s", exc)
            return []

    def close(self) -> None:
        if self._owns_backend and self._backend is not None:
            self._backend.close()
            self._backend = None


# Backward-compatible class name alias.
LanceDBFrameIndexer = FrameSearchEngine


def get_embedder(
    *,
    index_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    table_name: str = Config.TABLE_NAME,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
) -> FrameSearchEngine:
    """Return a cached default ``FrameSearchEngine`` or a configured instance."""
    global indexer
    if (
        index_path is not None
        or db_path is not None
        or isinstance(backend, FrameEmbeddingBackend)
        or backend != DEFAULT_SEARCH_BACKEND
        or backend_params
    ):
        target = (
            index_path if index_path is not None else (db_path or Config.INDEX_PATH)
        )
        return FrameSearchEngine(
            db_path=target,
            table_name=table_name,
            backend=backend,
            backend_params=backend_params,
        )

    if indexer is None:
        default_target: Union[str, Path] = (
            Config.INDEX_PATH if Path(Config.INDEX_PATH).is_file() else Config.DB_PATH
        )
        indexer = FrameSearchEngine(db_path=default_target, table_name=table_name)
    return indexer


def search_frames(
    query_image: Union[str, Path, Image.Image, np.ndarray],
    limit: int = 3,
    *,
    index_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    table_name: str = Config.TABLE_NAME,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    threshold: float = -1.0,
) -> List[Dict[str, Any]]:
    """Search indexed frames by image, image path, or natural-language behavior text."""
    owns_engine = (
        index_path is not None
        or db_path is not None
        or isinstance(backend, FrameEmbeddingBackend)
        or backend != DEFAULT_SEARCH_BACKEND
        or bool(backend_params)
    )
    engine = get_embedder(
        index_path=index_path,
        db_path=db_path,
        table_name=table_name,
        backend=backend,
        backend_params=backend_params,
    )
    try:
        results = engine.search_similar_images(
            query_image, limit=limit, threshold=threshold
        )
    finally:
        if owns_engine:
            engine.close()

    formatted: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, dict):
            entry = {
                "image_uri": res.get("image_uri", ""),
                "flags": res.get("flags") or [],
                "caption": res.get("caption") or "",
            }
            if res.get("similarity") is not None:
                entry["similarity"] = float(res["similarity"])
            if res.get("frame_index") is not None:
                entry["frame_index"] = int(res["frame_index"])
            formatted.append(entry)
        else:
            formatted.append(
                {
                    "image_uri": getattr(res, "image_uri", ""),
                    "flags": getattr(res, "flags", None) or [],
                    "caption": getattr(res, "caption", None) or "",
                }
            )
    return formatted


def search_video(
    video_path: Union[str, Path],
    frame_skip: int = 1,
    search_limit: int = 1,
    *,
    query_text: Optional[str] = None,
    query_frame_index: Optional[int] = None,
    index_path: Optional[Union[str, Path]] = None,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    threshold: float = 0.25,
    clip_frames: int = 1,
) -> List[Dict[str, Any]]:
    """Search a video by behavior text, reference frame, or an indexed frame store."""
    if query_text or query_frame_index is not None:
        result = search_video_behavior(
            video_path,
            query_text=query_text,
            query_frame_index=query_frame_index,
            backend=backend,
            backend_params=backend_params,
            stride=max(1, int(frame_skip)),
            top_k=max(1, int(search_limit)),
            threshold=float(threshold),
            clip_frames=max(1, int(clip_frames)),
        )
        return [m.to_dict() for m in result.matches]

    video_loader = CV2Video(video_path)
    matches_out: List[Dict[str, Any]] = []
    # Reuse a single FrameSearchEngine across all sampled frames instead of
    # reloading the .npz index and recreating ONNX Runtime sessions per frame.
    owns_engine = (
        index_path is not None
        or isinstance(backend, FrameEmbeddingBackend)
        or backend != DEFAULT_SEARCH_BACKEND
        or bool(backend_params)
    )
    engine = get_embedder(
        index_path=index_path,
        backend=backend,
        backend_params=backend_params,
    )
    try:
        num_frames = int(video_loader.total_frames())
        step = max(1, int(frame_skip))
        for i in range(0, num_frames, step):
            try:
                frame = video_loader.load_frame(i)
            except KeyError:
                break
            if frame is None:
                continue
            pil_img = cv2_to_pil(frame)
            raw_results = engine.search_similar_images(
                pil_img, limit=search_limit, threshold=float(threshold)
            )
            results: List[Dict[str, Any]] = []
            for res in raw_results:
                if isinstance(res, dict):
                    entry: Dict[str, Any] = {
                        "image_uri": res.get("image_uri", ""),
                        "flags": res.get("flags") or [],
                        "caption": res.get("caption") or "",
                    }
                    if res.get("similarity") is not None:
                        entry["similarity"] = float(res["similarity"])
                    if res.get("frame_index") is not None:
                        entry["frame_index"] = int(res["frame_index"])
                    results.append(entry)
                else:
                    results.append(
                        {
                            "image_uri": getattr(res, "image_uri", ""),
                            "flags": getattr(res, "flags", None) or [],
                            "caption": getattr(res, "caption", None) or "",
                        }
                    )
            matches_out.append({"frame_index": i, "results": results})
    finally:
        if owns_engine:
            engine.close()
        video_loader.release()
    return matches_out


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Search video frames or a frame index using ONNX EmbeddingGemma 2."
    )
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument(
        "--query",
        default=None,
        help="Behavior text query (e.g. 'mouse grooming').",
    )
    parser.add_argument(
        "--query-frame",
        type=int,
        default=None,
        help="Reference frame index for visual similarity search.",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--clip-frames", type=int, default=1)
    parser.add_argument(
        "--quantization",
        default="q4",
        choices=("q4", "quantized", "q4f16", "fp32", "fp16"),
    )
    args = parser.parse_args()

    out = search_video(
        args.video_path,
        frame_skip=args.stride,
        search_limit=args.top_k,
        query_text=args.query,
        query_frame_index=args.query_frame,
        threshold=args.threshold,
        clip_frames=args.clip_frames,
        backend_params={"quantization": args.quantization},
    )
    print(json.dumps(out, indent=2))
