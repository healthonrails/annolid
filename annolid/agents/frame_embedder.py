"""Frame indexing utilities powered by ONNX EmbeddingGemma 2.

Replaces the legacy import-time OpenCLIP/LanceDB global initialization with a
lazy, ONNX-first indexer (``OnnxFrameIndexer``) backed by portable ``.npz``
storage, while retaining an optional ``LanceDBFrameIndexer`` wrapper that also
stores EmbeddingGemma 2 ONNX vectors when ``lancedb`` is installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from PIL import Image

from annolid.infrastructure import AnnotationStoreError, load_labelme_json
from annolid.services.embedding_search import (
    DEFAULT_SEARCH_BACKEND,
    FolderFrameIndex,
    FrameEmbeddingBackend,
    build_embedding_backend,
    index_folder_frames,
    resolve_query_vector,
)

logger = logging.getLogger(__name__)

# Constants
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
ANNOTATION_EXTENSIONS = {".json"}


class Config:
    DB_PATH = "lancedb"
    TABLE_NAME = "video_frames"
    INDEX_PATH = "frame_embeddings.npz"
    MAX_INDEX_IMAGES = 1000
    BACKEND = DEFAULT_SEARCH_BACKEND
    QUANTIZATION = "q4"
    MATRYOSHKA_DIM: Optional[int] = None


@dataclass
class LanceDBFrame:
    """Lightweight frame record schema compatible with legacy imports."""

    image_uri: str
    flags: Optional[List[str]] = None
    caption: Optional[str] = None
    vector: Optional[List[float]] = None


def _extract_annotation_metadata(image_path: Path) -> Dict[str, Any]:
    """Read optional LabelMe flags and caption next to an image frame."""
    annotation_file = image_path.with_suffix(".json")
    if not annotation_file.exists():
        return {"flags": [], "caption": ""}
    try:
        annotation_data = load_labelme_json(annotation_file)
        raw_flags = annotation_data.get("flags", {}) or {}
        if isinstance(raw_flags, dict):
            flags = [str(k) for k, v in raw_flags.items() if v]
        elif isinstance(raw_flags, list):
            flags = [str(k) for k in raw_flags]
        else:
            flags = []
        caption = str(annotation_data.get("caption", "") or "")
        return {"flags": flags, "caption": caption}
    except (AnnotationStoreError, Exception) as exc:
        logger.error("Error reading annotation for %s: %s", image_path, exc)
        return {"flags": [], "caption": ""}


class OnnxFrameIndexer:
    """Portable ONNX EmbeddingGemma 2 frame indexer backed by ``FolderFrameIndex``."""

    def __init__(
        self,
        index_path: Union[str, Path] = Config.INDEX_PATH,
        *,
        backend: Union[str, FrameEmbeddingBackend] = Config.BACKEND,
        backend_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.index_path = Path(index_path).expanduser()
        self._owns_backend = not isinstance(backend, FrameEmbeddingBackend)
        self._backend_spec = backend
        self._backend_params = dict(backend_params or {})
        self._backend: Optional[FrameEmbeddingBackend] = (
            backend if isinstance(backend, FrameEmbeddingBackend) else None
        )
        self._index: Optional[FolderFrameIndex] = None

    def _get_backend(self) -> FrameEmbeddingBackend:
        if self._backend is None:
            self._backend = build_embedding_backend(
                str(self._backend_spec), self._backend_params
            )
        return self._backend

    def load_index(self) -> Optional[FolderFrameIndex]:
        if self._index is None and self.index_path.is_file():
            self._index = FolderFrameIndex.load_npz(self.index_path)
        return self._index

    def index_folder(
        self,
        folder_path: Union[str, Path],
        *,
        max_images: Optional[int] = Config.MAX_INDEX_IMAGES,
        batch_size: int = 8,
        recursive: bool = False,
    ) -> FolderFrameIndex:
        self._index = index_folder_frames(
            folder_path,
            output_path=self.index_path,
            backend=self._get_backend(),
            max_images=max_images,
            batch_size=batch_size,
            recursive=recursive,
        )
        return self._index

    async def index_frames(self, frame_data: List[Dict[str, Any]]) -> None:
        """Append frame records to the portable ONNX index and save."""
        if not frame_data:
            return
        backend = self._get_backend()
        existing = self.load_index()

        new_uris: List[str] = []
        new_flags: List[tuple[str, ...]] = []
        new_captions: List[str] = []
        new_images: List[np.ndarray] = []

        for row in frame_data:
            uri = str(row.get("image_uri") or "")
            if not uri:
                continue
            path = Path(uri)
            if not path.is_file():
                continue
            try:
                with Image.open(path) as opened:
                    new_images.append(np.asarray(opened.convert("RGB"), dtype=np.uint8))
            except Exception as exc:
                logger.warning("Skipping unreadable frame %s: %s", path, exc)
                continue
            new_uris.append(uri)
            raw_flags = row.get("flags") or []
            new_flags.append(tuple(str(f) for f in raw_flags))
            new_captions.append(str(row.get("caption") or ""))

        if not new_images:
            return

        new_vecs = backend.embed_frames_batch(new_images)
        if existing is not None and len(existing) > 0:
            merged_uris = existing.image_uris + tuple(new_uris)
            merged_flags = existing.flags + tuple(new_flags)
            merged_captions = existing.captions + tuple(new_captions)
            merged_vecs = np.concatenate([existing.embeddings, new_vecs], axis=0)
            metadata = dict(existing.metadata)
        else:
            merged_uris = tuple(new_uris)
            merged_flags = tuple(new_flags)
            merged_captions = tuple(new_captions)
            merged_vecs = new_vecs
            metadata = {"backend_key": list(backend.key)}

        self._index = FolderFrameIndex(
            image_uris=merged_uris,
            embeddings=merged_vecs,
            flags=merged_flags,
            captions=merged_captions,
            metadata=metadata,
        )
        self._index.save_npz(self.index_path)
        logger.info("Indexed %d frames into %s.", len(new_uris), self.index_path)

    async def search_similar_images(
        self, query_image: Any, limit: int = 3
    ) -> List[Dict[str, Any]]:
        index = self.load_index()
        if index is None or len(index) == 0:
            return []
        qvec, _ = resolve_query_vector(
            self._get_backend(), query_image=query_image, query_mode="frame"
        )
        matches = index.search(qvec, top_k=limit)
        return [m.to_dict() for m in matches]

    async def search_by_text(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        index = self.load_index()
        if index is None or len(index) == 0:
            return []
        qvec, _ = resolve_query_vector(
            self._get_backend(), query_text=query_text, query_mode="text"
        )
        matches = index.search(qvec, top_k=limit)
        return [m.to_dict() for m in matches]

    async def optimize_table(self) -> None:
        if self._index is not None:
            self._index.save_npz(self.index_path)

    def close(self) -> None:
        if self._owns_backend and self._backend is not None:
            self._backend.close()
            self._backend = None


class LanceDBFrameIndexer:
    """Async LanceDB frame indexer using ONNX EmbeddingGemma 2 embeddings."""

    def __init__(
        self,
        db_path: str = Config.DB_PATH,
        table_name: str = Config.TABLE_NAME,
        *,
        backend: Union[str, FrameEmbeddingBackend] = Config.BACKEND,
        backend_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.db_uri = db_path
        self.table_name = table_name
        self.db = None
        self.table = None
        self._owns_backend = not isinstance(backend, FrameEmbeddingBackend)
        self._backend_spec = backend
        self._backend_params = dict(backend_params or {})
        self._backend: Optional[FrameEmbeddingBackend] = (
            backend if isinstance(backend, FrameEmbeddingBackend) else None
        )

    def _get_backend(self) -> FrameEmbeddingBackend:
        if self._backend is None:
            self._backend = build_embedding_backend(
                str(self._backend_spec), self._backend_params
            )
        return self._backend

    async def get_async_client(self):
        if self.db is None:
            import lancedb  # type: ignore

            self.db = await lancedb.connect_async(self.db_uri)
        return self.db

    async def get_async_table(self):
        if self.table is None:
            db = await self.get_async_client()
            if self.table_name in await db.table_names():
                self.table = await db.open_table(self.table_name)
        return self.table

    async def index_frames(self, frame_data: List[Dict[str, Any]]) -> None:
        """Embed frame images with ONNX EmbeddingGemma 2 and store in LanceDB."""
        if not frame_data:
            return
        try:
            backend = self._get_backend()
            rows: List[Dict[str, Any]] = []
            images: List[np.ndarray] = []
            valid_meta: List[Dict[str, Any]] = []
            for item in frame_data:
                uri = str(item.get("image_uri") or "")
                if not uri or not Path(uri).is_file():
                    continue
                with Image.open(uri) as opened:
                    images.append(np.asarray(opened.convert("RGB"), dtype=np.uint8))
                valid_meta.append(item)

            if not images:
                return
            vectors = backend.embed_frames_batch(images)
            for item, vec in zip(valid_meta, vectors):
                rows.append(
                    {
                        "vector": [float(v) for v in vec.tolist()],
                        "image_uri": str(item.get("image_uri") or ""),
                        "flags": list(item.get("flags") or []),
                        "caption": str(item.get("caption") or ""),
                    }
                )

            db = await self.get_async_client()
            if self.table is None:
                if self.table_name in await db.table_names():
                    self.table = await db.open_table(self.table_name)
                    await self.table.add(rows)
                else:
                    self.table = await db.create_table(self.table_name, data=rows)
            else:
                await self.table.add(rows)
            logger.info(
                "Indexed %d frames into LanceDB '%s'.", len(rows), self.table_name
            )
        except Exception as exc:
            logger.error("Error indexing frames: %s", exc)

    async def search_similar_images(
        self, query_image: Any, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Search the LanceDB table by image or text query."""
        try:
            table = await self.get_async_table()
            if table is None:
                return []
            backend = self._get_backend()
            if isinstance(query_image, str) and not Path(query_image).is_file():
                qvec, _ = resolve_query_vector(
                    backend, query_text=query_image, query_mode="text"
                )
            else:
                qvec, _ = resolve_query_vector(
                    backend, query_image=query_image, query_mode="frame"
                )
            rows = await table.vector_search(qvec.tolist()).limit(limit).to_list()
            return [
                {
                    "image_uri": r.get("image_uri", ""),
                    "flags": r.get("flags") or [],
                    "caption": r.get("caption") or "",
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("Error searching images: %s", exc)
            return []

    async def optimize_table(self) -> None:
        try:
            table = await self.get_async_table()
            if table is not None:
                await table.optimize()
                logger.info("Optimization of LanceDB table is done.")
        except Exception as exc:
            logger.error("Error optimizing table: %s", exc)

    def close(self) -> None:
        if self._owns_backend and self._backend is not None:
            self._backend.close()
            self._backend = None


def _build_folder_monitor_handler(
    folder_path: Path,
    indexer: Union[OnnxFrameIndexer, LanceDBFrameIndexer],
    loop: Optional[asyncio.AbstractEventLoop] = None,
):
    from watchdog.events import FileSystemEventHandler  # type: ignore

    class _FolderMonitorHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self.folder_path = folder_path
            self.indexer = indexer
            self._loop = loop

        def on_created(self, event):
            if event.is_directory:
                return
            file_path = Path(event.src_path)
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                return
            coro = self.process_new_image(file_path)
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                asyncio.run(coro)

        async def process_new_image(self, file_path: Path) -> None:
            logger.info("New image detected: %s", file_path)
            meta = _extract_annotation_metadata(file_path)
            frame_data = [
                {
                    "image_uri": str(file_path),
                    "flags": meta["flags"],
                    "caption": meta["caption"],
                }
            ]
            await self.indexer.index_frames(frame_data)

    return _FolderMonitorHandler()


async def index_existing_images(
    folder_path: Path,
    indexer: Union[OnnxFrameIndexer, LanceDBFrameIndexer],
    *,
    max_images: int = Config.MAX_INDEX_IMAGES,
) -> None:
    """Index existing images and their associated annotations in a folder."""
    image_paths = sorted(
        f
        for f in folder_path.glob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.warning("No images found in folder: %s", folder_path)
        return

    if len(image_paths) > max_images:
        sampled = sample(image_paths, max_images)
    else:
        sampled = image_paths

    logger.info("Indexing %d images from folder %s.", len(sampled), folder_path)
    frame_data_list = []
    for image_file in sampled:
        meta = _extract_annotation_metadata(image_file)
        frame_data_list.append(
            {
                "image_uri": str(image_file),
                "flags": meta["flags"],
                "caption": meta["caption"],
            }
        )

    if frame_data_list:
        await indexer.index_frames(frame_data_list)


async def monitor_folder(
    folder_path: Path, indexer: Union[OnnxFrameIndexer, LanceDBFrameIndexer]
) -> None:
    """Monitor a folder for new images and index them asynchronously."""
    from watchdog.observers import Observer  # type: ignore

    loop = asyncio.get_running_loop()
    event_handler = _build_folder_monitor_handler(folder_path, indexer, loop=loop)
    observer = Observer()
    observer.schedule(event_handler, str(folder_path), recursive=False)
    observer.start()

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        observer.stop()
    finally:
        observer.stop()
        observer.join()


async def embed_frames(
    folder_path: str,
    *,
    output_index: Optional[str] = None,
    use_lancedb: bool = False,
    quantization: str = Config.QUANTIZATION,
    matryoshka_dim: Optional[int] = Config.MATRYOSHKA_DIM,
    watch: bool = False,
) -> None:
    """Index (and optionally monitor) a folder of frames using ONNX EmbeddingGemma 2."""
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Invalid folder path: %s", folder)
        return

    backend_params: Dict[str, Any] = {"quantization": quantization}
    if matryoshka_dim is not None:
        backend_params["matryoshka_dim"] = int(matryoshka_dim)

    indexer: Union[OnnxFrameIndexer, LanceDBFrameIndexer]
    if use_lancedb:
        indexer = LanceDBFrameIndexer(
            db_path=output_index or Config.DB_PATH,
            backend=Config.BACKEND,
            backend_params=backend_params,
        )
    else:
        default_out = output_index or str(folder / Config.INDEX_PATH)
        indexer = OnnxFrameIndexer(
            index_path=default_out,
            backend=Config.BACKEND,
            backend_params=backend_params,
        )

    try:
        await index_existing_images(folder, indexer)
        if watch:
            logger.info("Monitoring folder %s for new images...", folder)
            await monitor_folder(folder, indexer)
    except KeyboardInterrupt:
        logger.info("Shutting down folder indexing.")
    finally:
        await indexer.optimize_table()
        indexer.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Index and optionally monitor a folder of frames with ONNX EmbeddingGemma 2."
    )
    parser.add_argument("folder_path", help="Path to the folder containing images.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz index path (or LanceDB URI with --lancedb).",
    )
    parser.add_argument(
        "--quantization",
        default=Config.QUANTIZATION,
        choices=("q4", "quantized", "q4f16", "fp32", "fp16"),
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        default=None,
        choices=(128, 256, 512, 768),
    )
    parser.add_argument(
        "--lancedb",
        action="store_true",
        help="Store embeddings in LanceDB instead of a portable .npz index.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching folder_path for newly created frames.",
    )
    args = parser.parse_args()

    asyncio.run(
        embed_frames(
            args.folder_path,
            output_index=args.out,
            use_lancedb=bool(args.lancedb),
            quantization=args.quantization,
            matryoshka_dim=args.matryoshka_dim,
            watch=bool(args.watch),
        )
    )
