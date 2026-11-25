"""Shared frame and behavior embedding search services.

Provides a unified, framework-agnostic service layer used by the Annolid GUI,
CLI, agent tools, and timeline scripts:
- ONNX EmbeddingGemma 2 (default), DINOv3, and Qwen3-VL backends behind a
  common ``FrameEmbeddingBackend`` contract.
- Vectorized single-frame, multi-frame temporal clip, text-to-frame, and
  hybrid (frame + text) behavior search.
- Reusable in-memory and ``.npz``-backed frame embedding indices so repeat
  text or exemplar queries on the same video run in milliseconds.
- Deterministic annotation-overlay rendering and contiguous behavior bout
  grouping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
from PIL import Image

from annolid.core.media.video import CV2Video
from annolid.core.models.adapters.embeddinggemma2_onnx import (
    DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO,
    DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION,
    EmbeddingGemma2OnnxAdapter,
    EmbeddingGemma2OnnxConfig,
    l2_normalize_np,
    normalize_quantization_preset,
)
from annolid.infrastructure import AnnotationStore, load_labelme_json

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_BACKEND = "embeddinggemma2_onnx"
DEFAULT_DINOV3_SEARCH_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_QWEN3VL_SEARCH_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
IMAGE_FILE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"})
DEFAULT_EMBEDDING_CACHE_MAX_FILES = 32
DEFAULT_EMBEDDING_CACHE_MAX_BYTES = 256 * 1024 * 1024


def resolve_default_embedding_cache_dir() -> Path:
    """Return the persistent directory for cached video frame embedding ``.npz`` indices."""
    env_dir = str(os.getenv("ANNOLID_EMBEDDING_CACHE_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "annolid" / "embedding_search"


def default_video_index_cache_path(
    *,
    video_path: Union[str, Path],
    backend_key: Sequence[str],
    stride: int,
    max_frames: Optional[int],
    clip_frames: int,
    overlay_shapes: bool = False,
    annotation_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Compute a deterministic content-addressed ``.npz`` cache path for a video scan."""
    resolved = Path(video_path).expanduser().resolve()
    try:
        stat = resolved.stat()
        mtime_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    except OSError:
        mtime_ns = 0
        file_size = 0
    fingerprint = json.dumps(
        {
            "video": str(resolved),
            "mtime_ns": mtime_ns,
            "size": file_size,
            "backend_key": [str(part) for part in backend_key],
            "stride": max(1, int(stride)),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "clip_frames": max(1, int(clip_frames)),
            "overlay_shapes": bool(overlay_shapes),
            "annotation_dir": (
                str(Path(annotation_dir).expanduser().resolve())
                if overlay_shapes and annotation_dir is not None
                else ""
            ),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    stem = (
        "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in resolved.stem)[:32]
        or "video"
    )
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else resolve_default_embedding_cache_dir()
    )
    return root / f"{stem}_{digest}.npz"


def prune_embedding_cache_dir(
    cache_dir: Union[str, Path],
    *,
    max_files: int = DEFAULT_EMBEDDING_CACHE_MAX_FILES,
    max_bytes: int = DEFAULT_EMBEDDING_CACHE_MAX_BYTES,
) -> int:
    """Evict oldest ``.npz`` cache files so disk usage stays within bounded limits."""
    root = Path(cache_dir).expanduser()
    if not root.is_dir():
        return 0
    entries: List[Tuple[float, int, Path]] = []
    for path in root.glob("*.npz"):
        try:
            stat = path.stat()
            entries.append((float(stat.st_mtime), int(stat.st_size), path))
        except OSError:
            continue
    # Oldest first for LRU-like eviction.
    entries.sort(key=lambda item: item[0])
    total_bytes = sum(item[1] for item in entries)
    total_files = len(entries)
    removed = 0
    file_limit = max(1, int(max_files))
    byte_limit = max(1, int(max_bytes))
    for _mtime, size, path in entries:
        if total_files <= file_limit and total_bytes <= byte_limit:
            break
        try:
            path.unlink()
            removed += 1
            total_files -= 1
            total_bytes -= size
        except OSError:
            continue
    return removed


@dataclass(frozen=True)
class EmbeddingSearchMatch:
    frame_index: int
    similarity: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    label: Optional[str] = None
    image_uri: Optional[str] = None
    flags: Optional[Tuple[str, ...]] = None
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "frame_index": int(self.frame_index),
            "similarity": float(self.similarity),
        }
        if self.start_frame is not None:
            payload["start_frame"] = int(self.start_frame)
        if self.end_frame is not None:
            payload["end_frame"] = int(self.end_frame)
        if self.label is not None:
            payload["label"] = str(self.label)
        if self.image_uri is not None:
            payload["image_uri"] = str(self.image_uri)
        if self.flags is not None:
            payload["flags"] = list(self.flags)
        if self.caption is not None:
            payload["caption"] = str(self.caption)
        return payload


@dataclass(frozen=True)
class BehaviorSearchBout:
    """Contiguous candidate behavior interval merged from nearby frame matches."""

    start_frame: int
    end_frame: int
    peak_frame: int
    peak_similarity: float
    mean_similarity: float
    match_count: int
    label: Optional[str] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "peak_frame": int(self.peak_frame),
            "peak_similarity": float(self.peak_similarity),
            "mean_similarity": float(self.mean_similarity),
            "match_count": int(self.match_count),
        }
        if self.label is not None:
            payload["label"] = str(self.label)
        if self.start_time_sec is not None:
            payload["start_time_sec"] = round(float(self.start_time_sec), 3)
        if self.end_time_sec is not None:
            payload["end_time_sec"] = round(float(self.end_time_sec), 3)
        return payload


@dataclass(frozen=True)
class ScoreSummary:
    """Distribution statistics of cosine similarities across indexed frames."""

    count: int
    min_similarity: float
    median_similarity: float
    mean_similarity: float
    std_similarity: float
    p90_similarity: float
    max_similarity: float
    best_frame_index: Optional[int]
    effective_threshold: float
    threshold_mode: str = "fixed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "min_similarity": round(float(self.min_similarity), 4),
            "median_similarity": round(float(self.median_similarity), 4),
            "mean_similarity": round(float(self.mean_similarity), 4),
            "std_similarity": round(float(self.std_similarity), 4),
            "p90_similarity": round(float(self.p90_similarity), 4),
            "max_similarity": round(float(self.max_similarity), 4),
            "best_frame_index": (
                int(self.best_frame_index)
                if self.best_frame_index is not None
                else None
            ),
            "effective_threshold": round(float(self.effective_threshold), 4),
            "threshold_mode": str(self.threshold_mode),
        }


def recommended_similarity_threshold(
    backend: Optional[str] = DEFAULT_SEARCH_BACKEND,
    query_mode: str = "text",
) -> float:
    """Return empirically calibrated default cosine threshold per backend & query mode.

    Calibrated against ``onnx-community/embeddinggemma-2-ONNX`` where unrelated
    text-to-image negatives score ~0.54-0.62, positive matches score ~0.68-0.78,
    and same-modality frame matches score ~0.82-0.99.
    """
    canonical = normalize_backend_name(backend)
    mode = str(query_mode or "text").strip().lower()
    if canonical == "embeddinggemma2_onnx":
        if mode in {"frame", "image"}:
            return 0.82
        if mode == "hybrid":
            return 0.72
        return 0.66
    return 0.35


def compute_adaptive_threshold(
    scores: np.ndarray,
    *,
    backend: Optional[str] = DEFAULT_SEARCH_BACKEND,
    query_mode: str = "text",
    min_z: float = 1.0,
) -> float:
    """Compute a robust adaptive threshold from a video's similarity distribution.

    Uses ``median + max(min_z * 1.4826 * MAD, min_margin)`` floored by a gentle
    backend-specific minimum so unrelated negative queries do not match
    background noise while valid behaviors above the video's baseline are
    isolated automatically.
    """
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    default_thr = recommended_similarity_threshold(backend, query_mode)
    if arr.size < 3:
        return float(default_thr)

    std = float(np.std(arr))
    if std < 1e-4:
        return float(default_thr)

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_std = 1.4826 * mad if mad > 1e-6 else std

    canonical = normalize_backend_name(backend)
    mode = str(query_mode or "text").strip().lower()
    if canonical == "embeddinggemma2_onnx":
        if mode in {"frame", "image"}:
            floor, min_margin = 0.78, 0.025
        elif mode == "hybrid":
            floor, min_margin = 0.68, 0.025
        else:
            floor, min_margin = 0.64, 0.025
    else:
        floor, min_margin = 0.25, 0.04

    max_val = float(np.max(arr))
    if max_val < floor:
        return float(floor)
    if (max_val - med) < min_margin:
        # Nearly uniform scores across the video: fall back to calibrated default
        # so flat negative scores return 0 matches while uniformly high videos match.
        return float(default_thr)

    upper = max(med + min_margin, med + 0.5 * (max_val - med))
    adaptive = min(med + max(float(min_z) * robust_std, min_margin), upper)
    return float(max(adaptive, floor))


def summarize_similarity_scores(
    frame_indices: np.ndarray,
    scores: np.ndarray,
    *,
    effective_threshold: float,
    threshold_mode: str = "fixed",
    exclude_frame_index: Optional[int] = None,
) -> ScoreSummary:
    """Summarize full-video similarity scores for diagnostics and UI feedback."""
    idx_arr = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    scr_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if exclude_frame_index is not None and idx_arr.size > 0:
        keep = idx_arr != int(exclude_frame_index)
        idx_arr = idx_arr[keep]
        scr_arr = scr_arr[keep]

    if scr_arr.size == 0:
        return ScoreSummary(
            count=0,
            min_similarity=0.0,
            median_similarity=0.0,
            mean_similarity=0.0,
            std_similarity=0.0,
            p90_similarity=0.0,
            max_similarity=0.0,
            best_frame_index=None,
            effective_threshold=float(effective_threshold),
            threshold_mode=str(threshold_mode),
        )

    best_pos = int(np.argmax(scr_arr))
    return ScoreSummary(
        count=int(scr_arr.size),
        min_similarity=float(np.min(scr_arr)),
        median_similarity=float(np.median(scr_arr)),
        mean_similarity=float(np.mean(scr_arr)),
        std_similarity=float(np.std(scr_arr)),
        p90_similarity=float(np.percentile(scr_arr, 90)),
        max_similarity=float(scr_arr[best_pos]),
        best_frame_index=int(idx_arr[best_pos]),
        effective_threshold=float(effective_threshold),
        threshold_mode=str(threshold_mode),
    )


@dataclass
class VideoFrameEmbeddingIndex:
    """Reusable L2-normalized embedding matrix for sampled video frames/clips."""

    frame_indices: np.ndarray  # [N] int64
    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    metadata: Dict[str, Any] = field(default_factory=dict)
    clip_spans: Optional[np.ndarray] = None  # [N, 2] int64 (start_frame, end_frame)

    def __post_init__(self) -> None:
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64).reshape(-1)
        emb = np.asarray(self.embeddings, dtype=np.float32)
        if emb.ndim == 1 and emb.size == 0:
            emb = emb.reshape(0, 0)
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[0] != self.frame_indices.shape[0]:
            raise ValueError(
                f"frame_indices length ({self.frame_indices.shape[0]}) does not match "
                f"embeddings rows ({emb.shape[0]})."
            )
        if emb.shape[0] > 0 and emb.shape[1] > 0:
            emb = l2_normalize_np(emb, axis=1)
        self.embeddings = emb
        if self.clip_spans is not None:
            spans = np.asarray(self.clip_spans, dtype=np.int64).reshape(-1, 2)
            if spans.shape[0] != self.frame_indices.shape[0]:
                raise ValueError("clip_spans rows must match frame_indices length.")
            self.clip_spans = spans

    def __len__(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def score_query(self, query_vector: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between ``query_vector`` and all indexed frames."""
        if len(self) == 0:
            return np.zeros((0,), dtype=np.float32)
        qvec = l2_normalize_np(np.asarray(query_vector, dtype=np.float32).reshape(-1))
        if qvec.shape[0] != self.dim:
            raise ValueError(
                f"Query embedding dimension ({qvec.shape[0]}) does not match "
                f"index dimension ({self.dim})."
            )
        return (self.embeddings @ qvec).astype(np.float32, copy=False)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        threshold: float = 0.25,
        top_k: int = 50,
        exclude_frame_index: Optional[int] = None,
        label: Optional[str] = None,
    ) -> List[EmbeddingSearchMatch]:
        if len(self) == 0:
            return []
        scores = self.score_query(query_vector)
        order = np.argsort(-scores)
        thr = float(threshold)
        limit = max(1, int(top_k))
        matches: List[EmbeddingSearchMatch] = []
        for pos in order.tolist():
            sim = float(scores[pos])
            if sim < thr:
                break
            frame_idx = int(self.frame_indices[pos])
            if exclude_frame_index is not None and frame_idx == int(
                exclude_frame_index
            ):
                continue
            start_f: Optional[int] = None
            end_f: Optional[int] = None
            if self.clip_spans is not None:
                start_f = int(self.clip_spans[pos, 0])
                end_f = int(self.clip_spans[pos, 1])
            matches.append(
                EmbeddingSearchMatch(
                    frame_index=frame_idx,
                    similarity=sim,
                    start_frame=start_f,
                    end_frame=end_f,
                    label=label,
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def score_labels(
        self,
        label_vectors: Mapping[str, np.ndarray],
        *,
        top_n: int = 3,
        min_similarity: float = -1.0,
    ) -> List[Dict[str, Any]]:
        """Score every indexed frame against a dictionary of behavior label vectors."""
        if len(self) == 0 or not label_vectors:
            return []
        labels = list(label_vectors.keys())
        mat = np.stack(
            [
                l2_normalize_np(
                    np.asarray(label_vectors[lbl], dtype=np.float32).reshape(-1)
                )
                for lbl in labels
            ],
            axis=0,
        )
        # [N_frames, N_labels]
        sims = self.embeddings @ mat.T
        limit = max(1, int(top_n))
        min_sim = float(min_similarity)
        rows: List[Dict[str, Any]] = []
        for row_idx in range(len(self)):
            row_scores = sims[row_idx]
            ranked = np.argsort(-row_scores)[:limit]
            top_behaviors = [
                {
                    "label": labels[int(j)],
                    "similarity": float(row_scores[int(j)]),
                }
                for j in ranked.tolist()
                if float(row_scores[int(j)]) >= min_sim
            ]
            entry: Dict[str, Any] = {
                "frame_index": int(self.frame_indices[row_idx]),
                "behaviors": top_behaviors,
            }
            if self.clip_spans is not None:
                entry["start_frame"] = int(self.clip_spans[row_idx, 0])
                entry["end_frame"] = int(self.clip_spans[row_idx, 1])
            rows.append(entry)
        return rows

    def save_npz(self, path: Union[str, Path]) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        meta_json = np.asarray(
            [json.dumps(self.metadata, sort_keys=True)], dtype=np.str_
        )
        payload: Dict[str, Any] = {
            "frame_indices": self.frame_indices.astype(np.int64, copy=False),
            "embeddings": self.embeddings.astype(np.float32, copy=False),
            "metadata_json": meta_json,
        }
        if self.clip_spans is not None:
            payload["clip_spans"] = self.clip_spans.astype(np.int64, copy=False)
        np.savez_compressed(str(target), **payload)
        return target

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "VideoFrameEmbeddingIndex":
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Embedding index file not found: {source}")
        with np.load(str(source), allow_pickle=False) as data:
            frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            raw_meta = data["metadata_json"][0] if "metadata_json" in data else "{}"
            metadata = json.loads(str(raw_meta))
            clip_spans = (
                np.asarray(data["clip_spans"], dtype=np.int64)
                if "clip_spans" in data
                else None
            )
        return cls(
            frame_indices=frame_indices,
            embeddings=embeddings,
            metadata=metadata if isinstance(metadata, dict) else {},
            clip_spans=clip_spans,
        )


@dataclass
class FolderFrameIndex:
    """Portable ONNX embedding index for a directory of frame images + annotations."""

    image_uris: Tuple[str, ...]
    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    flags: Tuple[Tuple[str, ...], ...] = ()
    captions: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_uris = tuple(str(u) for u in self.image_uris)
        emb = np.asarray(self.embeddings, dtype=np.float32)
        if emb.ndim == 1 and emb.size == 0:
            emb = emb.reshape(0, 0)
        elif emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[0] != len(self.image_uris):
            raise ValueError("image_uris length must match embeddings rows.")
        if emb.shape[0] > 0 and emb.shape[1] > 0:
            emb = l2_normalize_np(emb, axis=1)
        self.embeddings = emb
        if not self.flags:
            self.flags = tuple(() for _ in self.image_uris)
        else:
            self.flags = tuple(tuple(str(f) for f in row) for row in self.flags)
        if not self.captions:
            self.captions = tuple("" for _ in self.image_uris)
        else:
            self.captions = tuple(str(c or "") for c in self.captions)

    def __len__(self) -> int:
        return len(self.image_uris)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        threshold: float = -1.0,
    ) -> List[EmbeddingSearchMatch]:
        if len(self) == 0:
            return []
        qvec = l2_normalize_np(np.asarray(query_vector, dtype=np.float32).reshape(-1))
        scores = self.embeddings @ qvec
        order = np.argsort(-scores)
        limit = max(1, int(top_k))
        thr = float(threshold)
        results: List[EmbeddingSearchMatch] = []
        for pos in order.tolist():
            sim = float(scores[pos])
            if sim < thr:
                break
            uri = self.image_uris[pos]
            frame_idx = AnnotationStore.frame_number_from_path(Path(uri))
            results.append(
                EmbeddingSearchMatch(
                    frame_index=int(frame_idx) if frame_idx is not None else int(pos),
                    similarity=sim,
                    image_uri=uri,
                    flags=self.flags[pos],
                    caption=self.captions[pos],
                )
            )
            if len(results) >= limit:
                break
        return results

    def save_npz(self, path: Union[str, Path]) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "image_uri": uri,
                "flags": list(flg),
                "caption": cap,
            }
            for uri, flg, cap in zip(self.image_uris, self.flags, self.captions)
        ]
        np.savez_compressed(
            str(target),
            embeddings=self.embeddings.astype(np.float32, copy=False),
            records_json=np.asarray([json.dumps(records)], dtype=np.str_),
            metadata_json=np.asarray(
                [json.dumps(self.metadata, sort_keys=True)], dtype=np.str_
            ),
        )
        return target

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "FolderFrameIndex":
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Folder embedding index not found: {source}")
        with np.load(str(source), allow_pickle=False) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            records = json.loads(str(data["records_json"][0]))
            meta_raw = data["metadata_json"][0] if "metadata_json" in data else "{}"
            metadata = json.loads(str(meta_raw))
        image_uris = tuple(str(r.get("image_uri", "")) for r in records)
        flags = tuple(tuple(str(f) for f in (r.get("flags") or [])) for r in records)
        captions = tuple(str(r.get("caption") or "") for r in records)
        return cls(
            image_uris=image_uris,
            embeddings=embeddings,
            flags=flags,
            captions=captions,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


def normalize_backend_name(backend: Optional[str]) -> str:
    """Normalize backend aliases into canonical service identifiers."""
    raw = str(backend or DEFAULT_SEARCH_BACKEND).strip().lower()
    if raw in {
        "embeddinggemma2_onnx",
        "embeddinggemma2",
        "embeddinggemma",
        "embeddinggemma-2",
        "embeddinggemma-2-onnx",
        "gemma2_onnx",
        "onnx",
    }:
        return "embeddinggemma2_onnx"
    if raw in {"dinov3", "dino", "dino_v3"}:
        return "dinov3"
    if raw in {"qwen3vl", "qwen3_vl", "qwen3"}:
        return "qwen3vl"
    raise ValueError(
        f"Unknown embedding backend: {backend!r}. "
        "Supported backends: 'embeddinggemma2_onnx', 'dinov3', 'qwen3vl'."
    )


class FrameEmbeddingBackend(ABC):
    """Unified interface for frame, clip, and optional text embedding backends."""

    supports_text: bool = False
    supports_video_clip: bool = False

    @property
    @abstractmethod
    def key(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_frames_batch(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if not frames_rgb:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self.embed_frame_rgb(frame) for frame in frames_rgb], axis=0)

    def embed_text(
        self, text: str, *, prompt_name: Optional[str] = "SearchQuery"
    ) -> np.ndarray:
        _ = (text, prompt_name)
        raise NotImplementedError(
            f"Backend {self.key[0]!r} does not support text queries. "
            "Switch to 'embeddinggemma2_onnx' or 'qwen3vl'."
        )

    def embed_texts_batch(
        self, texts: Sequence[str], *, prompt_name: Optional[str] = "SearchQuery"
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(
            [self.embed_text(t, prompt_name=prompt_name) for t in texts], axis=0
        )

    def embed_video_clip(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        if not frames_rgb:
            raise ValueError("embed_video_clip requires at least one frame.")
        vecs = self.embed_frames_batch(frames_rgb)
        return l2_normalize_np(vecs.mean(axis=0))

    def close(self) -> None:
        return None


class EmbeddingGemma2OnnxBackend(FrameEmbeddingBackend):
    """Pure ONNX Runtime EmbeddingGemma 2 backend (text + image + video clip)."""

    supports_text: bool = True
    supports_video_clip: bool = True

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO,
        quantization: str = DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION,
        matryoshka_dim: Optional[int] = None,
        image_max_soft_tokens: int = 140,
        video_max_soft_tokens: int = 70,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        providers: Optional[Sequence[str]] = None,
        adapter: Optional[EmbeddingGemma2OnnxAdapter] = None,
    ) -> None:
        if adapter is not None:
            self._adapter = adapter
            cfg = adapter.config
        else:
            cfg = EmbeddingGemma2OnnxConfig(
                model_id=model_id,
                quantization=quantization,
                matryoshka_dim=matryoshka_dim,
                image_max_soft_tokens=image_max_soft_tokens,
                video_max_soft_tokens=video_max_soft_tokens,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                providers=tuple(providers) if providers else None,
            )
            self._adapter = EmbeddingGemma2OnnxAdapter(config=cfg, load_vision=True)
        self._key = (
            "embeddinggemma2_onnx",
            str(cfg.model_id),
            str(cfg.quantization),
            str(cfg.matryoshka_dim or ""),
            str(cfg.image_max_soft_tokens),
            str(cfg.video_max_soft_tokens),
            ",".join(cfg.providers or ()),
        )

    @property
    def adapter(self) -> EmbeddingGemma2OnnxAdapter:
        return self._adapter

    @property
    def key(self) -> Tuple[str, ...]:
        return self._key

    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self._adapter.embed_images([frame_rgb])[0]

    def embed_frames_batch(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        return self._adapter.embed_images(frames_rgb)

    def embed_text(
        self, text: str, *, prompt_name: Optional[str] = "SearchQuery"
    ) -> np.ndarray:
        return self._adapter.embed_texts([text], prompt_name=prompt_name)[0]

    def embed_texts_batch(
        self, texts: Sequence[str], *, prompt_name: Optional[str] = "SearchQuery"
    ) -> np.ndarray:
        return self._adapter.embed_texts(texts, prompt_name=prompt_name)

    def embed_video_clip(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray:
        return self._adapter.embed_video_clip(frames_rgb)

    def close(self) -> None:
        self._adapter.close()


class Dinov3EmbeddingBackend(FrameEmbeddingBackend):
    """DINOv3 / Google vision dense-feature average-pooled backend (image-only)."""

    supports_text: bool = False
    supports_video_clip: bool = False

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DINOV3_SEARCH_MODEL,
        short_side: int = 384,
        device: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> None:
        from annolid.features import Dinov3Config, Dinov3FeatureExtractor

        cfg = Dinov3Config(
            model_name=model_name,
            short_side=int(short_side),
            device=device,
            matryoshka_dim=matryoshka_dim,
        )
        self._extractor = Dinov3FeatureExtractor(cfg)
        self._key = (
            "dinov3",
            str(model_name),
            str(int(short_side)),
            str(device or ""),
            str(matryoshka_dim or ""),
        )

    @property
    def key(self) -> Tuple[str, ...]:
        return self._key

    def embed_frame_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        feats = self._extractor.extract(
            frame_rgb, return_type="numpy", return_layer="last", normalize=True
        )
        vec = feats.mean(axis=(1, 2)).astype(np.float32, copy=False)
        return l2_normalize_np(vec)


class Qwen3VLEmbeddingBackend(FrameEmbeddingBackend):
    """Qwen3-VL multimodal PyTorch embedding backend."""

    supports_text: bool = True
    supports_video_clip: bool = True

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN3VL_SEARCH_MODEL,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
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
        return l2_normalize_np(np.asarray(vec, dtype=np.float32))

    def embed_text(
        self, text: str, *, prompt_name: Optional[str] = "SearchQuery"
    ) -> np.ndarray:
        _ = prompt_name
        resp = self._adapter.predict(self._ModelRequest(task="embed", text=str(text)))
        vec = (resp.output or {}).get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("Qwen3VL embedding adapter returned no embedding.")
        return l2_normalize_np(np.asarray(vec, dtype=np.float32))

    def close(self) -> None:
        self._adapter.close()


def backend_cache_key(
    backend: Optional[str], backend_params: Optional[Mapping[str, Any]] = None
) -> Tuple[str, ...]:
    """Compute a deterministic cache key for a backend configuration."""
    canonical = normalize_backend_name(backend)
    params = dict(backend_params or {})
    if canonical == "embeddinggemma2_onnx":
        model_id = str(
            params.get("model_id") or DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO
        ).strip()
        quant = normalize_quantization_preset(
            params.get("quantization") or DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION
        )
        mrl = params.get("matryoshka_dim")
        mrl_str = str(int(mrl)) if mrl not in (None, "", 0) else ""
        img_tokens = int(params.get("image_max_soft_tokens") or 140)
        vid_tokens = int(params.get("video_max_soft_tokens") or 70)
        providers = params.get("providers") or ()
        if isinstance(providers, str):
            providers_str = providers.strip()
        else:
            providers_str = ",".join(
                str(p).strip() for p in providers if str(p).strip()
            )
        return (
            "embeddinggemma2_onnx",
            model_id,
            quant,
            mrl_str,
            str(img_tokens),
            str(vid_tokens),
            providers_str,
        )
    if canonical == "dinov3":
        model_name = str(params.get("model_name") or DEFAULT_DINOV3_SEARCH_MODEL)
        short_side = int(params.get("short_side") or 384)
        device = str(params.get("device") or "").strip()
        mrl = params.get("matryoshka_dim")
        mrl_str = str(int(mrl)) if mrl not in (None, "", 0) else ""
        return ("dinov3", model_name, str(short_side), device, mrl_str)
    model_id = str(params.get("model_id") or DEFAULT_QWEN3VL_SEARCH_MODEL)
    torch_dtype = str(params.get("torch_dtype") or "").strip()
    attn_impl = str(params.get("attn_implementation") or "").strip()
    return ("qwen3vl", model_id, torch_dtype, attn_impl)


def build_embedding_backend(
    backend: Optional[str] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
) -> FrameEmbeddingBackend:
    """Instantiate a ``FrameEmbeddingBackend`` from name + parameters."""
    canonical = normalize_backend_name(backend)
    params = dict(backend_params or {})
    if canonical == "embeddinggemma2_onnx":
        mrl = params.get("matryoshka_dim")
        mrl_int = int(mrl) if mrl not in (None, "", 0) else None
        providers_raw = params.get("providers")
        providers: Optional[Sequence[str]] = None
        if isinstance(providers_raw, str) and providers_raw.strip():
            providers = [p.strip() for p in providers_raw.split(",") if p.strip()]
        elif isinstance(providers_raw, (list, tuple)):
            providers = [str(p).strip() for p in providers_raw if str(p).strip()]
        return EmbeddingGemma2OnnxBackend(
            model_id=str(
                params.get("model_id") or DEFAULT_EMBEDDINGGEMMA2_ONNX_REPO
            ).strip(),
            quantization=str(
                params.get("quantization") or DEFAULT_EMBEDDINGGEMMA2_QUANTIZATION
            ),
            matryoshka_dim=mrl_int,
            image_max_soft_tokens=int(params.get("image_max_soft_tokens") or 140),
            video_max_soft_tokens=int(params.get("video_max_soft_tokens") or 70),
            cache_dir=params.get("cache_dir"),
            local_files_only=bool(params.get("local_files_only", False)),
            providers=providers,
        )
    if canonical == "dinov3":
        mrl = params.get("matryoshka_dim")
        mrl_int = int(mrl) if mrl not in (None, "", 0) else None
        device = params.get("device")
        return Dinov3EmbeddingBackend(
            model_name=str(params.get("model_name") or DEFAULT_DINOV3_SEARCH_MODEL),
            short_side=int(params.get("short_side") or 384),
            device=str(device).strip() if device else None,
            matryoshka_dim=mrl_int,
        )
    torch_dtype = params.get("torch_dtype")
    attn_impl = params.get("attn_implementation")
    return Qwen3VLEmbeddingBackend(
        model_id=str(params.get("model_id") or DEFAULT_QWEN3VL_SEARCH_MODEL),
        torch_dtype=str(torch_dtype).strip() if torch_dtype else None,
        attn_implementation=str(attn_impl).strip() if attn_impl else None,
    )


def _deterministic_color_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministic RGB color derived via CRC32 (stable across processes)."""
    value = zlib.crc32((label or "shape").encode("utf-8")) & 0xFFFFFFFF
    r = 64 + (value & 0x7F)
    g = 64 + ((value >> 7) & 0x7F)
    b = 64 + ((value >> 14) & 0x7F)
    return int(r), int(g), int(b)


def _clip_point(pt: Tuple[int, int], *, width: int, height: int) -> Tuple[int, int]:
    x = max(0, min(int(pt[0]), width - 1))
    y = max(0, min(int(pt[1]), height - 1))
    return x, y


def overlay_shapes_on_frame_rgb(
    frame_rgb: np.ndarray,
    shapes: List[Dict[str, Any]],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    """Render LabelMe-style shapes onto an RGB frame deterministically."""
    if not shapes or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return frame_rgb

    height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    base = frame_rgb.astype(np.uint8, copy=True)
    overlay = base.copy()
    outlines: List[Tuple[str, Tuple[int, int, int], Any]] = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "")
        shape_type = str(shape.get("shape_type") or "").lower()
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            continue

        color = _deterministic_color_for_label(label)
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
            x1, y1 = _clip_point(
                (int(pts_i[0, 0]), int(pts_i[0, 1])), width=width, height=height
            )
            x2, y2 = _clip_point(
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
            cv2.circle(blended, payload, 6, color, thickness=2)

    return blended


def open_annotation_store_if_present(
    annotation_dir: Optional[Path],
) -> Optional[object]:
    """Open the NDJSON AnnotationStore for ``annotation_dir`` if one exists."""
    if annotation_dir is None or not annotation_dir.exists():
        return None
    try:
        stub = annotation_dir / f"{annotation_dir.name}_000000000.json"
        store = AnnotationStore.for_frame_path(stub)
        if not store.store_path.exists():
            return None
        return store
    except Exception:
        return None


def load_shapes_for_frame(
    *,
    annotation_dir: Path,
    frame_index: int,
    store: Optional[object] = None,
) -> List[Dict[str, Any]]:
    """Load shapes for a single frame from discrete LabelMe JSON or AnnotationStore."""
    if not annotation_dir.exists():
        return []
    try:
        idx = int(frame_index)
    except Exception:
        return []

    primary = annotation_dir / f"{annotation_dir.name}_{idx:09d}.json"
    legacy = annotation_dir / f"{idx:09d}.json"
    for candidate in (primary, legacy):
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
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


def run_embedding_search(
    *,
    query_vector: np.ndarray,
    query_frame_index: Optional[int],
    total_frames: int,
    stride: int,
    max_frames: Optional[int],
    threshold: float,
    top_k: int,
    embed_frame: Callable[[int], np.ndarray],
    is_stopped: Optional[Callable[[], bool]] = None,
    on_match: Optional[Callable[[EmbeddingSearchMatch], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[EmbeddingSearchMatch]:
    """Run vector similarity search over frames using caller-provided embedding IO."""
    if total_frames <= 0:
        raise ValueError("No frames selected for search.")

    stride = max(1, int(stride))
    top_k = max(1, int(top_k))
    threshold = float(threshold)
    max_steps = (int(total_frames) + stride - 1) // stride
    if max_frames is not None:
        max_steps = min(max_steps, max(1, int(max_frames)))
    if max_steps <= 0:
        raise ValueError("No frames selected for search.")

    qvec = np.asarray(query_vector, dtype=np.float32)
    matches: List[EmbeddingSearchMatch] = []
    step = 0

    for frame_idx in range(0, int(total_frames), stride):
        if step >= max_steps:
            break
        step += 1
        if is_stopped is not None and is_stopped():
            break
        if query_frame_index is not None and int(frame_idx) == int(query_frame_index):
            if on_progress is not None and (step % 10 == 0 or step == max_steps):
                on_progress(step, max_steps)
            continue

        vec = np.asarray(embed_frame(int(frame_idx)), dtype=np.float32)
        sim = float(vec @ qvec)
        if sim >= threshold:
            match = EmbeddingSearchMatch(frame_index=int(frame_idx), similarity=sim)
            matches.append(match)
            if on_match is not None:
                on_match(match)

        if on_progress is not None and (step % 10 == 0 or step == max_steps):
            on_progress(step, max_steps)

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:top_k]


def group_matches_into_bouts(
    matches: Sequence[EmbeddingSearchMatch],
    *,
    max_gap_frames: int = 10,
    min_matches: int = 1,
    fps: Optional[float] = None,
    label: Optional[str] = None,
    sort_by: str = "peak_similarity",
) -> List[BehaviorSearchBout]:
    """Merge temporally proximate frame matches into contiguous behavior bouts."""
    if not matches:
        return []
    gap = max(0, int(max_gap_frames))
    min_count = max(1, int(min_matches))
    ordered = sorted(matches, key=lambda m: (int(m.frame_index), -float(m.similarity)))

    clusters: List[List[EmbeddingSearchMatch]] = []
    for item in ordered:
        item_start = (
            int(item.start_frame)
            if item.start_frame is not None
            else int(item.frame_index)
        )
        if not clusters:
            clusters.append([item])
            continue
        prev = clusters[-1][-1]
        prev_end = (
            int(prev.end_frame) if prev.end_frame is not None else int(prev.frame_index)
        )
        if item_start - prev_end <= gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    fps_val = float(fps) if fps and float(fps) > 0 else None
    bouts: List[BehaviorSearchBout] = []
    for group in clusters:
        if len(group) < min_count:
            continue
        start_f = min(
            int(m.start_frame) if m.start_frame is not None else int(m.frame_index)
            for m in group
        )
        end_f = max(
            int(m.end_frame) if m.end_frame is not None else int(m.frame_index)
            for m in group
        )
        peak = max(group, key=lambda m: float(m.similarity))
        mean_sim = float(np.mean([float(m.similarity) for m in group]))
        bouts.append(
            BehaviorSearchBout(
                start_frame=start_f,
                end_frame=end_f,
                peak_frame=int(peak.frame_index),
                peak_similarity=float(peak.similarity),
                mean_similarity=mean_sim,
                match_count=len(group),
                label=label or peak.label,
                start_time_sec=(start_f / fps_val) if fps_val else None,
                end_time_sec=(end_f / fps_val) if fps_val else None,
            )
        )

    if sort_by == "start_frame":
        bouts.sort(key=lambda b: (b.start_frame, -b.peak_similarity))
    else:
        bouts.sort(key=lambda b: (-b.peak_similarity, b.start_frame))
    return bouts


def _compute_sampled_indices(
    total_frames: int, *, stride: int, max_frames: Optional[int]
) -> List[int]:
    if total_frames <= 0:
        raise ValueError("Video contains no frames.")
    step = max(1, int(stride))
    indices = list(range(0, int(total_frames), step))
    if max_frames is not None:
        indices = indices[: max(1, int(max_frames))]
    if not indices:
        raise ValueError("No frames selected for search.")
    return indices


def _clip_window_indices(
    center_idx: int,
    *,
    clip_frames: int,
    clip_step: int,
    total_frames: int,
) -> List[int]:
    count = max(1, int(clip_frames))
    if count == 1 or total_frames <= 1:
        return [max(0, min(int(center_idx), total_frames - 1))]
    step = max(1, int(clip_step))
    half = (count - 1) / 2.0
    raw = [int(round(center_idx + (i - half) * step)) for i in range(count)]
    clamped = [max(0, min(idx, total_frames - 1)) for idx in raw]
    # Keep order and preserve exact length when possible while avoiding trivial collapse.
    return clamped


def _load_frame_with_optional_overlay(
    video: CV2Video,
    frame_index: int,
    *,
    overlay_shapes: bool,
    annotation_dir: Optional[Path],
    store: Optional[object],
) -> np.ndarray:
    frame_rgb = video.load_frame(int(frame_index))
    if overlay_shapes and annotation_dir is not None:
        try:
            shapes = load_shapes_for_frame(
                annotation_dir=annotation_dir,
                frame_index=int(frame_index),
                store=store,
            )
            if shapes:
                frame_rgb = overlay_shapes_on_frame_rgb(frame_rgb, shapes)
        except Exception:
            pass
    return frame_rgb


def build_video_embedding_index(
    video_path: Union[str, Path],
    *,
    backend: FrameEmbeddingBackend,
    stride: int = 5,
    max_frames: Optional[int] = None,
    clip_frames: int = 1,
    clip_step: int = 1,
    batch_size: int = 4,
    overlay_shapes: bool = False,
    annotation_dir: Optional[Union[str, Path]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    on_vector: Optional[
        Callable[[int, np.ndarray, Optional[int], Optional[int]], None]
    ] = None,
) -> Tuple[VideoFrameEmbeddingIndex, bool]:
    """Extract and index embeddings across sampled video frames or temporal clips.

    Returns ``(index, completed)`` where ``completed=False`` indicates early
    cancellation via ``is_stopped``.
    """
    resolved_video = Path(video_path).expanduser().resolve()
    if not resolved_video.exists():
        raise FileNotFoundError(f"Video does not exist: {resolved_video}")

    resolved_ann_dir = (
        Path(annotation_dir).expanduser().resolve()
        if annotation_dir is not None
        else resolved_video.with_suffix("")
    )
    store = (
        open_annotation_store_if_present(resolved_ann_dir) if overlay_shapes else None
    )

    video = CV2Video(resolved_video)
    completed = True
    try:
        total_frames = int(video.total_frames())
        fps = float(video.get_fps() or 0.0)
        sampled = _compute_sampled_indices(
            total_frames, stride=stride, max_frames=max_frames
        )
        total_steps = len(sampled)
        clip_n = max(1, int(clip_frames))
        bs = max(1, int(batch_size))

        kept_indices: List[int] = []
        kept_vectors: List[np.ndarray] = []
        kept_spans: List[Tuple[int, int]] = []

        if clip_n == 1:
            done = 0
            for start in range(0, total_steps, bs):
                if is_stopped is not None and is_stopped():
                    completed = False
                    break
                chunk_indices = sampled[start : start + bs]
                frames_chunk: List[np.ndarray] = []
                valid_indices: List[int] = []
                for idx in chunk_indices:
                    if is_stopped is not None and is_stopped():
                        completed = False
                        break
                    try:
                        frame_rgb = _load_frame_with_optional_overlay(
                            video,
                            idx,
                            overlay_shapes=overlay_shapes,
                            annotation_dir=resolved_ann_dir,
                            store=store,
                        )
                    except Exception:
                        continue
                    frames_chunk.append(frame_rgb)
                    valid_indices.append(int(idx))
                if not completed and not frames_chunk:
                    break
                if frames_chunk:
                    vecs = backend.embed_frames_batch(frames_chunk)
                    for idx, vec in zip(valid_indices, vecs):
                        vec_norm = l2_normalize_np(vec)
                        kept_indices.append(idx)
                        kept_vectors.append(vec_norm)
                        if on_vector is not None:
                            on_vector(idx, vec_norm, None, None)
                done += len(chunk_indices)
                if on_progress is not None:
                    on_progress(min(done, total_steps), total_steps)
        else:
            for step_idx, center_idx in enumerate(sampled, start=1):
                if is_stopped is not None and is_stopped():
                    completed = False
                    break
                win_indices = _clip_window_indices(
                    center_idx,
                    clip_frames=clip_n,
                    clip_step=clip_step,
                    total_frames=total_frames,
                )
                clip_rgb: List[np.ndarray] = []
                for w_idx in win_indices:
                    try:
                        clip_rgb.append(
                            _load_frame_with_optional_overlay(
                                video,
                                w_idx,
                                overlay_shapes=overlay_shapes,
                                annotation_dir=resolved_ann_dir,
                                store=store,
                            )
                        )
                    except Exception:
                        continue
                if not clip_rgb:
                    continue
                vec_norm = l2_normalize_np(backend.embed_video_clip(clip_rgb))
                span_start, span_end = min(win_indices), max(win_indices)
                kept_indices.append(int(center_idx))
                kept_vectors.append(vec_norm)
                kept_spans.append((span_start, span_end))
                if on_vector is not None:
                    on_vector(int(center_idx), vec_norm, span_start, span_end)
                if on_progress is not None and (
                    step_idx % 5 == 0 or step_idx == total_steps
                ):
                    on_progress(step_idx, total_steps)
    finally:
        video.release()

    try:
        stat = resolved_video.stat()
        mtime_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    except OSError:
        mtime_ns = 0
        file_size = 0

    embeddings_mat = (
        np.stack(kept_vectors, axis=0).astype(np.float32, copy=False)
        if kept_vectors
        else np.zeros((0, 0), dtype=np.float32)
    )
    index = VideoFrameEmbeddingIndex(
        frame_indices=np.asarray(kept_indices, dtype=np.int64),
        embeddings=embeddings_mat,
        clip_spans=(np.asarray(kept_spans, dtype=np.int64) if kept_spans else None),
        metadata={
            "video_path": str(resolved_video),
            "video_mtime_ns": mtime_ns,
            "video_size": file_size,
            "fps": fps,
            "total_frames": total_frames,
            "stride": max(1, int(stride)),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "clip_frames": max(1, int(clip_frames)),
            "clip_step": max(1, int(clip_step)),
            "overlay_shapes": bool(overlay_shapes),
            "annotation_dir": str(resolved_ann_dir) if overlay_shapes else "",
            "backend_key": list(backend.key),
            "completed": bool(completed),
        },
    )
    return index, completed


def resolve_query_vector(
    backend: FrameEmbeddingBackend,
    *,
    video_path: Optional[Union[str, Path]] = None,
    query_frame_index: Optional[int] = None,
    query_text: Optional[str] = None,
    query_image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
    query_mode: str = "auto",
    prompt_name: Optional[str] = "SearchQuery",
    overlay_shapes: bool = False,
    annotation_dir: Optional[Union[str, Path]] = None,
    clip_frames: int = 1,
    clip_step: int = 1,
    text_weight: float = 0.5,
) -> Tuple[np.ndarray, str]:
    """Compute a normalized query vector for text, frame, image, or hybrid search."""
    clean_text = str(query_text or "").strip()
    mode = str(query_mode or "auto").strip().lower()
    if mode == "auto":
        has_visual = query_image is not None or query_frame_index is not None
        if clean_text and has_visual:
            mode = "hybrid"
        elif clean_text:
            mode = "text"
        elif has_visual:
            mode = "frame"
        else:
            raise ValueError(
                "Provide query_text, query_frame_index, or query_image to search."
            )

    if mode not in {"text", "frame", "image", "hybrid"}:
        raise ValueError(
            f"Unsupported query_mode {query_mode!r}. "
            "Use 'auto', 'text', 'frame', or 'hybrid'."
        )

    text_vec: Optional[np.ndarray] = None
    visual_vec: Optional[np.ndarray] = None

    if mode in {"text", "hybrid"}:
        if not clean_text:
            raise ValueError("query_text is required for text or hybrid search.")
        if not backend.supports_text:
            raise ValueError(
                f"Selected backend '{backend.key[0]}' does not support text queries. "
                "Select 'EmbeddingGemma 2 (ONNX)' (embeddinggemma2_onnx) or 'Qwen3-VL' for text behavior search."
            )
        text_vec = l2_normalize_np(
            backend.embed_text(clean_text, prompt_name=prompt_name)
        )

    if mode in {"frame", "image", "hybrid"}:
        if query_image is not None:
            if isinstance(query_image, (str, Path)):
                with Image.open(Path(query_image)) as opened:
                    img_arr = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            elif isinstance(query_image, Image.Image):
                img_arr = np.asarray(query_image.convert("RGB"), dtype=np.uint8)
            else:
                img_arr = np.asarray(query_image)
            visual_vec = l2_normalize_np(backend.embed_frame_rgb(img_arr))
        elif query_frame_index is not None:
            if video_path is None:
                raise ValueError(
                    "video_path is required when searching by query_frame_index."
                )
            resolved_video = Path(video_path).expanduser().resolve()
            resolved_ann_dir = (
                Path(annotation_dir).expanduser().resolve()
                if annotation_dir is not None
                else resolved_video.with_suffix("")
            )
            store = (
                open_annotation_store_if_present(resolved_ann_dir)
                if overlay_shapes
                else None
            )
            video = CV2Video(resolved_video)
            try:
                total_frames = int(video.total_frames())
                if clip_frames > 1:
                    win_indices = _clip_window_indices(
                        int(query_frame_index),
                        clip_frames=clip_frames,
                        clip_step=clip_step,
                        total_frames=total_frames,
                    )
                    frames = [
                        _load_frame_with_optional_overlay(
                            video,
                            w_idx,
                            overlay_shapes=overlay_shapes,
                            annotation_dir=resolved_ann_dir,
                            store=store,
                        )
                        for w_idx in win_indices
                    ]
                    visual_vec = l2_normalize_np(backend.embed_video_clip(frames))
                else:
                    frame_rgb = _load_frame_with_optional_overlay(
                        video,
                        int(query_frame_index),
                        overlay_shapes=overlay_shapes,
                        annotation_dir=resolved_ann_dir,
                        store=store,
                    )
                    visual_vec = l2_normalize_np(backend.embed_frame_rgb(frame_rgb))
            finally:
                video.release()
        else:
            raise ValueError(
                "Provide query_frame_index or query_image for frame search."
            )

    if mode == "text":
        assert text_vec is not None
        return text_vec, "text"
    if mode in {"frame", "image"}:
        assert visual_vec is not None
        return visual_vec, "frame"

    assert text_vec is not None and visual_vec is not None
    w = float(np.clip(text_weight, 0.0, 1.0))
    combined = l2_normalize_np((w * text_vec) + ((1.0 - w) * visual_vec))
    return combined, "hybrid"


@dataclass(frozen=True)
class VideoBehaviorSearchResult:
    """Structured result for video frame/behavior embedding search."""

    query_mode: str
    backend: str
    matches: List[EmbeddingSearchMatch]
    bouts: List[BehaviorSearchBout]
    index: VideoFrameEmbeddingIndex
    completed: bool = True
    score_summary: Optional[ScoreSummary] = None
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query_mode": self.query_mode,
            "backend": self.backend,
            "completed": bool(self.completed),
            "cache_hit": bool(self.cache_hit),
            "indexed_frames": len(self.index),
            "embedding_dim": self.index.dim,
            "match_count": len(self.matches),
            "matches": [m.to_dict() for m in self.matches],
            "bouts": [b.to_dict() for b in self.bouts],
            "metadata": dict(self.index.metadata),
        }
        if self.score_summary is not None:
            payload["score_summary"] = self.score_summary.to_dict()
        return payload


def search_video_behavior(
    video_path: Union[str, Path],
    *,
    query_text: Optional[str] = None,
    query_frame_index: Optional[int] = None,
    query_image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
    query_mode: str = "auto",
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    stride: int = 5,
    max_frames: Optional[int] = None,
    clip_frames: int = 1,
    clip_step: int = 1,
    batch_size: int = 4,
    threshold: Optional[float] = None,
    top_k: int = 50,
    overlay_shapes: bool = False,
    annotation_dir: Optional[Union[str, Path]] = None,
    prompt_name: Optional[str] = "SearchQuery",
    text_weight: float = 0.5,
    group_bouts: bool = True,
    bout_max_gap_frames: Optional[int] = None,
    index: Optional[VideoFrameEmbeddingIndex] = None,
    index_cache_path: Optional[Union[str, Path]] = None,
    auto_disk_cache: bool = False,
    is_stopped: Optional[Callable[[], bool]] = None,
    on_match: Optional[Callable[[EmbeddingSearchMatch], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> VideoBehaviorSearchResult:
    """Search a video for matching frames/behaviors via text, reference frame, or hybrid.

    When ``threshold is None``, computes an adaptive threshold from the video's
    similarity distribution (via ``compute_adaptive_threshold``).
    """
    resolved_video = Path(video_path).expanduser().resolve()
    owns_backend = not isinstance(backend, FrameEmbeddingBackend)
    active_backend = (
        backend
        if isinstance(backend, FrameEmbeddingBackend)
        else build_embedding_backend(backend, backend_params)
    )

    try:
        query_vec, resolved_mode = resolve_query_vector(
            active_backend,
            video_path=resolved_video,
            query_frame_index=query_frame_index,
            query_text=query_text,
            query_image=query_image,
            query_mode=query_mode,
            prompt_name=prompt_name,
            overlay_shapes=overlay_shapes,
            annotation_dir=annotation_dir,
            clip_frames=clip_frames,
            clip_step=clip_step,
            text_weight=text_weight,
        )

        exclude_idx = (
            int(query_frame_index)
            if resolved_mode == "frame" and query_frame_index is not None
            else None
        )
        label_tag = (
            str(query_text).strip() if query_text and str(query_text).strip() else None
        )
        threshold_mode = "auto" if threshold is None else "fixed"
        stream_thr = (
            float(threshold)
            if threshold is not None
            else recommended_similarity_threshold(active_backend.key[0], resolved_mode)
        )

        effective_cache_path: Optional[Path] = (
            Path(index_cache_path).expanduser()
            if index_cache_path is not None
            else (
                default_video_index_cache_path(
                    video_path=resolved_video,
                    backend_key=active_backend.key,
                    stride=stride,
                    max_frames=max_frames,
                    clip_frames=clip_frames,
                    overlay_shapes=overlay_shapes,
                    annotation_dir=annotation_dir,
                )
                if auto_disk_cache and not overlay_shapes
                else None
            )
        )
        managed_auto_cache = (
            index_cache_path is None and effective_cache_path is not None
        )

        active_index = index
        cache_hit = active_index is not None
        if active_index is None and effective_cache_path is not None:
            if effective_cache_path.is_file():
                try:
                    loaded = VideoFrameEmbeddingIndex.load_npz(effective_cache_path)
                    if _index_matches_request(
                        loaded,
                        video_path=resolved_video,
                        backend_key=active_backend.key,
                        stride=stride,
                        max_frames=max_frames,
                        clip_frames=clip_frames,
                        overlay_shapes=overlay_shapes,
                    ):
                        active_index = loaded
                        cache_hit = True
                        try:
                            effective_cache_path.touch()
                        except OSError:
                            pass
                except Exception as exc:
                    logger.warning(
                        "Ignoring incompatible cached embedding index %s: %s",
                        effective_cache_path,
                        exc,
                    )

        completed = True
        if active_index is not None:
            if on_progress is not None:
                total = max(1, len(active_index))
                on_progress(total, total)
        else:

            def _stream_match(
                frame_idx: int,
                vec: np.ndarray,
                start_f: Optional[int],
                end_f: Optional[int],
            ) -> None:
                if exclude_idx is not None and frame_idx == exclude_idx:
                    return
                sim = float(vec @ query_vec)
                if sim >= stream_thr and on_match is not None:
                    on_match(
                        EmbeddingSearchMatch(
                            frame_index=frame_idx,
                            similarity=sim,
                            start_frame=start_f,
                            end_frame=end_f,
                            label=label_tag,
                        )
                    )

            active_index, completed = build_video_embedding_index(
                resolved_video,
                backend=active_backend,
                stride=stride,
                max_frames=max_frames,
                clip_frames=clip_frames,
                clip_step=clip_step,
                batch_size=batch_size,
                overlay_shapes=overlay_shapes,
                annotation_dir=annotation_dir,
                is_stopped=is_stopped,
                on_progress=on_progress,
                on_vector=_stream_match if on_match is not None else None,
            )
            if completed and effective_cache_path is not None and len(active_index) > 0:
                try:
                    active_index.save_npz(effective_cache_path)
                    if managed_auto_cache:
                        prune_embedding_cache_dir(effective_cache_path.parent)
                except OSError as exc:
                    if not managed_auto_cache:
                        raise
                    logger.warning(
                        "Failed writing auto embedding cache %s: %s",
                        effective_cache_path,
                        exc,
                    )

        all_scores = active_index.score_query(query_vec)
        if threshold is None:
            eval_scores = (
                all_scores[active_index.frame_indices != int(exclude_idx)]
                if exclude_idx is not None and len(active_index) > 0
                else all_scores
            )
            effective_thr = compute_adaptive_threshold(
                eval_scores,
                backend=active_backend.key[0],
                query_mode=resolved_mode,
            )
        else:
            effective_thr = float(threshold)

        score_summary = summarize_similarity_scores(
            active_index.frame_indices,
            all_scores,
            effective_threshold=effective_thr,
            threshold_mode=threshold_mode,
            exclude_frame_index=exclude_idx,
        )
        matches = active_index.search(
            query_vec,
            threshold=effective_thr,
            top_k=top_k,
            exclude_frame_index=exclude_idx,
            label=label_tag,
        )
        if cache_hit and on_match is not None:
            for m in matches:
                on_match(m)

        gap = (
            int(bout_max_gap_frames)
            if bout_max_gap_frames is not None
            else max(1, int(stride)) * 2
        )
        fps_meta = active_index.metadata.get("fps")
        bouts = (
            group_matches_into_bouts(
                matches,
                max_gap_frames=gap,
                fps=float(fps_meta) if fps_meta else None,
                label=label_tag,
            )
            if group_bouts
            else []
        )
        return VideoBehaviorSearchResult(
            query_mode=resolved_mode,
            backend=active_backend.key[0],
            matches=matches,
            bouts=bouts,
            index=active_index,
            completed=completed,
            score_summary=score_summary,
            cache_hit=cache_hit,
        )
    finally:
        if owns_backend:
            active_backend.close()


def _index_matches_request(
    index: VideoFrameEmbeddingIndex,
    *,
    video_path: Path,
    backend_key: Sequence[str],
    stride: int,
    max_frames: Optional[int],
    clip_frames: int,
    overlay_shapes: bool,
) -> bool:
    meta = index.metadata or {}
    if not meta.get("completed", True):
        return False
    if list(meta.get("backend_key") or []) != list(backend_key):
        return False
    if int(meta.get("stride") or 0) != max(1, int(stride)):
        return False
    if meta.get("max_frames") != (int(max_frames) if max_frames is not None else None):
        return False
    if int(meta.get("clip_frames") or 1) != max(1, int(clip_frames)):
        return False
    if bool(meta.get("overlay_shapes", False)) != bool(overlay_shapes):
        return False
    try:
        stat = video_path.stat()
        if int(meta.get("video_mtime_ns") or 0) != int(stat.st_mtime_ns):
            return False
        if int(meta.get("video_size") or 0) != int(stat.st_size):
            return False
    except OSError:
        return False
    return True


def score_video_behaviors(
    video_path: Union[str, Path],
    behaviors: Sequence[str],
    *,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    stride: int = 15,
    max_frames: Optional[int] = None,
    clip_frames: int = 1,
    batch_size: int = 4,
    top_n: int = 3,
    min_similarity: float = 0.0,
    prompt_name: Optional[str] = "SearchQuery",
    index: Optional[VideoFrameEmbeddingIndex] = None,
    index_cache_path: Optional[Union[str, Path]] = None,
    auto_disk_cache: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Zero-shot score a video against a list of behavior text descriptions."""
    clean_behaviors = [str(b).strip() for b in behaviors if str(b).strip()]
    if not clean_behaviors:
        raise ValueError("Provide at least one behavior description to score.")

    owns_backend = not isinstance(backend, FrameEmbeddingBackend)
    active_backend = (
        backend
        if isinstance(backend, FrameEmbeddingBackend)
        else build_embedding_backend(backend, backend_params)
    )
    try:
        if not active_backend.supports_text:
            raise ValueError(
                f"Backend '{active_backend.key[0]}' does not support text behavior scoring. "
                "Use 'embeddinggemma2_onnx' or 'qwen3vl'."
            )
        resolved_video = Path(video_path).expanduser().resolve()
        effective_cache_path: Optional[Path] = (
            Path(index_cache_path).expanduser()
            if index_cache_path is not None
            else (
                default_video_index_cache_path(
                    video_path=resolved_video,
                    backend_key=active_backend.key,
                    stride=stride,
                    max_frames=max_frames,
                    clip_frames=clip_frames,
                    overlay_shapes=False,
                )
                if auto_disk_cache
                else None
            )
        )
        managed_auto_cache = (
            index_cache_path is None and effective_cache_path is not None
        )

        active_index = index
        if active_index is None and effective_cache_path is not None:
            if effective_cache_path.is_file():
                try:
                    loaded = VideoFrameEmbeddingIndex.load_npz(effective_cache_path)
                    if _index_matches_request(
                        loaded,
                        video_path=resolved_video,
                        backend_key=active_backend.key,
                        stride=stride,
                        max_frames=max_frames,
                        clip_frames=clip_frames,
                        overlay_shapes=False,
                    ):
                        active_index = loaded
                except Exception as exc:
                    logger.warning(
                        "Ignoring incompatible cached embedding index %s: %s",
                        effective_cache_path,
                        exc,
                    )
        if active_index is None:
            active_index, completed = build_video_embedding_index(
                resolved_video,
                backend=active_backend,
                stride=stride,
                max_frames=max_frames,
                clip_frames=clip_frames,
                batch_size=batch_size,
                on_progress=on_progress,
            )
            if completed and effective_cache_path is not None and len(active_index) > 0:
                try:
                    active_index.save_npz(effective_cache_path)
                    if managed_auto_cache:
                        prune_embedding_cache_dir(effective_cache_path.parent)
                except OSError as exc:
                    if not managed_auto_cache:
                        raise
                    logger.warning(
                        "Failed writing auto embedding cache %s: %s",
                        effective_cache_path,
                        exc,
                    )

        text_vecs = active_backend.embed_texts_batch(
            clean_behaviors, prompt_name=prompt_name
        )
        label_map = {lbl: text_vecs[i] for i, lbl in enumerate(clean_behaviors)}
        frame_scores = active_index.score_labels(
            label_map, top_n=top_n, min_similarity=min_similarity
        )
        return {
            "video_path": str(resolved_video),
            "backend": active_backend.key[0],
            "behaviors": clean_behaviors,
            "indexed_frames": len(active_index),
            "fps": active_index.metadata.get("fps"),
            "frames": frame_scores,
        }
    finally:
        if owns_backend:
            active_backend.close()


def index_folder_frames(
    folder_path: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    max_images: Optional[int] = None,
    batch_size: int = 8,
    recursive: bool = False,
) -> FolderFrameIndex:
    """Index images and optional LabelMe JSON flags/captions in a folder."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame folder does not exist: {folder}")

    iterator = folder.rglob("*") if recursive else folder.glob("*")
    image_paths = sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_FILE_EXTENSIONS
    )
    if max_images is not None:
        image_paths = image_paths[: max(1, int(max_images))]

    owns_backend = not isinstance(backend, FrameEmbeddingBackend)
    active_backend = (
        backend
        if isinstance(backend, FrameEmbeddingBackend)
        else build_embedding_backend(backend, backend_params)
    )
    try:
        uris: List[str] = []
        flags_list: List[Tuple[str, ...]] = []
        captions_list: List[str] = []
        vectors: List[np.ndarray] = []

        bs = max(1, int(batch_size))
        for start in range(0, len(image_paths), bs):
            chunk = image_paths[start : start + bs]
            imgs: List[np.ndarray] = []
            valid_paths: List[Path] = []
            for p in chunk:
                try:
                    with Image.open(p) as opened:
                        imgs.append(np.asarray(opened.convert("RGB"), dtype=np.uint8))
                    valid_paths.append(p)
                except Exception as exc:
                    logger.warning("Skipping unreadable frame %s: %s", p, exc)
            if not imgs:
                continue
            vecs = active_backend.embed_frames_batch(imgs)
            for p, vec in zip(valid_paths, vecs):
                ann_file = p.with_suffix(".json")
                frame_flags: Tuple[str, ...] = ()
                caption = ""
                if ann_file.is_file():
                    try:
                        ann = load_labelme_json(ann_file)
                        raw_flags = ann.get("flags") or {}
                        if isinstance(raw_flags, dict):
                            frame_flags = tuple(
                                str(k) for k, v in raw_flags.items() if v
                            )
                        caption = str(ann.get("caption") or "")
                    except Exception as exc:
                        logger.warning(
                            "Failed reading annotation %s: %s", ann_file, exc
                        )
                uris.append(str(p))
                flags_list.append(frame_flags)
                captions_list.append(caption)
                vectors.append(l2_normalize_np(vec))

        emb_mat = (
            np.stack(vectors, axis=0).astype(np.float32, copy=False)
            if vectors
            else np.zeros((0, 0), dtype=np.float32)
        )
        index = FolderFrameIndex(
            image_uris=tuple(uris),
            embeddings=emb_mat,
            flags=tuple(flags_list),
            captions=tuple(captions_list),
            metadata={
                "folder_path": str(folder),
                "backend_key": list(active_backend.key),
            },
        )
        if output_path is not None:
            index.save_npz(output_path)
        return index
    finally:
        if owns_backend:
            active_backend.close()


def search_folder_frames(
    index_or_path: Union[FolderFrameIndex, str, Path],
    *,
    query_text: Optional[str] = None,
    query_image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
    backend: Union[str, FrameEmbeddingBackend] = DEFAULT_SEARCH_BACKEND,
    backend_params: Optional[Mapping[str, Any]] = None,
    top_k: int = 10,
    threshold: float = -1.0,
    prompt_name: Optional[str] = "SearchQuery",
) -> List[EmbeddingSearchMatch]:
    """Query a ``FolderFrameIndex`` by text behavior query, image, or both."""
    index = (
        index_or_path
        if isinstance(index_or_path, FolderFrameIndex)
        else FolderFrameIndex.load_npz(index_or_path)
    )
    owns_backend = not isinstance(backend, FrameEmbeddingBackend)
    active_backend = (
        backend
        if isinstance(backend, FrameEmbeddingBackend)
        else build_embedding_backend(backend, backend_params)
    )
    try:
        qvec, _ = resolve_query_vector(
            active_backend,
            query_text=query_text,
            query_image=query_image,
            prompt_name=prompt_name,
        )
        return index.search(qvec, top_k=top_k, threshold=threshold)
    finally:
        if owns_backend:
            active_backend.close()


__all__ = [
    "DEFAULT_DINOV3_SEARCH_MODEL",
    "DEFAULT_EMBEDDING_CACHE_MAX_BYTES",
    "DEFAULT_EMBEDDING_CACHE_MAX_FILES",
    "DEFAULT_QWEN3VL_SEARCH_MODEL",
    "DEFAULT_SEARCH_BACKEND",
    "BehaviorSearchBout",
    "Dinov3EmbeddingBackend",
    "EmbeddingGemma2OnnxBackend",
    "EmbeddingSearchMatch",
    "FolderFrameIndex",
    "FrameEmbeddingBackend",
    "Qwen3VLEmbeddingBackend",
    "ScoreSummary",
    "VideoBehaviorSearchResult",
    "VideoFrameEmbeddingIndex",
    "backend_cache_key",
    "build_embedding_backend",
    "build_video_embedding_index",
    "compute_adaptive_threshold",
    "default_video_index_cache_path",
    "group_matches_into_bouts",
    "index_folder_frames",
    "load_shapes_for_frame",
    "normalize_backend_name",
    "open_annotation_store_if_present",
    "overlay_shapes_on_frame_rgb",
    "prune_embedding_cache_dir",
    "recommended_similarity_threshold",
    "resolve_default_embedding_cache_dir",
    "resolve_query_vector",
    "run_embedding_search",
    "score_video_behaviors",
    "search_folder_frames",
    "search_video_behavior",
    "summarize_similarity_scores",
]
