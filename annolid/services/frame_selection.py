"""Unified frame selection & optional extraction service for pose and segmentation labeling.

Coordinates:
- Task-aware presets (``pose``, ``segmentation``, ``general``),
- Built-in zero-dependency visual descriptors (foreground+motion ``visual_fg`` and
  full-frame ``visual``),
- Foundation frame embeddings via ``annolid.services.embedding_search``
  (``embeddinggemma2_onnx``, ``dinov3`` / vision backbone catalog) with
  persistent ``.npz`` disk cache auto-discovery & reuse,
- Dual-storage already-labeled frame discovery (per-frame LabelMe ``.json`` +
  NDJSON ``AnnotationStore``) for both exclusion and labeled-seed diversity expansion,
- Both **Queue in Video (no extraction)** and **Extract Images to Folder** workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from annolid.core.media.video import CV2Video, read_capture_frame_bgr
from annolid.data.frame_sampling import (
    _compute_foreground_visual_features,
    _frame_quality_metrics,
    _frame_visual_descriptor,
    _kmeans_cluster_medoids,
    _normalize_feature_matrix,
    _resolve_frame_bounds,
    _write_selected_frames,
    discover_labeled_frame_indices,
    select_farthest_point_indices,
    select_flow_frame_indices,
    select_hybrid_frame_indices,
    select_keyframe_indices,
    select_uniform_frame_indices,
)
from annolid.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

QUEUE_MANIFEST_FILENAME = "frame_selection_queue.json"
EXTRACT_MANIFEST_FILENAME = "frame_selection_manifest.json"

SUPPORTED_TASKS = ("pose", "segmentation", "general")
SUPPORTED_ALGOS = (
    "hybrid",
    "kmeans",
    "farthest",
    "uniform",
    "random",
    "flow",
    "keyframes",
)
SUPPORTED_FEATURE_BACKENDS = (
    "auto",
    "visual_fg",
    "visual",
    "embeddinggemma2_onnx",
    "dinov3",
    "cached_index",
)


@dataclass(frozen=True)
class TaskProfile:
    """Scientifically grounded frame selection defaults for a labeling task."""

    key: str
    label: str
    description: str
    default_algo: str
    default_num_frames: int
    default_feature_backend: str
    representative_ratio: float
    include_boundaries: bool
    exclude_labeled: bool
    filter_blur: bool
    foreground_weight: float
    appearance_weight: float
    edge_weight: float
    recommended_min_gap: int


TASK_PROFILES: Dict[str, TaskProfile] = {
    "pose": TaskProfile(
        key="pose",
        label="Pose (Keypoints)",
        description=(
            "Balances representative postures with rare/extreme poses using "
            "Hybrid (K-Means + Farthest Coreset), foreground focus, blur filtering, "
            "and diversity seeding from already-labeled frames."
        ),
        default_algo="hybrid",
        default_num_frames=40,
        default_feature_backend="auto",
        representative_ratio=0.60,
        include_boundaries=False,
        exclude_labeled=True,
        filter_blur=True,
        foreground_weight=1.65,
        appearance_weight=0.70,
        edge_weight=1.15,
        recommended_min_gap=10,
    ),
    "segmentation": TaskProfile(
        key="segmentation",
        label="Segmentation (Masks)",
        description=(
            "Maximizes appearance, occlusion, and lighting coverage while pinning "
            "a sharp start anchor frame for VOS/Cutie/SAM propagation."
        ),
        default_algo="hybrid",
        default_num_frames=25,
        default_feature_backend="auto",
        representative_ratio=0.72,
        include_boundaries=True,
        exclude_labeled=True,
        filter_blur=True,
        foreground_weight=1.20,
        appearance_weight=1.05,
        edge_weight=1.00,
        recommended_min_gap=12,
    ),
    "general": TaskProfile(
        key="general",
        label="General / Custom",
        description="Full control over algorithm, visual or neural embedding features, and output mode.",
        default_algo="kmeans",
        default_num_frames=50,
        default_feature_backend="auto",
        representative_ratio=0.65,
        include_boundaries=False,
        exclude_labeled=True,
        filter_blur=True,
        foreground_weight=1.35,
        appearance_weight=0.90,
        edge_weight=1.05,
        recommended_min_gap=8,
    ),
}


@dataclass(frozen=True)
class CachedEmbeddingIndexInfo:
    """Metadata summary of a precomputed ``.npz`` video embedding cache."""

    path: str
    backend_name: str
    backend_key: Tuple[str, ...]
    frame_count: int
    embedding_dim: int
    stride: int
    clip_frames: int
    mtime: float = 0.0

    @property
    def display_label(self) -> str:
        return (
            f"{self.backend_name} ({self.frame_count} frames, "
            f"{self.embedding_dim}d, step={self.stride})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "backend_name": self.backend_name,
            "backend_key": list(self.backend_key),
            "frame_count": int(self.frame_count),
            "embedding_dim": int(self.embedding_dim),
            "stride": int(self.stride),
            "clip_frames": int(self.clip_frames),
        }


@dataclass(frozen=True)
class FeatureBackendOption:
    """UI/CLI capability descriptor for a frame feature representation."""

    key: str
    label: str
    description: str
    available: bool
    supports_text_query: bool = False
    requires_download: bool = False
    install_hint: str = ""
    cached_index: Optional[CachedEmbeddingIndexInfo] = None


@dataclass(frozen=True)
class SelectedFrameItem:
    """A single selected frame with rich selection diagnostics."""

    frame_index: int
    timestamp_sec: Optional[float] = None
    cluster_id: Optional[int] = None
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    motion_score: float = 0.0
    sharpness_score: float = 0.0
    relevance_score: Optional[float] = None
    selection_role: str = "representative"
    already_labeled: bool = False
    reason: str = ""
    saved_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "frame_index": int(self.frame_index),
            "timestamp_sec": (
                round(float(self.timestamp_sec), 3)
                if self.timestamp_sec is not None
                else None
            ),
            "cluster_id": (
                int(self.cluster_id) if self.cluster_id is not None else None
            ),
            "diversity_score": round(float(self.diversity_score), 4),
            "novelty_score": round(float(self.novelty_score), 4),
            "motion_score": round(float(self.motion_score), 4),
            "sharpness_score": round(float(self.sharpness_score), 4),
            "selection_role": str(self.selection_role or "representative"),
            "already_labeled": bool(self.already_labeled),
            "reason": str(self.reason or ""),
        }
        if self.relevance_score is not None:
            payload["relevance_score"] = round(float(self.relevance_score), 4)
        if self.saved_path is not None:
            payload["saved_path"] = str(self.saved_path)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectedFrameItem":
        return cls(
            frame_index=int(data.get("frame_index", 0)),
            timestamp_sec=(
                float(data["timestamp_sec"])
                if data.get("timestamp_sec") is not None
                else None
            ),
            cluster_id=(
                int(data["cluster_id"]) if data.get("cluster_id") is not None else None
            ),
            diversity_score=float(data.get("diversity_score", 0.0) or 0.0),
            novelty_score=float(data.get("novelty_score", 0.0) or 0.0),
            motion_score=float(data.get("motion_score", 0.0) or 0.0),
            sharpness_score=float(data.get("sharpness_score", 0.0) or 0.0),
            relevance_score=(
                float(data["relevance_score"])
                if data.get("relevance_score") is not None
                else None
            ),
            selection_role=str(data.get("selection_role") or "representative"),
            already_labeled=bool(data.get("already_labeled", False)),
            reason=str(data.get("reason") or ""),
            saved_path=(
                str(data["saved_path"]) if data.get("saved_path") is not None else None
            ),
        )


@dataclass(frozen=True)
class FrameSelectionConfig:
    """Complete configuration for frame selection and optional extraction."""

    task: str = "pose"
    algo: str = "hybrid"
    num_frames: int = 40
    feature_backend: str = "auto"
    backend_params: Dict[str, Any] = field(default_factory=dict)
    extract_images: bool = False
    output_dir: Optional[str] = None
    annotation_dir: Optional[str] = None
    image_format: str = "png"
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    max_candidates: int = 600
    min_frame_gap: Optional[int] = None
    exclude_labeled: bool = True
    seed_from_labeled: bool = True
    filter_blur: bool = True
    include_boundaries: Optional[bool] = None
    representative_ratio: Optional[float] = None
    query_text: Optional[str] = None
    relevance_weight: float = 0.35
    embedding_index_path: Optional[str] = None
    use_embedding_disk_cache: bool = True
    persist_queue_manifest: bool = True
    open_output_in_files: bool = True
    seed: int = 42

    @classmethod
    def for_task(cls, task: str, **overrides: Any) -> "FrameSelectionConfig":
        profile = get_task_profile(task)
        defaults: Dict[str, Any] = {
            "task": profile.key,
            "algo": profile.default_algo,
            "num_frames": profile.default_num_frames,
            "feature_backend": profile.default_feature_backend,
            "include_boundaries": profile.include_boundaries,
            "exclude_labeled": profile.exclude_labeled,
            "filter_blur": profile.filter_blur,
            "representative_ratio": profile.representative_ratio,
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass(frozen=True)
class FrameSelectionResult:
    """Structured result for both in-video selection queues and disk extraction."""

    video_path: str
    task: str
    algo: str
    feature_backend: str
    extract_images: bool
    output_dir: Optional[str]
    total_video_frames: int
    fps: float
    start_frame: int
    end_frame: int
    candidate_count: int
    excluded_labeled_count: int
    filtered_blur_count: int
    selected_items: Tuple[SelectedFrameItem, ...]
    saved_paths: Tuple[str, ...] = ()
    manifest_path: Optional[str] = None
    embedding_cache_path: Optional[str] = None
    embedding_cache_hit: bool = False
    query_text: Optional[str] = None
    cancelled: bool = False

    @property
    def selected_indices(self) -> Tuple[int, ...]:
        return tuple(int(item.frame_index) for item in self.selected_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "video_path": self.video_path,
            "task": self.task,
            "algo": self.algo,
            "feature_backend": self.feature_backend,
            "extract_images": bool(self.extract_images),
            "output_dir": self.output_dir,
            "total_video_frames": int(self.total_video_frames),
            "fps": round(float(self.fps), 4),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "candidate_count": int(self.candidate_count),
            "excluded_labeled_count": int(self.excluded_labeled_count),
            "filtered_blur_count": int(self.filtered_blur_count),
            "num_selected": len(self.selected_items),
            "num_extracted": len(self.saved_paths),
            "selected_indices": list(self.selected_indices),
            "selected_items": [item.to_dict() for item in self.selected_items],
            "saved_paths": list(self.saved_paths),
            "manifest_path": self.manifest_path,
            "embedding_cache_path": self.embedding_cache_path,
            "embedding_cache_hit": bool(self.embedding_cache_hit),
            "query_text": self.query_text,
            "cancelled": bool(self.cancelled),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameSelectionResult":
        raw_items = data.get("selected_items")
        if isinstance(raw_items, list) and raw_items:
            items = tuple(
                SelectedFrameItem.from_dict(row)
                for row in raw_items
                if isinstance(row, Mapping)
            )
        else:
            raw_indices = data.get("selected_indices") or []
            fps = float(data.get("fps") or 0.0)
            items = tuple(
                SelectedFrameItem(
                    frame_index=int(idx),
                    timestamp_sec=(float(idx) / fps) if fps > 0.0 else None,
                )
                for idx in raw_indices
            )
        return cls(
            video_path=str(data.get("video_path") or ""),
            task=str(data.get("task") or "general"),
            algo=str(data.get("algo") or "hybrid"),
            feature_backend=str(data.get("feature_backend") or "visual_fg"),
            extract_images=bool(data.get("extract_images", False)),
            output_dir=(
                str(data["output_dir"]) if data.get("output_dir") is not None else None
            ),
            total_video_frames=int(data.get("total_video_frames", 0) or 0),
            fps=float(data.get("fps", 0.0) or 0.0),
            start_frame=int(data.get("start_frame", 0) or 0),
            end_frame=int(data.get("end_frame", 0) or 0),
            candidate_count=int(data.get("candidate_count", 0) or 0),
            excluded_labeled_count=int(data.get("excluded_labeled_count", 0) or 0),
            filtered_blur_count=int(data.get("filtered_blur_count", 0) or 0),
            selected_items=items,
            saved_paths=tuple(str(p) for p in (data.get("saved_paths") or ())),
            manifest_path=(
                str(data["manifest_path"])
                if data.get("manifest_path") is not None
                else None
            ),
            embedding_cache_path=(
                str(data["embedding_cache_path"])
                if data.get("embedding_cache_path") is not None
                else None
            ),
            embedding_cache_hit=bool(data.get("embedding_cache_hit", False)),
            query_text=(
                str(data["query_text"]) if data.get("query_text") is not None else None
            ),
            cancelled=bool(data.get("cancelled", False)),
        )


def get_task_profile(task: Optional[str]) -> TaskProfile:
    key = str(task or "pose").strip().lower()
    if key in {"seg", "mask", "masks", "vos"}:
        key = "segmentation"
    elif key in {"kp", "keypoints", "keypoint", "skeleton"}:
        key = "pose"
    return TASK_PROFILES.get(key, TASK_PROFILES["general"])


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _resolve_embedding_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    import os

    env_dir = str(os.getenv("ANNOLID_EMBEDDING_CACHE_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "annolid" / "embedding_search"


def find_cached_video_embedding_indices(
    video_path: Optional[Union[str, Path]],
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[CachedEmbeddingIndexInfo]:
    """Discover compatible cached ``.npz`` frame embedding indices for ``video_path``."""
    if not video_path:
        return []
    resolved = Path(video_path).expanduser().resolve()
    if not resolved.is_file():
        return []
    root = _resolve_embedding_cache_dir(cache_dir)
    if not root.is_dir():
        return []

    try:
        stat = resolved.stat()
        mtime_ns = int(stat.st_mtime_ns)
        file_size = int(stat.st_size)
    except OSError:
        mtime_ns = 0
        file_size = 0

    stem = (
        "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in resolved.stem)[:32]
        or "video"
    )
    matches: List[CachedEmbeddingIndexInfo] = []
    # Probe stem-matching caches first, falling back to any .npz if stem changed.
    candidates = list(root.glob(f"{stem}_*.npz"))
    if not candidates:
        candidates = list(root.glob("*.npz"))

    for path in candidates:
        try:
            with np.load(str(path), allow_pickle=False) as data:
                if "metadata_json" not in data or "frame_indices" not in data:
                    continue
                meta = json.loads(str(data["metadata_json"][0]))
                if not isinstance(meta, dict):
                    continue
                if not meta.get("completed", True):
                    continue
                meta_video = str(meta.get("video_path") or "")
                same_path = bool(meta_video and Path(meta_video) == resolved)
                same_stat = (
                    file_size > 0
                    and int(meta.get("video_size") or 0) == file_size
                    and int(meta.get("video_mtime_ns") or 0) == mtime_ns
                )
                if not (same_path or same_stat):
                    continue
                if bool(meta.get("overlay_shapes", False)):
                    continue
                frame_count = int(data["frame_indices"].shape[0])
                emb_shape = data["embeddings"].shape
                emb_dim = int(emb_shape[1]) if len(emb_shape) == 2 else 0
                if frame_count <= 0 or emb_dim <= 0:
                    continue
                backend_key = tuple(str(x) for x in (meta.get("backend_key") or ()))
                backend_name = backend_key[0] if backend_key else "embedding"
                matches.append(
                    CachedEmbeddingIndexInfo(
                        path=str(path),
                        backend_name=backend_name,
                        backend_key=backend_key,
                        frame_count=frame_count,
                        embedding_dim=emb_dim,
                        stride=max(1, int(meta.get("stride") or 1)),
                        clip_frames=max(1, int(meta.get("clip_frames") or 1)),
                        mtime=float(path.stat().st_mtime),
                    )
                )
        except Exception:
            continue

    matches.sort(key=lambda item: (-item.frame_count, -item.mtime))
    return matches


def list_available_feature_backends(
    video_path: Optional[Union[str, Path]] = None,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[FeatureBackendOption]:
    """Return available frame feature representations without importing heavy ML runtimes."""
    cached = (
        find_cached_video_embedding_indices(video_path, cache_dir=cache_dir)
        if video_path
        else []
    )
    best_cached = cached[0] if cached else None
    has_onnx = _has_module("onnxruntime")
    has_torch = _has_module("torch")
    has_dino_deps = has_torch and (
        _has_module("transformers") or _has_module("timm") or _has_module("torchvision")
    )

    options = [
        FeatureBackendOption(
            key="auto",
            label=(
                f"Auto (Reuse Cached {best_cached.backend_name} · {best_cached.frame_count}f)"
                if best_cached is not None
                else "Auto (Foreground + Motion Visual — Instant)"
            ),
            description=(
                "Instantly reuses an existing cached embedding index for this video when "
                "available, otherwise uses built-in Foreground + Motion visual descriptors."
            ),
            available=True,
            supports_text_query=bool(
                best_cached is not None
                and best_cached.backend_name == "embeddinggemma2_onnx"
                and has_onnx
            ),
            cached_index=best_cached,
        ),
        FeatureBackendOption(
            key="visual_fg",
            label="Foreground + Motion Visual (Built-in, Fast)",
            description=(
                "Background-subtracted animal focus + LAB appearance + Sobel edges. "
                "Zero ML dependencies; ideal for fixed-camera lab setups."
            ),
            available=True,
        ),
        FeatureBackendOption(
            key="visual",
            label="Full-Frame Visual (LAB + Edges, Built-in)",
            description=(
                "Whole-frame spatial color and edge descriptors. Zero ML dependencies; "
                "best when the camera moves."
            ),
            available=True,
        ),
        FeatureBackendOption(
            key="embeddinggemma2_onnx",
            label=(
                "EmbeddingGemma 2 (ONNX · Multimodal + Text Focus)"
                if has_onnx
                else "EmbeddingGemma 2 (ONNX — requires onnxruntime)"
            ),
            description=(
                "Semantic multimodal frame embeddings cached to .npz (shared with "
                "Embedding Search). Supports optional text focus prompts (e.g. 'rearing')."
            ),
            available=has_onnx,
            supports_text_query=True,
            requires_download=True,
            install_hint="pip install onnxruntime tokenizers huggingface-hub",
            cached_index=next(
                (c for c in cached if c.backend_name == "embeddinggemma2_onnx"),
                None,
            ),
        ),
        FeatureBackendOption(
            key="dinov3",
            label=(
                "DINOv3 / Vision Backbone (Dense Semantic Features)"
                if has_dino_deps
                else "DINOv3 / Vision Backbone (requires torch + transformers/timm)"
            ),
            description=(
                "Dense visual foundation embeddings (DINOv3, DINOv2, SigLIP 2, MobileNetV4) "
                "cached to .npz. Strongest semantic pose & appearance separation."
            ),
            available=has_dino_deps,
            requires_download=True,
            install_hint="pip install torch transformers",
            cached_index=next(
                (c for c in cached if c.backend_name == "dinov3"),
                None,
            ),
        ),
        FeatureBackendOption(
            key="cached_index",
            label=(
                f"Cached Embedding Index (.npz · {best_cached.display_label})"
                if best_cached is not None
                else "Cached Embedding Index (.npz — Browse file…)"
            ),
            description=(
                "Reuse an existing precomputed VideoFrameEmbeddingIndex .npz file "
                "for near-instant frame diversity selection."
            ),
            available=True,
            cached_index=best_cached,
        ),
    ]
    return options


def default_queue_manifest_path(
    video_path: Union[str, Path],
    annotation_dir: Optional[Union[str, Path]] = None,
) -> Path:
    vpath = Path(video_path).expanduser().resolve()
    base_dir = (
        Path(annotation_dir).expanduser().resolve()
        if annotation_dir is not None and str(annotation_dir).strip()
        else vpath.with_suffix("")
    )
    return base_dir / QUEUE_MANIFEST_FILENAME


def save_frame_selection_queue(
    result: FrameSelectionResult,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    target = (
        Path(path).expanduser().resolve()
        if path is not None
        else default_queue_manifest_path(result.video_path, result.output_dir)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, result.to_dict(), indent=2)
    return target


def load_frame_selection_queue(path: Union[str, Path]) -> FrameSelectionResult:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Frame selection queue not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid frame selection queue file: {source}")
    return FrameSelectionResult.from_dict(data)


def _resolve_video_bounds(
    vpath: Path,
    *,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")
    try:
        total_frames, lo, hi = _resolve_frame_bounds(
            cap, start_seconds=start_seconds, end_seconds=end_seconds
        )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if total_frames <= 0:
        return 0, fps, 0, 0
    if start_frame is not None:
        lo = max(0, min(int(start_frame), total_frames - 1))
    if end_frame is not None:
        hi = max(lo, min(int(end_frame), total_frames - 1))
    return total_frames, fps, lo, hi


def _read_candidate_thumbnails_and_quality(
    vpath: Path,
    indices: Sequence[int],
    *,
    resize_width: int = 48,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
    pct_start: float = 5.0,
    pct_end: float = 65.0,
) -> Tuple[List[int], List[np.ndarray], np.ndarray, np.ndarray, bool]:
    """Decode candidate frames once into compact thumbnails + quality metrics."""
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")
    valid_indices: List[int] = []
    thumbs: List[np.ndarray] = []
    sharpness_list: List[float] = []
    luma_list: List[float] = []
    cancelled = False
    try:
        next_expected: Optional[int] = 0
        grab_skip_disabled = False
        sorted_idx = sorted({int(i) for i in indices if int(i) >= 0})
        total = max(1, len(sorted_idx))
        target_w = max(24, int(resize_width))
        for pos, fidx in enumerate(sorted_idx):
            if is_stopped is not None and is_stopped():
                cancelled = True
                break
            ret, frame, next_expected, grab_skip_disabled = read_capture_frame_bgr(
                cap,
                fidx,
                expected_next=next_expected,
                max_forward_grab_skip=CV2Video._MAX_FORWARD_GRAB_SKIP,
                grab_skip_disabled=grab_skip_disabled,
            )
            if not ret or frame is None:
                continue
            sharp, luma = _frame_quality_metrics(frame)
            h, w = frame.shape[:2]
            target_h = max(24, int(round(h * (target_w / float(max(w, 1))))))
            thumb = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            valid_indices.append(int(fidx))
            thumbs.append(thumb)
            sharpness_list.append(sharp)
            luma_list.append(luma)
            if progress_callback is not None and (pos % 20 == 0 or pos == total - 1):
                pct = pct_start + (pct_end - pct_start) * (pos + 1) / float(total)
                progress_callback(
                    pct,
                    f"Scanning candidate frames ({pos + 1}/{total})…",
                )
    finally:
        cap.release()
    return (
        valid_indices,
        thumbs,
        np.asarray(sharpness_list, dtype=np.float32),
        np.asarray(luma_list, dtype=np.float32),
        cancelled,
    )


def _params_from_cached_backend_key(
    backend_key: Sequence[str],
    user_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive matching backend constructor params from a cached ``backend_key``."""
    params: Dict[str, Any] = dict(user_params or {})
    if not backend_key:
        return params
    name = str(backend_key[0]).strip().lower()
    if name == "embeddinggemma2_onnx":
        if len(backend_key) >= 2 and backend_key[1]:
            params["model_id"] = str(backend_key[1])
        if len(backend_key) >= 3 and backend_key[2]:
            params["quantization"] = str(backend_key[2])
        if len(backend_key) >= 4 and str(backend_key[3]).isdigit():
            params["matryoshka_dim"] = int(backend_key[3])
        if len(backend_key) >= 5 and str(backend_key[4]).isdigit():
            params["image_max_soft_tokens"] = int(backend_key[4])
        if len(backend_key) >= 6 and str(backend_key[5]).isdigit():
            params["video_max_soft_tokens"] = int(backend_key[5])
    elif name == "dinov3":
        if len(backend_key) >= 2 and backend_key[1]:
            params["model_name"] = str(backend_key[1])
        if len(backend_key) >= 3 and str(backend_key[2]).isdigit():
            params["short_side"] = int(backend_key[2])
        if len(backend_key) >= 5 and str(backend_key[4]).isdigit():
            params["matryoshka_dim"] = int(backend_key[4])
    return params


def _compute_embedding_features_for_video(
    vpath: Path,
    *,
    backend_or_name: Any,
    backend_params: Optional[Mapping[str, Any]],
    total_frames: int,
    lo: int,
    hi: int,
    k_target: int,
    max_candidates: int,
    query_text: Optional[str],
    precomputed_index: Optional[Any],
    embedding_index_path: Optional[str],
    use_disk_cache: bool,
    cache_dir: Optional[Union[str, Path]],
    progress_callback: Optional[Callable[[float, str], None]],
    is_stopped: Optional[Callable[[], bool]],
) -> Tuple[
    List[int],
    np.ndarray,
    Optional[np.ndarray],
    Optional[str],
    bool,
    str,
    bool,
]:
    """Compute or load L2-normalized neural frame embeddings for candidate frames.

    Checks disk caches via ``backend_cache_key`` *before* instantiating heavy ML
    models so cached selections run offline in milliseconds.

    Returns ``(frame_indices, embeddings, relevance_scores, cache_path, cache_hit,
    resolved_backend_name, cancelled)``.
    """
    from annolid.services.embedding_search import (
        FrameEmbeddingBackend,
        VideoFrameEmbeddingIndex,
        _index_matches_request,
        backend_cache_key,
        build_embedding_backend,
        build_video_embedding_index,
        default_video_index_cache_path,
        prune_embedding_cache_dir,
    )

    span = max(1, hi - lo + 1)
    min_required = min(span, max(int(k_target), 16))
    stride = max(1, int(round(span / float(max(1, int(max_candidates))))))
    if span // stride > int(max_candidates):
        stride = max(1, (span + int(max_candidates) - 1) // int(max_candidates))

    active_index: Optional[VideoFrameEmbeddingIndex] = None
    cache_path_used: Optional[str] = None
    cache_hit = False
    resolved_backend_name = (
        backend_or_name.key[0]
        if isinstance(backend_or_name, FrameEmbeddingBackend)
        else str(backend_or_name)
    )
    resolved_backend_key: Optional[Tuple[str, ...]] = (
        tuple(backend_or_name.key)
        if isinstance(backend_or_name, FrameEmbeddingBackend)
        else None
    )

    # 1. Explicit precomputed index object or .npz path.
    if isinstance(precomputed_index, VideoFrameEmbeddingIndex):
        active_index = precomputed_index
        cache_hit = True
        bkey = tuple(str(x) for x in (active_index.metadata.get("backend_key") or ()))
        if bkey:
            resolved_backend_key = bkey
            resolved_backend_name = bkey[0]
    elif embedding_index_path:
        p = Path(embedding_index_path).expanduser().resolve()
        active_index = VideoFrameEmbeddingIndex.load_npz(p)
        cache_path_used = str(p)
        cache_hit = True
        bkey = tuple(str(x) for x in (active_index.metadata.get("backend_key") or ()))
        if bkey:
            resolved_backend_key = bkey
            resolved_backend_name = bkey[0]
    elif isinstance(backend_or_name, str) and backend_or_name == "cached_index":
        discovered = find_cached_video_embedding_indices(vpath, cache_dir=cache_dir)
        if not discovered:
            raise FileNotFoundError(
                f"No cached embedding index found for {vpath.name}. "
                "Select another feature backend or browse for an .npz index file."
            )
        p = Path(discovered[0].path)
        active_index = VideoFrameEmbeddingIndex.load_npz(p)
        cache_path_used = str(p)
        cache_hit = True
        resolved_backend_key = tuple(discovered[0].backend_key)
        resolved_backend_name = discovered[0].backend_name

    # 2. Compute deterministic backend cache key WITHOUT loading weights, and
    #    probe compatible disk caches first for zero-load offline reuse!
    effective_cache_path: Optional[Path] = None
    if active_index is None:
        if resolved_backend_key is None and resolved_backend_name in {
            "embeddinggemma2_onnx",
            "dinov3",
            "qwen3vl",
        }:
            resolved_backend_key = tuple(
                backend_cache_key(resolved_backend_name, backend_params)
            )
        if use_disk_cache and resolved_backend_key is not None:
            for existing in find_cached_video_embedding_indices(
                vpath, cache_dir=cache_dir
            ):
                if tuple(existing.backend_key) == resolved_backend_key:
                    try:
                        loaded = VideoFrameEmbeddingIndex.load_npz(existing.path)
                        in_range = (loaded.frame_indices >= lo) & (
                            loaded.frame_indices <= hi
                        )
                        if int(np.sum(in_range)) >= min_required:
                            active_index = loaded
                            cache_path_used = existing.path
                            cache_hit = True
                            break
                    except Exception:
                        continue

            if active_index is None:
                effective_cache_path = default_video_index_cache_path(
                    video_path=vpath,
                    backend_key=resolved_backend_key,
                    stride=stride,
                    max_frames=None,
                    clip_frames=1,
                    overlay_shapes=False,
                    cache_dir=cache_dir,
                )
                if effective_cache_path.is_file():
                    try:
                        loaded = VideoFrameEmbeddingIndex.load_npz(effective_cache_path)
                        in_range = (loaded.frame_indices >= lo) & (
                            loaded.frame_indices <= hi
                        )
                        if (
                            _index_matches_request(
                                loaded,
                                video_path=vpath,
                                backend_key=resolved_backend_key,
                                stride=stride,
                                max_frames=None,
                                clip_frames=1,
                                overlay_shapes=False,
                            )
                            and int(np.sum(in_range)) >= min_required
                        ):
                            active_index = loaded
                            cache_path_used = str(effective_cache_path)
                            cache_hit = True
                    except Exception:
                        pass

    # 3. Only instantiate a live backend if cache missed OR we need text encoding.
    clean_query = str(query_text or "").strip() or None
    needs_backend = active_index is None or clean_query is not None
    owns_backend = False
    active_backend: Optional[FrameEmbeddingBackend] = None

    try:
        if needs_backend:
            if isinstance(backend_or_name, FrameEmbeddingBackend):
                active_backend = backend_or_name
                resolved_backend_name = active_backend.key[0]
            else:
                bname = (
                    resolved_backend_name
                    if resolved_backend_name
                    in {"embeddinggemma2_onnx", "dinov3", "qwen3vl"}
                    else str(backend_or_name)
                )
                if bname in {"embeddinggemma2_onnx", "dinov3", "qwen3vl"}:
                    eff_params = (
                        _params_from_cached_backend_key(
                            resolved_backend_key or (), backend_params
                        )
                        if active_index is not None and resolved_backend_key
                        else dict(backend_params or {})
                    )
                    if progress_callback is not None:
                        progress_callback(8.0, f"Loading {bname} embedding backend…")
                    active_backend = build_embedding_backend(bname, eff_params)
                    owns_backend = True
                    resolved_backend_name = active_backend.key[0]

        if active_index is None:
            if active_backend is None:
                raise RuntimeError(
                    f"Unable to initialize embedding backend '{backend_or_name}'."
                )

            if effective_cache_path is None and use_disk_cache:
                effective_cache_path = default_video_index_cache_path(
                    video_path=vpath,
                    backend_key=active_backend.key,
                    stride=stride,
                    max_frames=None,
                    clip_frames=1,
                    overlay_shapes=False,
                    cache_dir=cache_dir,
                )

            def _on_embed_progress(done: int, total: int) -> None:
                if progress_callback is not None and total > 0:
                    pct = 10.0 + 62.0 * (float(done) / float(total))
                    progress_callback(
                        pct,
                        f"Embedding frames with {resolved_backend_name} ({done}/{total})…",
                    )

            active_index, completed = build_video_embedding_index(
                vpath,
                backend=active_backend,
                stride=stride,
                max_frames=None,
                clip_frames=1,
                batch_size=int((backend_params or {}).get("batch_size") or 4),
                overlay_shapes=False,
                is_stopped=is_stopped,
                on_progress=_on_embed_progress,
            )
            if not completed:
                return (
                    [],
                    np.zeros((0, 0), dtype=np.float32),
                    None,
                    None,
                    False,
                    resolved_backend_name,
                    True,
                )
            if completed and effective_cache_path is not None and len(active_index) > 0:
                try:
                    active_index.save_npz(effective_cache_path)
                    prune_embedding_cache_dir(effective_cache_path.parent)
                    cache_path_used = str(effective_cache_path)
                except OSError as exc:
                    logger.warning(
                        "Could not persist embedding cache %s: %s",
                        effective_cache_path,
                        exc,
                    )

        assert active_index is not None
        if cache_hit and progress_callback is not None:
            progress_callback(
                68.0,
                f"Reused cached {resolved_backend_name} embeddings ({len(active_index)} frames).",
            )

        mask = (active_index.frame_indices >= int(lo)) & (
            active_index.frame_indices <= int(hi)
        )
        frame_indices = [int(v) for v in active_index.frame_indices[mask].tolist()]
        embeddings = active_index.embeddings[mask].astype(np.float32, copy=False)

        relevance_scores: Optional[np.ndarray] = None
        if clean_query is not None and len(frame_indices) > 0:
            if active_backend is not None and active_backend.supports_text:
                qvec = active_backend.embed_text(clean_query).astype(
                    np.float32, copy=False
                )
                emb_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
                # Support Matryoshka truncation if text encoder returned full 768d
                # while cached index was truncated (e.g. 256d).
                if emb_dim > 0 and qvec.shape[0] > emb_dim:
                    qvec = qvec[:emb_dim]
                qnorm = float(np.linalg.norm(qvec))
                if qnorm > 1e-8:
                    qvec = (qvec / qnorm).astype(np.float32, copy=False)
                if qvec.shape[0] == emb_dim:
                    relevance_scores = (embeddings @ qvec).astype(
                        np.float32, copy=False
                    )
            else:
                logger.info(
                    "Ignoring text query %r because backend %r does not support text.",
                    clean_query,
                    resolved_backend_name,
                )

        return (
            frame_indices,
            _normalize_feature_matrix(embeddings),
            relevance_scores,
            cache_path_used,
            cache_hit,
            resolved_backend_name,
            False,
        )
    finally:
        if owns_backend and active_backend is not None:
            try:
                active_backend.close()
            except Exception:
                pass


def _compute_quality_keep_mask(
    sharpness: np.ndarray,
    luma: np.ndarray,
    *,
    min_keep: int,
) -> np.ndarray:
    """Filter severely blurred or nearly black frames while preserving at least ``min_keep``."""
    n = int(sharpness.shape[0])
    if n == 0 or n <= min_keep:
        return np.ones((n,), dtype=bool)
    med_sharp = float(np.median(sharpness)) if n > 0 else 0.0
    sharp_floor = max(4.0, 0.22 * med_sharp) if med_sharp > 12.0 else 0.0
    keep = (sharpness >= sharp_floor) & (luma >= 8.0)
    if int(np.sum(keep)) >= min_keep:
        return keep
    # Safety fallback: keep the sharpest min_keep frames.
    order = np.argsort(-sharpness)
    fallback = np.zeros((n,), dtype=bool)
    fallback[order[:min_keep]] = True
    return fallback


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32, copy=False)


def _format_selection_reason(
    *,
    role: str,
    cluster_id: Optional[int],
    diversity_score: float,
    novelty_score: float,
    motion_score: float,
    relevance_score: Optional[float],
) -> str:
    parts: List[str] = []
    if role == "boundary":
        parts.append("Start anchor (VOS seed)")
    elif role == "representative":
        if cluster_id is not None:
            parts.append(f"Cluster #{int(cluster_id) + 1} medoid")
        else:
            parts.append("Representative state")
    elif role == "diverse_extreme":
        parts.append("Novel / rare state")
    elif role == "motion_peak":
        parts.append("High motion peak")
    elif role == "uniform":
        parts.append("Uniform interval")
    elif role == "random":
        parts.append("Random sample")
    elif role == "keyframe":
        parts.append("Video keyframe")
    else:
        parts.append(role)

    if relevance_score is not None:
        parts.append(f"match={relevance_score:.2f}")
    if diversity_score > 0.0:
        parts.append(f"div={diversity_score:.2f}")
    elif novelty_score > 0.0:
        parts.append(f"nov={novelty_score:.2f}")
    if motion_score >= 0.70:
        parts.append("high motion")
    return " · ".join(parts)


def select_frames_for_labeling(
    video_path: Union[str, Path],
    *,
    config: Optional[FrameSelectionConfig] = None,
    backend: Optional[Any] = None,
    precomputed_index: Optional[Any] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    is_stopped: Optional[Callable[[], bool]] = None,
) -> FrameSelectionResult:
    """Select diverse frames for pose or segmentation labeling — with or without disk extraction.

    When ``config.extract_images`` is ``False``, no PNG/JPG images are written
    to disk; instead, a structured ``FrameSelectionResult`` (and optional
    ``frame_selection_queue.json``) is returned for direct in-video labeling.
    When ``config.extract_images`` is ``True``, selected frames are also written
    to ``output_dir`` along with ``frame_selection_manifest.json``.
    """
    cfg = config if config is not None else FrameSelectionConfig()
    profile = get_task_profile(cfg.task)
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    algo_norm = str(cfg.algo or profile.default_algo).strip().lower()
    if algo_norm not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported selection algorithm {cfg.algo!r}. "
            f"Choose one of: {', '.join(SUPPORTED_ALGOS)}."
        )

    ann_dir = (
        Path(cfg.annotation_dir).expanduser().resolve()
        if cfg.annotation_dir is not None and str(cfg.annotation_dir).strip()
        else vpath.with_suffix("")
    )
    total_frames, fps, lo, hi = _resolve_video_bounds(
        vpath,
        start_seconds=cfg.start_seconds,
        end_seconds=cfg.end_seconds,
        start_frame=cfg.start_frame,
        end_frame=cfg.end_frame,
    )

    if cfg.extract_images:
        out_dir_path: Optional[Path] = (
            Path(cfg.output_dir).expanduser().resolve()
            if cfg.output_dir is not None and str(cfg.output_dir).strip()
            else vpath.parent / f"{vpath.stem.replace(' ', '_')}_{algo_norm}"
        )
    else:
        out_dir_path = None

    if total_frames <= 0:
        return FrameSelectionResult(
            video_path=str(vpath),
            task=profile.key,
            algo=algo_norm,
            feature_backend="none",
            extract_images=bool(cfg.extract_images),
            output_dir=str(out_dir_path) if out_dir_path is not None else None,
            total_video_frames=0,
            fps=fps,
            start_frame=0,
            end_frame=0,
            candidate_count=0,
            excluded_labeled_count=0,
            filtered_blur_count=0,
            selected_items=(),
        )

    span = hi - lo + 1
    k_target = (
        span if int(cfg.num_frames) == -1 else max(0, min(int(cfg.num_frames), span))
    )

    # Discover already-labeled frames from both disk JSONs and NDJSON AnnotationStore.
    labeled_frames: set[int] = (
        discover_labeled_frame_indices(ann_dir, require_shapes=True)
        if cfg.exclude_labeled or cfg.seed_from_labeled
        else set()
    )
    labeled_in_range = {f for f in labeled_frames if lo <= f <= hi}

    if cfg.min_frame_gap is None or int(cfg.min_frame_gap) < 0:
        if k_target > 1 and span > k_target * 2:
            eff_min_gap = min(
                profile.recommended_min_gap, max(1, span // max(k_target * 4, 1))
            )
        else:
            eff_min_gap = 0
    else:
        eff_min_gap = max(0, int(cfg.min_frame_gap))

    include_boundaries = (
        profile.include_boundaries
        if cfg.include_boundaries is None
        else bool(cfg.include_boundaries)
    )
    rep_ratio = (
        profile.representative_ratio
        if cfg.representative_ratio is None
        else float(cfg.representative_ratio)
    )

    if k_target == 0:
        return FrameSelectionResult(
            video_path=str(vpath),
            task=profile.key,
            algo=algo_norm,
            feature_backend="none",
            extract_images=bool(cfg.extract_images),
            output_dir=str(out_dir_path) if out_dir_path is not None else None,
            total_video_frames=total_frames,
            fps=fps,
            start_frame=lo,
            end_frame=hi,
            candidate_count=0,
            excluded_labeled_count=0,
            filtered_blur_count=0,
            selected_items=(),
        )

    # Fast-path when requesting all frames (`num_frames == -1`) without blur/text filtering.
    if (
        int(cfg.num_frames) == -1
        and algo_norm not in {"keyframes"}
        and not cfg.filter_blur
        and not (cfg.query_text and str(cfg.query_text).strip())
    ):
        chosen_all = [
            idx
            for idx in range(lo, hi + 1)
            if not (cfg.exclude_labeled and idx in labeled_in_range)
        ]
        all_items = [
            SelectedFrameItem(
                frame_index=idx,
                timestamp_sec=(float(idx) / fps) if fps > 0.0 else None,
                selection_role="all",
                already_labeled=(idx in labeled_frames),
                reason="All frames in range",
            )
            for idx in chosen_all
        ]
        return _finalize_selection_result(
            vpath=vpath,
            cfg=cfg,
            profile=profile,
            algo_norm=algo_norm,
            resolved_backend_name="none",
            out_dir_path=out_dir_path,
            ann_dir=ann_dir,
            total_frames=total_frames,
            fps=fps,
            lo=lo,
            hi=hi,
            candidate_count=span,
            excluded_labeled_count=len(labeled_in_range) if cfg.exclude_labeled else 0,
            filtered_blur_count=0,
            selected_items=all_items,
            embedding_cache_path=None,
            embedding_cache_hit=False,
            progress_callback=progress_callback,
            is_stopped=is_stopped,
        )

    # ------------------------------------------------------------------
    # Branch 1: Non-embedding / structural algorithms (uniform, random, flow, keyframes)
    # ------------------------------------------------------------------
    if algo_norm in {"uniform", "random", "flow", "keyframes"}:
        resolved_backend_name = "none"
        motion_by_frame: Dict[int, float] = {}
        role_tag = algo_norm
        if algo_norm == "uniform":
            if progress_callback is not None:
                progress_callback(
                    20.0, f"Selecting {k_target} uniformly spaced frames…"
                )
            pool = [
                idx
                for idx in range(lo, hi + 1)
                if not (cfg.exclude_labeled and idx in labeled_in_range)
            ]
            if not pool:
                pool = list(range(lo, hi + 1))
            pos = select_uniform_frame_indices(len(pool), k_target)
            chosen_indices = [pool[p] for p in pos]
        elif algo_norm == "random":
            if progress_callback is not None:
                progress_callback(20.0, f"Sampling {k_target} random frames…")
            pool = [
                idx
                for idx in range(lo, hi + 1)
                if not (cfg.exclude_labeled and idx in labeled_in_range)
            ]
            if not pool:
                pool = list(range(lo, hi + 1))
            rng = np.random.default_rng(int(cfg.seed))
            if k_target >= len(pool):
                chosen_indices = list(pool)
            else:
                chosen_indices = sorted(
                    int(v) for v in rng.choice(pool, size=k_target, replace=False)
                )
        elif algo_norm == "flow":
            role_tag = "motion_peak"
            chosen_indices, motion_by_frame = select_flow_frame_indices(
                vpath,
                k_target,
                start_frame=lo,
                end_frame=hi,
                max_candidates=max(int(cfg.max_candidates), k_target * 4),
                min_frame_gap=eff_min_gap,
                progress_callback=progress_callback,
                is_stopped=is_stopped,
            )
            if cfg.exclude_labeled and labeled_in_range:
                chosen_indices = [
                    idx for idx in chosen_indices if idx not in labeled_in_range
                ]
        else:  # keyframes
            role_tag = "keyframe"
            if progress_callback is not None:
                progress_callback(25.0, "Reading video keyframe timestamps…")
            max_kf = None if int(cfg.num_frames) == -1 else k_target
            chosen_indices = select_keyframe_indices(
                vpath, start_frame=lo, end_frame=hi, max_frames=max_kf
            )
            if cfg.exclude_labeled and labeled_in_range:
                chosen_indices = [
                    idx for idx in chosen_indices if idx not in labeled_in_range
                ]

        if is_stopped is not None and is_stopped():
            return FrameSelectionResult(
                video_path=str(vpath),
                task=profile.key,
                algo=algo_norm,
                feature_backend=resolved_backend_name,
                extract_images=bool(cfg.extract_images),
                output_dir=str(out_dir_path) if out_dir_path is not None else None,
                total_video_frames=total_frames,
                fps=fps,
                start_frame=lo,
                end_frame=hi,
                candidate_count=span,
                excluded_labeled_count=len(labeled_in_range)
                if cfg.exclude_labeled
                else 0,
                filtered_blur_count=0,
                selected_items=(),
                cancelled=True,
            )

        max_m = max(motion_by_frame.values(), default=0.0)
        items_list: List[SelectedFrameItem] = []
        for idx in sorted(set(int(f) for f in chosen_indices)):
            m_norm = (
                float(motion_by_frame.get(idx, 0.0)) / max_m if max_m > 1e-6 else 0.0
            )
            items_list.append(
                SelectedFrameItem(
                    frame_index=idx,
                    timestamp_sec=(float(idx) / fps) if fps > 0.0 else None,
                    motion_score=m_norm,
                    selection_role=role_tag,
                    already_labeled=(idx in labeled_frames),
                    reason=_format_selection_reason(
                        role=role_tag,
                        cluster_id=None,
                        diversity_score=0.0,
                        novelty_score=0.0,
                        motion_score=m_norm,
                        relevance_score=None,
                    ),
                )
            )
        return _finalize_selection_result(
            vpath=vpath,
            cfg=cfg,
            profile=profile,
            algo_norm=algo_norm,
            resolved_backend_name=resolved_backend_name,
            out_dir_path=out_dir_path,
            ann_dir=ann_dir,
            total_frames=total_frames,
            fps=fps,
            lo=lo,
            hi=hi,
            candidate_count=span,
            excluded_labeled_count=len(labeled_in_range) if cfg.exclude_labeled else 0,
            filtered_blur_count=0,
            selected_items=items_list,
            embedding_cache_path=None,
            embedding_cache_hit=False,
            progress_callback=progress_callback,
            is_stopped=is_stopped,
        )

    # ------------------------------------------------------------------
    # Branch 2: Feature & Embedding Diversity Algorithms (hybrid, kmeans, farthest)
    # ------------------------------------------------------------------
    requested_backend = str(cfg.feature_backend or "auto").strip().lower()
    auto_cached_path: Optional[str] = None
    if backend is not None:
        effective_backend_mode = "embedding_instance"
    elif precomputed_index is not None or cfg.embedding_index_path:
        effective_backend_mode = "cached_index"
    elif requested_backend == "auto":
        cached_list = (
            find_cached_video_embedding_indices(vpath, cache_dir=cache_dir)
            if cfg.use_embedding_disk_cache
            else []
        )
        # Only auto-reuse a cached .npz index if it has sufficient frames in [lo, hi].
        effective_backend_mode = "visual_fg"
        if cached_list:
            from annolid.services.embedding_search import VideoFrameEmbeddingIndex

            min_needed = min(span, max(k_target, 1))
            for cand_cache in cached_list:
                try:
                    loaded_check = VideoFrameEmbeddingIndex.load_npz(cand_cache.path)
                    in_rng = int(
                        np.sum(
                            (loaded_check.frame_indices >= lo)
                            & (loaded_check.frame_indices <= hi)
                        )
                    )
                    if in_rng >= min_needed:
                        effective_backend_mode = "cached_index"
                        auto_cached_path = cand_cache.path
                        break
                except Exception:
                    continue
    else:
        effective_backend_mode = requested_backend

    embedding_cache_path: Optional[str] = None
    embedding_cache_hit = False
    relevance_scores_all: Optional[np.ndarray] = None
    sharpness_all: Optional[np.ndarray] = None
    luma_all: Optional[np.ndarray] = None
    motion_all: Optional[np.ndarray] = None
    filtered_blur_count = 0

    if effective_backend_mode in {
        "embedding_instance",
        "embeddinggemma2_onnx",
        "dinov3",
        "qwen3vl",
        "cached_index",
    }:
        (
            cand_indices,
            feat_matrix,
            relevance_scores_all,
            embedding_cache_path,
            embedding_cache_hit,
            resolved_backend_name,
            cancelled,
        ) = _compute_embedding_features_for_video(
            vpath,
            backend_or_name=(
                backend if backend is not None else effective_backend_mode
            ),
            backend_params=cfg.backend_params,
            total_frames=total_frames,
            lo=lo,
            hi=hi,
            k_target=k_target,
            max_candidates=max(int(cfg.max_candidates), k_target * 4),
            query_text=cfg.query_text,
            precomputed_index=precomputed_index,
            embedding_index_path=cfg.embedding_index_path or auto_cached_path,
            use_disk_cache=cfg.use_embedding_disk_cache,
            cache_dir=cache_dir,
            progress_callback=progress_callback,
            is_stopped=is_stopped,
        )
        if cancelled:
            return FrameSelectionResult(
                video_path=str(vpath),
                task=profile.key,
                algo=algo_norm,
                feature_backend=resolved_backend_name,
                extract_images=bool(cfg.extract_images),
                output_dir=str(out_dir_path) if out_dir_path is not None else None,
                total_video_frames=total_frames,
                fps=fps,
                start_frame=lo,
                end_frame=hi,
                candidate_count=0,
                excluded_labeled_count=0,
                filtered_blur_count=0,
                selected_items=(),
                cancelled=True,
            )
        # Approximate motion energy directly in embedding space (cosine delta to prev frame).
        if len(cand_indices) > 1:
            diffs = 1.0 - np.sum(feat_matrix[1:] * feat_matrix[:-1], axis=1)
            motion_all = np.concatenate([[float(diffs[0])], diffs]).astype(
                np.float32, copy=False
            )
        else:
            motion_all = np.zeros((len(cand_indices),), dtype=np.float32)
    else:
        resolved_backend_name = (
            "visual" if effective_backend_mode == "visual" else "visual_fg"
        )
        cand_count = min(span, max(int(cfg.max_candidates), k_target * 4))
        raw_candidates = select_uniform_frame_indices(
            total_frames, cand_count, start_frame=lo, end_frame=hi
        )
        # Ensure already-labeled seeds in range (capped at 120) are represented if needed.
        if cfg.seed_from_labeled and labeled_in_range:
            seed_subset = sorted(labeled_in_range)
            if len(seed_subset) > 120:
                sub_pos = select_uniform_frame_indices(len(seed_subset), 120)
                seed_subset = [seed_subset[p] for p in sub_pos]
            raw_candidates = sorted(set(raw_candidates) | set(seed_subset))

        (
            cand_indices,
            thumbs,
            sharpness_all,
            luma_all,
            cancelled,
        ) = _read_candidate_thumbnails_and_quality(
            vpath,
            raw_candidates,
            resize_width=48,
            progress_callback=progress_callback,
            is_stopped=is_stopped,
        )
        if cancelled:
            return FrameSelectionResult(
                video_path=str(vpath),
                task=profile.key,
                algo=algo_norm,
                feature_backend=resolved_backend_name,
                extract_images=bool(cfg.extract_images),
                output_dir=str(out_dir_path) if out_dir_path is not None else None,
                total_video_frames=total_frames,
                fps=fps,
                start_frame=lo,
                end_frame=hi,
                candidate_count=0,
                excluded_labeled_count=0,
                filtered_blur_count=0,
                selected_items=(),
                cancelled=True,
            )

        if not cand_indices:
            return FrameSelectionResult(
                video_path=str(vpath),
                task=profile.key,
                algo=algo_norm,
                feature_backend=resolved_backend_name,
                extract_images=bool(cfg.extract_images),
                output_dir=str(out_dir_path) if out_dir_path is not None else None,
                total_video_frames=total_frames,
                fps=fps,
                start_frame=lo,
                end_frame=hi,
                candidate_count=0,
                excluded_labeled_count=0,
                filtered_blur_count=0,
                selected_items=(),
            )

        if progress_callback is not None:
            progress_callback(
                70.0,
                f"Computing {resolved_backend_name} features for {len(cand_indices)} frames…",
            )
        if resolved_backend_name == "visual_fg":
            feat_matrix, motion_all = _compute_foreground_visual_features(
                thumbs,
                foreground_weight=profile.foreground_weight,
                appearance_weight=profile.appearance_weight,
                edge_weight=profile.edge_weight,
            )
        else:
            feat_matrix = _normalize_feature_matrix(
                np.stack(
                    [_frame_visual_descriptor(t, resize_width=48) for t in thumbs],
                    axis=0,
                )
            )
            grays = [
                cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                for t in thumbs
            ]
            motion_all = np.zeros((len(thumbs),), dtype=np.float32)
            for i in range(1, len(grays)):
                motion_all[i] = float(np.mean(np.abs(grays[i] - grays[i - 1])))
            if len(thumbs) > 1:
                motion_all[0] = motion_all[1]

    if not cand_indices or feat_matrix.shape[0] == 0:
        return FrameSelectionResult(
            video_path=str(vpath),
            task=profile.key,
            algo=algo_norm,
            feature_backend=resolved_backend_name,
            extract_images=bool(cfg.extract_images),
            output_dir=str(out_dir_path) if out_dir_path is not None else None,
            total_video_frames=total_frames,
            fps=fps,
            start_frame=lo,
            end_frame=hi,
            candidate_count=0,
            excluded_labeled_count=0,
            filtered_blur_count=0,
            selected_items=(),
            embedding_cache_path=embedding_cache_path,
            embedding_cache_hit=embedding_cache_hit,
            query_text=cfg.query_text,
        )

    total_candidates = len(cand_indices)
    cand_arr = np.asarray(cand_indices, dtype=np.int64)

    # Separate already-labeled seeds from unlabeled candidate pool.
    labeled_mask = np.asarray(
        [int(f) in labeled_in_range for f in cand_indices], dtype=bool
    )
    seed_features: Optional[np.ndarray] = None
    if cfg.seed_from_labeled and labeled_in_range:
        seed_rows: set[int] = set(int(r) for r in np.where(labeled_mask)[0].tolist())
        # For strided embedding indices where a labeled frame may fall between sampled
        # indices, map nearby labeled frames to their nearest candidate within stride/2.
        if len(cand_arr) > 1 and len(seed_rows) < len(labeled_in_range):
            approx_step = int(max(1, round(float(np.median(np.diff(cand_arr))))))
            tol = max(1, approx_step // 2)
            for lf in sorted(labeled_in_range):
                pos = int(np.argmin(np.abs(cand_arr - int(lf))))
                if abs(int(cand_arr[pos]) - int(lf)) <= tol:
                    seed_rows.add(pos)
        if seed_rows:
            seed_features = feat_matrix[sorted(seed_rows)]

    keep_mask = np.ones((total_candidates,), dtype=bool)
    excluded_labeled_count = 0
    if cfg.exclude_labeled and np.any(labeled_mask):
        unlabeled_count = int(np.sum(~labeled_mask))
        if unlabeled_count > 0:
            keep_mask &= ~labeled_mask
            excluded_labeled_count = int(np.sum(labeled_mask))

    if (
        cfg.filter_blur
        and sharpness_all is not None
        and luma_all is not None
        and int(np.sum(keep_mask)) > k_target
    ):
        active_rows = np.where(keep_mask)[0]
        q_mask = _compute_quality_keep_mask(
            sharpness_all[active_rows],
            luma_all[active_rows],
            min_keep=min(len(active_rows), max(k_target, 1)),
        )
        filtered_blur_count = int(len(active_rows) - int(np.sum(q_mask)))
        new_keep = np.zeros_like(keep_mask)
        new_keep[active_rows[q_mask]] = True
        keep_mask = new_keep

    pool_rows = np.where(keep_mask)[0]
    pool_indices = [int(cand_arr[r]) for r in pool_rows.tolist()]
    pool_features = feat_matrix[pool_rows]
    pool_rel = (
        relevance_scores_all[pool_rows] if relevance_scores_all is not None else None
    )

    # When a text focus prompt is supplied, narrow candidate pool to top matching quantile
    # before running diversity clustering so picks are both relevant AND visually diverse.
    rel_weight = float(cfg.relevance_weight) if pool_rel is not None else 0.0
    if pool_rel is not None and len(pool_indices) > max(k_target * 3, 12):
        keep_n = min(len(pool_indices), max(k_target * 4, len(pool_indices) // 3))
        top_rel_order = np.argsort(-pool_rel)[:keep_n]
        top_rel_sorted = np.sort(top_rel_order)
        pool_rows = pool_rows[top_rel_sorted]
        pool_indices = [int(cand_arr[r]) for r in pool_rows.tolist()]
        pool_features = feat_matrix[pool_rows]
        pool_rel = relevance_scores_all[pool_rows]

    k_eff = min(k_target, len(pool_indices))
    if progress_callback is not None:
        progress_callback(
            78.0,
            f"Selecting {k_eff} diverse frames ({algo_norm} · {resolved_backend_name})…",
        )

    role_by_local: Dict[int, str] = {}
    cluster_labels: Optional[np.ndarray] = None
    if algo_norm == "kmeans":
        chosen_local, cluster_labels, _centroids = _kmeans_cluster_medoids(
            pool_features,
            k_eff,
            frame_indices=pool_indices,
            min_frame_gap=eff_min_gap,
            seed=int(cfg.seed),
        )
        role_by_local = {r: "representative" for r in chosen_local}
        if include_boundaries and 0 not in set(chosen_local) and len(pool_indices) > 0:
            if len(chosen_local) >= k_eff and chosen_local:
                chosen_local = sorted([0] + chosen_local[:-1])
            else:
                chosen_local = sorted([0] + chosen_local)
            role_by_local[0] = "boundary"
    elif algo_norm == "farthest":
        chosen_local = sorted(
            select_farthest_point_indices(
                pool_features,
                k_eff,
                frame_indices=pool_indices,
                seed_features=seed_features,
                min_frame_gap=eff_min_gap,
                relevance_scores=pool_rel,
                relevance_weight=rel_weight,
            )
        )
        role_by_local = {r: "diverse_extreme" for r in chosen_local}
        if include_boundaries and 0 not in set(chosen_local) and len(pool_indices) > 0:
            if len(chosen_local) >= k_eff and chosen_local:
                chosen_local = sorted([0] + chosen_local[:-1])
            else:
                chosen_local = sorted([0] + chosen_local)
            role_by_local[0] = "boundary"
    else:  # hybrid
        chosen_local, role_by_local, cluster_labels = select_hybrid_frame_indices(
            pool_features,
            k_eff,
            frame_indices=pool_indices,
            seed_features=seed_features,
            min_frame_gap=eff_min_gap,
            representative_ratio=rep_ratio,
            include_start_boundary=include_boundaries,
            relevance_scores=pool_rel,
            relevance_weight=rel_weight,
            seed=int(cfg.seed),
        )

    # Compute clean diagnostics per chosen frame.
    motion_norm_all = (
        _normalize_0_1(motion_all)
        if motion_all is not None
        else np.zeros((total_candidates,), dtype=np.float32)
    )
    sharp_norm_all = (
        _normalize_0_1(sharpness_all)
        if sharpness_all is not None
        else np.ones((total_candidates,), dtype=np.float32)
    )

    chosen_feats = (
        pool_features[chosen_local]
        if chosen_local
        else np.zeros((0, pool_features.shape[1]), dtype=np.float32)
    )
    pairwise_dist: Optional[np.ndarray] = None
    if chosen_feats.shape[0] > 1:
        pairwise_dist = np.clip(1.0 - (chosen_feats @ chosen_feats.T), 0.0, 2.0)
        np.fill_diagonal(pairwise_dist, np.inf)

    global_centroid = np.mean(pool_features, axis=0)
    gc_norm = float(np.linalg.norm(global_centroid))
    if gc_norm > 1e-8:
        global_centroid = global_centroid / gc_norm

    items: List[SelectedFrameItem] = []
    for pos_in_chosen, local_row in enumerate(chosen_local):
        orig_row = int(pool_rows[local_row])
        fidx = int(pool_indices[local_row])
        vec = pool_features[local_row]
        div_score = (
            float(np.min(pairwise_dist[pos_in_chosen]))
            if pairwise_dist is not None
            else 0.0
        )
        if seed_features is not None and seed_features.shape[0] > 0:
            nov_score = float(np.min(1.0 - (seed_features @ vec)))
        else:
            nov_score = float(max(0.0, 1.0 - float(vec @ global_centroid)))
        m_score = float(motion_norm_all[orig_row])
        s_score = float(sharp_norm_all[orig_row])
        rel_val = float(pool_rel[local_row]) if pool_rel is not None else None
        role = role_by_local.get(local_row, "representative")
        cid = (
            int(cluster_labels[local_row])
            if cluster_labels is not None and local_row < len(cluster_labels)
            else None
        )
        reason = _format_selection_reason(
            role=role,
            cluster_id=cid if role == "representative" else None,
            diversity_score=div_score,
            novelty_score=nov_score,
            motion_score=m_score,
            relevance_score=rel_val,
        )
        items.append(
            SelectedFrameItem(
                frame_index=fidx,
                timestamp_sec=(float(fidx) / fps) if fps > 0.0 else None,
                cluster_id=cid,
                diversity_score=div_score,
                novelty_score=nov_score,
                motion_score=m_score,
                sharpness_score=s_score,
                relevance_score=rel_val,
                selection_role=role,
                already_labeled=(fidx in labeled_frames),
                reason=reason,
            )
        )

    return _finalize_selection_result(
        vpath=vpath,
        cfg=cfg,
        profile=profile,
        algo_norm=algo_norm,
        resolved_backend_name=resolved_backend_name,
        out_dir_path=out_dir_path,
        ann_dir=ann_dir,
        total_frames=total_frames,
        fps=fps,
        lo=lo,
        hi=hi,
        candidate_count=total_candidates,
        excluded_labeled_count=excluded_labeled_count,
        filtered_blur_count=filtered_blur_count,
        selected_items=items,
        embedding_cache_path=embedding_cache_path,
        embedding_cache_hit=embedding_cache_hit,
        progress_callback=progress_callback,
        is_stopped=is_stopped,
    )


def _finalize_selection_result(
    *,
    vpath: Path,
    cfg: FrameSelectionConfig,
    profile: TaskProfile,
    algo_norm: str,
    resolved_backend_name: str,
    out_dir_path: Optional[Path],
    ann_dir: Path,
    total_frames: int,
    fps: float,
    lo: int,
    hi: int,
    candidate_count: int,
    excluded_labeled_count: int,
    filtered_blur_count: int,
    selected_items: List[SelectedFrameItem],
    embedding_cache_path: Optional[str],
    embedding_cache_hit: bool,
    progress_callback: Optional[Callable[[float, str], None]],
    is_stopped: Optional[Callable[[], bool]],
) -> FrameSelectionResult:
    saved_paths: Tuple[str, ...] = ()
    manifest_path: Optional[str] = None
    final_items = list(selected_items)

    if cfg.extract_images and out_dir_path is not None:
        indices_to_save = [item.frame_index for item in final_items]
        saved_idx, saved_files = _write_selected_frames(
            vpath,
            out_dir_path,
            indices_to_save,
            image_format=cfg.image_format,
            progress_callback=progress_callback,
            progress_start=84.0,
        )
        saved_by_idx = {
            int(idx): str(path) for idx, path in zip(saved_idx, saved_files)
        }
        final_items = [
            SelectedFrameItem(
                frame_index=item.frame_index,
                timestamp_sec=item.timestamp_sec,
                cluster_id=item.cluster_id,
                diversity_score=item.diversity_score,
                novelty_score=item.novelty_score,
                motion_score=item.motion_score,
                sharpness_score=item.sharpness_score,
                relevance_score=item.relevance_score,
                selection_role=item.selection_role,
                already_labeled=item.already_labeled,
                reason=item.reason,
                saved_path=saved_by_idx.get(item.frame_index),
            )
            for item in final_items
            if item.frame_index in saved_by_idx
        ]
        saved_paths = tuple(saved_files)
        manifest_file = out_dir_path / EXTRACT_MANIFEST_FILENAME
        manifest_path = str(manifest_file)
    elif cfg.persist_queue_manifest and final_items:
        if cfg.output_dir is not None and str(cfg.output_dir).strip():
            custom_q = Path(cfg.output_dir).expanduser().resolve()
            queue_file = (
                custom_q
                if custom_q.suffix.lower() == ".json"
                else custom_q / QUEUE_MANIFEST_FILENAME
            )
        else:
            queue_file = default_queue_manifest_path(vpath, ann_dir)
        manifest_path = str(queue_file)

    result = FrameSelectionResult(
        video_path=str(vpath),
        task=profile.key,
        algo=algo_norm,
        feature_backend=resolved_backend_name,
        extract_images=bool(cfg.extract_images),
        output_dir=str(out_dir_path) if out_dir_path is not None else None,
        total_video_frames=int(total_frames),
        fps=float(fps),
        start_frame=int(lo),
        end_frame=int(hi),
        candidate_count=int(candidate_count),
        excluded_labeled_count=int(excluded_labeled_count),
        filtered_blur_count=int(filtered_blur_count),
        selected_items=tuple(final_items),
        saved_paths=saved_paths,
        manifest_path=manifest_path,
        embedding_cache_path=embedding_cache_path,
        embedding_cache_hit=bool(embedding_cache_hit),
        query_text=str(cfg.query_text).strip() if cfg.query_text else None,
        cancelled=bool(is_stopped is not None and is_stopped()),
    )

    if manifest_path is not None:
        try:
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(Path(manifest_path), result.to_dict(), indent=2)
        except OSError as exc:
            logger.warning(
                "Failed writing selection manifest %s: %s", manifest_path, exc
            )

    if progress_callback is not None:
        if cfg.extract_images:
            progress_callback(
                100.0,
                f"Extracted {len(saved_paths)} frames to {out_dir_path}",
            )
        else:
            progress_callback(
                100.0,
                f"Selected {len(final_items)} frames for in-video labeling (no files extracted).",
            )
    return result


def materialize_selected_frames(
    video_path: Union[str, Path],
    items_or_indices: Sequence[Union[int, SelectedFrameItem]],
    output_dir: Union[str, Path],
    *,
    image_format: str = "png",
    copy_existing_labels: bool = False,
    annotation_dir: Optional[Union[str, Path]] = None,
    base_result: Optional[FrameSelectionResult] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> FrameSelectionResult:
    """Write an existing virtual selection queue to disk as images + manifest."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_frames, fps, _lo, _hi = _resolve_video_bounds(
        vpath,
        start_seconds=None,
        end_seconds=None,
        start_frame=None,
        end_frame=None,
    )

    items: List[SelectedFrameItem] = []
    for entry in items_or_indices:
        if isinstance(entry, SelectedFrameItem):
            items.append(entry)
        else:
            idx = int(entry)
            items.append(
                SelectedFrameItem(
                    frame_index=idx,
                    timestamp_sec=(float(idx) / fps) if fps > 0.0 else None,
                    selection_role="queued",
                    reason="Exported from label queue",
                )
            )
    items.sort(key=lambda item: item.frame_index)

    indices = [item.frame_index for item in items]
    saved_idx, saved_files = _write_selected_frames(
        vpath,
        out_root,
        indices,
        image_format=image_format,
        progress_callback=progress_callback,
        progress_start=5.0,
    )
    saved_by_idx = {int(i): str(p) for i, p in zip(saved_idx, saved_files)}

    if copy_existing_labels:
        from annolid.infrastructure import AnnotationStore, load_labelme_json

        src_ann_dir = (
            Path(annotation_dir).expanduser().resolve()
            if annotation_dir is not None and str(annotation_dir).strip()
            else vpath.with_suffix("")
        )
        if src_ann_dir.is_dir():
            store: Optional[Any] = None
            try:
                stub = src_ann_dir / f"{src_ann_dir.name}_000000000.json"
                cand_store = AnnotationStore.for_frame_path(stub)
                if cand_store.store_path.is_file():
                    store = cand_store
            except Exception:
                store = None

            for idx, img_path_str in saved_by_idx.items():
                img_path = Path(img_path_str)
                dst_json = img_path.with_suffix(".json")
                payload: Optional[Dict[str, Any]] = None
                candidates = (
                    src_ann_dir / f"{src_ann_dir.name}_{idx:09d}.json",
                    src_ann_dir / f"{vpath.stem}_{idx:09d}.json",
                    src_ann_dir / f"{idx:09d}.json",
                )
                for src_json in candidates:
                    if src_json.is_file():
                        try:
                            loaded = load_labelme_json(src_json)
                            if isinstance(loaded, dict):
                                payload = dict(loaded)
                                break
                        except Exception:
                            continue
                if payload is None and store is not None:
                    try:
                        rec = store.get_frame(int(idx))
                        if isinstance(rec, dict) and rec.get("shapes"):
                            payload = {
                                "version": str(rec.get("version") or "5.0.1"),
                                "flags": dict(rec.get("flags") or {}),
                                "shapes": list(rec.get("shapes") or []),
                                "imagePath": img_path.name,
                                "imageData": None,
                                "imageHeight": rec.get("imageHeight"),
                                "imageWidth": rec.get("imageWidth"),
                            }
                    except Exception:
                        payload = None
                if payload is not None:
                    payload["imagePath"] = img_path.name
                    payload["imageData"] = None
                    try:
                        atomic_write_json(dst_json, payload, indent=2)
                    except OSError:
                        pass

    final_items = tuple(
        SelectedFrameItem(
            frame_index=item.frame_index,
            timestamp_sec=item.timestamp_sec,
            cluster_id=item.cluster_id,
            diversity_score=item.diversity_score,
            novelty_score=item.novelty_score,
            motion_score=item.motion_score,
            sharpness_score=item.sharpness_score,
            relevance_score=item.relevance_score,
            selection_role=item.selection_role,
            already_labeled=item.already_labeled,
            reason=item.reason,
            saved_path=saved_by_idx.get(item.frame_index),
        )
        for item in items
        if item.frame_index in saved_by_idx
    )

    manifest_file = out_root / EXTRACT_MANIFEST_FILENAME
    result = FrameSelectionResult(
        video_path=str(vpath),
        task=base_result.task if base_result is not None else "general",
        algo=base_result.algo if base_result is not None else "queue_export",
        feature_backend=(
            base_result.feature_backend if base_result is not None else "none"
        ),
        extract_images=True,
        output_dir=str(out_root),
        total_video_frames=int(total_frames),
        fps=float(fps),
        start_frame=min((item.frame_index for item in final_items), default=0),
        end_frame=max((item.frame_index for item in final_items), default=0),
        candidate_count=(
            base_result.candidate_count if base_result is not None else len(final_items)
        ),
        excluded_labeled_count=(
            base_result.excluded_labeled_count if base_result is not None else 0
        ),
        filtered_blur_count=(
            base_result.filtered_blur_count if base_result is not None else 0
        ),
        selected_items=final_items,
        saved_paths=tuple(saved_files),
        manifest_path=str(manifest_file),
        embedding_cache_path=(
            base_result.embedding_cache_path if base_result is not None else None
        ),
        embedding_cache_hit=(
            base_result.embedding_cache_hit if base_result is not None else False
        ),
        query_text=base_result.query_text if base_result is not None else None,
    )
    atomic_write_json(manifest_file, result.to_dict(), indent=2)
    return result


def build_selection_result_from_indices(
    video_path: Union[str, Path],
    frame_indices: Sequence[int],
    *,
    task: str = "general",
    algo: str = "custom",
    feature_backend: str = "none",
    annotation_dir: Optional[Union[str, Path]] = None,
    reasons_by_frame: Optional[Mapping[int, str]] = None,
    scores_by_frame: Optional[Mapping[int, float]] = None,
    role: str = "custom",
    persist_queue_manifest: bool = True,
) -> FrameSelectionResult:
    """Create a ``FrameSelectionResult`` from an explicit list of frames (e.g. search or outliers)."""
    vpath = Path(video_path).expanduser().resolve()
    total_frames = 0
    fps = 0.0
    if vpath.is_file():
        total_frames, fps, _lo, _hi = _resolve_video_bounds(
            vpath,
            start_seconds=None,
            end_seconds=None,
            start_frame=None,
            end_frame=None,
        )
    ann_dir = (
        Path(annotation_dir).expanduser().resolve()
        if annotation_dir is not None and str(annotation_dir).strip()
        else vpath.with_suffix("")
    )
    labeled = discover_labeled_frame_indices(ann_dir, require_shapes=True)
    clean_indices = sorted(
        {
            int(f)
            for f in frame_indices
            if int(f) >= 0 and (total_frames <= 0 or int(f) < total_frames)
        }
    )
    items = tuple(
        SelectedFrameItem(
            frame_index=idx,
            timestamp_sec=(float(idx) / fps) if fps > 0.0 else None,
            diversity_score=float((scores_by_frame or {}).get(idx, 0.0)),
            selection_role=role,
            already_labeled=(idx in labeled),
            reason=str((reasons_by_frame or {}).get(idx) or role),
        )
        for idx in clean_indices
    )
    manifest_path: Optional[str] = None
    if persist_queue_manifest and items:
        manifest_path = str(default_queue_manifest_path(vpath, ann_dir))

    res = FrameSelectionResult(
        video_path=str(vpath),
        task=get_task_profile(task).key,
        algo=str(algo or "custom"),
        feature_backend=str(feature_backend or "none"),
        extract_images=False,
        output_dir=None,
        total_video_frames=int(total_frames),
        fps=float(fps),
        start_frame=clean_indices[0] if clean_indices else 0,
        end_frame=clean_indices[-1] if clean_indices else 0,
        candidate_count=len(clean_indices),
        excluded_labeled_count=0,
        filtered_blur_count=0,
        selected_items=items,
        manifest_path=manifest_path,
    )
    if manifest_path is not None:
        try:
            save_frame_selection_queue(res, manifest_path)
        except OSError:
            pass
    return res


__all__ = [
    "EXTRACT_MANIFEST_FILENAME",
    "QUEUE_MANIFEST_FILENAME",
    "SUPPORTED_ALGOS",
    "SUPPORTED_FEATURE_BACKENDS",
    "SUPPORTED_TASKS",
    "TASK_PROFILES",
    "CachedEmbeddingIndexInfo",
    "FeatureBackendOption",
    "FrameSelectionConfig",
    "FrameSelectionResult",
    "SelectedFrameItem",
    "TaskProfile",
    "build_selection_result_from_indices",
    "default_queue_manifest_path",
    "find_cached_video_embedding_indices",
    "get_task_profile",
    "list_available_feature_backends",
    "load_frame_selection_queue",
    "materialize_selected_frames",
    "save_frame_selection_queue",
    "select_frames_for_labeling",
]
