"""Service-layer entry points for search workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from annolid.services.embedding_search import (
    DEFAULT_SEARCH_BACKEND,
    BehaviorSearchBout,
    EmbeddingSearchMatch,
    FolderFrameIndex,
    ScoreSummary,
    VideoBehaviorSearchResult,
    VideoFrameEmbeddingIndex,
    build_embedding_backend,
    build_video_embedding_index,
    compute_adaptive_threshold,
    default_video_index_cache_path,
    group_matches_into_bouts,
    index_folder_frames,
    prune_embedding_cache_dir,
    recommended_similarity_threshold,
    resolve_default_embedding_cache_dir,
    run_embedding_search,
    score_video_behaviors,
    search_folder_frames,
    search_video_behavior,
    summarize_similarity_scores,
)
from annolid.services.literature_search import search_literature


def search_indexed_frames(*args: Any, **kwargs: Any):
    from annolid.agents.frame_search import search_frames

    return search_frames(*args, **kwargs)


def search_video_frames(*args: Any, **kwargs: Any):
    from annolid.agents.frame_search import search_video

    return search_video(*args, **kwargs)


def run_behavior_search_cli(
    *,
    video_path: str,
    query: Optional[str] = None,
    query_frame: Optional[int] = None,
    query_image: Optional[str] = None,
    behaviors: Optional[Sequence[str]] = None,
    backend: str = DEFAULT_SEARCH_BACKEND,
    model_id: Optional[str] = None,
    quantization: str = "q4",
    matryoshka_dim: Optional[int] = None,
    soft_tokens: Optional[int] = None,
    stride: int = 5,
    max_frames: Optional[int] = None,
    clip_frames: int = 1,
    batch_size: int = 4,
    threshold: Optional[float] = None,
    top_k: int = 50,
    overlay_shapes: bool = False,
    annotation_dir: Optional[str] = None,
    index_cache: Optional[str] = None,
    auto_disk_cache: bool = False,
    out_path: Optional[str] = None,
    emit_json: bool = False,
) -> tuple[dict[str, Any], int]:
    """Execute CLI frame/behavior embedding search and optionally write JSON."""
    backend_params: dict[str, Any] = {}
    if model_id:
        if backend in {"dinov3", "dino", "dino_v3"}:
            backend_params["model_name"] = model_id
        else:
            backend_params["model_id"] = model_id
    if quantization:
        backend_params["quantization"] = quantization
    if matryoshka_dim is not None:
        backend_params["matryoshka_dim"] = int(matryoshka_dim)
    if soft_tokens is not None:
        backend_params["image_max_soft_tokens"] = int(soft_tokens)

    behavior_list = [str(b).strip() for b in (behaviors or []) if str(b).strip()]
    if behavior_list:
        payload = score_video_behaviors(
            video_path,
            behavior_list,
            backend=backend,
            backend_params=backend_params,
            stride=stride,
            max_frames=max_frames,
            clip_frames=clip_frames,
            batch_size=batch_size,
            top_n=max(1, min(int(top_k), len(behavior_list))),
            min_similarity=float(threshold) if threshold is not None else 0.0,
            index_cache_path=index_cache,
            auto_disk_cache=bool(auto_disk_cache),
        )
    else:
        result = search_video_behavior(
            video_path,
            query_text=query,
            query_frame_index=query_frame,
            query_image=query_image,
            backend=backend,
            backend_params=backend_params,
            stride=stride,
            max_frames=max_frames,
            clip_frames=clip_frames,
            batch_size=batch_size,
            threshold=threshold,
            top_k=top_k,
            overlay_shapes=overlay_shapes,
            annotation_dir=annotation_dir,
            index_cache_path=index_cache,
            auto_disk_cache=bool(auto_disk_cache),
        )
        payload = result.to_dict()

    if out_path:
        target = Path(out_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _ = emit_json
    return payload, 0


__all__ = [
    "DEFAULT_SEARCH_BACKEND",
    "BehaviorSearchBout",
    "EmbeddingSearchMatch",
    "FolderFrameIndex",
    "ScoreSummary",
    "VideoBehaviorSearchResult",
    "VideoFrameEmbeddingIndex",
    "build_embedding_backend",
    "build_video_embedding_index",
    "compute_adaptive_threshold",
    "default_video_index_cache_path",
    "group_matches_into_bouts",
    "index_folder_frames",
    "prune_embedding_cache_dir",
    "recommended_similarity_threshold",
    "resolve_default_embedding_cache_dir",
    "run_behavior_search_cli",
    "run_embedding_search",
    "score_video_behaviors",
    "search_folder_frames",
    "search_indexed_frames",
    "search_literature",
    "search_video_behavior",
    "search_video_frames",
    "summarize_similarity_scores",
]
