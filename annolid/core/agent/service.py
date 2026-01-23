from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from annolid.core.agent.runner import AgentRunConfig, AgentRunner
from annolid.core.behavior.spec import BehaviorSpec, load_behavior_spec
from annolid.core.models.base import RuntimeModel
from annolid.core.output.validate import validate_agent_record
from annolid.core.agent.tools.artifacts import FileArtifactStore, content_hash
from annolid.utils.annotation_store import AnnotationStore


@dataclass(frozen=True)
class AgentServiceResult:
    results_dir: Path
    ndjson_path: Path
    store_path: Path
    records_written: int
    total_frames: Optional[int]
    stopped: bool
    cached: bool = False


ProgressCallback = Callable[[int, int, Optional[int]], None]


def resolve_results_dir(
    video_path: str | Path, results_dir: Optional[str | Path] = None
) -> Path:
    if results_dir is not None:
        return Path(results_dir).expanduser().resolve()
    return Path(video_path).expanduser().resolve().with_suffix("")


def _build_agent_meta(
    runner: AgentRunner,
    schema: BehaviorSpec,
    schema_path: Optional[Path],
    video_path: Path,
) -> dict[str, Any]:
    meta = runner._build_agent_meta(schema, schema_path, video_path)
    if runner._config.include_llm_summary and runner._llm_model is not None:
        meta["llm_summary"] = runner._compute_llm_summary(schema)
    return meta


def _store_record_from_agent(record: dict[str, Any]) -> dict[str, Any]:
    other = dict(record.get("otherData") or {})
    if "timestamp_sec" in record:
        other.setdefault("timestamp_sec", record.get("timestamp_sec"))
    return {
        "frame": record.get("frame_index"),
        "version": record.get("version") or "annolid",
        "flags": record.get("flags") or {},
        "shapes": record.get("shapes") or [],
        "imagePath": record.get("imagePath") or "",
        "imageHeight": record.get("imageHeight"),
        "imageWidth": record.get("imageWidth"),
        "otherData": other,
    }


def _build_cache_payload(
    *,
    video_path: Path,
    schema: BehaviorSpec,
    schema_path: Optional[Path],
    config: AgentRunConfig,
    out_ndjson_name: str,
    vision_model: Optional[RuntimeModel],
    llm_model: Optional[RuntimeModel],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_path": str(video_path),
        "behavior_spec_path": str(schema_path) if schema_path is not None else None,
        "behavior_spec": asdict(schema),
        "config": asdict(config),
        "ndjson_name": out_ndjson_name,
        "vision_model_id": getattr(vision_model, "model_id", None),
        "llm_model_id": getattr(llm_model, "model_id", None),
    }
    try:
        stat = video_path.stat()
        payload["video_size"] = stat.st_size
        payload["video_mtime"] = stat.st_mtime
    except Exception:
        payload["video_size"] = None
        payload["video_mtime"] = None
    return payload


def _count_ndjson_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def run_agent_to_results(
    *,
    video_path: str | Path,
    behavior_spec_path: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    out_ndjson_name: str = "agent.ndjson",
    vision_model: Optional[RuntimeModel] = None,
    llm_model: Optional[RuntimeModel] = None,
    config: Optional[AgentRunConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stop_event: Optional[Any] = None,
    reuse_cache: bool = True,
) -> AgentServiceResult:
    resolved_video = Path(video_path).expanduser().resolve()
    results_dir_path = resolve_results_dir(resolved_video, results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    ndjson_path = results_dir_path / out_ndjson_name
    store_stub = results_dir_path / f"{results_dir_path.name}_000000000.json"
    store = AnnotationStore.for_frame_path(store_stub)
    cache_store = FileArtifactStore(base_dir=results_dir_path, run_id="agent")
    cache_meta_path = cache_store.resolve("agent_cache.json", kind="cache")

    runner = AgentRunner(
        vision_model=vision_model,
        llm_model=llm_model,
        config=config or AgentRunConfig(),
    )
    schema, schema_path = load_behavior_spec(
        path=behavior_spec_path,
        video_path=resolved_video,
    )
    cache_payload = _build_cache_payload(
        video_path=resolved_video,
        schema=schema,
        schema_path=schema_path,
        config=runner._config,
        out_ndjson_name=out_ndjson_name,
        vision_model=vision_model,
        llm_model=llm_model,
    )
    cache_hash = content_hash(cache_payload)
    if reuse_cache and cache_store.should_reuse_cache(cache_meta_path, cache_hash):
        if ndjson_path.exists() and store.store_path.exists():
            cached_records = _count_ndjson_records(ndjson_path)
            cached_total = None
            try:
                cached_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                cached_total = cached_meta.get("total_frames")
            except Exception:
                cached_total = None
            return AgentServiceResult(
                results_dir=results_dir_path,
                ndjson_path=ndjson_path,
                store_path=store.store_path,
                records_written=cached_records,
                total_frames=cached_total,
                stopped=False,
                cached=True,
            )
    agent_meta = _build_agent_meta(runner, schema, schema_path, resolved_video)

    total_frames: Optional[int] = None
    try:
        from annolid.core.media.video import CV2Video

        video = CV2Video(resolved_video)
        total_frames = int(video.total_frames())
        video.release()
    except Exception:
        total_frames = None

    records_written = 0
    stopped = False
    iterator = runner.iter_records(
        video_path=resolved_video,
        schema=schema,
        agent_meta=agent_meta,
    )
    with ndjson_path.open("w", encoding="utf-8") as fh:
        for record in iterator:
            if stop_event is not None and getattr(stop_event, "is_set", None):
                if stop_event.is_set():
                    stopped = True
                    break

            validate_agent_record(record)
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")

            store.append_frame(_store_record_from_agent(record))

            records_written += 1
            if progress_callback is not None:
                progress_callback(
                    int(record["frame_index"]), records_written, total_frames
                )

    if stopped:
        try:
            iterator.close()
        except Exception:
            pass

    if not stopped:
        cache_store.write_meta(
            cache_meta_path,
            {
                "content_hash": cache_hash,
                "video_path": str(resolved_video),
                "behavior_spec_path": str(schema_path)
                if schema_path is not None
                else None,
                "results_dir": str(results_dir_path),
                "ndjson_path": str(ndjson_path),
                "store_path": str(store.store_path),
                "records_written": records_written,
                "total_frames": total_frames,
            },
        )

    return AgentServiceResult(
        results_dir=results_dir_path,
        ndjson_path=ndjson_path,
        store_path=store.store_path,
        records_written=records_written,
        total_frames=total_frames,
        stopped=stopped,
        cached=False,
    )
