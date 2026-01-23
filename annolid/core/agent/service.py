from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from annolid.core.agent.runner import AgentRunConfig, AgentRunner
from annolid.core.behavior.spec import BehaviorSpec, load_behavior_spec
from annolid.core.models.base import RuntimeModel
from annolid.core.output.validate import validate_agent_record
from annolid.utils.annotation_store import AnnotationStore


@dataclass(frozen=True)
class AgentServiceResult:
    results_dir: Path
    ndjson_path: Path
    store_path: Path
    records_written: int
    total_frames: Optional[int]
    stopped: bool


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
) -> AgentServiceResult:
    resolved_video = Path(video_path).expanduser().resolve()
    results_dir_path = resolve_results_dir(resolved_video, results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    ndjson_path = results_dir_path / out_ndjson_name
    store_stub = results_dir_path / f"{results_dir_path.name}_000000000.json"
    store = AnnotationStore.for_frame_path(store_stub)

    runner = AgentRunner(
        vision_model=vision_model,
        llm_model=llm_model,
        config=config or AgentRunConfig(),
    )
    schema, schema_path = load_behavior_spec(
        path=behavior_spec_path,
        video_path=resolved_video,
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

    return AgentServiceResult(
        results_dir=results_dir_path,
        ndjson_path=ndjson_path,
        store_path=store.store_path,
        records_written=records_written,
        total_frames=total_frames,
        stopped=stopped,
    )
