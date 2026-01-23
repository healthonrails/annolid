from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

from annolid.core.behavior.spec import BehaviorSpec
from annolid.core.types import BBoxGeometry, PolygonGeometry, RLEGeometry

from .behavior_engine import BehaviorEngine, BehaviorEngineConfig
from .frame_source import FrameSource
from .pipeline import AgentPipelineConfig
from .runner import AgentRunner
from .track_store import TrackState, TrackStore
from .tools.base import FrameBatch, Instances, Tool, ToolContext, ToolError
from .tools.detection import DetectionResult
from .tools.embedding import EmbeddingResult
from .tools.llm import CaptionResult
from .tools.tracking import TrackingResult


class AnnolidAgent:
    """Orchestrator placeholder for the unified agent pipeline."""

    def __init__(
        self,
        *,
        runner: Optional[AgentRunner] = None,
        tools: Optional[Sequence[Tool[Any, Any]]] = None,
        config: Optional[AgentPipelineConfig] = None,
        track_store: Optional[TrackStore] = None,
        behavior_engine: Optional[BehaviorEngine] = None,
    ) -> None:
        self._runner = runner or AgentRunner()
        self._tools = list(tools or [])
        self._config = config or AgentPipelineConfig()
        self._track_store = track_store
        self._behavior_engine = behavior_engine

    @property
    def config(self) -> AgentPipelineConfig:
        return self._config

    def iter_records(
        self,
        *,
        video_path: Path,
        schema: BehaviorSpec,
        agent_meta: Dict[str, Any],
        ctx: Optional[ToolContext] = None,
    ) -> Iterator[Dict[str, Any]]:
        if self._tools:
            yield from self._iter_records_with_tools(
                video_path=Path(video_path),
                schema=schema,
                agent_meta=agent_meta,
                ctx=ctx,
            )
            return

        yield from self._runner.iter_records(
            video_path=Path(video_path),
            schema=schema,
            agent_meta=agent_meta,
        )

    def _iter_records_with_tools(
        self,
        *,
        video_path: Path,
        schema: BehaviorSpec,
        agent_meta: Dict[str, Any],
        ctx: Optional[ToolContext] = None,
    ) -> Iterator[Dict[str, Any]]:
        _ = schema
        _ = agent_meta
        source = FrameSource(
            video_path=video_path,
            stride=self._config.stride,
            target_fps=self._config.target_fps,
            random_count=self._config.random_count,
            random_seed=self._config.random_seed,
            random_replace=self._config.random_replace,
            random_include_ends=self._config.random_include_ends,
        )
        tool_ctx = ctx
        if tool_ctx is None:
            tool_ctx = ToolContext(
                video_path=video_path,
                results_dir=Path("."),
                run_id="agent",
            )

        track_store = self._track_store or TrackStore()
        behavior_engine = self._behavior_engine or self._build_behavior_engine(schema)

        for batch in source.iter_batches(batch_size=1):
            if tool_ctx.cancelled():
                break
            frame = batch.frames[0]
            if frame.image_rgb is None:
                continue
            height, width = int(frame.image_rgb.shape[0]), int(frame.image_rgb.shape[1])
            try:
                outputs = self._run_pipeline(tool_ctx, batch)
                outputs = self._post_process_outputs(outputs, track_store)
            except ToolError as exc:
                if self._config.fail_fast:
                    raise
                tool_ctx.logger.warning("Agent tool error: %s", exc)
                if self._config.skip_on_error:
                    continue
                outputs = []
            except Exception as exc:
                if self._config.fail_fast:
                    raise
                tool_ctx.logger.exception("Agent pipeline error: %s", exc)
                if self._config.skip_on_error:
                    continue
                outputs = []

            behavior_update = behavior_engine.update(
                int(frame.ref.frame_index),
                track_store.active_tracks(),
            )
            record = {
                "version": "AnnolidAgentPipeline.1",
                "video_name": video_path.name,
                "frame_index": int(frame.ref.frame_index),
                "imagePath": str(frame.image_path) if frame.image_path else "",
                "imageHeight": height,
                "imageWidth": width,
                "flags": {},
                "shapes": self._shapes_from_outputs(outputs),
                "otherData": self._other_data_from_outputs(
                    outputs,
                    track_store.active_tracks(),
                    behavior_update.active,
                    behavior_update.completed,
                ),
            }
            if frame.ref.timestamp_sec is not None:
                record["timestamp_sec"] = float(frame.ref.timestamp_sec)
            yield record

    def _run_pipeline(self, ctx: ToolContext, batch: FrameBatch) -> Sequence[Any]:
        payload: Any = batch
        outputs: list[Any] = []
        for tool in self._tools:
            tool_input = payload
            if isinstance(payload, DetectionResult):
                tool_input = payload.frames
            payload = tool.run(ctx, tool_input)
            outputs.append(payload)
        return outputs

    def _post_process_outputs(
        self,
        outputs: Sequence[Any],
        track_store: TrackStore,
    ) -> list[Any]:
        updated: list[Any] = []
        for output in outputs:
            if isinstance(output, DetectionResult):
                frames = [track_store.update(frame) for frame in output.frames]
                updated.append(DetectionResult(frames=frames))
            else:
                updated.append(output)
        return updated

    def _shapes_from_outputs(self, outputs: Sequence[Any]) -> list[Dict[str, Any]]:
        shapes: list[Dict[str, Any]] = []
        for output in outputs:
            if isinstance(output, DetectionResult):
                for frame_instances in output.frames:
                    shapes.extend(self._instances_to_shapes(frame_instances))
        return shapes

    def _instances_to_shapes(self, instances: Instances) -> list[Dict[str, Any]]:
        shapes: list[Dict[str, Any]] = []
        for inst in instances.instances:
            geom = inst.geometry
            if isinstance(geom, BBoxGeometry):
                x1, y1, x2, y2 = geom.xyxy
                shapes.append(
                    {
                        "label": inst.label or "instance",
                        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                        "shape_type": "rectangle",
                        "flags": {"track_id": inst.track_id} if inst.track_id else {},
                        "score": float(inst.score) if inst.score is not None else None,
                    }
                )
            elif isinstance(geom, PolygonGeometry):
                shapes.append(
                    {
                        "label": inst.label or "instance",
                        "points": [[float(x), float(y)] for x, y in geom.points],
                        "shape_type": "polygon",
                        "flags": {"track_id": inst.track_id} if inst.track_id else {},
                        "score": float(inst.score) if inst.score is not None else None,
                    }
                )
            elif isinstance(geom, RLEGeometry):
                shapes.append(
                    {
                        "label": inst.label or "instance",
                        "points": [],
                        "shape_type": "mask",
                        "mask": geom.counts,
                        "flags": {"track_id": inst.track_id} if inst.track_id else {},
                        "score": float(inst.score) if inst.score is not None else None,
                    }
                )
        return shapes

    def _other_data_from_outputs(
        self,
        outputs: Sequence[Any],
        tracks: Sequence[TrackState],
        active_events: Sequence[Any],
        completed_events: Sequence[Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tracks_state": [self._track_state_to_dict(track) for track in tracks],
            "behavior_active": [event.to_dict() for event in active_events],
            "behavior_completed": [event.to_dict() for event in completed_events],
        }
        for output in outputs:
            if isinstance(output, CaptionResult):
                payload["captions"] = dict(output.frames)
            elif isinstance(output, TrackingResult):
                payload["tracks"] = [track.to_dict() for track in output.tracks]
            elif isinstance(output, EmbeddingResult):
                payload["embeddings"] = [
                    {"frame": frame.to_dict(), "vector": list(vec)}
                    for frame, vec in zip(output.frames, output.embeddings)
                ]
        return payload

    def _track_state_to_dict(self, track: TrackState) -> Dict[str, object]:
        return {
            "track_id": track.track_id,
            "label": track.label,
            "last_frame": track.last_frame,
            "bbox_xyxy": list(track.last_bbox),
            "hits": track.hits,
            "misses": track.misses,
        }

    def _build_behavior_engine(self, schema: BehaviorSpec) -> BehaviorEngine:
        params = dict(self._config.behavior_params or {})
        config_fields = {field.name for field in fields(BehaviorEngineConfig)}
        filtered = {k: v for k, v in params.items() if k in config_fields}
        config = BehaviorEngineConfig(**filtered)
        allowed = [behavior.code for behavior in schema.behaviors]
        return BehaviorEngine(config=config, allowed_codes=allowed)
