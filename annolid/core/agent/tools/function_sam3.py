from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .common import _resolve_read_path, _resolve_write_path
from .function_base import FunctionTool


def _default_sam3_output_dir(
    *,
    video_path: Path,
    allowed_dir: Path | None,
) -> Path:
    if allowed_dir is not None:
        return (
            Path(allowed_dir).expanduser().resolve() / f"{video_path.stem}_sam3_agent"
        )
    return video_path.parent / f"{video_path.stem}_sam3_agent"


class Sam3AgentVideoTrackTool(FunctionTool):
    """Run SAM3 agent-seeded long-video tracking from a bot/tool call."""

    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "sam3_agent_video_track"

    @property
    def description(self) -> str:
        return (
            "Run SAM3 Agent on the first frame of sliding video windows, then "
            "propagate masks through each window with overlap-aware state carry-over."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_path": {"type": "string"},
                "agent_prompt": {"type": "string", "minLength": 1},
                "window_size": {"type": "integer", "minimum": 1},
                "stride": {"type": "integer", "minimum": 1},
                "output_dir": {"type": "string"},
                "summary_filename": {"type": "string"},
                "checkpoint_path": {"type": "string"},
                "propagation_direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "both"],
                },
                "device": {"type": "string"},
                "agent_det_thresh": {"type": "number", "minimum": 0.0},
                "score_threshold_detection": {"type": "number"},
                "new_det_thresh": {"type": "number"},
                "max_num_objects": {"type": "integer", "minimum": 1},
                "multiplex_count": {"type": "integer", "minimum": 1},
                "compile_model": {"type": "boolean"},
                "offload_video_to_cpu": {"type": "boolean"},
                "use_explicit_window_reseed": {"type": "boolean"},
                "boundary_mask_match_iou_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "allow_private_state_mutation": {"type": "boolean"},
                "llm_provider": {"type": "string"},
                "llm_model": {"type": "string"},
                "llm_profile": {"type": "string"},
                "max_generations": {"type": "integer", "minimum": 1},
                "debug": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["video_path", "agent_prompt"],
        }

    async def execute(
        self,
        video_path: str,
        agent_prompt: str,
        window_size: int = 5,
        stride: int | None = None,
        output_dir: str | None = None,
        summary_filename: str | None = None,
        checkpoint_path: str | None = None,
        propagation_direction: str = "forward",
        device: str | None = None,
        agent_det_thresh: float = 0.3,
        score_threshold_detection: float | None = None,
        new_det_thresh: float | None = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        compile_model: bool = False,
        offload_video_to_cpu: bool = True,
        use_explicit_window_reseed: bool = True,
        boundary_mask_match_iou_threshold: float = 0.2,
        allow_private_state_mutation: bool = False,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_profile: str | None = None,
        max_generations: int = 100,
        debug: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            resolved_video = _resolve_read_path(
                video_path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not resolved_video.exists():
                return json.dumps(
                    {"ok": False, "error": f"File not found: {video_path}"}
                )
            if not resolved_video.is_file():
                return json.dumps({"ok": False, "error": f"Not a file: {video_path}"})

            resolved_output_dir: Path | None
            if output_dir:
                resolved_output_dir = _resolve_write_path(
                    output_dir, allowed_dir=self._allowed_dir
                )
            else:
                resolved_output_dir = _default_sam3_output_dir(
                    video_path=resolved_video,
                    allowed_dir=self._allowed_dir,
                )

            planned_summary_name = str(
                summary_filename or f"{resolved_video.stem}_sam3_agent_tracking.json"
            )
            summary_path = resolved_output_dir / planned_summary_name

            if dry_run:
                return json.dumps(
                    {
                        "ok": True,
                        "dry_run": True,
                        "video_path": str(resolved_video),
                        "agent_prompt": str(agent_prompt),
                        "output_dir": str(resolved_output_dir),
                        "summary_path": str(summary_path),
                        "config": {
                            "window_size": int(window_size),
                            "stride": int(stride) if stride is not None else None,
                            "checkpoint_path": checkpoint_path,
                            "propagation_direction": propagation_direction,
                            "device": device,
                            "agent_det_thresh": float(agent_det_thresh),
                            "score_threshold_detection": score_threshold_detection,
                            "new_det_thresh": new_det_thresh,
                            "max_num_objects": int(max_num_objects),
                            "multiplex_count": int(multiplex_count),
                            "compile_model": bool(compile_model),
                            "offload_video_to_cpu": bool(offload_video_to_cpu),
                            "use_explicit_window_reseed": bool(
                                use_explicit_window_reseed
                            ),
                            "boundary_mask_match_iou_threshold": float(
                                boundary_mask_match_iou_threshold
                            ),
                            "allow_private_state_mutation": bool(
                                allow_private_state_mutation
                            ),
                            "llm_provider": llm_provider,
                            "llm_model": llm_model,
                            "llm_profile": llm_profile,
                            "max_generations": int(max_generations),
                            "debug": bool(debug),
                        },
                    }
                )

            if resolved_output_dir is not None:
                resolved_output_dir.mkdir(parents=True, exist_ok=True)

            from annolid.segmentation.SAM.sam3.adapter import (
                process_video_with_agent,
            )

            frames_processed, masks_written = process_video_with_agent(
                video_path=str(resolved_video),
                agent_prompt=str(agent_prompt),
                agent_det_thresh=float(agent_det_thresh),
                window_size=int(window_size),
                stride=int(stride) if stride is not None else None,
                output_dir=str(resolved_output_dir) if resolved_output_dir else None,
                checkpoint_path=checkpoint_path,
                propagation_direction=str(propagation_direction),
                device=device,
                score_threshold_detection=score_threshold_detection,
                new_det_thresh=new_det_thresh,
                max_num_objects=int(max_num_objects),
                multiplex_count=int(multiplex_count),
                compile_model=bool(compile_model),
                offload_video_to_cpu=bool(offload_video_to_cpu),
                use_explicit_window_reseed=bool(use_explicit_window_reseed),
                boundary_mask_match_iou_threshold=float(
                    boundary_mask_match_iou_threshold
                ),
                allow_private_state_mutation=bool(allow_private_state_mutation),
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_profile=llm_profile,
                max_generations=int(max_generations),
            )

            payload = {
                "ok": True,
                "dry_run": False,
                "video_path": str(resolved_video),
                "agent_prompt": str(agent_prompt),
                "output_dir": str(resolved_output_dir) if resolved_output_dir else None,
                "summary_path": str(summary_path),
                "frames_processed": int(frames_processed),
                "masks_written": int(masks_written),
                "config": {
                    "window_size": int(window_size),
                    "stride": int(stride) if stride is not None else None,
                    "checkpoint_path": checkpoint_path,
                    "propagation_direction": propagation_direction,
                    "device": device,
                    "agent_det_thresh": float(agent_det_thresh),
                    "score_threshold_detection": score_threshold_detection,
                    "new_det_thresh": new_det_thresh,
                    "max_num_objects": int(max_num_objects),
                    "multiplex_count": int(multiplex_count),
                    "compile_model": bool(compile_model),
                    "offload_video_to_cpu": bool(offload_video_to_cpu),
                    "use_explicit_window_reseed": bool(use_explicit_window_reseed),
                    "boundary_mask_match_iou_threshold": float(
                        boundary_mask_match_iou_threshold
                    ),
                    "allow_private_state_mutation": bool(allow_private_state_mutation),
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "llm_profile": llm_profile,
                    "max_generations": int(max_generations),
                    "debug": bool(debug),
                },
            }
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return json.dumps(payload)
        except PermissionError as exc:
            return json.dumps(
                {"ok": False, "error": str(exc), "video_path": video_path}
            )
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc), "video_path": video_path}
            )
