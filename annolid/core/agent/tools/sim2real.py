from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool


class Sim2RealTool(FunctionTool):
    """Run Annolid's end-to-end Sim-to-Real workflow (video match -> 3D Mouse/Fly sim -> Colab train/iterate -> report)."""

    _ACTIONS = {
        "analyze",
        "generate",
        "iterate",
        "evaluate",
        "apply_recommendation",
        "report",
        "status",
    }

    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
        default_accelerator: str = "T4",
        default_colab_backend: str = "auto",
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir, allowed_read_roots
        )
        self._default_accelerator = (
            str(default_accelerator or "T4").strip().upper() or "T4"
        )
        self._default_colab_backend = (
            str(default_colab_backend or "auto").strip().lower() or "auto"
        )

    @property
    def name(self) -> str:
        return "sim2real_workflow"

    @property
    def description(self) -> str:
        return (
            "Analyze real example videos, generate environment-matched Mouse 3D or FlyBody 3D "
            "synthetic datasets, train/fine-tune and iterate models on Google Colab (or locally), "
            "evaluate on example videos, apply advisor recommendations, and write Markdown/JSON/HTML reports."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(self._ACTIONS)},
                "project_dir": {
                    "type": "string",
                    "description": "Local project directory storing profile, datasets, iterations, and reports.",
                },
                "example_videos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of example video paths or image directories (for analyze).",
                },
                "species": {
                    "type": "string",
                    "enum": ["auto", "mouse", "fly"],
                },
                "task": {
                    "type": "string",
                    "enum": ["yolo_seg", "yolo_pose", "yolo_detect"],
                },
                "export_level": {
                    "type": "string",
                    "enum": ["instances", "parts"],
                },
                "arena": {"type": "string"},
                "camera_view": {"type": "string"},
                "lighting": {"type": "string"},
                "coat_pair": {"type": "string"},
                "scenarios": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "frames_per_scenario": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 200,
                },
                "iterations": {"type": "integer", "minimum": 1, "maximum": 10},
                "train_backend": {"type": "string", "enum": ["colab", "local"]},
                "colab_backend": {
                    "type": "string",
                    "enum": ["auto", "cli", "api", "notebook"],
                },
                "accelerator": {"type": "string"},
                "epochs": {"type": "integer", "minimum": 1},
                "imgsz": {"type": "integer", "minimum": 64},
                "weights": {"type": "string"},
                "conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "keep_runtime": {"type": "boolean"},
                "allow_fallback": {"type": "boolean"},
            },
            "required": ["action", "project_dir"],
        }

    def _check_read(self, raw_path: str) -> str:
        return str(
            _resolve_read_path(
                raw_path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
        )

    def _check_write(self, raw_path: str) -> str:
        return str(_resolve_write_path(raw_path, allowed_dir=self._allowed_dir))

    async def execute(self, action: str, project_dir: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return "Error: Unsupported action. Use one of: " + ", ".join(
                sorted(self._ACTIONS)
            )
        try:
            payload = await asyncio.to_thread(
                self._execute_sync, action_name, project_dir, kwargs
            )
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Error: sim2real_workflow '{action_name}' failed: {exc}"

    def _execute_sync(
        self, action_name: str, project_dir: str, kwargs: dict[str, Any]
    ) -> Any:
        from annolid.sim2real.models import SimDatasetConfig
        from annolid.sim2real.service import Sim2RealWorkflowService

        proj_resolved = self._check_write(project_dir)
        svc = Sim2RealWorkflowService(proj_resolved)

        if action_name == "status":
            return svc.load_state().to_dict()

        if action_name == "analyze":
            raw_videos = kwargs.get("example_videos") or []
            if not raw_videos:
                raise ValueError("analyze requires `example_videos`.")
            videos = [self._check_read(str(v)) for v in raw_videos]
            profile, sim_cfg = svc.analyze_videos(
                videos,
                species_hint=str(kwargs.get("species") or "auto"),
                task=str(kwargs.get("task") or "yolo_seg"),
                export_level=str(kwargs.get("export_level") or "instances"),
                frames_per_scenario=int(kwargs.get("frames_per_scenario") or 24),
            )
            return {
                "project_dir": proj_resolved,
                "profile": profile.to_dict(),
                "sim_config": sim_cfg.to_dict(),
            }

        if action_name == "generate":
            state = svc.load_state()
            cfg = (
                SimDatasetConfig.from_dict(state.current_sim_config)
                if isinstance(state.current_sim_config, dict)
                else SimDatasetConfig()
            )
            for field_name in (
                "species",
                "task",
                "export_level",
                "arena",
                "camera_view",
                "lighting",
                "coat_pair",
            ):
                val = kwargs.get(field_name)
                if val and val != "auto":
                    setattr(cfg, field_name, str(val))
            if kwargs.get("scenarios"):
                cfg.scenarios = [str(s) for s in kwargs["scenarios"]]
            if kwargs.get("frames_per_scenario"):
                cfg.frames_per_scenario = int(kwargs["frames_per_scenario"])
            if kwargs.get("allow_fallback"):
                cfg.allow_fallback = True
            return svc.generate_dataset(cfg).to_dict()

        if action_name == "iterate":
            state = svc.load_state()
            cfg = (
                SimDatasetConfig.from_dict(state.current_sim_config)
                if isinstance(state.current_sim_config, dict)
                else SimDatasetConfig()
            )
            if kwargs.get("allow_fallback"):
                cfg.allow_fallback = True
            weights = kwargs.get("weights")
            if weights and Path(str(weights)).exists():
                weights = self._check_read(str(weights))
            records = svc.run_loop(
                iterations=int(kwargs.get("iterations") or 2),
                sim_config=cfg,
                train_backend=str(kwargs.get("train_backend") or "colab"),
                colab_backend=str(
                    kwargs.get("colab_backend") or self._default_colab_backend
                ),
                accelerator=str(kwargs.get("accelerator") or self._default_accelerator),
                epochs=int(kwargs.get("epochs") or 20),
                imgsz=int(kwargs.get("imgsz") or 640),
                weights=str(weights) if weights else None,
                conf=float(kwargs.get("conf") or 0.25),
                keep_runtime=bool(kwargs.get("keep_runtime", False)),
            )
            final_state = svc.load_state()
            return {
                "iterations": [r.to_dict() for r in records],
                "best_iteration": final_state.best_iteration,
                "best_score": final_state.best_score,
                "best_weights_path": final_state.best_weights_path,
                "report_paths": final_state.report_paths,
            }

        if action_name == "evaluate":
            weights = str(kwargs.get("weights") or "").strip()
            if not weights:
                raise ValueError("evaluate requires `weights`.")
            weights = self._check_read(weights)
            ev = svc.evaluate_existing_weights(
                weights_path=weights,
                train_backend=str(kwargs.get("train_backend") or "colab"),
                colab_backend=str(
                    kwargs.get("colab_backend") or self._default_colab_backend
                ),
                accelerator=str(kwargs.get("accelerator") or self._default_accelerator),
                conf=float(kwargs.get("conf") or 0.25),
            )
            return ev.to_dict()

        if action_name == "apply_recommendation":
            cfg, overrides = svc.apply_latest_recommendation()
            return {
                "sim_config": cfg.to_dict(),
                "suggested_train_overrides": overrides,
            }

        if action_name == "report":
            return svc.write_report()

        raise ValueError(f"Unsupported action: {action_name}")
