from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from annolid.datasets.coco import build_coco_spec_from_dataset_path
from annolid.engine.registry import get_model, list_models, load_builtin_models

from .annolid_run import AnnolidRunTool
from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool
from .shell_sessions import get_shell_session_manager

_MODEL_HINTS: dict[str, dict[str, object]] = {
    "dino_kpseg": {
        "aliases": (
            "dino",
            "dino keypoint segmentation",
            "keypoint segmentation",
        ),
        "tasks": ("keypoint_segmentation", "pose"),
        "default_task": "keypoint_segmentation",
    },
    "yolo": {
        "aliases": (
            "yolo",
            "yolo pose",
            "yolo segmentation",
            "yolo detection",
        ),
        "tasks": ("pose", "segmentation", "detection"),
        "default_task": "segmentation",
    },
}

_YOLO_TASK_WEIGHTS = {
    "pose": "yolo11n-pose.pt",
    "segmentation": "yolo11n-seg.pt",
    "detection": "yolo11n.pt",
}


def _find_dataset_config_candidates(dataset_root: Path) -> dict[str, Path]:
    root = Path(dataset_root).expanduser().resolve()
    candidates: dict[str, Path] = {}
    labelme_spec = root / "labelme_spec.yaml"
    if labelme_spec.exists():
        candidates["labelme_spec"] = labelme_spec.resolve()
    coco_spec = root / "coco_spec.yaml"
    if coco_spec.exists():
        candidates["coco_spec"] = coco_spec.resolve()
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        candidates["data_yaml"] = data_yaml.resolve()
    for candidate in sorted(root.rglob("data.yaml")):
        resolved = candidate.resolve()
        if data_yaml.exists() and resolved == data_yaml.resolve():
            continue
        candidates.setdefault("nested_data_yaml", resolved)
        break
    return candidates


def _resolve_dataset_training_input(
    dataset_folder: str,
    *,
    model: str,
    task: str,
    allowed_dir: Path | None,
    allowed_read_roots: Sequence[str | Path] | None,
    write_dir: Path | None = None,
) -> tuple[str, str]:
    root = _resolve_read_path(
        dataset_folder,
        allowed_dir=allowed_dir,
        allowed_read_roots=allowed_read_roots,
    )
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a dataset folder, got: {root}")
    candidates = _find_dataset_config_candidates(root)
    model_name = str(model or "").strip().lower()

    if model_name == "dino_kpseg":
        if "labelme_spec" in candidates:
            return str(candidates["labelme_spec"]), "labelme_spec"
        if "coco_spec" in candidates:
            return str(candidates["coco_spec"]), "coco_spec"
        inferred_coco = build_coco_spec_from_dataset_path(root)
        if inferred_coco is not None:
            return (
                str(
                    _stage_inferred_coco_spec(
                        dataset_root=root,
                        payload=inferred_coco,
                        write_dir=write_dir or allowed_dir or Path.cwd().resolve(),
                    )
                ),
                "coco_spec",
            )
        if "data_yaml" in candidates:
            return str(candidates["data_yaml"]), "data_yaml"
        if "nested_data_yaml" in candidates:
            return str(candidates["nested_data_yaml"]), "data_yaml"
        raise ValueError(
            "No trainable dataset spec found in dataset_folder. Expected "
            "`labelme_spec.yaml`, `coco_spec.yaml`, or `data.yaml`. Use "
            "annolid_dataset_prepare first."
        )

    if model_name == "yolo":
        if "data_yaml" in candidates:
            return str(candidates["data_yaml"]), "data_yaml"
        if "nested_data_yaml" in candidates:
            return str(candidates["nested_data_yaml"]), "data_yaml"
        if "coco_spec" in candidates:
            raise ValueError(
                "dataset_folder contains a COCO spec but YOLO training requires "
                "`data.yaml`. Run annolid_dataset_prepare with mode=coco_to_yolo first."
            )
        raise ValueError(
            "No YOLO data.yaml found in dataset_folder. Use annolid_dataset_prepare "
            "with mode=yolo_from_labelme or mode=coco_to_yolo first."
        )

    if "labelme_spec" in candidates:
        return str(candidates["labelme_spec"]), "labelme_spec"
    if "coco_spec" in candidates:
        return str(candidates["coco_spec"]), "coco_spec"
    if "data_yaml" in candidates:
        return str(candidates["data_yaml"]), "data_yaml"
    if "nested_data_yaml" in candidates:
        return str(candidates["nested_data_yaml"]), "data_yaml"
    raise ValueError(
        "No trainable dataset config found in dataset_folder. Use annolid_dataset_prepare first."
    )


def _stage_inferred_coco_spec(
    *,
    dataset_root: Path,
    payload: dict[str, object],
    write_dir: Path,
) -> Path:
    digest = hashlib.sha1(str(dataset_root).encode("utf-8")).hexdigest()[:10]
    cache_dir = (
        Path(write_dir).expanduser().resolve()
        / ".annolid"
        / "agent_cache"
        / "datasets"
        / f"{dataset_root.name}_{digest}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec_path = cache_dir / "coco_spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return spec_path


def _resolve_training_python(working_dir: Path) -> str:
    candidates = [
        working_dir / ".venv" / "bin" / "python",
        Path.cwd() / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    executable = str(sys.executable or "").strip()
    return executable or "python3"


def _model_metadata(name: str) -> dict[str, object]:
    hints = dict(_MODEL_HINTS.get(name, {}))
    return {
        "aliases": list(hints.get("aliases", ())),
        "tasks": list(hints.get("tasks", ())),
        "default_task": hints.get("default_task"),
    }


def _build_train_parser(model_name: str) -> argparse.ArgumentParser:
    plugin = get_model(model_name)
    parser = argparse.ArgumentParser(add_help=False)
    plugin.add_train_args(parser)
    return parser


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    options: set[str] = set()
    for action in getattr(parser, "_actions", ()):
        for opt in getattr(action, "option_strings", ()):
            if opt:
                options.add(str(opt))
    return options


def _append_supported_option(
    parser: argparse.ArgumentParser,
    argv: list[str],
    candidates: Sequence[str],
    value: object,
) -> bool:
    if value is None:
        return False
    options = _option_strings(parser)
    for candidate in candidates:
        if candidate in options:
            argv.extend([candidate, str(value)])
            return True
    return False


def _append_supported_flag(
    parser: argparse.ArgumentParser,
    argv: list[str],
    candidates: Sequence[str],
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    options = _option_strings(parser)
    for candidate in candidates:
        if candidate in options:
            argv.append(candidate)
            return True
    return False


def _resolve_input_path(
    value: str,
    *,
    allowed_dir: Path | None,
    allowed_read_roots: Sequence[str | Path] | None,
    required: bool = True,
) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("A non-empty path is required.")
        return ""
    resolved = _resolve_read_path(
        text,
        allowed_dir=allowed_dir,
        allowed_read_roots=allowed_read_roots,
    )
    if required and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {text}")
    return str(resolved)


def _resolve_weights_arg(
    value: str,
    *,
    allowed_dir: Path | None,
    allowed_read_roots: Sequence[str | Path] | None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    looks_like_path = any(sep in text for sep in ("/", "\\")) or text.startswith(
        (".", "~")
    )
    if not looks_like_path:
        return text
    resolved = _resolve_read_path(
        text,
        allowed_dir=allowed_dir,
        allowed_read_roots=allowed_read_roots,
    )
    if not resolved.exists():
        raise FileNotFoundError(f"Weights path not found: {text}")
    return str(resolved)


class AnnolidTrainModelsTool(FunctionTool):
    @property
    def name(self) -> str:
        return "annolid_train_models"

    @property
    def description(self) -> str:
        return (
            "List trainable Annolid model families the bot can fine-tune, including "
            "common aliases and task hints such as DINO keypoint segmentation and "
            "YOLO pose."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        load_builtin_models()
        rows = []
        for model in list_models(load_builtins=False):
            if not model.supports_train:
                continue
            rows.append(
                {
                    "name": model.name,
                    "description": model.description,
                    "supports_train": bool(model.supports_train),
                    "supports_predict": bool(model.supports_predict),
                    **_model_metadata(model.name),
                }
            )
        return json.dumps({"ok": True, "models": rows}, ensure_ascii=False)


class AnnolidTrainHelpTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._runner = AnnolidRunTool(
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_train_help"

    @property
    def description(self) -> str:
        return (
            "Return model-specific Annolid training help so the bot can discover the "
            "right fine-tuning flags before launching a run."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string", "minLength": 1},
            },
            "required": ["model"],
        }

    async def execute(self, model: str, **kwargs: Any) -> str:
        del kwargs
        return await self._runner.execute(
            argv=["train", str(model).strip(), "--help-model"],
        )


class AnnolidTrainStartTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_train_start"

    @property
    def description(self) -> str:
        return (
            "Launch a model training or fine-tuning job through `annolid-run train` "
            "using structured parameters. Prefers the workspace `.venv` Python when "
            "available and can start long YOLO or DINO runs in the background."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string", "minLength": 1},
                "task": {
                    "type": "string",
                    "enum": [
                        "pose",
                        "segmentation",
                        "detection",
                        "keypoint_segmentation",
                    ],
                },
                "data": {"type": "string"},
                "dataset_folder": {"type": "string"},
                "run_config": {"type": "string"},
                "weights": {"type": "string"},
                "working_dir": {"type": "string"},
                "output_dir": {"type": "string"},
                "runs_root": {"type": "string"},
                "run_name": {"type": "string"},
                "device": {"type": "string"},
                "video_folder": {"type": "string"},
                "dataset_dir": {"type": "string"},
                "checkpoint_dir": {"type": "string"},
                "tensorboard_log_dir": {"type": "string"},
                "epochs": {"type": "integer", "minimum": 1},
                "batch": {"type": "integer", "minimum": 1},
                "batch_size": {"type": "integer", "minimum": 1},
                "imgsz": {"type": "integer", "minimum": 32},
                "short_side": {"type": "integer", "minimum": 32},
                "learning_rate": {"type": "number", "minimum": 0.0},
                "base_lr": {"type": "number", "minimum": 0.0},
                "validation_split": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "max_iterations": {"type": "integer", "minimum": 1},
                "num_workers": {"type": "integer", "minimum": 0},
                "checkpoint_period": {"type": "integer", "minimum": 1},
                "score_threshold": {"type": "number", "minimum": 0.0},
                "overlap_threshold": {"type": "number", "minimum": 0.0},
                "feature_backbone": {"type": "string"},
                "dinov3_model_name": {"type": "string"},
                "model_arch": {"type": "string"},
                "background": {"type": "boolean"},
                "timeout_s": {"type": "number", "minimum": 0.0, "maximum": 86400.0},
                "allow_mutation": {"type": "boolean"},
                "unfreeze_dinov3": {"type": "boolean"},
                "plots": {"type": "boolean"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["model"],
        }

    async def execute(
        self,
        model: str,
        task: str = "",
        data: str = "",
        dataset_folder: str = "",
        run_config: str = "",
        weights: str = "",
        working_dir: str = "",
        output_dir: str = "",
        runs_root: str = "",
        run_name: str = "",
        device: str = "",
        video_folder: str = "",
        dataset_dir: str = "",
        checkpoint_dir: str = "",
        tensorboard_log_dir: str = "",
        epochs: int | None = None,
        batch: int | None = None,
        batch_size: int | None = None,
        imgsz: int | None = None,
        short_side: int | None = None,
        learning_rate: float | None = None,
        base_lr: float | None = None,
        validation_split: float | None = None,
        max_iterations: int | None = None,
        num_workers: int | None = None,
        checkpoint_period: int | None = None,
        score_threshold: float | None = None,
        overlap_threshold: float | None = None,
        feature_backbone: str = "",
        dinov3_model_name: str = "",
        model_arch: str = "",
        background: bool = True,
        timeout_s: float = 0.0,
        allow_mutation: bool = False,
        unfreeze_dinov3: bool = False,
        plots: bool = False,
        extra_args: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        model_name = str(model or "").strip()
        if not allow_mutation:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Starting a training run modifies state. Retry with "
                        "allow_mutation=true only when that is intended."
                    ),
                    "model": model_name,
                },
                ensure_ascii=False,
            )

        try:
            plugin = get_model(model_name)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc), "model": model_name},
                ensure_ascii=False,
            )
        if not plugin.__class__.supports_train():
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Model {model_name!r} does not support training.",
                    "model": model_name,
                },
                ensure_ascii=False,
            )

        cwd = self._resolve_working_dir(working_dir)
        task_label = str(task or "").strip().lower()
        try:
            argv, applied_task = self._build_argv(
                model=model_name,
                task=task_label,
                data=data,
                dataset_folder=dataset_folder,
                run_config=run_config,
                weights=weights,
                output_dir=output_dir,
                runs_root=runs_root,
                run_name=run_name,
                device=device,
                video_folder=video_folder,
                dataset_dir=dataset_dir,
                checkpoint_dir=checkpoint_dir,
                tensorboard_log_dir=tensorboard_log_dir,
                epochs=epochs,
                batch=batch,
                batch_size=batch_size,
                imgsz=imgsz,
                short_side=short_side,
                learning_rate=learning_rate,
                base_lr=base_lr,
                validation_split=validation_split,
                max_iterations=max_iterations,
                num_workers=num_workers,
                checkpoint_period=checkpoint_period,
                score_threshold=score_threshold,
                overlap_threshold=overlap_threshold,
                feature_backbone=feature_backbone,
                dinov3_model_name=dinov3_model_name,
                model_arch=model_arch,
                unfreeze_dinov3=unfreeze_dinov3,
                plots=plots,
                extra_args=extra_args or [],
            )
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc), "model": model_name},
                ensure_ascii=False,
            )

        python_exec = _resolve_training_python(cwd)
        command = " ".join(
            [
                shlex.quote(python_exec),
                "-m",
                "annolid.engine.cli",
                *(shlex.quote(part) for part in argv),
            ]
        )

        manager = get_shell_session_manager()
        try:
            session = await manager.start(
                command=command,
                cwd=str(cwd),
                timeout_s=max(0.0, float(timeout_s)),
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "model": model_name,
                    "command": command,
                },
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {
            "ok": True,
            "model": model_name,
            "task": applied_task,
            "argv": argv,
            "command": command,
            "cwd": str(cwd),
            "session_id": session.session_id,
            "background": bool(background),
            "follow_up_tool": "exec_process",
        }
        if not background:
            while True:
                polled = await manager.poll(session.session_id, wait_ms=50)
                if not bool(polled.get("ok", False)) or not bool(polled.get("running")):
                    break
            log = await manager.log(session.session_id, tail_lines=400)
            polled = await manager.poll(session.session_id)
            payload.update(
                {
                    "status": polled.get("status"),
                    "return_code": polled.get("return_code"),
                    "output": log.get("text", ""),
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    def _resolve_working_dir(self, raw_path: str) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            return self._allowed_dir or Path.cwd().resolve()
        return _resolve_read_path(
            text,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )

    def _build_argv(
        self,
        *,
        model: str,
        task: str,
        data: str,
        dataset_folder: str,
        run_config: str,
        weights: str,
        output_dir: str,
        runs_root: str,
        run_name: str,
        device: str,
        video_folder: str,
        dataset_dir: str,
        checkpoint_dir: str,
        tensorboard_log_dir: str,
        epochs: int | None,
        batch: int | None,
        batch_size: int | None,
        imgsz: int | None,
        short_side: int | None,
        learning_rate: float | None,
        base_lr: float | None,
        validation_split: float | None,
        max_iterations: int | None,
        num_workers: int | None,
        checkpoint_period: int | None,
        score_threshold: float | None,
        overlap_threshold: float | None,
        feature_backbone: str,
        dinov3_model_name: str,
        model_arch: str,
        unfreeze_dinov3: bool,
        plots: bool,
        extra_args: Sequence[str],
    ) -> tuple[list[str], str]:
        argv = ["train", model]
        parser = _build_train_parser(model)
        options = _option_strings(parser)
        if (
            "--data" in options
            or "--dataset-dir" in options
            or "--video-folder" in options
        ) and not (
            str(data or "").strip()
            or str(dataset_folder or "").strip()
            or str(dataset_dir or "").strip()
            or str(video_folder or "").strip()
            or str(run_config or "").strip()
        ):
            raise ValueError(
                "Provide dataset input for training via `data`, `dataset_folder`, "
                "`dataset_dir`, `video_folder`, or `run_config`."
            )

        if str(run_config or "").strip():
            _append_supported_option(
                parser,
                argv,
                ("--run-config",),
                _resolve_input_path(
                    run_config,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                ),
            )
        resolved_data = ""
        resolved_data_kind = ""
        if str(data or "").strip():
            resolved_data = _resolve_input_path(
                data,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if resolved_data.endswith("coco_spec.yaml"):
                resolved_data_kind = "coco_spec"
        elif str(dataset_folder or "").strip():
            resolved_data, resolved_data_kind = _resolve_dataset_training_input(
                dataset_folder,
                model=model,
                task=task,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
                write_dir=self._allowed_dir or Path.cwd().resolve(),
            )
        if resolved_data:
            _append_supported_option(parser, argv, ("--data",), resolved_data)
            _append_supported_option(parser, argv, ("--dataset-dir",), resolved_data)
            _append_supported_option(parser, argv, ("--video-folder",), resolved_data)
        if model == "dino_kpseg" and resolved_data_kind == "coco_spec":
            _append_supported_option(parser, argv, ("--data-format",), "coco")
        if str(dataset_dir or "").strip():
            _append_supported_option(
                parser,
                argv,
                ("--dataset-dir",),
                _resolve_input_path(
                    dataset_dir,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                ),
            )
        if str(video_folder or "").strip():
            _append_supported_option(
                parser,
                argv,
                ("--video-folder",),
                _resolve_input_path(
                    video_folder,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                ),
            )

        applied_task = task or str(_model_metadata(model).get("default_task") or "")
        if model == "yolo":
            valid_tasks = set(_MODEL_HINTS["yolo"]["tasks"])
            if applied_task and applied_task not in valid_tasks:
                raise ValueError(
                    f"Unsupported YOLO training task {applied_task!r}. "
                    f"Expected one of {sorted(valid_tasks)}."
                )
            chosen_weights = str(weights or "").strip() or _YOLO_TASK_WEIGHTS.get(
                applied_task or "segmentation",
                _YOLO_TASK_WEIGHTS["segmentation"],
            )
            argv.extend(
                [
                    "--weights",
                    _resolve_weights_arg(
                        chosen_weights,
                        allowed_dir=self._allowed_dir,
                        allowed_read_roots=self._allowed_read_roots,
                    ),
                ]
            )
            if output_dir:
                argv.extend(
                    [
                        "--project",
                        str(
                            _resolve_write_path(
                                output_dir,
                                allowed_dir=self._allowed_dir,
                            )
                        ),
                    ]
                )
            if imgsz is not None:
                argv.extend(["--imgsz", str(int(imgsz))])
        elif model == "dino_kpseg":
            valid_tasks = set(_MODEL_HINTS["dino_kpseg"]["tasks"])
            if applied_task and applied_task not in valid_tasks:
                raise ValueError(
                    f"Unsupported DinoKPSEG task {applied_task!r}. "
                    f"Expected one of {sorted(valid_tasks)}."
                )
            if output_dir:
                argv.extend(
                    [
                        "--output",
                        str(
                            _resolve_write_path(
                                output_dir,
                                allowed_dir=self._allowed_dir,
                            )
                        ),
                    ]
                )
            if runs_root:
                argv.extend(
                    [
                        "--runs-root",
                        str(
                            _resolve_write_path(
                                runs_root,
                                allowed_dir=self._allowed_dir,
                            )
                        ),
                    ]
                )
            if run_name:
                argv.extend(["--run-name", str(run_name).strip()])
        resolved_output_dir = (
            str(_resolve_write_path(output_dir, allowed_dir=self._allowed_dir))
            if output_dir
            else ""
        )
        if resolved_output_dir:
            _append_supported_option(parser, argv, ("--output",), resolved_output_dir)
            _append_supported_option(
                parser, argv, ("--output-dir",), resolved_output_dir
            )
            _append_supported_option(parser, argv, ("--project",), resolved_output_dir)
            _append_supported_option(
                parser, argv, ("--checkpoint-dir",), resolved_output_dir
            )
        if checkpoint_dir:
            _append_supported_option(
                parser,
                argv,
                ("--checkpoint-dir",),
                str(_resolve_write_path(checkpoint_dir, allowed_dir=self._allowed_dir)),
            )
        if tensorboard_log_dir:
            _append_supported_option(
                parser,
                argv,
                ("--tensorboard-log-dir",),
                str(
                    _resolve_write_path(
                        tensorboard_log_dir,
                        allowed_dir=self._allowed_dir,
                    )
                ),
            )
        if runs_root:
            _append_supported_option(
                parser,
                argv,
                ("--runs-root",),
                str(_resolve_write_path(runs_root, allowed_dir=self._allowed_dir)),
            )
        if run_name:
            _append_supported_option(
                parser, argv, ("--run-name",), str(run_name).strip()
            )
        if device:
            _append_supported_option(parser, argv, ("--device",), str(device).strip())
        if epochs is not None:
            _append_supported_option(parser, argv, ("--epochs",), int(epochs))
        if batch is not None:
            _append_supported_option(parser, argv, ("--batch",), int(batch))
            _append_supported_option(parser, argv, ("--batch-size",), int(batch))
        if batch_size is not None:
            _append_supported_option(parser, argv, ("--batch-size",), int(batch_size))
        if imgsz is not None:
            _append_supported_option(parser, argv, ("--imgsz",), int(imgsz))
        if short_side is not None:
            _append_supported_option(parser, argv, ("--short-side",), int(short_side))
        if learning_rate is not None:
            _append_supported_option(
                parser, argv, ("--learning-rate", "--lr"), float(learning_rate)
            )
        if base_lr is not None:
            _append_supported_option(parser, argv, ("--base-lr",), float(base_lr))
        if validation_split is not None:
            _append_supported_option(
                parser, argv, ("--validation-split",), float(validation_split)
            )
        if max_iterations is not None:
            _append_supported_option(
                parser, argv, ("--max-iterations",), int(max_iterations)
            )
        if num_workers is not None:
            _append_supported_option(parser, argv, ("--num-workers",), int(num_workers))
        if checkpoint_period is not None:
            _append_supported_option(
                parser, argv, ("--checkpoint-period",), int(checkpoint_period)
            )
        if score_threshold is not None:
            _append_supported_option(
                parser, argv, ("--score-threshold",), float(score_threshold)
            )
        if overlap_threshold is not None:
            _append_supported_option(
                parser, argv, ("--overlap-threshold",), float(overlap_threshold)
            )
        if feature_backbone:
            _append_supported_option(
                parser, argv, ("--feature-backbone",), str(feature_backbone).strip()
            )
        if dinov3_model_name:
            _append_supported_option(
                parser, argv, ("--dinov3-model-name",), str(dinov3_model_name).strip()
            )
        if model_arch:
            _append_supported_option(
                parser, argv, ("--model-arch",), str(model_arch).strip()
            )
        _append_supported_flag(
            parser, argv, ("--unfreeze-dinov3",), bool(unfreeze_dinov3)
        )
        _append_supported_flag(parser, argv, ("--plots",), bool(plots))
        for item in extra_args:
            token = str(item or "").strip()
            if token:
                argv.append(token)
        return argv, applied_task


__all__ = [
    "AnnolidTrainHelpTool",
    "AnnolidTrainModelsTool",
    "AnnolidTrainStartTool",
]
