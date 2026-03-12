from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from annolid.yolo import resolve_weight_path
from annolid.yolo.ultralytics_cli import build_yolo_val_command, ensure_parent_dir

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool
from .shell_sessions import get_shell_session_manager
from .training import _resolve_training_python


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
        raise ValueError("Weights are required for evaluation.")
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


def _parse_overrides(items: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid override {text!r}; expected key=value")
        key, value = text.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


class AnnolidEvalStartTool(FunctionTool):
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
        return "annolid_eval_start"

    @property
    def description(self) -> str:
        return (
            "Launch a model evaluation job for supported Annolid model families. "
            "Currently supports DinoKPSEG evaluation, YOLO validation, and "
            "behavior-classifier evaluation, returning a managed background session "
            "for later polling."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_family": {
                    "type": "string",
                    "enum": ["dino_kpseg", "yolo", "behavior_classifier"],
                },
                "data": {"type": "string"},
                "weights": {"type": "string"},
                "working_dir": {"type": "string"},
                "split": {"type": "string"},
                "device": {"type": "string"},
                "background": {"type": "boolean"},
                "timeout_s": {"type": "number", "minimum": 0.0, "maximum": 86400.0},
                "allow_mutation": {"type": "boolean"},
                "out": {"type": "string"},
                "report_dir": {"type": "string"},
                "report_basename": {"type": "string"},
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "imgsz": {"type": "integer", "minimum": 32},
                "batch": {"type": "integer", "minimum": 1},
                "project": {"type": "string"},
                "run_name": {"type": "string"},
                "plots": {"type": "boolean"},
                "save_json": {"type": "boolean"},
                "workers": {"type": "integer", "minimum": 0},
                "data_format": {
                    "type": "string",
                    "enum": ["auto", "yolo", "labelme", "coco"],
                },
                "thresholds": {"type": "string"},
                "max_images": {"type": "integer", "minimum": 1},
                "per_keypoint": {"type": "boolean"},
                "paper_report": {"type": "boolean"},
                "auto_threshold": {"type": "boolean"},
                "auto_threshold_metric": {
                    "type": "string",
                    "enum": ["pck", "ap"],
                },
                "auto_threshold_pck_px": {"type": "number", "minimum": 0.0},
                "auto_threshold_grid": {"type": "string"},
                "auto_threshold_per_keypoint": {"type": "boolean"},
                "video_folder": {"type": "string"},
                "checkpoint_path": {"type": "string"},
                "feature_backbone": {"type": "string"},
                "dinov3_model_name": {"type": "string"},
                "feature_dim": {"type": "integer"},
                "transformer_dim": {"type": "integer", "minimum": 1},
                "val_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "random_seed": {"type": "integer"},
                "plot_dir": {"type": "string"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "override": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["model_family", "data", "weights"],
        }

    async def execute(
        self,
        model_family: str,
        data: str,
        weights: str,
        working_dir: str = "",
        split: str = "test",
        device: str = "",
        background: bool = True,
        timeout_s: float = 0.0,
        allow_mutation: bool = False,
        out: str = "",
        report_dir: str = "",
        report_basename: str = "eval_report",
        dataset_name: str = "",
        model_name: str = "",
        imgsz: int | None = None,
        batch: int | None = None,
        project: str = "",
        run_name: str = "",
        plots: bool = False,
        save_json: bool = False,
        workers: int | None = None,
        data_format: str = "auto",
        thresholds: str = "",
        max_images: int | None = None,
        per_keypoint: bool = False,
        paper_report: bool = False,
        auto_threshold: bool = False,
        auto_threshold_metric: str = "pck",
        auto_threshold_pck_px: float = 8.0,
        auto_threshold_grid: str = "",
        auto_threshold_per_keypoint: bool = False,
        video_folder: str = "",
        checkpoint_path: str = "",
        feature_backbone: str = "",
        dinov3_model_name: str = "",
        feature_dim: int | None = None,
        transformer_dim: int = 768,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        plot_dir: str = "",
        extra_args: list[str] | None = None,
        override: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if not allow_mutation:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Starting an evaluation run modifies state. Retry with "
                        "allow_mutation=true only when that is intended."
                    ),
                    "model_family": model_family,
                },
                ensure_ascii=False,
            )

        family = str(model_family or "").strip().lower()
        cwd = self._resolve_working_dir(working_dir)
        try:
            if family == "dino_kpseg":
                command, output_path = self._build_dino_command(
                    cwd=cwd,
                    data=data,
                    weights=weights,
                    split=split,
                    device=device,
                    out=out,
                    report_dir=report_dir,
                    report_basename=report_basename,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    data_format=data_format,
                    thresholds=thresholds,
                    max_images=max_images,
                    per_keypoint=per_keypoint,
                    paper_report=paper_report,
                    auto_threshold=auto_threshold,
                    auto_threshold_metric=auto_threshold_metric,
                    auto_threshold_pck_px=auto_threshold_pck_px,
                    auto_threshold_grid=auto_threshold_grid,
                    auto_threshold_per_keypoint=auto_threshold_per_keypoint,
                    extra_args=extra_args or [],
                )
            elif family == "yolo":
                command, output_path = self._build_yolo_command(
                    cwd=cwd,
                    data=data,
                    weights=weights,
                    split=split,
                    device=device,
                    project=project,
                    run_name=run_name,
                    imgsz=imgsz,
                    batch=batch,
                    plots=plots,
                    save_json=save_json,
                    workers=workers,
                    overrides=override or [],
                )
            elif family == "behavior_classifier":
                command, output_path = self._build_behavior_command(
                    cwd=cwd,
                    video_folder=(video_folder or data),
                    checkpoint_path=(checkpoint_path or weights),
                    batch=batch,
                    device=device,
                    out=out,
                    split=split,
                    feature_backbone=feature_backbone,
                    dinov3_model_name=dinov3_model_name,
                    feature_dim=feature_dim,
                    transformer_dim=transformer_dim,
                    val_ratio=val_ratio,
                    random_seed=random_seed,
                    plot_dir=plot_dir,
                    extra_args=extra_args or [],
                )
            else:
                raise ValueError(
                    "Unsupported model_family. Expected 'dino_kpseg', 'yolo', or 'behavior_classifier'."
                )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "model_family": family,
                },
                ensure_ascii=False,
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
                    "model_family": family,
                    "command": command,
                },
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {
            "ok": True,
            "model_family": family,
            "command": command,
            "cwd": str(cwd),
            "session_id": session.session_id,
            "background": bool(background),
            "expected_output_path": output_path,
            "follow_up_tools": ["exec_process", "annolid_eval_report"],
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

    def _build_dino_command(
        self,
        *,
        cwd: Path,
        data: str,
        weights: str,
        split: str,
        device: str,
        out: str,
        report_dir: str,
        report_basename: str,
        dataset_name: str,
        model_name: str,
        data_format: str,
        thresholds: str,
        max_images: int | None,
        per_keypoint: bool,
        paper_report: bool,
        auto_threshold: bool,
        auto_threshold_metric: str,
        auto_threshold_pck_px: float,
        auto_threshold_grid: str,
        auto_threshold_per_keypoint: bool,
        extra_args: Sequence[str],
    ) -> tuple[str, str]:
        python_exec = _resolve_training_python(cwd)
        argv = [
            "-m",
            "annolid.segmentation.dino_kpseg.eval",
            "--data",
            _resolve_input_path(
                data,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            ),
            "--weights",
            _resolve_weights_arg(
                weights,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            ),
            "--split",
            str(split or "test").strip() or "test",
            "--data-format",
            str(data_format or "auto").strip() or "auto",
        ]
        if device:
            argv.extend(["--device", str(device).strip()])
        if thresholds:
            argv.extend(["--thresholds", str(thresholds).strip()])
        if max_images is not None:
            argv.extend(["--max-images", str(int(max_images))])
        if per_keypoint:
            argv.append("--per-keypoint")
        if paper_report:
            argv.append("--paper-report")
        if dataset_name:
            argv.extend(["--dataset-name", str(dataset_name).strip()])
        if model_name:
            argv.extend(["--model-name", str(model_name).strip()])

        output_path = ""
        if out:
            output_path = str(_resolve_write_path(out, allowed_dir=self._allowed_dir))
            argv.extend(["--out", output_path])
        if report_dir:
            argv.extend(
                [
                    "--report-dir",
                    str(_resolve_write_path(report_dir, allowed_dir=self._allowed_dir)),
                    "--report-basename",
                    str(report_basename or "eval_report").strip() or "eval_report",
                ]
            )
        if auto_threshold:
            argv.append("--auto-threshold")
            argv.extend(["--auto-threshold-metric", str(auto_threshold_metric)])
            argv.extend(["--auto-threshold-pck-px", str(float(auto_threshold_pck_px))])
            if auto_threshold_grid:
                argv.extend(["--auto-threshold-grid", str(auto_threshold_grid).strip()])
            if auto_threshold_per_keypoint:
                argv.append("--auto-threshold-per-keypoint")
        for item in extra_args:
            token = str(item or "").strip()
            if token:
                argv.append(token)

        command = " ".join(
            [shlex.quote(python_exec), *(shlex.quote(part) for part in argv)]
        )
        return command, output_path

    def _build_yolo_command(
        self,
        *,
        cwd: Path,
        data: str,
        weights: str,
        split: str,
        device: str,
        project: str,
        run_name: str,
        imgsz: int | None,
        batch: int | None,
        plots: bool,
        save_json: bool,
        workers: int | None,
        overrides: Sequence[str],
    ) -> tuple[str, str]:
        weight_path = resolve_weight_path(
            _resolve_weights_arg(
                weights,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
        )
        model_arg = ensure_parent_dir(str(weight_path))
        project_dir = (
            str(_resolve_write_path(project, allowed_dir=self._allowed_dir))
            if project
            else None
        )
        cmd = build_yolo_val_command(
            model=str(model_arg),
            data=_resolve_input_path(
                data,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            ),
            imgsz=(int(imgsz) if imgsz is not None else None),
            batch=(int(batch) if batch is not None else None),
            device=(str(device).strip() if device else None),
            project=project_dir,
            name=(str(run_name).strip() if run_name else None),
            split=(str(split or "test").strip() or None),
            plots=(True if plots else None),
            save_json=(True if save_json else None),
            workers=(int(workers) if workers is not None else None),
            overrides=_parse_overrides(overrides),
        )
        command = " ".join(shlex.quote(part) for part in cmd)
        return command, (project_dir or str(cwd))

    def _build_behavior_command(
        self,
        *,
        cwd: Path,
        video_folder: str,
        checkpoint_path: str,
        batch: int | None,
        device: str,
        out: str,
        split: str,
        feature_backbone: str,
        dinov3_model_name: str,
        feature_dim: int | None,
        transformer_dim: int,
        val_ratio: float,
        random_seed: int,
        plot_dir: str,
        extra_args: Sequence[str],
    ) -> tuple[str, str]:
        python_exec = _resolve_training_python(cwd)
        output_path = (
            str(_resolve_write_path(out, allowed_dir=self._allowed_dir)) if out else ""
        )
        argv = [
            "-m",
            "annolid.behavior.eval",
            "--video-folder",
            _resolve_input_path(
                video_folder,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            ),
            "--checkpoint-path",
            _resolve_input_path(
                checkpoint_path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            ),
            "--split",
            str(split or "all").strip() or "all",
            "--val-ratio",
            str(float(val_ratio)),
            "--random-seed",
            str(int(random_seed)),
            "--transformer-dim",
            str(int(transformer_dim)),
        ]
        if batch is not None:
            argv.extend(["--batch-size", str(int(batch))])
        if device:
            argv.extend(["--device", str(device).strip()])
        if feature_backbone:
            argv.extend(["--feature-backbone", str(feature_backbone).strip()])
        if dinov3_model_name:
            argv.extend(["--dinov3-model-name", str(dinov3_model_name).strip()])
        if feature_dim is not None:
            argv.extend(["--feature-dim", str(int(feature_dim))])
        if output_path:
            argv.extend(["--out", output_path])
        if plot_dir:
            argv.extend(
                [
                    "--plot-dir",
                    str(_resolve_write_path(plot_dir, allowed_dir=self._allowed_dir)),
                ]
            )
        for item in extra_args:
            token = str(item or "").strip()
            if token:
                argv.append(token)
        command = " ".join(
            [shlex.quote(python_exec), *(shlex.quote(part) for part in argv)]
        )
        return command, output_path


__all__ = ["AnnolidEvalStartTool"]
