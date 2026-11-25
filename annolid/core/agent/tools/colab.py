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


class GoogleColabTool(FunctionTool):
    """Manage Google Colab authentication, runtimes, and Annolid remote train/predict jobs."""

    _ACTIONS = {
        "auth_status",
        "doctor",
        "list_specs",
        "list_sessions",
        "list_jobs",
        "list_templates",
        "stop_session",
        "generate_notebook",
        "train",
        "predict",
        "pull_results",
    }

    def __init__(
        self,
        *,
        credentials_file: str = "~/.annolid/agent/google_oauth_credentials.json",
        token_file: str = "~/.annolid/agent/google_oauth_token.json",
        allow_interactive_auth: bool = False,
        default_backend: str = "auto",
        default_accelerator: str = "T4",
        default_drive_folder: str = "Annolid/colab",
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._credentials_file = str(credentials_file or "").strip()
        self._token_file = str(token_file or "").strip()
        self._allow_interactive_auth = bool(allow_interactive_auth)
        self._default_backend = str(default_backend or "auto").strip().lower() or "auto"
        self._default_accelerator = (
            str(default_accelerator or "T4").strip().upper() or "T4"
        )
        self._default_drive_folder = (
            str(default_drive_folder or "Annolid/colab").strip() or "Annolid/colab"
        )
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir, allowed_read_roots
        )

    @property
    def name(self) -> str:
        return "google_colab"

    @property
    def description(self) -> str:
        return (
            "Authenticate status, inspect Colab runtimes/templates, generate Colab notebooks, "
            "launch Annolid model training or inference on Google Colab GPUs/TPUs, "
            "and pull finished outputs from Google Drive."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(self._ACTIONS)},
                "model": {
                    "type": "string",
                    "description": "Model plugin name (yolo, yolo_labelme, dino_kpseg, maskrcnn_detectron2, keypoint_rcnn).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["train", "predict"],
                    "description": "Notebook mode for generate_notebook.",
                },
                "dataset_path": {
                    "type": "string",
                    "description": "Local dataset YAML, folder, or .zip for training.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Local video/image/folder for inference.",
                },
                "weights": {"type": "string"},
                "epochs": {"type": "integer", "minimum": 1},
                "batch": {"type": "integer", "minimum": 1},
                "imgsz": {"type": "integer", "minimum": 32},
                "conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "accelerator": {
                    "type": "string",
                    "description": "GPU/TPU accelerator (T4, L4, A100, H100, CPU, TPU_V5E1, TPU_V6E1).",
                },
                "backend": {
                    "type": "string",
                    "enum": ["auto", "cli", "api", "notebook"],
                },
                "session": {"type": "string"},
                "job_id": {"type": "string"},
                "output_dir": {"type": "string"},
                "drive_folder": {"type": "string"},
                "upload_to_drive": {"type": "boolean"},
                "save_csv": {"type": "boolean"},
                "save_video": {"type": "boolean"},
                "keep_runtime": {"type": "boolean"},
                "live": {
                    "type": "boolean",
                    "description": "Enable live read-only API/Drive/CLI probes for doctor action.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["action"],
        }

    def _service(self) -> Any:
        from annolid.colab.auth import ColabAuthManager
        from annolid.colab.service import ColabWorkflowService

        auth = ColabAuthManager(
            credentials_file=self._credentials_file,
            token_file=self._token_file,
            allow_interactive_auth=self._allow_interactive_auth,
        )
        return ColabWorkflowService(
            auth_manager=auth,
            allowed_read_roots=self._allowed_read_roots,
        )

    def _check_read_path(self, raw_path: str) -> str:
        resolved = _resolve_read_path(
            raw_path,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )
        return str(resolved)

    def _check_write_path(self, raw_path: str | None) -> str | None:
        if not raw_path or not str(raw_path).strip():
            return None
        resolved = _resolve_write_path(str(raw_path), allowed_dir=self._allowed_dir)
        return str(resolved)

    async def execute(self, action: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return "Error: Unsupported action. Use one of: " + ", ".join(
                sorted(self._ACTIONS)
            )
        try:
            payload = await asyncio.to_thread(self._execute_sync, action_name, kwargs)
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Error: Google Colab action '{action_name}' failed: {exc}"

    def _execute_sync(self, action_name: str, kwargs: dict[str, Any]) -> Any:
        from annolid.colab.models import ColabPredictRequest, ColabTrainRequest

        svc = self._service()
        if action_name == "auth_status":
            return svc.auth_status()
        if action_name == "doctor":
            return svc.run_diagnostics(live=bool(kwargs.get("live", False)))
        if action_name == "list_templates":
            return {"templates": svc.list_templates()}
        if action_name == "list_jobs":
            limit = int(kwargs.get("limit", 20) or 20)
            return {"jobs": [j.to_dict() for j in svc.list_jobs(limit=limit)]}
        if action_name == "list_specs":
            return {"specs": [s.to_dict() for s in svc.list_runtime_specs()]}
        if action_name == "list_sessions":
            backend = str(kwargs.get("backend") or self._default_backend)
            return {
                "sessions": [
                    s.to_public_dict() for s in svc.list_sessions(backend=backend)
                ]
            }
        if action_name == "stop_session":
            session = str(kwargs.get("session") or "").strip()
            if not session:
                raise ValueError("stop_session requires `session`.")
            backend = str(kwargs.get("backend") or self._default_backend)
            svc.stop_session(session, backend=backend)
            return {"ok": True, "stopped": session}
        if action_name == "pull_results":
            job_id = str(kwargs.get("job_id") or "").strip()
            if not job_id:
                raise ValueError("pull_results requires `job_id`.")
            out_dir = self._check_write_path(kwargs.get("output_dir"))
            rec = svc.pull_results(job_id, output_dir=out_dir)
            return rec.to_dict()

        if action_name == "train" or (
            action_name == "generate_notebook"
            and str(kwargs.get("mode") or "train").lower() == "train"
        ):
            dataset_raw = str(kwargs.get("dataset_path") or "").strip()
            if not dataset_raw:
                raise ValueError(f"{action_name} requires `dataset_path`.")
            dataset_path = self._check_read_path(dataset_raw)
            weights_raw = str(kwargs.get("weights") or "").strip() or None
            if weights_raw and Path(weights_raw).expanduser().is_file():
                weights_raw = self._check_read_path(weights_raw)
            out_dir = self._check_write_path(kwargs.get("output_dir"))
            req = ColabTrainRequest(
                model=str(kwargs.get("model") or "yolo"),
                dataset_path=dataset_path,
                weights=weights_raw,
                epochs=kwargs.get("epochs"),
                batch=kwargs.get("batch"),
                imgsz=kwargs.get("imgsz"),
                accelerator=str(kwargs.get("accelerator") or self._default_accelerator),
                backend=(
                    "notebook"
                    if action_name == "generate_notebook"
                    else str(kwargs.get("backend") or self._default_backend)
                ),
                session_name=kwargs.get("session"),
                keep_runtime=bool(kwargs.get("keep_runtime", False)),
                output_dir=out_dir,
                drive_folder=str(
                    kwargs.get("drive_folder") or self._default_drive_folder
                ),
                upload_to_drive=bool(kwargs.get("upload_to_drive", True)),
            )
            return svc.run_train(req).to_dict()

        if action_name == "predict" or (
            action_name == "generate_notebook"
            and str(kwargs.get("mode") or "").lower() == "predict"
        ):
            source_raw = str(kwargs.get("source_path") or "").strip()
            if not source_raw:
                raise ValueError(f"{action_name} requires `source_path`.")
            source_path = self._check_read_path(source_raw)
            weights_raw = str(kwargs.get("weights") or "").strip() or None
            if weights_raw and Path(weights_raw).expanduser().is_file():
                weights_raw = self._check_read_path(weights_raw)
            out_dir = self._check_write_path(kwargs.get("output_dir"))
            req_p = ColabPredictRequest(
                model=str(kwargs.get("model") or "yolo_labelme"),
                source_path=source_path,
                weights=weights_raw,
                conf=kwargs.get("conf"),
                imgsz=kwargs.get("imgsz"),
                batch=kwargs.get("batch"),
                save_csv=bool(kwargs.get("save_csv", True)),
                save_video=bool(kwargs.get("save_video", False)),
                accelerator=str(kwargs.get("accelerator") or self._default_accelerator),
                backend=(
                    "notebook"
                    if action_name == "generate_notebook"
                    else str(kwargs.get("backend") or self._default_backend)
                ),
                session_name=kwargs.get("session"),
                keep_runtime=bool(kwargs.get("keep_runtime", False)),
                output_dir=out_dir,
                drive_folder=str(
                    kwargs.get("drive_folder") or self._default_drive_folder
                ),
                upload_to_drive=bool(kwargs.get("upload_to_drive", True)),
            )
            return svc.run_predict(req_p).to_dict()

        raise ValueError(f"Unhandled action: {action_name}")


__all__ = ["GoogleColabTool"]
