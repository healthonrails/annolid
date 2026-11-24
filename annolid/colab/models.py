from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

ColabBackendKind = Literal["auto", "cli", "api", "notebook"]
ColabAuthMethod = Literal["auto", "personal", "oauth2", "adc", "colab-cli", "drive"]

VALID_BACKENDS: tuple[str, ...] = ("auto", "cli", "api", "notebook")
VALID_AUTH_METHODS: tuple[str, ...] = (
    "auto",
    "personal",
    "oauth2",
    "adc",
    "colab-cli",
    "drive",
)
SUPPORTED_ACCELERATORS: tuple[str, ...] = (
    "T4",
    "L4",
    "A100",
    "H100",
    "CPU",
    "TPU_V5E1",
    "TPU_V6E1",
)

TRAIN_MODEL_CHOICES: tuple[str, ...] = (
    "yolo",
    "dino_kpseg",
    "maskrcnn_detectron2",
    "keypoint_rcnn",
    "tcn_behavior",
    "polygon_classifier",
    "behavior_classifier",
)

PREDICT_MODEL_CHOICES: tuple[str, ...] = (
    "yolo_labelme",
    "yolo",
    "dino_kpseg",
    "maskrcnn_detectron2",
    "keypoint_rcnn",
    "tcn_behavior",
    "polygon_classifier",
    "behavior_classifier",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_job_id(prefix: str = "colab") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prefix)
    return f"{clean}_{stamp}_{suffix}"


def colab_home_dir() -> Path:
    override = str(os.environ.get("ANNOLID_COLAB_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".annolid" / "colab"


def normalize_accelerator(value: Optional[str]) -> str:
    raw = str(value or "T4").strip().upper().replace("-", "_")
    aliases = {
        "": "T4",
        "NONE": "CPU",
        "GPU": "T4",
        "GPU_T4": "T4",
        "NVIDIA_T4": "T4",
        "GPU_L4": "L4",
        "NVIDIA_L4": "L4",
        "GPU_A100": "A100",
        "NVIDIA_A100": "A100",
        "GPU_H100": "H100",
        "NVIDIA_H100": "H100",
        "V5E1": "TPU_V5E1",
        "TPU_V5E": "TPU_V5E1",
        "V6E1": "TPU_V6E1",
        "TPU_V6E": "TPU_V6E1",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in SUPPORTED_ACCELERATORS:
        return normalized
    return normalized


def accelerator_to_api_spec(
    accelerator: str, *, high_mem: bool = False
) -> dict[str, str]:
    acc = normalize_accelerator(accelerator)
    shape = "SHAPE_HIGHMEM" if high_mem else "SHAPE_STANDARD"
    if acc == "CPU":
        return {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": shape}
    if acc.startswith("TPU_"):
        tpu_name = acc.removeprefix("TPU_")
        return {"variant": "VARIANT_TPU", "accelerator": tpu_name, "shape": shape}
    return {"variant": "VARIANT_GPU", "accelerator": acc, "shape": shape}


@dataclass(frozen=True)
class ColabRuntimeSpec:
    variant: str
    accelerator: str
    shape: str = "SHAPE_STANDARD"
    eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ColabRuntimeInfo:
    name: str
    backend: str
    session_name: str = ""
    accelerator: str = ""
    variant: str = ""
    shape: str = ""
    state: str = "UNKNOWN"
    browser_url: str = ""
    proxy_url: str = ""
    proxy_token: str = ""
    expire_time: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize runtime metadata without leaking proxy tokens."""
        return {
            "name": self.name,
            "backend": self.backend,
            "session_name": self.session_name,
            "accelerator": self.accelerator,
            "variant": self.variant,
            "shape": self.shape,
            "state": self.state,
            "browser_url": self.browser_url,
            "proxy_url": self.proxy_url,
            "expire_time": self.expire_time,
        }


@dataclass(frozen=True)
class ColabTrainRequest:
    model: str
    dataset_path: str
    weights: Optional[str] = None
    epochs: Optional[int] = None
    max_iterations: Optional[int] = None
    batch: Optional[int] = None
    imgsz: Optional[int] = None
    preset: Optional[str] = None
    data_format: Optional[str] = None
    plots: bool = True
    resume: bool = False
    extra_args: tuple[str, ...] = ()
    accelerator: str = "T4"
    high_mem: bool = False
    backend: str = "auto"
    session_name: Optional[str] = None
    keep_runtime: bool = False
    output_dir: Optional[str] = None
    drive_folder: str = "Annolid/colab"
    upload_to_drive: bool = True
    open_browser: bool = False
    annolid_install_spec: str = "annolid"
    job_name: Optional[str] = None


@dataclass(frozen=True)
class ColabPredictRequest:
    model: str
    source_path: str
    weights: Optional[str] = None
    dataset_dir: Optional[str] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    imgsz: Optional[int] = None
    batch: Optional[int] = None
    save_csv: bool = True
    save_video: bool = False
    no_tracking: bool = False
    classes: Optional[str] = None
    extra_args: tuple[str, ...] = ()
    accelerator: str = "T4"
    high_mem: bool = False
    backend: str = "auto"
    session_name: Optional[str] = None
    keep_runtime: bool = False
    output_dir: Optional[str] = None
    drive_folder: str = "Annolid/colab"
    upload_to_drive: bool = True
    open_browser: bool = False
    annolid_install_spec: str = "annolid"
    job_name: Optional[str] = None


@dataclass
class ColabJobRecord:
    job_id: str
    kind: str  # "train" | "predict"
    model: str
    backend: str
    accelerator: str
    status: str  # "created" | "running" | "notebook_ready" | "completed" | "failed" | "cancelled"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    session_name: str = ""
    runtime_name: str = ""
    remote_command: list[str] = field(default_factory=list)
    notebook_path: str = ""
    local_staging_dir: str = ""
    local_output_dir: str = ""
    drive_folder: str = ""
    drive_folder_id: str = ""
    drive_notebook_id: str = ""
    colab_url: str = ""
    downloaded_files: list[str] = field(default_factory=list)
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColabJobRecord":
        payload = dict(data or {})
        return cls(
            job_id=str(payload.get("job_id") or ""),
            kind=str(payload.get("kind") or "train"),
            model=str(payload.get("model") or ""),
            backend=str(payload.get("backend") or "notebook"),
            accelerator=str(payload.get("accelerator") or "T4"),
            status=str(payload.get("status") or "created"),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            session_name=str(payload.get("session_name") or ""),
            runtime_name=str(payload.get("runtime_name") or ""),
            remote_command=[str(x) for x in list(payload.get("remote_command") or [])],
            notebook_path=str(payload.get("notebook_path") or ""),
            local_staging_dir=str(payload.get("local_staging_dir") or ""),
            local_output_dir=str(payload.get("local_output_dir") or ""),
            drive_folder=str(payload.get("drive_folder") or ""),
            drive_folder_id=str(payload.get("drive_folder_id") or ""),
            drive_notebook_id=str(payload.get("drive_notebook_id") or ""),
            colab_url=str(payload.get("colab_url") or ""),
            downloaded_files=[
                str(x) for x in list(payload.get("downloaded_files") or [])
            ],
            result_summary=dict(payload.get("result_summary") or {}),
            error=str(payload.get("error") or ""),
        )


class ColabJobStore:
    """Atomic JSON persistence for Colab job manifests under ~/.annolid/colab/jobs/."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir is not None else colab_home_dir()
        self._jobs_dir = self._root / "jobs"

    @property
    def jobs_dir(self) -> Path:
        return self._jobs_dir

    def save(self, record: ColabJobRecord) -> Path:
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        record.updated_at = utc_now_iso()
        target = self._jobs_dir / f"{record.job_id}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    def load(self, job_id: str) -> Optional[ColabJobRecord]:
        clean = str(job_id or "").strip()
        if not clean or "/" in clean or "\\" in clean or ".." in clean:
            return None
        target = self._jobs_dir / f"{clean}.json"
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return ColabJobRecord.from_dict(payload)

    def list_jobs(self, *, limit: int = 50) -> list[ColabJobRecord]:
        if not self._jobs_dir.is_dir():
            return []
        records: list[ColabJobRecord] = []
        for path in self._jobs_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    records.append(ColabJobRecord.from_dict(payload))
            except Exception:
                continue
        records.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
        return records[: max(1, int(limit))]
