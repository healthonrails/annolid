"""Live log parser and metric state tracker for Google Colab training & inference."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_NVIDIA_GPU_RE = re.compile(r"\|\s*\d+\s+(?P<gpu>[A-Za-z0-9 _.-]+?)\s+(?:On|Off)\s*\|")
_NVIDIA_MEM_RE = re.compile(r"(?P<used>\d+)MiB\s*/\s*(?P<total>\d+)MiB")
_COLAB_URL_RE = re.compile(r"(https://colab\.research\.google\.com/\S+)")

# Ultralytics YOLO training row, e.g.:
#   3/50  0.467G  1.754  3.534  3.752  1.485  0  41  320: 100% ... 9/9
_YOLO_EPOCH_RE = re.compile(
    r"^\s*(?P<epoch>\d+)/(?P<total>\d+)\s+"
    r"(?P<mem>[0-9.]+[GMK]?)\s+"
    r"(?P<nums>(?:[0-9.]+(?:\s+|$)){2,8})"
)

# Ultralytics validation summary row ("all <images> <instances> <P> <R> <mAP50> <mAP50-95> [Mask P R mAP50 mAP50-95]")
_YOLO_VAL_ALL_RE = re.compile(
    r"^\s*all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+(?P<metrics>[0-9.eE+-]+(?:\s+[0-9.eE+-]+){3,7})\s*$"
)

# Generic PyTorch / DINO KPSEG epoch lines, e.g.:
#   Epoch 12/150 | train_loss=0.412 | val_loss=0.388
_GENERIC_EPOCH_RE = re.compile(
    r"\bepoch\s*[:#]?\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\b",
    re.IGNORECASE,
)

# Detectron2 / Torchvision iteration lines, e.g.:
#   iter: 200/3000  total_loss: 0.512
_GENERIC_ITER_RE = re.compile(
    r"\b(?:iter|iteration|step)\s*[:#]?\s*(?P<step>\d+)\s*/\s*(?P<total>\d+)\b",
    re.IGNORECASE,
)

_KV_FLOAT_RE = re.compile(
    r"\b(?P<key>train_loss|val_loss|total_loss|loss|box_loss|seg_loss|cls_loss|dfl_loss|map50|mAP50)\s*[:=]\s*(?P<val>[0-9.eE+-]+)",
    re.IGNORECASE,
)

# yolo_labelme progress output, e.g. "Done#10" or "frame 12/100"
_LABELME_DONE_RE = re.compile(r"\bDone#(?P<frames>\d+)\b")
_FRAME_PROGRESS_RE = re.compile(
    r"\bframe\s*[:#]?\s*(?P<frame>\d+)\s*/\s*(?P<total>\d+)\b",
    re.IGNORECASE,
)


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", str(text or ""))


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


@dataclass
class ColabMetricPoint:
    """Single epoch or iteration snapshot for live Colab monitoring."""

    step: int
    total_steps: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    box_loss: Optional[float] = None
    seg_loss: Optional[float] = None
    cls_loss: Optional[float] = None
    dfl_loss: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    mask_map50: Optional[float] = None
    mask_map50_95: Optional[float] = None
    pose_map50: Optional[float] = None
    pose_map50_95: Optional[float] = None
    gpu_mem: str = ""
    label: str = ""


@dataclass
class ColabLiveMonitorState:
    """Structured state rendered by the Colab Studio Live Monitor tab."""

    job_id: str = ""
    kind: str = ""
    model: str = ""
    session_name: str = ""
    stage: str = "idle"
    gpu_name: str = ""
    gpu_mem_total_mib: Optional[int] = None
    gpu_mem_used: str = ""
    colab_url: str = ""
    current_step: int = 0
    total_steps: Optional[int] = None
    step_unit: str = "epoch"
    latest_metrics: Optional[ColabMetricPoint] = None
    history: list[ColabMetricPoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifact_count: int = 0
    elapsed_remote_s: Optional[float] = None


class ColabLogMonitorParser:
    """Incremental line parser extracting live stage, GPU, and metrics from Colab logs."""

    def __init__(self, state: Optional[ColabLiveMonitorState] = None) -> None:
        self.state = state or ColabLiveMonitorState()
        self._pending_gpu_name = ""

    def reset(
        self,
        *,
        job_id: str = "",
        kind: str = "",
        model: str = "",
        session_name: str = "",
    ) -> None:
        self.state = ColabLiveMonitorState(
            job_id=job_id,
            kind=kind,
            model=model,
            session_name=session_name,
            stage="starting" if kind else "idle",
            step_unit="frame" if kind == "predict" else "epoch",
        )
        self._pending_gpu_name = ""

    def _upsert_point(self, point: ColabMetricPoint) -> None:
        for idx, existing in enumerate(self.state.history):
            if existing.step == point.step:
                # Preserve validation metrics if updating the same epoch
                if point.map50 is None:
                    point.map50 = existing.map50
                if point.map50_95 is None:
                    point.map50_95 = existing.map50_95
                if point.mask_map50 is None:
                    point.mask_map50 = existing.mask_map50
                if point.mask_map50_95 is None:
                    point.mask_map50_95 = existing.mask_map50_95
                if point.val_loss is None:
                    point.val_loss = existing.val_loss
                if point.train_loss is None:
                    point.train_loss = existing.train_loss
                self.state.history[idx] = point
                self.state.latest_metrics = point
                return
        self.state.history.append(point)
        self.state.latest_metrics = point

    def feed_line(self, raw_line: str) -> bool:
        """Feed one streamed log line; returns True if structured monitor state changed."""
        # Handle carriage-return progress overwrites by inspecting the final segment
        cleaned = strip_ansi(raw_line)
        segments = [seg.strip() for seg in cleaned.split("\r") if seg.strip()]
        if not segments:
            return False

        changed = False
        for line in segments:
            if self._feed_single_line(line):
                changed = True
        return changed

    def _feed_single_line(self, line: str) -> bool:
        changed = False
        low = line.lower()

        # Warnings
        if "[warn]" in low:
            if line not in self.state.warnings:
                self.state.warnings.append(line)
                changed = True

        # Colab live browser URL
        url_m = _COLAB_URL_RE.search(line)
        if url_m:
            url = url_m.group(1).rstrip(".,)")
            if url != self.state.colab_url:
                self.state.colab_url = url
                changed = True

        # Lifecycle stages
        if "packaging inputs for" in low:
            self.state.stage = "packaging"
            changed = True
        elif "provisioning colab" in low or "creating session" in low:
            self.state.stage = "provisioning"
            m_sess = re.search(r"session\s+'([^']+)'", line)
            if m_sess:
                self.state.session_name = m_sess.group(1)
            changed = True
        elif "uploading " in low:
            self.state.stage = "uploading"
            changed = True
        elif (
            "ensuring annolid runtime" in low
            or "reusing already-installed annolid" in low
        ):
            self.state.stage = "installing"
            changed = True
        elif (
            "running: annolid-run train" in low or "executing remote annolid job" in low
        ):
            if self.state.kind == "predict":
                self.state.stage = "inference"
            elif self.state.stage in {
                "idle",
                "starting",
                "packaging",
                "provisioning",
                "uploading",
                "installing",
            }:
                self.state.stage = (
                    "training" if self.state.kind != "predict" else "inference"
                )
            changed = True
        elif "running: annolid-run predict" in low:
            self.state.stage = "inference"
            self.state.step_unit = "frame"
            changed = True
        elif "downloading " in low and "output_bundle.zip" in low:
            self.state.stage = "downloading"
            changed = True

        # GPU detection from nvidia-smi table
        gpu_m = _NVIDIA_GPU_RE.search(line)
        if gpu_m:
            self._pending_gpu_name = gpu_m.group("gpu").strip()
            if not self.state.gpu_name:
                self.state.gpu_name = self._pending_gpu_name
                changed = True
        mem_m = _NVIDIA_MEM_RE.search(line)
        if mem_m and self._pending_gpu_name:
            total_mib = int(mem_m.group("total"))
            self.state.gpu_mem_total_mib = total_mib
            self.state.gpu_name = f"{self._pending_gpu_name} ({total_mib} MiB)"
            self._pending_gpu_name = ""
            changed = True

        # Ultralytics YOLO epoch row
        ym = _YOLO_EPOCH_RE.match(line)
        if ym:
            epoch = int(ym.group("epoch"))
            total = int(ym.group("total"))
            mem_str = ym.group("mem")
            nums = [
                _safe_float(tok)
                for tok in ym.group("nums").split()
                if _safe_float(tok) is not None
            ]
            # YOLO detect has (box, cls, dfl, instances, size); seg has (box, seg, cls, dfl, ...)
            box_l = nums[0] if len(nums) >= 1 else None
            seg_l = nums[1] if len(nums) >= 4 else None
            cls_l = nums[2] if len(nums) >= 4 else (nums[1] if len(nums) >= 2 else None)
            dfl_l = nums[3] if len(nums) >= 4 else (nums[2] if len(nums) >= 3 else None)
            loss_parts = [v for v in (box_l, seg_l, cls_l, dfl_l) if v is not None]
            total_loss = round(sum(loss_parts), 4) if loss_parts else None

            self.state.stage = "training"
            self.state.step_unit = "epoch"
            self.state.current_step = epoch
            self.state.total_steps = total
            self.state.gpu_mem_used = mem_str
            self._upsert_point(
                ColabMetricPoint(
                    step=epoch,
                    total_steps=total,
                    train_loss=total_loss,
                    box_loss=box_l,
                    seg_loss=seg_l,
                    cls_loss=cls_l,
                    dfl_loss=dfl_l,
                    gpu_mem=mem_str,
                    label=f"Epoch {epoch}/{total}",
                )
            )
            return True

        # Ultralytics validation summary row
        vm = _YOLO_VAL_ALL_RE.match(line)
        if vm:
            vals = [_safe_float(tok) for tok in vm.group("metrics").split()]
            # Format: Box(P, R, mAP50, mAP50-95) [Mask(P, R, mAP50, mAP50-95)]
            box_map50 = vals[2] if len(vals) >= 4 else None
            box_map50_95 = vals[3] if len(vals) >= 4 else None
            mask_map50 = vals[6] if len(vals) >= 8 else None
            mask_map50_95 = vals[7] if len(vals) >= 8 else None
            step = self.state.current_step or (len(self.state.history) + 1)
            existing = self.state.latest_metrics
            pt = ColabMetricPoint(
                step=step,
                total_steps=self.state.total_steps,
                train_loss=existing.train_loss
                if existing and existing.step == step
                else None,
                box_loss=existing.box_loss
                if existing and existing.step == step
                else None,
                seg_loss=existing.seg_loss
                if existing and existing.step == step
                else None,
                cls_loss=existing.cls_loss
                if existing and existing.step == step
                else None,
                dfl_loss=existing.dfl_loss
                if existing and existing.step == step
                else None,
                gpu_mem=existing.gpu_mem
                if existing and existing.step == step
                else self.state.gpu_mem_used,
                map50=box_map50,
                map50_95=box_map50_95,
                mask_map50=mask_map50,
                mask_map50_95=mask_map50_95,
                label=f"Epoch {step}/{self.state.total_steps or '?'}",
            )
            self._upsert_point(pt)
            return True

        # Generic Epoch / Iter progress (DINO KPSEG, Torchvision Mask/Keypoint R-CNN, etc.)
        em = _GENERIC_EPOCH_RE.search(line)
        im = _GENERIC_ITER_RE.search(line) if not em else None
        if em or im:
            step = int((em or im).group(1))
            total = int((em or im).group(2))
            kvs = {
                m.group("key").lower(): _safe_float(m.group("val"))
                for m in _KV_FLOAT_RE.finditer(line)
            }
            self.state.stage = "training"
            self.state.step_unit = "epoch" if em else "iter"
            self.state.current_step = step
            self.state.total_steps = total
            train_loss = (
                kvs.get("train_loss") or kvs.get("total_loss") or kvs.get("loss")
            )
            val_loss = kvs.get("val_loss")
            map50 = kvs.get("map50")
            if train_loss is not None or val_loss is not None or map50 is not None:
                self._upsert_point(
                    ColabMetricPoint(
                        step=step,
                        total_steps=total,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        map50=map50,
                        label=f"{self.state.step_unit.title()} {step}/{total}",
                    )
                )
            return True

        # Inference frame counters
        fm = _FRAME_PROGRESS_RE.search(line)
        if fm:
            self.state.stage = "inference"
            self.state.step_unit = "frame"
            self.state.current_step = int(fm.group("frame"))
            self.state.total_steps = int(fm.group("total"))
            return True
        dm = _LABELME_DONE_RE.search(line)
        if dm:
            self.state.stage = "inference"
            self.state.step_unit = "frame"
            self.state.current_step = int(dm.group("frames"))
            return True

        # Structured remote result JSON
        marker = "ANNOLID_COLAB_RESULT_JSON="
        idx = line.find(marker)
        if idx != -1:
            try:
                payload = json.loads(line[idx + len(marker) :].strip())
                if isinstance(payload, dict):
                    self.state.artifact_count = int(payload.get("artifact_count") or 0)
                    if payload.get("elapsed_seconds") is not None:
                        self.state.elapsed_remote_s = float(payload["elapsed_seconds"])
                    if payload.get("job_id"):
                        self.state.job_id = str(payload["job_id"])
                    return True
            except Exception:
                pass

        return changed


def load_metrics_from_results_csv(csv_path: str | Path) -> list[ColabMetricPoint]:
    """Load epoch metrics from an Ultralytics YOLO `results.csv` for Monitor replay."""
    path = Path(csv_path)
    if not path.is_file():
        return []
    points: list[ColabMetricPoint] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [
                {
                    str(k).strip(): str(v).strip()
                    for k, v in row.items()
                    if k is not None
                }
                for row in reader
                if row
            ]
    except Exception:
        return []

    total = len(rows) or None
    for idx, row in enumerate(rows, start=1):
        ep = int(float(row["epoch"])) if row.get("epoch") else idx
        box_l = _safe_float(row.get("train/box_loss"))
        seg_l = _safe_float(row.get("train/seg_loss"))
        cls_l = _safe_float(row.get("train/cls_loss"))
        dfl_l = _safe_float(row.get("train/dfl_loss"))
        loss_parts = [v for v in (box_l, seg_l, cls_l, dfl_l) if v is not None]
        train_loss = (
            round(sum(loss_parts), 4)
            if loss_parts
            else _safe_float(row.get("train_loss") or row.get("loss"))
        )
        val_parts = [
            v
            for v in (
                _safe_float(row.get("val/box_loss")),
                _safe_float(row.get("val/seg_loss")),
                _safe_float(row.get("val/cls_loss")),
                _safe_float(row.get("val/dfl_loss")),
            )
            if v is not None
        ]
        val_loss = (
            round(sum(val_parts), 4) if val_parts else _safe_float(row.get("val_loss"))
        )
        points.append(
            ColabMetricPoint(
                step=ep,
                total_steps=total,
                train_loss=train_loss,
                val_loss=val_loss,
                box_loss=box_l,
                seg_loss=seg_l,
                cls_loss=cls_l,
                dfl_loss=dfl_l,
                map50=_safe_float(row.get("metrics/mAP50(B)")),
                map50_95=_safe_float(row.get("metrics/mAP50-95(B)")),
                mask_map50=_safe_float(row.get("metrics/mAP50(M)")),
                mask_map50_95=_safe_float(row.get("metrics/mAP50-95(M)")),
                pose_map50=_safe_float(row.get("metrics/mAP50(P)")),
                pose_map50_95=_safe_float(row.get("metrics/mAP50-95(P)")),
                label=f"Epoch {ep}/{total or '?'}",
            )
        )
    return points
