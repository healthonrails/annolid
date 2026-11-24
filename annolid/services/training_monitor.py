"""Domain & IO service for discovering, inspecting, and summarizing training runs.

This module intentionally has zero Qt dependencies and zero heavy ML/TensorBoard
top-level imports so it can be imported safely during lean GUI startup and unit
tests. Optional TensorBoard ``EventAccumulator`` reading is imported lazily and
degrades gracefully when ``tensorboard`` is not installed.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import statistics
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from annolid.utils.atomic_io import atomic_write_json
from annolid.utils.logger import logger

RUN_METADATA_FILENAME = "annolid_run_meta.json"
RUN_METADATA_SCHEMA_VERSION = 1


_TB_SUBDIR_NAMES = frozenset({"tensorboard", "tb", "tb_logs", "tfevents"})
_CSV_CANDIDATE_NAMES = ("results.csv", "training_log.csv", "metrics.csv")
_CONFIG_CANDIDATE_NAMES = (
    "args.yaml",
    "config.yaml",
    "hparams.yaml",
    "train_config.json",
)
_LOG_CANDIDATE_NAMES = ("train.log", "training.log")
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

_TRAIN_LOSS_DIRECT_KEYS: Tuple[str, ...] = (
    "train_loss",
    "loss/total",
    "train_loss_total",
    "total_loss",
    "loss",
)
_TRAIN_LOSS_COMPONENT_KEYS: Tuple[str, ...] = (
    "train/box_loss",
    "train/seg_loss",
    "train/pose_loss",
    "train/kobj_loss",
    "train/cls_loss",
    "train/dfl_loss",
)
_VAL_LOSS_DIRECT_KEYS: Tuple[str, ...] = (
    "val_loss",
    "val/loss",
    "val_loss_total",
)
_VAL_LOSS_COMPONENT_KEYS: Tuple[str, ...] = (
    "val/box_loss",
    "val/seg_loss",
    "val/pose_loss",
    "val/kobj_loss",
    "val/cls_loss",
    "val/dfl_loss",
)
_PRIMARY_METRIC_CANDIDATES: Tuple[str, ...] = (
    "metrics/mAP50(M)",
    "metrics/mAP50(P)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95(P)",
    "metrics/mAP50-95(B)",
    "metrics/pck@8px",
    "metrics/pck@4px",
    "metrics/pck@16px",
    "metrics/pck@2px",
    "metrics/segm_AP",
    "metrics/bbox_AP",
    "metrics/keypoint_AP",
    "metrics/AP",
    "val/accuracy",
    "val_acc",
    "accuracy",
    "val_f1",
    "f1",
)

_TIMELINE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("train_loss", "train"),
    ("val_loss", "val"),
    ("loss_classifier", "cls"),
    ("loss_box_reg", "box"),
    ("loss_mask", "mask"),
    ("loss_keypoint", "kpt"),
    ("train/box_loss", "train_box"),
    ("train/seg_loss", "train_seg"),
    ("train/pose_loss", "train_pose"),
    ("train/cls_loss", "train_cls"),
    ("train/dfl_loss", "train_dfl"),
    ("val/box_loss", "val_box"),
    ("val/seg_loss", "val_seg"),
    ("val/pose_loss", "val_pose"),
    ("val/cls_loss", "val_cls"),
    ("val/dfl_loss", "val_dfl"),
    ("metrics/precision(B)", "prec_B"),
    ("metrics/recall(B)", "recall_B"),
    ("metrics/mAP50(B)", "mAP50_B"),
    ("metrics/mAP50-95(B)", "mAP5095_B"),
    ("metrics/mAP50(M)", "mAP50_M"),
    ("metrics/mAP50-95(M)", "mAP5095_M"),
    ("metrics/mAP50(P)", "mAP50_P"),
    ("metrics/mAP50-95(P)", "mAP5095_P"),
    ("metrics/precision(P)", "prec_P"),
    ("metrics/recall(P)", "recall_P"),
    ("metrics/pck@2px", "pck2"),
    ("metrics/pck@4px", "pck4"),
    ("metrics/pck@8px", "pck8"),
    ("metrics/pck@16px", "pck16"),
    ("metrics/swap_rate", "swap"),
    ("metrics/precision(M)", "prec_M"),
    ("metrics/recall(M)", "recall_M"),
    ("lr", "lr"),
    ("lr/pg0", "lr"),
    ("seconds", "sec"),
)

_TB_SCALAR_CACHE: Dict[
    Tuple[str, Tuple[Tuple[str, int, int], ...]], Dict[str, "MetricSeries"]
] = {}
_TB_SCALAR_CACHE_MAX = 64


@dataclass(frozen=True)
class MetricSeries:
    """Ordered scalar series keyed by epoch, iteration, or step."""

    tag: str
    steps: Tuple[int, ...]
    values: Tuple[float, ...]
    source: str = "csv"  # csv | tensorboard | derived

    @property
    def latest_step(self) -> Optional[int]:
        return self.steps[-1] if self.steps else None

    @property
    def latest_value(self) -> Optional[float]:
        return self.values[-1] if self.values else None


@dataclass(frozen=True)
class RunMetadata:
    """User-editable sidecar metadata stored non-destructively in ``annolid_run_meta.json``."""

    alias: str = ""
    tags: Tuple[str, ...] = ()
    notes: str = ""
    starred: bool = False
    updated_at: Optional[float] = None


@dataclass(frozen=True)
class RunHealthFinding:
    """Automated training health or anomaly diagnostic for a run."""

    severity: str  # info | warning | critical
    code: str
    message: str


@dataclass(frozen=True)
class HparamDiffRow:
    """One hyperparameter row compared across one or more training runs."""

    key: str
    values: Tuple[str, ...]
    differs: bool


@dataclass
class DiscoveredRun:
    """Canonical summary & metrics payload for a single training run directory."""

    run_dir: Path
    display_name: str
    task: str = "train"
    model: str = ""
    status: str = "recorded"  # running | finished | failed | recorded
    started_at: float = 0.0
    updated_at: Optional[float] = None
    step_unit: str = "epoch"  # epoch | iteration | step
    last_step: Optional[int] = None
    total_steps: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    min_train_loss: Optional[float] = None
    min_val_loss: Optional[float] = None
    best_metric_value: Optional[float] = None
    best_metric_step: Optional[int] = None
    results_csv: Optional[Path] = None
    tensorboard_dir: Optional[Path] = None
    has_projector: bool = False
    log_path: Optional[Path] = None
    command_path: Optional[Path] = None
    config_path: Optional[Path] = None
    checkpoints: Tuple[Path, ...] = ()
    plot_images: Tuple[Path, ...] = ()
    csv_columns: Tuple[str, ...] = ()
    csv_rows: List[Dict[str, str]] = field(default_factory=list)
    metrics: Dict[str, MetricSeries] = field(default_factory=dict)
    hparams: Dict[str, Any] = field(default_factory=dict)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    health_findings: Tuple[RunHealthFinding, ...] = ()


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return int(v)
    except Exception:
        return None


def smooth_ema(values: Sequence[float], weight: float) -> Tuple[float, ...]:
    """Apply TensorBoard-style bias-corrected exponential moving average."""
    if not values:
        return ()
    w = max(0.0, min(0.99, float(weight)))
    if w <= 1e-9:
        return tuple(float(v) for v in values)
    smoothed: List[float] = []
    acc = 0.0
    denom = 0.0
    for val in values:
        v = float(val)
        acc = acc * w + (1.0 - w) * v
        denom = denom * w + (1.0 - w)
        smoothed.append(acc / denom if denom > 0 else v)
    return tuple(smoothed)


def canonicalize_run_dir(path: Path, *, runs_root: Optional[Path] = None) -> Path:
    """Normalize a TensorBoard subdir (e.g. ``<run>/tensorboard``) back to ``<run>``."""
    resolved = Path(path).expanduser().resolve()
    root_resolved = (
        Path(runs_root).expanduser().resolve() if runs_root is not None else None
    )
    if resolved.is_file():
        resolved = resolved.parent
    if (
        resolved.name.lower() in _TB_SUBDIR_NAMES
        and resolved.parent != resolved
        and (root_resolved is None or resolved != root_resolved)
    ):
        parent = resolved.parent
        # Step up when parent is inside runs_root (or not root itself) and looks like a run folder.
        if root_resolved is None or parent != root_resolved:
            return parent
        # Even if parent == runs_root, step up only if parent has run artifacts.
        if any(
            (parent / name).exists()
            for name in _CSV_CANDIDATE_NAMES + _LOG_CANDIDATE_NAMES
        ):
            return parent
    return resolved


def _bounded_walk(
    root: Path, *, max_depth: int = 6
) -> Iterable[Tuple[Path, List[str], List[str]]]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return
    root_parts = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=True):
        cur = Path(dirpath)
        depth = len(cur.parts) - root_parts
        if depth > int(max_depth):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d and not d.startswith(".") and d not in {"__pycache__", ".git"}
        )
        yield cur, dirnames, filenames


def find_results_csv(run_dir: Path) -> Optional[Path]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return None
    for name in _CSV_CANDIDATE_NAMES:
        direct = root / name
        if direct.is_file():
            return direct
    for cur, _dirs, files in _bounded_walk(root, max_depth=3):
        for name in _CSV_CANDIDATE_NAMES:
            if name in files:
                return cur / name
    return None


def find_tensorboard_dir(run_dir: Path) -> Optional[Path]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return None
    for candidate_dir in (root, root / "tensorboard", root / "tb"):
        if candidate_dir.is_dir():
            try:
                if any(
                    p.is_file() and p.name.startswith("events.out.tfevents.")
                    for p in candidate_dir.iterdir()
                ):
                    return candidate_dir
            except OSError:
                continue
    for cur, _dirs, files in _bounded_walk(root, max_depth=3):
        if any(f.startswith("events.out.tfevents.") for f in files):
            return cur
    return None


def has_projector_config(run_dir: Path) -> bool:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return False
    for candidate in (
        root / "projector_config.pbtxt",
        root / "tensorboard" / "projector_config.pbtxt",
        root / "tb" / "projector_config.pbtxt",
    ):
        if candidate.is_file():
            return True
    for _cur, _dirs, files in _bounded_walk(root, max_depth=3):
        if "projector_config.pbtxt" in files:
            return True
    return False


def find_train_log(run_dir: Path) -> Optional[Path]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return None
    for name in _LOG_CANDIDATE_NAMES:
        direct = root / name
        if direct.is_file():
            return direct
    for cur, _dirs, files in _bounded_walk(root, max_depth=2):
        for name in _LOG_CANDIDATE_NAMES:
            if name in files:
                return cur / name
    return None


def find_config_file(run_dir: Path) -> Optional[Path]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return None
    for name in _CONFIG_CANDIDATE_NAMES:
        direct = root / name
        if direct.is_file():
            return direct
    for cur, _dirs, files in _bounded_walk(root, max_depth=2):
        for name in _CONFIG_CANDIDATE_NAMES:
            if name in files:
                return cur / name
    return None


def find_checkpoints(run_dir: Path) -> Tuple[Path, ...]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return ()
    found: List[Path] = []
    seen: set[str] = set()
    preferred = (
        root / "weights" / "best.pt",
        root / "weights" / "last.pt",
        root / "best.pt",
        root / "last.pt",
        root / "model_final.pth",
        root / "best_model.pth",
    )
    for cand in preferred:
        if cand.is_file():
            key = str(cand.resolve())
            if key not in seen:
                seen.add(key)
                found.append(cand.resolve())
    for cur, _dirs, files in _bounded_walk(root, max_depth=2):
        for fname in sorted(files):
            low = fname.lower()
            if low.endswith((".pt", ".pth", ".onnx")):
                cand = (cur / fname).resolve()
                key = str(cand)
                if key not in seen:
                    seen.add(key)
                    found.append(cand)
    return tuple(found)


def _plot_priority(path: Path) -> Tuple[int, float, str]:
    name = path.name.lower()
    if name == "results.png":
        rank = 0
    elif "confusion_matrix" in name:
        rank = 1
    elif name.endswith("_curve.png") or "loss" in name:
        rank = 2
    elif name.startswith("val_batch"):
        rank = 3
    elif name.startswith("train_batch"):
        rank = 4
    elif name.startswith("labels"):
        rank = 5
    else:
        rank = 6
    try:
        mtime = -float(path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return (rank, mtime, name)


def find_plot_images(run_dir: Path, *, max_images: int = 80) -> Tuple[Path, ...]:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        return ()
    images: List[Path] = []
    seen: set[str] = set()
    for cur, dirs, files in _bounded_walk(root, max_depth=3):
        # Skip raw dataset or projector sprite caches
        dirs[:] = [
            d
            for d in dirs
            if d.lower() not in {"weights", "checkpoints", "__pycache__"}
        ]
        for fname in files:
            p = cur / fname
            if p.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            if "sprite" in fname.lower():
                continue
            resolved = p.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            images.append(resolved)
    images.sort(key=_plot_priority)
    return tuple(images[: max(1, int(max_images))])


@lru_cache(maxsize=4)
def _has_yaml() -> bool:
    try:
        import yaml  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def read_run_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {}
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if config_path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    if _has_yaml():
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    parsed: Dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        if key:
            parsed[key] = v.strip()
    return parsed


def extract_total_steps(
    hparams: Dict[str, Any], *, command_path: Optional[Path] = None
) -> Tuple[Optional[int], str]:
    """Return ``(total_steps, step_unit)`` from config or command line."""
    for key, unit in (
        ("epochs", "epoch"),
        ("max_epochs", "epoch"),
        ("max_iterations", "iteration"),
        ("max_iter", "iteration"),
    ):
        val = safe_int(hparams.get(key))
        if val is not None and val > 0:
            return val, unit

    if command_path is not None and command_path.is_file():
        try:
            cmd_text = command_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            cmd_text = ""
        for pat, unit in (
            (r"--epochs(?:=|\s+)(\d+)", "epoch"),
            (r"\bepochs=(\d+)", "epoch"),
            (r"--max-iterations(?:=|\s+)(\d+)", "iteration"),
            (r"--max_iterations(?:=|\s+)(\d+)", "iteration"),
        ):
            m = re.search(pat, cmd_text)
            if m:
                parsed = safe_int(m.group(1))
                if parsed is not None and parsed > 0:
                    return parsed, unit
    return None, "epoch"


def read_csv_rows(
    path: Path, *, max_rows: int = 2000
) -> Tuple[Tuple[str, ...], List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            raw_fields = list(reader.fieldnames or ())
            columns = tuple(
                str(c).strip() for c in raw_fields if c is not None and str(c).strip()
            )
            rows: List[Dict[str, str]] = []
            for raw_row in reader:
                if not raw_row:
                    continue
                cleaned = {
                    str(k).strip(): ("" if v is None else str(v).strip())
                    for k, v in raw_row.items()
                    if k is not None and str(k).strip()
                }
                if any(cleaned.values()):
                    rows.append(cleaned)
            if max_rows and len(rows) > int(max_rows):
                rows = rows[-int(max_rows) :]
            return columns, rows
    except Exception:
        return (), []


def _lookup_ci(row: Dict[str, str], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            v = safe_float(row.get(key))
            if v is not None:
                return v
    lowered = {k.lower(): v for k, v in row.items()}
    for key in keys:
        v = safe_float(lowered.get(key.lower()))
        if v is not None:
            return v
    return None


def _extract_step_from_row(row: Dict[str, str], fallback_idx: int) -> Tuple[int, str]:
    for key, unit in (
        ("epoch", "epoch"),
        ("Epoch", "epoch"),
        ("iteration", "iteration"),
        ("iter", "iteration"),
        ("step", "step"),
        ("Step", "step"),
    ):
        val = safe_int(row.get(key))
        if val is not None:
            return val, unit
    return int(fallback_idx), "step"


def _build_csv_metrics(
    columns: Sequence[str], rows: Sequence[Dict[str, str]]
) -> Tuple[Dict[str, MetricSeries], str]:
    if not rows:
        return {}, "epoch"

    steps_by_tag: Dict[str, List[int]] = {}
    vals_by_tag: Dict[str, List[float]] = {}
    detected_unit = "epoch"

    derived_train_steps: List[int] = []
    derived_train_vals: List[float] = []
    derived_val_steps: List[int] = []
    derived_val_vals: List[float] = []

    step_col_names = {"epoch", "iteration", "iter", "step"}
    col_list = list(columns) if columns else list(rows[0].keys())

    for idx, row in enumerate(rows, start=1):
        step, unit = _extract_step_from_row(row, idx)
        if idx == 1:
            detected_unit = unit

        for col in col_list:
            if col.lower() in step_col_names:
                continue
            v = safe_float(row.get(col))
            if v is None:
                continue
            steps_by_tag.setdefault(col, []).append(step)
            vals_by_tag.setdefault(col, []).append(v)

        # Compute aggregate train_loss if not already an explicit column
        direct_train = _lookup_ci(row, _TRAIN_LOSS_DIRECT_KEYS)
        if direct_train is None:
            parts = [
                v
                for k in _TRAIN_LOSS_COMPONENT_KEYS
                for v in (_lookup_ci(row, (k,)),)
                if v is not None
            ]
            if parts:
                derived_train_steps.append(step)
                derived_train_vals.append(round(sum(parts), 6))

        direct_val = _lookup_ci(row, _VAL_LOSS_DIRECT_KEYS)
        if direct_val is None:
            parts = [
                v
                for k in _VAL_LOSS_COMPONENT_KEYS
                for v in (_lookup_ci(row, (k,)),)
                if v is not None
            ]
            if parts:
                derived_val_steps.append(step)
                derived_val_vals.append(round(sum(parts), 6))

    series: Dict[str, MetricSeries] = {}
    for tag, st_list in steps_by_tag.items():
        series[tag] = MetricSeries(
            tag=tag,
            steps=tuple(st_list),
            values=tuple(vals_by_tag[tag]),
            source="csv",
        )
    if derived_train_steps and "train_loss" not in series:
        series["train_loss"] = MetricSeries(
            tag="train_loss",
            steps=tuple(derived_train_steps),
            values=tuple(derived_train_vals),
            source="derived",
        )
    if derived_val_steps and "val_loss" not in series:
        series["val_loss"] = MetricSeries(
            tag="val_loss",
            steps=tuple(derived_val_steps),
            values=tuple(derived_val_vals),
            source="derived",
        )
    return series, detected_unit


def read_tensorboard_scalars(
    tb_dir: Path, *, max_points_per_tag: int = 1000
) -> Dict[str, MetricSeries]:
    """Read scalar series from TensorBoard event files with graceful fallback.

    Returns an empty dict when ``tensorboard`` is not installed or no scalar
    events exist. Results are cached by event file mtime + byte size so live
    polling stays fast.
    """
    root = Path(tb_dir).expanduser().resolve()
    if not root.is_dir():
        return {}
    try:
        event_files = sorted(
            p
            for p in root.iterdir()
            if p.is_file() and p.name.startswith("events.out.tfevents.")
        )
    except OSError:
        return {}
    if not event_files:
        return {}

    fingerprint_items: List[Tuple[str, int, int]] = []
    for ef in event_files:
        try:
            st = ef.stat()
            fingerprint_items.append((ef.name, int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            continue
    cache_key = (str(root), tuple(fingerprint_items))
    cached = _TB_SCALAR_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore
    except Exception:
        return {}

    try:
        acc = event_accumulator.EventAccumulator(
            str(root),
            size_guidance={
                event_accumulator.SCALARS: int(max_points_per_tag),
                event_accumulator.IMAGES: 0,
                event_accumulator.AUDIO: 0,
                event_accumulator.HISTOGRAMS: 0,
                event_accumulator.TENSORS: 0,
            },
        )
        acc.Reload()
        scalar_tags = list(acc.Tags().get("scalars") or [])
    except Exception:
        return {}

    result: Dict[str, MetricSeries] = {}
    for tag in scalar_tags:
        try:
            events = acc.Scalars(tag)
        except Exception:
            continue
        steps: List[int] = []
        vals: List[float] = []
        for ev in events:
            v = safe_float(getattr(ev, "value", None))
            if v is None:
                continue
            steps.append(int(getattr(ev, "step", len(steps))))
            vals.append(v)
        if steps:
            result[str(tag)] = MetricSeries(
                tag=str(tag),
                steps=tuple(steps),
                values=tuple(vals),
                source="tensorboard",
            )

    if len(_TB_SCALAR_CACHE) >= _TB_SCALAR_CACHE_MAX:
        _TB_SCALAR_CACHE.clear()
    _TB_SCALAR_CACHE[cache_key] = dict(result)
    return result


def _normalize_tag_key(tag: str) -> str:
    s = str(tag or "").strip().lower()
    if s.startswith("ultralytics/"):
        s = s[len("ultralytics/") :]
    return re.sub(r"[^a-z0-9/]+", "_", s).strip("_")


def _resolve_loss_series(
    metrics: Dict[str, MetricSeries],
    *,
    kind: str,
) -> Optional[MetricSeries]:
    """Find the primary train or validation loss series, including TB-prefixed tags."""
    candidates = (
        _TRAIN_LOSS_DIRECT_KEYS + _TRAIN_LOSS_COMPONENT_KEYS
        if kind == "train"
        else _VAL_LOSS_DIRECT_KEYS + _VAL_LOSS_COMPONENT_KEYS
    )
    hit = _find_metric_series_ci(metrics, candidates)
    if hit is not None:
        return hit
    for tag, metric_series in metrics.items():
        if not metric_series.values:
            continue
        tl = tag.lower()
        if kind == "train":
            if ("train" in tl and "loss" in tl) or tl in {"loss/total", "loss"}:
                return metric_series
        else:
            if "val" in tl and "loss" in tl:
                return metric_series
    return None


def _pick_summary_metrics(
    metrics: Dict[str, MetricSeries],
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[float]]:
    lower_map = {k.lower(): (k, s) for k, s in metrics.items()}

    def _first_val(candidates: Sequence[str]) -> Tuple[Optional[str], Optional[float]]:
        for cand in candidates:
            hit = lower_map.get(cand.lower())
            if hit is not None and hit[1].latest_value is not None:
                return hit[0], hit[1].latest_value
        return None, None

    train_series = _resolve_loss_series(metrics, kind="train")
    val_series = _resolve_loss_series(metrics, kind="val")
    train_loss = train_series.latest_value if train_series is not None else None
    val_loss = val_series.latest_value if val_series is not None else None
    primary_name, primary_val = _first_val(_PRIMARY_METRIC_CANDIDATES)

    if primary_val is None:
        for tag, metric_series in metrics.items():
            tl = tag.lower()
            if any(
                tok in tl for tok in ("map50", "pck@", "metrics/ap", "accuracy", "f1")
            ):
                if metric_series.latest_value is not None:
                    primary_name, primary_val = tag, metric_series.latest_value
                    break
    return train_loss, val_loss, primary_name, primary_val


def infer_task_and_model(
    run_dir: Path,
    *,
    runs_root: Optional[Path] = None,
    hparams: Optional[Dict[str, Any]] = None,
    results_csv: Optional[Path] = None,
    command_path: Optional[Path] = None,
) -> Tuple[str, str, str]:
    """Infer ``(task, model, display_name)`` for a training run directory."""
    resolved = Path(run_dir).expanduser().resolve()
    rel_parts: Tuple[str, ...] = ()
    if runs_root is not None:
        try:
            rel = resolved.relative_to(Path(runs_root).expanduser().resolve())
            rel_parts = rel.parts
        except Exception:
            rel_parts = ()

    display_name = "/".join(rel_parts) if rel_parts else resolved.name
    task = "train"
    model = ""

    if len(rel_parts) >= 3:
        task = rel_parts[0]
        model = rel_parts[1]
    elif len(rel_parts) == 2:
        task = rel_parts[0]
        model = rel_parts[1]
    elif len(rel_parts) == 1:
        model = rel_parts[0]

    hp = hparams or {}
    if hp.get("task"):
        task = str(hp["task"])
    if hp.get("model"):
        m_str = str(hp["model"]).strip()
        if m_str:
            model = (
                Path(m_str).stem
                if ("/" in m_str or m_str.endswith((".pt", ".pth")))
                else m_str
            )

    if command_path is not None and command_path.is_file():
        try:
            cmd_txt = command_path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            cmd_txt = ""
        if "dino_kpseg" in cmd_txt:
            task = "dino_kpseg"
        elif "keypoint_rcnn" in cmd_txt:
            task = "keypoint_rcnn"
        elif "maskrcnn" in cmd_txt:
            task = "maskrcnn"
        elif "yolo" in cmd_txt:
            task = "yolo"

    if task == "train" and results_csv is not None:
        if results_csv.name == "training_log.csv":
            task = "maskrcnn"

    if not model:
        model = resolved.name
    return task, model, display_name


def inspect_run_dir(
    run_dir: Path,
    *,
    runs_root: Optional[Path] = None,
    include_tb_scalars: bool = False,
    max_csv_rows: int = 2000,
    status: str = "recorded",
    task_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> DiscoveredRun:
    """Inspect a run directory and return structured metadata, artifacts, and metrics."""
    canonical = canonicalize_run_dir(run_dir, runs_root=runs_root)
    results_csv = find_results_csv(canonical)
    tb_dir = find_tensorboard_dir(canonical)
    log_path = find_train_log(canonical)
    config_path = find_config_file(canonical)
    command_path = (
        (canonical / "command.txt") if (canonical / "command.txt").is_file() else None
    )
    hparams = read_run_config(config_path)
    checkpoints = find_checkpoints(canonical)
    plot_images = find_plot_images(canonical)
    projector = has_projector_config(canonical)

    task, model, display_name = infer_task_and_model(
        canonical,
        runs_root=runs_root,
        hparams=hparams,
        results_csv=results_csv,
        command_path=command_path,
    )
    if task_override:
        task = str(task_override)
    if model_override:
        model = str(model_override)

    columns: Tuple[str, ...] = ()
    rows: List[Dict[str, str]] = []
    metrics: Dict[str, MetricSeries] = {}
    step_unit = "epoch"

    if results_csv is not None:
        columns, rows = read_csv_rows(results_csv, max_rows=max_csv_rows)
        metrics, step_unit = _build_csv_metrics(columns, rows)

    if include_tb_scalars and tb_dir is not None:
        tb_series = read_tensorboard_scalars(tb_dir)
        existing_norm = {_normalize_tag_key(k) for k in metrics}
        for tag, metric_series in tb_series.items():
            norm = _normalize_tag_key(tag)
            if tag in metrics or norm in existing_norm:
                continue
            metrics[tag] = metric_series
            existing_norm.add(norm)
        if not rows and tb_series:
            # Infer step unit if only TensorBoard scalars exist
            if any(t.startswith("loss/") for t in tb_series):
                step_unit = "iteration"

    total_steps, cfg_unit = extract_total_steps(hparams, command_path=command_path)
    if total_steps is not None and not rows:
        step_unit = cfg_unit

    last_step: Optional[int] = None
    if rows:
        last_step, step_unit = _extract_step_from_row(rows[-1], len(rows))
    elif metrics:
        last_steps = [
            s.latest_step for s in metrics.values() if s.latest_step is not None
        ]
        if last_steps:
            last_step = max(last_steps)

    train_loss, val_loss, primary_name, primary_val = _pick_summary_metrics(metrics)
    min_train_loss, min_val_loss, best_metric_val, best_metric_step = (
        summarize_run_best_metrics(metrics, primary_metric_name=primary_name)
    )
    metadata = load_run_metadata(canonical)

    # Best-effort timestamps from most recently updated artifact
    mtimes: List[float] = []
    for candidate in (results_csv, log_path, tb_dir, canonical):
        if candidate is not None and candidate.exists():
            try:
                mtimes.append(float(candidate.stat().st_mtime))
            except OSError:
                pass
    updated_at = max(mtimes) if mtimes else None
    try:
        started_at = float(canonical.stat().st_ctime) if canonical.exists() else 0.0
    except OSError:
        started_at = updated_at or 0.0

    discovered = DiscoveredRun(
        run_dir=canonical,
        display_name=display_name,
        task=task,
        model=model,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        step_unit=step_unit,
        last_step=last_step,
        total_steps=total_steps,
        train_loss=train_loss,
        val_loss=val_loss,
        primary_metric_name=primary_name,
        primary_metric_value=primary_val,
        min_train_loss=min_train_loss,
        min_val_loss=min_val_loss,
        best_metric_value=best_metric_val,
        best_metric_step=best_metric_step,
        results_csv=results_csv,
        tensorboard_dir=tb_dir,
        has_projector=projector,
        log_path=log_path,
        command_path=command_path,
        config_path=config_path,
        checkpoints=checkpoints,
        plot_images=plot_images,
        csv_columns=columns,
        csv_rows=rows,
        metrics=metrics,
        hparams=hparams,
        metadata=metadata,
    )
    discovered.health_findings = tuple(analyze_run_health(discovered))
    return discovered


def discover_training_runs(
    runs_root: Path,
    *,
    max_runs: int = 200,
    max_depth: int = 6,
    include_tb_scalars: bool = False,
) -> List[DiscoveredRun]:
    """Discover and deduplicate training run directories beneath ``runs_root``."""
    root = Path(runs_root).expanduser().resolve()
    if not root.exists():
        return []

    candidate_dirs: set[Path] = set()
    for cur, _dirs, files in _bounded_walk(root, max_depth=max_depth):
        has_signal = (
            any(name in files for name in _CSV_CANDIDATE_NAMES)
            or any(name in files for name in _LOG_CANDIDATE_NAMES)
            or any(f.startswith("events.out.tfevents.") for f in files)
            or "projector_config.pbtxt" in files
        )
        if has_signal:
            candidate_dirs.add(canonicalize_run_dir(cur, runs_root=root))

    def _mtime(p: Path) -> float:
        best = 0.0
        for cand in (
            p,
            p / "results.csv",
            p / "training_log.csv",
            p / "train.log",
            p / "tensorboard",
        ):
            if cand.exists():
                try:
                    best = max(best, float(cand.stat().st_mtime))
                except OSError:
                    pass
        return best

    sorted_dirs = sorted(candidate_dirs, key=_mtime, reverse=True)[
        : max(1, int(max_runs))
    ]
    return [
        inspect_run_dir(
            d,
            runs_root=root,
            include_tb_scalars=include_tb_scalars,
            max_csv_rows=500,
        )
        for d in sorted_dirs
    ]


def tail_log_text(path: Path, *, max_lines: int = 250) -> str:
    """Read the tail of a log file efficiently without loading the whole file."""
    max_lines = max(1, int(max_lines))
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end <= 0:
                return ""
            block_size = 8192
            data = b""
            lines = 0
            pos = end
            while pos > 0 and lines <= max_lines:
                read_size = min(block_size, pos)
                pos -= read_size
                fh.seek(pos, os.SEEK_SET)
                chunk = fh.read(read_size)
                data = chunk + data
                lines = data.count(b"\n")
                if len(data) > 8_000_000:
                    break
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except Exception:
        return ""


def format_timeline_line(
    step: int, row: Dict[str, str], *, step_unit: str = "epoch"
) -> str:
    prefix = (
        "iter"
        if step_unit == "iteration"
        else ("step" if step_unit == "step" else "epoch")
    )
    parts = [f"{prefix}={int(step)}"]
    for raw_key, label in _TIMELINE_KEYS:
        if raw_key not in row:
            continue
        val = safe_float(row.get(raw_key))
        if val is None:
            continue
        if label == "sec":
            parts.append(f"{label}={val:.1f}")
        elif label == "lr":
            parts.append(f"{label}={val:.2e}")
        else:
            parts.append(f"{label}={val:.4f}")
    return "  ".join(parts)


def format_timeline_lines(
    rows: Sequence[Dict[str, str]],
    *,
    step_unit: str = "epoch",
    max_lines: int = 800,
) -> List[str]:
    out: List[str] = []
    for idx, row in enumerate(rows, start=1):
        step, unit = _extract_step_from_row(row, idx)
        out.append(format_timeline_line(step, row, step_unit=unit or step_unit))
    if max_lines and len(out) > int(max_lines):
        return out[-int(max_lines) :]
    return out


def build_tensorboard_logdir_spec(
    entries: Sequence[Tuple[str, Path]],
) -> str:
    """Build a TensorBoard ``--logdir_spec`` string for comparing multiple runs.

    Format: ``name1:/abs/path1,name2:/abs/path2`` with sanitized, unique labels.
    """
    parts: List[str] = []
    used_names: Dict[str, int] = {}
    for raw_name, raw_path in entries:
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            continue
        clean = re.sub(r"[^a-zA-Z0-9._/-]+", "_", str(raw_name or resolved.name)).strip(
            "_/"
        )
        clean = clean.replace(":", "_").replace(",", "_")
        if not clean:
            clean = resolved.name or "run"
        count = used_names.get(clean, 0)
        used_names[clean] = count + 1
        label = clean if count == 0 else f"{clean}_{count + 1}"
        # Prefer the tensorboard subdir if present for faster startup, else run dir.
        tb_dir = find_tensorboard_dir(resolved) or resolved
        parts.append(f"{label}:{tb_dir}")
    return ",".join(parts)


def normalize_run_tags(raw_tags: Sequence[str] | str) -> Tuple[str, ...]:
    """Normalize comma-separated or list tags into a clean deduplicated tuple."""
    if isinstance(raw_tags, str):
        items = raw_tags.split(",")
    else:
        items = list(raw_tags)
    seen: set[str] = set()
    cleaned: List[str] = []
    for item in items:
        tok = re.sub(r"\s+", "-", str(item or "").strip())
        if not tok:
            continue
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(tok)
    return tuple(cleaned)


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Read optional ``annolid_run_meta.json`` sidecar safely."""
    meta_path = Path(run_dir).expanduser().resolve() / RUN_METADATA_FILENAME
    if not meta_path.is_file():
        return RunMetadata()
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable run metadata %s: %s", meta_path, exc)
        return RunMetadata()
    if not isinstance(payload, dict):
        logger.warning("Ignoring non-object run metadata in %s", meta_path)
        return RunMetadata()
    raw_tags = payload.get("tags") or ()
    tags = normalize_run_tags(
        raw_tags if isinstance(raw_tags, (list, tuple, str)) else ()
    )
    return RunMetadata(
        alias=str(payload.get("alias") or "").strip(),
        tags=tags,
        notes=str(payload.get("notes") or ""),
        starred=bool(payload.get("starred", False)),
        updated_at=safe_float(payload.get("updated_at")),
    )


def save_run_metadata(run_dir: Path, metadata: RunMetadata) -> Path:
    """Persist ``annolid_run_meta.json`` atomically without modifying training artifacts."""
    target_dir = Path(run_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = target_dir / RUN_METADATA_FILENAME
    payload = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "alias": str(metadata.alias or "").strip(),
        "tags": list(normalize_run_tags(metadata.tags)),
        "notes": str(metadata.notes or ""),
        "starred": bool(metadata.starred),
        "updated_at": float(metadata.updated_at or time.time()),
    }
    atomic_write_json(meta_path, payload, indent=2)
    return meta_path


def _find_metric_series_ci(
    metrics: Dict[str, MetricSeries], candidates: Sequence[str]
) -> Optional[MetricSeries]:
    lowered = {k.lower(): v for k, v in metrics.items()}
    for cand in candidates:
        hit = lowered.get(cand.lower())
        if hit is not None and hit.values:
            return hit
    return None


def summarize_run_best_metrics(
    metrics: Dict[str, MetricSeries],
    *,
    primary_metric_name: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """Return ``(min_train_loss, min_val_loss, best_metric_value, best_metric_step)``."""
    train_series = _resolve_loss_series(metrics, kind="train")
    val_series = _resolve_loss_series(metrics, kind="val")
    min_train = (
        min(train_series.values) if train_series and train_series.values else None
    )
    min_val = min(val_series.values) if val_series and val_series.values else None

    best_val: Optional[float] = None
    best_step: Optional[int] = None
    if primary_metric_name and primary_metric_name in metrics:
        p_series = metrics[primary_metric_name]
        if p_series.values:
            best_idx = max(
                range(len(p_series.values)), key=lambda i: p_series.values[i]
            )
            best_val = float(p_series.values[best_idx])
            best_step = int(p_series.steps[best_idx])
    return min_train, min_val, best_val, best_step


def _parse_command_flags(command_path: Optional[Path]) -> Dict[str, str]:
    """Extract ``--key=value`` / ``key=value`` tokens from ``command.txt`` for hparam diffing."""
    if command_path is None or not command_path.is_file():
        return {}
    try:
        raw = command_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    parsed: Dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                if k:
                    parsed[k.replace("-", "_")] = v
            elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                parsed[body.replace("-", "_")] = tokens[idx + 1]
                idx += 1
            else:
                parsed[body.replace("-", "_")] = "true"
        elif "=" in tok and not tok.startswith(("/", ".")):
            k, v = tok.split("=", 1)
            if k and re.fullmatch(r"[A-Za-z0-9_.-]+", k):
                parsed[k.replace("-", "_")] = v
        idx += 1
    return parsed


def extract_effective_hparams(run: DiscoveredRun) -> Dict[str, str]:
    """Build a normalized string hyperparameter map combining config + CLI args."""
    effective: Dict[str, str] = {
        "task": str(run.task or ""),
        "model": str(run.model or ""),
    }
    cmd_flags = _parse_command_flags(run.command_path)
    for k, v in cmd_flags.items():
        if k.lower() in {"project", "name", "output", "output_dir", "save_dir"}:
            continue
        effective[str(k)] = str(v)

    for k, v in (run.hparams or {}).items():
        kl = str(k).strip()
        if not kl or kl.lower() in {"save_dir", "project", "name", "output_dir"}:
            continue
        if isinstance(v, (dict, list)):
            effective[kl] = json.dumps(v, sort_keys=True)
        else:
            effective[kl] = "" if v is None else str(v)
    return effective


def compute_hparam_diff(
    runs: Sequence[DiscoveredRun], *, only_diffs: bool = False
) -> List[HparamDiffRow]:
    """Compare hyperparameters across runs and highlight keys whose values differ."""
    if not runs:
        return []
    maps = [extract_effective_hparams(run) for run in runs]
    all_keys = sorted({k for m in maps for k in m.keys()})
    rows: List[HparamDiffRow] = []
    for key in all_keys:
        vals = tuple(m.get(key, "—") for m in maps)
        differs = len(set(vals)) > 1
        if only_diffs and not differs:
            continue
        rows.append(HparamDiffRow(key=key, values=vals, differs=differs))
    # Surface differing keys first when showing all rows.
    rows.sort(key=lambda r: (not r.differs, r.key))
    return rows


def _value_at_or_before_step(
    metric_series: MetricSeries, target_step: int
) -> Optional[float]:
    """Return the series value at ``target_step`` (or nearest preceding step)."""
    best_val: Optional[float] = None
    for step, val in zip(metric_series.steps, metric_series.values):
        if step <= target_step:
            best_val = float(val)
        else:
            break
    if best_val is None and metric_series.values:
        return float(metric_series.values[0])
    return best_val


def analyze_run_health(
    run: DiscoveredRun,
    *,
    disk_free_gb: Optional[float] = None,
    spike_factor: float = 4.0,
    overfit_ratio: float = 1.15,
    plateau_steps: int = 15,
    low_disk_gb: float = 2.0,
) -> List[RunHealthFinding]:
    """Detect actionable training anomalies (NaN/Inf, loss spike, overfitting, plateau, low disk)."""
    findings: List[RunHealthFinding] = []

    # 1. Check raw CSV rows for literal NaN / Inf entries (all rows for loss cols, last 25 for others)
    non_finite_tokens = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}
    nan_cols: List[str] = []
    recent_start = max(0, len(run.csv_rows) - 25)
    for row_idx, row in enumerate(run.csv_rows):
        in_recent = row_idx >= recent_start
        for k, raw_val in row.items():
            if not in_recent and "loss" not in k.lower():
                continue
            tok = str(raw_val or "").strip().lower()
            if tok in non_finite_tokens:
                if k not in nan_cols:
                    nan_cols.append(k)
    if nan_cols:
        cols_preview = ", ".join(nan_cols[:4])
        findings.append(
            RunHealthFinding(
                severity="critical",
                code="nan_metric",
                message=f"Non-finite (NaN/Inf) values detected in: {cols_preview}",
            )
        )

    # 2. Loss spike / divergence detection
    train_series = _resolve_loss_series(run.metrics, kind="train")
    if train_series is not None and len(train_series.values) >= 5:
        prev_window = [v for v in train_series.values[:-1][-10:] if v > 0]
        latest = train_series.values[-1]
        if prev_window:
            baseline = statistics.median(prev_window)
            if baseline > 1e-6 and latest > max(0.5, baseline * float(spike_factor)):
                findings.append(
                    RunHealthFinding(
                        severity="critical",
                        code="loss_spike",
                        message=(
                            f"Train loss spiked to {latest:.4f} "
                            f"(recent median {baseline:.4f})"
                        ),
                    )
                )

    # 3. Overfitting signal: step-aligned val_loss trending up after best step while train_loss falls
    val_series = _resolve_loss_series(run.metrics, kind="val")
    if (
        train_series is not None
        and val_series is not None
        and len(val_series.values) >= 6
        and len(train_series.values) >= 6
    ):
        best_val_idx = min(
            range(len(val_series.values)), key=lambda i: val_series.values[i]
        )
        tail_count = len(val_series.values) - 1 - best_val_idx
        best_val = val_series.values[best_val_idx]
        best_val_step = val_series.steps[best_val_idx]
        latest_val = val_series.values[-1]
        train_at_best = _value_at_or_before_step(train_series, best_val_step)
        train_falling = (
            train_at_best is not None and train_series.values[-1] < train_at_best * 0.95
        )
        if (
            tail_count >= 4
            and best_val > 1e-6
            and latest_val >= best_val * float(overfit_ratio)
            and train_falling
        ):
            findings.append(
                RunHealthFinding(
                    severity="warning",
                    code="overfitting",
                    message=(
                        f"Possible overfitting: val loss rose from {best_val:.4f} "
                        f"({run.step_unit} {best_val_step}) to {latest_val:.4f} "
                        "while train loss kept dropping"
                    ),
                )
            )

    # 4. Metric plateau detection
    min_plateau = max(3, int(plateau_steps))
    if run.primary_metric_name and run.primary_metric_name in run.metrics:
        p_series = run.metrics[run.primary_metric_name]
        if len(p_series.values) >= max(20, min_plateau + 5):
            best_idx = max(
                range(len(p_series.values)), key=lambda i: p_series.values[i]
            )
            stale_steps = len(p_series.values) - 1 - best_idx
            if stale_steps >= min_plateau:
                findings.append(
                    RunHealthFinding(
                        severity="info",
                        code="metric_plateau",
                        message=(
                            f"{run.primary_metric_name.split('/')[-1]} peaked at "
                            f"{p_series.values[best_idx]:.4f} ({run.step_unit} {p_series.steps[best_idx]}); "
                            f"no improvement in last {stale_steps} {run.step_unit}s"
                        ),
                    )
                )

    # 5. Missing checkpoints on completed runs
    if run.status == "finished" and not run.checkpoints:
        findings.append(
            RunHealthFinding(
                severity="warning",
                code="no_checkpoint",
                message="Run finished without a discoverable checkpoint (.pt/.pth/.onnx)",
            )
        )

    # 6. Low disk space warning
    warn_disk = float(low_disk_gb)
    if disk_free_gb is not None and disk_free_gb < warn_disk:
        findings.append(
            RunHealthFinding(
                severity="critical"
                if disk_free_gb < min(0.5, warn_disk * 0.25)
                else "warning",
                code="low_disk",
                message=f"Low free disk space on runs volume ({disk_free_gb:.2f} GB remaining)",
            )
        )

    return findings


__all__ = [
    "DiscoveredRun",
    "HparamDiffRow",
    "MetricSeries",
    "RUN_METADATA_FILENAME",
    "RUN_METADATA_SCHEMA_VERSION",
    "RunHealthFinding",
    "RunMetadata",
    "analyze_run_health",
    "build_tensorboard_logdir_spec",
    "canonicalize_run_dir",
    "compute_hparam_diff",
    "discover_training_runs",
    "extract_effective_hparams",
    "extract_total_steps",
    "find_checkpoints",
    "find_config_file",
    "find_plot_images",
    "find_results_csv",
    "find_tensorboard_dir",
    "find_train_log",
    "format_timeline_line",
    "format_timeline_lines",
    "has_projector_config",
    "inspect_run_dir",
    "load_run_metadata",
    "normalize_run_tags",
    "read_csv_rows",
    "read_run_config",
    "read_tensorboard_scalars",
    "safe_float",
    "safe_int",
    "save_run_metadata",
    "smooth_ema",
    "summarize_run_best_metrics",
    "tail_log_text",
]
