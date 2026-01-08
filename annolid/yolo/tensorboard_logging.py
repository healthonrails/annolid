from __future__ import annotations

import csv
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Protocol, Tuple


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float,
                   global_step: int) -> None: ...

    def add_text(self, tag: str, text_string: str,
                 global_step: int) -> None: ...
    def add_image(self, tag: str, img_tensor: Any,
                  global_step: int, dataformats: str = "HWC") -> None: ...

    def flush(self) -> None: ...
    def close(self) -> None: ...


def _try_create_summary_writer(log_dir: Path) -> Optional[ScalarWriter]:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


_TAG_SAFE = re.compile(r"[^a-zA-Z0-9._/-]+")


def sanitize_tensorboard_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    tag = tag.replace(" ", "_")
    tag = tag.replace("(", "_").replace(")", "_")
    tag = tag.replace("[", "_").replace("]", "_")
    tag = tag.replace("{", "_").replace("}", "_")
    tag = _TAG_SAFE.sub("_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag or "metric"


def _find_results_csv(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "results.csv"
    if direct.exists():
        return direct
    for candidate in run_dir.rglob("results.csv"):
        return candidate
    return None


def _read_last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            last: Optional[Dict[str, str]] = None
            for row in reader:
                if row:
                    last = row
            return last
    except Exception:
        return None


def _summarize_run_tree(run_dir: Path, *, max_files: int = 2000) -> str:
    lines = [f"Run dir: {run_dir}"]
    if not run_dir.exists():
        return "\n".join(lines + ["(missing)"])

    count = 0
    for path in sorted(run_dir.rglob("*")):
        if count >= max_files:
            lines.append(f"... ({count} files shown; truncated)")
            break
        try:
            rel = path.relative_to(run_dir)
        except Exception:
            rel = path
        try:
            if path.is_dir():
                continue
            size = path.stat().st_size
        except Exception:
            size = -1
        suffix = "" if size < 0 else f" ({size} bytes)"
        lines.append(f"- {rel}{suffix}")
        count += 1
    return "\n".join(lines)


@dataclass
class YOLORunTBConfig:
    poll_interval_s: float = 2.0
    log_tail_lines: int = 200
    text_refresh_s: float = 60.0
    images_refresh_s: float = 30.0
    max_images_per_poll: int = 12
    max_image_edge_px: int = 1280


class YOLORunTensorBoardLogger:
    """Augment an Ultralytics run dir with TensorBoard-friendly metrics and text summaries."""

    def __init__(
        self,
        *,
        run_dir: Path,
        writer: Optional[ScalarWriter] = None,
        config: Optional[YOLORunTBConfig] = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self._writer = writer
        self._config = config or YOLORunTBConfig()
        self._last_epoch_logged: Optional[int] = None
        self._output_tail: Deque[str] = deque(
            maxlen=int(self._config.log_tail_lines))
        self._last_text_write_at: float = 0.0
        self._last_images_write_at: float = 0.0
        self._image_mtime_by_path: Dict[str, float] = {}

    @property
    def writer(self) -> Optional[ScalarWriter]:
        if self._writer is not None:
            return self._writer
        self._writer = _try_create_summary_writer(self.run_dir)
        return self._writer

    def record_output_line(self, line: str) -> None:
        s = str(line or "").rstrip("\n")
        if s:
            self._output_tail.append(s)

    def write_static_metadata(
        self,
        *,
        command: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        w = self.writer
        if w is None:
            return
        if command:
            w.add_text("annolid/command", f"```\n{command}\n```", 0)
        if hparams:
            lines = ["Hyperparameters:"]
            for k in sorted(hparams.keys()):
                lines.append(f"- {k}: {hparams[k]}")
            w.add_text("annolid/hparams", "\n".join(lines), 0)
        w.flush()

    def poll_and_log_metrics(self) -> Tuple[bool, Optional[int]]:
        """Read results.csv and write scalar summaries when a new epoch is observed."""
        w = self.writer
        if w is None:
            return False, None
        results_csv = _find_results_csv(self.run_dir)
        if results_csv is None:
            return False, None
        row = _read_last_csv_row(results_csv)
        if not row:
            return False, None

        epoch = _safe_int(row.get("epoch") or row.get("Epoch"))
        if epoch is None:
            return False, None
        if self._last_epoch_logged is not None and epoch <= self._last_epoch_logged:
            return False, epoch

        scalars_logged = 0
        for key, value in row.items():
            if key is None:
                continue
            if key.lower() in {"epoch"}:
                continue
            f = _safe_float(value)
            if f is None:
                continue
            tag = sanitize_tensorboard_tag(str(key))
            w.add_scalar(f"ultralytics/{tag}", float(f), int(epoch))
            scalars_logged += 1

        w.add_scalar("annolid/progress/epoch", float(epoch), int(epoch))
        self._last_epoch_logged = int(epoch)

        now = time.time()
        if now - self._last_text_write_at >= float(self._config.text_refresh_s):
            self._last_text_write_at = now
            self._write_log_tail(step=int(epoch))
            w.flush()

        return bool(scalars_logged), int(epoch)

    def poll_and_log_images(self, *, step: Optional[int] = None) -> int:
        """Scan the run directory for plot images and emit them to TensorBoard."""
        w = self.writer
        if w is None or not hasattr(w, "add_image"):
            return 0

        now = time.time()
        if now - self._last_images_write_at < float(self._config.images_refresh_s):
            return 0
        self._last_images_write_at = now

        step_i = int(step if step is not None else (
            self._last_epoch_logged or 0))
        written = 0
        for path in self._discover_plot_images():
            if written >= int(self._config.max_images_per_poll):
                break
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                continue
            key = str(path.resolve())
            if self._image_mtime_by_path.get(key) == mtime:
                continue
            arr = _read_image_rgb(
                path, max_edge_px=int(self._config.max_image_edge_px))
            if arr is None:
                continue
            try:
                rel = path.relative_to(self.run_dir)
            except Exception:
                rel = path.name
            tag = sanitize_tensorboard_tag(f"annolid/images/{rel}")
            try:
                w.add_image(tag, arr, step_i, dataformats="HWC")
            except Exception:
                continue
            self._image_mtime_by_path[key] = mtime
            written += 1

        if written:
            try:
                w.flush()
            except Exception:
                pass
        return written

    def _write_log_tail(self, *, step: int) -> None:
        w = self.writer
        if w is None:
            return
        if not self._output_tail:
            return
        tail = "\n".join(self._output_tail)
        w.add_text("annolid/train_log_tail", f"```\n{tail}\n```", int(step))

    def finalize(self, *, ok: bool, error: Optional[str] = None) -> None:
        w = self.writer
        if w is None:
            return
        step = int(self._last_epoch_logged or 0)
        self._write_log_tail(step=step)
        self.poll_and_log_images(step=step)
        w.add_text("annolid/run_tree",
                   f"```\n{_summarize_run_tree(self.run_dir)}\n```", step)
        w.add_text("annolid/status", "ok" if ok else "failed", step)
        if error:
            w.add_text("annolid/error", f"```\n{error}\n```", step)
        w.flush()

    def close(self) -> None:
        w = self.writer
        if w is None:
            return
        try:
            w.flush()
        except Exception:
            pass
        try:
            w.close()
        except Exception:
            pass

    def _discover_plot_images(self) -> list[Path]:
        patterns = [
            "results.png",
            "confusion_matrix*.png",
            "*_curve.png",
            "labels*.jpg",
            "labels*.png",
            "train_batch*.jpg",
            "train_batch*.png",
            "val_batch*.jpg",
            "val_batch*.png",
        ]
        out: list[Path] = []
        seen: set[str] = set()
        for pat in patterns:
            for path in sorted(self.run_dir.rglob(pat)):
                try:
                    if not path.is_file():
                        continue
                except Exception:
                    continue
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                out.append(path)

        def key_fn(p: Path) -> float:
            try:
                return float(p.stat().st_mtime)
            except Exception:
                return 0.0

        out.sort(key=key_fn, reverse=True)
        return out


def _read_image_rgb(path: Path, *, max_edge_px: int = 1280) -> Optional[Any]:
    """Read an image into an RGB uint8 HWC array for TensorBoard."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    arr: Optional[Any] = None
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        arr = None

    if arr is None:
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
        except Exception:
            return None

    try:
        h, w = int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        return None

    edge = max(h, w)
    if max_edge_px and edge > int(max_edge_px):
        scale = float(max_edge_px) / float(edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2  # type: ignore

            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            try:
                from PIL import Image  # type: ignore

                im = Image.fromarray(arr)
                im = im.resize((new_w, new_h))
                arr = np.asarray(im)
            except Exception:
                pass

    if getattr(arr, "dtype", None) != np.uint8:
        try:
            arr = arr.astype(np.uint8, copy=False)
        except Exception:
            return None
    return arr
