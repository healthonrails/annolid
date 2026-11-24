"""Lightweight non-blocking system & GPU telemetry sampler for training monitoring.

Collects host CPU, RAM, disk free space (for the runs root), and optional GPU
metrics (NVIDIA via ``nvidia-smi`` or Apple Silicon / PyTorch MPS status) using
only stdlib + optional ``psutil``. Never blocks the Qt UI thread.
"""

from __future__ import annotations

import collections
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple


@dataclass(frozen=True)
class TelemetrySample:
    """Single timestamped hardware utilization snapshot."""

    timestamp: float
    cpu_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    gpu_name: str = ""
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_mib: Optional[float] = None
    gpu_mem_total_mib: Optional[float] = None
    gpu_temp_c: Optional[float] = None

    @property
    def gpu_mem_percent(self) -> Optional[float]:
        if (
            self.gpu_mem_used_mib is not None
            and self.gpu_mem_total_mib is not None
            and self.gpu_mem_total_mib > 0
        ):
            return min(
                100.0,
                max(0.0, (self.gpu_mem_used_mib / self.gpu_mem_total_mib) * 100.0),
            )
        return None


def _parse_nvidia_smi_csv(
    stdout: str,
) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Parse ``nvidia-smi`` CSV lines and select the most active GPU when multi-GPU."""
    lines = [ln.strip() for ln in str(stdout or "").splitlines() if ln.strip()]
    if not lines:
        return "", None, None, None, None

    def _num(tok: str) -> Optional[float]:
        try:
            return float(tok)
        except Exception:
            return None

    parsed: List[
        Tuple[
            int, str, Optional[float], Optional[float], Optional[float], Optional[float]
        ]
    ] = []
    for gpu_idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            if parts and parts[0]:
                parsed.append((gpu_idx, parts[0], None, None, None, None))
            continue
        parsed.append(
            (
                gpu_idx,
                parts[0],
                _num(parts[1]),
                _num(parts[2]),
                _num(parts[3]),
                _num(parts[4]) if len(parts) >= 5 else None,
            )
        )
    if not parsed:
        return "", None, None, None, None

    chosen = max(
        parsed,
        key=lambda item: (
            item[2] if item[2] is not None else -1.0,
            item[3] if item[3] is not None else -1.0,
            -item[0],
        ),
    )
    gpu_idx, name, util, mem_used, mem_total, temp = chosen
    if len(parsed) > 1:
        name = f"{name} [GPU {gpu_idx}/{len(parsed)}]"
    return name, util, mem_used, mem_total, temp


def _probe_nvidia_gpu(
    timeout_s: float = 1.5,
) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Query NVIDIA GPU(s) via ``nvidia-smi`` CSV output."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", None, None, None, None
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0 or not proc.stdout:
            return "", None, None, None, None
        return _parse_nvidia_smi_csv(proc.stdout)
    except Exception:
        return "", None, None, None, None


def _detect_accelerator_label() -> str:
    """Fast non-blocking accelerator hint when nvidia-smi is unavailable."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "darwin" and machine in {"arm64", "aarch64"}:
        return "Apple Silicon (MPS / Unified RAM)"
    return ""


def collect_telemetry_sample(
    *,
    disk_path: Optional[Path] = None,
    probe_gpu: bool = True,
) -> TelemetrySample:
    """Capture one hardware snapshot synchronously (safe to call from worker thread)."""
    now = time.time()
    cpu_pct: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_pct: Optional[float] = None

    try:
        import psutil  # type: ignore

        cpu_pct = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        ram_used_gb = float(vm.used) / (1024.0**3)
        ram_total_gb = float(vm.total) / (1024.0**3)
        ram_pct = float(vm.percent)
    except Exception:
        pass

    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    try:
        target = Path(disk_path).expanduser().resolve() if disk_path else Path.cwd()
        while not target.exists() and target.parent != target:
            target = target.parent
        usage = shutil.disk_usage(str(target))
        disk_free_gb = float(usage.free) / (1024.0**3)
        disk_total_gb = float(usage.total) / (1024.0**3)
    except Exception:
        pass

    gpu_name = ""
    gpu_util: Optional[float] = None
    gpu_mem_used: Optional[float] = None
    gpu_mem_total: Optional[float] = None
    gpu_temp: Optional[float] = None

    if probe_gpu:
        gpu_name, gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = _probe_nvidia_gpu()
    if not gpu_name:
        gpu_name = _detect_accelerator_label()

    return TelemetrySample(
        timestamp=now,
        cpu_percent=cpu_pct,
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        ram_percent=ram_pct,
        disk_free_gb=disk_free_gb,
        disk_total_gb=disk_total_gb,
        gpu_name=gpu_name,
        gpu_util_percent=gpu_util,
        gpu_mem_used_mib=gpu_mem_used,
        gpu_mem_total_mib=gpu_mem_total,
        gpu_temp_c=gpu_temp,
    )


class SystemTelemetrySampler:
    """Thread-safe rolling hardware sampler whose GPU probes never block the GUI."""

    def __init__(
        self,
        *,
        max_samples: int = 120,
        disk_path: Optional[Path] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._history: Deque[TelemetrySample] = collections.deque(
            maxlen=max(10, int(max_samples))
        )
        self._disk_path = disk_path
        self._gpu_probe_in_flight = False
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self._last_gpu_snapshot: Tuple[
            str, Optional[float], Optional[float], Optional[float], Optional[float]
        ] = (
            "" if self._has_nvidia else _detect_accelerator_label(),
            None,
            None,
            None,
            None,
        )

        # Prime psutil cpu_percent counter so first real reading isn't always 0.0.
        try:
            import psutil  # type: ignore

            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def set_disk_path(self, path: Optional[Path]) -> None:
        with self._lock:
            self._disk_path = path

    def _trigger_async_gpu_probe(self) -> None:
        if not self._has_nvidia:
            return
        with self._lock:
            if self._gpu_probe_in_flight:
                return
            self._gpu_probe_in_flight = True

        def _worker() -> None:
            result: Tuple[
                str, Optional[float], Optional[float], Optional[float], Optional[float]
            ] = ("", None, None, None, None)
            try:
                result = _probe_nvidia_gpu(timeout_s=1.5)
            finally:
                with self._lock:
                    if result[0]:
                        self._last_gpu_snapshot = result
                    self._gpu_probe_in_flight = False

        threading.Thread(
            target=_worker, name="annolid-gpu-telemetry", daemon=True
        ).start()

    def sample_now(self) -> TelemetrySample:
        """Record a non-blocking hardware sample and trigger async GPU refresh."""
        with self._lock:
            disk_path = self._disk_path
            gpu_name, gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = (
                self._last_gpu_snapshot
            )
        self._trigger_async_gpu_probe()

        base = collect_telemetry_sample(disk_path=disk_path, probe_gpu=False)
        sample = TelemetrySample(
            timestamp=base.timestamp,
            cpu_percent=base.cpu_percent,
            ram_used_gb=base.ram_used_gb,
            ram_total_gb=base.ram_total_gb,
            ram_percent=base.ram_percent,
            disk_free_gb=base.disk_free_gb,
            disk_total_gb=base.disk_total_gb,
            gpu_name=gpu_name or base.gpu_name,
            gpu_util_percent=gpu_util,
            gpu_mem_used_mib=gpu_mem_used,
            gpu_mem_total_mib=gpu_mem_total,
            gpu_temp_c=gpu_temp,
        )
        with self._lock:
            self._history.append(sample)
        return sample

    def history(self) -> List[TelemetrySample]:
        with self._lock:
            return list(self._history)

    def latest(self) -> Optional[TelemetrySample]:
        with self._lock:
            return self._history[-1] if self._history else None


def format_telemetry_summary(sample: Optional[TelemetrySample]) -> str:
    """Format a compact single-line hardware readout for dashboard headers."""
    if sample is None:
        return "System: —"
    parts: List[str] = []
    if sample.cpu_percent is not None:
        parts.append(f"CPU {sample.cpu_percent:.0f}%")
    if sample.ram_used_gb is not None and sample.ram_total_gb is not None:
        parts.append(f"RAM {sample.ram_used_gb:.1f}/{sample.ram_total_gb:.0f} GB")
    if sample.gpu_util_percent is not None:
        gpu_str = f"GPU {sample.gpu_util_percent:.0f}%"
        if sample.gpu_mem_used_mib is not None and sample.gpu_mem_total_mib is not None:
            gpu_str += (
                f" ({sample.gpu_mem_used_mib / 1024.0:.1f}/"
                f"{sample.gpu_mem_total_mib / 1024.0:.1f} GB)"
            )
        if sample.gpu_temp_c is not None:
            gpu_str += f" {sample.gpu_temp_c:.0f}°C"
        parts.append(gpu_str)
    elif sample.gpu_name:
        parts.append(sample.gpu_name)
    if sample.disk_free_gb is not None:
        parts.append(f"Disk {sample.disk_free_gb:.1f} GB free")
    return " · ".join(parts) if parts else "System: —"


__all__ = [
    "SystemTelemetrySampler",
    "TelemetrySample",
    "collect_telemetry_sample",
    "format_telemetry_summary",
]
