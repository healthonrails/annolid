"""Client utilities to run SAM 3D Objects from Annolid.

Supports two modes:
- In-process: uses the current Python environment (requires SAM 3D installed).
- Subprocess: calls a separate Python executable (e.g., a dedicated conda env)
  via the runner CLI and passes image/mask paths through a JSON spec.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from annolid.three_d.sam3d_backend import (
    Sam3DBackend,
    Sam3DBackendError,
    Sam3DConfig,
    Sam3DResult,
)
from annolid.utils.logger import logger


@dataclass
class Sam3DClientConfig(Sam3DConfig):
    python_executable: Optional[str] = None
    timeout_s: Optional[int] = None


@dataclass
class Sam3DAvailability:
    ok: bool
    reason: Optional[str] = None
    mode: str = "inprocess"


@dataclass
class Sam3DJobSpec:
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    image_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    basename: str = "sam3d_object"
    metadata: Dict[str, Any] = field(default_factory=dict)


def sam3d_available(cfg: Sam3DClientConfig, mode: str = "auto") -> Sam3DAvailability:
    """Probe whether SAM 3D can be run (either in-process or via subprocess)."""
    pipeline = cfg.pipeline_path()
    if not pipeline.exists():
        return Sam3DAvailability(
            ok=False,
            reason=f"Missing pipeline config: {pipeline}",
            mode=mode,
        )
    notebook_dir = cfg.repo_path / "notebook"
    if not notebook_dir.exists():
        return Sam3DAvailability(
            ok=False,
            reason=f"Missing SAM 3D notebook dir: {notebook_dir}",
            mode=mode,
        )

    if cfg.python_executable:
        probe = _probe_subprocess(cfg)
        return Sam3DAvailability(ok=probe[0], reason=probe[1], mode="subprocess")

    # In-process probe: import mildly
    try:
        if str(notebook_dir) not in sys.path:
            sys.path.insert(0, str(notebook_dir))
        importlib.import_module("inference")  # type: ignore
        return Sam3DAvailability(ok=True, mode="inprocess")
    except Exception as exc:  # pragma: no cover - probe depends on external pkg
        return Sam3DAvailability(ok=False, reason=str(exc), mode="inprocess")


def _probe_subprocess(cfg: Sam3DClientConfig) -> tuple[bool, Optional[str]]:
    exe = cfg.python_executable
    if not exe:
        return False, "python_executable not set"
    if not Path(exe).exists():
        return False, f"python executable not found: {exe}"
    cmd = [exe, "-m", "annolid.three_d.sam3d_runner_cli", "--probe"]
    env = _runner_env()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s or 15,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)
    if proc.returncode != 0:
        return False, proc.stdout.strip() or proc.stderr.strip() or "probe failed"
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
        if payload.get("ok"):
            return True, None
        return False, payload.get("error")
    except Exception as exc:  # pragma: no cover
        return False, f"probe parse failed: {exc}"


def run_inprocess(cfg: Sam3DClientConfig, job: Sam3DJobSpec) -> Sam3DResult:
    """Run SAM 3D inside the current Python environment."""
    if job.image is None or job.mask is None:
        raise Sam3DBackendError(
            "image and mask arrays are required for in-process run")
    output_dir = job.output_dir or Path(
        tempfile.mkdtemp(prefix="annolid_sam3d_"))
    backend = Sam3DBackend(cfg)
    return backend.run_single(
        image_rgb=job.image,
        mask_bool=job.mask,
        output_dir=output_dir,
        basename=job.basename,
        extra_metadata=job.metadata,
    )


def run_subprocess(cfg: Sam3DClientConfig, job: Sam3DJobSpec) -> Sam3DResult:
    """Run SAM 3D in a separate Python environment via the runner CLI."""
    exe = cfg.python_executable or sys.executable
    if not Path(exe).exists():
        raise Sam3DBackendError(f"python executable not found: {exe}")

    with tempfile.TemporaryDirectory(prefix="annolid_sam3d_job_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_path = job.image_path or (tmpdir_path / "image.png")
        mask_path = job.mask_path or (tmpdir_path / "mask.png")
        if job.image is not None:
            Image.fromarray(np.asarray(
                job.image, dtype=np.uint8)).save(image_path)
        if job.mask is not None:
            mask_u8 = (np.asarray(job.mask).astype(
                np.uint8) * 255).clip(0, 255)
            Image.fromarray(mask_u8).save(mask_path)

        output_dir = job.output_dir or Path(
            tempfile.mkdtemp(prefix="annolid_sam3d_out_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_path = tmpdir_path / "spec.json"
        spec_payload = {
            "image": str(image_path),
            "mask": str(mask_path),
            "repo_path": str(cfg.repo_path),
            "checkpoints_dir": str(cfg.checkpoints_dir or cfg.repo_path / "checkpoints"),
            "checkpoint_tag": cfg.checkpoint_tag,
            "compile": cfg.compile,
            "seed": cfg.seed,
            "output_dir": str(output_dir),
            "basename": job.basename,
            "metadata": job.metadata,
        }
        spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")

        cmd = [exe, "-m", "annolid.three_d.sam3d_runner_cli",
               "--spec", str(spec_path)]
        env = _runner_env()
        logger.info("Launching SAM 3D subprocess: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s or 3600,
            env=env,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            raise Sam3DBackendError(
                f"SAM3D subprocess failed ({proc.returncode}): {stdout or stderr}"
            )
        try:
            payload = json.loads(stdout or "{}")
        except Exception as exc:
            raise Sam3DBackendError(
                f"Failed to parse subprocess output: {exc}") from exc
        if not payload.get("ok"):
            raise Sam3DBackendError(payload.get(
                "error") or "Unknown SAM3D error")
        result = payload.get("result") or {}
        return Sam3DResult(
            ply_path=Path(result["ply"]),
            sidecar_path=Path(result["sidecar"]),
            duration_s=float(result.get("duration_s", 0.0)),
            metadata=result,
        )


def _runner_env() -> Dict[str, str]:
    """Provide PYTHONPATH so runner can import Annolid from source."""
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    )
    return env
