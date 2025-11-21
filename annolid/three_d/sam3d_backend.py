"""Thin wrapper around SAM 3D Objects inference for Annolid.

This module is intentionally dependency-light and only imports the heavy SAM 3D
dependencies when `Sam3DBackend` is instantiated. It can be used within the
Annolid process or by a separate runner CLI.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from annolid.utils.logger import logger


@dataclass
class Sam3DConfig:
    """Runtime configuration for SAM 3D Objects."""

    repo_path: Path = field(default_factory=lambda: Path("sam-3d-objects"))
    checkpoints_dir: Optional[Path] = None
    checkpoint_tag: str = "hf"
    compile: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.repo_path, Path):
            self.repo_path = Path(self.repo_path)
        if self.checkpoints_dir is not None and not isinstance(
            self.checkpoints_dir, Path
        ):
            self.checkpoints_dir = Path(self.checkpoints_dir)

    def pipeline_path(self) -> Path:
        base = self.checkpoints_dir or (self.repo_path / "checkpoints")
        return base / self.checkpoint_tag / "pipeline.yaml"


@dataclass
class Sam3DResult:
    ply_path: Path
    sidecar_path: Path
    duration_s: float
    metadata: Dict[str, Any]


class Sam3DBackendError(RuntimeError):
    """Raised when SAM 3D inference fails."""


class Sam3DBackend:
    """Lazy SAM 3D Objects runner."""

    def __init__(self, cfg: Sam3DConfig):
        self.cfg = cfg
        self._inference = None

    # Public API -----------------------------------------------------
    def run_single(
        self,
        image_rgb: np.ndarray,
        mask_bool: np.ndarray,
        *,
        output_dir: Path,
        basename: str = "sam3d_object",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Sam3DResult:
        """Run SAM 3D on a single mask and save a PLY + sidecar JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply_path = output_dir / f"{basename}.ply"
        sidecar_path = output_dir / f"{basename}.json"

        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise Sam3DBackendError("image_rgb must be HxWx3 uint8")
        mask_bool = np.asarray(mask_bool).astype(bool)
        if mask_bool.shape[:2] != image_rgb.shape[:2]:
            raise Sam3DBackendError("mask and image size mismatch")

        inference = self._load_inference()
        start = time.time()
        try:
            output = inference(image_rgb, mask_bool, seed=self.cfg.seed)
        except Exception as exc:  # pragma: no cover - depends on external repo
            raise Sam3DBackendError(f"SAM 3D inference failed: {exc}") from exc
        duration = time.time() - start

        if not output or "gs" not in output:
            raise Sam3DBackendError("SAM 3D output missing gaussian splat")
        try:
            output["gs"].save_ply(str(ply_path))
        except Exception as exc:  # pragma: no cover - external writer
            raise Sam3DBackendError(f"Failed to save PLY: {exc}") from exc

        sidecar = {
            "ply_path": str(ply_path),
            "checkpoint_tag": self.cfg.checkpoint_tag,
            "pipeline": str(self.cfg.pipeline_path()),
            "repo_path": str(self.cfg.repo_path),
            "duration_s": duration,
            "seed": self.cfg.seed,
        }
        if extra_metadata:
            sidecar.update(extra_metadata)
        try:
            sidecar_path.write_text(json.dumps(
                sidecar, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem
            logger.warning("Unable to write SAM3D sidecar: %s", exc)

        return Sam3DResult(
            ply_path=ply_path,
            sidecar_path=sidecar_path,
            duration_s=duration,
            metadata=sidecar,
        )

    # Internal helpers ----------------------------------------------
    def _load_inference(self):
        """Import SAM3D inference lazily to keep Annolid light."""
        if self._inference is not None:
            return self._inference

        pipeline = self.cfg.pipeline_path()
        if not pipeline.exists():
            raise Sam3DBackendError(
                f"SAM 3D pipeline config not found: {pipeline}"
            )

        notebook_path = self.cfg.repo_path / "notebook"
        if not notebook_path.exists():
            raise Sam3DBackendError(
                f"SAM 3D notebook directory missing: {notebook_path}"
            )

        # Ensure both repo root (for utils3d, etc.) and notebook are importable
        repo_path = self.cfg.repo_path.resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        if str(notebook_path) not in sys.path:
            sys.path.insert(0, str(notebook_path))
        try:
            import inference as sam3d_inference  # type: ignore
        except Exception as exc:  # pragma: no cover - external dependency
            raise Sam3DBackendError(
                "Unable to import SAM 3D inference. "
                "Verify the SAM 3D repo is present, installed (pip install -e .), "
                f"and PYTHONPATH includes {repo_path}. Original error: {exc}"
            ) from exc

        # Compatibility: some older builds may miss run_layout_model. Provide a no-op.
        try:
            import sam3d_objects.pipeline.inference_pipeline_pointmap as _ipp  # type: ignore

            if not hasattr(_ipp.InferencePipelinePointMap, "run_layout_model"):  # pragma: no cover - compat
                def _run_layout_model_stub(self, *_args, **_kwargs):
                    return {}
                logger.warning(
                    "SAM 3D InferencePipelinePointMap missing run_layout_model; "
                    "install/upgrade SAM 3D to avoid degraded output."
                )
                setattr(_ipp.InferencePipelinePointMap,
                        "run_layout_model", _run_layout_model_stub)
        except Exception:
            # Non-fatal; continue and let initialization raise if needed
            pass

        try:
            self._inference = sam3d_inference.Inference(
                str(pipeline),
                compile=self.cfg.compile,
            )
            # Safety: ensure pipeline has run_layout_model
            try:
                pipeline_obj = getattr(self._inference, "_pipeline", None)
                if pipeline_obj and not hasattr(pipeline_obj, "run_layout_model"):  # pragma: no cover - compat
                    def _run_layout_model_stub(*_args, **_kwargs):
                        return {}
                    logger.warning(
                        "SAM 3D pipeline missing run_layout_model; continuing without layout postprocess."
                    )
                    setattr(pipeline_obj, "run_layout_model",
                            _run_layout_model_stub)
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - external dependency
            raise Sam3DBackendError(
                f"Failed to initialize SAM 3D: {exc}") from exc
        return self._inference
