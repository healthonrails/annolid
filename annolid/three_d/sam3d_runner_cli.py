"""CLI entrypoint to run SAM 3D Objects in a dedicated environment.

Usage:
    python -m annolid.three_d.sam3d_runner_cli --spec /path/to/spec.json

The spec JSON should include:
{
  "image": ".../frame.png",
  "mask": ".../mask.png",
  "repo_path": ".../sam-3d-objects",
  "checkpoints_dir": ".../sam-3d-objects/checkpoints",
  "checkpoint_tag": "hf",
  "output_dir": "...",
  "basename": "example",
  "seed": 42,
  "compile": false,
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from annolid.three_d.sam3d_backend import (
    Sam3DBackend,
    Sam3DBackendError,
    Sam3DConfig,
)


def _read_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    return np.array(img)


def _load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_from_spec(spec_path: Path) -> Dict[str, Any]:
    spec = _load_spec(spec_path)
    repo_path = Path(spec.get("repo_path", "sam-3d-objects"))
    checkpoints_dir = spec.get("checkpoints_dir")
    cfg = Sam3DConfig(
        repo_path=repo_path,
        checkpoints_dir=Path(checkpoints_dir)
        if checkpoints_dir
        else None,
        checkpoint_tag=spec.get("checkpoint_tag", "hf"),
        compile=bool(spec.get("compile", False)),
        seed=spec.get("seed"),
    )

    output_dir = Path(spec.get("output_dir", "."))
    basename = spec.get("basename", "sam3d_object")
    metadata: Optional[Dict[str, Any]] = spec.get("metadata") or {}

    image_path = Path(spec["image"])
    mask_path = Path(spec["mask"])
    image_arr = _read_image(image_path)
    mask_arr = _read_image(mask_path)
    mask_bool = np.asarray(mask_arr).astype(bool)

    backend = Sam3DBackend(cfg)
    result = backend.run_single(
        image_rgb=image_arr,
        mask_bool=mask_bool,
        output_dir=output_dir,
        basename=basename,
        extra_metadata=metadata,
    )
    return {
        "ply": str(result.ply_path),
        "sidecar": str(result.sidecar_path),
        "duration_s": result.duration_s,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run SAM 3D Objects once.")
    parser.add_argument(
        "--spec",
        required=False,
        help="Path to JSON spec describing image/mask and config.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Import-time probe; exits 0 if imports succeed.",
    )
    args = parser.parse_args(argv)

    if args.probe:
        try:
            from annolid.three_d.sam3d_backend import Sam3DBackend  # noqa: F401
        except Exception as exc:  # pragma: no cover - probe path
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1
        print(json.dumps({"ok": True}))
        return 0

    if not args.spec:
        parser.error("--spec is required unless --probe is used")
    spec_path = Path(args.spec)
    try:
        result = _run_from_spec(spec_path)
        print(json.dumps({"ok": True, "result": result}))
        return 0
    except (OSError, Sam3DBackendError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
