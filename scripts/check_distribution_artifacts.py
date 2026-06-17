#!/usr/bin/env python3
"""Reject large/generated model artifacts in release distributions.

This complements ``annolid.spec``: PyInstaller bundles and Python source/wheel
archives must both stay lean and must not ship local checkpoints, generated
runs, or exported model runtimes.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Literal


ArtifactKind = Literal["distribution", "bundle"]


FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".h5",
    ".mlmodel",
    ".onnx",
    ".pth",
    ".pt",
    ".safetensors",
    ".tflite",
    ".weights",
}

FORBIDDEN_NAMES = {
    ".cache",
    "__pycache__",
    "checkpoints",
    "mlpackage",
    "weights",
}

FORBIDDEN_FRAGMENTS = {
    "annolid/depth/checkpoints",
    "annolid/detector/countgd/checkpoints",
    "annolid/realtime/models",
    "annolid/realtime/runs",
    "annolid/realtime/yolo11n",
    "annolid/segmentation/MEDIAR/weights",
    "annolid/segmentation/SAM/segment-anything-2/.github",
    "annolid/segmentation/SAM/segment-anything-2/SAM_2.egg-info",
    "annolid/segmentation/SAM/segment-anything-2/checkpoints",
    "annolid/segmentation/SAM/segment-anything-2/demo",
    "annolid/segmentation/SAM/segment-anything-2/notebooks",
    "annolid/segmentation/SAM/segment-anything-2/training",
    "annolid/segmentation/cutie_vos/weights",
    "openvino_model",
}

BUNDLE_FORBIDDEN_NAMES = FORBIDDEN_NAMES | {
    "diffusers",
    "h5py",
    "matplotlib",
    "numba",
    "onnxruntime",
    "pandas",
    "runs",
    "scipy",
    "skimage",
    "sklearn",
    "tensorboard",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
}

BUNDLE_FORBIDDEN_FRAGMENTS = FORBIDDEN_FRAGMENTS | {
    "matplotlib",
    "onnxruntime",
    "pandas",
    "scikit_image",
    "scikit_learn",
    "scipy",
    "tensorboard",
    "torchvision",
}


def _normalize_archive_member(name: str) -> str:
    return str(name or "").replace("\\", "/").lstrip("./")


def _strip_archive_root(name: str) -> str:
    normalized = _normalize_archive_member(name)
    parts = normalized.split("/")
    if parts and parts[0].startswith("annolid-"):
        return "/".join(parts[1:])
    return normalized


def is_forbidden_member(
    name: str, artifact_kind: ArtifactKind = "distribution"
) -> bool:
    normalized = _normalize_archive_member(name)
    without_root = _strip_archive_root(normalized)
    parts = {part for part in without_root.split("/") if part}
    suffix = Path(without_root).suffix.lower()
    forbidden_names = (
        BUNDLE_FORBIDDEN_NAMES if artifact_kind == "bundle" else FORBIDDEN_NAMES
    )
    forbidden_fragments = (
        BUNDLE_FORBIDDEN_FRAGMENTS if artifact_kind == "bundle" else FORBIDDEN_FRAGMENTS
    )
    if suffix in FORBIDDEN_SUFFIXES:
        return True
    if parts.intersection(forbidden_names):
        return True
    return any(fragment in without_root for fragment in forbidden_fragments)


def iter_distribution_members(path: Path) -> Iterable[str]:
    if path.is_dir():
        for item in path.rglob("*"):
            yield item.relative_to(path).as_posix()
        return

    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffixes == ".tar.gz" or path.suffix.lower() in {".tgz", ".tar"}:
        with tarfile.open(path, "r:*") as archive:
            yield from archive.getnames()
        return

    if path.suffix.lower() in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            yield from archive.namelist()
        return

    raise ValueError(f"Unsupported distribution artifact: {path}")


def find_forbidden_members(
    paths: Iterable[Path], artifact_kind: ArtifactKind = "distribution"
) -> list[str]:
    matches: list[str] = []
    for path in paths:
        for member in iter_distribution_members(path):
            if is_forbidden_member(member, artifact_kind=artifact_kind):
                matches.append(f"{path}: {member}")
    return sorted(matches)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject model/checkpoint/runtime artifacts in distributions."
    )
    parser.add_argument(
        "paths", nargs="+", help="Distribution archives or directories."
    )
    parser.add_argument(
        "--kind",
        choices=("distribution", "bundle"),
        default="distribution",
        help=(
            "Artifact policy to apply. Use 'bundle' for frozen desktop app "
            "folders/archives that must exclude heavyweight runtimes."
        ),
    )
    args = parser.parse_args(argv)

    paths = [Path(item).expanduser() for item in args.paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print("Missing distribution artifact(s):", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 2

    matches = find_forbidden_members(paths, artifact_kind=args.kind)
    if matches:
        print("Forbidden model/runtime artifacts found:", file=sys.stderr)
        print("\n".join(matches[:100]), file=sys.stderr)
        if len(matches) > 100:
            print(f"... and {len(matches) - 100} more", file=sys.stderr)
        return 1

    print("Distribution artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
