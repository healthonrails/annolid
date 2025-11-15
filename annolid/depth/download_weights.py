"""Download Video-Depth-Anything checkpoints on demand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from huggingface_hub import hf_hub_download

MODEL_SPECS: Mapping[str, Mapping[str, str]] = {
    "video_depth_anything_vits": {
        "repo_id": "depth-anything/Video-Depth-Anything-Small",
        "filename": "video_depth_anything_vits.pth",
    },
    "video_depth_anything_vitb": {
        "repo_id": "depth-anything/Video-Depth-Anything-Base",
        "filename": "video_depth_anything_vitb.pth",
    },
    "video_depth_anything_vitl": {
        "repo_id": "depth-anything/Video-Depth-Anything-Large",
        "filename": "video_depth_anything_vitl.pth",
    },
    "metric_video_depth_anything_vits": {
        "repo_id": "depth-anything/Metric-Video-Depth-Anything-Small",
        "filename": "metric_video_depth_anything_vits.pth",
    },
    "metric_video_depth_anything_vitb": {
        "repo_id": "depth-anything/Metric-Video-Depth-Anything-Base",
        "filename": "metric_video_depth_anything_vitb.pth",
    },
    "metric_video_depth_anything_vitl": {
        "repo_id": "depth-anything/Metric-Video-Depth-Anything-Large",
        "filename": "metric_video_depth_anything_vitl.pth",
    },
}


def _download_checkpoint(repo_id: str, filename: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Skipping {target.name} because it already exists.")
        return

    try:
        output_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
            repo_type="model",
            revision="main",
            cache_dir=None,
        )
        if Path(output_path) != target:
            Path(output_path).rename(target)
    except Exception as exc:  # pragma: no cover - runtime failure
        raise RuntimeError(f"Failed to download {filename} from {repo_id}") from exc


def ensure_checkpoints(models: Sequence[str], *, dest: Optional[Path] = None) -> None:
    """Download the requested checkpoints if they are missing."""
    if dest is None:
        dest = Path(__file__).resolve().parent / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)

    invalid = [m for m in models if m not in MODEL_SPECS]
    if invalid:
        raise ValueError(
            f"Unknown model(s): {', '.join(invalid)}. Available: {', '.join(MODEL_SPECS)}"
        )

    for model in models:
        spec = MODEL_SPECS[model]
        checkpoint_path = dest / f"{model}.pth"
        if checkpoint_path.exists():
            print(f"Already have {model}.pth")
            continue
        print(f"Downloading {model}...")
        _download_checkpoint(spec["repo_id"], spec["filename"], checkpoint_path)


def _normalize_selection(items: Sequence[str]) -> Sequence[str]:
    lower = {name.lower(): name for name in MODEL_SPECS}
    selected = []
    for entry in items:
        key = entry.lower()
        if key not in lower:
            raise ValueError(f"Unknown model {entry!r}. Available: {', '.join(MODEL_SPECS)}")
        selected.append(lower[key])
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Video-Depth-Anything checkpoints.")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        help="Checkpoint to download (can be repeated). Case insensitive.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all checkpoints (only missing files are fetched).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints and exit.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory to save checkpoints.",
    )

    args = parser.parse_args(argv)
    if args.list:
        print("Available checkpoints:")
        for name in MODEL_SPECS:
            print(f"  - {name}")
        return 0

    if args.all:
        selected = list(MODEL_SPECS)
    else:
        if not args.model:
            parser.error("No model requested. Use --model or --all.")
        try:
            selected = _normalize_selection(args.model)
        except ValueError as exc:
            parser.error(str(exc))

    ensure_checkpoints(selected, dest=args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
