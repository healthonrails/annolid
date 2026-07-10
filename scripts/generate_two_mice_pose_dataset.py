#!/usr/bin/env python3
"""Generate a deterministic two-mouse pose and keypoint-tracking fixture."""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "two_mice_pose_tracking_tiny"
SCENE_PATH = "annolid/gui/assets/threejs/two_mice.html"


class _QuietRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _serve_repo() -> Iterator[str]:
    handler = partial(_QuietRequestHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _subset(sequence: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        "info": sequence["info"],
        "licenses": sequence["licenses"],
        "videos": sequence["videos"],
        "images": [image for image in sequence["images"] if image["id"] in image_ids],
        "annotations": [
            annotation
            for annotation in sequence["annotations"]
            if annotation["image_id"] in image_ids
        ],
        "categories": sequence["categories"],
    }


def _write_support_files(
    output: Path,
    *,
    frame_count: int,
    train_frames: int,
    fps: int,
    seed: int,
    width: int,
    height: int,
) -> None:
    keypoint_names = [
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "spine_1",
        "spine_2",
        "tail_base",
        "left_forepaw",
        "right_forepaw",
        "left_hindpaw",
        "right_hindpaw",
        "tail_mid",
        "tail_tip",
    ]
    spec_lines = [
        "format: coco",
        "path: .",
        "image_root: .",
        "train: annotations/train.json",
        "val: annotations/val.json",
        "kpt_shape: [13, 3]",
        "keypoint_names:",
        *[f"  - {name}" for name in keypoint_names],
        "",
    ]
    (output / "coco_spec.yaml").write_text("\n".join(spec_lines), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generator": "scripts/generate_two_mice_pose_dataset.py",
        "scene": SCENE_PATH,
        "seed": seed,
        "fps": fps,
        "frame_count": frame_count,
        "train_frames": train_frames,
        "validation_frames": frame_count - train_frames,
        "width": width,
        "height": height,
        "keypoint_count": len(keypoint_names),
        "track_ids": [1, 2],
    }
    _write_json(output / "manifest.json", manifest)

    readme = f"""# Two-Mouse Pose and Tracking Fixture

This deterministic synthetic fixture contains {frame_count} sequential {width}x{height}
JPEG frames at {fps} FPS. Every frame has two COCO keypoint annotations with stable
`track_id` values (`1` and `2`) and 13 anatomical landmarks.

Files:

- `annotations/sequence.json`: the uninterrupted sequence for tracking tests.
- `annotations/train.json`: frames 0-{train_frames - 1} for pose-loader tests.
- `annotations/val.json`: frames {train_frames}-{frame_count - 1} for validation tests.
- `coco_spec.yaml`: Annolid COCO keypoint dataset configuration.
- `manifest.json`: deterministic generation parameters.

COCO visibility is `2` for a landmark projected inside the image and `0` outside.
Labels are geometric ground truth and do not attempt pixel-level self-occlusion
classification. The nonstandard `video_id`, `frame_id`, and `track_id` fields provide
the temporal identity contract used by keypoint-tracking tests.

Each image record also carries contact-quality ground truth:
`minimum_subject_clearance`, `subject_overlap`, `minimum_tail_clearance`,
`tail_overlap`, `maximum_foot_slip`, and `maximum_paw_ground_error`. These fields
guard against body/tail intersections and stance sliding in generated sequences.

Regenerate from the repository root:

```bash
source .venv/bin/activate
python scripts/generate_two_mice_pose_dataset.py
```

Generation seed: `{seed}`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def generate_dataset(
    output: Path,
    *,
    frame_count: int,
    train_frames: int,
    fps: int,
    seed: int,
    width: int,
    height: int,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required. Activate the repository .venv before running."
        ) from exc

    images_dir = output / "images"
    annotations_dir = output / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in images_dir.glob("frame_*.jpg"):
        stale_image.unlink()

    frames: list[dict[str, Any]] = []
    browser_errors: list[str] = []
    with _serve_repo() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": width, "height": height}, device_scale_factor=1
        )
        page.on(
            "console",
            lambda message: browser_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        query = urlencode({"dataset": 1, "frame": 0, "fps": fps, "seed": seed})
        page.goto(
            f"{base_url}/{SCENE_PATH}?{query}", wait_until="load", timeout=120_000
        )
        page.wait_for_selector("#c[data-dataset-ready='true']", timeout=120_000)
        for frame_index in range(frame_count):
            frame_payload = page.evaluate(
                "frameIndex => window.annolidPoseDataset.setFrame(frameIndex)",
                frame_index,
            )
            if frame_payload["image"]["frame_id"] != frame_index:
                raise RuntimeError(
                    f"Scene returned the wrong frame for index {frame_index}"
                )
            image_path = output / frame_payload["image"]["file_name"]
            page.screenshot(path=str(image_path), type="jpeg", quality=90)
            frames.append(frame_payload)
        browser.close()

    if browser_errors:
        details = "\n".join(f"- {message}" for message in browser_errors)
        raise RuntimeError(
            f"Browser errors occurred during dataset generation:\n{details}"
        )

    category = frames[0]["category"]
    sequence = {
        "info": {
            "description": "Deterministic two-mouse synthetic pose and tracking fixture",
            "version": "1.0",
            "generator": "scripts/generate_two_mice_pose_dataset.py",
            "seed": seed,
        },
        "licenses": [],
        "videos": [
            {
                "id": 1,
                "file_name": "two_mice_synthetic",
                "fps": fps,
                "num_frames": frame_count,
                "width": width,
                "height": height,
            }
        ],
        "images": [frame["image"] for frame in frames],
        "annotations": [
            annotation for frame in frames for annotation in frame["annotations"]
        ],
        "categories": [category],
    }
    _write_json(annotations_dir / "sequence.json", sequence)
    train_ids = {image["id"] for image in sequence["images"][:train_frames]}
    val_ids = {image["id"] for image in sequence["images"][train_frames:]}
    _write_json(annotations_dir / "train.json", _subset(sequence, train_ids))
    _write_json(annotations_dir / "val.json", _subset(sequence, val_ids))
    _write_support_files(
        output,
        frame_count=frame_count,
        train_frames=train_frames,
        fps=fps,
        seed=seed,
        width=width,
        height=height,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--train-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if not 0 < args.train_frames < args.frames:
        parser.error(
            "--train-frames must leave at least one train and validation frame"
        )
    if not 1 <= args.fps <= 30:
        parser.error("--fps must be between 1 and 30")
    if args.width < 128 or args.height < 128:
        parser.error("--width and --height must be at least 128")
    return args


def main() -> None:
    args = _parse_args()
    generate_dataset(
        args.output.resolve(),
        frame_count=args.frames,
        train_frames=args.train_frames,
        fps=args.fps,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print(f"Generated {args.frames} frames in {args.output.resolve()}")


if __name__ == "__main__":
    main()
