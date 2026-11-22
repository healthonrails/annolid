"""Convert COCO annotations (keypoints or bounding boxes) to DeepLabCut format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from annolid.annotation.pose_interop.deeplabcut import write_dlc_analysis_csv
from annolid.annotation.pose_interop.labelme import read_pose_from_coco_json
from annolid.utils.atomic_io import atomic_write_json


def convert_coco_to_dlc(
    coco_file: str | Path,
    dlc_file: str | Path,
    *,
    scorer: str = "annolid",
    multi_animal: Optional[bool] = None,
    output_format: Optional[str] = None,
) -> Path:
    """Convert COCO annotations to DeepLabCut multi-index CSV (or legacy JSON).

    When ``dlc_file`` ends with ``.csv`` (or ``output_format == "csv"``), this writes a
    real DeepLabCut/Lightning Pose/SimBA-compatible multi-index CSV preserving all COCO
    keypoints (or bounding-box centers when keypoints are absent).

    When ``dlc_file`` ends with ``.json`` and ``output_format`` is not ``"csv"``, this
    preserves the legacy JSON list format for backward compatibility.
    """
    src = Path(coco_file).expanduser().resolve()
    dst = Path(dlc_file).expanduser().resolve()
    fmt = str(output_format or "").strip().lower()
    if not fmt:
        fmt = "json" if dst.suffix.lower() == ".json" else "csv"

    if fmt == "csv":
        dataset = read_pose_from_coco_json(src)
        return write_dlc_analysis_csv(
            dataset,
            dst,
            scorer=scorer,
            multi_animal=multi_animal,
        )

    # Legacy JSON compatibility path
    with src.open("r", encoding="utf-8") as fh:
        coco_outputs = json.load(fh)

    dlc_outputs = []
    for annotation in coco_outputs.get("annotations", []):
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        dlc_outputs.append(
            {
                "frame": annotation.get("image_id"),
                "x": cx,
                "y": cy,
                "likelihood": float(annotation.get("score", 1.0)),
                "label": annotation.get("category_id"),
            }
        )

    atomic_write_json(dst, dlc_outputs, indent=2)
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert COCO keypoints or detections JSON to DeepLabCut multi-index CSV."
    )
    parser.add_argument("coco_file", help="Input COCO annotations JSON path.")
    parser.add_argument(
        "dlc_file", help="Output DeepLabCut .csv (or legacy .json) path."
    )
    parser.add_argument(
        "--scorer", default="annolid", help="Scorer name (default: annolid)."
    )
    parser.add_argument(
        "--multi-animal",
        action="store_true",
        default=None,
        help="Force 4-level multi-animal DLC CSV header (auto-detected by default).",
    )
    args = parser.parse_args(argv)
    try:
        out = convert_coco_to_dlc(
            args.coco_file,
            args.dlc_file,
            scorer=args.scorer,
            multi_animal=args.multi_animal,
        )
        print(f"[coco2deeplabcut] wrote: {out}")
        return 0
    except Exception as exc:
        print(f"[coco2deeplabcut] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
