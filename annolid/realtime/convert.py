from __future__ import annotations

import argparse
from typing import Sequence


DEFAULT_MODEL = "yolo11n-seg.pt"
DEFAULT_FORMAT = "openvino"


def _load_yolo_class():
    from annolid.yolo import import_ultralytics_symbol

    return import_ultralytics_symbol("YOLO", purpose="realtime YOLO export")


def export_yolo_model(
    model: str = DEFAULT_MODEL,
    *,
    export_format: str = DEFAULT_FORMAT,
):
    """Export a YOLO model for realtime inference."""
    from annolid.yolo import configure_ultralytics_cache

    configure_ultralytics_cache()
    yolo_cls = _load_yolo_class()
    yolo_model = yolo_cls(model)
    return yolo_model.export(format=export_format)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a YOLO realtime model to an Ultralytics-supported format."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model path/name to export (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--format",
        dest="export_format",
        default=DEFAULT_FORMAT,
        help=f"Ultralytics export format (default: {DEFAULT_FORMAT}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exported = export_yolo_model(args.model, export_format=args.export_format)
    if exported is not None:
        print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
