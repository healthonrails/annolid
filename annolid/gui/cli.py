import argparse
import codecs
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from annolid.configs import get_config

__all__ = ["build_parser", "parse_cli"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser used by the Annolid GUI entry point."""
    parser = argparse.ArgumentParser(description="Launch the Annolid GUI.")
    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="show version and exit",
    )
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save annotations",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        default=argparse.SUPPRESS,
        help="comma separated list of labels or file containing labels",
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )

    default_config_file = str(Path.home() / ".labelmerc")
    parser.add_argument(
        "--config",
        dest="config",
        default=default_config_file,
        help=f"config file or yaml format string (default {default_config_file})",
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    return parser


def _load_items_from_source(raw_value: str) -> List[str]:
    """Interpret a flag/label source as newline separated file or comma list."""
    if os.path.isfile(raw_value):
        with codecs.open(raw_value, "r", encoding="utf-8") as file_obj:
            return [line.strip() for line in file_obj if line.strip()]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _normalize_cli_collections(namespace: argparse.Namespace) -> None:
    """Mutate the namespace in-place to expand flags/labels into lists."""
    if hasattr(namespace, "flags"):
        namespace.flags = _load_items_from_source(namespace.flags)
    if hasattr(namespace, "labels"):
        source = namespace.labels
        if Path(source).is_file():
            with codecs.open(source, "r", encoding="utf-8") as file_obj:
                namespace.labels = [
                    line.strip() for line in file_obj if line.strip()
                ]
        else:
            namespace.labels = [
                item for item in source.split(",") if item
            ]


def parse_cli(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[dict, argparse.Namespace, bool]:
    """Parse CLI arguments and return `(config, namespace, version_requested)`."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    _normalize_cli_collections(namespace)

    overrides = vars(namespace).copy()
    version_requested = overrides.pop("version", False)
    config_file_or_yaml = overrides.pop("config")
    config = get_config(config_file_or_yaml, overrides)
    return config, namespace, bool(version_requested)
