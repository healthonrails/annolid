"""Sanitize LabelMe JSON files for editor compatibility.

LabelMe treats both image-level and shape-level `flags` as boolean checkboxes.
Non-boolean values (floats, strings, lists, ...) can crash the LabelMe editor.

This module rewrites payloads so:
- `payload["flags"]` is `dict[str, bool]`
- `shape["flags"]` is `dict[str, bool]`
- any removed non-bool entries are preserved outside of `flags`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

from annolid.utils.labelme_flags import (
    sanitize_labelme_flags_with_meta,
    sanitize_labelme_shape_dict,
)


def sanitize_labelme_payload(payload: Any) -> Tuple[Dict[str, Any], bool]:
    """Return `(sanitized_payload, changed)` for a LabelMe JSON dict."""
    if not isinstance(payload, dict):
        return {}, False

    changed = False
    sanitized: Dict[str, Any] = dict(payload)

    safe_flags, flags_meta = sanitize_labelme_flags_with_meta(sanitized.get("flags"))
    if sanitized.get("flags") != safe_flags:
        sanitized["flags"] = safe_flags
        changed = True
    if flags_meta:
        existing = sanitized.get("annolid_flags_meta")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(flags_meta)
            if merged != existing:
                sanitized["annolid_flags_meta"] = merged
                changed = True
        else:
            sanitized["annolid_flags_meta"] = dict(flags_meta)
            changed = True

    shapes = sanitized.get("shapes")
    if isinstance(shapes, list):
        sanitized_shapes = []
        for shape in shapes:
            if isinstance(shape, dict):
                before = dict(shape)
                after = sanitize_labelme_shape_dict(dict(shape))
                if before != after:
                    changed = True
                sanitized_shapes.append(after)
            else:
                sanitized_shapes.append(shape)
        sanitized["shapes"] = sanitized_shapes

    return sanitized, changed


def _iter_json_files(root: Path, *, recursive: bool) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    pattern = "**/*.json" if recursive else "*.json"
    for path in root.glob(pattern):
        if path.is_file():
            yield path


def sanitize_path(path: Path, *, recursive: bool = True) -> int:
    """Sanitize all LabelMe JSON files under `path` in-place.

    Returns the number of files modified.
    """
    modified = 0
    for json_path in _iter_json_files(path, recursive=recursive):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sanitized, changed = sanitize_labelme_payload(payload)
        if not changed:
            continue
        json_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        modified += 1
    return modified


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanitize LabelMe JSON files to keep flags boolean-only."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="LabelMe JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse when PATH is a directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    modified = sanitize_path(args.path, recursive=not args.no_recursive)
    print(f"Sanitized {modified} file(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

