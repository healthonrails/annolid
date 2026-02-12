from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


TEMPLATE_ROOT = Path(__file__).parent / "workspace"


def _iter_template_files() -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in TEMPLATE_ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(TEMPLATE_ROOT).as_posix()
        files[rel] = path
    return files


def bootstrap_workspace(workspace: Path, *, overwrite: bool = False) -> Dict[str, str]:
    """
    Copy built-in workspace templates into `workspace`.

    Returns a mapping `{relative_path: status}` where status is one of:
    - `created`
    - `overwritten`
    - `skipped`
    """
    root = Path(workspace).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    outcomes: Dict[str, str] = {}
    for rel, src in _iter_template_files().items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        existed = dst.exists()
        if existed and not overwrite:
            outcomes[rel] = "skipped"
            continue
        shutil.copyfile(src, dst)
        outcomes[rel] = "overwritten" if existed else "created"
    return outcomes
