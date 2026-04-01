from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict
import json


TEMPLATE_ROOT = Path(__file__).parent / "workspace"
MANIFEST_REL = ".annolid/bootstrap-manifest.json"


def _iter_template_files() -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in TEMPLATE_ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(TEMPLATE_ROOT).as_posix()
        files[rel] = path
    return files


def _manifest_path(workspace: Path) -> Path:
    return Path(workspace).expanduser() / MANIFEST_REL


def _read_manifest(workspace: Path) -> dict:
    path = _manifest_path(workspace)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_manifest(workspace: Path, managed_files: list[str]) -> None:
    path = _manifest_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"managed_files": sorted(set(str(p) for p in managed_files))}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _status(name: str, *, dry_run: bool) -> str:
    return f"would_{name}" if dry_run else name


def _backup_existing(dst: Path, backup_root: Path, rel: str, *, dry_run: bool) -> None:
    backup_path = backup_root / rel
    if dry_run:
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dst, backup_path)


def bootstrap_workspace(
    workspace: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    backup_root: Path | None = None,
) -> Dict[str, str]:
    """
    Copy built-in workspace templates into `workspace`.

    Returns a mapping `{relative_path: status}` where status is one of:
    - `created`
    - `overwritten`
    - `skipped`
    - `unchanged`

    When `dry_run` is enabled, statuses are prefixed with `would_`.
    """
    root = Path(workspace).expanduser()
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)

    outcomes: Dict[str, str] = {}
    template_files = sorted(_iter_template_files().items())
    for rel, src in template_files:
        dst = root / rel
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
        existed = dst.exists()
        if not existed:
            if not dry_run:
                shutil.copyfile(src, dst)
            outcomes[rel] = _status("created", dry_run=dry_run)
            continue

        src_bytes = src.read_bytes()
        dst_bytes = dst.read_bytes()
        if src_bytes == dst_bytes:
            outcomes[rel] = _status("unchanged", dry_run=dry_run)
            continue

        if not overwrite:
            outcomes[rel] = _status("skipped", dry_run=dry_run)
            continue

        if backup_root is not None:
            _backup_existing(dst, backup_root, rel, dry_run=dry_run)
        if not dry_run:
            shutil.copyfile(src, dst)
        outcomes[rel] = _status("overwritten", dry_run=dry_run)
    if not dry_run:
        _write_manifest(root, [rel for rel, _ in template_files])
    return outcomes


def prune_bootstrap_workspace(
    workspace: Path,
    *,
    dry_run: bool = False,
    backup_root: Path | None = None,
) -> Dict[str, str]:
    """
    Remove stale bootstrap-managed files that are no longer in template catalog.

    Stale candidates are resolved from the persisted bootstrap manifest.
    Returns `{relative_path: status}` with status in:
    - `removed`
    - `missing`
    - `would_removed`
    - `would_missing`
    """
    root = Path(workspace).expanduser()
    if not root.exists():
        return {}

    manifest = _read_manifest(root)
    previous_files = {
        str(item)
        for item in (manifest.get("managed_files") or [])
        if isinstance(item, str) and item.strip()
    }
    current_files = set(_iter_template_files().keys())
    stale_files = sorted(previous_files - current_files)
    outcomes: Dict[str, str] = {}

    for rel in stale_files:
        target = root / rel
        if not target.exists():
            outcomes[rel] = _status("missing", dry_run=dry_run)
            continue
        if backup_root is not None:
            _backup_existing(target, backup_root, rel, dry_run=dry_run)
        if not dry_run:
            target.unlink()
            parent = target.parent
            while parent != root and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        outcomes[rel] = _status("removed", dry_run=dry_run)

    if not dry_run:
        _write_manifest(root, sorted(current_files))
    return outcomes
