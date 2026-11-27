from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict


TEMPLATE_ROOT = Path(__file__).parent / "workspace"
MANIFEST_REL = ".annolid/bootstrap-manifest.json"

# SHA-256 hashes (LF-normalized) of historical unmodified bundled templates that
# predate manifest hash tracking, enabling safe in-place upgrades of untouched
# default files without ever overwriting user customizations.
_LEGACY_UNMODIFIED_TEMPLATE_HASHES: Dict[str, frozenset[str]] = {
    "TOOLS.md": frozenset(
        {
            "1ea2c84ac21af66fe3961d05ab7a48b74573eb6e5a72d4ed2b2c4c1f87ef6877",
            "c25c2e2e9aacf987ec6498f1a6cc265a7756f3748cd17ca40b7f18d593f96e34",
        }
    ),
}


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


def _hash_bytes(data: bytes) -> str:
    normalized = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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


def _write_manifest(
    workspace: Path,
    managed_files: list[str],
    managed_hashes: Dict[str, str] | None = None,
) -> None:
    path = _manifest_path(workspace)
    payload: dict[str, object] = {
        "managed_files": sorted(set(str(p) for p in managed_files))
    }
    if managed_hashes:
        payload["managed_hashes"] = {
            str(k): str(v) for k, v in sorted(managed_hashes.items()) if k and v
        }
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _is_unmodified_managed_file(
    rel: str,
    dst_hash: str,
    recorded_hashes: Dict[str, str],
) -> bool:
    recorded = str(recorded_hashes.get(rel) or "").strip()
    if recorded and dst_hash == recorded:
        return True
    legacy = _LEGACY_UNMODIFIED_TEMPLATE_HASHES.get(rel)
    if legacy and dst_hash in legacy:
        return True
    return False


def _status(name: str, *, dry_run: bool) -> str:
    return f"would_{name}" if dry_run else name


def _backup_existing(dst: Path, backup_root: Path, rel: str, *, dry_run: bool) -> None:
    backup_path = backup_root / rel
    if dry_run:
        return
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dst, backup_path)


def maybe_upgrade_unmodified_template(workspace: Path, rel: str) -> str | None:
    """Safely upgrade a single workspace template if it is an untouched bundled copy.

    Returns the upgraded UTF-8 text when an upgrade occurs (even if disk write
    is skipped on a read-only workspace), or ``None`` when the file is missing,
    already up to date, or customized by the user.
    """
    src = TEMPLATE_ROOT / rel
    root = Path(workspace).expanduser()
    dst = root / rel
    if not src.is_file() or not dst.is_file():
        return None
    try:
        src_bytes = src.read_bytes()
        dst_bytes = dst.read_bytes()
    except OSError:
        return None
    src_hash = _hash_bytes(src_bytes)
    dst_hash = _hash_bytes(dst_bytes)
    if dst_hash == src_hash:
        return None

    manifest = _read_manifest(root)
    raw_hashes = manifest.get("managed_hashes")
    recorded_hashes: Dict[str, str] = (
        {str(k): str(v) for k, v in raw_hashes.items()}
        if isinstance(raw_hashes, dict)
        else {}
    )
    if not _is_unmodified_managed_file(rel, dst_hash, recorded_hashes):
        return None

    try:
        _atomic_write_bytes(dst, src_bytes)
        managed_files = [
            str(item)
            for item in (manifest.get("managed_files") or [])
            if isinstance(item, str) and item.strip()
        ]
        if rel not in managed_files:
            managed_files.append(rel)
        recorded_hashes[rel] = src_hash
        _write_manifest(root, managed_files, recorded_hashes)
    except OSError:
        pass
    return src_bytes.decode("utf-8", errors="replace")


def bootstrap_workspace(
    workspace: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    backup_root: Path | None = None,
    upgrade_unmodified: bool = True,
) -> Dict[str, str]:
    """
    Copy built-in workspace templates into `workspace`.

    Returns a mapping `{relative_path: status}` where status is one of:
    - `created`
    - `updated` (untouched bundled template safely upgraded)
    - `overwritten`
    - `skipped` (user-customized file preserved when `overwrite=False`)
    - `unchanged`

    When `dry_run` is enabled, statuses are prefixed with `would_`.
    """
    root = Path(workspace).expanduser()
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(root)
    raw_hashes = manifest.get("managed_hashes")
    recorded_hashes: Dict[str, str] = (
        {str(k): str(v) for k, v in raw_hashes.items()}
        if isinstance(raw_hashes, dict)
        else {}
    )
    next_hashes: Dict[str, str] = dict(recorded_hashes)

    outcomes: Dict[str, str] = {}
    template_files = sorted(_iter_template_files().items())
    for rel, src in template_files:
        dst = root / rel
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
        src_bytes = src.read_bytes()
        src_hash = _hash_bytes(src_bytes)

        if not dst.exists():
            if not dry_run:
                _atomic_write_bytes(dst, src_bytes)
            next_hashes[rel] = src_hash
            outcomes[rel] = _status("created", dry_run=dry_run)
            continue

        dst_bytes = dst.read_bytes()
        dst_hash = _hash_bytes(dst_bytes)
        if dst_hash == src_hash:
            next_hashes[rel] = src_hash
            outcomes[rel] = _status("unchanged", dry_run=dry_run)
            continue

        if not overwrite:
            if upgrade_unmodified and _is_unmodified_managed_file(
                rel, dst_hash, recorded_hashes
            ):
                if backup_root is not None:
                    _backup_existing(dst, backup_root, rel, dry_run=dry_run)
                if not dry_run:
                    _atomic_write_bytes(dst, src_bytes)
                next_hashes[rel] = src_hash
                outcomes[rel] = _status("updated", dry_run=dry_run)
                continue
            outcomes[rel] = _status("skipped", dry_run=dry_run)
            continue

        if backup_root is not None:
            _backup_existing(dst, backup_root, rel, dry_run=dry_run)
        if not dry_run:
            _atomic_write_bytes(dst, src_bytes)
        next_hashes[rel] = src_hash
        outcomes[rel] = _status("overwritten", dry_run=dry_run)

    if not dry_run:
        _write_manifest(root, [rel for rel, _ in template_files], next_hashes)
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
    raw_hashes = manifest.get("managed_hashes")
    recorded_hashes: Dict[str, str] = (
        {str(k): str(v) for k, v in raw_hashes.items()}
        if isinstance(raw_hashes, dict)
        else {}
    )
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
        recorded_hashes.pop(rel, None)
        outcomes[rel] = _status("removed", dry_run=dry_run)

    if not dry_run:
        _write_manifest(root, sorted(current_files), recorded_hashes)
    return outcomes
