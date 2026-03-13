from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES = 12
DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES = 20 * 1024 * 1024 * 1024


def large_image_cache_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "annolid" / "large_image"


@dataclass(frozen=True)
class LargeImageCacheEntry:
    path: Path
    size_bytes: int
    modified_time: float


def optimized_large_image_cache_path(source_path: str | Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    stat = source.stat()
    fingerprint = hashlib.sha1(
        f"{source}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    ).hexdigest()[:12]
    safe_stem = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem
    )
    return large_image_cache_root() / f"{safe_stem}_{fingerprint}.pyramidal.tif"


def resolve_fresh_optimized_large_image_path(source_path: str | Path) -> Path | None:
    cache_path = optimized_large_image_cache_path(source_path)
    return cache_path if cache_path.exists() else None


def list_large_image_cache_entries() -> list[LargeImageCacheEntry]:
    root = large_image_cache_root()
    if not root.exists():
        return []
    entries: list[LargeImageCacheEntry] = []
    for path in root.glob("*.pyramidal.tif"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            LargeImageCacheEntry(
                path=path,
                size_bytes=int(stat.st_size),
                modified_time=float(stat.st_mtime),
            )
        )
    entries.sort(key=lambda entry: entry.modified_time, reverse=True)
    return entries


def large_image_cache_size_bytes() -> int:
    return sum(entry.size_bytes for entry in list_large_image_cache_entries())


def remove_large_image_cache_file(cache_path: str | Path) -> bool:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return False
    path.unlink()
    return True


def clear_all_large_image_caches() -> int:
    removed = 0
    for entry in list_large_image_cache_entries():
        try:
            entry.path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def prune_large_image_caches(
    *,
    max_entries: int = DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES,
    max_size_bytes: int = DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES,
    keep_paths: tuple[str | Path, ...] = (),
) -> list[Path]:
    entries = list_large_image_cache_entries()
    protected = {str(Path(path).expanduser().resolve()) for path in keep_paths}
    survivors = list(entries)
    total_size = sum(entry.size_bytes for entry in survivors)
    removed: list[Path] = []
    limit_entries = max(1, int(max_entries))
    limit_bytes = max(0, int(max_size_bytes))

    for entry in reversed(entries):
        if len(survivors) <= limit_entries and total_size <= limit_bytes:
            break
        try:
            resolved = str(entry.path.expanduser().resolve())
        except OSError:
            resolved = str(entry.path)
        if resolved in protected:
            continue
        try:
            entry.path.unlink()
        except OSError:
            continue
        survivors = [
            candidate for candidate in survivors if candidate.path != entry.path
        ]
        total_size -= entry.size_bytes
        removed.append(entry.path)
    return removed


def format_large_image_cache_size(size_bytes: int) -> str:
    size = float(max(0, int(size_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0


def pyvips_optimization_available() -> tuple[bool, str | None]:
    try:
        import pyvips  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


def optimize_large_tiff_for_viewing(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    tile_size: int = 512,
    jpeg_quality: int = 90,
    max_cache_entries: int = DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES,
    max_cache_size_bytes: int = DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES,
) -> Path:
    import pyvips

    source = Path(source_path).expanduser().resolve()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else optimized_large_image_cache_path(source)
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    image = pyvips.Image.new_from_file(str(source), access="sequential")
    image.tiffsave(
        str(target),
        tile=True,
        pyramid=True,
        subifd=True,
        bigtiff=True,
        tile_width=max(128, int(tile_size)),
        tile_height=max(128, int(tile_size)),
        compression="jpeg",
        Q=max(1, min(100, int(jpeg_quality))),
        properties=True,
    )
    prune_large_image_caches(
        max_entries=max_cache_entries,
        max_size_bytes=max_cache_size_bytes,
        keep_paths=(target,),
    )
    return target
