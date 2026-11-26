"""Framework-agnostic project workspace, workflow progress, and safe file-pair services."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
import zipfile

from annolid.infrastructure import AnnotationStore, atomic_write_json, atomic_write_text
from annolid.utils.logger import logger

PROJECT_CONFIG_FILENAME = "project.annolid.json"

DEFAULT_PROJECT_DIRECTORIES: Dict[str, str] = {
    "images": "images",
    "annotations": "annotations",
    "models": "models",
    "videos": "videos",
    "exports": "exports",
}

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
)

DEFAULT_MODEL_EXTENSIONS: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".onnx",
    ".ckpt",
    ".safetensors",
)

DEFAULT_ANNOLIDIGNORE = """# Annolid ignore file
# Files and directories to ignore during project operations

# Temporary files
*.tmp
*.temp
*~

# System files
.DS_Store
Thumbs.db

# Cache directories
__pycache__/
*.pyc
*.pyo

# Log files
*.log

# Backup files
*.bak
*.backup

# Export directories (can be large)
exports/
"""

_INVALID_NAME_CHARS = frozenset({"\x00", "/", "\\", "\n", "\r", "\t"})
_NON_ANNOTATION_JSON_NAMES = frozenset({PROJECT_CONFIG_FILENAME, "pose_schema.json"})
_EXPORT_COPY_CHUNK_BYTES = 1024 * 1024


class _ExportCancelled(Exception):
    """Internal control-flow signal when export_project_bundle is cancelled."""


def is_non_annotation_json(path: Path) -> bool:
    """Return True when *path* is an Annolid project/metadata JSON rather than a LabelMe frame."""
    name = path.name
    low = name.lower()
    return bool(
        name in _NON_ANNOTATION_JSON_NAMES
        or low.endswith("_tracking_stats.json")
        or low.endswith(".segments.json")
    )


def current_iso_timestamp() -> str:
    """Return current local timestamp in ISO-8601 format."""
    return datetime.now().isoformat()


def validate_project_name(name: str) -> Tuple[bool, str]:
    """Validate a project or file basename against path traversal and control chars."""
    raw = str(name or "").strip()
    if not raw:
        return False, "Name cannot be empty."
    if raw in {".", ".."}:
        return False, "Name cannot be '.' or '..'."
    if any(ch in _INVALID_NAME_CHARS for ch in raw):
        return False, "Name cannot contain path separators or control characters."
    if Path(raw).name != raw:
        return False, "Name must be a single folder or file name without path parts."
    return True, ""


def resolve_project_dir(path_like: str | Path) -> Path:
    """Resolve either a project directory or a direct path to `project.annolid.json`."""
    candidate = Path(path_like).expanduser()
    if candidate.is_file() and candidate.name == PROJECT_CONFIG_FILENAME:
        return candidate.parent
    return candidate


def default_project_config(
    project_name: str,
    *,
    description: str = "",
) -> Dict[str, Any]:
    """Build the canonical default workspace configuration envelope."""
    now = current_iso_timestamp()
    return {
        "name": project_name,
        "version": "1.0.0",
        "description": str(description or f"Annolid project: {project_name}"),
        "created": now,
        "modified": now,
        "directories": dict(DEFAULT_PROJECT_DIRECTORIES),
        "settings": {
            "annotation_format": "labelme",
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "video_extensions": [".mp4", ".avi", ".mov", ".mkv"],
            "auto_save": True,
            "backup_interval": 300,
        },
        "classes": [],
        "models": [],
        "videos": [],
    }


def deep_merge_dict(base_dict: Dict[str, Any], update_dict: Mapping[str, Any]) -> None:
    """Deep-merge *update_dict* into *base_dict* in place."""
    for key, value in update_dict.items():
        if (
            key in base_dict
            and isinstance(base_dict[key], dict)
            and isinstance(value, Mapping)
        ):
            deep_merge_dict(base_dict[key], value)
        else:
            base_dict[key] = value


def is_schema_only_project_config(config: Mapping[str, Any]) -> bool:
    """Return True when a project.annolid.json contains schema keys without workspace envelope."""
    has_schema = any(
        key in config
        for key in ("behaviors", "subjects", "categories", "modifiers", "pose_schema")
    )
    has_workspace = "name" in config and "directories" in config
    return bool(has_schema and not has_workspace)


def normalize_project_config(
    project_dir: Path,
    raw_config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Ensure a project config dict has workspace metadata without dropping schema fields.

    Returns ``(normalized_config, was_upgraded)``.
    """
    upgraded = False
    merged: Dict[str, Any] = dict(raw_config or {})
    defaults = default_project_config(project_dir.name or "AnnolidProject")

    if not str(merged.get("name") or "").strip():
        merged["name"] = defaults["name"]
        upgraded = True

    if not isinstance(merged.get("directories"), dict) or not merged["directories"]:
        merged["directories"] = dict(DEFAULT_PROJECT_DIRECTORIES)
        upgraded = True

    if not isinstance(merged.get("settings"), dict):
        merged["settings"] = dict(defaults["settings"])
        upgraded = True
    else:
        for key, val in defaults["settings"].items():
            if key not in merged["settings"]:
                merged["settings"][key] = val
                upgraded = True

    for key in ("description", "created", "modified", "classes", "models", "videos"):
        if key not in merged:
            merged[key] = defaults[key]
            upgraded = True

    if "version" not in merged:
        merged["version"] = defaults["version"]
        upgraded = True

    return merged, upgraded


def ensure_project_directories(
    project_dir: Path,
    directories: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    """Create standard (or configured) subdirectories inside *project_dir*."""
    root = Path(project_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mapping = dict(directories or DEFAULT_PROJECT_DIRECTORIES)
    created: Dict[str, Path] = {}
    for key, rel_name in mapping.items():
        clean_rel = str(rel_name or "").strip()
        if not clean_rel:
            continue
        candidate = (root / clean_rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Project directory '{key}' escapes project root: {clean_rel}"
            ) from exc
        candidate.mkdir(parents=True, exist_ok=True)
        created[key] = candidate
    return created


def write_project_ignore_file(project_dir: Path, *, overwrite: bool = False) -> Path:
    """Write `.annolidignore` atomically if missing (or when *overwrite=True*)."""
    ignore_path = Path(project_dir) / ".annolidignore"
    if overwrite or not ignore_path.exists():
        atomic_write_text(ignore_path, DEFAULT_ANNOLIDIGNORE, encoding="utf-8")
    return ignore_path


def resolve_project_video_paths(
    project_dir: Path,
    config: Mapping[str, Any],
) -> List[Path]:
    """Return deduplicated absolute paths of videos registered or stored in *project_dir*."""
    root = Path(project_dir).resolve()
    seen: set[str] = set()
    resolved: List[Path] = []

    raw_exts = (config.get("settings") or {}).get(
        "video_extensions", DEFAULT_VIDEO_EXTENSIONS
    )
    video_exts = {
        str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in (raw_exts or DEFAULT_VIDEO_EXTENSIONS)
    }

    def _add(path_obj: Path) -> None:
        key = str(path_obj)
        if key not in seen:
            seen.add(key)
            resolved.append(path_obj)

    for entry in list(config.get("videos") or []):
        raw_path = entry.get("path") if isinstance(entry, Mapping) else entry
        text = str(raw_path or "").strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        _add(candidate)

    videos_rel = str(
        (config.get("directories") or {}).get("videos", "videos") or "videos"
    )
    videos_dir = root / videos_rel
    if videos_dir.is_dir():
        for file_path in sorted(videos_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in video_exts:
                _add(file_path.resolve())

    return resolved


def register_project_videos(
    project_dir: Path,
    config: Dict[str, Any],
    video_paths: Sequence[str | Path],
    *,
    copy_into_project: bool = False,
) -> List[str]:
    """Register *video_paths* into *config['videos']* (optionally copying into `videos/`).

    Returns the list of absolute resolved paths for all newly/currently registered videos.
    """
    root = Path(project_dir).resolve()
    videos_rel = str(
        (config.get("directories") or {}).get("videos", "videos") or "videos"
    )
    videos_dir = root / videos_rel
    if copy_into_project:
        videos_dir.mkdir(parents=True, exist_ok=True)

    existing_entries = list(config.get("videos") or [])
    normalized_existing: List[str] = []
    seen_resolved: set[str] = set()
    for item in existing_entries:
        raw = item.get("path") if isinstance(item, Mapping) else item
        text = str(raw or "").strip()
        if not text:
            continue
        p = Path(text).expanduser()
        full = (root / p).resolve() if not p.is_absolute() else p.resolve()
        if str(full) not in seen_resolved:
            seen_resolved.add(str(full))
            normalized_existing.append(text)

    registered_abs: List[str] = []
    for raw_video in video_paths:
        src = Path(raw_video).expanduser()
        if not str(raw_video or "").strip():
            continue
        final_path = src.resolve() if src.exists() else src
        if copy_into_project and src.is_file():
            dst = videos_dir / src.name
            if dst.resolve() != src.resolve():
                if dst.exists():
                    stem = dst.stem
                    suffix = dst.suffix
                    idx = 1
                    while dst.exists():
                        dst = videos_dir / f"{stem}_{idx}{suffix}"
                        idx += 1
                shutil.copy2(src, dst)
            final_path = dst.resolve()

        abs_str = str(final_path)
        registered_abs.append(abs_str)
        if abs_str in seen_resolved:
            continue
        seen_resolved.add(abs_str)
        try:
            stored = final_path.relative_to(root).as_posix()
        except ValueError:
            stored = abs_str
        normalized_existing.append(stored)

    config["videos"] = normalized_existing
    return registered_abs


def _count_dual_annotations_in_dir(directory: Path) -> Tuple[int, int]:
    """Count per-frame JSON files and deduplicated AnnotationStore records under *directory*.

    Returns ``(total_annotated_frames, store_only_records)``.
    """
    if not directory.is_dir():
        return 0, 0

    json_frame_keys: set[tuple[str, int]] = set()
    non_frame_json_count = 0
    ndjson_paths: List[Path] = []

    for root_str, _dirs, files in os.walk(directory):
        root_path = Path(root_str)
        for name in files:
            low = name.lower()
            if low.endswith("_annotations.ndjson"):
                ndjson_paths.append(root_path / name)
                continue
            if not low.endswith(".json"):
                continue
            file_path = root_path / name
            if is_non_annotation_json(file_path):
                continue
            frame_num = AnnotationStore.frame_number_from_path(file_path)
            if frame_num is not None:
                json_frame_keys.add((str(root_path.resolve()), int(frame_num)))
            else:
                non_frame_json_count += 1

    store_only = 0
    for ndjson_path in ndjson_paths:
        parent_key = str(ndjson_path.parent.resolve())
        try:
            store = AnnotationStore(ndjson_path)
            for frame_num in store.iter_frame_numbers_fast():
                if (parent_key, int(frame_num)) not in json_frame_keys:
                    store_only += 1
        except Exception as exc:
            logger.debug("Skipping unreadable AnnotationStore %s: %s", ndjson_path, exc)

    total = len(json_frame_keys) + non_frame_json_count + store_only
    return total, store_only


def scan_project_statistics(
    project_dir: Path,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Gather accurate project statistics across nested folders and AnnotationStore files."""
    root = Path(project_dir).resolve()
    dirs_cfg = dict(config.get("directories") or {})
    settings = dict(config.get("settings") or {})

    raw_img_exts = settings.get("image_extensions") or DEFAULT_IMAGE_EXTENSIONS
    image_exts = {
        str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in raw_img_exts
    }

    stats: Dict[str, Any] = {
        "total_images": 0,
        "total_annotations": 0,
        "store_only_annotations": 0,
        "total_videos": 0,
        "missing_videos": 0,
        "total_models": 0,
        "total_exports": 0,
        "has_behavior_schema": False,
        "has_pose_schema": False,
    }

    try:
        images_dir = root / str(dirs_cfg.get("images", "images") or "images")
        if images_dir.is_dir():
            stats["total_images"] = sum(
                1
                for p in images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in image_exts
            )

        annotations_dir = root / str(
            dirs_cfg.get("annotations", "annotations") or "annotations"
        )
        ann_total, ann_store_only = _count_dual_annotations_in_dir(annotations_dir)

        # Also check video sibling results directories inside the project's videos folder.
        videos_dir = root / str(dirs_cfg.get("videos", "videos") or "videos")
        if videos_dir.is_dir():
            vid_ann_total, vid_store_only = _count_dual_annotations_in_dir(videos_dir)
            ann_total += vid_ann_total
            ann_store_only += vid_store_only

        stats["total_annotations"] = ann_total
        stats["store_only_annotations"] = ann_store_only

        video_paths = resolve_project_video_paths(root, config)
        existing_videos = [p for p in video_paths if p.exists()]
        stats["total_videos"] = len(existing_videos)
        stats["missing_videos"] = len(video_paths) - len(existing_videos)

        models_dir = root / str(dirs_cfg.get("models", "models") or "models")
        if models_dir.is_dir():
            stats["total_models"] = sum(
                1
                for p in models_dir.rglob("*")
                if p.is_file()
                and (
                    p.suffix.lower() in DEFAULT_MODEL_EXTENSIONS
                    or p.parent == models_dir
                )
            )

        exports_dir = root / str(dirs_cfg.get("exports", "exports") or "exports")
        if exports_dir.is_dir():
            stats["total_exports"] = sum(
                1 for p in exports_dir.iterdir() if not p.name.startswith(".")
            )

        stats["has_behavior_schema"] = bool(
            config.get("behaviors") or config.get("subjects")
        )
        pose_path_rel = str(config.get("pose_schema_path") or "").strip()
        stats["has_pose_schema"] = bool(
            config.get("pose_schema")
            or (pose_path_rel and (root / pose_path_rel).is_file())
            or (root / "pose_schema.json").is_file()
        )
    except Exception as exc:
        logger.warning("Failed to gather project statistics for %s: %s", root, exc)

    return stats


def compute_project_workflow_status(
    project_dir: Optional[Path],
    config: Optional[Mapping[str, Any]] = None,
    statistics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute live 6-step Annolid workflow status from project artifacts on disk.

    Steps: ``project``, ``collect``, ``annotate``, ``train``, ``inference``, ``analyze``.
    """
    if project_dir is None:
        return {
            "steps": {
                "project": "active",
                "collect": "pending",
                "annotate": "pending",
                "train": "pending",
                "inference": "pending",
                "analyze": "pending",
            },
            "next_step": "project",
            "completed_count": 0,
            "total_steps": 6,
            "summary": "Create or open a project to start tracking workflow progress.",
        }

    root = Path(project_dir).resolve()
    cfg = dict(config or {})
    stats = dict(statistics or scan_project_statistics(root, cfg))

    has_project = (root / PROJECT_CONFIG_FILENAME).is_file()
    has_media = (
        int(stats.get("total_videos", 0)) > 0 or int(stats.get("total_images", 0)) > 0
    )
    has_annotations = int(stats.get("total_annotations", 0)) > 0
    has_models = int(stats.get("total_models", 0)) > 0 or bool(cfg.get("models"))

    exports_dir = root / str(
        (cfg.get("directories") or {}).get("exports", "exports") or "exports"
    )
    has_exports = int(stats.get("total_exports", 0)) > 0
    has_inference = bool(
        (root / "predictions").is_dir()
        and any((root / "predictions").iterdir())
        or any(
            p.is_dir() and "_tracking_results" in p.name
            for p in root.iterdir()
            if root.is_dir()
        )
        or (
            exports_dir.is_dir()
            and any(
                p.suffix.lower() in {".csv", ".parquet"}
                for p in exports_dir.rglob("*")
                if p.is_file()
            )
        )
    )
    has_reports = bool(
        exports_dir.is_dir()
        and any(
            p.suffix.lower() in {".html", ".svg", ".pdf", ".md", ".boris"}
            for p in exports_dir.rglob("*")
            if p.is_file()
        )
    ) or (has_exports and has_inference)

    flags = [
        ("project", has_project),
        ("collect", has_media),
        ("annotate", has_annotations),
        ("train", has_models),
        ("inference", has_inference),
        ("analyze", has_reports),
    ]

    steps: Dict[str, str] = {}
    next_step = "analyze"
    found_active = False
    completed_count = 0
    for step_id, done in flags:
        if done:
            steps[step_id] = "completed"
            completed_count += 1
        elif not found_active:
            steps[step_id] = "active"
            next_step = step_id
            found_active = True
        else:
            steps[step_id] = "pending"

    labels = {
        "project": "Define Project",
        "collect": "Import Videos or Images",
        "annotate": "Annotate Frames",
        "train": "Train a Model",
        "inference": "Run Inference / Tracking",
        "analyze": "Export & Analyze Results",
    }
    if completed_count == len(flags):
        summary = "All 6 workflow stages have artifacts in this project."
    else:
        summary = (
            f"{completed_count}/{len(flags)} stages complete — "
            f"recommended next step: {labels.get(next_step, next_step)}."
        )

    return {
        "steps": steps,
        "next_step": next_step,
        "completed_count": completed_count,
        "total_steps": len(flags),
        "summary": summary,
    }


def _copy_file_cancellable(
    src: Path,
    dst: Path,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    chunk_size: int = _EXPORT_COPY_CHUNK_BYTES,
) -> bool:
    """Copy *src* to *dst* in chunks so large video copies can be cancelled promptly."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            if cancel_check is not None and cancel_check():
                fout.close()
                dst.unlink(missing_ok=True)
                return False
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass
    return True


def _collect_export_manifest(
    src_root: Path,
    dirs_cfg: Mapping[str, Any],
    dir_keys: Sequence[str],
    *,
    exclude_paths: Optional[set[Path]] = None,
) -> List[Tuple[Path, str]]:
    """Build deterministic ``(source_file, posix_rel_path)`` pairs to export."""
    excluded = exclude_paths or set()
    manifest: List[Tuple[Path, str]] = []
    seen_arcnames: set[str] = set()

    for fname in (PROJECT_CONFIG_FILENAME, "pose_schema.json", ".annolidignore"):
        candidate = (src_root / fname).resolve()
        if (
            candidate.is_file()
            and candidate not in excluded
            and fname not in seen_arcnames
        ):
            seen_arcnames.add(fname)
            manifest.append((candidate, fname))

    for key in dir_keys:
        rel_dir = str(dirs_cfg.get(key, key) or key).strip()
        if not rel_dir:
            continue
        src_dir = (src_root / rel_dir).resolve()
        if not src_dir.is_dir():
            continue
        try:
            src_dir.relative_to(src_root)
        except ValueError:
            continue
        for item in sorted(src_dir.rglob("*")):
            if not item.is_file():
                continue
            resolved_item = item.resolve()
            if resolved_item in excluded:
                continue
            try:
                arcname = resolved_item.relative_to(src_root).as_posix()
            except ValueError:
                continue
            if arcname not in seen_arcnames:
                seen_arcnames.add(arcname)
                manifest.append((resolved_item, arcname))

    return manifest


def export_project_bundle(
    project_dir: Path,
    export_path: Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    include_data: bool = True,
    include_models: bool = False,
    as_zip: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[bool, str]:
    """Export a project folder or atomic `.zip` archive safely with progress & cancellation.

    Guards against exporting into *project_dir* itself or a subdirectory of it (folder mode)
    or overwriting an active project source file (ZIP mode).
    """
    src_root = Path(project_dir).expanduser().resolve()
    dst = Path(export_path).expanduser().resolve()

    if not src_root.is_dir():
        return False, f"Project directory does not exist: {src_root}"

    config_src = src_root / PROJECT_CONFIG_FILENAME
    if not config_src.is_file():
        return False, f"Project configuration file not found: {config_src}"

    # Prevent infinite recursion when exporting into a subfolder of the project.
    if not as_zip:
        if dst == src_root or src_root in dst.parents:
            return (
                False,
                "Export destination cannot be the project folder or inside it.",
            )

    dirs_cfg = dict((config or {}).get("directories") or DEFAULT_PROJECT_DIRECTORIES)
    dir_keys = ["annotations"]
    if include_data:
        dir_keys.extend(["images", "videos"])
    if include_models:
        dir_keys.append("models")

    manifest = _collect_export_manifest(
        src_root,
        dirs_cfg,
        dir_keys,
    )
    if as_zip and any(src_file == dst for src_file, _ in manifest):
        return False, "Export archive destination conflicts with a project file."

    total_files = len(manifest)

    def _report(done: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total_files, label)

    def _is_cancelled() -> bool:
        return bool(cancel_check is not None and cancel_check())

    try:
        if _is_cancelled():
            return False, "Export cancelled by user."
        _report(0, "Preparing export…")

        if as_zip:
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_zip = tempfile.mkstemp(
                prefix=f".{dst.stem}_", suffix=".zip", dir=str(dst.parent)
            )
            os.close(fd)
            tmp_zip_path = Path(tmp_zip)
            try:
                with zipfile.ZipFile(
                    tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as zf:
                    for idx, (src_file, arcname) in enumerate(manifest):
                        if _is_cancelled():
                            raise _ExportCancelled()
                        _report(idx, arcname)
                        zinfo = zipfile.ZipInfo.from_file(src_file, arcname=arcname)
                        zinfo.compress_type = zipfile.ZIP_DEFLATED
                        with (
                            open(src_file, "rb") as fin,
                            zf.open(zinfo, "w") as zout,
                        ):
                            while True:
                                if _is_cancelled():
                                    raise _ExportCancelled()
                                chunk = fin.read(_EXPORT_COPY_CHUNK_BYTES)
                                if not chunk:
                                    break
                                zout.write(chunk)
                        _report(idx + 1, arcname)
                if _is_cancelled():
                    raise _ExportCancelled()
                os.replace(tmp_zip_path, dst)
            except _ExportCancelled:
                tmp_zip_path.unlink(missing_ok=True)
                return False, "Export cancelled by user."
            except Exception:
                tmp_zip_path.unlink(missing_ok=True)
                raise
            _report(total_files, dst.name)
            return True, f"Project archive exported successfully to {dst}"

        dst_existed = dst.exists()
        dst.mkdir(parents=True, exist_ok=True)
        written_files: List[Path] = []
        for idx, (src_file, arcname) in enumerate(manifest):
            if _is_cancelled():
                raise _ExportCancelled()
            _report(idx, arcname)
            target_file = dst / arcname
            copied = _copy_file_cancellable(
                src_file, target_file, cancel_check=cancel_check
            )
            if not copied:
                raise _ExportCancelled()
            written_files.append(target_file)
            _report(idx + 1, arcname)

        _report(total_files, dst.name)
        return True, f"Project exported successfully to {dst}"
    except _ExportCancelled:
        if not as_zip:
            if not dst_existed:
                shutil.rmtree(dst, ignore_errors=True)
            else:
                for written in reversed(written_files):
                    try:
                        written.unlink(missing_ok=True)
                    except OSError:
                        pass
        return False, "Export cancelled by user."
    except Exception as exc:
        logger.error("Failed to export project %s to %s: %s", src_root, dst, exc)
        return False, f"Failed to export project: {exc}"


def _resolve_audit_image_path(
    json_path: Path,
    payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Resolve the image corresponding to a LabelMe JSON without optional dependencies."""
    base = json_path.with_suffix("")
    for ext in DEFAULT_IMAGE_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.is_file():
            return candidate

    if isinstance(payload, Mapping):
        raw_img = str(payload.get("imagePath") or "").strip()
        if raw_img:
            candidate = Path(raw_img).expanduser()
            if not candidate.is_absolute():
                candidate = json_path.parent / candidate
            if candidate.is_file():
                return candidate
            alt_base = json_path.parent / Path(raw_img).stem
            for ext in DEFAULT_IMAGE_EXTENSIONS:
                alt = alt_base.with_suffix(ext)
                if alt.is_file():
                    return alt
    return None


def scan_annotation_audit(
    roots: Sequence[str | Path],
    *,
    recursive: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Audit LabelMe `.json` files and `AnnotationStore` NDJSON files across *roots*.

    Excludes Annolid project/schema/tracking metadata JSON files so auditing a project
    folder only reports actual annotation files.
    """
    json_files: List[Path] = []
    ndjson_files: List[Path] = []
    seen_json: set[Path] = set()
    seen_ndjson: set[Path] = set()

    for raw_root in roots:
        if cancel_check is not None and cancel_check():
            return {
                "items": [],
                "ndjson_stores": [],
                "ndjson_total_frames": 0,
                "labels": [],
                "cancelled": True,
            }
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        if root.is_file():
            resolved = root.resolve()
            low = resolved.name.lower()
            if low.endswith("_annotations.ndjson") and resolved not in seen_ndjson:
                seen_ndjson.add(resolved)
                ndjson_files.append(resolved)
            elif (
                low.endswith(".json")
                and not is_non_annotation_json(resolved)
                and resolved not in seen_json
            ):
                seen_json.add(resolved)
                json_files.append(resolved)
            continue

        iterator = root.rglob("*") if recursive else root.iterdir()
        for entry in iterator:
            if cancel_check is not None and cancel_check():
                return {
                    "items": [],
                    "ndjson_stores": [],
                    "ndjson_total_frames": 0,
                    "labels": [],
                    "cancelled": True,
                }
            if not entry.is_file():
                continue
            low = entry.name.lower()
            if low.endswith("_annotations.ndjson"):
                resolved = entry.resolve()
                if resolved not in seen_ndjson:
                    seen_ndjson.add(resolved)
                    ndjson_files.append(resolved)
            elif low.endswith(".json") and not is_non_annotation_json(entry):
                resolved = entry.resolve()
                if resolved not in seen_json:
                    seen_json.add(resolved)
                    json_files.append(resolved)

    json_files.sort()
    ndjson_files.sort()

    items: List[Dict[str, Any]] = []
    detected_labels: set[str] = set()
    for jf in json_files:
        if cancel_check is not None and cancel_check():
            return {
                "items": items,
                "ndjson_stores": [],
                "ndjson_total_frames": 0,
                "labels": sorted(detected_labels),
                "cancelled": True,
            }
        status = "valid"
        shapes_len = 0
        image_abs_path: Optional[Path] = None
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                status = "error"
            else:
                image_abs_path = _resolve_audit_image_path(jf, payload=data)
                if image_abs_path is None:
                    status = "missing_image"
                shapes = data.get("shapes", [])
                if not isinstance(shapes, list):
                    status = "error"
                else:
                    shapes_len = len(shapes)
                    if not shapes and status == "valid":
                        status = "empty"
                    for shape in shapes:
                        if isinstance(shape, Mapping):
                            lbl = str(shape.get("label") or "").strip()
                            if lbl:
                                detected_labels.add(lbl)
        except Exception:
            status = "error"

        items.append(
            {
                "json_path": jf,
                "image_path": image_abs_path,
                "status": status,
                "shape_count": shapes_len,
            }
        )

    ndjson_stores: List[Dict[str, Any]] = []
    ndjson_total_frames = 0
    for ndjson_path in ndjson_files:
        if cancel_check is not None and cancel_check():
            return {
                "items": items,
                "ndjson_stores": ndjson_stores,
                "ndjson_total_frames": ndjson_total_frames,
                "labels": sorted(detected_labels),
                "cancelled": True,
            }
        try:
            store = AnnotationStore(ndjson_path)
            frame_count = len(set(store.iter_frame_numbers_fast()))
            ndjson_total_frames += frame_count
            ndjson_stores.append(
                {
                    "store_path": ndjson_path,
                    "frame_count": frame_count,
                    "error": "",
                }
            )
        except Exception as exc:
            ndjson_stores.append(
                {
                    "store_path": ndjson_path,
                    "frame_count": 0,
                    "error": str(exc),
                }
            )

    return {
        "items": items,
        "ndjson_stores": ndjson_stores,
        "ndjson_total_frames": ndjson_total_frames,
        "labels": sorted(detected_labels),
        "cancelled": False,
    }


def audit_project_annotations(
    project_dir: str | Path,
    config: Optional[Mapping[str, Any]] = None,
    *,
    recursive: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Audit annotations across an Annolid project workspace (JSON + NDJSON stores)."""
    root = resolve_project_dir(project_dir).resolve()
    dirs_cfg = dict((config or {}).get("directories") or DEFAULT_PROJECT_DIRECTORIES)
    scan_roots: List[Path] = []
    for key in ("annotations", "videos", "images"):
        rel = str(dirs_cfg.get(key, key) or key).strip()
        if not rel:
            continue
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_dir() and candidate not in scan_roots:
            scan_roots.append(candidate)
    if not scan_roots and root.is_dir():
        scan_roots = [root]
    return scan_annotation_audit(
        scan_roots,
        recursive=recursive,
        cancel_check=cancel_check,
    )


def rename_annotation_file_pair(
    source_path: str | Path,
    new_name: str,
    *,
    label_path: Optional[str | Path] = None,
    sync_sidecar: bool = True,
) -> Dict[str, Any]:
    """Rename a media/annotation file and keep its companion LabelMe `.json` in sync.

    - Validates *new_name* against path separators / traversal.
    - When renaming an image with an existing companion `.json` (and *sync_sidecar=True*),
      also renames the companion `.json` to the matching stem and atomically updates its
      `imagePath` field if it referenced the old image filename.
    - Rolls back the primary rename if companion rename fails.
    """
    src = Path(source_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    ok, reason = validate_project_name(new_name)
    if not ok:
        raise ValueError(reason)

    clean_name = str(new_name).strip()
    target = src.with_name(clean_name)
    if target == src:
        return {
            "source": src,
            "target": target,
            "sidecar_source": None,
            "sidecar_target": None,
            "renamed_sidecar": False,
        }
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")

    sidecar_src: Optional[Path] = None
    sidecar_dst: Optional[Path] = None
    if sync_sidecar and src.suffix.lower() != ".json":
        candidate_json = (
            Path(label_path).expanduser()
            if label_path is not None and str(label_path).strip()
            else src.with_suffix(".json")
        )
        if candidate_json.is_file() and candidate_json.resolve() != src.resolve():
            dst_json = candidate_json.with_name(f"{target.stem}.json")
            if dst_json != candidate_json:
                if dst_json.exists():
                    raise FileExistsError(
                        f"Companion label target already exists: {dst_json}"
                    )
                sidecar_src = candidate_json
                sidecar_dst = dst_json

    src.rename(target)
    renamed_sidecar = False
    if sidecar_src is not None and sidecar_dst is not None:
        try:
            sidecar_src.rename(sidecar_dst)
            renamed_sidecar = True
        except Exception:
            # Roll back primary rename so image/label stay paired.
            try:
                target.rename(src)
            except OSError:
                pass
            raise

        # Keep LabelMe imagePath synchronized atomically when it points to old filename.
        try:
            payload = json.loads(sidecar_dst.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("imagePath"):
                old_img_ref = str(payload["imagePath"])
                if Path(old_img_ref).name == src.name:
                    if sidecar_dst.parent.resolve() == target.parent.resolve():
                        payload["imagePath"] = target.name
                    else:
                        try:
                            payload["imagePath"] = Path(
                                os.path.relpath(target, sidecar_dst.parent)
                            ).as_posix()
                        except ValueError:
                            payload["imagePath"] = target.name
                    atomic_write_json(
                        sidecar_dst, payload, indent=2, ensure_ascii=False
                    )
        except Exception as exc:
            logger.debug(
                "Could not update imagePath inside renamed sidecar %s: %s",
                sidecar_dst,
                exc,
            )

    return {
        "source": src,
        "target": target,
        "sidecar_source": sidecar_src,
        "sidecar_target": sidecar_dst,
        "renamed_sidecar": renamed_sidecar,
    }


def upsert_recent_project_entries(
    existing: Iterable[Mapping[str, Any] | str],
    project_path: str | Path,
    *,
    name: str = "",
    timestamp: Optional[str] = None,
    max_entries: int = 12,
) -> List[Dict[str, str]]:
    """Insert or bump a project at the top of a recent-projects list (deduplicated)."""
    resolved = str(resolve_project_dir(project_path).resolve())
    display_name = str(name or Path(resolved).name or "Project").strip()
    ts = str(timestamp or current_iso_timestamp())

    out: List[Dict[str, str]] = [
        {"name": display_name, "path": resolved, "last_opened": ts}
    ]
    seen = {resolved}

    for raw in existing or ():
        if isinstance(raw, Mapping):
            raw_path = str(raw.get("path") or "").strip()
            raw_name = str(raw.get("name") or "").strip()
            raw_ts = str(raw.get("last_opened") or "").strip()
        else:
            raw_path = str(raw or "").strip()
            raw_name = ""
            raw_ts = ""
        if not raw_path:
            continue
        norm = str(resolve_project_dir(raw_path).resolve())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(
            {
                "name": raw_name or Path(norm).name,
                "path": norm,
                "last_opened": raw_ts or ts,
            }
        )
        if len(out) >= max(1, int(max_entries)):
            break

    return out


def prune_missing_recent_entries(
    entries: Iterable[Mapping[str, Any] | str],
) -> List[Dict[str, str]]:
    """Filter recent-project entries to those whose directory still exists on disk."""
    kept: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in entries or ():
        if isinstance(raw, Mapping):
            raw_path = str(raw.get("path") or "").strip()
            raw_name = str(raw.get("name") or "").strip()
            raw_ts = str(raw.get("last_opened") or "").strip()
        else:
            raw_path = str(raw or "").strip()
            raw_name = ""
            raw_ts = ""
        if not raw_path:
            continue
        p = resolve_project_dir(raw_path)
        if not p.exists():
            continue
        norm = str(p.resolve())
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(
            {
                "name": raw_name or p.name,
                "path": norm,
                "last_opened": raw_ts,
            }
        )
    return kept


__all__ = [
    "DEFAULT_ANNOLIDIGNORE",
    "DEFAULT_IMAGE_EXTENSIONS",
    "DEFAULT_MODEL_EXTENSIONS",
    "DEFAULT_PROJECT_DIRECTORIES",
    "DEFAULT_VIDEO_EXTENSIONS",
    "PROJECT_CONFIG_FILENAME",
    "audit_project_annotations",
    "compute_project_workflow_status",
    "current_iso_timestamp",
    "deep_merge_dict",
    "default_project_config",
    "ensure_project_directories",
    "export_project_bundle",
    "is_non_annotation_json",
    "is_schema_only_project_config",
    "normalize_project_config",
    "prune_missing_recent_entries",
    "register_project_videos",
    "rename_annotation_file_pair",
    "resolve_project_dir",
    "resolve_project_video_paths",
    "scan_annotation_audit",
    "scan_project_statistics",
    "upsert_recent_project_entries",
    "validate_project_name",
    "write_project_ignore_file",
]
