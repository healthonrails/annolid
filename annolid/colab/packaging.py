from __future__ import annotations

import json
import shlex
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from .models import ColabPredictRequest, ColabTrainRequest

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

_MODEL_EXTRAS: dict[str, tuple[str, ...]] = {
    "yolo": ("yolo",),
    "yolo_labelme": ("yolo",),
    "dino_kpseg": ("ml", "training"),
    "maskrcnn_detectron2": ("ml", "training"),
    "keypoint_rcnn": ("ml", "training"),
    "tcn_behavior": ("ml", "training"),
    "polygon_classifier": ("ml", "scientific"),
    "behavior_classifier": ("ml", "scientific"),
}


def required_extras_for_model(model_name: str) -> tuple[str, ...]:
    key = str(model_name or "").strip().lower()
    return _MODEL_EXTRAS.get(key, ("ml",))


def build_pip_install_spec(base_spec: str, model_name: str) -> str:
    """Attach model-appropriate Annolid extras when installing from PyPI/git."""
    raw = str(base_spec or "annolid").strip() or "annolid"
    if "[" in raw and "]" in raw:
        return raw
    extras = ",".join(required_extras_for_model(model_name))
    if not extras:
        return raw
    # git+https://... or editable path vs package name
    if raw.startswith(("git+", "http://", "https://")):
        return f"annolid[{extras}] @ {raw}"
    return f"{raw}[{extras}]"


def _is_local_weight_file(weights: Optional[str]) -> Optional[Path]:
    raw = str(weights or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    return None


@dataclass(frozen=True)
class StagedInputBundle:
    archive_path: Path
    dataset_entry: str  # relative path inside extracted input dir (e.g. "data.yaml", "dataset", "source/video.mp4")
    weights_entry: Optional[str] = (
        None  # relative path inside extracted input dir if local weights were bundled
    )
    extra_dataset_entry: Optional[str] = (
        None  # optional COCO dataset dir for maskrcnn predict
    )
    file_count: int = 0
    archive_bytes: int = 0
    notes: tuple[str, ...] = ()


def _validate_allowed_read(
    path: Path, allowed_read_roots: Optional[Sequence[Path]] = None
) -> Path:
    resolved = path.expanduser().resolve()
    if not allowed_read_roots:
        return resolved
    for root in allowed_read_roots:
        root_resolved = Path(root).expanduser().resolve()
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            return resolved
    raise PermissionError(f"Refusing to package path outside allowed roots: {resolved}")


def _write_dir_to_zip(
    zf: zipfile.ZipFile,
    src_dir: Path,
    arc_prefix: str,
    *,
    allowed_read_roots: Optional[Sequence[Path]] = None,
) -> int:
    count = 0
    src_resolved = _validate_allowed_read(src_dir, allowed_read_roots)
    for path in sorted(src_resolved.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_resolved)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        # Reject symlinks escaping src_dir or allowed_read_roots.
        resolved_target = path.resolve()
        if path.is_symlink() and not resolved_target.is_relative_to(src_resolved):
            raise PermissionError(
                f"Refusing to package symlink escaping dataset root: {path} -> {resolved_target}"
            )
        _validate_allowed_read(resolved_target, allowed_read_roots)
        arcname = f"{arc_prefix}/{rel.as_posix()}" if arc_prefix else rel.as_posix()
        zf.write(resolved_target, arcname)
        count += 1
    return count


def _resolve_yolo_split_path(
    base_root: Path, yaml_dir: Path, entry: str
) -> Optional[Path]:
    raw = Path(str(entry).strip()).expanduser()
    candidates = [raw] if raw.is_absolute() else [base_root / raw, yaml_dir / raw]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _find_corresponding_yolo_labels_dir(images_dir: Path) -> Optional[Path]:
    parts = list(images_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            cand = Path(*parts[:idx], "labels", *parts[idx + 1 :])
            if cand.is_dir():
                return cand.resolve()
    sibling = images_dir.parent / "labels"
    if sibling.is_dir():
        return sibling.resolve()
    return None


def _package_yolo_yaml_dataset(
    yaml_path: Path,
    archive_path: Path,
    *,
    allowed_read_roots: Optional[Sequence[Path]] = None,
) -> StagedInputBundle:
    """Create a portable zip archive from a YOLO data.yaml without mutating the source."""
    yaml_resolved = _validate_allowed_read(yaml_path, allowed_read_roots)
    raw_cfg = yaml.safe_load(yaml_resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_resolved}")

    yaml_dir = yaml_resolved.parent
    raw_base = raw_cfg.get("path")
    if raw_base:
        base_cand = Path(str(raw_base)).expanduser()
        base_root = (
            base_cand.resolve()
            if base_cand.is_absolute()
            else (yaml_dir / base_cand).resolve()
        )
        if not base_root.exists():
            base_root = yaml_dir
    else:
        base_root = yaml_dir
    _validate_allowed_read(base_root, allowed_read_roots)

    portable_cfg: dict[str, Any] = dict(raw_cfg)
    portable_cfg["path"] = "."
    file_count = 0
    notes: list[str] = []

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for split_key in ("train", "val", "test"):
            split_val = raw_cfg.get(split_key)
            if not split_val:
                continue
            entries = split_val if isinstance(split_val, list) else [split_val]
            portable_entries: list[str] = []
            for idx, entry in enumerate(entries):
                resolved = _resolve_yolo_split_path(base_root, yaml_dir, str(entry))
                if resolved is None:
                    notes.append(f"Skipped missing split entry {split_key}={entry}")
                    continue
                resolved = _validate_allowed_read(resolved, allowed_read_roots)
                suffix = f"_{idx}" if len(entries) > 1 else ""
                if resolved.is_dir():
                    img_arc = f"images/{split_key}{suffix}"
                    file_count += _write_dir_to_zip(
                        zf,
                        resolved,
                        img_arc,
                        allowed_read_roots=allowed_read_roots,
                    )
                    labels_dir = _find_corresponding_yolo_labels_dir(resolved)
                    if labels_dir is not None and labels_dir != resolved:
                        lbl_arc = f"labels/{split_key}{suffix}"
                        file_count += _write_dir_to_zip(
                            zf,
                            labels_dir,
                            lbl_arc,
                            allowed_read_roots=allowed_read_roots,
                        )
                    portable_entries.append(img_arc)
                elif resolved.is_file():
                    # Handle .txt image list file by bundling listed images + matching labels.
                    if resolved.suffix.lower() == ".txt":
                        img_arc = f"images/{split_key}{suffix}"
                        lbl_arc = f"labels/{split_key}{suffix}"
                        listed_arcs: list[str] = []
                        for line in resolved.read_text(encoding="utf-8").splitlines():
                            item = line.strip()
                            if not item:
                                continue
                            img_path = Path(item).expanduser()
                            if not img_path.is_absolute():
                                img_path = (resolved.parent / img_path).resolve()
                            else:
                                img_path = img_path.resolve()
                            img_path = _validate_allowed_read(
                                img_path, allowed_read_roots
                            )
                            if not img_path.is_file():
                                continue
                            arc_img = f"{img_arc}/{img_path.name}"
                            zf.write(img_path, arc_img)
                            file_count += 1
                            listed_arcs.append(f"./{arc_img}")
                            # Check matching label .txt
                            lbl_cand = img_path.with_suffix(".txt")
                            if not lbl_cand.is_file():
                                parts = list(img_path.parts)
                                for p_idx in range(len(parts) - 1, -1, -1):
                                    if parts[p_idx].lower() == "images":
                                        alt = Path(
                                            *parts[:p_idx],
                                            "labels",
                                            *parts[p_idx + 1 :],
                                        ).with_suffix(".txt")
                                        if alt.is_file():
                                            lbl_cand = alt
                                            break
                            if lbl_cand.is_file():
                                lbl_cand = _validate_allowed_read(
                                    lbl_cand, allowed_read_roots
                                )
                                zf.write(lbl_cand, f"{lbl_arc}/{lbl_cand.name}")
                                file_count += 1
                        manifest_name = f"{split_key}{suffix}.txt"
                        zf.writestr(manifest_name, "\n".join(listed_arcs) + "\n")
                        file_count += 1
                        portable_entries.append(manifest_name)
                    else:
                        arc_file = f"splits/{split_key}{suffix}/{resolved.name}"
                        zf.write(resolved, arc_file)
                        file_count += 1
                        portable_entries.append(arc_file)
            if portable_entries:
                portable_cfg[split_key] = (
                    portable_entries[0]
                    if len(portable_entries) == 1
                    else portable_entries
                )

        # Preserve pose_schema.json or labels.txt next to data.yaml if present
        for extra_name in ("pose_schema.json", "labels.txt", "classes.txt"):
            extra_file = yaml_dir / extra_name
            if extra_file.is_file():
                zf.write(extra_file, extra_name)
                file_count += 1

        zf.writestr(
            "data.yaml",
            yaml.safe_dump(portable_cfg, sort_keys=False),
        )
        file_count += 1

    return StagedInputBundle(
        archive_path=archive_path,
        dataset_entry="data.yaml",
        file_count=file_count,
        archive_bytes=int(archive_path.stat().st_size),
        notes=tuple(notes),
    )


def _package_generic_yaml_spec(
    yaml_path: Path,
    archive_path: Path,
    *,
    allowed_read_roots: Optional[Sequence[Path]] = None,
) -> StagedInputBundle:
    """Package a LabelMe/COCO/DINO spec.yaml + referenced dataset dirs portably."""
    yaml_resolved = _validate_allowed_read(yaml_path, allowed_read_roots)
    raw_cfg = yaml.safe_load(yaml_resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_resolved}")

    yaml_dir = yaml_resolved.parent
    portable_cfg: dict[str, Any] = dict(raw_cfg)
    if "path" in portable_cfg:
        portable_cfg["path"] = "."

    file_count = 0
    notes: list[str] = []

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in (
            "train",
            "val",
            "valid",
            "test",
            "train_images",
            "val_images",
            "test_images",
            "train_ann",
            "val_ann",
            "test_ann",
            "images_dir",
            "annotations",
        ):
            val = raw_cfg.get(key)
            if not val or not isinstance(val, (str, Path)):
                continue
            cand = Path(str(val)).expanduser()
            resolved = (
                cand.resolve() if cand.is_absolute() else (yaml_dir / cand).resolve()
            )
            resolved = _validate_allowed_read(resolved, allowed_read_roots)
            if not resolved.exists():
                notes.append(f"Skipped missing spec path {key}={val}")
                continue
            if resolved.is_dir():
                arc_dir = f"data/{key}"
                file_count += _write_dir_to_zip(
                    zf,
                    resolved,
                    arc_dir,
                    allowed_read_roots=allowed_read_roots,
                )
                portable_cfg[key] = arc_dir
            elif resolved.is_file():
                arc_file = f"data/{key}/{resolved.name}"
                zf.write(resolved, arc_file)
                file_count += 1
                portable_cfg[key] = arc_file

        zf.writestr(
            yaml_resolved.name,
            yaml.safe_dump(portable_cfg, sort_keys=False),
        )
        file_count += 1

    return StagedInputBundle(
        archive_path=archive_path,
        dataset_entry=yaml_resolved.name,
        file_count=file_count,
        archive_bytes=int(archive_path.stat().st_size),
        notes=tuple(notes),
    )


def _validate_zip_members(zip_path: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            member = info.filename
            if member.startswith("/") or member.startswith("\\"):
                raise ValueError(f"Unsafe absolute path in zip archive: {member}")
            parts = Path(member).parts
            if ".." in parts:
                raise ValueError(
                    f"Unsafe path traversal entry in zip archive: {member}"
                )
            names.append(member)
    return names


def package_training_dataset(
    dataset_path: str | Path,
    *,
    model: str,
    staging_dir: Path,
    weights: Optional[str] = None,
    allowed_read_roots: Optional[Sequence[Path]] = None,
) -> StagedInputBundle:
    """Package a local dataset (and optional local weights) into a portable Colab zip."""
    src = Path(str(dataset_path).strip()).expanduser().resolve()
    src = _validate_allowed_read(src, allowed_read_roots)
    if not src.exists():
        raise FileNotFoundError(f"Dataset path not found: {src}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path = staging_dir / "input_dataset.zip"
    model_key = str(model or "").strip().lower()
    local_weights = _is_local_weight_file(weights)
    if local_weights is not None:
        local_weights = _validate_allowed_read(local_weights, allowed_read_roots)
    weights_entry: Optional[str] = None

    if src.is_file() and src.suffix.lower() == ".zip":
        members = _validate_zip_members(src)
        shutil.copy2(src, archive_path)
        entry = "."
        for candidate in ("data.yaml", "labelme_spec.yaml", "coco_spec.yaml"):
            matches = [
                m for m in members if m.endswith(f"/{candidate}") or m == candidate
            ]
            if matches:
                entry = sorted(matches, key=len)[0]
                break
        if local_weights is not None:
            weights_entry = f"weights/{local_weights.name}"
            with zipfile.ZipFile(
                archive_path, "a", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.write(local_weights, weights_entry)
        return StagedInputBundle(
            archive_path=archive_path,
            dataset_entry=entry,
            weights_entry=weights_entry,
            file_count=len(members) + (1 if weights_entry else 0),
            archive_bytes=int(archive_path.stat().st_size),
        )

    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        if model_key == "yolo" or src.name.lower() == "data.yaml":
            bundle = _package_yolo_yaml_dataset(
                src, archive_path, allowed_read_roots=allowed_read_roots
            )
        else:
            bundle = _package_generic_yaml_spec(
                src, archive_path, allowed_read_roots=allowed_read_roots
            )
    elif src.is_dir():
        data_yaml = src / "data.yaml"
        if model_key == "yolo" and data_yaml.is_file():
            bundle = _package_yolo_yaml_dataset(
                data_yaml, archive_path, allowed_read_roots=allowed_read_roots
            )
        else:
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                count = _write_dir_to_zip(
                    zf, src, "dataset", allowed_read_roots=allowed_read_roots
                )
            entry = "dataset"
            if (src / "data.yaml").is_file():
                entry = "dataset/data.yaml"
            elif (src / "labelme_spec.yaml").is_file():
                entry = "dataset/labelme_spec.yaml"
            elif (src / "coco_spec.yaml").is_file():
                entry = "dataset/coco_spec.yaml"
            bundle = StagedInputBundle(
                archive_path=archive_path,
                dataset_entry=entry,
                file_count=count,
                archive_bytes=int(archive_path.stat().st_size),
            )
    else:
        raise ValueError(
            f"Unsupported dataset input {src}. Provide a YOLO data.yaml, "
            "LabelMe/COCO spec.yaml, dataset folder, or .zip archive."
        )

    if local_weights is not None:
        weights_entry = f"weights/{local_weights.name}"
        with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(local_weights, weights_entry)
        return StagedInputBundle(
            archive_path=bundle.archive_path,
            dataset_entry=bundle.dataset_entry,
            weights_entry=weights_entry,
            file_count=bundle.file_count + 1,
            archive_bytes=int(archive_path.stat().st_size),
            notes=bundle.notes,
        )
    return bundle


def package_inference_inputs(
    source_path: str | Path,
    *,
    staging_dir: Path,
    weights: Optional[str] = None,
    dataset_dir: Optional[str | Path] = None,
    allowed_read_roots: Optional[Sequence[Path]] = None,
) -> StagedInputBundle:
    """Package inference media (video/image/folder) and optional local weights into a zip."""
    src = Path(str(source_path).strip()).expanduser().resolve()
    src = _validate_allowed_read(src, allowed_read_roots)
    if not src.exists():
        raise FileNotFoundError(f"Inference source not found: {src}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path = staging_dir / "input_inference.zip"
    local_weights = _is_local_weight_file(weights)
    if local_weights is not None:
        local_weights = _validate_allowed_read(local_weights, allowed_read_roots)
    weights_entry: Optional[str] = None
    extra_dataset_entry: Optional[str] = None
    file_count = 0

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if src.is_file():
            source_entry = f"source/{src.name}"
            zf.write(src, source_entry)
            file_count += 1
        else:
            source_entry = "source"
            file_count += _write_dir_to_zip(
                zf, src, source_entry, allowed_read_roots=allowed_read_roots
            )

        if local_weights is not None:
            weights_entry = f"weights/{local_weights.name}"
            zf.write(local_weights, weights_entry)
            file_count += 1

        if dataset_dir:
            ds_path = Path(str(dataset_dir).strip()).expanduser().resolve()
            ds_path = _validate_allowed_read(ds_path, allowed_read_roots)
            if not ds_path.exists():
                raise FileNotFoundError(f"Dataset directory not found: {ds_path}")
            if ds_path.is_dir():
                extra_dataset_entry = "dataset_meta"
                file_count += _write_dir_to_zip(
                    zf,
                    ds_path,
                    extra_dataset_entry,
                    allowed_read_roots=allowed_read_roots,
                )
            else:
                extra_dataset_entry = f"dataset_meta/{ds_path.name}"
                zf.write(ds_path, extra_dataset_entry)
                file_count += 1

    return StagedInputBundle(
        archive_path=archive_path,
        dataset_entry=source_entry,
        weights_entry=weights_entry,
        extra_dataset_entry=extra_dataset_entry,
        file_count=file_count,
        archive_bytes=int(archive_path.stat().st_size),
    )


def safe_extract_archive(archive_path: str | Path, dest_dir: str | Path) -> list[Path]:
    """Safely extract a .zip or .tar.gz archive with path-traversal protection."""
    src = Path(archive_path).expanduser().resolve()
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src, "r") as zf:
            for info in zf.infolist():
                member_path = (dest / info.filename).resolve()
                if dest != member_path and dest not in member_path.parents:
                    raise ValueError(
                        f"Refusing to extract archive member outside destination: {info.filename}"
                    )
                if info.is_dir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src_f, open(member_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                extracted.append(member_path)
        return extracted

    if tarfile.is_tarfile(src):
        with tarfile.open(src, "r:*") as tf:
            for member in tf.getmembers():
                member_path = (dest / member.name).resolve()
                if dest != member_path and dest not in member_path.parents:
                    raise ValueError(
                        f"Refusing to extract tar member outside destination: {member.name}"
                    )
                if member.isdir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                member_path.parent.mkdir(parents=True, exist_ok=True)
                src_obj = tf.extractfile(member)
                if src_obj is None:
                    continue
                with src_obj, open(member_path, "wb") as dst_f:
                    shutil.copyfileobj(src_obj, dst_f)
                extracted.append(member_path)
        return extracted

    raise ValueError(f"Unsupported archive format: {src}")


def build_remote_train_command(
    request: ColabTrainRequest,
    *,
    remote_input_root: str,
    remote_output_root: str,
    dataset_entry: str,
    weights_entry: Optional[str] = None,
) -> list[str]:
    """Construct the exact remote `annolid-run train <model> ...` command."""
    model = str(request.model or "yolo").strip()
    remote_data = (
        remote_input_root
        if dataset_entry in {"", "."}
        else f"{remote_input_root.rstrip('/')}/{dataset_entry.lstrip('/')}"
    )
    remote_weights: Optional[str] = None
    if weights_entry:
        remote_weights = f"{remote_input_root.rstrip('/')}/{weights_entry.lstrip('/')}"
    elif request.weights and not _is_local_weight_file(request.weights):
        remote_weights = str(request.weights).strip()

    cmd: list[str] = ["annolid-run", "train", model]

    if model == "yolo":
        cmd.extend(
            ["--data", remote_data, "--project", remote_output_root, "--name", "run"]
        )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.preset:
            cmd.extend(["--preset", str(request.preset)])
        if request.epochs is not None:
            cmd.extend(["--epochs", str(int(request.epochs))])
        if request.batch is not None:
            cmd.extend(["--batch", str(int(request.batch))])
        if request.imgsz is not None:
            cmd.extend(["--imgsz", str(int(request.imgsz))])
        if request.plots:
            cmd.append("--plots")
        if request.resume:
            cmd.append("--resume")
    elif model == "dino_kpseg":
        cmd.extend(["--data", remote_data, "--output", remote_output_root])
        if request.data_format:
            cmd.extend(["--data-format", str(request.data_format)])
        if remote_weights:
            cmd.extend(["--model-name", remote_weights])
        if request.epochs is not None:
            cmd.extend(["--epochs", str(int(request.epochs))])
        if request.batch is not None:
            cmd.extend(["--batch", str(int(request.batch))])
    elif model in {"maskrcnn_detectron2", "keypoint_rcnn"}:
        cmd.extend(["--dataset-dir", remote_data, "--output-dir", remote_output_root])
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.max_iterations is not None:
            cmd.extend(["--max-iterations", str(int(request.max_iterations))])
        elif request.epochs is not None:
            cmd.extend(["--max-iterations", str(int(request.epochs))])
        if request.batch is not None:
            cmd.extend(["--batch-size", str(int(request.batch))])
    else:
        cmd.extend(["--data", remote_data, "--output", remote_output_root])
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.epochs is not None:
            cmd.extend(["--epochs", str(int(request.epochs))])
        if request.batch is not None:
            cmd.extend(["--batch", str(int(request.batch))])

    for extra in request.extra_args:
        token = str(extra).strip()
        if token:
            cmd.append(token)
    return cmd


def build_remote_predict_command(
    request: ColabPredictRequest,
    *,
    remote_input_root: str,
    remote_output_root: str,
    source_entry: str,
    weights_entry: Optional[str] = None,
    extra_dataset_entry: Optional[str] = None,
) -> list[str]:
    """Construct the exact remote `annolid-run predict <model> ...` command."""
    model = str(request.model or "yolo_labelme").strip()
    remote_source = f"{remote_input_root.rstrip('/')}/{source_entry.lstrip('/')}"
    remote_weights: Optional[str] = None
    if weights_entry:
        remote_weights = f"{remote_input_root.rstrip('/')}/{weights_entry.lstrip('/')}"
    elif request.weights and not _is_local_weight_file(request.weights):
        remote_weights = str(request.weights).strip()

    cmd: list[str] = ["annolid-run", "predict", model]

    if model == "yolo_labelme":
        cmd.extend(
            [
                "--source",
                remote_source,
                "--output-dir",
                f"{remote_output_root}/predictions",
            ]
        )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.conf is not None:
            cmd.extend(["--conf", str(float(request.conf))])
        if request.iou is not None:
            cmd.extend(["--iou", str(float(request.iou))])
        if request.imgsz is not None:
            cmd.extend(["--imgsz", str(int(request.imgsz))])
        if request.batch is not None:
            cmd.extend(["--batch", str(int(request.batch))])
        if request.save_csv:
            cmd.append("--save-csv")
        if request.save_video:
            cmd.extend(["--save-video", f"{remote_output_root}/preview.mp4"])
        if request.no_tracking:
            cmd.append("--no-tracking")
        if request.classes:
            cmd.extend(["--classes", str(request.classes)])
    elif model == "yolo":
        cmd.extend(
            [
                "--source",
                remote_source,
                "--project",
                remote_output_root,
                "--name",
                "predict",
                "--save",
            ]
        )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.conf is not None:
            cmd.extend(["--conf", str(float(request.conf))])
        if request.iou is not None:
            cmd.extend(["--iou", str(float(request.iou))])
        if request.imgsz is not None:
            cmd.extend(["--imgsz", str(int(request.imgsz))])
        if not request.no_tracking:
            cmd.append("--track")
        if request.classes:
            cmd.extend(["--classes", str(request.classes)])
    elif model == "dino_kpseg":
        cmd.extend(
            [
                "--image",
                remote_source,
                "--out",
                f"{remote_output_root}/predictions.json",
            ]
        )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.conf is not None:
            cmd.extend(["--threshold", str(float(request.conf))])
    elif model == "keypoint_rcnn":
        cmd.extend(
            [
                "--source",
                remote_source,
                "--output-dir",
                f"{remote_output_root}/predictions",
                "--no-display",
            ]
        )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.conf is not None:
            cmd.extend(["--score-threshold", str(float(request.conf))])
        if request.batch is not None:
            cmd.extend(["--batch-size", str(int(request.batch))])
        if request.save_video:
            cmd.extend(["--save-video", f"{remote_output_root}/preview.mp4"])
    elif model == "maskrcnn_detectron2":
        cmd.extend(["--image", remote_source, "--no-display"])
        if extra_dataset_entry:
            cmd.extend(
                [
                    "--dataset-dir",
                    f"{remote_input_root.rstrip('/')}/{extra_dataset_entry.lstrip('/')}",
                ]
            )
        if remote_weights:
            cmd.extend(["--weights", remote_weights])
        if request.conf is not None:
            cmd.extend(["--score-threshold", str(float(request.conf))])
    else:
        cmd.extend(["--source", remote_source])
        if remote_weights:
            cmd.extend(["--weights", remote_weights])

    for extra in request.extra_args:
        token = str(extra).strip()
        if token:
            cmd.append(token)
    return cmd


def build_remote_runner_script(
    *,
    job_id: str,
    kind: str,
    model: str,
    install_spec: str,
    remote_job_root: str,
    input_archive_remote_path: str,
    remote_command: Sequence[str],
    drive_job_dir: Optional[str] = None,
) -> str:
    """Generate a self-contained Python script executed on a remote Colab runtime."""
    pip_spec = build_pip_install_spec(install_spec, model)
    cmd_json = json.dumps(list(remote_command))
    drive_sync_block = (
        f"DRIVE_JOB_DIR = Path({drive_job_dir!r})\n"
        if drive_job_dir
        else "DRIVE_JOB_DIR = None\n"
    )
    return f'''"""Auto-generated Annolid Colab runner for job {job_id}."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JOB_ID = {job_id!r}
JOB_KIND = {kind!r}
MODEL_NAME = {model!r}
PIP_SPEC = {pip_spec!r}
JOB_ROOT = Path({remote_job_root!r})
INPUT_ARCHIVE = Path({input_archive_remote_path!r})
INPUT_DIR = JOB_ROOT / "input"
OUTPUT_DIR = JOB_ROOT / "output"
OUTPUT_ARCHIVE = JOB_ROOT / "output_bundle.zip"
RESULT_MANIFEST = JOB_ROOT / "job_result.json"
REMOTE_CMD = {cmd_json}
{drive_sync_block}

def log(msg: str) -> None:
    print(f"[annolid-colab][{{JOB_ID}}] {{msg}}", flush=True)


def run_cmd(cmd: list[str], *, check: bool = True) -> int:
    log("Running: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = int(proc.wait())
    if check and rc != 0:
        raise RuntimeError(f"Command exited with code {{rc}}: {{cmd}}")
    return rc


def filter_supported_cli_flags(cmd: list[str]) -> list[str]:
    if len(cmd) < 3:
        return cmd
    sub_idx = -1
    for idx, tok in enumerate(cmd[:-1]):
        if tok in {{"train", "predict"}}:
            sub_idx = idx
            break
    if sub_idx == -1 or sub_idx + 1 >= len(cmd):
        return cmd
    help_cmd = [*cmd[: sub_idx + 2], "--help-model"]
    try:
        proc = subprocess.run(help_cmd, capture_output=True, text=True, timeout=30)
        help_text = (proc.stdout or "") + "\\n" + (proc.stderr or "")
    except Exception:
        return cmd
    if proc.returncode != 0 or "--help" not in help_text:
        return cmd
    supported = set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))
    optional_value_flags = {{
        "--name",
        "--preset",
        "--conf",
        "--iou",
        "--imgsz",
        "--batch",
        "--save-video",
    }}
    optional_bool_flags = {{"--resume", "--save-csv", "--plots"}}
    filtered = list(cmd[: sub_idx + 2])
    i = sub_idx + 2
    while i < len(cmd):
        tok = cmd[i]
        if tok in optional_value_flags and tok not in supported:
            log(
                f"[WARN] Installed remote annolid does not support {{tok}}; skipping "
                "(pass --annolid-spec git+https://github.com/healthonrails/annolid.git for latest CLI flags)."
            )
            i += 2 if i + 1 < len(cmd) and not cmd[i + 1].startswith("-") else 1
            continue
        if tok in optional_bool_flags and tok not in supported:
            log(
                f"[WARN] Installed remote annolid does not support {{tok}}; skipping "
                "(pass --annolid-spec git+https://github.com/healthonrails/annolid.git for latest CLI flags)."
            )
            i += 1
            continue
        filtered.append(tok)
        i += 1
    return filtered


def _pip_install(spec: str) -> None:
    uv_bin = shutil.which("uv")
    if uv_bin:
        rc = run_cmd(
            [uv_bin, "pip", "install", "--system", "--no-progress", spec],
            check=False,
        )
        if rc == 0:
            return
    run_cmd([sys.executable, "-m", "pip", "install", "--no-input", "--upgrade", spec])


def ensure_annolid_installed() -> None:
    marker_path = Path("/content/.annolid_colab_env.json")
    installed_specs: list[str] = []
    if marker_path.is_file():
        try:
            raw = json.loads(marker_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("specs"), list):
                installed_specs = [str(s) for s in raw["specs"]]
        except Exception:
            installed_specs = []

    already_installed = PIP_SPEC in installed_specs
    if already_installed:
        probe = subprocess.run(
            [sys.executable, "-c", "import annolid"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            log(f"Reusing already-installed Annolid runtime ({{PIP_SPEC}}) in this session.")
        else:
            already_installed = False

    if not already_installed:
        log(f"Ensuring Annolid runtime ({{PIP_SPEC}})...")
        _pip_install(PIP_SPEC)
        if PIP_SPEC not in installed_specs:
            installed_specs.append(PIP_SPEC)
        try:
            marker_path.write_text(
                json.dumps({{"specs": installed_specs}}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Legacy PyPI Annolid (<=1.7.1) imported qtpy eagerly inside yolo_labelme.
    # Probe headlessly and install lean PySide6-Essentials only when needed.
    if MODEL_NAME == "yolo_labelme":
        qt_probe = subprocess.run(
            [sys.executable, "-c", "import annolid.segmentation.yolos"],
            capture_output=True,
            text=True,
        )
        if qt_probe.returncode != 0 and "QtBindingsNotFoundError" in (
            (qt_probe.stderr or "") + (qt_probe.stdout or "")
        ):
            log(
                "Installed Annolid build requires Qt bindings for yolo_labelme; "
                "installing PySide6-Essentials..."
            )
            _pip_install("PySide6-Essentials>=6.7.0")


def main() -> None:
    started = time.time()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Log hardware
    if shutil.which("nvidia-smi"):
        run_cmd(["nvidia-smi"], check=False)

    # 2. Ensure Annolid and model extras are installed (cached across warm sessions)
    ensure_annolid_installed()

    # 3. Extract staged input bundle safely
    if not INPUT_ARCHIVE.is_file():
        raise FileNotFoundError(f"Staged input archive missing: {{INPUT_ARCHIVE}}")
    log(f"Extracting {{INPUT_ARCHIVE}} -> {{INPUT_DIR}}")
    with zipfile.ZipFile(INPUT_ARCHIVE, "r") as zf:
        for info in zf.infolist():
            dest = (INPUT_DIR / info.filename).resolve()
            if INPUT_DIR.resolve() != dest and INPUT_DIR.resolve() not in dest.parents:
                raise ValueError(f"Unsafe archive entry: {{info.filename}}")
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src_f, open(dest, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
    try:
        INPUT_ARCHIVE.unlink()
    except OSError:
        pass

    # Resolve relative `path: .` in extracted YOLO data.yaml to the absolute
    # remote directory because Ultralytics resolves `path: .` against cwd (`/content`).
    try:
        import yaml as _yaml

        for _yml in list(INPUT_DIR.glob("data.yaml")) + list(INPUT_DIR.glob("*/data.yaml")):
            _cfg = _yaml.safe_load(_yml.read_text(encoding="utf-8"))
            if isinstance(_cfg, dict) and str(_cfg.get("path", "")).strip() in {{"", "."}}:
                _cfg["path"] = str(_yml.parent.resolve())
                _yml.write_text(_yaml.safe_dump(_cfg, sort_keys=False), encoding="utf-8")
                log(f"Resolved dataset YAML path -> {{_cfg['path']}}")
    except Exception as _yaml_exc:
        log(f"[WARN] Could not normalize extracted data.yaml path: {{_yaml_exc}}")

    # 4. Run Annolid training or inference
    cmd = list(REMOTE_CMD)
    if cmd and cmd[0] == "annolid-run" and not shutil.which("annolid-run"):
        cmd = [sys.executable, "-m", "annolid.engine.cli", *cmd[1:]]
    cmd = filter_supported_cli_flags(cmd)
    run_cmd(cmd, check=True)

    # 5. Bundle outputs for download
    artifacts: list[str] = []
    with zipfile.ZipFile(OUTPUT_ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(OUTPUT_DIR).as_posix()
            zf.write(path, rel)
            artifacts.append(rel)

    elapsed_s = round(time.time() - started, 2)
    result = {{
        "job_id": JOB_ID,
        "kind": JOB_KIND,
        "model": MODEL_NAME,
        "status": "completed",
        "elapsed_seconds": elapsed_s,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "output_archive": str(OUTPUT_ARCHIVE),
    }}
    RESULT_MANIFEST.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if DRIVE_JOB_DIR is not None and DRIVE_JOB_DIR.parent.exists():
        DRIVE_JOB_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_ARCHIVE, DRIVE_JOB_DIR / OUTPUT_ARCHIVE.name)
        shutil.copy2(RESULT_MANIFEST, DRIVE_JOB_DIR / RESULT_MANIFEST.name)
        log(f"Synced outputs to Google Drive: {{DRIVE_JOB_DIR}}")

    log(f"Finished in {{elapsed_s}}s ({{len(artifacts)}} artifacts).")
    print("ANNOLID_COLAB_RESULT_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _exc:
        print(
            "ANNOLID_COLAB_ERROR_JSON="
            + json.dumps({{"job_id": JOB_ID, "status": "failed", "error": str(_exc)}}),
            flush=True,
        )
        raise
'''


def format_command_shell(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(tok)) for tok in cmd)
