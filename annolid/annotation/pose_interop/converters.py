"""Unified high-level import and export converters for pose/tracking formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from annolid.annotation.pose_interop.deeplabcut import (
    find_dlc_analysis_file_for_video,
    read_dlc_analysis,
    read_dlc_training_project,
    write_dlc_analysis_csv,
    write_dlc_analysis_h5,
    write_dlc_training_project,
)
from annolid.annotation.pose_interop.labelme import (
    read_pose_from_coco_json,
    read_pose_from_labelme_dir,
    read_pose_from_tracking_csv,
    write_pose_to_coco_json,
    write_pose_to_labelme_dir,
)
from annolid.annotation.pose_interop.model import (
    PoseDataset,
    detect_pose_source_format,
)
from annolid.annotation.pose_interop.sleap import (
    SleapH5Reader,
    _extract_video_frames,
    detect_sleap_container,
    detect_sleap_h5_kind,
    extract_sleap_zip_frames,
    read_sleap,
    write_sleap_analysis_h5,
    write_sleap_slp,
)
from annolid.annotation.pose_schema import PoseSchema

_FORMAT_ALIASES = {
    "auto": "auto",
    "dlc": "deeplabcut_csv",
    "dlc-csv": "deeplabcut_csv",
    "deeplabcut": "deeplabcut_csv",
    "deeplabcut-csv": "deeplabcut_csv",
    "deeplabcut_csv": "deeplabcut_csv",
    "lightning-pose": "deeplabcut_csv",
    "lightning_pose": "deeplabcut_csv",
    "simba": "deeplabcut_csv",
    "movement": "deeplabcut_csv",
    "dlc-h5": "deeplabcut_h5",
    "deeplabcut-h5": "deeplabcut_h5",
    "deeplabcut_h5": "deeplabcut_h5",
    "dlc-training": "deeplabcut_training",
    "deeplabcut-training": "deeplabcut_training",
    "deeplabcut_training": "deeplabcut_training",
    "dlc-project": "deeplabcut_project",
    "deeplabcut-project": "deeplabcut_project",
    "deeplabcut_project": "deeplabcut_project",
    "sleap": "sleap",
    "sleap-slp": "sleap_slp",
    "sleap_slp": "sleap_slp",
    "slp": "sleap_slp",
    "sleap-h5": "sleap_analysis_h5",
    "sleap_h5": "sleap_analysis_h5",
    "sleap-analysis-h5": "sleap_analysis_h5",
    "sleap_analysis_h5": "sleap_analysis_h5",
    "labelme": "labelme_dir",
    "labelme-dir": "labelme_dir",
    "labelme_dir": "labelme_dir",
    "tracking-csv": "annolid_tracking_csv",
    "annolid-tracking-csv": "annolid_tracking_csv",
    "annolid_tracking_csv": "annolid_tracking_csv",
    "coco": "coco_json",
    "coco-json": "coco_json",
    "coco_json": "coco_json",
}


def normalize_pose_format(name: Optional[str]) -> str:
    raw = str(name or "auto").strip().lower()
    if raw not in _FORMAT_ALIASES:
        supported = sorted(set(_FORMAT_ALIASES.values()))
        raise ValueError(
            f"Unsupported pose format {name!r}. Supported formats: {', '.join(supported)}"
        )
    return _FORMAT_ALIASES[raw]


def read_pose_dataset(
    source: str | Path,
    *,
    source_format: str = "auto",
    video_path: Optional[str | Path] = None,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    config_path: Optional[str | Path] = None,
    min_likelihood: Optional[float] = None,
    is_multi_animal: Optional[bool] = None,
    default_instance: str = "mouse",
    video_index: Optional[int] = None,
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    recursive: bool = False,
) -> PoseDataset:
    """Load any supported pose/tracking source into a canonical ``PoseDataset``."""
    src_path = Path(source).expanduser().resolve()
    fmt = normalize_pose_format(source_format)
    if fmt == "auto":
        fmt = detect_pose_source_format(src_path)
    if fmt == "sleap":
        fmt = (
            "sleap_analysis_h5"
            if detect_sleap_h5_kind(src_path) == "analysis_h5"
            else "sleap_slp"
        )

    if fmt == "deeplabcut_training":
        return read_dlc_training_project(
            src_path,
            config_path=config_path,
            default_instance=default_instance,
            recursive=recursive or True,
        )
    if fmt in ("deeplabcut_csv", "deeplabcut_h5"):
        return read_dlc_analysis(
            src_path,
            video_path=video_path,
            config_path=config_path,
            min_likelihood=min_likelihood,
            is_multi_animal=is_multi_animal,
            default_instance=default_instance,
        )
    if fmt in ("sleap_slp", "sleap_analysis_h5"):
        return read_sleap(
            src_path,
            video_path=video_path,
            video_index=video_index,
            prefer_user_instances=prefer_user_instances,
            include_predicted=include_predicted,
            include_occluded=include_occluded,
            min_score=min_likelihood,
        )
    if fmt in ("labelme_dir", "labelme_json"):
        return read_pose_from_labelme_dir(
            src_path,
            pose_schema=pose_schema,
            recursive=recursive,
        )
    if fmt == "annolid_tracking_csv":
        return read_pose_from_tracking_csv(
            src_path,
            pose_schema=pose_schema,
            default_instance=default_instance,
        )
    if fmt == "coco_json":
        return read_pose_from_coco_json(src_path)

    raise ValueError(
        f"Could not detect a supported pose format for {src_path} (detected={fmt!r})."
    )


def import_pose_to_labelme(
    source: str | Path,
    out_dir: Optional[str | Path] = None,
    *,
    source_format: str = "auto",
    video_path: Optional[str | Path] = None,
    pose_schema_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    save_frames: bool = True,
    video_index: Optional[int] = None,
    min_likelihood: Optional[float] = None,
    is_multi_animal: Optional[bool] = None,
    default_instance: str = "mouse",
    prefer_user_instances: bool = True,
    include_predicted: bool = True,
    include_occluded: bool = True,
    write_pose_schema: bool = True,
    pose_schema_preset: str = "auto",
    filename_prefix: Optional[str] = None,
    prefix_labels_with_instance: bool = False,
    instance_separator: str = "_",
    overwrite: bool = True,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Import DeepLabCut, Lightning Pose, SLEAP, or COCO pose data into Annolid LabelMe JSONs."""
    src_path = Path(source).expanduser().resolve()
    fmt = normalize_pose_format(source_format)
    if fmt == "auto":
        # Allow passing a video path directly when a sibling DLC CSV/H5 exists.
        if src_path.is_file() and src_path.suffix.lower() in (
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".m4v",
        ):
            dlc_match = find_dlc_analysis_file_for_video(src_path)
            if dlc_match is None:
                raise FileNotFoundError(
                    f"No DeepLabCut analysis CSV or H5 found next to video: {src_path}"
                )
            video_path = video_path or src_path
            src_path = dlc_match
            fmt = detect_pose_source_format(src_path)
        else:
            fmt = detect_pose_source_format(src_path)

    if fmt == "deeplabcut_training":
        from annolid.datasets.importers.deeplabcut_training_data import (
            DeepLabCutTrainingImportConfig,
            import_deeplabcut_training_data,
        )

        labeled_candidate = (
            src_path / "labeled-data"
            if (src_path / "labeled-data").is_dir()
            else src_path
        )
        source_root = src_path if labeled_candidate != src_path else src_path.parent
        return import_deeplabcut_training_data(
            DeepLabCutTrainingImportConfig(
                source_dir=source_root,
                labeled_data_root=labeled_candidate,
                instance_label=default_instance,
                overwrite=overwrite,
                recursive=recursive,
            ),
            write_pose_schema=write_pose_schema,
            pose_schema_out=Path(out_dir) / "pose_schema.json" if out_dir else None,
            pose_schema_preset=pose_schema_preset,
            instance_separator=instance_separator,
        )

    if fmt == "sleap":
        fmt = (
            "sleap_analysis_h5"
            if detect_sleap_h5_kind(src_path) == "analysis_h5"
            else "sleap_slp"
        )

    if out_dir is None:
        if src_path.name.lower().endswith(".pkg.slp"):
            stem = src_path.name[: -len(".pkg.slp")]
        else:
            stem = src_path.stem
        out_root = src_path.parent / f"{stem}_labelme"
    else:
        out_root = Path(out_dir).expanduser().resolve()

    dataset = read_pose_dataset(
        src_path,
        source_format=fmt,
        video_path=video_path,
        config_path=config_path,
        min_likelihood=min_likelihood,
        is_multi_animal=is_multi_animal,
        default_instance=default_instance,
        video_index=video_index,
        prefer_user_instances=prefer_user_instances,
        include_predicted=include_predicted,
        include_occluded=include_occluded,
    )

    frame_to_image: Dict[Tuple[int, int], str] = {}
    frame_to_wh: Dict[Tuple[int, int], Tuple[int, int]] = {}
    extracted_frames = 0

    if save_frames and fmt == "sleap_slp":
        from PIL import Image

        used_vids = sorted({int(f.video_index) for f in dataset.frames})
        multi_video = len(used_vids) > 1
        vid_names = dataset.metadata.get("video_names") or {}

        if detect_sleap_container(src_path) == "zip":
            zip_img_map, zip_wh_map = extract_sleap_zip_frames(
                src_path,
                dataset,
                out_root,
                filename_prefix=filename_prefix or "",
            )
            frame_to_image.update(zip_img_map)
            frame_to_wh.update(zip_wh_map)
            extracted_frames += len(zip_img_map)
            if not zip_img_map:
                for vid in used_vids:
                    prefix = (
                        str(vid_names.get(vid, f"video{vid}"))
                        if multi_video
                        else (filename_prefix or "")
                    )
                    ext_raw = (
                        video_path
                        if (video_path and vid == 0)
                        else dataset.video_paths.get(vid)
                    )
                    ext_video = Path(ext_raw).expanduser() if ext_raw else None
                    if ext_video is not None and ext_video.is_file():
                        f_indices = [
                            int(f.frame_index)
                            for f in dataset.frames
                            if int(f.video_index) == vid
                        ]
                        mapping, sizes = _extract_video_frames(
                            ext_video, f_indices, out_root, name_prefix=prefix
                        )
                        for fidx, rel in mapping.items():
                            frame_to_image[(vid, fidx)] = rel
                            if fidx in sizes:
                                frame_to_wh[(vid, fidx)] = sizes[fidx]
                        extracted_frames += len(mapping)
                    else:
                        dataset.warnings.append(
                            f"No embedded images found in ZIP SLEAP archive for video {vid} and external video is unavailable; "
                            "wrote LabelMe JSONs without extracting images."
                        )
        else:
            with SleapH5Reader(src_path) as reader:
                for vid in used_vids:
                    prefix = (
                        str(vid_names.get(vid, f"video{vid}"))
                        if multi_video
                        else (filename_prefix or "")
                    )
                    if reader.has_embedded_frames(vid):
                        mapping = reader.save_embedded_frames(
                            out_root,
                            video_index=vid,
                            name_prefix=prefix,
                        )
                        for fidx, rel in mapping.items():
                            frame_to_image[(vid, fidx)] = rel
                            with Image.open(out_root / rel) as im:
                                frame_to_wh[(vid, fidx)] = (im.width, im.height)
                        extracted_frames += len(mapping)
                    else:
                        ext_video = (
                            Path(video_path).expanduser()
                            if (video_path and vid == 0)
                            else (
                                Path(reader.video_filename(vid)).expanduser()
                                if reader.video_filename(vid)
                                else None
                            )
                        )
                        if ext_video is not None and ext_video.is_file():
                            f_indices = [
                                int(f.frame_index)
                                for f in dataset.frames
                                if int(f.video_index) == vid
                            ]
                            mapping, sizes = _extract_video_frames(
                                ext_video,
                                f_indices,
                                out_root,
                                name_prefix=prefix,
                            )
                            for fidx, rel in mapping.items():
                                frame_to_image[(vid, fidx)] = rel
                                if fidx in sizes:
                                    frame_to_wh[(vid, fidx)] = sizes[fidx]
                            extracted_frames += len(mapping)
                        else:
                            dataset.warnings.append(
                                f"No embedded frames found at /video{vid}/video and external video is unavailable; "
                                "wrote LabelMe JSONs without extracting images."
                            )
    elif save_frames and video_path and Path(video_path).expanduser().is_file():
        f_indices = [int(f.frame_index) for f in dataset.frames]
        mapping, sizes = _extract_video_frames(
            Path(video_path).expanduser(),
            f_indices,
            out_root,
            name_prefix=filename_prefix or "",
        )
        for fidx, rel in mapping.items():
            frame_to_image[(0, fidx)] = rel
            if fidx in sizes:
                frame_to_wh[(0, fidx)] = sizes[fidx]
        extracted_frames += len(mapping)

    summary = write_pose_to_labelme_dir(
        dataset,
        out_root,
        overwrite=overwrite,
        write_pose_schema=write_pose_schema,
        pose_schema_override=pose_schema_path,
        pose_schema_preset=pose_schema_preset,
        filename_prefix=filename_prefix,
        prefix_labels_with_instance=prefix_labels_with_instance,
        instance_separator=instance_separator,
        include_occluded=include_occluded,
        frame_to_image=frame_to_image or None,
        frame_to_wh=frame_to_wh or None,
    )
    summary["frames_extracted"] = extracted_frames
    summary["source_path"] = str(src_path)
    return summary


def export_pose_dataset(
    source: str | Path,
    output_path: str | Path,
    *,
    target_format: str,
    source_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    video_path: Optional[str | Path] = None,
    scorer: str = "annolid",
    multi_animal: Optional[bool] = None,
    video_name: Optional[str] = None,
    copy_images: bool = True,
    embed_images: bool = False,
    write_config_yaml: bool = True,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Export Annolid LabelMe / Tracking CSV / COCO / pose data to DLC, Lightning Pose, or SLEAP."""
    tgt_fmt = normalize_pose_format(target_format)
    dataset = read_pose_dataset(
        source,
        source_format=source_format,
        video_path=video_path,
        pose_schema=pose_schema,
        recursive=recursive,
    )
    if not dataset.frames:
        raise ValueError(f"No pose frames found in {source}.")

    out = Path(output_path).expanduser().resolve()
    if tgt_fmt == "deeplabcut_csv":
        if out.is_dir():
            out = out / f"{Path(source).stem}_dlc.csv"
        written = write_dlc_analysis_csv(
            dataset,
            out,
            scorer=scorer,
            multi_animal=multi_animal,
        )
        return {
            "target_format": "deeplabcut_csv",
            "output_path": str(written),
            "frames_exported": len(dataset.frames),
            "multi_animal": bool(
                multi_animal if multi_animal is not None else dataset.is_multi_animal()
            ),
            "keypoints": dataset.resolved_keypoints(),
            "instances": dataset.resolved_instances(),
        }

    if tgt_fmt == "deeplabcut_h5":
        if out.is_dir():
            out = out / f"{Path(source).stem}_dlc.h5"
        written = write_dlc_analysis_h5(
            dataset,
            out,
            scorer=scorer,
            multi_animal=multi_animal,
        )
        return {
            "target_format": "deeplabcut_h5",
            "output_path": str(written),
            "frames_exported": len(dataset.frames),
            "multi_animal": bool(
                multi_animal if multi_animal is not None else dataset.is_multi_animal()
            ),
            "keypoints": dataset.resolved_keypoints(),
            "instances": dataset.resolved_instances(),
        }

    if tgt_fmt == "deeplabcut_project":
        vname = video_name or Path(source).stem or "video"
        res = write_dlc_training_project(
            dataset,
            out,
            scorer=scorer,
            video_name=vname,
            copy_images=copy_images,
            write_config_yaml=write_config_yaml,
            multi_animal=multi_animal,
        )
        res["target_format"] = "deeplabcut_project"
        return res

    if tgt_fmt == "sleap_analysis_h5":
        if out.is_dir():
            out = out / f"{Path(source).stem}.analysis.h5"
        written = write_sleap_analysis_h5(
            dataset,
            out,
            video_path=video_path,
        )
        return {
            "target_format": "sleap_analysis_h5",
            "output_path": str(written),
            "frames_exported": len(dataset.frames),
            "multi_animal": dataset.is_multi_animal(),
            "keypoints": dataset.resolved_keypoints(),
            "instances": dataset.resolved_instances(),
        }

    if tgt_fmt in ("sleap_slp", "sleap"):
        if out.is_dir():
            ext = ".pkg.slp" if embed_images else ".slp"
            out = out / f"{Path(source).stem}{ext}"
        written = write_sleap_slp(
            dataset,
            out,
            video_path=video_path,
            embed_images=embed_images,
        )
        return {
            "target_format": "sleap_slp",
            "output_path": str(written),
            "embed_images": bool(embed_images),
            "frames_exported": len(dataset.frames),
            "multi_animal": dataset.is_multi_animal(),
            "keypoints": dataset.resolved_keypoints(),
            "instances": dataset.resolved_instances(),
        }

    if tgt_fmt == "labelme_dir":
        return write_pose_to_labelme_dir(dataset, out)

    if tgt_fmt == "coco_json":
        if out.is_dir():
            out = out / f"{Path(source).stem}_coco_keypoints.json"
        return write_pose_to_coco_json(dataset, out)

    raise ValueError(
        f"Unsupported export target_format={target_format!r}. "
        "Choose one of: deeplabcut_csv, deeplabcut_project, sleap_analysis_h5, sleap_slp, labelme_dir, coco_json."
    )
